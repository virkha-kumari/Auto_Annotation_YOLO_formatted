"""
Debug script: SAM3 raw few-shot capability via canvas-composite exemplar prompting.

SAM3 has no native cross-image few-shot API (image-exemplar boxes only work
within the same image the box was drawn on). This script uses the canvas trick
from "Few-Shot Semantic Segmentation Meets SAM3" (WongKinYiu/FSS-SAM3): paste a
reference image and a target image into one shared canvas, remap the reference's
known bbox into canvas-normalized coords, run SAM3 once with that box as a
positive geometric exemplar (box-only, no text), then crop the target half of
the prediction back out and resize to the target's original size.

For each (class id, ref instance, target image) triple, saves a 4-panel figure:
  1. Composite canvas (ref top, target bottom) with the ref exemplar box drawn
  2. Raw SAM3 prediction on the full canvas, with SAM3's own boxes + scores
  3. Prediction cropped back to target image, overlaid as mask + bbox + scores
  4. Tightened bbox only, per instance, with scores

Multi-class: pass --class-ids as explicit ids ("0 1 2") or omit for "all"
(auto-discovers every class id present in --refs-labels). Output is flat,
one file per (ref, target) pair: output_dir/<target_stem>__cls<id>_<ref>.png

Targets are batched per ref (--batch-size) into a single SAM3 forward pass
for speed; lower --batch-size if you hit OOM on 8GB VRAM.

Speed features:
  - SAM3 runs in bf16 on CUDA by default (--fp32 to disable)
  - Figure rendering offloaded to a 2-worker thread pool (GPU never waits on matplotlib)
  - --max-refs-per-class N (default 3): DINOv2 CLS-embeds every ref crop and
    farthest-point-samples N diverse refs per class (0 = all refs). Chosen crops
    saved to output_dir/temp_refs/cls{id}/ for inspection. DINOv2 is loaded and
    unloaded before SAM3 loads (VRAM rule).

Usage:
    python test/debug_sam3.py \\
        --refs-dir    "D:/path/to/labelled_ref_images" \\
        --refs-labels "D:/path/to/labelled_ref_images"  (YOLO .txt, same stem) \\
        --class-ids   0 1 2 \\
        --targets-dir "D:/path/to/target_images"
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import matplotlib.patches as patches
from matplotlib.figure import Figure
from PIL import Image
from transformers import Sam3Model, Sam3Processor

SAM3_MODEL_ID = "facebook/sam3"
CANVAS_SIZE = 1008
SPLIT_RATIO = 0.5   # ref gets 50% of canvas, target gets 50%
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def discover_class_ids(refs_labels: Path) -> list[int]:
    ids = set()
    for label_path in refs_labels.glob("*.txt"):
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                ids.add(int(parts[0]))
    return sorted(ids)


def collect_ref_instances(refs_dir: Path, refs_labels: Path, class_id: int) -> list[dict]:
    """Each YOLO label line matching class_id in each ref image → one ref instance dict."""
    instances = []
    for img_path in sorted(refs_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = refs_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        for i, line in enumerate(label_path.read_text().splitlines()):
            parts = line.strip().split()
            if len(parts) < 5 or int(parts[0]) != class_id:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            box_xywh = [(cx - w / 2) * iw, (cy - h / 2) * ih, w * iw, h * ih]  # x,y,w,h px
            instances.append({
                "image": img,
                "box": box_xywh,
                "name": f"{img_path.stem}_inst{i}",
            })
    return instances


def farthest_point_sample(vectors: np.ndarray, k: int) -> list[int]:
    """Greedy max-min diversity selection (same as scripts/extract_crops_labelled.py)."""
    if k <= 0 or len(vectors) == 0:
        return []
    k = min(k, len(vectors))
    seed = int(np.argmax(np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)))
    selected = [seed]
    dists = np.full(len(vectors), np.inf)
    for _ in range(k - 1):
        last = vectors[selected[-1]]
        d = np.linalg.norm(vectors - last, axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return selected


def select_diverse_refs(ref_instances_by_class: dict, max_refs: int, output_dir: Path,
                         device: str) -> dict:
    """
    DINOv2 CLS-embed each ref crop, farthest-point-sample max_refs diverse refs per
    class. Chosen ref crops saved to output_dir/temp_refs/cls{id}/ for inspection.
    Loads and fully unloads DINOv2 (VRAM rule: must run BEFORE SAM3 is loaded).
    """
    from transformers import AutoImageProcessor, AutoModel

    needs_selection = any(len(v) > max_refs for v in ref_instances_by_class.values())
    dproc = dmodel = None
    if needs_selection:
        print("[refs] Loading DINOv2 for diverse ref selection ...")
        dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        dmodel = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    selected_by_class = {}
    with torch.no_grad():
        for class_id, instances in ref_instances_by_class.items():
            crops = []
            for inst in instances:
                x, y, w, h = inst["box"]
                crops.append(inst["image"].crop((int(x), int(y), int(x + w), int(y + h))))

            if len(instances) <= max_refs:
                keep = list(range(len(instances)))
            else:
                embs = []
                for i in range(0, len(crops), 16):
                    inputs = dproc(images=crops[i:i + 16], return_tensors="pt").to(device)
                    out = dmodel(**inputs)
                    embs.append(out.last_hidden_state[:, 0, :].float().cpu().numpy())
                embs = np.vstack(embs)
                normed = embs / np.linalg.norm(embs, axis=1, keepdims=True)
                keep = farthest_point_sample(normed, max_refs)
                print(f"[refs] class {class_id}: {len(instances)} -> {len(keep)} diverse ref(s): "
                      f"{', '.join(instances[i]['name'] for i in keep)}")

            selected_by_class[class_id] = [instances[i] for i in keep]

            temp_refs_dir = output_dir / "temp_refs" / f"cls{class_id}"
            temp_refs_dir.mkdir(parents=True, exist_ok=True)
            for i in keep:
                crops[i].save(temp_refs_dir / f"{instances[i]['name']}.jpg", quality=92)

    if dmodel is not None:
        del dmodel, dproc
        if device == "cuda":
            torch.cuda.empty_cache()
    return selected_by_class


def create_canvas(ref_img: Image.Image, ref_box: list[float], target_img: Image.Image,
                   canvas_size: int, orientation: str, split_ratio: float):
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    split_pos = int(canvas_size * split_ratio)
    rem_pos = canvas_size - split_pos

    if orientation == "vertical":
        s_rect = (0, 0, canvas_size, split_pos)
        t_rect = (0, split_pos, canvas_size, rem_pos)
    else:
        s_rect = (0, 0, split_pos, canvas_size)
        t_rect = (split_pos, 0, rem_pos, canvas_size)

    layouts = [
        {"offset": (s_rect[0], s_rect[1]), "max_dim": (s_rect[2], s_rect[3]), "image": ref_img, "type": "ref", "box": ref_box},
        {"offset": (t_rect[0], t_rect[1]), "max_dim": (t_rect[2], t_rect[3]), "image": target_img, "type": "tgt", "box": None},
    ]

    placements = {"canvas_size": (canvas_size, canvas_size)}
    for lay in layouts:
        target_w, target_h = int(lay["max_dim"][0]), int(lay["max_dim"][1])
        img_resized = lay["image"].resize((target_w, target_h), Image.BILINEAR)
        canvas.paste(img_resized, lay["offset"])
        placements[lay["type"]] = {
            "offset": lay["offset"], "curr_size": (target_w, target_h),
            "orig_size": lay["image"].size, "orig_box": lay["box"],
        }
    return canvas, placements


def get_norm_box(placements: dict) -> list[float]:
    """Remap ref's pixel-space box into canvas-normalized (cx, cy, w, h) in [0,1]."""
    p = placements["ref"]
    cw, ch = placements["canvas_size"]
    bx, by, bw, bh = p["orig_box"]
    ox, oy = p["offset"]
    sx, sy = p["curr_size"][0] / p["orig_size"][0], p["curr_size"][1] / p["orig_size"][1]
    px, py = bx * sx + ox, by * sy + oy
    return [(px + bw * sx / 2) / cw, (py + bh * sy / 2) / ch, (bw * sx) / cw, (bh * sy) / ch]


def norm_cxcywh_to_xyxy_px(norm_box: list[float], w: int, h: int) -> list[float]:
    cx, cy, bw, bh = norm_box
    return [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]


def mask_to_bbox(mask: np.ndarray) -> list[int] | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


def save_step_figure(canvas, ref_box_canvas_xyxy, raw_mask_canvas, raw_boxes_canvas, raw_scores_canvas,
                      target_img, pred_mask_target, pred_boxes_target, pred_scores_target,
                      ref_name, target_name, output_path):
    # OO API (Figure, not pyplot) — no global state, safe to call from worker threads
    fig = Figure(figsize=(32, 8))
    axes = fig.subplots(1, 4)

    axes[0].imshow(canvas)
    x1, y1, x2, y2 = ref_box_canvas_xyxy
    axes[0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                       linewidth=2, edgecolor="lime", facecolor="none"))
    axes[0].set_title(f"Canvas: ref={ref_name}  target={target_name}\n(green box = exemplar prompt)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(canvas)
    axes[1].imshow(raw_mask_canvas, alpha=0.5, cmap="jet")
    for box, score in zip(raw_boxes_canvas, raw_scores_canvas):
        x1, y1, x2, y2 = box
        axes[1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=2, edgecolor="yellow", facecolor="none"))
        axes[1].text(x1, max(y1 - 4, 0), f"{score:.2f}", color="yellow", fontsize=8,
                      bbox=dict(facecolor="black", alpha=0.5, pad=1))
    axes[1].set_title(f"Raw SAM3 prediction on full canvas\n({len(raw_boxes_canvas)} box(es), yellow = SAM3 boxes)", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(target_img)
    if pred_mask_target is not None:
        axes[2].imshow(pred_mask_target, alpha=0.5, cmap="jet")
    for box, score in zip(pred_boxes_target, pred_scores_target):
        x1, y1, x2, y2 = box
        axes[2].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=2, edgecolor="cyan", facecolor="none"))
        axes[2].text(x1, max(y1 - 4, 0), f"{score:.2f}", color="cyan", fontsize=8,
                      bbox=dict(facecolor="black", alpha=0.5, pad=1))
    scores_str = ", ".join(f"{s:.2f}" for s in pred_scores_target) or "-"
    axes[2].set_title(f"Cropped back to target\n({len(pred_boxes_target)} instance(s), scores: {scores_str})", fontsize=9)
    axes[2].axis("off")

    axes[3].imshow(target_img)
    for box, score in zip(pred_boxes_target, pred_scores_target):
        x1, y1, x2, y2 = box
        axes[3].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=2, edgecolor="orange", facecolor="none"))
        axes[3].text(x1, max(y1 - 4, 0), f"{score:.2f}", color="orange", fontsize=8,
                      bbox=dict(facecolor="black", alpha=0.5, pad=1))
    max_score_str = f"{max(pred_scores_target):.2f}" if pred_scores_target else "-"
    axes[3].set_title(f"Tightened bbox only, per instance ({len(pred_boxes_target)})\nmax score: {max_score_str}", fontsize=9)
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"[saved] {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--refs-dir", required=True, help="Folder of full reference images")
    p.add_argument("--refs-labels", required=True, help="Folder of YOLO .txt labels (same stem as ref images)")
    p.add_argument("--class-ids", nargs="+", default=["all"],
                    help="Class ids to pull ref instances for, e.g. '0 1 2'. Default 'all' auto-discovers every class id present in --refs-labels.")
    p.add_argument("--targets-dir", required=True, help="Folder of target images (can include ref images too)")
    p.add_argument("--orientation", choices=["vertical", "horizontal"], default="vertical")
    p.add_argument("--split-ratio", type=float, default=SPLIT_RATIO)
    p.add_argument("--canvas-size", type=int, default=CANVAS_SIZE)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--mask-threshold", type=float, default=0.6)
    p.add_argument("--output-dir", default="output_sam3_fewshot")
    p.add_argument("--batch-size", type=int, default=4,
                    help="Targets batched per forward pass (same ref). Lower if OOM on 8GB VRAM.")
    p.add_argument("--max-refs-per-class", type=int, default=3,
                    help="Diverse refs kept per class via DINOv2+farthest-point sampling "
                         "(0 = use all refs). Chosen crops saved to output_dir/temp_refs/.")
    p.add_argument("--fp32", action="store_true",
                    help="Run SAM3 in fp32 (default bf16 on CUDA).")
    return p.parse_args()


def process_result(canvas, placements, ref_box_canvas_xyxy, target_img, masks_canvas,
                    scores_canvas, boxes_canvas, ref_name, target_name, out_path):
    ch_cw = placements["canvas_size"]
    cw, ch = ch_cw

    raw_mask_canvas = np.zeros((ch, cw), dtype=np.uint8)
    for m in masks_canvas:
        raw_mask_canvas |= m

    tgt = placements["tgt"]
    tx, ty = tgt["offset"]
    tw, th = tgt["curr_size"]

    pred_masks_target = []
    pred_boxes_target = []
    pred_scores_target = []
    for m, score in zip(masks_canvas, scores_canvas):
        crop = m[ty:ty + th, tx:tx + tw]
        if not crop.any():
            continue
        mask_target = np.array(
            Image.fromarray(crop * 255).resize(tgt["orig_size"], Image.NEAREST)
        ) > 0
        box = mask_to_bbox(mask_target)
        if box is None:
            continue
        pred_masks_target.append(mask_target)
        pred_boxes_target.append(box)
        pred_scores_target.append(score)

    if pred_scores_target:
        scores_log = ", ".join(f"{s:.3f}" for s in pred_scores_target)
        print(f"  [scores] {len(pred_scores_target)} instance(s): {scores_log}")
    else:
        print("  [scores] no instances passed into target region")

    combined_mask_target = None
    if pred_masks_target:
        combined_mask_target = np.zeros_like(pred_masks_target[0])
        for m in pred_masks_target:
            combined_mask_target |= m

    save_step_figure(
        canvas, ref_box_canvas_xyxy, raw_mask_canvas, boxes_canvas, scores_canvas,
        target_img, combined_mask_target, pred_boxes_target, pred_scores_target,
        ref_name, target_name, out_path,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    refs_labels_dir = Path(args.refs_labels)
    if args.class_ids == ["all"]:
        class_ids = discover_class_ids(refs_labels_dir)
        print(f"[class-ids] auto-discovered: {class_ids}")
    else:
        class_ids = [int(c) for c in args.class_ids]
    if not class_ids:
        print("[abort] No class ids found.")
        return

    ref_instances_by_class = {}
    for class_id in class_ids:
        instances = collect_ref_instances(Path(args.refs_dir), refs_labels_dir, class_id)
        print(f"[refs] {len(instances)} ref instance(s) for class {class_id}")
        if instances:
            ref_instances_by_class[class_id] = instances
    if not ref_instances_by_class:
        print("[abort] No ref instances found for any class.")
        return

    targets_dir = Path(args.targets_dir)
    target_paths = sorted(p for p in targets_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"[targets] {len(target_paths)} target image(s)")
    if not target_paths:
        print("[abort] No target images found.")
        return

    # Diverse ref selection (DINOv2) must run before SAM3 loads — VRAM rule
    if args.max_refs_per_class > 0:
        ref_instances_by_class = select_diverse_refs(
            ref_instances_by_class, args.max_refs_per_class, output_dir, device,
        )

    dtype = torch.float32 if (args.fp32 or device != "cuda") else torch.bfloat16
    print(f"[model] Loading SAM3: {SAM3_MODEL_ID} ({dtype}) ...")
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID, torch_dtype=dtype, device_map=device)
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)

    target_imgs = {p: Image.open(p).convert("RGB") for p in target_paths}

    total = sum(len(insts) for insts in ref_instances_by_class.values()) * len(target_paths)
    done = 0
    batch_size = args.batch_size
    # Figure rendering is CPU-bound and slow — offload so GPU keeps working
    save_pool = ThreadPoolExecutor(max_workers=2)
    save_futures = []
    for class_id, ref_instances in ref_instances_by_class.items():
        for ref in ref_instances:
            ref_tag = f"cls{class_id}_{ref['name']}"
            for batch_start in range(0, len(target_paths), batch_size):
                batch_paths = target_paths[batch_start:batch_start + batch_size]
                print(f"\n[class={class_id} ref={ref['name']}] batch {batch_start // batch_size + 1} "
                      f"({len(batch_paths)} target(s): {', '.join(p.name for p in batch_paths)})")

                canvases, placements_list, ref_box_xyxys = [], [], []
                for target_path in batch_paths:
                    canvas, placements = create_canvas(
                        ref["image"], ref["box"], target_imgs[target_path],
                        args.canvas_size, args.orientation, args.split_ratio,
                    )
                    norm_box = get_norm_box(placements)
                    cw, ch = placements["canvas_size"]
                    box_xyxy = norm_cxcywh_to_xyxy_px(norm_box, cw, ch)
                    canvases.append(canvas)
                    placements_list.append(placements)
                    ref_box_xyxys.append(box_xyxy)

                inputs = processor(
                    images=canvases,
                    input_boxes=[[b] for b in ref_box_xyxys],
                    input_boxes_labels=[[1] for _ in ref_box_xyxys],
                    return_tensors="pt",
                ).to(model.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
                if "input_boxes" in inputs:
                    inputs["input_boxes"] = inputs["input_boxes"].to(model.dtype)

                with torch.no_grad():
                    outputs = model(**inputs)

                results_list = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=args.threshold,
                    mask_threshold=args.mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )

                for target_path, canvas, placements, ref_box_xyxy, results in zip(
                    batch_paths, canvases, placements_list, ref_box_xyxys, results_list
                ):
                    done += 1
                    print(f"  [{done}/{total}] target={target_path.name}")

                    # .to(uint8) before .numpy() — numpy has no bf16 dtype
                    masks_canvas = [m.to(torch.uint8).cpu().numpy() for m in results["masks"]]
                    scores_canvas = [float(s) for s in results["scores"]]
                    boxes_canvas = [b.float().cpu().numpy().tolist() for b in results["boxes"]]

                    out_path = output_dir / f"{target_path.stem}__{ref_tag}.png"
                    save_futures.append(save_pool.submit(
                        process_result,
                        canvas, placements, ref_box_xyxy, target_imgs[target_path],
                        masks_canvas, scores_canvas, boxes_canvas,
                        ref_tag, target_path.name, out_path,
                    ))

                del inputs, outputs

    print("\n[save] waiting for figure saves to finish ...")
    for f in save_futures:
        f.result()   # re-raise any worker exception
    save_pool.shutdown()

    print(f"\n[done] Output -> {output_dir.resolve()}")


if __name__ == "__main__":
    main()
