"""
SAM3 cross-image few-shot via canvas-composite exemplar prompting.

No native cross-image few-shot API in SAM3 (exemplar boxes only work within
same image). Trick (WongKinYiu/FSS-SAM3): paste ref + target on one canvas,
remap ref bbox to canvas coords, run SAM3 once (box-only exemplar, no text),
crop target half back out.

Refs sharing a source image are grouped onto one canvas as a multi-box prompt
(one SAM3 call per frame, not per instance).

Per (class, ref group, target) triple, saves 4-panel figure: composite canvas
w/ ref box(es) -> raw SAM3 pred on canvas -> pred cropped to target (mask+bbox+score)
-> tightened bbox only.

--class-ids: explicit ("0 1 2") or "all" (auto-discover from --refs-labels).
Output: output_dir/<target_stem>__cls<id>_<ref>.png

Targets batched per ref group (--batch-size) into one SAM3 forward pass;
lower it on 8GB VRAM OOM.

Speed: bf16 on CUDA by default (--fp32 disables); figure rendering offloaded
to thread pool; --max-refs-per-class N does DINOv2 CLS-embed + farthest-point
sample of N diverse refs per class (0 = all), saved to output_dir/temp_refs/
for inspection (DINOv2 loaded/unloaded before SAM3, VRAM rule).

Usage:
    python experiments/debug_sam3.py \\
        --refs-dir    "D:/path/to/labelled_ref_images" \\
        --refs-labels "D:/path/to/labelled_ref_images"  (YOLO .txt, same stem) \\
        --class-ids   0 1 2 \\
        --targets-dir "D:/path/to/target_images"
"""

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch
from tqdm import tqdm
import matplotlib.patches as patches
from matplotlib.figure import Figure
from PIL import Image

RAM_GUARD_PCT = 90.0


def ram_guard():
    """GC pass if RAM usage crosses RAM_GUARD_PCT."""
    if psutil.virtual_memory().percent >= RAM_GUARD_PCT:
        gc.collect()
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
            if len(parts) < 5:
                continue
            try:
                ids.add(int(parts[0]))
            except ValueError:
                print(f"[warn] malformed class id in {label_path.name}: {parts[0]!r} — skipped")
    return sorted(ids)


def collect_ref_boxes(refs_dir: Path, refs_labels: Path, class_ids: list[int]) -> dict[int, list[dict]]:
    """Metadata-only: parse labels for boxes matching class_ids, read image size from header (no pixel decode) to convert YOLO norm coords to px. 
    Full decode deferred until an instance survives diverse-ref selection."""
    wanted = set(class_ids)
    boxes_by_class: dict[int, list[dict]] = {c: [] for c in class_ids}
    for img_path in sorted(refs_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = refs_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        lines = label_path.read_text().splitlines()
        matches = []
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                print(f"[warn] malformed class id in {label_path.name} line {i}: {parts[0]!r} — skipped")
                continue
            if cid not in wanted:
                continue
            matches.append((i, cid, parts))
        if not matches:
            continue
        try:
            with Image.open(img_path) as img:
                iw, ih = img.size  # header-only, no pixel decode
        except Exception as e:
            print(f"[warn] cannot read {img_path.name}: {e} — skipped")
            continue
        if iw <= 0 or ih <= 0:
            print(f"[warn] {img_path.name} has invalid size {iw}x{ih} — skipped")
            continue
        for i, cid, parts in matches:
            try:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                print(f"[warn] malformed box in {label_path.name} line {i} — skipped")
                continue
            box_xywh = [(cx - w / 2) * iw, (cy - h / 2) * ih, w * iw, h * ih]  # x,y,w,h px
            boxes_by_class[cid].append({
                "img_path": img_path,
                "box": box_xywh,
                "name": f"{img_path.stem}_inst{i}",
            })
    return boxes_by_class


def farthest_point_sample(vectors: np.ndarray, k: int) -> list[int]:
    """Greedy max-min diversity selection (cf. scripts/extract_crops_labelled.py)."""
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


def _load_crop(inst: dict) -> Image.Image:
    """Decode source image on demand, crop box region."""
    with Image.open(inst["img_path"]) as img:
        img = img.convert("RGB")
        x, y, w, h = inst["box"]
        return img.crop((int(x), int(y), int(x + w), int(y + h)))


def _phash_dedup(instances: list[dict], max_dist: int, class_id) -> list[dict]:
    """Perceptual-hash near-dup filter over ref crops (decoded lazily). Growable
    hash matrix, hamming distance <= max_dist rejects as dup of one already kept
    (cf. extract_crops_labelled.py phase1_extract_class)."""
    import imagehash

    hash_mat = np.empty((256, 64), dtype=np.uint8)
    n_hashes = 0
    kept = []
    for inst in tqdm(instances, desc=f"  cls{class_id} phash", unit="crop", leave=False):
        crop = _load_crop(inst)
        ph = imagehash.phash(crop)
        bits = np.array(ph.hash, dtype=np.uint8).ravel()
        if n_hashes > 0:
            dists = (hash_mat[:n_hashes] != bits).sum(axis=1)
            if dists.min() <= max_dist:
                continue
        if n_hashes >= len(hash_mat):
            hash_mat = np.vstack([hash_mat, np.empty_like(hash_mat)])
        hash_mat[n_hashes] = bits
        n_hashes += 1
        kept.append(inst)
    return kept


def select_diverse_refs(ref_boxes_by_class: dict, max_refs: int, output_dir: Path,
                         device: str, dinov2_batch_size: int = 16,
                         phash_max_dist: int = 0, ref_jpeg_quality: int = 92) -> dict:
    """
        Optional phash dedup, then DINOv2 CLS-embed surviving ref crops (lazy decode, one frame at a time), farthest-point-sample max_refs diverse refs
        per class. Chosen crops saved to output_dir/temp_refs/cls{id}/; full images decoded/kept only for surviving instances. Loads+unloads DINOv2 fully
    """
    from transformers import AutoImageProcessor, AutoModel

    if phash_max_dist > 0:
        deduped = {}
        for class_id, boxes in ref_boxes_by_class.items():
            deduped[class_id] = _phash_dedup(boxes, phash_max_dist, class_id)
            if len(deduped[class_id]) != len(boxes):
                print(f"[refs] class {class_id}: phash dedup {len(boxes)} -> {len(deduped[class_id])}")
        ref_boxes_by_class = deduped  # don't mutate caller's dict

    keep_all = max_refs <= 0  # 0 = no pruning, decode-and-keep every candidate
    needs_selection = (not keep_all) and any(len(v) > max_refs for v in ref_boxes_by_class.values())
    dproc = dmodel = None
    if needs_selection:
        print("[refs] Loading DINOv2 for diverse ref selection ...")
        dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        dmodel = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    selected_by_class = {}
    with torch.no_grad():
        for class_id, instances in ref_boxes_by_class.items():
            if keep_all or len(instances) <= max_refs:
                keep = list(range(len(instances)))
            else:
                embs = []
                for i in tqdm(range(0, len(instances), dinov2_batch_size),
                               desc=f"  cls{class_id} dinov2", unit="batch", leave=False):
                    batch_insts = instances[i:i + dinov2_batch_size]
                    crops = [_load_crop(inst) for inst in batch_insts]
                    inputs = dproc(images=crops, return_tensors="pt").to(device)
                    out = dmodel(**inputs)
                    embs.append(out.last_hidden_state[:, 0, :].float().cpu().numpy())
                    del crops, inputs, out
                    ram_guard()
                embs = np.vstack(embs)
                normed = embs / np.linalg.norm(embs, axis=1, keepdims=True)
                keep = farthest_point_sample(normed, max_refs)
                print(f"[refs] class {class_id}: {len(instances)} -> {len(keep)} diverse ref(s): "
                      f"{', '.join(instances[i]['name'] for i in keep)}")

            temp_refs_dir = output_dir / "temp_refs" / f"cls{class_id}"
            temp_refs_dir.mkdir(parents=True, exist_ok=True)

            # Group by source image: shared frame -> one canvas ref-half, one multi-box prompt
            groups_by_path = {}
            for i in keep:
                inst = instances[i]
                groups_by_path.setdefault(inst["img_path"], []).append(inst)

            kept_groups = []
            for img_path, insts in groups_by_path.items():
                with Image.open(img_path) as img:
                    full_img = img.convert("RGB")
                boxes, names = [], []
                for inst in insts:
                    x, y, w, h = inst["box"]
                    crop = full_img.crop((int(x), int(y), int(x + w), int(y + h)))
                    crop.save(temp_refs_dir / f"{inst['name']}.jpg", quality=ref_jpeg_quality)
                    boxes.append(inst["box"])
                    names.append(inst["name"])
                kept_groups.append({"img_path": img_path, "image": full_img, "boxes": boxes, "names": names})
            selected_by_class[class_id] = kept_groups
            ram_guard()

    if dmodel is not None:
        del dmodel, dproc
        if device == "cuda":
            torch.cuda.empty_cache()
    return selected_by_class


def create_canvas(ref_img: Image.Image, ref_boxes: list[list[float]], target_img: Image.Image,
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
        {"offset": (s_rect[0], s_rect[1]), "max_dim": (s_rect[2], s_rect[3]), "image": ref_img, "type": "ref", "box": ref_boxes},
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


def get_norm_boxes(placements: dict) -> list[list[float]]:
    """Remap ref's pixel-space boxes into canvas-normalized (cx, cy, w, h) in [0,1], one per ref instance."""
    p = placements["ref"]
    cw, ch = placements["canvas_size"]
    ox, oy = p["offset"]
    sx, sy = p["curr_size"][0] / p["orig_size"][0], p["curr_size"][1] / p["orig_size"][1]
    norm_boxes = []
    for bx, by, bw, bh in p["orig_box"]:
        px, py = bx * sx + ox, by * sy + oy
        norm_boxes.append([(px + bw * sx / 2) / cw, (py + bh * sy / 2) / ch, (bw * sx) / cw, (bh * sy) / ch])
    return norm_boxes


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


def save_step_figure(canvas, ref_boxes_canvas_xyxy, raw_mask_canvas, raw_boxes_canvas, raw_scores_canvas,
                      target_img, pred_mask_target, pred_boxes_target, pred_scores_target,
                      ref_name, target_name, output_path):
    # OO API (Figure, not pyplot) — no global state, safe to call from worker threads
    fig = Figure(figsize=(32, 8))
    axes = fig.subplots(1, 4)

    axes[0].imshow(canvas)
    for x1, y1, x2, y2 in ref_boxes_canvas_xyxy:
        axes[0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=2, edgecolor="lime", facecolor="none"))
    axes[0].set_title(f"Canvas: ref={ref_name}  target={target_name}\n"
                       f"(green box(es) = {len(ref_boxes_canvas_xyxy)} exemplar prompt(s))", fontsize=9)
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


def process_result(canvas, placements, ref_boxes_canvas_xyxy, target_img, masks_canvas,
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
        canvas, ref_boxes_canvas_xyxy, raw_mask_canvas, boxes_canvas, scores_canvas,
        target_img, combined_mask_target, pred_boxes_target, pred_scores_target,
        ref_name, target_name, out_path,
    )


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
    p.add_argument("--batch-size", type=int, default=8,
                    help="Targets batched per forward pass (same ref). Lower if OOM on 8GB VRAM.")
    p.add_argument("--max-refs-per-class", type=int, default=5,
                    help="Diverse refs kept per class via DINOv2+farthest-point sampling "
                         "(0 = use all refs). Chosen crops saved to output_dir/temp_refs/.")
    p.add_argument("--dinov2-batch-size", type=int, default=32,
                    help="Crops per forward pass during diverse-ref DINOv2 embedding.")
    p.add_argument("--phash-max-dist", type=int, default=0,
                    help="Perceptual-hash dedup of ref crops before DINOv2 embedding "
                         "(hamming distance <= this = duplicate, skipped). 0 = disabled. ")
    p.add_argument("--fp32", action="store_true",
                    help="Run SAM3 in fp32 (default bf16 on CUDA).")
    p.add_argument("--save-workers", type=int, default=2,
                    help="Thread pool size for offloaded figure rendering/saving.")
    p.add_argument("--ref-jpeg-quality", type=int, default=80,
                    help="JPEG quality for saved ref crops in output_dir/temp_refs/.")
    return p.parse_args()

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

    raw_ref_boxes_by_class = collect_ref_boxes(Path(args.refs_dir), refs_labels_dir, class_ids)
    ref_boxes_by_class = {}
    for class_id in class_ids:
        boxes = raw_ref_boxes_by_class.get(class_id, [])
        print(f"[refs] {len(boxes)} ref instance(s) for class {class_id}")
        if boxes:
            ref_boxes_by_class[class_id] = boxes
    if not ref_boxes_by_class:
        print("[abort] No ref instances found for any class.")
        return

    targets_dir = Path(args.targets_dir)
    target_paths = sorted(p for p in targets_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"[targets] {len(target_paths)} target image(s)")
    if not target_paths:
        print("[abort] No target images found.")
        return

    # Diverse ref selection (DINOv2) must run before SAM3 loads — VRAM rule.
    ref_instances_by_class = select_diverse_refs(
        ref_boxes_by_class, args.max_refs_per_class, output_dir, device,
        dinov2_batch_size=args.dinov2_batch_size, phash_max_dist=args.phash_max_dist,
        ref_jpeg_quality=args.ref_jpeg_quality,
    )

    dtype = torch.float32 if (args.fp32 or device != "cuda") else torch.bfloat16
    print(f"[model] Loading SAM3: {SAM3_MODEL_ID} ({dtype}) ...")
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID, torch_dtype=dtype, device_map=device)
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)

    total = sum(len(groups) for groups in ref_instances_by_class.values()) * len(target_paths)
    done = 0
    batch_size = args.batch_size
    # Offload CPU-bound figure rendering so GPU keeps working
    save_pool = ThreadPoolExecutor(max_workers=args.save_workers)
    save_futures = []
    max_pending_saves = args.save_workers * 4  # drain periodically: bound memory, surface worker errors promptly
    for class_id, ref_groups in ref_instances_by_class.items():
        for ref_group in ref_groups:
            # Cap tag length (Windows MAX_PATH)
            names = ref_group["names"]
            tag_suffix = names[0] if len(names) == 1 else f"{names[0]}+{len(names) - 1}more"
            ref_tag = f"cls{class_id}_{tag_suffix}"
            for batch_start in range(0, len(target_paths), batch_size):
                batch_paths = target_paths[batch_start:batch_start + batch_size]
                print(f"\n[class={class_id} ref_group={ref_group['names']}] "
                      f"batch {batch_start // batch_size + 1} "
                      f"({len(batch_paths)} target(s): {', '.join(p.name for p in batch_paths)})")

                # Decode only this batch's targets — freed at end of the batch iteration
                batch_target_imgs = {p: Image.open(p).convert("RGB") for p in batch_paths}
                canvases, placements_list, ref_box_xyxys_list = [], [], []
                for target_path in batch_paths:
                    canvas, placements = create_canvas(
                        ref_group["image"], ref_group["boxes"], batch_target_imgs[target_path],
                        args.canvas_size, args.orientation, args.split_ratio,
                    )
                    norm_boxes = get_norm_boxes(placements)
                    cw, ch = placements["canvas_size"]
                    box_xyxys = [norm_cxcywh_to_xyxy_px(nb, cw, ch) for nb in norm_boxes]
                    canvases.append(canvas)
                    placements_list.append(placements)
                    ref_box_xyxys_list.append(box_xyxys)

                inputs = processor(
                    images=canvases,
                    input_boxes=ref_box_xyxys_list,
                    input_boxes_labels=[[1] * len(b) for b in ref_box_xyxys_list],
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

                for target_path, canvas, placements, ref_box_xyxys, results in zip(
                    batch_paths, canvases, placements_list, ref_box_xyxys_list, results_list
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
                        canvas, placements, ref_box_xyxys, batch_target_imgs[target_path],
                        masks_canvas, scores_canvas, boxes_canvas,
                        ref_tag, target_path.name, out_path,
                    ))

                del inputs, outputs
                ram_guard()

                if len(save_futures) >= max_pending_saves:
                    for f in save_futures:
                        f.result()   # re-raise any worker exception promptly
                    save_futures.clear()

    print("\n[save] waiting for remaining figure saves to finish ...")
    for f in save_futures:
        f.result()
    save_pool.shutdown()

    print(f"\n[done] Output -> {output_dir.resolve()}")


if __name__ == "__main__":
    main()
