"""
Debug script: SAM3 raw few-shot capability via canvas-composite exemplar prompting.

SAM3 has no native cross-image few-shot API (image-exemplar boxes only work
within the same image the box was drawn on). This script uses the canvas trick
from "Few-Shot Semantic Segmentation Meets SAM3" (WongKinYiu/FSS-SAM3): paste a
reference image and a target image into one shared canvas, remap the reference's
known bbox into canvas-normalized coords, run SAM3 once with that box as a
positive geometric exemplar (box-only, no text), then crop the target half of
the prediction back out and resize to the target's original size.

For each (ref instance, target image) pair, saves a 3-panel figure:
  1. Composite canvas (ref top, target bottom) with the ref exemplar box drawn
  2. Raw SAM3 prediction mask on the full canvas
  3. Prediction cropped back to target image, overlaid as mask + bbox

Usage:
    python test/debug_sam3.py \\
        --refs-dir    "D:/path/to/labelled_ref_images" \\
        --refs-labels "D:/path/to/labelled_ref_images"  (YOLO .txt, same stem) \\
        --class-id    0 \\
        --targets-dir "D:/path/to/target_images"
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import Sam3Model, Sam3Processor

SAM3_MODEL_ID = "facebook/sam3"
CANVAS_SIZE = 1008
SPLIT_RATIO = 0.5   # ref gets 50% of canvas, target gets 50%
IMG_EXTS = {".jpg", ".jpeg", ".png"}


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


def save_step_figure(canvas, ref_box_canvas_xyxy, raw_mask_canvas, target_img,
                      pred_mask_target, pred_boxes_target, ref_name, target_name, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(32, 8))

    axes[0].imshow(canvas)
    x1, y1, x2, y2 = ref_box_canvas_xyxy
    axes[0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                       linewidth=2, edgecolor="lime", facecolor="none"))
    axes[0].set_title(f"Canvas: ref={ref_name}  target={target_name}\n(green box = exemplar prompt)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(canvas)
    axes[1].imshow(raw_mask_canvas, alpha=0.5, cmap="jet")
    axes[1].set_title("Raw SAM3 prediction on full canvas", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(target_img)
    if pred_mask_target is not None:
        axes[2].imshow(pred_mask_target, alpha=0.5, cmap="jet")
    for box in pred_boxes_target:
        x1, y1, x2, y2 = box
        axes[2].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=2, edgecolor="cyan", facecolor="none"))
    axes[2].set_title(f"Cropped back to target\n({len(pred_boxes_target)} instance(s) found)", fontsize=9)
    axes[2].axis("off")

    axes[3].imshow(target_img)
    for box in pred_boxes_target:
        x1, y1, x2, y2 = box
        axes[3].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=2, edgecolor="orange", facecolor="none"))
    axes[3].set_title(f"Tightened bbox only, per instance ({len(pred_boxes_target)})", fontsize=9)
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--refs-dir", required=True, help="Folder of full reference images")
    p.add_argument("--refs-labels", required=True, help="Folder of YOLO .txt labels (same stem as ref images)")
    p.add_argument("--class-id", type=int, required=True, help="Class id to pull ref instances for")
    p.add_argument("--targets-dir", required=True, help="Folder of target images (can include ref images too)")
    p.add_argument("--orientation", choices=["vertical", "horizontal"], default="vertical")
    p.add_argument("--split-ratio", type=float, default=SPLIT_RATIO)
    p.add_argument("--canvas-size", type=int, default=CANVAS_SIZE)
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--mask-threshold", type=float, default=0.2)
    p.add_argument("--output-dir", default="output_sam3_fewshot")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    ref_instances = collect_ref_instances(Path(args.refs_dir), Path(args.refs_labels), args.class_id)
    print(f"[refs] {len(ref_instances)} ref instance(s) for class {args.class_id}")
    if not ref_instances:
        print("[abort] No ref instances found.")
        return

    targets_dir = Path(args.targets_dir)
    target_paths = sorted(p for p in targets_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"[targets] {len(target_paths)} target image(s)")
    if not target_paths:
        print("[abort] No target images found.")
        return

    print(f"[model] Loading SAM3: {SAM3_MODEL_ID} ...")
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID, device_map=device)
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)

    total = len(ref_instances) * len(target_paths)
    done = 0
    for target_path in target_paths:
        target_img = Image.open(target_path).convert("RGB")
        target_out_dir = output_dir / target_path.stem
        target_out_dir.mkdir(exist_ok=True)

        for ref in ref_instances:
            done += 1
            print(f"\n[{done}/{total}] ref={ref['name']}  target={target_path.name}")

            canvas, placements = create_canvas(
                ref["image"], ref["box"], target_img,
                args.canvas_size, args.orientation, args.split_ratio,
            )
            norm_box = get_norm_box(placements)
            cw, ch = placements["canvas_size"]
            ref_box_canvas_xyxy = norm_cxcywh_to_xyxy_px(norm_box, cw, ch)

            box_xyxy = norm_cxcywh_to_xyxy_px(norm_box, cw, ch)
            inputs = processor(
                images=canvas,
                input_boxes=[[box_xyxy]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            masks_canvas = [m.cpu().numpy().astype(np.uint8) for m in results["masks"]]

            raw_mask_canvas = np.zeros((ch, cw), dtype=np.uint8)
            for m in masks_canvas:
                raw_mask_canvas |= m

            tgt = placements["tgt"]
            tx, ty = tgt["offset"]
            tw, th = tgt["curr_size"]

            # Per-mask: crop to target region, resize to orig target size, tight bbox.
            # Masks with no pixels inside the target region are dropped (ref-side only).
            pred_masks_target = []
            pred_boxes_target = []
            for m in masks_canvas:
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

            combined_mask_target = None
            if pred_masks_target:
                combined_mask_target = np.zeros_like(pred_masks_target[0])
                for m in pred_masks_target:
                    combined_mask_target |= m

            out_path = target_out_dir / f"{ref['name']}.png"
            save_step_figure(
                canvas, ref_box_canvas_xyxy, raw_mask_canvas, target_img,
                combined_mask_target, pred_boxes_target, ref["name"], target_path.name, out_path,
            )

    print(f"\n[done] Output -> {output_dir.resolve()}")


if __name__ == "__main__":
    main()
