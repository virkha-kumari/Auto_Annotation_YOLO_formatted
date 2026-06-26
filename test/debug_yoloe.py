"""
Debug YOLOe visual-prompt proposals — pure YOLOe, no DINOv2.

For each query crop:
  1. Resolve source image + all class bboxes from label
  2. Run YOLOe visual-prompt detection on target (conf=0.01)
  3. Show side-by-side: source with ref bboxes | target with proposals

python test/debug_yoloe.py --queries "path/to/crops/cls0" --target "path/to/target.jpg" --source-images "path/to/source/images" --labels "path/to/source/labels"
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import torch


YOLOE_MODEL_ID = "yoloe-11l-seg.pt"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_STEM_RE = re.compile(r"^(.+)_cls(\d+)_(\d+)$")


def resolve_source_image(crop_path: Path, source_dir: Path) -> Path | None:
    m = CROP_STEM_RE.match(crop_path.stem)
    if not m:
        return None
    src_stem = m.group(1)
    for ext in IMG_EXTS:
        candidate = source_dir / (src_stem + ext)
        if candidate.exists():
            return candidate
    return None


def resolve_all_class_bboxes(crop_path: Path, labels_dir: Path, source_img: Image.Image) -> list[list[float]] | None:
    m = CROP_STEM_RE.match(crop_path.stem)
    if not m:
        return None
    src_stem = m.group(1)
    cls_id = int(m.group(2))

    label_path = labels_dir / (src_stem + ".txt")
    if not label_path.exists():
        return None

    iw, ih = source_img.size
    bboxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        if int(parts[0]) != cls_id:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = max(0, (cx - w / 2) * iw)
        y1 = max(0, (cy - h / 2) * ih)
        x2 = min(iw, (cx + w / 2) * iw)
        y2 = min(ih, (cy + h / 2) * ih)
        bboxes.append([x1, y1, x2, y2])

    return bboxes if bboxes else None


def main():
    p = argparse.ArgumentParser(description="Debug YOLOe proposals (no DINOv2)")
    p.add_argument("--queries", required=True, help="Folder of query crop images")
    p.add_argument("--target", required=True, help="Target image")
    p.add_argument("--source-images", required=True, help="Folder of raw source images")
    p.add_argument("--labels", required=True, help="Folder of YOLO .txt label files")
    p.add_argument("--conf", type=float, default=0.06, help="YOLOe confidence (default: 0.01)")
    p.add_argument("--yoloe-model", default=YOLOE_MODEL_ID)
    args = p.parse_args()

    query_dir = Path(args.queries)
    target_path = Path(args.target)
    source_dir = Path(args.source_images)
    labels_dir = Path(args.labels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    target_img = Image.open(target_path).convert("RGB")
    print(f"[target] {target_path.name}  size={target_img.size}")

    # Load all crop paths
    exts = {".jpg", ".jpeg", ".png"}
    crop_paths = sorted(p for p in query_dir.iterdir() if p.suffix.lower() in exts)
    print(f"[crops] {len(crop_paths)} crop(s) in {query_dir}")

    print(f"[model] Loading YOLOe: {args.yoloe_model} ...")
    model = YOLO(args.yoloe_model)

    # Group by source image (same source + class = same visual prompt)
    source_groups: dict[str, list[Path]] = {}
    src_path_map: dict[str, Path] = {}

    for cp in crop_paths:
        src = resolve_source_image(cp, source_dir)
        if src is None:
            print(f"[skip] {cp.name} — cannot resolve source")
            continue
        stem = src.stem
        src_path_map[stem] = src
        source_groups.setdefault(stem, []).append(cp)

    print(f"[sources] {len(source_groups)} unique source image(s)\n")

    for stem, crops in source_groups.items():
        src_path = src_path_map[stem]
        src_img = Image.open(src_path).convert("RGB")
        bboxes = resolve_all_class_bboxes(crops[0], labels_dir, src_img)
        if bboxes is None:
            print(f"[{stem}] no bboxes from label — skipping")
            continue

        visual_prompts = dict(
            bboxes=np.array(bboxes),
            cls=np.array([0] * len(bboxes)),
        )

        results = model.predict(
            source=str(target_path),
            refer_image=str(src_path),
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            conf=args.conf,
            iou=0.7,
            verbose=False,
        )

        proposals = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for j in range(len(boxes)):
                proposals.append((boxes.xyxy[j].tolist(), float(boxes.conf[j])))

        # Sort by confidence
        proposals.sort(key=lambda x: -x[1])

        crop_names = ", ".join(c.name for c in crops)
        print(f"[{stem}] {len(bboxes)} ref bbox | {len(proposals)} proposal(s) | crops: {crop_names}")
        for box, conf in proposals:
            print(f"  conf={conf:.4f}  box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Left: source with ref bboxes
        axes[0].imshow(src_img)
        axes[0].set_title(f"Source: {stem}\n{len(bboxes)} ref bbox(es)")
        for bb in bboxes:
            rect = patches.Rectangle(
                (bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                linewidth=2, edgecolor="cyan", facecolor="none",
            )
            axes[0].add_patch(rect)
        axes[0].axis("off")

        # Right: target with proposals
        axes[1].imshow(target_img)
        axes[1].set_title(f"Target: {len(proposals)} proposal(s) (conf>={args.conf})")
        for box, conf in proposals:
            color = "lime" if conf >= 0.15 else ("yellow" if conf >= 0.05 else "red")
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor=color, facecolor="none",
            )
            axes[1].add_patch(rect)
            axes[1].text(box[0], box[1]-3, f"{conf:.3f}", color=color, fontsize=7,
                         bbox=dict(facecolor="black", alpha=0.5, pad=1))
        axes[1].axis("off")

        plt.suptitle(f"{stem}  |  green>=0.15  yellow>=0.05  red<0.05", fontsize=9)
        plt.tight_layout()
        plt.show()
        print()


if __name__ == "__main__":
    main()
