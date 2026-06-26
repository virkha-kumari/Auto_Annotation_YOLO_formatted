"""
Extract cropped object images from a YOLO-annotated dataset.
No dedup, no clustering — just save every crop for the specified classes.

Usage:
    python scripts/extract_crops_varied.py \
        --images "D:/path/to/images" \
        --labels "D:/path/to/labels" \
        --classes 0 2 17 \
        --output "D:/path/to/crops" \
        --stride 1 \
        --padding 0.05

Each saved crop is named:
    <frame_stem>_cls<class_id>_<ann_idx>.png
"""

import argparse
import sys
from pathlib import Path

from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract YOLO-annotated crops without any filtering or clustering."
    )
    p.add_argument("--images",   required=True, help="Folder containing frame images")
    p.add_argument("--labels",   required=True, help="Folder containing YOLO .txt label files")
    p.add_argument("--classes",  required=True, nargs="+", type=int,
                   help="Class IDs to extract (space-separated, e.g. 0 2 17)")
    p.add_argument("--output",   required=True, help="Folder to save crops into")
    p.add_argument("--stride",   type=int, default=1,
                   help="Process every Nth image (default: 1)")
    p.add_argument("--padding",  type=float, default=0.0,
                   help="Fractional padding around each box, e.g. 0.05 = 5%% (default: 0)")
    return p.parse_args()


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h, padding=0.0):
    pad_x = w * padding
    pad_y = h * padding
    x1 = max(0,     int((cx - w / 2 - pad_x) * img_w))
    y1 = max(0,     int((cy - h / 2 - pad_y) * img_h))
    x2 = min(img_w, int((cx + w / 2 + pad_x) * img_w))
    y2 = min(img_h, int((cy + h / 2 + pad_y) * img_h))
    return x1, y1, x2, y2


def main():
    args = parse_args()

    images_dir    = Path(args.images)
    labels_dir    = Path(args.labels)
    output_dir    = Path(args.output)
    target_classes = set(args.classes)

    if not images_dir.is_dir():
        sys.exit(f"[error] Images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        sys.exit(f"[error] Labels folder not found: {labels_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for cls_id in target_classes:
        (output_dir / f"cls{cls_id}").mkdir(exist_ok=True)

    img_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not img_paths:
        sys.exit(f"[error] No images found in {images_dir}")

    strided = img_paths[::args.stride]
    print(f"[info] Processing {len(strided)} image(s)...")

    saved = 0
    skipped_zero = 0

    for img_path in strided:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        annotations = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id in target_classes:
                annotations.append((cls_id, float(parts[1]), float(parts[2]),
                                     float(parts[3]), float(parts[4])))
        if not annotations:
            continue

        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size

        for ann_idx, (cls_id, cx, cy, w, h) in enumerate(annotations):
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, iw, ih, args.padding)
            if x2 <= x1 or y2 <= y1:
                skipped_zero += 1
                continue

            crop = img.crop((x1, y1, x2, y2))
            name = f"{img_path.stem}_cls{cls_id}_{ann_idx}.png"
            crop.save(output_dir / f"cls{cls_id}" / name)
            saved += 1

    if skipped_zero:
        print(f"[warn] Skipped {skipped_zero} zero-area box(es)")
    print(f"[done] Saved {saved} crop(s) to {output_dir}")


if __name__ == "__main__":
    main()
