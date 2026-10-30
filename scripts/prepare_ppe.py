"""
Rebuild the Construction-PPE eval layout from the public Ultralytics zip:
all images + YOLO labels flattened into <out>/label, then the 150 stems listed in
experiments/ppe_test_split.txt moved to <out>/test (same split as the reported runs).

Usage:
    curl -L -o construction-ppe.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
    unzip -q construction-ppe.zip -d ppe_raw
    python scripts/prepare_ppe.py --src ppe_raw --out data
"""

import argparse
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Extracted zip root (searched recursively)")
    p.add_argument("--out", default="data")
    p.add_argument("--split", default=str(Path(__file__).resolve().parent.parent / "experiments" / "ppe_test_split.txt"))
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    label_dir, test_dir = out / "label", out / "test"
    for d in (label_dir, test_dir):
        if d.exists() and any(d.iterdir()):
            sys.exit(f"[abort] {d} is not empty")
        d.mkdir(parents=True, exist_ok=True)

    images: dict[str, Path] = {}
    for img in src.rglob("*"):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        if img.stem in images:
            sys.exit(f"[abort] duplicate image stem {img.stem}: {images[img.stem]} vs {img}")
        images[img.stem] = img
    labels = {t.stem: t for t in src.rglob("*.txt") if "labels" in t.parts}
    print(f"[scan] {len(images)} image(s), {len(labels)} label file(s)")

    test_names = [l.strip() for l in Path(args.split).read_text().splitlines() if l.strip()]
    test_stems = {Path(n).stem for n in test_names}
    missing = test_stems - images.keys()
    if missing:
        sys.exit(f"[abort] {len(missing)} test image(s) not in zip, e.g. {sorted(missing)[:5]}")

    n_label = n_test = n_nolabel = 0
    for stem, img in sorted(images.items()):
        dst = test_dir if stem in test_stems else label_dir
        shutil.copy2(img, dst / img.name)
        if stem in labels:
            shutil.copy2(labels[stem], dst / f"{stem}.txt")
        else:
            n_nolabel += 1
        n_test += dst is test_dir
        n_label += dst is label_dir

    print(f"[done] pool {n_label} → {label_dir}, test {n_test} → {test_dir}, {n_nolabel} image(s) without label")


if __name__ == "__main__":
    main()
