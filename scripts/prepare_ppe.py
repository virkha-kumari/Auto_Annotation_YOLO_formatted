"""
Rebuild the Construction-PPE eval layout from the public Ultralytics zip (2026-11-03 release:
800 train / 200 val, 12 classes). Official split, flattened:
images/train + labels/train → <out>/label (labelled pool), images/val + labels/val → <out>/test.
Also writes <out>/classes.txt (one name per line, from the zip's yaml) and copies the yaml.

Usage:
    curl -L -o construction-ppe.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
    unzip -q construction-ppe.zip -d ppe_raw
    python scripts/prepare_ppe.py --src ppe_raw --out data
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = {"train": "label", "val": "test"}  # zip split → output folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Extracted zip root (searched recursively)")
    p.add_argument("--out", default="data")
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    for d in SPLITS.values():
        if (out / d).exists() and any((out / d).iterdir()):
            sys.exit(f"[abort] {out / d} is not empty")
        (out / d).mkdir(parents=True, exist_ok=True)

    yamls = list(src.rglob("*.yaml"))
    if len(yamls) != 1:
        sys.exit(f"[abort] expected one dataset yaml, found {len(yamls)}")
    names = yaml.safe_load(yamls[0].read_text())["names"]
    (out / "classes.txt").write_text("\n".join(names[i] for i in sorted(names)) + "\n")
    shutil.copy2(yamls[0], out / "data.yaml")

    seen: set[str] = set()
    for split, dst_name in SPLITS.items():
        dst = out / dst_name
        n_img = n_nolabel = 0
        for img in sorted(src.rglob(f"images/{split}/*")):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            if img.stem in seen:
                sys.exit(f"[abort] duplicate image stem across splits: {img.stem}")
            seen.add(img.stem)
            shutil.copy2(img, dst / img.name)
            label = img.parent.parent.parent / "labels" / split / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, dst / label.name)
            else:
                n_nolabel += 1
            n_img += 1
        print(f"[done] {split}: {n_img} image(s) → {dst} ({n_nolabel} without label)")
    print(f"[classes] {len(names)}: {' '.join(names[i] for i in sorted(names))}")


if __name__ == "__main__":
    main()
