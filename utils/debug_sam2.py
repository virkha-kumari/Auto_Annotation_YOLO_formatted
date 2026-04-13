"""
Debug script: visualize SAM2 raw proposals for a single image.
Shows ALL masks before any filtering, and filtered candidates separately.

Usage:
    python utils/debug_sam2.py --image "test_imgs_black_region/vlcsnap-2026-01-20-19h15m49s165.png"
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sam2.build_sam import build_sam2_hf
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

SAM2_MODEL_ID = "facebook/sam2.1-hiera-base-plus"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to a single image")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pil_img = Image.open(args.image).convert("RGB")
    img_np = np.array(pil_img)
    img_h, img_w = img_np.shape[:2]
    img_area = img_h * img_w

    print(f"Image: {args.image}  ({img_w}x{img_h})")
    print(f"Loading SAM2...")

    sam2 = build_sam2_hf(SAM2_MODEL_ID, device=device)
    generator = SAM2AutomaticMaskGenerator(sam2)
    masks = generator.generate(img_np)
    print(f"SAM2 generated {len(masks)} masks")

    print(f"SAM2 raw proposals: {len(masks)}")

    # --- Plot: all SAM2 bboxes, no filtering ---
    fig, ax = plt.subplots(1, figsize=(16, 9))
    fig.suptitle(f"{args.image}  |  {len(masks)} SAM2 masks (NO filtering)", fontsize=10)
    ax.imshow(pil_img)
    for m in masks:
        x, y, w, h = m["bbox"]
        area_pct = round((w * h) / img_area * 100, 1)
        iou = round(float(m.get("predicted_iou", 0)), 2)
        ax.add_patch(patches.Rectangle(
            (x, y), w, h,
            linewidth=1, edgecolor="cyan", facecolor="none", alpha=0.8
        ))
        ax.text(x, y - 3, f"{area_pct}% iou={iou}",
                color="cyan", fontsize=6,
                bbox=dict(facecolor="black", alpha=0.4, pad=1))
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("\n--- All SAM2 proposals ---")
    for i, m in enumerate(masks):
        x, y, w, h = m["bbox"]
        area_pct = round((w * h) / img_area * 100, 2)
        ar = round(w / h, 2) if h > 0 else 0
        iou = round(float(m.get("predicted_iou", 0)), 3)
        print(f"  [{i}] box=[{int(x)},{int(y)},{int(x+w)},{int(y+h)}]  area={area_pct}%  AR={ar}  iou={iou}")


if __name__ == "__main__":
    main()
