"""
Debug script: visualize SAM3 raw proposals for a single image.
No filtering — shows everything SAM3 proposes.

Usage:
    python utils/debug_sam3.py --image "test_imgs_black_region/vlcsnap-2026-01-20-19h15m49s165.png"
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import pipeline

SAM3_MODEL_ID = "facebook/sam3"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to a single image")
    p.add_argument("--points-per-batch", type=int, default=64)
    return p.parse_args()


def mask_to_bbox(mask_pil):
    """Convert a PIL binary mask to [x1, y1, x2, y2]."""
    arr = np.array(mask_pil)
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


def main():
    args = parse_args()
    device = 0 if torch.cuda.is_available() else -1

    pil_img = Image.open(args.image).convert("RGB")
    img_w, img_h = pil_img.size
    img_area = img_w * img_h

    print(f"Image: {args.image}  ({img_w}x{img_h})")
    print(f"Loading SAM3...")

    generator = pipeline(
        "mask-generation",
        model=SAM3_MODEL_ID,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    outputs = generator(pil_img, points_per_batch=args.points_per_batch)
    masks = outputs["masks"]
    scores = outputs.get("scores", [None] * len(masks))

    print(f"SAM3 generated {len(masks)} masks")

    # Compute bboxes from masks
    proposals = []
    for mask, score in zip(masks, scores):
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        area_pct = round((w * h) / img_area * 100, 2)
        proposals.append({
            "box": bbox,
            "area_pct": area_pct,
            "score": round(float(score), 3) if score is not None else None,
        })

    print(f"Valid proposals (non-empty masks): {len(proposals)}")

    # --- Plot ---
    fig, ax = plt.subplots(1, figsize=(16, 9))
    fig.suptitle(f"{args.image}  |  {len(masks)} SAM3 masks (NO filtering)", fontsize=10)
    ax.imshow(pil_img)
    for p in proposals:
        x1, y1, x2, y2 = p["box"]
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor="cyan", facecolor="none", alpha=0.8
        ))
        label = f"{p['area_pct']}%"
        if p["score"] is not None:
            label += f" s={p['score']}"
        ax.text(x1, y1 - 3, label, color="cyan", fontsize=6,
                bbox=dict(facecolor="black", alpha=0.4, pad=1))
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("\n--- All SAM3 proposals ---")
    for i, p in enumerate(proposals):
        print(f"  [{i}] box={p['box']}  area={p['area_pct']}%  score={p['score']}")


if __name__ == "__main__":
    main()
