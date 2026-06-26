"""
Debug YOLOe proposals + DINOv2 scoring — 3-panel visualization.

For each source group:
  Panel 1 (left):   refer_image with reference bboxes (cyan)
  Panel 2 (middle): YOLOe proposals on target — color by YOLOe conf
  Panel 3 (right):  Same proposals — color by DINOv2 cosine similarity vs query crops

VRAM rule: YOLOe runs first (all images), then del + empty_cache, then DINOv2 loads.
Saves each figure to output_results/ in cwd.

python test/debug_yoloe_dinov2.py --queries "path/to/crops/cls0" --target "path/to/target.jpg" --source-images "path/to/source/images" --labels "path/to/source/labels"

"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


YOLOE_MODEL_ID = "yoloe-11l-seg.pt"
DINOV2_MODEL_ID = "facebook/dinov2-base"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_STEM_RE = re.compile(r"^(.+)_cls(\d+)_(\d+)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def conf_color(conf: float) -> str:
    """YOLOe confidence color coding (same as debug_yoloe.py)."""
    if conf >= 0.15:
        return "lime"
    elif conf >= 0.05:
        return "yellow"
    return "red"


def dino_color(sim: float) -> str:
    """DINOv2 similarity color coding."""
    if sim >= 0.65:
        return "lime"
    elif sim >= 0.4:
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# DINOv2 embedding helpers
# ---------------------------------------------------------------------------

def build_dinov2(device: str):
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(device).eval()
    return processor, model


@torch.no_grad()
def embed_images(imgs: list[Image.Image], processor, model, device: str, batch_size: int = 16) -> torch.Tensor:
    """Return L2-normalised [N, D] embeddings for a list of PIL images."""
    all_embs = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        out = model(**inputs)
        # CLS token
        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def crop_pil(img: Image.Image, box: list[float]) -> Image.Image:
    """Crop PIL image by [x1, y1, x2, y2], clamped to image bounds."""
    iw, ih = img.size
    x1, y1, x2, y2 = (
        max(0, int(box[0])), max(0, int(box[1])),
        min(iw, int(box[2])), min(ih, int(box[3])),
    )
    if x2 <= x1 or y2 <= y1:
        return img  # degenerate — return full image as fallback
    return img.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="YOLOe proposals + DINOv2 scoring — 3-panel")
    p.add_argument("--queries", required=True, help="Folder of query crop images")
    p.add_argument("--target", required=True, help="Target image")
    p.add_argument("--source-images", required=True, help="Folder of raw source images")
    p.add_argument("--labels", required=True, help="Folder of YOLO .txt label files")
    p.add_argument("--yoloe-conf", type=float, default=0.1, help="YOLOe confidence threshold (default: 0.1)")
    p.add_argument("--nms-iou", type=float, default=0.6, help="NMS IoU threshold for YOLOe proposals (default: 0.5)")
    p.add_argument("--wbf-score", type=float, default=0.1, help="Minimum WBF fused score to keep a final box (default: 0.1)")
    p.add_argument("--dino-thresh", type=float, default=0.55, help="DINOv2 cosine similarity threshold — proposals below this are discarded (default: 0.55)")
    p.add_argument("--yoloe-model", default=YOLOE_MODEL_ID)
    p.add_argument("--output-dir", default="output_results torque placed", help="Folder to save figures (default: output_results)")
    args = p.parse_args()

    query_dir = Path(args.queries)
    target_path = Path(args.target)
    source_dir = Path(args.source_images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    target_img = Image.open(target_path).convert("RGB")
    print(f"[target] {target_path.name}  size={target_img.size}")

    exts = {".jpg", ".jpeg", ".png"}
    crop_paths = sorted(cp for cp in query_dir.iterdir() if cp.suffix.lower() in exts)
    print(f"[crops] {len(crop_paths)} crop(s) in {query_dir}")

    # Group crops by source image
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

    # -----------------------------------------------------------------------
    # PHASE 1: Run YOLOe on all source groups — collect results
    # -----------------------------------------------------------------------
    print("[phase 1] Running YOLOe proposals ...")
    print(f"[model] Loading YOLOe: {args.yoloe_model} ...")
    yoloe_model = YOLO(args.yoloe_model)

    # results_per_stem: stem -> {bboxes, proposals, src_path, crop_paths}
    results_per_stem: dict[str, dict] = {}

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

        yolo_results = yoloe_model.predict(
            source=str(target_path),
            refer_image=str(src_path),
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            conf=args.yoloe_conf,
            iou=args.nms_iou,
            verbose=False,
        )

        proposals = []
        if yolo_results and yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes
            for j in range(len(boxes)):
                proposals.append((boxes.xyxy[j].tolist(), float(boxes.conf[j])))
        proposals.sort(key=lambda x: -x[1])

        crop_names = ", ".join(c.name for c in crops)
        print(f"[{stem}] {len(bboxes)} ref bbox | {len(proposals)} proposal(s) | crops: {crop_names}")
        for box, conf in proposals:
            print(f"  yoloe_conf={conf:.4f}  box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

        results_per_stem[stem] = {
            "src_path": src_path,
            "src_img": src_img,
            "bboxes": bboxes,
            "proposals": proposals,
            "crop_paths": crops,
        }

    # VRAM rule: unload YOLOe before loading DINOv2
    del yoloe_model
    torch.cuda.empty_cache()
    print("\n[vram] YOLOe unloaded.\n")

    # -----------------------------------------------------------------------
    # PHASE 2: DINOv2 scoring of proposals
    # -----------------------------------------------------------------------
    print("[phase 2] Loading DINOv2 for proposal scoring ...")
    dino_processor, dino_model = build_dinov2(device)

    # Embed all query crops once (shared prototype pool)
    query_imgs = [Image.open(cp).convert("RGB") for cp in crop_paths]
    print(f"[dino] Embedding {len(query_imgs)} query crop(s) ...")
    query_embs = embed_images(query_imgs, dino_processor, dino_model, device)  # [N_crops, D]

    for stem, data in results_per_stem.items():
        proposals = data["proposals"]

        if not proposals:
            dino_scores = []
        else:
            # Crop each proposal region from the target image
            proposal_crops = [crop_pil(target_img, box) for box, _ in proposals]
            prop_embs = embed_images(proposal_crops, dino_processor, dino_model, device)  # [N_props, D]

            # Per-proposal: max cosine similarity across all query crops
            # query_embs: [N_crops, D],  prop_embs: [N_props, D]
            sim_matrix = prop_embs @ query_embs.T  # [N_props, N_crops]
            dino_scores = sim_matrix.max(dim=1).values.tolist()  # best match per proposal

        print(f"\n[{stem}] DINOv2 scores vs YOLOe conf (dino-thresh={args.dino_thresh}):")
        kept_proposals = []
        kept_dino_scores = []
        for i, ((box, yconf), dsim) in enumerate(zip(proposals, dino_scores)):
            status = "KEEP" if dsim >= args.dino_thresh else "DROP"
            print(f"  prop {i:02d}  yoloe={yconf:.4f}  dino={dsim:.4f}  [{status}]")
            if dsim >= args.dino_thresh:
                kept_proposals.append((box, yconf))
                kept_dino_scores.append(dsim)

        print(f"  → {len(kept_proposals)}/{len(proposals)} proposals passed dino-thresh")
        data["dino_scores"] = kept_dino_scores
        data["proposals"] = kept_proposals

    # VRAM cleanup
    del dino_model
    torch.cuda.empty_cache()
    print("\n[vram] DINOv2 unloaded.\n")

    # -----------------------------------------------------------------------
    # PHASE 3: WBF — aggregate all kept proposals across all source groups
    # -----------------------------------------------------------------------
    print("[phase 3] Running WBF across all source groups ...")
    from ensemble_boxes import weighted_boxes_fusion

    iw, ih = target_img.size

    # Collect all kept proposals from every source group
    all_boxes_list = []   # one list of boxes per source group (normalised 0-1)
    all_scores_list = []  # DINOv2 scores
    all_labels_list = []  # all class 0

    for stem, data in results_per_stem.items():
        proposals = data["proposals"]
        dino_scores = data["dino_scores"]
        if not proposals:
            continue
        norm_boxes = [
            [box[0] / iw, box[1] / ih, box[2] / iw, box[3] / ih]
            for box, _ in proposals
        ]
        all_boxes_list.append(norm_boxes)
        all_scores_list.append(dino_scores)
        all_labels_list.append([0] * len(proposals))

    if all_boxes_list:
        wbf_boxes, wbf_scores, _ = weighted_boxes_fusion(
            all_boxes_list,
            all_scores_list,
            all_labels_list,
            iou_thr=args.nms_iou,
            skip_box_thr=0.0,
        )
        # Denormalise back to pixel coords
        wbf_boxes_px = [
            [b[0] * iw, b[1] * ih, b[2] * iw, b[3] * ih]
            for b, s in zip(wbf_boxes, wbf_scores)
            if s >= args.wbf_score
        ]
        wbf_scores = [s for s in wbf_scores if s >= args.wbf_score]
        print(f"  WBF produced {len(wbf_boxes_px)} final box(es) (wbf-score>={args.wbf_score})")
        for i, (b, s) in enumerate(zip(wbf_boxes_px, wbf_scores)):
            print(f"  wbf {i:02d}  score={s:.4f}  box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")
    else:
        wbf_boxes_px = []
        wbf_scores = []
        print("  No proposals survived — WBF skipped.")

    # -----------------------------------------------------------------------
    # PHASE 4: Save 3-panel figures + resultant image
    # -----------------------------------------------------------------------
    print("[phase 3] Saving figures ...")

    for stem, data in results_per_stem.items():
        src_img = data["src_img"]
        bboxes = data["bboxes"]
        proposals = data["proposals"]
        dino_scores = data["dino_scores"]

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        # --- Panel 1: refer_image with ref bboxes ---
        axes[0].imshow(src_img)
        axes[0].set_title(f"refer_image: {stem}\n{len(bboxes)} ref bbox(es)", fontsize=9)
        for bb in bboxes:
            rect = patches.Rectangle(
                (bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                linewidth=2, edgecolor="cyan", facecolor="none",
            )
            axes[0].add_patch(rect)
        axes[0].axis("off")

        # --- Panel 2: YOLOe proposals — color by YOLOe conf ---
        axes[1].imshow(target_img)
        axes[1].set_title(
            f"YOLOe proposals  ({len(proposals)} passed conf>={args.yoloe_conf})\n"
            "green>=0.15  yellow>=0.05  red<0.05",
            fontsize=9,
        )
        for box, conf in proposals:
            color = conf_color(conf)
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor=color, facecolor="none",
            )
            axes[1].add_patch(rect)
            axes[1].text(
                box[0], box[1] - 3, f"{conf:.3f}",
                color=color, fontsize=7,
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )
        axes[1].axis("off")

        # --- Panel 3: DINOv2 scores — same boxes, color by DINOv2 sim ---
        axes[2].imshow(target_img)
        axes[2].set_title(
            f"DINOv2 similarity  ({len(proposals)} passed thresh>={args.dino_thresh})\n"
            "green>=0.65  yellow>=0.40  red<0.40",
            fontsize=9,
        )
        for i, (box, _conf) in enumerate(proposals):
            sim = dino_scores[i] if i < len(dino_scores) else 0.0
            color = dino_color(sim)
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor=color, facecolor="none",
            )
            axes[2].add_patch(rect)
            axes[2].text(
                box[0], box[1] - 3, f"{sim:.3f}",
                color=color, fontsize=7,
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )
        axes[2].axis("off")

        plt.suptitle(
            f"{stem}  |  target: {target_path.name}",
            fontsize=10,
        )
        plt.tight_layout()

        out_path = output_dir / f"{stem}_yoloe_dino.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_path}")

    # --- Resultant image: WBF final boxes on target ---
    fig_r, ax_r = plt.subplots(1, 1, figsize=(12, 7))
    ax_r.imshow(target_img)
    ax_r.set_title(
        f"WBF result — {len(wbf_boxes_px)} final box(es)  "
        f"(yoloe-conf={args.yoloe_conf}, dino-thresh={args.dino_thresh}, nms-iou={args.nms_iou})\n"
        f"target: {target_path.name}",
        fontsize=9,
    )
    for b, s in zip(wbf_boxes_px, wbf_scores):
        rect = patches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax_r.add_patch(rect)
        ax_r.text(
            b[0], b[1] - 3, f"{s:.3f}",
            color="lime", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )
    ax_r.axis("off")
    plt.tight_layout()
    resultant_path = output_dir / "resultant_output.png"
    fig_r.savefig(resultant_path, dpi=150, bbox_inches="tight")
    plt.close(fig_r)
    print(f"[saved] resultant → {resultant_path}")

    print(f"\n[done] All figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
