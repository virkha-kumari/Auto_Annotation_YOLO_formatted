"""
YOLOe + SAM2 + DINOv2 masked-patch scoring — few-shot auto-annotation debug script.

Pipeline (3 model loads, strictly sequential — never two models in VRAM at once):
  Phase 1 — YOLOe:  load once → for ALL targets: per source frame → bbox visual prompts
                     → proposals on target → unload YOLOe
  Phase 2 — SAM2:   load once →
               Job A (refs):      for each source frame → set_image → bbox predict →
                                  clean masked crops → prototype bank input
               Job B (proposals): for each target → set_image → per proposal bbox →
                                  SAM2 predict → masked crop (drop if empty)
               → unload SAM2
  Phase 3 — DINOv2: load once →
               embed ref masked crops → prototype bank [N_refs, 768]
               for each target: embed proposal masked crops → cosine sim →
               filter by --dino-thresh → unload DINOv2
  Phase 4 — WBF + figures (no model): consolidated score = 0.3×yoloe + 0.7×dino → WBF

DINOv2 embedding: masked patch pooling
    patch tokens last_hidden_state[:, 1:, :] → reshaped to 16×16 grid
    mask resized to 16×16 → mean pool tokens inside mask
    fallback to CLS if mask covers zero patch tokens

Usage:
    python test/debug_yoloe_sam2_dino.py \\
        --queries      "D:/path/to/crops/cls0" \\
        --targets-dir  "D:/path/to/target/images" \\
        --source-images "D:/path/to/source/frames" \\
        --labels       "D:/path/to/labels"
"""

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


YOLOE_MODEL_ID  = "yoloe-11l-seg.pt"
DINOV2_MODEL_ID = "facebook/dinov2-base"
SAM2_MODEL_ID   = "facebook/sam2.1-hiera-base-plus"
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_STEM_RE    = re.compile(r"^(.+)_cls(\d+)_(\d+)$")
DINO_PATCH_GRID = 16   # DINOv2-base: 224px / 14px = 16×16 patch grid

THUMB = 128
COLS  = 10


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def resolve_class_bboxes_padded(
    crop_path: Path,
    labels_dir: Path,
    src_img: Image.Image,
    padding: float,
) -> list[np.ndarray] | None:
    """Return all padded xyxy bbox arrays for the crop's class from its label file."""
    m = CROP_STEM_RE.match(crop_path.stem)
    if not m:
        return None
    src_stem, cls_id = m.group(1), int(m.group(2))
    label_path = labels_dir / (src_stem + ".txt")
    if not label_path.exists():
        return None
    iw, ih = src_img.size
    bboxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5 or int(parts[0]) != cls_id:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        pad_x, pad_y = w * padding, h * padding
        x1 = max(0,      int((cx - w / 2 - pad_x) * iw))
        y1 = max(0,      int((cy - h / 2 - pad_y) * ih))
        x2 = min(iw - 1, int((cx + w / 2 + pad_x) * iw))
        y2 = min(ih - 1, int((cy + h / 2 + pad_y) * ih))
        bboxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
    return bboxes if bboxes else None


def make_masked_crop(img_np: np.ndarray, mask_hw: np.ndarray,
                     bbox: np.ndarray) -> Image.Image:
    """Crop bbox region, zero pixels outside mask, return PIL RGB."""
    ih, iw = img_np.shape[:2]
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(iw, int(bbox[2])), min(ih, int(bbox[3]))
    crop = img_np[y1:y2, x1:x2].copy()
    mask_crop = mask_hw[y1:y2, x1:x2]
    crop[mask_crop == 0] = 0
    return Image.fromarray(crop)


def sam2_predict_bbox(predictor, bbox: np.ndarray) -> np.ndarray | None:
    """
    Run SAM2 predict for one bbox. Returns (H,W) uint8 mask or None if empty.
    """
    raw_masks, scores, _ = predictor.predict(box=bbox, multimask_output=True)
    mask = raw_masks[int(np.argmax(scores))].astype(np.uint8)
    return mask if mask.sum() > 0 else None


def dino_color(sim: float) -> str:
    return "lime" if sim >= 0.65 else ("yellow" if sim >= 0.40 else "red")


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 masked patch pooling
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_masked_crops(
    masked_crops: list[Image.Image],
    processor,
    model,
    device: str,
    batch_size: int = 16,
    desc: str = "embedding",
) -> torch.Tensor:
    """
    DINOv2 masked patch pooling.
    Resizes each crop's non-zero region to 16×16 patch grid → mean pool
    patch tokens inside mask. Falls back to CLS if mask covers no patches.
    Returns L2-normalised [N, 768] tensor.
    """
    all_embs = []
    for i in tqdm(range(0, len(masked_crops), batch_size),
                  desc=f"  [dino] {desc}", unit="batch", leave=False):
        batch = masked_crops[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        out    = model(**inputs)
        hs         = out.last_hidden_state          # [B, 1+256, 768]
        cls_tok    = hs[:, 0, :]                    # [B, 768]
        patch_toks = hs[:, 1:, :]                   # [B, 256, 768]

        batch_embs = []
        for b in range(len(batch)):
            crop_np = np.array(batch[b].convert("L"))
            if crop_np.max() == 0:
                batch_embs.append(F.normalize(cls_tok[b:b+1], dim=-1))
                continue
            mask_small = cv2.resize(
                (crop_np > 0).astype(np.uint8),
                (DINO_PATCH_GRID, DINO_PATCH_GRID),
                interpolation=cv2.INTER_NEAREST,
            ).ravel()
            patch_sel = patch_toks[b][mask_small.astype(bool)]
            if len(patch_sel) == 0:
                emb = F.normalize(cls_tok[b:b+1], dim=-1)
            else:
                emb = F.normalize(patch_sel.mean(dim=0, keepdim=True), dim=-1)
            batch_embs.append(emb)

        all_embs.append(torch.cat(batch_embs, dim=0).cpu())

    return torch.cat(all_embs, dim=0)   # [N, 768]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — YOLOe → proposals on target
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_yoloe(
    source_groups:  dict[str, list[Path]],
    src_path_map:   dict[str, Path],
    labels_dir:     Path,
    target_path:    str,
    yoloe_model,
    yoloe_conf:     float,
    nms_iou:        float,
    mask_padding:   float,
) -> list[tuple[list[float], float, str]]:
    """
    For each source frame: run YOLOe with bbox visual prompts → proposals on target.
    Returns flat list of (box_xyxy, yoloe_conf, src_stem).
    YOLOe seg masks intentionally discarded — SAM2 will re-segment in Phase 2.
    Caller owns model lifecycle (load before, del+empty_cache after all targets).
    """
    import ultralytics
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    ultralytics.utils.LOGGER.setLevel("WARNING")

    all_proposals: list[tuple[list[float], float, str]] = []

    for stem, crops in tqdm(source_groups.items(), desc="[phase1] YOLOe frames", unit="frame"):
        src_path = src_path_map[stem]
        src_img  = Image.open(src_path).convert("RGB")

        bboxes_px = resolve_class_bboxes_padded(crops[0], labels_dir, src_img, mask_padding)
        if not bboxes_px:
            continue

        visual_prompts = dict(
            bboxes=[b.tolist() for b in bboxes_px],
            cls=list(range(len(bboxes_px))),
        )

        try:
            results = yoloe_model.predict(
                source=target_path,
                refer_image=str(src_path),
                visual_prompts=visual_prompts,
                predictor=YOLOEVPSegPredictor,
                conf=yoloe_conf,
                iou=nms_iou,
                verbose=False,
            )
        except Exception as e:
            print(f"  [{stem}] YOLOe error: {e} — skipping")
            continue

        if not results or results[0].boxes is None:
            continue

        for j in range(len(results[0].boxes)):
            box  = results[0].boxes.xyxy[j].tolist()
            conf = float(results[0].boxes.conf[j])
            all_proposals.append((box, conf, stem))

    all_proposals.sort(key=lambda x: -x[1])
    print(f"[phase1] {len(all_proposals)} proposal(s) across all source frames")
    return all_proposals


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — SAM2: segment ref crops (Job B) then target proposals (Job A)
# ─────────────────────────────────────────────────────────────────────────────

def build_sam2_predictor(sam2_model_id: str, device: str):
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2      = build_sam2_hf(sam2_model_id, device=device)
    predictor = SAM2ImagePredictor(sam2)
    return sam2, predictor


def run_phase2a_sam2_refs(
    source_groups: dict[str, list[Path]],
    src_path_map:  dict[str, Path],
    labels_dir:    Path,
    predictor,
    mask_padding:  float,
    sam_score_min: float,
    sam_area_min:  float,
) -> tuple[list[Image.Image], list[str], list[float]]:
    """
    Job B: segment reference frames → masked crops for prototype bank.
    Uses already-loaded predictor (no model load/unload here).
    """
    ref_crops:      list[Image.Image] = []
    ref_names:      list[str]         = []
    ref_sam_scores: list[float]       = []
    skipped = 0

    for stem, crops in tqdm(source_groups.items(), desc="[phase2a] refs", unit="frame", leave=False):
        src_path = src_path_map[stem]
        src_img  = Image.open(src_path).convert("RGB")
        img_np   = np.array(src_img)

        bboxes_px = resolve_class_bboxes_padded(crops[0], labels_dir, src_img, mask_padding)
        if not bboxes_px:
            continue

        predictor.set_image(img_np)

        for bi, bbox in enumerate(bboxes_px):
            raw_masks, scores, _ = predictor.predict(box=bbox, multimask_output=True)
            best      = int(np.argmax(scores))
            mask      = raw_masks[best].astype(np.uint8)
            sam_score = float(scores[best])

            if mask.sum() == 0:
                skipped += 1
                continue

            bbox_area  = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)
            area_ratio = float(mask.sum()) / bbox_area
            if sam_score < sam_score_min or area_ratio < sam_area_min:
                skipped += 1
                continue

            ref_crops.append(make_masked_crop(img_np, mask, bbox))
            ref_names.append(f"{stem}_inst{bi}")
            ref_sam_scores.append(sam_score)

    print(f"[phase2a] {len(ref_crops)} ref instance(s) ({skipped} skipped low-quality)")
    return ref_crops, ref_names, ref_sam_scores


def run_phase2b_sam2_proposals(
    proposals:  list[tuple[list[float], float, str]],
    target_img: Image.Image,
    predictor,
) -> list[tuple[list[float], float, Image.Image, str]]:
    """
    Job A: segment target proposals → masked crops for DINOv2 scoring.
    Uses already-loaded predictor (no model load/unload here).
    """
    target_np = np.array(target_img)
    predictor.set_image(target_np)

    kept: list[tuple[list[float], float, Image.Image, str]] = []
    skipped = 0

    for box, conf, stem in tqdm(proposals, desc="[phase2b] proposals", unit="prop", leave=False):
        bbox = np.array([box[0], box[1], box[2], box[3]], dtype=np.float32)
        mask = sam2_predict_bbox(predictor, bbox)
        if mask is None:
            skipped += 1
            continue
        kept.append((box, conf, make_masked_crop(target_np, mask, bbox), stem))

    print(f"[phase2b] {len(kept)} proposal(s) kept ({skipped} dropped — SAM2 empty mask)")
    return kept



# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — WBF + containment filter
# ─────────────────────────────────────────────────────────────────────────────

def filter_contained_boxes(
    boxes_px:          list[list[float]],
    scores:            list[float],
    containment_thresh: float,
) -> tuple[list[list[float]], list[float]]:
    """
    Remove boxes that are mostly inside another box.
    For each pair (i, j): if intersection/min_area > containment_thresh,
    drop the lower-scoring one.
    Returns surviving (boxes, scores).
    """
    n = len(boxes_px)
    if n <= 1:
        return boxes_px, scores

    drop = set()
    for i in range(n):
        for j in range(i + 1, n):
            if i in drop or j in drop:
                continue
            bi, bj = boxes_px[i], boxes_px[j]
            ix1 = max(bi[0], bj[0]); iy1 = max(bi[1], bj[1])
            ix2 = min(bi[2], bj[2]); iy2 = min(bi[3], bj[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
            area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
            ratio = inter / max(min(area_i, area_j), 1.0)
            if ratio > containment_thresh:
                drop.add(i if scores[i] < scores[j] else j)

    surviving_boxes  = [b for k, b in enumerate(boxes_px) if k not in drop]
    surviving_scores = [s for k, s in enumerate(scores)   if k not in drop]
    print(f"[containment] {len(drop)} box(es) removed → {len(surviving_boxes)} remaining")
    return surviving_boxes, surviving_scores

def run_phase4_wbf(
    kept:        list[tuple[list[float], float, float, str, str]],
    target_img:  Image.Image,
    nms_iou:     float,
    wbf_score:   float,
) -> tuple[list[list[float]], list[float]]:
    from ensemble_boxes import weighted_boxes_fusion

    iw, ih = target_img.size
    if not kept:
        return [], []

    norm_boxes      = [[b[0]/iw, b[1]/ih, b[2]/iw, b[3]/ih] for b, *_ in kept]
    combined_scores = [0.3 * yconf + 0.7 * dsim for _, yconf, dsim, _, _ in kept]

    wbf_boxes, wbf_scores, _ = weighted_boxes_fusion(
        [norm_boxes], [combined_scores], [[0] * len(kept)],
        iou_thr=nms_iou, skip_box_thr=0.0,
    )
    surviving  = [(b, s) for b, s in zip(wbf_boxes, wbf_scores) if s >= wbf_score]
    boxes_px   = [[b[0]*iw, b[1]*ih, b[2]*iw, b[3]*ih] for b, _ in surviving]
    scores_out = [s for _, s in surviving]
    print(f"[phase4] WBF → {len(boxes_px)} final box(es) (wbf-score>={wbf_score})")
    return boxes_px, scores_out


# ─────────────────────────────────────────────────────────────────────────────
# Grid figures
# ─────────────────────────────────────────────────────────────────────────────

def _thumb(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((THUMB, THUMB), Image.LANCZOS)
    out = Image.new("RGB", (THUMB, THUMB), (0, 0, 0))
    out.paste(img, ((THUMB - img.width) // 2, (THUMB - img.height) // 2))
    return out


def save_ref_grid(
    ref_crops: list[Image.Image],
    ref_names: list[str],
    ref_sam_scores: list[float],
    output_path: Path,
) -> None:
    n    = len(ref_crops)
    cols = min(COLS, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            ax.imshow(_thumb(ref_crops[i]))
            inst  = ref_names[i].rsplit("_inst", 1)[-1]
            stem  = ref_names[i].rsplit("_inst", 1)[0][-18:]
            ax.set_title(f"{stem}\ninst{inst} s={ref_sam_scores[i]:.2f}", fontsize=5, pad=2)
        ax.axis("off")
    plt.suptitle(f"SAM2 reference masked crops ({n} instances)", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")


def save_proposal_grid(
    all_scored: list[tuple[Image.Image, float, float, str, bool]],
    dino_thresh: float,
    output_path: Path,
) -> None:
    n    = len(all_scored)
    cols = min(COLS, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.9))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            crop, yconf, dsim, stem, passed = all_scored[i]
            ax.imshow(_thumb(crop))
            color = "lime" if passed else "red"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax.set_title(f"d={dsim:.2f} y={yconf:.2f}\n{stem[-18:]}",
                         fontsize=5, pad=2,
                         color="lime" if passed else "salmon")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.axis("off")
    plt.suptitle(
        f"SAM2-masked proposal crops ({n} total)  "
        f"green=KEEP(≥{dino_thresh})  red=DROP", fontsize=10
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Result figures
# ─────────────────────────────────────────────────────────────────────────────

def save_figures(
    target_img:            Image.Image,
    target_name:           str,
    kept:                  list[tuple[list[float], float, float, str, str]],
    wbf_boxes_px:          list[list[float]],
    wbf_scores:            list[float],
    final_boxes_px:        list[list[float]],
    final_scores:          list[float],
    output_dir:            Path,
    dino_thresh:           float,
    yoloe_conf:            float,
    nms_iou:               float,
    result_panel3_thresh:  float,
    containment_thresh:    float,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(40, 7))

    # Panel 1: proposals passing dino-thresh
    axes[0].imshow(target_img)
    axes[0].set_title(
        f"Proposals passing dino-thresh={dino_thresh}\n"
        f"green≥0.65  yellow≥0.40  red<0.40  ({len(kept)} kept)", fontsize=9
    )
    for box, yconf, dsim, pname, stem in kept:
        color = dino_color(dsim)
        axes[0].add_patch(patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=1.5, edgecolor=color, facecolor="none",
        ))
        axes[0].text(box[0], box[1]-3, f"d={dsim:.2f} y={yconf:.2f}",
                     color=color, fontsize=6,
                     bbox=dict(facecolor="black", alpha=0.5, pad=1))
    axes[0].axis("off")

    # Panel 2: all WBF boxes
    axes[1].imshow(target_img)
    axes[1].set_title(
        f"WBF result — {len(wbf_boxes_px)} final box(es)\n"
        f"yoloe-conf={yoloe_conf}  dino-thresh={dino_thresh}  nms-iou={nms_iou}", fontsize=9
    )
    for b, s in zip(wbf_boxes_px, wbf_scores):
        axes[1].add_patch(patches.Rectangle(
            (b[0], b[1]), b[2]-b[0], b[3]-b[1],
            linewidth=2, edgecolor="lime", facecolor="none",
        ))
        axes[1].text(b[0], b[1]-3, f"{s:.3f}", color="lime", fontsize=8,
                     bbox=dict(facecolor="black", alpha=0.5, pad=1))
    axes[1].axis("off")

    # Panel 3: all WBF boxes >= result_panel3_thresh
    axes[2].imshow(target_img)
    result_boxes = [(b, s) for b, s in zip(wbf_boxes_px, wbf_scores) if s >= result_panel3_thresh]
    for b, s in result_boxes:
        axes[2].add_patch(patches.Rectangle(
            (b[0], b[1]), b[2]-b[0], b[3]-b[1],
            linewidth=3, edgecolor="cyan", facecolor="none",
        ))
        axes[2].text(b[0], b[1]-5, f"{s:.3f}",
                     color="cyan", fontsize=9, fontweight="bold",
                     bbox=dict(facecolor="black", alpha=0.6, pad=2))
    axes[2].set_title(
        f"Result ≥{result_panel3_thresh}  ({len(result_boxes)}/{len(wbf_boxes_px)} box(es))\n"
        f"score = 0.3×yoloe + 0.7×dino → WBF", fontsize=9
    )
    axes[2].axis("off")

    # Panel 4: after containment filter
    axes[3].imshow(target_img)
    for b, s in zip(final_boxes_px, final_scores):
        axes[3].add_patch(patches.Rectangle(
            (b[0], b[1]), b[2]-b[0], b[3]-b[1],
            linewidth=3, edgecolor="orange", facecolor="none",
        ))
        axes[3].text(b[0], b[1]-5, f"{s:.3f}",
                     color="orange", fontsize=9, fontweight="bold",
                     bbox=dict(facecolor="black", alpha=0.6, pad=2))
    axes[3].set_title(
        f"After containment filter (thresh={containment_thresh})\n"
        f"{len(final_boxes_px)} box(es) from {len(wbf_boxes_px)} WBF box(es)", fontsize=9
    )
    axes[3].axis("off")

    plt.suptitle(f"target: {target_name}", fontsize=10)
    plt.tight_layout()
    out = output_dir / "resultant_output.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Args + main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOe + SAM2 + DINOv2 masked-patch few-shot annotation"
    )
    p.add_argument("--queries",           required=True,  help="Folder of reference crop images (one class)")
    p.add_argument("--targets-dir",       required=True,  help="Folder of target images to annotate")
    p.add_argument("--source-images",     required=True,  help="Folder of raw source frames")
    p.add_argument("--labels",            required=True,  help="Folder of YOLO .txt label files")
    p.add_argument("--yoloe-conf",        type=float, default=0.06)
    p.add_argument("--nms-iou",           type=float, default=0.45)
    p.add_argument("--wbf-score",                  type=float, default=0.10)
    p.add_argument("--result-panel3-thresh",       type=float, default=0.5,
                   help="Min WBF score to show in Panel 3 result (default: 0.5)")
    p.add_argument("--final-containment-thresh",   type=float, default=0.7,
                   help="Soft containment ratio to drop nested boxes (default: 0.7)")
    p.add_argument("--dino-thresh",       type=float, default=0.65,
                   help="Min cosine sim to keep a proposal (default: 0.65)")
    p.add_argument("--sam2-mask-padding", type=float, default=0.05,
                   help="Fractional bbox padding before SAM2 prompt (default: 0.05)")
    p.add_argument("--sam-score-min",     type=float, default=0.50,
                   help="Min SAM2 score to keep ref instance (default: 0.50)")
    p.add_argument("--sam-area-min",      type=float, default=0.10,
                   help="Min mask/bbox area ratio to keep ref instance (default: 0.10)")
    p.add_argument("--yoloe-model",       default=YOLOE_MODEL_ID)
    p.add_argument("--sam2-model",        default=SAM2_MODEL_ID)
    p.add_argument("--dino-batch-size",   type=int,   default=16)
    p.add_argument("--output-dir",        default="output_results")
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # collect target images
    targets_dir = Path(args.targets_dir)
    if not targets_dir.is_dir():
        print(f"[abort] targets-dir not found: {targets_dir}")
        return
    target_paths = sorted(p for p in targets_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not target_paths:
        print(f"[abort] No images in targets-dir: {targets_dir}")
        return
    print(f"[targets] {len(target_paths)} image(s) in {targets_dir}")

    # resolve source groups from crop filenames (done once)
    crop_paths = sorted(
        cp for cp in Path(args.queries).iterdir() if cp.suffix.lower() in IMG_EXTS
    )
    print(f"[crops] {len(crop_paths)} reference crop(s)")

    source_groups: dict[str, list[Path]] = {}
    src_path_map:  dict[str, Path]       = {}
    for cp in crop_paths:
        src = resolve_source_image(cp, Path(args.source_images))
        if src is None:
            print(f"  [skip] {cp.name} — cannot resolve source frame")
            continue
        source_groups.setdefault(src.stem, []).append(cp)
        src_path_map[src.stem] = src

    print(f"[sources] {len(source_groups)} unique source frame(s)")
    if not source_groups:
        print("[abort] No source frames resolved.")
        return

    t_global_start = time.time()

    # ── Phase 1: YOLOe — load once, all targets, then unload ───────────────
    from ultralytics import YOLO
    print(f"\n[phase1] Loading YOLOe: {args.yoloe_model} ...")
    yoloe_model = YOLO(args.yoloe_model)

    proposals_per_target: dict[Path, list[tuple[list[float], float, str]]] = {}
    for target_path in target_paths:
        print(f"\n{'='*60}")
        print(f"[phase1] target: {target_path.name}")
        proposals = run_phase1_yoloe(
            source_groups, src_path_map, Path(args.labels),
            str(target_path), yoloe_model,
            args.yoloe_conf, args.nms_iou, args.sam2_mask_padding,
        )
        proposals_per_target[target_path] = proposals

    del yoloe_model
    torch.cuda.empty_cache()
    t1_done = time.time()
    total_props = sum(len(v) for v in proposals_per_target.values())
    print(f"\n[phase1] Done — {total_props} proposal(s) across {len(target_paths)} target(s)  "
          f"({t1_done - t_global_start:.1f}s)")

    # ── Phase 2: SAM2 — refs + all target proposals, then unload ────────────
    print(f"\n[phase2] Loading SAM2: {args.sam2_model} ...")
    sam2_obj, predictor = build_sam2_predictor(args.sam2_model, device)

    # 2a: ref masked crops (prototype bank input)
    ref_crops, ref_names, ref_sam_scores = run_phase2a_sam2_refs(
        source_groups, src_path_map, Path(args.labels),
        predictor, args.sam2_mask_padding,
        args.sam_score_min, args.sam_area_min,
    )
    if not ref_crops:
        print("[abort] Empty prototype bank after SAM2 filtering.")
        del predictor, sam2_obj; torch.cuda.empty_cache()
        return
    save_ref_grid(ref_crops, ref_names, ref_sam_scores,
                  output_dir / "ref_masked_crops.png")

    # 2b: target proposal masked crops
    kept_proposals_per_target: dict[Path, list[tuple[list[float], float, Image.Image, str]]] = {}
    for target_path, proposals in proposals_per_target.items():
        if not proposals:
            kept_proposals_per_target[target_path] = []
            continue
        target_img = Image.open(target_path).convert("RGB")
        kept = run_phase2b_sam2_proposals(proposals, target_img, predictor)
        kept_proposals_per_target[target_path] = kept
        print(f"  [phase2b] {target_path.name}: {len(kept)}/{len(proposals)} proposals kept")

    del predictor, sam2_obj
    torch.cuda.empty_cache()
    t2_done = time.time()
    print(f"[phase2] Done — SAM2 unloaded  ({t2_done - t1_done:.1f}s)")

    # ── Phase 3: DINOv2 — embed refs + all proposals, then unload ───────────
    from transformers import AutoImageProcessor, AutoModel

    print(f"\n[phase3] Loading DINOv2: {DINOV2_MODEL_ID} ...")
    try:
        processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
        dino_model = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(device).eval()
    except OSError:
        processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID, local_files_only=True)
        dino_model = AutoModel.from_pretrained(DINOV2_MODEL_ID, local_files_only=True).to(device).eval()

    print(f"[phase3] Embedding {len(ref_crops)} reference instance(s)...")
    proto_bank = embed_masked_crops(ref_crops, processor, dino_model, device,
                                    batch_size=args.dino_batch_size, desc="prototype bank")
    print(f"[phase3] Prototype bank: {proto_bank.shape}")

    # score all targets
    scored_per_target: dict[Path, tuple[
        list[tuple[list[float], float, float, str, str]],  # kept
        list[tuple[Image.Image, float, float, str, bool]], # all_scored
    ]] = {}
    for target_path, kept_proposals in kept_proposals_per_target.items():
        if not kept_proposals:
            scored_per_target[target_path] = ([], [])
            continue

        prop_crops = [mc for _, _, mc, _ in kept_proposals]
        prop_embs  = embed_masked_crops(prop_crops, processor, dino_model, device,
                                        batch_size=args.dino_batch_size,
                                        desc=f"{target_path.name} proposals")

        sim_matrix           = prop_embs @ proto_bank.T
        best_sims, best_idxs = sim_matrix.max(dim=1)

        kept       = []
        all_scored = []
        print(f"\n[phase3] {target_path.name}  (dino-thresh={args.dino_thresh}):")
        for i, ((box, yconf, mc, stem), dsim, bidx) in enumerate(
            zip(kept_proposals, best_sims.tolist(), best_idxs.tolist())
        ):
            pname  = ref_names[bidx] if bidx < len(ref_names) else str(bidx)
            passed = dsim >= args.dino_thresh
            status = "KEEP" if passed else "DROP"
            print(f"  prop {i:03d}  yoloe={yconf:.4f}  dino={dsim:.4f}  "
                  f"best_proto={pname}  src={stem}  [{status}]")
            if passed:
                kept.append((box, yconf, dsim, pname, stem))
            all_scored.append((mc, yconf, dsim, stem, passed))

        print(f"  → {len(kept)}/{len(kept_proposals)} passed dino-thresh")
        scored_per_target[target_path] = (kept, all_scored)

    del dino_model
    torch.cuda.empty_cache()
    t3_done = time.time()
    print(f"\n[phase3] Done — DINOv2 unloaded  ({t3_done - t2_done:.1f}s)")

    # ── Phase 4: WBF + figures (no model needed) ────────────────────────────
    for target_path in target_paths:
        target_name = target_path.name
        target_img  = Image.open(target_path).convert("RGB")
        kept, all_scored = scored_per_target.get(target_path, ([], []))

        tgt_out = output_dir / target_path.stem
        tgt_out.mkdir(exist_ok=True)

        wbf_boxes_px, wbf_scores = run_phase4_wbf(
            kept, target_img, args.nms_iou, args.wbf_score,
        )
        print(f"[phase4] {target_name}: WBF → {len(wbf_boxes_px)} final box(es)")

        p3_boxes  = [b for b, s in zip(wbf_boxes_px, wbf_scores) if s >= args.result_panel3_thresh]
        p3_scores = [s for s in wbf_scores if s >= args.result_panel3_thresh]
        final_boxes_px, final_scores = filter_contained_boxes(
            p3_boxes, p3_scores, args.final_containment_thresh,
        )

        save_figures(
            target_img, target_name, kept,
            wbf_boxes_px, wbf_scores,
            final_boxes_px, final_scores,
            tgt_out,
            args.dino_thresh, args.yoloe_conf, args.nms_iou,
            args.result_panel3_thresh, args.final_containment_thresh,
        )
        if all_scored:
            save_proposal_grid(all_scored, args.dino_thresh,
                               tgt_out / "proposal_masked_crops.png")

        avg_sim = (sum(d for _, _, d, _, _ in kept) / len(kept)) if kept else 0.0
        print(f"[done] {target_name}: boxes={len(wbf_boxes_px)}  avg_sim={avg_sim:.4f}  → {tgt_out}")

    print(f"\n[done] All targets processed in {time.time()-t_global_start:.1f}s")
    print(f"[done] Output → {output_dir.resolve()}")


if __name__ == "__main__":
    main()
