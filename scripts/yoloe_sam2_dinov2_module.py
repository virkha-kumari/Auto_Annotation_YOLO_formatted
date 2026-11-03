"""
Auto-annotation pipeline — production script.
YOLOe → SAM2 → DINOv2 → WBF → containment filter → YOLO .txt output.

All selected classes run in ONE pipeline pass:
  Phase 1 — YOLOe:   load once → per source frame: ALL class bboxes in one visual_prompts call
                      → proposals tagged with yoloe cls index → unload
                      → cross-stem dedup per class (IoU >= --yoloe-iou keeps highest conf)
  Phase 2 — SAM2:    load once → ref masks per class + ALL target proposal masks → unload
                      (masks box-local + bit-packed; proposal masks cached to disk per target)
  Phase 3 — DINOv2:  load once → proto bank per class → score proposals → unload
                      (crops re-cut from images on the fly, streamed across targets in full batches)
  Phase 4 — WBF + containment filter → merged YOLO .txt per target (all classes)

Reference instances = exactly the crops in --queries-dirs. Crop names must be
<stem>_cls<id>_<line_idx> where line_idx is the raw label-file line index
(as written by scripts/extract_crops_labelled.py / extract_crops_varied.py).

Usage:
    python scripts/yoloe_sam2_dinov2_module.py \\
        --queries-dirs  "D:/crops/cls0"  "D:/crops/cls6" \\
        --class-ids     0  6 \\
        --targets-dir   "D:/unlabeled/images" \\
        --source-images "D:/labelled/images" \\
        --labels        "D:/labelled/labels" \\
        --output-dir    "D:/output"
"""

import argparse
import json
import pickle
import re
import shutil
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tqdm import tqdm


YOLOE_MODEL_ID  = "yoloe-11l-seg.pt"
DINOV2_MODEL_ID = "facebook/dinov2-base"
SAM2_MODEL_ID   = "facebook/sam2.1-hiera-base-plus"
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_STEM_RE    = re.compile(r"^(.+)_cls(\d+)_(\d+)$")
DINO_PATCH_GRID = 16
DINO_INPUT      = 224   # square resize, no center crop — whole crop maps onto the 16×16 patch grid


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


def crop_label_row(crop_path: Path, labels_dir: Path) -> tuple[float, float, float, float] | None:
    """(cx, cy, w, h) of the exact label line the crop was cut from (crop name = <stem>_cls<id>_<line_idx>)."""
    m = CROP_STEM_RE.match(crop_path.stem)
    if not m:
        return None
    src_stem, cls_id, line_idx = m.group(1), int(m.group(2)), int(m.group(3))
    label_path = labels_dir / (src_stem + ".txt")
    if not label_path.exists():
        return None
    lines = label_path.read_text().splitlines()
    parts = lines[line_idx].split() if line_idx < len(lines) else []
    if len(parts) < 5 or int(parts[0]) != cls_id:
        print(f"[warn] {crop_path.name}: label line {line_idx} is not cls{cls_id} — "
              f"re-extract crops (names must use raw label line index)")
        return None
    return float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def rows_to_bboxes_px(
    rows: list[tuple[float, float, float, float]],
    img_size: tuple[int, int],
    padding: float,
) -> list[np.ndarray]:
    iw, ih = img_size
    bboxes = []
    for cx, cy, w, h in rows:
        pad_x, pad_y = w * padding, h * padding
        x1 = max(0,      int((cx - w / 2 - pad_x) * iw))
        y1 = max(0,      int((cy - h / 2 - pad_y) * ih))
        x2 = min(iw - 1, int((cx + w / 2 + pad_x) * iw))
        y2 = min(ih - 1, int((cy + h / 2 + pad_y) * ih))
        bboxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
    return bboxes


def bbox_int(img_np: np.ndarray, bbox) -> tuple[int, int, int, int]:
    """Clamped int (x1, y1, x2, y2) — one convention for mask slicing and crop re-cutting."""
    ih, iw = img_np.shape[:2]
    return max(0, int(bbox[0])), max(0, int(bbox[1])), min(iw, int(bbox[2])), min(ih, int(bbox[3]))


def make_masked_crop(img_np: np.ndarray, local_mask: np.ndarray, bbox) -> Image.Image:
    """Bbox crop with background zeroed by the box-local bool mask (mask also passed to DINOv2)."""
    x1, y1, x2, y2 = bbox_int(img_np, bbox)
    crop = img_np[y1:y2, x1:x2].copy()
    crop[~local_mask] = 0
    return Image.fromarray(crop)


def make_bbox_crop(img_np: np.ndarray, bbox) -> Image.Image:
    """Raw bbox crop — no mask, no resize. Used for small objects."""
    x1, y1, x2, y2 = bbox_int(img_np, bbox)
    return Image.fromarray(img_np[y1:y2, x1:x2])


def local_mask(img_np: np.ndarray, mask_hw: np.ndarray, bbox) -> np.ndarray:
    """Full-image mask → box-local bool mask."""
    x1, y1, x2, y2 = bbox_int(img_np, bbox)
    return mask_hw[y1:y2, x1:x2] > 0


def pack_mask(mask: np.ndarray) -> dict:
    """Box-local bool mask → 1 bit/pixel."""
    return {"bits": np.packbits(mask, axis=None), "shape": mask.shape}


def unpack_mask(packed: dict) -> np.ndarray:
    shape = packed["shape"]
    return np.unpackbits(packed["bits"], count=int(np.prod(shape))).reshape(shape).astype(bool)


def warn_skip(where: str, e: Exception, log_path: Path) -> None:
    """Warn + append traceback to errors.log; caller skips the batch. Non-OOM CUDA errors leave the
    CUDA context unusable (every later batch would fail too) → re-raised instead."""
    msg = f"[WARN] {where}: {type(e).__name__}: {e} — batch skipped"
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n" + traceback.format_exc() + "\n")
    if not isinstance(e, torch.cuda.OutOfMemoryError) and "CUDA" in str(e):
        print(f"[ERROR] {where}: non-OOM CUDA error — aborting (see {log_path})")
        raise e
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def p90_area(areas: list[float]) -> float:
    """90th-percentile bbox area (w*h normalised) of the selected ref instances.
    If p90 < --small-obj-thresh the class is small — even its largest typical instances
    are tiny, so SAM2 masking adds no signal."""
    if not areas:
        return 1.0
    areas = sorted(areas)
    return areas[min(int(len(areas) * 0.9), len(areas) - 1)]


def xyxy_to_yolo(box: list[float], iw: int, ih: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / iw
    cy = (y1 + y2) / 2 / ih
    w  = (x2 - x1) / iw
    h  = (y2 - y1) / ih
    return cx, cy, w, h


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 masked patch pooling
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_patch_sel(mask: np.ndarray) -> np.ndarray:
    """(256,) bool: patches >= 50% covered by the (non-empty) mask; if none, the most-covered patch(es)."""
    cov = cv2.resize(mask.astype(np.float32), (DINO_PATCH_GRID, DINO_PATCH_GRID),
                     interpolation=cv2.INTER_AREA).ravel()
    sel = cov >= 0.5
    return sel if sel.any() else cov >= cov.max()


def prep_dino(crop: Image.Image, mask: np.ndarray | None) -> tuple[Image.Image, np.ndarray | None]:
    """Shrink to DINOv2 input before buffering: 224×224 crop (same BICUBIC square resize the processor
    does) + (256,) patch selection instead of the full mask. Keeps a batch's RAM independent of image size."""
    sel = None if mask is None else mask_to_patch_sel(mask)
    return crop.resize((DINO_INPUT, DINO_INPUT), Image.BICUBIC), sel


@torch.no_grad()
def embed_batch(
    crops: list[Image.Image],
    sels:  list[np.ndarray | None],
    processor,
    model,
    device: str,
) -> torch.Tensor:
    """
    One DINOv2 forward on prep_dino() outputs, L2-normalised, returned on CPU.
    sel None  → mean of all tokens (CLS + patches) on raw bbox crop — small objects
    sel given → masked patch pooling — mean of patch tokens selected by the mask
    """
    inputs = processor(images=crops, return_tensors="pt", do_center_crop=False,
                       size={"height": DINO_INPUT, "width": DINO_INPUT}).to(device)
    hs         = model(**inputs).last_hidden_state
    patch_toks = hs[:, 1:, :]

    batch_embs = []
    for b, sel in enumerate(sels):
        if sel is None:
            emb = hs[b:b+1].mean(dim=1)
        else:
            emb = patch_toks[b][torch.from_numpy(sel).to(patch_toks.device)].mean(dim=0, keepdim=True)
        batch_embs.append(F.normalize(emb, dim=-1))
    return torch.cat(batch_embs, dim=0).cpu()


def embed_crops(crops, masks, processor, model, device, batch_size: int = 32,
                desc: str = "embedding") -> torch.Tensor:
    all_embs = []
    for i in tqdm(range(0, len(crops), batch_size), desc=f"  [dino] {desc}", unit="batch", leave=False):
        prepped = [prep_dino(c, m) for c, m in zip(crops[i:i + batch_size], masks[i:i + batch_size])]
        all_embs.append(embed_batch([c for c, _ in prepped], [s for _, s in prepped],
                                    processor, model, device))
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def auto_dino_batch_size(model, device: str, frac: float = 0.7, cap: int = 4096) -> int:
    """Largest DINOv2 batch keeping total VRAM use under `frac` of the GPU: per-sample peak memory
    measured from two probe batches, remaining budget filled. CPU → 32."""
    if device != "cuda":
        return 32
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    base  = torch.cuda.memory_allocated()
    peaks = {}
    for n in (8, 32):
        torch.cuda.reset_peak_memory_stats()
        model(pixel_values=torch.zeros(n, 3, DINO_INPUT, DINO_INPUT, device=device))
        peaks[n] = torch.cuda.max_memory_allocated() - base
    per_sample = max((peaks[32] - peaks[8]) / 24, 1.0)
    fixed      = max(peaks[8] - 8 * per_sample, 0.0)
    budget     = min(frac * total - base, free - (1 - frac) * total) - fixed  # never eat others' VRAM
    bs = int(max(budget, 0) // per_sample)
    bs = max(8, min(cap, bs - bs % 8))
    torch.cuda.empty_cache()
    print(f"[dino] auto batch size {bs} (~{per_sample / 2**20:.1f} MiB/sample, "
          f"{frac:.0%} of {total / 2**30:.1f} GiB)")
    return bs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — YOLOe (multi-class, batched targets per stem)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_yoloe(
    class_source_groups: dict[int, dict[str, list[tuple]]],
    src_path_map:        dict[str, Path],
    target_paths:        list[Path],
    yoloe_model,
    yoloe_conf:          float,
    nms_iou:             float,
    mask_padding:        float,
    target_batch_size:   int = 8,
    log_path:            Path = Path("errors.log"),
) -> tuple[dict[Path, list[tuple[list[float], float, int, str]]], int]:
    """
    Outer loop = stems, inner loop = target batches.
    Per stem: get_vpe(refer_image) bakes VPE into model → plain predict(source=batch).
    ~3x speedup vs per-target outer loop (confirmed by experiments/test_yoloe_batch.py: 2.91x).
    Returns (proposals_per_target: {target: [(box, yoloe_conf, cls_idx, src_stem), ...]}, n_skipped_chunks)
    """
    import ultralytics
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    ultralytics.utils.LOGGER.setLevel("WARNING")

    class_ids  = sorted(class_source_groups.keys())
    idx_to_cls = {i: c for i, c in enumerate(class_ids)}
    all_stems  = sorted({stem for sg in class_source_groups.values() for stem in sg})

    # Pre-build visual_prompts per stem ONCE
    stem_prompts: dict[str, tuple[Path, dict]] = {}
    for stem in all_stems:
        if stem not in src_path_map:
            continue
        src_path = src_path_map[stem]
        with Image.open(src_path) as im:
            img_size = im.size
        bboxes_out: list[list[float]] = []
        cls_out:    list[int]         = []
        for cls_idx, cls_id in idx_to_cls.items():
            rows = class_source_groups.get(cls_id, {}).get(stem)
            if not rows:
                continue
            for b in rows_to_bboxes_px(rows, img_size, mask_padding):
                bboxes_out.append(b.tolist())
                cls_out.append(cls_idx)
        if bboxes_out:
            stem_prompts[stem] = (src_path, dict(bboxes=bboxes_out, cls=cls_out))

    print(f"[phase1] {len(stem_prompts)}/{len(all_stems)} stems have bbox prompts")
    proposals_per_target: dict[Path, list] = {t: [] for t in target_paths}
    target_strs = [str(t) for t in target_paths]
    n_skipped   = 0

    # stem → local_idx → global cls_idx mapping (needed after VPE bake loses cls info)
    stem_local_to_cls: dict[str, dict[int, int]] = {}
    for stem, (_, vp) in stem_prompts.items():
        unique_cls = sorted(set(vp["cls"]))
        stem_local_to_cls[stem] = {local: global_cls for local, global_cls in enumerate(unique_cls)}

    for stem, (src_path, visual_prompts) in tqdm(stem_prompts.items(),
                                                   desc="[phase1] stems", unit="stem"):
        local_to_cls = stem_local_to_cls[stem]
        num_cls      = len(local_to_cls)

        # Bake VPE for this stem into the model once
        if not isinstance(yoloe_model.predictor, YOLOEVPSegPredictor):
            yoloe_model.predictor = YOLOEVPSegPredictor(
                overrides={
                    "task": yoloe_model.model.task,
                    "mode": "predict",
                    "save": False,
                    "verbose": False,
                    "batch": 1,
                    "imgsz": 640,
                },
                _callbacks=yoloe_model.callbacks,
            )
        yoloe_model.model.model[-1].nc = num_cls
        yoloe_model.model.names = [f"object{i}" for i in range(num_cls)]
        yoloe_model.predictor.set_prompts(visual_prompts.copy())
        yoloe_model.predictor.setup_model(model=yoloe_model.model)
        vpe = yoloe_model.predictor.get_vpe(str(src_path))
        yoloe_model.model.set_classes(yoloe_model.model.names, vpe)
        yoloe_model.task = "segment"
        yoloe_model.predictor = None  # reset — now plain detection with baked VPE

        # Predict all targets in chunks
        for i in range(0, len(target_strs), target_batch_size):
            chunk       = target_strs[i:i + target_batch_size]
            chunk_paths = target_paths[i:i + target_batch_size]
            try:
                results = yoloe_model.predict(
                    source=chunk,
                    conf=yoloe_conf,
                    iou=nms_iou,
                    verbose=False,
                    agnostic_nms=True,
                )
            except Exception as e:
                warn_skip(f"YOLOe stem {stem} chunk {i}", e, log_path)
                n_skipped += 1
                continue

            for j, r in enumerate(results):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                tgt_path = chunk_paths[j]
                for k in range(len(r.boxes)):
                    box       = r.boxes.xyxy[k].tolist()
                    conf      = float(r.boxes.conf[k])
                    local_idx = int(r.boxes.cls[k]) if r.boxes.cls is not None else 0
                    cls_idx   = local_to_cls.get(local_idx, local_idx)
                    proposals_per_target[tgt_path].append((box, conf, cls_idx, stem))

    total = sum(len(v) for v in proposals_per_target.values())
    print(f"[phase1] {total} proposal(s) across {len(target_paths)} target(s)")
    return proposals_per_target, n_skipped


def dedup_proposals(
    proposals: list[tuple[list[float], float, int, str]],
    iou_thresh: float,
) -> list[tuple[list[float], float, int, str]]:
    """Cross-stem duplicates: every stem re-detects the same objects. Per class, NMS at iou_thresh
    keeps the highest-conf box. iou_thresh >= 1 disables."""
    if len(proposals) < 2 or iou_thresh >= 1.0:
        return proposals
    from torchvision.ops import batched_nms
    keep = batched_nms(
        torch.tensor([p[0] for p in proposals], dtype=torch.float32),
        torch.tensor([p[1] for p in proposals], dtype=torch.float32),
        torch.tensor([p[2] for p in proposals]),
        iou_thresh,
    )
    return [proposals[i] for i in keep.tolist()]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — SAM2
# ─────────────────────────────────────────────────────────────────────────────

def build_sam2_predictor(sam2_model_id: str, device: str):
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2      = build_sam2_hf(sam2_model_id, device=device)
    predictor = SAM2ImagePredictor(sam2)
    return sam2, predictor


def run_phase2a_sam2_refs(
    class_source_groups: dict[int, dict[str, list[tuple]]],
    src_path_map:        dict[str, Path],
    predictor,
    mask_padding:        float,
    sam_score_min:       float,
    sam_area_min:        float,
    small_cls:           set[int],
) -> dict[int, list[dict]]:
    """Ref instances per class as {stem, bbox, mask, name} — no crops held; DINOv2 re-cuts them.
    Small classes skip SAM2 (mask None); others keep a packed box-local SAM2 mask."""
    result: dict[int, list[dict]] = {}

    for cls_id, source_groups in class_source_groups.items():
        refs: list[dict] = []
        skipped = 0
        is_small = cls_id in small_cls

        for stem, rows in tqdm(source_groups.items(),
                               desc=f"[phase2a] cls{cls_id} refs", unit="frame", leave=False):
            src_img   = Image.open(src_path_map[stem]).convert("RGB")
            img_np    = np.array(src_img)
            bboxes_px = rows_to_bboxes_px(rows, src_img.size, mask_padding)

            if is_small:
                for bi, bbox in enumerate(bboxes_px):
                    refs.append({"stem": stem, "bbox": bbox, "mask": None, "name": f"{stem}_inst{bi}"})
            else:
                predictor.set_image(img_np)
                for bi, bbox in enumerate(bboxes_px):
                    raw_masks, scores, _ = predictor.predict(box=bbox, multimask_output=True)
                    best      = int(np.argmax(scores))
                    mask      = raw_masks[best].astype(np.uint8)
                    sam_score = float(scores[best])
                    bbox_area  = max((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]), 1.0)
                    area_ratio = float(mask.sum()) / bbox_area
                    if sam_score < sam_score_min or area_ratio < sam_area_min:
                        skipped += 1; continue
                    mask_crop = local_mask(img_np, mask, bbox)
                    if not mask_crop.any():
                        skipped += 1; continue
                    refs.append({"stem": stem, "bbox": bbox, "mask": pack_mask(mask_crop),
                                 "name": f"{stem}_inst{bi}"})

        mode = "bbox-crop/mean-pool" if is_small else "sam2-masked-patch"
        print(f"[phase2a] cls{cls_id} [{mode}]: {len(refs)} ref(s) ({skipped} skipped)")
        result[cls_id] = refs

    return result


def run_phase2b_sam2_proposals(
    proposals:  list[tuple[list[float], float, int, str]],
    target_img: Image.Image,
    predictor,
    small_cls:  set[int],
    idx_to_cls: dict[int, int],
    cache_path: Path,
) -> list[tuple[list[float], float, int, str]]:
    """
    Mask all proposals of one target; masks go to disk (cache_path), not RAM.
    Small-class proposals: no SAM2 (mask None).
    Normal proposals: batch through SAM2, box-local mask packed; empty masks dropped.
    Returns kept proposals; cache_path holds their masks in the same order.
    """
    target_np = np.array(target_img)

    small_props  = [(i, p) for i, p in enumerate(proposals) if idx_to_cls.get(p[2], p[2]) in small_cls]
    normal_props = [(i, p) for i, p in enumerate(proposals) if idx_to_cls.get(p[2], p[2]) not in small_cls]

    mask_map: dict[int, dict | None] = {i: None for i, _ in small_props}

    # normal objects — SAM2 in chunks to avoid OOM on images with many proposals
    SAM2_BOX_CHUNK = 50
    if normal_props:
        predictor.set_image(target_np)
        skipped = 0
        for chunk_start in range(0, len(normal_props), SAM2_BOX_CHUNK):
            chunk = normal_props[chunk_start:chunk_start + SAM2_BOX_CHUNK]
            boxes_np = np.array([p[0] for _, p in chunk], dtype=np.float32)
            raw_masks, scores, _ = predictor.predict(box=boxes_np, multimask_output=True)
            if raw_masks.ndim == 3:  # SAM2 squeezes the box dim when the chunk has a single box
                raw_masks, scores = raw_masks[None], scores[None]
            for k, (i, (box, *_)) in enumerate(chunk):
                best_idx  = int(np.argmax(scores[k]))
                mask_crop = local_mask(target_np, raw_masks[k, best_idx], box)
                if not mask_crop.any():
                    skipped += 1; continue
                mask_map[i] = pack_mask(mask_crop)
            del raw_masks, scores
        kept_normal = len(normal_props) - skipped
        print(f"[phase2b] normal: {kept_normal}/{len(normal_props)} kept | small: {len(small_props)} bbox-crop")
    else:
        print(f"[phase2b] small: {len(small_props)} bbox-crop (no SAM2 needed)")

    kept_idx = [i for i in range(len(proposals)) if i in mask_map]
    with open(cache_path, "wb") as f:
        pickle.dump([mask_map[i] for i in kept_idx], f, protocol=pickle.HIGHEST_PROTOCOL)
    return [proposals[i] for i in kept_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — WBF + containment filter
# ─────────────────────────────────────────────────────────────────────────────

def run_wbf(
    kept:       list[tuple[list[float], float, float, str, str]],
    target_img: Image.Image,
    nms_iou:    float,
    wbf_score:  float,
) -> tuple[list[list[float]], list[float]]:
    """Single-model WBF (conf_type='avg'): fused score = mean combined score of the cluster.
    No vote-count boost — one model means min(n_models, n_boxes) / n_models = 1."""
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
    return boxes_px, scores_out


def filter_contained_boxes(
    boxes_px: list[list[float]],
    scores:   list[float],
    thresh:   float,
) -> tuple[list[list[float]], list[float]]:
    n    = len(boxes_px)
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
            inter  = (ix2-ix1) * (iy2-iy1)
            area_i = (bi[2]-bi[0]) * (bi[3]-bi[1])
            area_j = (bj[2]-bj[0]) * (bj[3]-bj[1])
            ratio  = inter / max(min(area_i, area_j), 1.0)
            if ratio > thresh:
                drop.add(i if scores[i] < scores[j] else j)
    surviving_boxes  = [b for k, b in enumerate(boxes_px) if k not in drop]
    surviving_scores = [s for k, s in enumerate(scores)   if k not in drop]
    print(f"[containment] {len(drop)} removed → {len(surviving_boxes)} remaining")
    return surviving_boxes, surviving_scores


def save_preview(
    target_img: Image.Image,
    boxes_by_cls: dict[int, tuple[list[list[float]], list[float]]],
    out_path:   Path,
) -> None:
    COLORS = [(255,140,0),(0,200,255),(0,255,100),(255,80,80),(180,0,255)]
    img  = target_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for cls_id, (boxes, scores) in boxes_by_cls.items():
        color = COLORS[cls_id % len(COLORS)]
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1+4, y1+4), f"cls{cls_id} {score:.2f}", fill=color)
    img.save(out_path)


def save_yolo_label(
    boxes_by_cls: dict[int, tuple[list[list[float]], list[float]]],
    img_w: int,
    img_h: int,
    out_path: Path,
) -> None:
    lines = []
    for cls_id, (boxes, scores) in boxes_by_cls.items():
        for box in boxes:
            cx, cy, w, h = xyxy_to_yolo(box, img_w, img_h)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    out_path.write_text("\n".join(lines))
    print(f"[saved] {out_path}  ({len(lines)} box(es))")


# ─────────────────────────────────────────────────────────────────────────────
# Args + main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Auto-annotation: YOLOe → SAM2 → DINOv2 → WBF → YOLO .txt (multi-class)"
    )
    p.add_argument("--queries-dirs",         nargs="+", required=True,
                   help="One crops folder per class (same order as --class-ids)")
    p.add_argument("--class-ids",            nargs="+", type=int, required=True,
                   help="Class IDs matching --queries-dirs order")
    p.add_argument("--targets-dir",          required=True)
    p.add_argument("--source-images",        required=True)
    p.add_argument("--labels",               required=True)
    p.add_argument("--output-dir",           default="output_results")
    p.add_argument("--yoloe-conf",           type=float, default=0.06)
    p.add_argument("--nms-iou",              type=float, default=0.45)
    p.add_argument("--yoloe-iou",            type=float, default=0.90,
                   help="Cross-stem proposal dedup: per class, IoU >= this keeps the highest-conf box "
                        "before SAM2 (1.0 = off).")
    p.add_argument("--dino-thresh",          type=float, default=0.65)
    p.add_argument("--wbf-score",            type=float, default=0.10)
    p.add_argument("--result-thresh",        type=float, default=0.50)
    p.add_argument("--containment-thresh",   type=float, default=0.70)
    p.add_argument("--sam2-mask-padding",    type=float, default=0.05)
    p.add_argument("--sam-score-min",        type=float, default=0.50)
    p.add_argument("--sam-area-min",         type=float, default=0.10)
    p.add_argument("--small-obj-thresh",     type=float, default=0.01,
                   help="Classes whose ref crops have p90 bbox area (w*h normalised) below this skip SAM2 "
                        "and use mean-pool embedding on raw bbox crop instead of masked-patch pooling.")
    p.add_argument("--no-mask-classes",      nargs="+", type=int, default=[],
                   help="Class IDs to force into small-object handling (skip SAM2, mean-pool on raw "
                        "bbox crop for DINOv2) regardless of --small-obj-thresh. For classes too "
                        "hard/unreliable to mask (e.g. thin/reflective/occluded objects).")
    p.add_argument("--no-preview",           action="store_true",
                   help="Skip saving preview JPGs — labels/summary.json only.")
    p.add_argument("--yoloe-model",          default=YOLOE_MODEL_ID)
    p.add_argument("--yoloe-batch-size",     type=int,   default=8,
                   help="Targets per YOLOe predict call after VPE bake-in (default 8).")
    p.add_argument("--sam2-model",           default=SAM2_MODEL_ID)
    p.add_argument("--dino-batch-size",      type=int,   default=0,
                   help="DINOv2 crops per forward. 0 = auto (fills ~70%% of GPU VRAM).")
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.queries_dirs) != len(args.class_ids):
        print("[abort] --queries-dirs and --class-ids must have same length")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  class-ids={args.class_ids}")

    target_paths = sorted(
        p for p in Path(args.targets_dir).iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if not target_paths:
        print("[abort] No images in targets-dir"); return
    print(f"[targets] {len(target_paths)} image(s)")

    # Build per-class source groups: cls → stem → [(cx, cy, w, h)] of exactly the selected crops
    labels_dir = Path(args.labels)
    class_source_groups: dict[int, dict[str, list[tuple]]] = {}
    src_path_map: dict[str, Path] = {}

    for cls_id, queries_dir in zip(args.class_ids, args.queries_dirs):
        crop_paths = sorted(
            cp for cp in Path(queries_dir).iterdir() if cp.suffix.lower() in IMG_EXTS
        )
        sg: dict[str, list[tuple]] = {}
        for cp in crop_paths:
            src = resolve_source_image(cp, Path(args.source_images))
            row = crop_label_row(cp, labels_dir) if src else None
            if row is None:
                continue
            sg.setdefault(src.stem, []).append(row)
            src_path_map[src.stem] = src
        class_source_groups[cls_id] = sg
        n_inst = sum(len(r) for r in sg.values())
        print(f"[crops] cls{cls_id}: {len(crop_paths)} crop(s) → {n_inst} ref instance(s) "
              f"from {len(sg)} source frame(s)")

    if not any(class_source_groups.values()):
        print("[abort] No source frames resolved"); return

    # Detect small-object classes from the selected ref instances — skip SAM2, mean-pool embedding
    small_cls: set[int] = set()
    for cls_id in args.class_ids:
        p90 = p90_area([w * h for rows in class_source_groups[cls_id].values() for _, _, w, h in rows])
        mode = "bbox-crop/mean-pool" if p90 < args.small_obj_thresh else "SAM2/masked-patch"
        print(f"[cls{cls_id}] p90 bbox area={p90:.5f} → {mode}")
        if p90 < args.small_obj_thresh:
            small_cls.add(cls_id)
    forced = set(args.no_mask_classes) & set(args.class_ids)
    if forced:
        print(f"[no-mask-classes] forcing small-object handling for cls: {sorted(forced)}")
        small_cls |= forced

    class_ids  = sorted(class_source_groups.keys())
    idx_to_cls = {i: c for i, c in enumerate(class_ids)}

    t0 = time.time()
    timing: dict[str, float] = {}
    errors_log = output_dir / "errors.log"
    cache_dir  = output_dir / "_mask_cache"   # per-target SAM2 masks — disk, not RAM; removed at the end
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: YOLOe ──────────────────────────────────────────────────────
    t_phase = time.time()
    from ultralytics import YOLO
    print(f"\n[phase1] Loading YOLOe: {args.yoloe_model} ...")
    yoloe_model = YOLO(args.yoloe_model)

    proposals_per_target, n_skipped = run_phase1_yoloe(
        class_source_groups, src_path_map,
        target_paths, yoloe_model,
        args.yoloe_conf, args.nms_iou, args.sam2_mask_padding,
        target_batch_size=args.yoloe_batch_size,
        log_path=errors_log,
    )

    del yoloe_model
    torch.cuda.empty_cache()

    n_before = sum(len(v) for v in proposals_per_target.values())
    for t in proposals_per_target:
        proposals_per_target[t] = dedup_proposals(proposals_per_target[t], args.yoloe_iou)
    n_after = sum(len(v) for v in proposals_per_target.values())
    print(f"[phase1] cross-stem dedup (IoU >= {args.yoloe_iou}): {n_before} → {n_after} proposal(s)")
    timing["phase1_yoloe_s"] = time.time() - t_phase
    print(f"[phase1] Done ({timing['phase1_yoloe_s']:.1f}s)")

    # ── Phase 2: SAM2 ───────────────────────────────────────────────────────
    t_phase = time.time()
    print(f"\n[phase2] Loading SAM2: {args.sam2_model} ...")
    sam2_obj, predictor = build_sam2_predictor(args.sam2_model, device)

    refs_by_cls = run_phase2a_sam2_refs(
        class_source_groups, src_path_map,
        predictor, args.sam2_mask_padding,
        args.sam_score_min, args.sam_area_min,
        small_cls,
    )

    empty_classes = [c for c, refs in refs_by_cls.items() if not refs]
    if empty_classes:
        print(f"[warn] Empty prototype bank for cls: {empty_classes} — those classes skipped")
    if all(not refs for refs in refs_by_cls.values()):
        print("[abort] All classes have empty prototype banks")
        del predictor, sam2_obj; torch.cuda.empty_cache()
        shutil.rmtree(cache_dir, ignore_errors=True)
        return

    # target → kept proposals (box, conf, cls_idx, stem); masks in cache_dir/<target idx>.pkl
    kept_proposals_per_target: dict[Path, list] = {}
    for t_idx, (target_path, proposals) in enumerate(proposals_per_target.items()):
        if not proposals:
            kept_proposals_per_target[target_path] = []
            continue
        target_img = Image.open(target_path).convert("RGB")
        kept_proposals_per_target[target_path] = run_phase2b_sam2_proposals(
            proposals, target_img, predictor, small_cls, idx_to_cls,
            cache_dir / f"{t_idx}.pkl",
        )
    del proposals_per_target

    del predictor, sam2_obj
    torch.cuda.empty_cache()
    timing["phase2_sam2_s"] = time.time() - t_phase
    print(f"[phase2] Done ({timing['phase2_sam2_s']:.1f}s)")

    # ── Phase 3: DINOv2 ─────────────────────────────────────────────────────
    t_phase = time.time()
    from transformers import AutoImageProcessor, AutoModel
    print(f"\n[phase3] Loading DINOv2: {DINOV2_MODEL_ID} ...")
    try:
        processor  = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
        dino_model = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(device).eval()
    except OSError:
        processor  = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID, local_files_only=True)
        dino_model = AutoModel.from_pretrained(DINOV2_MODEL_ID, local_files_only=True).to(device).eval()
    dino_bs = args.dino_batch_size if args.dino_batch_size > 0 else auto_dino_batch_size(dino_model, device)

    # Proto bank per class (only classes with refs) — crops re-cut from source frames, one class at a time
    proto_banks: dict[int, tuple[torch.Tensor, list[str]]] = {}
    for cls_id, refs in refs_by_cls.items():
        if not refs:
            continue
        crops, masks = [], []
        for ref in refs:
            img_np = np.array(Image.open(src_path_map[ref["stem"]]).convert("RGB"))
            if ref["mask"] is None:
                crops.append(make_bbox_crop(img_np, ref["bbox"])); masks.append(None)
            else:
                m = unpack_mask(ref["mask"])
                crops.append(make_masked_crop(img_np, m, ref["bbox"])); masks.append(m)
        bank = embed_crops(crops, masks, processor, dino_model, device,
                           batch_size=dino_bs, desc=f"cls{cls_id} proto bank")
        del crops, masks
        proto_banks[cls_id] = (bank, [r["name"] for r in refs])
        print(f"[phase3] cls{cls_id} proto bank: {bank.shape}")

    # Proposals: crops streamed across targets into full batches — only one batch of 224px crops in RAM.
    # Each crop is scored against its own class bank right after its batch; only scalars kept.
    scored_per_target: dict[Path, dict[int, list]] = {t: {c: [] for c in class_ids} for t in target_paths}
    n_seen = {c: 0 for c in class_ids}
    buf: list[tuple] = []   # (target_path, cls_id, proposal, 224px crop, patch sel)

    def flush():
        embs = embed_batch([b[3] for b in buf], [b[4] for b in buf], processor, dino_model, device)
        for (tp, cls_id, (box, yconf, _, stem), _, _), emb in zip(buf, embs):
            bank, ref_names = proto_banks[cls_id]
            dsim, bidx = (emb @ bank.T).max(dim=0)
            if float(dsim) >= args.dino_thresh:
                scored_per_target[tp][cls_id].append((box, yconf, float(dsim), ref_names[int(bidx)], stem))
        buf.clear()

    n_total = sum(len(v) for v in kept_proposals_per_target.values())
    with tqdm(total=n_total, desc="[phase3] scoring proposals", unit="prop") as pbar:
        for t_idx, (target_path, props) in enumerate(kept_proposals_per_target.items()):
            if not props:
                continue
            with open(cache_dir / f"{t_idx}.pkl", "rb") as f:
                packed_masks = pickle.load(f)
            img_np = np.array(Image.open(target_path).convert("RGB"))
            for prop, packed in zip(props, packed_masks):
                pbar.update(1)
                cls_id = idx_to_cls.get(prop[2], prop[2])
                if cls_id not in proto_banks:
                    continue
                n_seen[cls_id] += 1
                if packed is None:
                    crop, sel = prep_dino(make_bbox_crop(img_np, prop[0]), None)
                else:
                    m = unpack_mask(packed)
                    crop, sel = prep_dino(make_masked_crop(img_np, m, prop[0]), m)
                buf.append((target_path, cls_id, prop, crop, sel))
                if len(buf) >= dino_bs:
                    flush()
            del packed_masks, img_np
        if buf:
            flush()

    for cls_id in class_ids:
        n_pass = sum(len(v[cls_id]) for v in scored_per_target.values())
        print(f"[phase3] cls{cls_id}: {n_pass}/{n_seen[cls_id]} passed dino-thresh")

    del dino_model, kept_proposals_per_target
    torch.cuda.empty_cache()
    shutil.rmtree(cache_dir, ignore_errors=True)
    timing["phase3_dinov2_s"] = time.time() - t_phase
    print(f"[phase3] Done ({timing['phase3_dinov2_s']:.1f}s)")

    # ── Phase 4: WBF + containment + save ────────────────────────────────────
    t_phase = time.time()
    summary = {}
    for target_path in target_paths:
        target_name = target_path.name
        target_img  = Image.open(target_path).convert("RGB")
        iw, ih      = target_img.size

        boxes_by_cls: dict[int, tuple[list, list]] = {}
        cls_summary = {}

        for cls_id in class_ids:
            kept = scored_per_target.get(target_path, {}).get(cls_id, [])
            wbf_boxes, wbf_scores = run_wbf(kept, target_img, args.nms_iou, args.wbf_score)
            p3_boxes  = [b for b, s in zip(wbf_boxes, wbf_scores) if s >= args.result_thresh]
            p3_scores = [s for s in wbf_scores if s >= args.result_thresh]
            final_boxes, final_scores = filter_contained_boxes(
                p3_boxes, p3_scores, args.containment_thresh
            )
            boxes_by_cls[cls_id] = (final_boxes, final_scores)
            cls_summary[cls_id]  = {
                "n_proposals": len(kept),
                "n_wbf":       len(wbf_boxes),
                "n_final":     len(final_boxes),
                "boxes": [{"xyxy": b, "score": float(s)}
                          for b, s in zip(final_boxes, final_scores)],
            }
            print(f"[done] {target_name} cls{cls_id}: {len(final_boxes)} box(es)")

        label_path   = output_dir / f"{target_path.stem}.txt"
        preview_path = output_dir / f"{target_path.stem}_preview.jpg"

        save_yolo_label(boxes_by_cls, iw, ih, label_path)
        if not args.no_preview:
            save_preview(target_img, boxes_by_cls, preview_path)

        summary[target_name] = {
            "label_file":   str(label_path),
            "preview_file": None if args.no_preview else str(preview_path),
            "classes":      cls_summary,
            "n_final_total": sum(len(b) for b, _ in boxes_by_cls.values()),
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    timing["phase4_wbf_save_s"] = time.time() - t_phase
    timing["total_s"] = time.time() - t0
    timing["n_targets"] = len(target_paths)
    timing["skipped_batches"] = n_skipped
    (output_dir / "timing.json").write_text(json.dumps(
        {k: round(v, 1) if isinstance(v, float) else v for k, v in timing.items()}, indent=2))
    print("\n[timing] per phase (not cumulative): " +
          "  ".join(f"{k}={v:.1f}" for k, v in timing.items() if k.endswith("_s")))
    if n_skipped:
        print(f"[WARN] {n_skipped} YOLOe batch(es) skipped — results incomplete, see {errors_log}")
    print(f"[done] All targets in {timing['total_s']:.1f}s → {output_dir.resolve()}")


if __name__ == "__main__":
    main()
