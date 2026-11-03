"""
SAM3 cross-image few-shot via canvas-composite exemplar prompting.

No native cross-image few-shot API in SAM3 (exemplar boxes only work within
same image). Trick (WongKinYiu/FSS-SAM3): paste ref + target on one canvas,
remap ref bbox to canvas coords, run SAM3 once (box-only exemplar, no text),
crop target half back out.

Refs sharing a source image are grouped onto one canvas as a multi-box prompt
(one SAM3 call per frame, not per instance) — each such frame is one "ref group".

Phases (one large model in VRAM at a time):
  1. DINOv2  — phash dedup + farthest-point sample --dino-proto-size diverse refs/class
               (most-diverse-first); first --max-refs-per-class of them = SAM3 exemplars.
  2. SAM3    — box-prompt masks for proto-bank refs (own image, own box), then
               canvas-composite proposals on every target (targets batched per ref group).
  3. DINOv2  — masked-patch pooling (SAM3 masks) + max-sim vs proto bank;
               combined_score = 0.2*sam3_score + 0.8*dino_sim, gated by --sam3-dino-thresh.
  4. Voting  — duplicates across ref groups merged (IoU >= --dup-iou-thresh);
               final_score = combined_score * voting_ref_groups / ref_groups; containment filter.

Small classes (p90 area of selected refs < --small-obj-thresh, or --no-mask-classes)
skip masking: DINOv2 mean of all tokens on the raw crop, for refs and proposals alike.

Output: output_dir/labels/<stem>.txt, output_dir/<stem>.png (3-panel preview),
summary.json, timing.json, temp_refs/ (selected ref crops + mask overlays).

Usage:
    python scripts/sam3_dinov2_module.py \\
        --refs-dir    "D:/path/to/labelled_ref_images" \\
        --refs-labels "D:/path/to/labelled_ref_images"  (YOLO .txt, same stem) \\
        --class-ids   0 1 2 \\
        --targets-dir "D:/path/to/target_images"
"""

import argparse
import gc
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.patches as patches
from matplotlib.figure import Figure
from PIL import Image

RAM_GUARD_PCT = 90.0


def ram_guard():
    """GC pass if RAM usage crosses RAM_GUARD_PCT."""
    if psutil.virtual_memory().percent >= RAM_GUARD_PCT:
        gc.collect()
from transformers import Sam3Model, Sam3Processor

SAM3_MODEL_ID = "facebook/sam3"
DINOV2_MODEL_ID = "facebook/dinov2-base"
CANVAS_SIZE = 1008
SPLIT_RATIO = 0.5   # ref gets 50% of canvas, target gets 50%
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DINO_PATCH_GRID = 16  # DINOv2-base patch grid for 224px input
DINO_INPUT = 224      # square resize, no center crop — whole crop maps onto the 16×16 patch grid


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


@torch.no_grad()
def auto_dino_batch_size(model, device: str, frac: float = 0.7, cap: int = 4096) -> int:
    """Largest DINOv2 batch keeping total VRAM use under `frac` of the GPU: per-sample peak memory
    measured from two probe batches, remaining budget filled. CPU → 32."""
    if device != "cuda":
        return 32
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    base = torch.cuda.memory_allocated()
    peaks = {}
    for n in (8, 32):
        torch.cuda.reset_peak_memory_stats()
        model(pixel_values=torch.zeros(n, 3, DINO_INPUT, DINO_INPUT, device=device))
        peaks[n] = torch.cuda.max_memory_allocated() - base
    per_sample = max((peaks[32] - peaks[8]) / 24, 1.0)
    fixed = max(peaks[8] - 8 * per_sample, 0.0)
    budget = min(frac * total - base, free - (1 - frac) * total) - fixed  # never eat others' VRAM
    bs = int(max(budget, 0) // per_sample)
    bs = max(8, min(cap, bs - bs % 8))
    torch.cuda.empty_cache()
    print(f"[dino] auto batch size {bs} (~{per_sample / 2**20:.1f} MiB/sample, "
          f"{frac:.0%} of {total / 2**30:.1f} GiB)")
    return bs


def discover_class_ids(refs_labels: Path) -> list[int]:
    ids = set()
    for label_path in refs_labels.glob("*.txt"):
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                ids.add(int(parts[0]))
            except ValueError:
                print(f"[warn] malformed class id in {label_path.name}: {parts[0]!r} — skipped")
    return sorted(ids)


def collect_ref_boxes(refs_dir: Path, refs_labels: Path, class_ids: list[int], box_padding: float = 0.0) -> dict[int, list[dict]]:
    """Metadata-only: parse labels for boxes matching class_ids, read image size from header (no pixel decode) to convert YOLO norm coords to px.
    Full decode deferred until an instance survives diverse-ref selection."""
    wanted = set(class_ids)
    boxes_by_class: dict[int, list[dict]] = {c: [] for c in class_ids}
    for img_path in sorted(refs_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = refs_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        lines = label_path.read_text().splitlines()
        matches = []
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                print(f"[warn] malformed class id in {label_path.name} line {i}: {parts[0]!r} — skipped")
                continue
            if cid not in wanted:
                continue
            matches.append((i, cid, parts))
        if not matches:
            continue
        try:
            with Image.open(img_path) as img:
                iw, ih = img.size  # header-only, no pixel decode
        except Exception as e:
            print(f"[warn] cannot read {img_path.name}: {e} — skipped")
            continue
        if iw <= 0 or ih <= 0:
            print(f"[warn] {img_path.name} has invalid size {iw}x{ih} — skipped")
            continue
        for i, cid, parts in matches:
            try:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                print(f"[warn] malformed box in {label_path.name} line {i} — skipped")
                continue
            x, y, bw, bh = (cx - w / 2) * iw, (cy - h / 2) * ih, w * iw, h * ih  # x,y,w,h px
            if box_padding > 0:
                px, py = bw * box_padding, bh * box_padding
                x1, y1 = max(0.0, x - px), max(0.0, y - py)
                x2, y2 = min(iw, x + bw + px), min(ih, y + bh + py)
                x, y, bw, bh = x1, y1, x2 - x1, y2 - y1
            box_xywh = [x, y, bw, bh]
            boxes_by_class[cid].append({
                "img_path": img_path,
                "box": box_xywh,
                "norm_area": w * h,  # unpadded label area, for the small-object rule
                "name": f"{img_path.stem}_inst{i}",
            })
    return boxes_by_class


def farthest_point_sample(vectors: np.ndarray, k: int) -> list[int]:
    """Greedy max-min diversity selection (cf. scripts/extract_crops_labelled.py)."""
    if k <= 0 or len(vectors) == 0:
        return []
    k = min(k, len(vectors))
    seed = int(np.argmax(np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)))
    selected = [seed]
    dists = np.full(len(vectors), np.inf)
    for _ in range(k - 1):
        last = vectors[selected[-1]]
        d = np.linalg.norm(vectors - last, axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return selected


def xywh_to_int_xyxy(box: list[float]) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return int(x), int(y), int(x + w), int(y + h)


def _load_crop(inst: dict) -> Image.Image:
    """Decode source image on demand, crop box region."""
    with Image.open(inst["img_path"]) as img:
        img = img.convert("RGB")
        return img.crop(xywh_to_int_xyxy(inst["box"]))


def p90_area(areas: list[float]) -> float:
    """90th-percentile bbox area (w*h normalised) of the selected ref instances.
    If p90 < --small-obj-thresh the class is small -- even its largest typical instances
    are tiny, so masking adds no signal (cf. yoloe_sam2_dinov2_module.py)."""
    if not areas:
        return 1.0
    areas = sorted(areas)
    return areas[min(int(len(areas) * 0.9), len(areas) - 1)]


def pack_mask(mask: np.ndarray) -> dict:
    """Box-local bool mask → 1 bit/pixel. Keeps per-det RAM small across thousands of targets."""
    return {"bits": np.packbits(mask, axis=None), "shape": mask.shape}


def unpack_mask(packed: dict) -> np.ndarray:
    shape = packed["shape"]
    return np.unpackbits(packed["bits"], count=int(np.prod(shape))).reshape(shape).astype(bool)


def save_mask_overlay(img_np: np.ndarray, local_mask: np.ndarray, box_int: tuple[int, int, int, int],
                      output_path, pad: float = 0.3):
    """Save image crop (padded window around box) with box-local mask as translucent overlay + box outline.
    Same style as debug_sam3.py panels — for inspecting mask alignment, not a model input."""
    ih, iw = img_np.shape[:2]
    x1, y1, x2, y2 = box_int
    bw, bh = x2 - x1, y2 - y1
    wx1 = max(0, int(x1 - bw * pad))
    wy1 = max(0, int(y1 - bh * pad))
    wx2 = min(iw, int(x2 + bw * pad))
    wy2 = min(ih, int(y2 + bh * pad))
    win_mask = np.zeros((wy2 - wy1, wx2 - wx1), dtype=bool)
    win_mask[y1 - wy1:y1 - wy1 + local_mask.shape[0], x1 - wx1:x1 - wx1 + local_mask.shape[1]] = local_mask

    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    ax.imshow(img_np[wy1:wy2, wx1:wx2])
    ax.imshow(win_mask, alpha=0.5, cmap="jet")
    ax.add_patch(patches.Rectangle((x1 - wx1, y1 - wy1), bw, bh,
                 linewidth=2, edgecolor="lime", facecolor="none"))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")


def _phash_dedup(instances: list[dict], max_dist: int, class_id) -> list[dict]:
    """Perceptual-hash near-dup filter over ref crops (decoded lazily). Growable
    hash matrix, hamming distance <= max_dist rejects as dup of one already kept
    (cf. extract_crops_labelled.py phase1_extract_class)."""
    import imagehash

    hash_mat = np.empty((256, 64), dtype=np.uint8)
    n_hashes = 0
    kept = []
    for inst in tqdm(instances, desc=f"  cls{class_id} phash", unit="crop", leave=False):
        crop = _load_crop(inst)
        ph = imagehash.phash(crop)
        bits = np.array(ph.hash, dtype=np.uint8).ravel()
        if n_hashes > 0:
            dists = (hash_mat[:n_hashes] != bits).sum(axis=1)
            if dists.min() <= max_dist:
                continue
        if n_hashes >= len(hash_mat):
            hash_mat = np.vstack([hash_mat, np.empty_like(hash_mat)])
        hash_mat[n_hashes] = bits
        n_hashes += 1
        kept.append(inst)
    return kept


def select_diverse_refs(ref_boxes_by_class: dict, max_refs: int, dino_proto_size: int, output_dir: Path,
                         device: str, dinov2_batch_size: int = 16,
                         phash_max_dist: int = 0, ref_jpeg_quality: int = 92) -> tuple[dict, dict]:
    """Phash dedup + DINOv2 farthest-point sample dino_proto_size refs/class (most-diverse-first),
    CLS-on-raw-crop embeddings for diversity ranking only (final scoring embeddings come later, see
    mask_ref_crops_sam3() + build_proto_banks()). SAM3's max_refs exemplars are the first N of that
    list (subset, not a separate selection).

    Returns (sam3_groups_by_class, proto_instances_by_class):
      sam3_groups_by_class: {class_id: [{img_path, image, boxes, names}]} grouped by source image.
      proto_instances_by_class: {class_id: [inst, ...]} kept ref instances for the proto bank."""
    from transformers import AutoImageProcessor, AutoModel

    if phash_max_dist > 0:
        deduped = {}
        for class_id, boxes in ref_boxes_by_class.items():
            deduped[class_id] = _phash_dedup(boxes, phash_max_dist, class_id)
            if len(deduped[class_id]) != len(boxes):
                print(f"[refs] class {class_id}: phash dedup {len(boxes)} -> {len(deduped[class_id])}")
        ref_boxes_by_class = deduped  # don't mutate caller's dict

    keep_all = dino_proto_size <= 0  # 0 = no pruning, decode-and-keep every candidate
    print("[refs] Loading DINOv2 for diverse ref selection ...")
    try:
        dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        dmodel = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    except OSError:
        dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
        dmodel = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).to(device).eval()
    if dinov2_batch_size <= 0:
        dinov2_batch_size = auto_dino_batch_size(dmodel, device)

    sam3_groups_by_class = {}
    proto_instances_by_class = {}
    with torch.no_grad():
        for class_id, instances in ref_boxes_by_class.items():
            embs_by_idx: dict[int, np.ndarray] = {}

            def embed_all(insts_with_idx: list[tuple[int, dict]]) -> None:
                for i in tqdm(range(0, len(insts_with_idx), dinov2_batch_size),
                               desc=f"  cls{class_id} dinov2", unit="batch", leave=False):
                    batch = insts_with_idx[i:i + dinov2_batch_size]
                    crops = []
                    for _, inst in batch:  # shrink to the processor's short-edge-256 step at load, not full-res
                        c = _load_crop(inst)
                        s = 256 / max(min(c.size), 1)
                        crops.append(c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))),
                                              Image.BICUBIC))
                    inputs = dproc(images=crops, return_tensors="pt").to(device)
                    out = dmodel(**inputs)
                    batch_embs = out.last_hidden_state[:, 0, :].float().cpu().numpy()
                    for (idx, _), e in zip(batch, batch_embs):
                        embs_by_idx[idx] = e
                    del crops, inputs, out
                    ram_guard()

            if keep_all or len(instances) <= dino_proto_size:
                keep = list(range(len(instances)))
                embed_all(list(enumerate(instances)))
            else:
                embed_all(list(enumerate(instances)))
                idx_order = sorted(embs_by_idx.keys())
                embs = np.stack([embs_by_idx[i] for i in idx_order])
                normed = embs / np.linalg.norm(embs, axis=1, keepdims=True)
                keep_local = farthest_point_sample(normed, dino_proto_size)
                keep = [idx_order[k] for k in keep_local]
                print(f"[refs] class {class_id}: {len(instances)} -> {len(keep)} diverse ref(s) for DINOv2 proto bank: "
                      f"{', '.join(instances[i]['name'] for i in keep)}")

            proto_instances_by_class[class_id] = [instances[i] for i in keep]

            # SAM3 exemplars: subset of the same ordered list (most-diverse-first), capped at max_refs.
            sam3_keep = keep[:max_refs] if max_refs > 0 else keep
            print(f"[refs] class {class_id}: {len(sam3_keep)} ref(s) for SAM3 exemplars (subset of proto bank)")

            temp_refs_dir = output_dir / "temp_refs" / f"cls{class_id}"
            temp_refs_dir.mkdir(parents=True, exist_ok=True)

            # Save every proto-bank crop for inspection (one at a time, never batch-held in RAM) —
            # excludes sam3_keep, which the SAM3 loop below saves via its own full-image decode.
            sam3_keep_set = set(sam3_keep)
            for i in keep:
                if i in sam3_keep_set:
                    continue
                inst = instances[i]
                crop = _load_crop(inst)
                crop.save(temp_refs_dir / f"{inst['name']}.jpg", quality=ref_jpeg_quality)
                del crop

            # Group by source image: shared frame -> one canvas ref-half, one multi-box prompt
            groups_by_path = {}
            for i in sam3_keep:
                inst = instances[i]
                groups_by_path.setdefault(inst["img_path"], []).append(inst)

            kept_groups = []
            for img_path, insts in groups_by_path.items():
                with Image.open(img_path) as img:
                    full_img = img.convert("RGB")
                boxes, names = [], []
                for inst in insts:
                    crop = full_img.crop(xywh_to_int_xyxy(inst["box"]))
                    crop.save(temp_refs_dir / f"{inst['name']}.jpg", quality=ref_jpeg_quality)
                    boxes.append(inst["box"])
                    names.append(inst["name"])
                kept_groups.append({"img_path": img_path, "image": full_img, "boxes": boxes, "names": names})
            sam3_groups_by_class[class_id] = kept_groups
            ram_guard()

    del dmodel, dproc
    if device == "cuda":
        torch.cuda.empty_cache()
    return sam3_groups_by_class, proto_instances_by_class


def create_canvas(ref_img: Image.Image, ref_boxes: list[list[float]], target_img: Image.Image,
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
        {"offset": (s_rect[0], s_rect[1]), "max_dim": (s_rect[2], s_rect[3]), "image": ref_img, "type": "ref", "box": ref_boxes},
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


def get_norm_boxes(placements: dict) -> list[list[float]]:
    """Remap ref's pixel-space boxes into canvas-normalized (cx, cy, w, h) in [0,1], one per ref instance."""
    p = placements["ref"]
    cw, ch = placements["canvas_size"]
    ox, oy = p["offset"]
    sx, sy = p["curr_size"][0] / p["orig_size"][0], p["curr_size"][1] / p["orig_size"][1]
    norm_boxes = []
    for bx, by, bw, bh in p["orig_box"]:
        px, py = bx * sx + ox, by * sy + oy
        norm_boxes.append([(px + bw * sx / 2) / cw, (py + bh * sy / 2) / ch, (bw * sx) / cw, (bh * sy) / ch])
    return norm_boxes


def norm_cxcywh_to_xyxy_px(norm_box: list[float], w: int, h: int) -> list[float]:
    cx, cy, bw, bh = norm_box
    return [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]


def remap_canvas_box_to_target(box_canvas: list[float], placements: dict) -> list[float] | None:
    """Remap a SAM3-predicted box (canvas px coords) into target-image px coords.
    Clips to the target region; returns None if the box doesn't overlap it."""
    tgt = placements["tgt"]
    tx, ty = tgt["offset"]
    tw, th = tgt["curr_size"]
    ow, oh = tgt["orig_size"]
    sx, sy = ow / tw, oh / th

    x1, y1, x2, y2 = box_canvas
    x1, x2 = max(x1, tx), min(x2, tx + tw)
    y1, y2 = max(y1, ty), min(y2, ty + th)
    if x2 <= x1 or y2 <= y1:
        return None
    return [(x1 - tx) * sx, (y1 - ty) * sy, (x2 - tx) * sx, (y2 - ty) * sy]


def crop_result_to_target(placements, masks_canvas, boxes_canvas, scores_canvas, class_id, ref_group,
                          keep_mask: bool):
    """Crop one batch's canvas-space masks/boxes/scores back to target-image space. Returns list
    of dicts: {box: [x1,y1,x2,y2] px (SAM3's own box, not mask-derived), mask: packed box-local
    mask (None if not keep_mask), score: float, class_id: int, ref_group: int}."""
    tgt = placements["tgt"]
    tx, ty = tgt["offset"]
    tw, th = tgt["curr_size"]

    out = []
    for m, box_canvas, score in zip(masks_canvas, boxes_canvas, scores_canvas):
        crop = m[ty:ty + th, tx:tx + tw]
        if not crop.any():
            continue
        box = remap_canvas_box_to_target(box_canvas, placements)
        if box is None:
            continue
        packed = None
        if keep_mask:
            mask_target = np.array(
                Image.fromarray(crop * 255).resize(tgt["orig_size"], Image.NEAREST)
            ) > 0
            x1, y1, x2, y2 = (int(v) for v in box)
            packed = pack_mask(mask_target[y1:y2, x1:x2])
        out.append({"box": [float(v) for v in box], "mask": packed,
                     "score": float(score), "class_id": class_id, "ref_group": ref_group})
    return out


def mask_ref_crops_sam3(proto_instances_by_class: dict, small_cls: set, model, processor, device: str,
                         output_dir: Path, threshold: float = 0.0, mask_threshold: float = 0.6,
                         box_chunk: int = 8, min_match_containment: float = 0.25,
                         log_path: Path = Path("errors.log")) -> int:
    """Box-prompt SAM3 per ref instance (own image, own box). Adds 'mask' (packed, box-local) in place;
    small classes and SAM3 misses get None. SAM3 output has no positional link to input box order
    (fixed-size DETR decoder) -- each prompt box matched to best-containment output box, not zipped by index.
    Returns the number of skipped (failed) SAM3 chunks."""
    n_skipped = 0
    if threshold <= 0.0:
        print(f"  [refs] warning: --threshold={threshold} keeps every SAM3 query regardless of "
              f"confidence, including detections nowhere near the ref instance -- relying entirely "
              f"on containment-matching (min_match_containment={min_match_containment}) to reject them.")
    for class_id, instances in proto_instances_by_class.items():
        if class_id in small_cls:
            for inst in instances:
                inst["mask"] = None
            continue

        temp_refs_dir = output_dir / "temp_refs" / f"cls{class_id}"
        temp_refs_dir.mkdir(parents=True, exist_ok=True)

        groups_by_path: dict = {}
        for inst in instances:
            groups_by_path.setdefault(inst["img_path"], []).append(inst)

        n_masked = n_failed = 0
        for img_path, insts in tqdm(groups_by_path.items(), desc=f"  cls{class_id} ref masking", unit="img", leave=False):
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb)

            for chunk_start in range(0, len(insts), box_chunk):
                chunk = insts[chunk_start:chunk_start + box_chunk]
                boxes_xyxy = []
                for inst in chunk:
                    x, y, w, h = inst["box"]
                    boxes_xyxy.append([x, y, x + w, y + h])

                inputs = processor(
                    images=[img_rgb],
                    input_boxes=[boxes_xyxy],
                    input_boxes_labels=[[1] * len(boxes_xyxy)],
                    return_tensors="pt",
                ).to(model.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
                if "input_boxes" in inputs:
                    inputs["input_boxes"] = inputs["input_boxes"].to(model.dtype)

                try:
                    with torch.no_grad():
                        outputs = model(**inputs)
                except Exception as e:
                    del inputs
                    warn_skip(f"SAM3 ref masking cls{class_id} {img_path.name} "
                              f"({len(chunk)} ref(s) dropped from bank)", e, log_path)
                    for inst in chunk:
                        inst["mask"] = None
                    n_failed += len(chunk)
                    n_skipped += 1
                    continue

                results = processor.post_process_instance_segmentation(
                    outputs, threshold=threshold, mask_threshold=mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]
                out_boxes = results["boxes"].float().cpu().numpy().tolist()
                out_masks = [m.to(torch.uint8).cpu().numpy() > 0 for m in results["masks"]]

                # No positional correspondence between prompt boxes and output boxes -- match each
                # prompt box to its best-containment output box instead.
                for inst in chunk:
                    x, y, w, h = inst["box"]
                    prompt_box = [x, y, x + w, y + h]
                    best_containment, best_j = 0.0, -1
                    for j, out_box in enumerate(out_boxes):
                        _, containment = box_iou(prompt_box, out_box)
                        if containment > best_containment:
                            best_containment, best_j = containment, j
                    box_int = xywh_to_int_xyxy(inst["box"])
                    local = (out_masks[best_j][box_int[1]:box_int[3], box_int[0]:box_int[2]]
                             if best_j >= 0 and best_containment >= min_match_containment else None)
                    if local is None or not local.any():
                        inst["mask"] = None
                        n_failed += 1
                        continue
                    inst["mask"] = pack_mask(local)
                    n_masked += 1
                    save_mask_overlay(img_np, local, box_int, temp_refs_dir / f"{inst['name']}_mask.jpg")

                del inputs, outputs
                ram_guard()

        print(f"[refs] cls{class_id}: {n_masked} ref(s) masked, {n_failed} SAM3 miss(es) — dropped from proto bank")
    return n_skipped


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0, 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    containment = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return iou, containment


def vote_and_filter(dets: list[dict], n_groups: int, containment_thresh: float,
                    iou_thresh: float) -> tuple[list[dict], list[dict]]:
    """Per-class voting across SAM3 ref groups + containment filter. dets already restricted to one class.
    1. Duplicates: by combined_score desc, a box with IoU >= iou_thresh to an already-kept box merges into
       it (kept box unchanged, its voter set gains the duplicate's ref_group).
    2. final_score = combined_score * n_voting_ref_groups / n_groups — found from more exemplars ranks higher.
    3. Containment: by final_score desc, a box contained (>= thresh) in a kept box is dropped.
    Returns (kept, rejected)."""
    clusters, rejected = [], []
    for d in sorted(dets, key=lambda d: d["combined_score"], reverse=True):
        for c in clusters:
            if box_iou(d["box"], c["box"])[0] >= iou_thresh:
                c["voters"].add(d["ref_group"])
                rejected.append(d)
                break
        else:
            d["voters"] = {d["ref_group"]}
            clusters.append(d)
    for c in clusters:
        c["votes"] = len(c.pop("voters"))
        c["final_score"] = c["combined_score"] * c["votes"] / max(n_groups, 1)

    kept = []
    for d in sorted(clusters, key=lambda d: d["final_score"], reverse=True):
        if any(box_iou(d["box"], k["box"])[1] >= containment_thresh for k in kept):
            rejected.append(d)
        else:
            kept.append(d)
    return kept, rejected


# tab10 cycle, indexed by class_id % 10 — stable color per class across both panels
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def class_color(class_id: int) -> str:
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def load_class_names(classes_file: str | None) -> dict[int, str]:
    """classes.txt: one class name per line, line N = name for class_id N (standard YOLO names file)."""
    if not classes_file:
        return {}
    p = Path(classes_file)
    if not p.is_file():
        print(f"[warn] --classes-file not found: {classes_file} — falling back to class ids")
        return {}
    names = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return {i: name for i, name in enumerate(names)}


def class_label(class_id: int, class_names: dict[int, str]) -> str:
    return class_names.get(class_id, f"cls{class_id}")


def _draw_box(ax, d: dict, class_names: dict[int, str], linestyle="solid"):
    x1, y1, x2, y2 = d["box"]
    color = class_color(d["class_id"])
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                 linewidth=2, edgecolor=color, facecolor="none", linestyle=linestyle))
    ax.text(x1, max(y1 - 4, 0), f"{class_label(d['class_id'], class_names)}:{d['score']:.2f}", color=color, fontsize=8,
            bbox=dict(facecolor="black", alpha=0.6, pad=1))


def _draw_box_dino(ax, d: dict, class_names: dict[int, str], linestyle="solid", key="combined_score"):
    """Same as _draw_box but label shows d[key] (falls back to combined_score) instead of raw SAM3 score."""
    x1, y1, x2, y2 = d["box"]
    color = class_color(d["class_id"])
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                 linewidth=2, edgecolor=color, facecolor="none", linestyle=linestyle))
    val = d.get(key, d.get("combined_score"))
    name = class_label(d["class_id"], class_names)
    label = f"{name}:{val:.2f}" if val is not None else f"{name}:?"
    ax.text(x1, max(y1 - 4, 0), label, color=color, fontsize=8,
            bbox=dict(facecolor="black", alpha=0.6, pad=1))


def save_target_figure(target_path: Path, raw_dets: list[dict], dino_kept_dets: list[dict], dino_rejected_dets: list[dict],
                        final_kept_dets: list[dict], final_rejected_dets: list[dict],
                        output_path, class_names: dict[int, str] | None = None):
    """3-panel figure: raw SAM3 boxes | kept/rejected after combined_score gate | kept/rejected
    after voting + containment (final, labelled with final_score). Box edge color = class_id."""
    class_names = class_names or {}
    with Image.open(target_path) as img:
        target_img = img.convert("RGB")
    fig = Figure(figsize=(24, 8))
    axes = fig.subplots(1, 3)

    class_ids_present = sorted({d["class_id"] for d in raw_dets})
    legend_handles = [patches.Patch(edgecolor=class_color(c), facecolor="none", linewidth=2,
                                     label=class_label(c, class_names))
                       for c in class_ids_present]

    axes[0].imshow(target_img)
    for d in raw_dets:
        _draw_box(axes[0], d, class_names)
    axes[0].set_title(f"Raw SAM3 proposals ({len(raw_dets)})", fontsize=10)
    axes[0].axis("off")
    if legend_handles:
        axes[0].legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

    axes[1].imshow(target_img)
    for d in dino_rejected_dets:
        _draw_box_dino(axes[1], d, class_names, linestyle="dashed")
    for d in dino_kept_dets:
        _draw_box_dino(axes[1], d, class_names, linestyle="solid")
    axes[1].set_title(f"After DINOv2 gate (solid={len(dino_kept_dets)} kept, dashed={len(dino_rejected_dets)} rejected)", fontsize=10)
    axes[1].axis("off")
    if legend_handles:
        axes[1].legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

    axes[2].imshow(target_img)
    for d in final_rejected_dets:
        _draw_box_dino(axes[2], d, class_names, linestyle="dashed", key="final_score")
    for d in final_kept_dets:
        _draw_box_dino(axes[2], d, class_names, linestyle="solid", key="final_score")
    axes[2].set_title(f"After voting + containment (solid={len(final_kept_dets)} kept, dashed={len(final_rejected_dets)} rejected)", fontsize=10)
    axes[2].axis("off")
    if legend_handles:
        axes[2].legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

    fig.suptitle(target_path.name, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"[saved] {output_path}")


def mask_to_patch_sel(mask: np.ndarray) -> np.ndarray:
    """(256,) bool: patches >= 50% covered by the (non-empty) mask; if none, the most-covered patch(es)."""
    cov = cv2.resize(mask.astype(np.float32), (DINO_PATCH_GRID, DINO_PATCH_GRID),
                     interpolation=cv2.INTER_AREA).ravel()
    sel = cov >= 0.5
    return sel if sel.any() else cov >= cov.max()


def prep_dino(crop: Image.Image, mask: np.ndarray | None) -> tuple[Image.Image, np.ndarray | None]:
    """Shrink to DINOv2 input before buffering: 224x224 crop (same BICUBIC square resize the processor
    does) + (256,) patch selection instead of the full mask. Keeps a batch's RAM independent of image size."""
    sel = None if mask is None else mask_to_patch_sel(mask)
    return crop.resize((DINO_INPUT, DINO_INPUT), Image.BICUBIC), sel


@torch.no_grad()
def embed_batch(crops: list[Image.Image], sels: list[np.ndarray | None], processor, model,
                device: str) -> torch.Tensor:
    """One DINOv2 forward on prep_dino() outputs, L2-normalised, returned on CPU.
    sel given -> masked-patch pooling. sel None (small class) -> mean of all tokens."""
    inputs = processor(images=crops, return_tensors="pt", do_center_crop=False,
                       size={"height": DINO_INPUT, "width": DINO_INPUT}).to(device)
    hs = model(**inputs).last_hidden_state
    patch_toks = hs[:, 1:, :]

    batch_embs = []
    for b, sel in enumerate(sels):
        if sel is None:
            emb = hs[b:b + 1].mean(dim=1)
        else:
            emb = patch_toks[b][torch.from_numpy(sel).to(patch_toks.device)].mean(dim=0, keepdim=True)
        batch_embs.append(F.normalize(emb, dim=-1))
    return torch.cat(batch_embs, dim=0).cpu()


def embed_masked_crops(crops: list[Image.Image], masks: list[np.ndarray | None], processor, model,
                        device: str, batch_size: int = 32, desc: str = "embedding") -> torch.Tensor:
    all_embs = []
    for i in tqdm(range(0, len(crops), batch_size), desc=f"  [dino] {desc}", unit="batch", leave=False):
        prepped = [prep_dino(c, m) for c, m in zip(crops[i:i + batch_size], masks[i:i + batch_size])]
        all_embs.append(embed_batch([c for c, _ in prepped], [s for _, s in prepped], processor, model, device))
        ram_guard()
    return torch.cat(all_embs, dim=0) if all_embs else torch.empty((0, model.config.hidden_size))


def build_proto_banks(proto_instances_by_class: dict, small_cls: set, dproc, dmodel, device: str,
                       dinov2_batch_size: int) -> dict[int, torch.Tensor]:
    """Embed proto bank instances. One embedding method per class: small classes mean-pool raw crops;
    other classes masked-patch only — refs SAM3 failed to mask are left out, not mixed in."""
    banks = {}
    for class_id, instances in proto_instances_by_class.items():
        is_small = class_id in small_cls
        usable = instances if is_small else [inst for inst in instances if inst.get("mask") is not None]
        if not usable:
            print(f"[warn] cls{class_id}: no usable refs for proto bank — class proposals will be rejected")
            continue
        crops = [_load_crop(inst) for inst in usable]
        masks = [None if is_small else unpack_mask(inst["mask"]) for inst in usable]
        banks[class_id] = embed_masked_crops(crops, masks, dproc, dmodel, device, dinov2_batch_size,
                                             desc=f"cls{class_id} proto bank")
        print(f"[dino] cls{class_id} proto bank: {tuple(banks[class_id].shape)}")
    return banks


def score_dets_dinov2(dets_by_target: dict[Path, list[dict]], proto_banks: dict[int, torch.Tensor],
                       small_cls: set, dproc, dmodel, device: str, dino_batch_size: int) -> None:
    """Adds 'dino_sim' (max cosine sim vs proto_banks) and 'combined_score' (0.2*sam3_score +
    0.8*dino_sim) to each det in-place. Non-small classes use the det's own SAM3 mask (masked-patch
    pooling); small classes mean-pool the raw crop. None for degenerate boxes / empty masks /
    classes without a bank -> routes to rejected. Crops streamed across targets into full batches
    (only one batch of 224px crops in RAM); targets decoded one at a time."""
    buf = []  # (det, 224px crop, patch sel)

    def flush():
        embs = embed_batch([b[1] for b in buf], [b[2] for b in buf], dproc, dmodel, device)
        for (d, _, _), emb in zip(buf, embs):
            sim = float((emb @ proto_banks[d["class_id"]].T).max())  # best-matching ref, same as Pipeline A
            d["dino_sim"] = sim
            d["combined_score"] = 0.2 * d["score"] + 0.8 * sim
        buf.clear()
        ram_guard()

    n_total = sum(len(v) for v in dets_by_target.values())
    with tqdm(total=n_total, desc="[dino] scoring proposals", unit="det") as pbar:
        for target_path, dets in dets_by_target.items():
            if not dets:
                continue
            with Image.open(target_path) as img:
                target_img = img.convert("RGB")
            for d in dets:
                pbar.update(1)
                d["dino_sim"] = d["combined_score"] = None
                packed, d["mask"] = d["mask"], None  # free RAM as dets are consumed
                if d["class_id"] not in proto_banks:
                    continue
                x1, y1, x2, y2 = (int(v) for v in d["box"])
                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate box (zero area) — can't embed
                mask = None if d["class_id"] in small_cls else unpack_mask(packed)
                if mask is not None and not mask.any():
                    continue  # SAM3 mask falls outside its own box — no foreground to pool
                buf.append((d, *prep_dino(target_img.crop((x1, y1, x2, y2)), mask)))
                if len(buf) >= dino_batch_size:
                    flush()
            del target_img
        if buf:
            flush()


def write_yolo_txt(kept_dets: list[dict], img_w: int, img_h: int, output_path):
    lines = []
    for d in kept_dets:
        x1, y1, x2, y2 = d["box"]
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{d['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--refs-dir", required=True, help="Folder of full reference images")
    p.add_argument("--refs-labels", required=True, help="Folder of YOLO .txt labels (same stem as ref images)")
    p.add_argument("--class-ids", nargs="+", default=["all"],
                    help="Class ids to pull ref instances for, e.g. '0 1 2'. Default 'all' auto-discovers every class id present in --refs-labels.")
    p.add_argument("--classes-file", default=None,
                    help="Optional YOLO classes.txt (one name per line, line N = name for class_id N). "
                         "If given, preview panels/legends show class names instead of 'clsN'.")
    p.add_argument("--targets-dir", required=True, help="Folder of target images (can include ref images too)")
    p.add_argument("--orientation", choices=["vertical", "horizontal"], default="vertical")
    p.add_argument("--split-ratio", type=float, default=SPLIT_RATIO)
    p.add_argument("--canvas-size", type=int, default=CANVAS_SIZE)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--mask-threshold", type=float, default=0.6)
    p.add_argument("--output-dir", default="output_sam3_fewshot")
    p.add_argument("--batch-size", type=int, default=8,
                    help="Targets batched per forward pass (same ref). Lower if OOM on 8GB VRAM.")
    p.add_argument("--max-refs-per-class", type=int, default=5,
                    help="SAM3 exemplar ref_groups per class — subset (most-diverse-first) of --dino-proto-size "
                         "pool. Small since SAM3 forward passes are expensive (0 = use all).")
    p.add_argument("--dino-proto-size", type=int, default=100,
                    help="Ref crops per class in the DINOv2 proto bank, farthest-point sampled (0 = use all). "
                         "Bigger than --max-refs-per-class — DINOv2 embedding is cheap, small pools are noisy.")
    p.add_argument("--dinov2-batch-size", type=int, default=0,
                    help="DINOv2 crops per forward (ref selection + scoring). 0 = auto (fills ~70%% of GPU VRAM).")
    p.add_argument("--phash-max-dist", type=int, default=4,
                    help="Perceptual-hash dedup of ref crops before DINOv2 embedding "
                         "(hamming distance <= this = duplicate, skipped). 0 = disabled. ")
    p.add_argument("--fp32", action="store_true",
                    help="Run SAM3 in fp32 (default bf16 on CUDA).")
    p.add_argument("--save-workers", type=int, default=2,
                    help="Thread pool size for offloaded figure rendering/saving.")
    p.add_argument("--ref-jpeg-quality", type=int, default=80,
                    help="JPEG quality for saved ref crops in output_dir/temp_refs/.")
    p.add_argument("--containment-thresh", type=float, default=0.85,
                    help="Box A inside box B if intersection/min_area >= this -> drop lower final_score box.")
    p.add_argument("--dup-iou-thresh", type=float, default=0.85,
                    help="Two boxes are the same object if IoU >= this -> merged, ref groups vote.")
    p.add_argument("--sam3-dino-thresh", type=float, default=0.2,
                    help="Gate on combined_score = 0.2*sam3_score + 0.8*dino_sim, before voting/containment. "
                         "Below thresh dropped. Shown in panel 2 (solid=kept, dashed=rejected).")
    p.add_argument("--small-obj-thresh", type=float, default=0.01,
                    help="p90 bbox area of selected refs below this -> class is small, skips masking, DINOv2 "
                         "mean-pools the raw crop. Same default as Pipeline A.")
    p.add_argument("--ref-box-padding", type=float, default=0.01,
                    help="Fractional ref bbox padding, clamped to image bounds.")
    p.add_argument("--no-mask-classes", nargs="+", type=int, default=[],
                    help="Class IDs to force into small-object handling (skip SAM3 mask usage, "
                         "DINOv2 mean-pools the raw crop) regardless of --small-obj-thresh. For classes "
                         "too hard/unreliable to mask (e.g. thin/reflective/occluded objects).")
    p.add_argument("--no-preview", action="store_true",
                    help="Skip the 3-panel figure and proposal mask-overlay jpgs — labels/summary.json only.")
    return p.parse_args()

def main():
    t_start = time.time()
    timing: dict[str, float] = {}
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    class_names = load_class_names(args.classes_file)
    if class_names:
        print(f"[classes-file] loaded {len(class_names)} class name(s) from {args.classes_file}")

    refs_labels_dir = Path(args.refs_labels)
    if args.class_ids == ["all"]:
        class_ids = discover_class_ids(refs_labels_dir)
        print(f"[class-ids] auto-discovered: {class_ids}")
    else:
        class_ids = [int(c) for c in args.class_ids]
    if not class_ids:
        print("[abort] No class ids found.")
        return

    raw_ref_boxes_by_class = collect_ref_boxes(Path(args.refs_dir), refs_labels_dir, class_ids, args.ref_box_padding)
    ref_boxes_by_class = {}
    for class_id in class_ids:
        boxes = raw_ref_boxes_by_class.get(class_id, [])
        print(f"[refs] {len(boxes)} ref instance(s) for class {class_id}")
        if boxes:
            ref_boxes_by_class[class_id] = boxes
    if not ref_boxes_by_class:
        print("[abort] No ref instances found for any class.")
        return

    targets_dir = Path(args.targets_dir)
    target_paths = sorted(p for p in targets_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"[targets] {len(target_paths)} target image(s)")
    if not target_paths:
        print("[abort] No target images found.")
        return

    # ── Phase 1: diverse ref selection (DINOv2) — must run before SAM3 loads (VRAM rule) ──
    t_phase = time.time()
    ref_instances_by_class, proto_instances_by_class = select_diverse_refs(
        ref_boxes_by_class, args.max_refs_per_class, args.dino_proto_size, output_dir, device,
        dinov2_batch_size=args.dinov2_batch_size, phash_max_dist=args.phash_max_dist,
        ref_jpeg_quality=args.ref_jpeg_quality,
    )
    timing["phase1_ref_select_s"] = time.time() - t_phase

    # p90 area of the SELECTED refs per class -> small classes skip masking (both ref + proposal side),
    # DINOv2 mean-pools the raw crop. Same rule as Pipeline A (yoloe_sam2_dinov2_module.py).
    small_cls = set()
    for c, insts in proto_instances_by_class.items():
        p90 = p90_area([inst["norm_area"] for inst in insts])
        if p90 < args.small_obj_thresh:
            small_cls.add(c)
    if small_cls:
        print(f"[small-obj] classes {sorted(small_cls)} below --small-obj-thresh={args.small_obj_thresh} "
              f"— SAM3 masking skipped, DINOv2 mean-pools the raw crop")
    forced = set(args.no_mask_classes) & set(class_ids)
    if forced:
        print(f"[no-mask-classes] forcing small-object handling for cls: {sorted(forced)}")
        small_cls |= forced

    # ── Phase 2: SAM3 (ref masks + canvas-composite proposals) ──
    t_phase = time.time()
    dtype = torch.float32 if (args.fp32 or device != "cuda") else torch.bfloat16
    print(f"[model] Loading SAM3: {SAM3_MODEL_ID} ({dtype}) ...")
    try:
        model = Sam3Model.from_pretrained(SAM3_MODEL_ID, torch_dtype=dtype, device_map=device)
        processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
    except OSError:
        model = Sam3Model.from_pretrained(SAM3_MODEL_ID, torch_dtype=dtype, device_map=device, local_files_only=True)
        processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, local_files_only=True)

    errors_log = output_dir / "errors.log"
    print("\n[refs] Masking ref crops with SAM3 (box-prompt) ...")
    n_skipped = mask_ref_crops_sam3(proto_instances_by_class, small_cls, model, processor, device, output_dir,
                                    threshold=args.threshold, mask_threshold=args.mask_threshold,
                                    log_path=errors_log)

    ref_pairs = [(class_id, gi, rg) for class_id, groups in ref_instances_by_class.items()
                 for gi, rg in enumerate(groups)]
    n_groups_by_class = {c: len(groups) for c, groups in ref_instances_by_class.items()}
    total = len(ref_pairs) * len(target_paths)
    done = 0
    batch_size = args.batch_size

    save_pool = ThreadPoolExecutor(max_workers=args.save_workers)
    save_futures = []
    max_pending_saves = args.save_workers * 4

    def drain_saves(force: bool = False):
        if force or len(save_futures) >= max_pending_saves:
            for f in save_futures:
                f.result()
            save_futures.clear()

    # Per-target accumulation across ALL classes/ref_groups: target_path -> list[det dict]
    dets_by_target: dict[Path, list[dict]] = {p: [] for p in target_paths}

    # Targets outer, (class, ref_group) inner — each target batch decoded+resized ONCE
    # and reused across every ref_group/class, instead of once per (class, ref_group) pair.
    for batch_start in range(0, len(target_paths), batch_size):
        batch_paths = target_paths[batch_start:batch_start + batch_size]
        batch_target_imgs = {p: Image.open(p).convert("RGB") for p in batch_paths}

        for class_id, group_idx, ref_group in ref_pairs:
            print(f"\n[class={class_id} ref_group={ref_group['names']}] "
                  f"batch {batch_start // batch_size + 1} "
                  f"({len(batch_paths)} target(s): {', '.join(p.name for p in batch_paths)})")

            canvases, placements_list = [], []
            for target_path in batch_paths:
                canvas, placements = create_canvas(
                    ref_group["image"], ref_group["boxes"], batch_target_imgs[target_path],
                    args.canvas_size, args.orientation, args.split_ratio,
                )
                canvases.append(canvas)
                placements_list.append(placements)

            inputs = processor(
                images=canvases,
                input_boxes=[[norm_cxcywh_to_xyxy_px(nb, *pl["canvas_size"]) for nb in get_norm_boxes(pl)] for pl in placements_list],
                input_boxes_labels=[[1] * len(ref_group["boxes"]) for _ in canvases],
                return_tensors="pt",
            ).to(model.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
            if "input_boxes" in inputs:
                inputs["input_boxes"] = inputs["input_boxes"].to(model.dtype)

            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            except Exception as e:
                del inputs
                warn_skip(f"SAM3 class={class_id} ref_group={group_idx} "
                          f"batch {batch_start // batch_size + 1} ({len(batch_paths)} target(s))", e, errors_log)
                n_skipped += 1
                done += len(batch_paths)
                continue

            results_list = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )

            keep_mask = class_id not in small_cls
            for target_path, placements, results in zip(batch_paths, placements_list, results_list):
                done += 1
                print(f"  [{done}/{total}] target={target_path.name}")

                # .to(uint8) before .numpy() — numpy has no bf16 dtype
                masks_canvas = [m.to(torch.uint8).cpu().numpy() for m in results["masks"]]
                scores_canvas = [float(s) for s in results["scores"]]
                boxes_canvas = [b.float().cpu().numpy().tolist() for b in results["boxes"]]

                dets = crop_result_to_target(placements, masks_canvas, boxes_canvas, scores_canvas,
                                             class_id, group_idx, keep_mask)
                if dets:
                    print(f"  [scores] {len(dets)} instance(s): " + ", ".join(f"{d['score']:.3f}" for d in dets))
                else:
                    print("  [scores] no instances passed into target region")

                if keep_mask and dets and not args.no_preview:
                    target_np = np.array(batch_target_imgs[target_path])
                    prop_dir = output_dir / "temp_refs" / f"cls{class_id}" / "proposals"
                    prop_dir.mkdir(parents=True, exist_ok=True)
                    for pi, d in enumerate(dets):
                        save_futures.append(save_pool.submit(
                            save_mask_overlay, target_np, unpack_mask(d["mask"]),
                            tuple(int(v) for v in d["box"]),
                            prop_dir / f"{target_path.stem}_g{group_idx}_inst{pi}.jpg"))
                    drain_saves()

                dets_by_target[target_path].extend(dets)

            del inputs, outputs
            ram_guard()
        del batch_target_imgs

    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()
    timing["phase2_sam3_s"] = time.time() - t_phase

    # ── Phase 3: DINOv2 proto bank + proposal scoring ──
    t_phase = time.time()
    from transformers import AutoImageProcessor, AutoModel
    print(f"\n[dino] Loading DINOv2: {DINOV2_MODEL_ID} for proto bank + proposal scoring ...")
    try:
        dproc = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
        dmodel = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(device).eval()
    except OSError:
        dproc = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID, local_files_only=True)
        dmodel = AutoModel.from_pretrained(DINOV2_MODEL_ID, local_files_only=True).to(device).eval()
    dino_bs = args.dinov2_batch_size if args.dinov2_batch_size > 0 else auto_dino_batch_size(dmodel, device)

    dino_proto_banks = build_proto_banks(proto_instances_by_class, small_cls, dproc, dmodel, device, dino_bs)

    # Gate on combined_score before voting/containment — runs first so a garbage oversized
    # SAM3 box gets killed on appearance before it can swallow real boxes via containment
    score_dets_dinov2(dets_by_target, dino_proto_banks, small_cls, dproc, dmodel, device, dino_bs)

    del dmodel, dproc
    if device == "cuda":
        torch.cuda.empty_cache()
    timing["phase3_dinov2_s"] = time.time() - t_phase

    # ── Phase 4: gate → voting + containment (per class) → save ──
    t_phase = time.time()
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for target_path in target_paths:
        raw_dets = dets_by_target[target_path]
        dino_kept, dino_rejected = [], []
        for d in raw_dets:
            combined = d.get("combined_score")
            (dino_kept if combined is not None and combined >= args.sam3_dino_thresh else dino_rejected).append(d)

        kept_all, rejected_all = [], []
        for class_id in sorted({d["class_id"] for d in dino_kept}):
            class_dets = [d for d in dino_kept if d["class_id"] == class_id]
            kept, rejected = vote_and_filter(class_dets, n_groups_by_class[class_id],
                                             args.containment_thresh, args.dup_iou_thresh)
            kept_all.extend(kept)
            rejected_all.extend(rejected)

        with Image.open(target_path) as img:
            img_w, img_h = img.size
        label_path = labels_dir / f"{target_path.stem}.txt"
        write_yolo_txt(kept_all, img_w, img_h, label_path)

        out_path = output_dir / f"{target_path.stem}.png"
        if not args.no_preview:
            save_futures.append(save_pool.submit(
                save_target_figure, target_path, raw_dets, dino_kept, dino_rejected,
                kept_all, rejected_all, out_path, class_names,
            ))
            drain_saves()

        summary[target_path.name] = {
            "label_file": str(label_path.resolve()),
            "preview_file": None if args.no_preview else str(out_path.resolve()),
            "n_final_total": len(kept_all),
            "boxes": [{"class_id": d["class_id"], "box": d["box"], "sam3_score": d["score"],
                       "dino_sim": d.get("dino_sim"), "combined_score": d.get("combined_score"),
                       "votes": d.get("votes"), "final_score": d.get("final_score")} for d in kept_all],
        }
        ram_guard()

    print("\n[save] waiting for remaining figure saves to finish ...")
    drain_saves(force=True)
    save_pool.shutdown()

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[saved] {output_dir / 'summary.json'}")
    timing["phase4_vote_save_s"] = time.time() - t_phase

    timing["total_s"] = time.time() - t_start
    timing["n_targets"] = len(target_paths)
    timing["skipped_batches"] = n_skipped
    (output_dir / "timing.json").write_text(json.dumps(
        {k: round(v, 1) if isinstance(v, float) else v for k, v in timing.items()}, indent=2))
    print(f"\n[done] Output -> {output_dir.resolve()}")
    print("[timing] per phase (not cumulative): " +
          "  ".join(f"{k}={v:.1f}" for k, v in timing.items() if k.endswith("_s")))
    if n_skipped:
        print(f"[WARN] {n_skipped} SAM3 batch(es) skipped — results incomplete, see {errors_log}")


if __name__ == "__main__":
    main()
