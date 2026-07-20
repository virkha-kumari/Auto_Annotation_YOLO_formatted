"""
SAM3 cross-image few-shot via canvas-composite exemplar prompting.

No native cross-image few-shot API in SAM3 (exemplar boxes only work within
same image). Trick (WongKinYiu/FSS-SAM3): paste ref + target on one canvas,
remap ref bbox to canvas coords, run SAM3 once (box-only exemplar, no text),
crop target half back out.

Refs sharing a source image are grouped onto one canvas as a multi-box prompt
(one SAM3 call per frame, not per instance).

Per (class, ref group, target) triple, saves 4-panel figure: composite canvas
w/ ref box(es) -> raw SAM3 pred on canvas -> pred cropped to target (mask+bbox+score)
-> tightened bbox only.

--class-ids: explicit ("0 1 2") or "all" (auto-discover from --refs-labels).
Output: output_dir/<target_stem>__cls<id>_<ref>.png

Targets batched per ref group (--batch-size) into one SAM3 forward pass;
lower it on 8GB VRAM OOM.

Speed: bf16 on CUDA by default (--fp32 disables); figure rendering offloaded
to thread pool. Ref selection: DINOv2 CLS-embed + farthest-point sample
--dino-proto-size diverse refs per class (0 = all) for the DINOv2 proto bank;
--max-refs-per-class is a subset of that same ordered list (most-diverse-first)
used as SAM3's exemplar ref_groups, saved to output_dir/temp_refs/ for
inspection (DINOv2 loaded/unloaded before SAM3, VRAM rule).

DINOv2 scoring: masked-patch pooling (not raw-crop CLS, too noisy on clutter).
SAM3's own masks reused for target proposals; ref crops get box-prompted SAM3
too (upfront, same VRAM phase, no extra model). Small classes (p90 bbox area
< --small-obj-thresh) skip masking, CLS-on-raw-crop instead.

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


def collect_ref_boxes(refs_dir: Path, refs_labels: Path, class_ids: list[int]) -> dict[int, list[dict]]:
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
            box_xywh = [(cx - w / 2) * iw, (cy - h / 2) * ih, w * iw, h * ih]  # x,y,w,h px
            boxes_by_class[cid].append({
                "img_path": img_path,
                "box": box_xywh,
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


def _load_crop(inst: dict) -> Image.Image:
    """Decode source image on demand, crop box region."""
    with Image.open(inst["img_path"]) as img:
        img = img.convert("RGB")
        x, y, w, h = inst["box"]
        return img.crop((int(x), int(y), int(x + w), int(y + h)))


def p90_bbox_area(refs_labels_dir: Path, cls_id: int) -> float:
    """90th-percentile bbox area (w*h normalised) for cls_id across all ref label files.
    If p90 < --small-obj-thresh the class is small -- even its largest typical instances
    are tiny, so masking adds no signal (cf. yoloe_sam2_dinov2_module.py)."""
    areas = []
    for f in refs_labels_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and int(parts[0]) == cls_id:
                areas.append(float(parts[3]) * float(parts[4]))
    if not areas:
        return 1.0
    areas.sort()
    idx = int(len(areas) * 0.9)
    return areas[min(idx, len(areas) - 1)]


def make_masked_crop(img_np: np.ndarray, mask_hw: np.ndarray, bbox_xyxy: list) -> Image.Image:
    """Crop to bbox, zero out pixels outside mask (mask must be same-size as img_np). DINOv2 input."""
    ih, iw = img_np.shape[:2]
    x1, y1 = max(0, int(bbox_xyxy[0])), max(0, int(bbox_xyxy[1]))
    x2, y2 = min(iw, int(bbox_xyxy[2])), min(ih, int(bbox_xyxy[3]))
    crop = img_np[y1:y2, x1:x2].copy()
    mask_crop = mask_hw[y1:y2, x1:x2]
    crop[mask_crop == 0] = 0
    return Image.fromarray(crop)


def save_mask_overlay(img_np: np.ndarray, mask_hw: np.ndarray, bbox_xyxy: list, output_path, pad: float = 0.3):
    """Save image crop (padded window around bbox) with mask as translucent overlay + box outline.
    Same style as debug_sam3.py panels — for inspecting mask alignment, not a model input."""
    ih, iw = img_np.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    wx1 = max(0, int(x1 - bw * pad))
    wy1 = max(0, int(y1 - bh * pad))
    wx2 = min(iw, int(x2 + bw * pad))
    wy2 = min(ih, int(y2 + bh * pad))

    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    ax.imshow(img_np[wy1:wy2, wx1:wx2])
    ax.imshow(mask_hw[wy1:wy2, wx1:wx2], alpha=0.5, cmap="jet")
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
    mask_ref_crops_sam3() + score_dets_dinov2()). SAM3's max_refs exemplars are the first N of that
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
    print("[refs] Loading DINOv2 for diverse ref selection + proto-bank embedding ...")
    try:
        dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        dmodel = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    except OSError:
        dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
        dmodel = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).to(device).eval()

    sam3_groups_by_class = {}
    proto_instances_by_class = {}
    with torch.no_grad():
        for class_id, instances in ref_boxes_by_class.items():
            embs_by_idx: dict[int, np.ndarray] = {}

            def embed_all(insts_with_idx: list[tuple[int, dict]]) -> None:
                for i in tqdm(range(0, len(insts_with_idx), dinov2_batch_size),
                               desc=f"  cls{class_id} dinov2", unit="batch", leave=False):
                    batch = insts_with_idx[i:i + dinov2_batch_size]
                    crops = [_load_crop(inst) for _, inst in batch]
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
            # Masked versions (non-small classes) get saved later by mask_ref_crops_sam3(), overwriting these.
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
                    x, y, w, h = inst["box"]
                    crop = full_img.crop((int(x), int(y), int(x + w), int(y + h)))
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


def crop_result_to_target(placements, masks_canvas, boxes_canvas, scores_canvas, class_id):
    """Crop one batch's canvas-space masks/boxes/scores back to target-image space. Returns list
    of dicts: {box: [x1,y1,x2,y2] px (SAM3's own box, not mask-derived), mask: np.ndarray (target-space,
    bool), score: float, class_id: int}."""
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
        mask_target = np.array(
            Image.fromarray(crop * 255).resize(tgt["orig_size"], Image.NEAREST)
        ) > 0
        out.append({"box": [float(v) for v in box], "mask": mask_target,
                     "score": float(score), "class_id": class_id})
    return out


def mask_ref_crops_sam3(proto_instances_by_class: dict, small_cls: set, model, processor, device: str,
                         output_dir: Path, threshold: float = 0.0, mask_threshold: float = 0.6,
                         ref_jpeg_quality: int = 80, box_chunk: int = 8, min_match_containment: float = 0.25) -> dict:
    """Box-prompt SAM3 per ref instance (own image, own box). Adds 'mask' in place; small
    classes get None. SAM3 output has no positional link to input box order (fixed-size DETR
    decoder) -- each prompt box matched to best-containment output box, not zipped by index."""
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
                    print(f"  [refs] cls{class_id} {img_path.name}: {e} — skipping chunk, raw crop kept")
                    for inst in chunk:
                        inst["mask"] = None
                    n_failed += len(chunk)
                    del inputs
                    if device == "cuda":
                        torch.cuda.empty_cache()
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
                    if best_j < 0 or best_containment < min_match_containment or out_masks[best_j].sum() == 0:
                        inst["mask"] = None
                        n_failed += 1
                        continue
                    mask = out_masks[best_j]
                    inst["mask"] = mask
                    n_masked += 1
                    save_mask_overlay(img_np, mask, prompt_box, temp_refs_dir / f"{inst['name']}_mask.jpg")

                del inputs, outputs
                ram_guard()

        print(f"[refs] cls{class_id}: {n_masked} ref(s) masked, {n_failed} kept as raw crop (SAM3 miss)")
    return proto_instances_by_class


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


def filter_containment_duplicates(dets: list[dict], containment_thresh: float, iou_thresh: float) -> tuple[list[dict], list[dict]]:
    """Per-class-id containment + duplicate filter. dets already restricted to one class.
    Sort by score desc; a lower-score box dies if it's contained in (containment>=thresh) or
    a near-duplicate of (iou>=thresh) an already-kept higher-score box. Returns (kept, rejected)."""
    order = sorted(range(len(dets)), key=lambda i: dets[i]["score"], reverse=True)
    kept_idx = []
    rejected_idx = []
    for i in order:
        box_i = dets[i]["box"]
        suppressed = False
        for j in kept_idx:
            iou, containment = box_iou(box_i, dets[j]["box"])
            if containment >= containment_thresh or iou >= iou_thresh:
                suppressed = True
                break
        if suppressed:
            rejected_idx.append(i)
        else:
            kept_idx.append(i)
    kept = [dets[i] for i in kept_idx]
    rejected = [dets[i] for i in rejected_idx]
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


def _draw_box_dino(ax, d: dict, class_names: dict[int, str], linestyle="solid"):
    """Same as _draw_box but label shows combined_score (the gating value) instead of raw SAM3 score."""
    x1, y1, x2, y2 = d["box"]
    color = class_color(d["class_id"])
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                 linewidth=2, edgecolor=color, facecolor="none", linestyle=linestyle))
    combined = d.get("combined_score")
    name = class_label(d["class_id"], class_names)
    label = f"{name}:{combined:.2f}" if combined is not None else f"{name}:?"
    ax.text(x1, max(y1 - 4, 0), label, color=color, fontsize=8,
            bbox=dict(facecolor="black", alpha=0.6, pad=1))


def save_target_figure(target_img, raw_dets: list[dict], dino_kept_dets: list[dict], dino_rejected_dets: list[dict],
                        final_kept_dets: list[dict], final_rejected_dets: list[dict],
                        target_name: str, output_path, class_names: dict[int, str] | None = None):
    """3-panel figure: raw SAM3 boxes | kept/rejected after combined_score gate | kept/rejected
    after containment+dup filter (final). Box edge color = class_id, same across panels."""
    class_names = class_names or {}
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
        _draw_box_dino(axes[2], d, class_names, linestyle="dashed")
    for d in final_kept_dets:
        _draw_box_dino(axes[2], d, class_names, linestyle="solid")
    axes[2].set_title(f"After containment + dup filter (solid={len(final_kept_dets)} kept, dashed={len(final_rejected_dets)} rejected)", fontsize=10)
    axes[2].axis("off")
    if legend_handles:
        axes[2].legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

    fig.suptitle(target_name, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"[saved] {output_path}")


@torch.no_grad()
def embed_masked_crops(crops: list[Image.Image], masks: list[np.ndarray | None], processor, model,
                        device: str, batch_size: int = 32, desc: str = "embedding") -> torch.Tensor:
    """DINOv2 embedding, L2-normalised. mask given (crop-local coords) -> masked-patch pooling.
    mask None (small class / SAM3 miss) -> mean of all tokens on raw crop."""
    all_embs = []
    for i in tqdm(range(0, len(crops), batch_size), desc=f"  [dino] {desc}", unit="batch", leave=False):
        batch = crops[i:i + batch_size]
        batch_masks = masks[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        out = model(**inputs)
        hs = out.last_hidden_state
        cls_tok = hs[:, 0, :]
        patch_toks = hs[:, 1:, :]

        batch_embs = []
        for b in range(len(batch)):
            mask = batch_masks[b]
            if mask is None:
                mean_tok = hs[b:b + 1].mean(dim=1)
                batch_embs.append(F.normalize(mean_tok, dim=-1))
                continue
            if mask.max() == 0:
                batch_embs.append(F.normalize(cls_tok[b:b + 1], dim=-1))
                continue
            mask_small = cv2.resize(
                mask.astype(np.uint8), (DINO_PATCH_GRID, DINO_PATCH_GRID),
                interpolation=cv2.INTER_NEAREST,
            ).ravel()
            patch_sel = patch_toks[b][mask_small.astype(bool)]
            emb = F.normalize(
                cls_tok[b:b + 1] if len(patch_sel) == 0 else patch_sel.mean(dim=0, keepdim=True),
                dim=-1,
            )
            batch_embs.append(emb)

        all_embs.append(torch.cat(batch_embs, dim=0).cpu())
        del batch, inputs, out
        ram_guard()
    return torch.cat(all_embs, dim=0) if all_embs else torch.empty((0, model.config.hidden_size))


def build_proto_banks(proto_instances_by_class: dict, small_cls: set, dproc, dmodel, device: str,
                       dinov2_batch_size: int) -> dict[int, torch.Tensor]:
    """Embed proto bank instances via embed_masked_crops, using masks from mask_ref_crops_sam3()."""
    banks = {}
    for class_id, instances in proto_instances_by_class.items():
        crops, masks = [], []
        for inst in instances:
            crop = _load_crop(inst)
            crops.append(crop)
            mask = inst.get("mask")
            if mask is None or class_id in small_cls:
                masks.append(None)
            else:
                x, y, w, h = inst["box"]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                masks.append(mask[y1:y2, x1:x2])
        embs = embed_masked_crops(crops, masks, dproc, dmodel, device, dinov2_batch_size,
                                   desc=f"cls{class_id} proto bank")
        banks[class_id] = embs
    return banks


def score_dets_dinov2(dets_by_target: dict[Path, list[dict]], proto_banks: dict[int, torch.Tensor],
                       target_imgs: dict[Path, Image.Image], small_cls: set, dproc, dmodel, device: str,
                       dino_batch_size: int, desc_prefix: str = "") -> None:
    """Adds 'dino_sim' (max cosine sim vs proto_banks) and 'combined_score' (0.2*sam3_score +
    0.8*dino_sim) to each det in-place. Non-small classes use the det's own SAM3 mask (masked-patch
    pooling); small classes / masks that came back empty use CLS-on-raw-crop. None propagates for
    degenerate boxes -> routes to rejected."""
    with torch.no_grad():
        for target_path, dets in dets_by_target.items():
            if not dets:
                continue
            target_img = target_imgs[target_path]
            for class_id in sorted({d["class_id"] for d in dets}):
                if class_id not in proto_banks:
                    continue
                bank = proto_banks[class_id]
                class_dets = [d for d in dets if d["class_id"] == class_id]
                valid_dets, crops, masks = [], [], []
                for d in class_dets:
                    x1, y1, x2, y2 = (int(v) for v in d["box"])
                    if x2 <= x1 or y2 <= y1:
                        d["dino_sim"] = None  # degenerate box (zero area) — can't embed, skip
                        d["combined_score"] = None
                        continue
                    valid_dets.append(d)
                    crops.append(target_img.crop((x1, y1, x2, y2)))
                    det_mask = d.get("mask")
                    masks.append(None if (det_mask is None or class_id in small_cls)
                                  else det_mask[y1:y2, x1:x2])
                if not crops:
                    continue
                prop_embs = embed_masked_crops(crops, masks, dproc, dmodel, device, dino_batch_size,
                                            desc=f"{desc_prefix}{target_path.name} cls{class_id}")
                sims, _ = (prop_embs @ bank.T).max(dim=1)  # best-matching ref, same as Pipeline A
                for d, sim in zip(valid_dets, sims.tolist()):
                    d["dino_sim"] = float(sim)
                    d["combined_score"] = 0.2 * d["score"] + 0.8 * float(sim)


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
    p.add_argument("--dinov2-batch-size", type=int, default=32,
                    help="Crops per forward pass during diverse-ref DINOv2 embedding.")
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
                    help="Box A inside box B if intersection/min_area >= this -> drop lower-score box.")
    p.add_argument("--dup-iou-thresh", type=float, default=0.85,
                    help="Two boxes are duplicates if IoU >= this -> drop lower-score box.")
    p.add_argument("--sam3-dino-thresh", type=float, default=0.2,
                    help="Gate on combined_score = 0.2*sam3_score + 0.8*dino_sim, before containment/dup "
                         "filter. Below thresh dropped. Shown in panel 2 (solid=kept, dashed=rejected).")
    p.add_argument("--small-obj-thresh", type=float, default=0.01,
                    help="p90 ref bbox area below this -> class is small, skips masking, DINOv2 "
                         "uses CLS-on-raw-crop. Same default as Pipeline A.")
    return p.parse_args()

def main():
    t_start = time.time()
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

    raw_ref_boxes_by_class = collect_ref_boxes(Path(args.refs_dir), refs_labels_dir, class_ids)
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

    # p90 ref bbox area per class -> small classes skip masking (both ref + proposal side),
    # DINOv2 falls back to CLS-on-raw-crop. Same rule as Pipeline A (yoloe_sam2_dinov2_module.py).
    small_cls = {c for c in class_ids if p90_bbox_area(refs_labels_dir, c) < args.small_obj_thresh}
    if small_cls:
        print(f"[small-obj] classes {sorted(small_cls)} below --small-obj-thresh={args.small_obj_thresh} "
              f"— SAM3 masking skipped, DINOv2 uses CLS-on-raw-crop")

    # Diverse ref selection (DINOv2) must run before SAM3 loads — VRAM rule.
    ref_instances_by_class, proto_instances_by_class = select_diverse_refs(
        ref_boxes_by_class, args.max_refs_per_class, args.dino_proto_size, output_dir, device,
        dinov2_batch_size=args.dinov2_batch_size, phash_max_dist=args.phash_max_dist,
        ref_jpeg_quality=args.ref_jpeg_quality,
    )

    dtype = torch.float32 if (args.fp32 or device != "cuda") else torch.bfloat16
    print(f"[model] Loading SAM3: {SAM3_MODEL_ID} ({dtype}) ...")
    try:
        model = Sam3Model.from_pretrained(SAM3_MODEL_ID, torch_dtype=dtype, device_map=device)
        processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
    except OSError:
        model = Sam3Model.from_pretrained(SAM3_MODEL_ID, torch_dtype=dtype, device_map=device, local_files_only=True)
        processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, local_files_only=True)

    print("\n[refs] Masking ref crops with SAM3 (box-prompt) ...")
    mask_ref_crops_sam3(proto_instances_by_class, small_cls, model, processor, device, output_dir,
                         threshold=args.threshold, mask_threshold=args.mask_threshold,
                         ref_jpeg_quality=args.ref_jpeg_quality)

    ref_pairs = [(class_id, rg) for class_id, groups in ref_instances_by_class.items() for rg in groups]
    total = len(ref_pairs) * len(target_paths)
    done = 0
    batch_size = args.batch_size

    # Per-target accumulation across ALL classes/ref_groups: target_path -> list[det dict]
    dets_by_target: dict[Path, list[dict]] = {p: [] for p in target_paths}

    # Targets outer, (class, ref_group) inner — each target batch decoded+resized ONCE
    # and reused across every ref_group/class, instead of once per (class, ref_group) pair.
    for batch_start in range(0, len(target_paths), batch_size):
        batch_paths = target_paths[batch_start:batch_start + batch_size]
        batch_target_imgs = {p: Image.open(p).convert("RGB") for p in batch_paths}

        for class_id, ref_group in ref_pairs:
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
                print(f"  [class={class_id} ref_group={ref_group['names']}] "
                      f"batch {batch_start // batch_size + 1}: {e} — skipping batch")
                del inputs
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

            results_list = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )

            for target_path, placements, results in zip(batch_paths, placements_list, results_list):
                done += 1
                print(f"  [{done}/{total}] target={target_path.name}")

                # .to(uint8) before .numpy() — numpy has no bf16 dtype
                masks_canvas = [m.to(torch.uint8).cpu().numpy() for m in results["masks"]]
                scores_canvas = [float(s) for s in results["scores"]]
                boxes_canvas = [b.float().cpu().numpy().tolist() for b in results["boxes"]]

                dets = crop_result_to_target(placements, masks_canvas, boxes_canvas, scores_canvas, class_id)
                if dets:
                    print(f"  [scores] {len(dets)} instance(s): " + ", ".join(f"{d['score']:.3f}" for d in dets))
                else:
                    print("  [scores] no instances passed into target region")

                if class_id not in small_cls and dets:
                    target_np = np.array(batch_target_imgs[target_path])
                    prop_dir = output_dir / "temp_refs" / f"cls{class_id}" / "proposals"
                    prop_dir.mkdir(parents=True, exist_ok=True)
                    for pi, d in enumerate(dets):
                        save_mask_overlay(target_np, d["mask"], d["box"], prop_dir / f"{target_path.stem}_inst{pi}.jpg")

                dets_by_target[target_path].extend(dets)

            del inputs, outputs
            ram_guard()

    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()

    from transformers import AutoImageProcessor, AutoModel
    print(f"\n[dino] Loading DINOv2: {DINOV2_MODEL_ID} for proto bank + proposal scoring ...")
    try:
        dproc = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
        dmodel = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(device).eval()
    except OSError:
        dproc = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID, local_files_only=True)
        dmodel = AutoModel.from_pretrained(DINOV2_MODEL_ID, local_files_only=True).to(device).eval()

    dino_proto_banks = build_proto_banks(proto_instances_by_class, small_cls, dproc, dmodel, device,
                                          args.dinov2_batch_size)

    # Gate on combined_score before containment/dup filter — runs first so a garbage oversized
    # SAM3 box gets killed on appearance before it can swallow real boxes via containment
    target_imgs = {p: Image.open(p).convert("RGB") for p in target_paths}
    score_dets_dinov2(dets_by_target, dino_proto_banks, target_imgs, small_cls, dproc, dmodel, device,
                       args.dinov2_batch_size)

    del dmodel, dproc
    if device == "cuda":
        torch.cuda.empty_cache()

    dino_kept_by_target: dict[Path, list[dict]] = {}
    dino_rejected_by_target: dict[Path, list[dict]] = {}
    for target_path in target_paths:
        raw_dets = dets_by_target[target_path]
        dino_kept, dino_rejected = [], []
        for d in raw_dets:
            combined = d.get("combined_score")
            (dino_kept if combined is not None and combined >= args.sam3_dino_thresh else dino_rejected).append(d)
        dino_kept_by_target[target_path] = dino_kept
        dino_rejected_by_target[target_path] = dino_rejected

    # Per-target: containment + duplicate filter (per class_id), on DINOv2-gate survivors only.
    kept_by_target: dict[Path, list[dict]] = {}
    rejected_by_target: dict[Path, list[dict]] = {}
    for target_path in target_paths:
        dino_kept = dino_kept_by_target[target_path]
        kept_all, rejected_all = [], []
        for class_id in sorted({d["class_id"] for d in dino_kept}):
            class_dets = [d for d in dino_kept if d["class_id"] == class_id]
            kept, rejected = filter_containment_duplicates(class_dets, args.containment_thresh, args.dup_iou_thresh)
            kept_all.extend(kept)
            rejected_all.extend(rejected)
        kept_by_target[target_path] = kept_all
        rejected_by_target[target_path] = rejected_all

    # Write YOLO txt + save 3-panel fig per target.
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    save_pool = ThreadPoolExecutor(max_workers=args.save_workers)
    save_futures = []
    max_pending_saves = args.save_workers * 4
    summary = {}
    for target_path in target_paths:
        raw_dets = dets_by_target[target_path]
        dino_kept = dino_kept_by_target[target_path]
        dino_rejected = dino_rejected_by_target[target_path]
        kept_all = kept_by_target[target_path]
        rejected_all = rejected_by_target[target_path]

        with Image.open(target_path) as img:
            img_w, img_h = img.size
        label_path = labels_dir / f"{target_path.stem}.txt"
        write_yolo_txt(kept_all, img_w, img_h, label_path)

        target_img = target_imgs[target_path]
        out_path = output_dir / f"{target_path.stem}.png"
        save_futures.append(save_pool.submit(
            save_target_figure, target_img, raw_dets, dino_kept, dino_rejected,
            kept_all, rejected_all, target_path.name, out_path, class_names,
        ))

        summary[target_path.name] = {
            "label_file": str(label_path.resolve()),
            "preview_file": str(out_path.resolve()),
            "n_final_total": len(kept_all),
            "boxes": [{"class_id": d["class_id"], "box": d["box"], "sam3_score": d["score"],
                       "dino_sim": d.get("dino_sim"), "combined_score": d.get("combined_score")} for d in kept_all],
        }

        if len(save_futures) >= max_pending_saves:
            for f in save_futures:
                f.result()
            save_futures.clear()
        ram_guard()

    print("\n[save] waiting for remaining figure saves to finish ...")
    for f in save_futures:
        f.result()
    save_pool.shutdown()

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[saved] {output_dir / 'summary.json'}")

    elapsed = time.time() - t_start
    print(f"\n[done] Output -> {output_dir.resolve()}")
    print(f"[time] total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
