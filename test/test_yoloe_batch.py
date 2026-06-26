"""
Test: can we call get_vpe + set_classes once per stem, then batch all targets?
Usage:
    python test/test_yoloe_batch.py \
        --queries "D:/path/to/crops/cls0" \
        --targets-dir "D:/path/to/targets" \
        --source-images "D:/path/to/source" \
        --labels "D:/path/to/labels" \
        --batch-size 8
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_STEM_RE = re.compile(r"^(.+)_cls(\d+)_(\d+)$")


def resolve_source_image(crop_path, source_dir):
    m = CROP_STEM_RE.match(crop_path.stem)
    if not m:
        return None
    src_stem = m.group(1)
    for ext in IMG_EXTS:
        c = source_dir / (src_stem + ext)
        if c.exists():
            return c
    return None


def resolve_bboxes(crop_path, labels_dir, src_img, padding=0.05):
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
        x1 = max(0,      int((cx - w/2 - pad_x) * iw))
        y1 = max(0,      int((cy - h/2 - pad_y) * ih))
        x2 = min(iw - 1, int((cx + w/2 + pad_x) * iw))
        y2 = min(ih - 1, int((cy + h/2 + pad_y) * ih))
        bboxes.append([x1, y1, x2, y2])
    return bboxes if bboxes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries",      required=True)
    ap.add_argument("--targets-dir",  required=True)
    ap.add_argument("--source-images",required=True)
    ap.add_argument("--labels",       required=True)
    ap.add_argument("--batch-size",   type=int, default=8)
    ap.add_argument("--conf",         type=float, default=0.06)
    ap.add_argument("--iou",          type=float, default=0.45)
    ap.add_argument("--max-stems",    type=int, default=999)
    args = ap.parse_args()

    queries_dir   = Path(args.queries)
    targets_dir   = Path(args.targets_dir)
    source_dir    = Path(args.source_images)
    labels_dir    = Path(args.labels)
    B             = args.batch_size

    crop_paths  = sorted(p for p in queries_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    target_paths = sorted(p for p in targets_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"[info] {len(crop_paths)} crops, {len(target_paths)} targets")

    # Build stem → (src_path, bboxes, cls_list)
    stem_data: dict[str, tuple[Path, list, list]] = {}
    for cp in crop_paths:
        src = resolve_source_image(cp, source_dir)
        if src is None:
            continue
        stem = src.stem
        if stem in stem_data:
            continue
        src_img = Image.open(src).convert("RGB")
        bboxes  = resolve_bboxes(cp, labels_dir, src_img)
        if not bboxes:
            continue
        cls_list = list(range(len(bboxes)))
        stem_data[stem] = (src, bboxes, cls_list)

    stems = list(stem_data.keys())[:args.max_stems]
    print(f"[info] {len(stems)} stems with bbox prompts")

    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    ultralytics.utils.LOGGER.setLevel("WARNING")

    print("[info] Loading YOLOe...")
    model = YOLO("yoloe-11l-seg.pt")

    # ── METHOD A: current approach (1 target × 1 stem per call) ──────────────
    print(f"\n[A] Current: 1-target × 1-stem calls")
    t0 = time.time()
    total_proposals_A = 0
    # Only test first 5 targets for speed
    test_targets = target_paths[:5]
    for target_path in tqdm(test_targets, desc="[A] targets"):
        for stem in stems:
            src_path, bboxes, cls_list = stem_data[stem]
            visual_prompts = dict(bboxes=bboxes, cls=cls_list)
            try:
                results = model.predict(
                    source=str(target_path),
                    refer_image=str(src_path),
                    visual_prompts=visual_prompts,
                    predictor=YOLOEVPSegPredictor,
                    conf=args.conf, iou=args.iou, verbose=False,
                )
                if results and results[0].boxes is not None:
                    total_proposals_A += len(results[0].boxes)
            except Exception as e:
                print(f"  [A] error: {e}")
    t_A = time.time() - t0
    print(f"[A] {t_A:.1f}s for {len(test_targets)} targets × {len(stems)} stems | {total_proposals_A} proposals")
    print(f"[A] per-target: {t_A/len(test_targets):.2f}s")

    import torch
    torch.cuda.empty_cache()
    time.sleep(2)

    # ── METHOD B: get_vpe once per stem → batch targets ──────────────────────
    print(f"\n[B] Batched: get_vpe once per stem → predict(source=[batch]) ")
    t0 = time.time()
    total_proposals_B = 0

    # proposals_per_target[path] = list of (box, conf)
    proposals_B: dict[str, list] = {str(t): [] for t in test_targets}

    for stem in tqdm(stems, desc="[B] stems"):
        src_path, bboxes, cls_list = stem_data[stem]
        visual_prompts = dict(bboxes=bboxes, cls=cls_list)

        # Step 1: set up predictor + get VPE from refer_image (once per stem)
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor as VP
        if not isinstance(model.predictor, VP):
            model.predictor = VP(
                overrides={
                    "task": model.model.task,
                    "mode": "predict",
                    "save": False,
                    "verbose": False,
                    "batch": 1,
                    "imgsz": 640,
                },
                _callbacks=model.callbacks,
            )
        num_cls = len(set(cls_list))
        model.model.model[-1].nc = num_cls
        model.model.names = [f"object{i}" for i in range(num_cls)]
        model.predictor.set_prompts(visual_prompts.copy())
        model.predictor.setup_model(model=model.model)

        # get_vpe bakes refer_image embedding into model weights
        vpe = model.predictor.get_vpe(str(src_path))
        model.model.set_classes(model.model.names, vpe)
        model.task = "segment"
        model.predictor = None  # reset — now it's plain detection with baked VPE

        # Step 2: batch predict all targets in chunks of B
        target_strs = [str(t) for t in test_targets]
        for i in range(0, len(target_strs), B):
            chunk = target_strs[i:i+B]
            try:
                results = model.predict(
                    source=chunk,
                    conf=args.conf, iou=args.iou,
                    verbose=False, agnostic_nms=True,
                )
                for j, r in enumerate(results):
                    if r.boxes is not None:
                        tgt_key = chunk[j]
                        for k in range(len(r.boxes)):
                            box  = r.boxes.xyxy[k].tolist()
                            conf = float(r.boxes.conf[k])
                            proposals_B[tgt_key].append((box, conf))
                            total_proposals_B += 1
            except Exception as e:
                print(f"  [B] stem={stem} chunk={i}: {e}")
                import traceback; traceback.print_exc()
                break

    t_B = time.time() - t0
    print(f"[B] {t_B:.1f}s for {len(test_targets)} targets × {len(stems)} stems | {total_proposals_B} proposals")
    print(f"[B] per-target-equivalent: {t_B/len(test_targets):.2f}s")

    print(f"\n{'='*50}")
    print(f"Speedup: {t_A/t_B:.2f}×  (A={t_A:.1f}s  B={t_B:.1f}s)")
    print(f"Proposals — A: {total_proposals_A}  B: {total_proposals_B}")
    if abs(total_proposals_A - total_proposals_B) > total_proposals_A * 0.1:
        print("⚠️  Proposal counts differ >10% — results may not be equivalent")
    else:
        print("✅  Proposal counts match within 10%")


if __name__ == "__main__":
    main()
