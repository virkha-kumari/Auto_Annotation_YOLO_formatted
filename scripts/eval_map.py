"""
Evaluate auto_annotate.py output against ground-truth YOLO labels.
Requires Python 3.10+.

Computes per-class and overall:
  - Precision, Recall, F1 @ configurable IoU (default 0.50)
  - AP@0.50  (101-point interpolation)
  - mAP@0.50:0.95  (COCO-style, 10 thresholds averaged; classes with 0 GT excluded)

Usage:
    python scripts/eval_map.py \\
        --preds   "D:/output"               \\  # auto_annotate output (.txt + summary.json)
        --gt      "D:/dataset/labels/train"  \\  # ground-truth YOLO .txt dir
        --classes helmet gloves vest         \\  # names in class-id order
        [--iou-thresh 0.50]                  \\  # IoU threshold for P/R/F1 row
        [--images  "D:/dataset/images/train"]    # optional: enumerate target stems from here

Box format (pred + GT):  standard YOLO .txt — <cls_id> <cx> <cy> <w> <h>  (normalised)
Scores: read from summary.json if present in preds dir; otherwise uniform 1.0.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Box utilities
# ─────────────────────────────────────────────────────────────────────────────

def yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou_pair(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_yolo_txt(path: Path) -> list[tuple]:
    """Returns [(cls_id, cx, cy, w, h), ...]."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        rows.append((int(parts[0]), float(parts[1]), float(parts[2]),
                     float(parts[3]), float(parts[4])))
    return rows


def load_summary_scores(summary_path: Path) -> dict[str, dict[int, list[float]]]:
    """Returns {stem: {cls_id: [score, ...]}} from a pipeline's summary.json.

    Pipeline A shape: {stem: {"classes": {cls_str: {"boxes": [{"score": ...}, ...]}}}}
    Pipeline B shape: {stem: {"boxes": [{"class_id": ..., "combined_score": ...}, ...]}} (flat, no "classes"
    nesting) — score is combined_score (0.2*sam3_score + 0.8*dino_sim, the same value the pipeline's own
    gate ranks/drops on), not sam3_score alone, since sam3_score is raw objectness/mask-quality, not
    class-discriminative."""
    if not summary_path.exists():
        return {}
    data = json.loads(summary_path.read_text())
    out: dict[str, dict[int, list[float]]] = {}
    for img_name, info in data.items():
        stem = Path(img_name).stem
        if "classes" in info:
            out[stem] = {
                int(cls_str): [b["score"] for b in cls_info.get("boxes", [])]
                for cls_str, cls_info in info.get("classes", {}).items()
            }
        else:
            by_cls: dict[int, list[float]] = {}
            for b in info.get("boxes", []):
                score = b.get("combined_score")
                by_cls.setdefault(b["class_id"], []).append(score if score is not None else 0.0)
            out[stem] = by_cls
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────

def match_preds(
    preds: list[tuple],   # [(score, xyxy), ...] — MUST be sorted desc by score
    gts:   list[tuple],   # [xyxy, ...]
    iou_thresh: float,
) -> list[int]:
    """
    Greedy confidence-sorted matching. Returns TP/FP flags (1/0) per pred.
    Each GT matched at most once. O(|preds| × |gts|).
    """
    matched: set[int] = set()
    flags: list[int] = []
    for _score, pbox in preds:
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(gts):
            if j in matched:
                continue
            v = iou_pair(pbox, gbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh:
            matched.add(best_j)
            flags.append(1)
        else:
            flags.append(0)
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# AP
# ─────────────────────────────────────────────────────────────────────────────

def compute_ap_101(tp_flags: list[int], n_gt: int) -> float:
    """
    101-point interpolated AP.
    tp_flags must be sorted by confidence descending (highest-confidence pred first).
    Returns NaN when n_gt == 0 — callers must handle and exclude from mAP mean.
    """
    if n_gt == 0 or not tp_flags:
        return float("nan")
    tp_cum = np.cumsum(tp_flags, dtype=float)
    fp_cum = np.cumsum([1 - f for f in tp_flags], dtype=float)
    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = recalls >= t
        ap += float(precisions[mask].max()) if mask.any() else 0.0
    return ap / 101.0


def global_tp_flags(
    stems: list[str],
    preds_by_img: dict[str, dict[int, list[tuple]]],
    gts_by_img:   dict[str, dict[int, list[tuple]]],
    cls_id: int,
    iou_thresh: float,
) -> list[int]:
    """
    Build a globally confidence-sorted TP/FP flag list for AP computation.

    AP requires matching in a single global confidence-sorted pass — not per-image
    sums — because a FP in image A must stay a FP even if a matching GT exists in
    image B. Matching is still per-image (each image has its own matched-GT set),
    which prevents a pred in image A from consuming a GT box in image B.

    This is a separate pass from the per-image P/R/F1 computation above: both use
    the same match_preds function, but this one sorts ALL predictions across ALL
    images by score before matching, while the P/R/F1 pass accumulates per-image.
    """
    global_preds = sorted(
        [(score, stem, box)
         for stem in stems
         for score, box in preds_by_img[stem].get(cls_id, [])],
        key=lambda x: -x[0],
    )
    matched_per_img: dict[str, set[int]] = {s: set() for s in stems}
    flags: list[int] = []
    for _score, stem, pbox in global_preds:
        gboxes = gts_by_img[stem].get(cls_id, [])
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(gboxes):
            if j in matched_per_img[stem]:
                continue
            v = iou_pair(pbox, gbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh:
            matched_per_img[stem].add(best_j)
            flags.append(1)
        else:
            flags.append(0)
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    preds_dir:   Path,
    gt_dir:      Path,
    class_names: list[str],
    report_iou:  float,
    images_dir:  Path | None,
) -> None:

    # Enumerate stems
    if images_dir and images_dir.exists():
        stems = sorted(p.stem for p in images_dir.iterdir()
                       if p.suffix.lower() in IMG_EXTS)
    else:
        stems = sorted(
            p.stem for p in preds_dir.glob("*.txt")
            if not p.stem.endswith("_preview") and p.stem != "summary"
        )

    n_cls = len(class_names)
    summary_scores = load_summary_scores(preds_dir / "summary.json")

    # Build per-image pred/GT dicts
    # preds_by_img[stem][cls_id] = [(score, xyxy), ...] sorted desc by score
    # gts_by_img[stem][cls_id]   = [xyxy, ...]
    preds_by_img: dict[str, dict[int, list[tuple]]] = {}
    gts_by_img:   dict[str, dict[int, list[tuple]]] = {}

    for stem in stems:
        pred_raw   = load_yolo_txt(preds_dir / f"{stem}.txt")
        gt_raw     = load_yolo_txt(gt_dir    / f"{stem}.txt")
        img_scores = summary_scores.get(stem, {})

        p_by_cls: dict[int, list] = defaultdict(list)
        g_by_cls: dict[int, list] = defaultdict(list)

        for cls_id, cx, cy, w, h in pred_raw:
            p_by_cls[cls_id].append(yolo_to_xyxy(cx, cy, w, h))
        for cls_id, cx, cy, w, h in gt_raw:
            g_by_cls[cls_id].append(yolo_to_xyxy(cx, cy, w, h))

        preds_by_img[stem] = {}
        for cls_id in range(n_cls):
            boxes  = p_by_cls.get(cls_id, [])
            scores = img_scores.get(cls_id, [1.0] * len(boxes))
            if len(scores) != len(boxes):
                scores = [1.0] * len(boxes)
            preds_by_img[stem][cls_id] = sorted(
                zip(scores, boxes), key=lambda x: -x[0]
            )
        gts_by_img[stem] = {cls_id: g_by_cls.get(cls_id, []) for cls_id in range(n_cls)}

    # ── Per-class metrics ─────────────────────────────────────────────────────
    iou_thresholds = np.round(np.arange(0.50, 0.955, 0.05), 2).tolist()

    print(f"\n{'─'*76}")
    print(f"  {len(stems)} images   P/R/F1 IoU={report_iou:.2f}   classes: {class_names}")
    print(f"{'─'*76}")
    print(f"  {'Class':<14} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'AP@.50':>7} {'AP@.5:.95':>10}"
          f"  {'nPred':>6} {'nGT':>6}")
    print(f"{'─'*76}")

    all_ap50:    list[float] = []
    all_ap_coco: list[float] = []
    per_img_stats: dict[str, dict[int, dict]] = {s: {} for s in stems}

    for cls_id in range(n_cls):
        n_gt_total   = sum(len(gts_by_img[s].get(cls_id, [])) for s in stems)
        n_pred_total = sum(len(preds_by_img[s].get(cls_id, [])) for s in stems)

        # P/R/F1: match per-image at report_iou, accumulate TP/FP/FN
        tp_total = fp_total = fn_total = 0
        for stem in stems:
            flags = match_preds(
                preds_by_img[stem].get(cls_id, []),
                gts_by_img[stem].get(cls_id, []),
                report_iou,
            )
            n_tp = flags.count(1)
            n_fp = flags.count(0)
            n_fn = len(gts_by_img[stem].get(cls_id, [])) - n_tp
            tp_total += n_tp
            fp_total += n_fp
            fn_total += n_fn
            per_img_stats[stem][cls_id] = {
                "tp": n_tp, "fp": n_fp, "fn": n_fn,
                "n_pred": len(preds_by_img[stem].get(cls_id, [])),
                "n_gt":   len(gts_by_img[stem].get(cls_id, [])),
            }

        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
        rec  = tp_total / (tp_total + fn_total) if n_gt_total else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        # AP: global confidence-sorted pass (see global_tp_flags docstring)
        if n_pred_total > 0 and n_gt_total > 0:
            ap50 = compute_ap_101(
                global_tp_flags(stems, preds_by_img, gts_by_img, cls_id, 0.50),
                n_gt_total,
            )
            ap_coco = float(np.nanmean([
                compute_ap_101(
                    global_tp_flags(stems, preds_by_img, gts_by_img, cls_id, float(t)),
                    n_gt_total,
                )
                for t in iou_thresholds
            ]))
        else:
            ap50 = ap_coco = float("nan")

        all_ap50.append(ap50)
        all_ap_coco.append(ap_coco)

        if n_gt_total == 0:
            print(f"  {class_names[cls_id]:<14}  [excluded from mAP — 0 GT instances in this eval set]")
            continue

        ap50_s = f"{ap50:.3f}"   if not np.isnan(ap50)   else "  N/A "
        coc_s  = f"{ap_coco:.3f}" if not np.isnan(ap_coco) else "   N/A  "
        print(f"  {class_names[cls_id]:<14} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}"
              f"  {ap50_s:>7} {coc_s:>10}  {n_pred_total:>6} {n_gt_total:>6}")

    map50    = float(np.nanmean(all_ap50))
    map_coco = float(np.nanmean(all_ap_coco))
    tot_pred = sum(len(preds_by_img[s].get(c, [])) for s in stems for c in range(n_cls))
    tot_gt   = sum(len(gts_by_img[s].get(c, []))   for s in stems for c in range(n_cls))

    print(f"{'─'*76}")
    print(f"  {'ALL':<14} {'':>6} {'':>6} {'':>6}  {map50:>7.3f} {map_coco:>10.3f}"
          f"  {tot_pred:>6} {tot_gt:>6}")
    print(f"{'─'*76}\n")

    # ── Per-image breakdown ───────────────────────────────────────────────────
    print(f"Per-image breakdown @ IoU={report_iou:.2f}  (tp/fp/fn per class)\n")
    col_w = 12
    header = f"  {'Stem':<35}" + "".join(f"  {n[:col_w]:>{col_w}}" for n in class_names)
    print(header)
    print(f"  {'':─<35}" + "".join(f"  {'─'*col_w}" for _ in class_names))
    for stem in stems:
        row = f"  {stem[:35]:<35}"
        for cls_id in range(n_cls):
            st = per_img_stats[stem].get(cls_id, {"tp": 0, "fp": 0, "fn": 0})
            cell = f"{st['tp']}/{st['fp']}/{st['fn']}"
            row += f"  {cell:>{col_w}}"
        print(row)
    print()


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO-format predictions vs ground truth."
    )
    parser.add_argument("--preds",      required=True,  help="Predicted YOLO .txt dir (+ summary.json)")
    parser.add_argument("--gt",         required=True,  help="Ground-truth YOLO .txt dir")
    parser.add_argument("--classes",    required=True,  nargs="+", help="Class names in id order")
    parser.add_argument("--iou-thresh", type=float, default=0.50, dest="iou_thresh",
                        help="IoU threshold for P/R/F1 report row (default 0.50)")
    parser.add_argument("--images",     default=None,   help="Images dir for stem enumeration")
    args = parser.parse_args()

    evaluate(
        preds_dir   = Path(args.preds),
        gt_dir      = Path(args.gt),
        class_names = args.classes,
        report_iou  = args.iou_thresh,
        images_dir  = Path(args.images) if args.images else None,
    )


if __name__ == "__main__":
    main()
