#!/usr/bin/env bash
# k-shot sweep: k labelled instances/class (default 15..50 step 5), Pipeline A and B,
# same labelled pool, same held-out test split → metrics.json + timing.json per run.
# Override any variable via env, e.g.  KS="15 30" B_ARGS="--batch-size 8" bash experiments/run_kshot.sh
set -euo pipefail
cd "$(dirname "$0")/.."

POOL=${POOL:-data/label}   # labelled pool (images + YOLO .txt, flat) — refs / seed crops come only from here
TEST=${TEST:-data/test}    # held-out eval split (images + YOLO .txt, flat) — never used as refs
OUT=${OUT:-runs/kshot}
KS=${KS:-"15 20 25 30 35 40 45 50"}
CLASS_IDS=${CLASS_IDS:-"0 1 2 3 4 5 6 7 8 9 10 11"}
CLASS_NAMES=${CLASS_NAMES:-"helmet gloves vest boots goggles Person no_helmet no_goggle no_gloves no_boots forklift no_vest"}
PHASH=${PHASH:-4}          # same near-duplicate dedup distance for both pipelines
PY=${PY:-python}
A_ARGS=${A_ARGS:-"--no-preview --yoloe-batch-size 16"}   # DINOv2 batch: auto (~70% VRAM)
B_ARGS=${B_ARGS:-"--no-preview --batch-size 16"}

mkdir -p "$OUT"

for K in $KS; do
  echo "================ k=$K ================"

  # ── Pipeline A: k crops/class (phash dedup + DINOv2/DBSCAN diversity) → YOLOe→SAM2→DINOv2→WBF ──
  CROPS="$OUT/crops_k$K"
  rm -rf "$CROPS"
  $PY scripts/extract_crops_labelled.py --images "$POOL" --labels "$POOL" --classes $CLASS_IDS \
      --output "$CROPS" --stride 1 --min-hash-dist "$PHASH" --max-per-class-crops "$K"
  QDIRS=""; for c in $CLASS_IDS; do QDIRS="$QDIRS $CROPS/cls$c"; done
  $PY scripts/yoloe_sam2_dinov2_module.py --queries-dirs $QDIRS --class-ids $CLASS_IDS \
      --targets-dir "$TEST" --source-images "$POOL" --labels "$POOL" \
      --output-dir "$OUT/A_k$K" $A_ARGS 2>&1 | tee "$OUT/A_k$K.log"
  $PY scripts/eval_map.py --preds "$OUT/A_k$K" --gt "$TEST" --images "$TEST" \
      --classes $CLASS_NAMES --out-json "$OUT/A_k$K/metrics.json" | tee "$OUT/A_k$K/eval.txt"

  # ── Pipeline B: k-instance DINOv2 proto bank (first 5 = SAM3 exemplars) → SAM3→DINOv2→voting ──
  $PY scripts/sam3_dinov2_module.py --refs-dir "$POOL" --refs-labels "$POOL" --class-ids $CLASS_IDS \
      --targets-dir "$TEST" --output-dir "$OUT/B_k$K" \
      --dino-proto-size "$K" --max-refs-per-class 5 --phash-max-dist "$PHASH" $B_ARGS 2>&1 | tee "$OUT/B_k$K.log"
  $PY scripts/eval_map.py --preds "$OUT/B_k$K/labels" --gt "$TEST" --images "$TEST" \
      --classes $CLASS_NAMES --out-json "$OUT/B_k$K/metrics.json" | tee "$OUT/B_k$K/eval.txt"
done

$PY experiments/summarize_kshot.py --runs "$OUT" --classes $CLASS_NAMES | tee "$OUT/summary.md"
