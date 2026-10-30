# CLAUDE.md — Auto-Annotation Pipeline

## What this project is

Few-shot auto-annotation pipeline for industrial computer vision datasets.
Goal: label 10–15 instances per class → auto-annotate large unlabeled image sets → YOLO `.txt` output.
Used across multiple industrial computer vision projects.

---

## Docs structure

| File | Purpose |
|---|---|
| `README.md` | Full project overview, ablation, quick start, parameters |
| `docs/plans.md` | What to do next — forward-looking only |
| `docs/log.md` | Chronological log — ADDED features, DROPPED approaches, FINDINGS with reasons |

**Always update `docs/log.md`** when:
- A new script or capability is added → `ADDED`
- A model, approach, or file is removed → `DROPPED` with reason
- An experimental result or insight is discovered → `FINDING`

---

## What's decided vs what's not

### DECIDED (finalized)

- **`scripts/extract_crops_labelled.py`** — seed crop extraction from YOLO-annotated datasets (phash dedup + DINOv2/DBSCAN clustering). Done.
- **`scripts/extract_crops_varied.py`** — seed crop extraction without clustering (all crops). Done.
- **Proposal generation → YOLOe** — confirmed working on a large industrial object class and Construction-PPE at conf>=0.06. Visual-prompt mode: `get_vpe(refer_image)` bakes VPE into model → batch predict all targets.
- **SAM2 role** — not a proposer. Masked crop producer for normal-size classes. Small classes skip SAM2 entirely.
- **DINOv2 embedding — two modes:**
  - Normal-size objects: masked patch pooling (mean of patch tokens ≥50% covered by mask, 16×16 grid; mask passed explicitly)
  - Small objects (p90 area of selected refs < `--small-obj-thresh`): "CLS mode" = mean of all tokens on raw bbox crop
  - DINOv2 input: square 224 resize, **no center crop** (keeps mask aligned with patch grid)
  - Auto-detected per class at startup from the selected ref instances (not the whole label folder)
- **Ref instances = exact crops** — crop names `<stem>_cls<id>_<line_idx>` carry the raw label line index; Pipeline A uses exactly those instances (k crops = k instances).
- **WBF consolidated score** — `0.3 × yoloe_conf + 0.7 × dino_sim` per box before WBF fusion. Single-model WBF = cluster average (no vote boost) — intended for Pipeline A.
- **Pipeline VRAM order** — YOLOe (all targets, batched) → unload → SAM2 (refs + all target proposals) → unload → DINOv2 (embed + score all) → unload → WBF + output.
- **Batched YOLOe** — outer=stems, inner=target batches. `get_vpe` bakes VPE once per stem, `predict(source=[batch])` runs plain detection on all targets. 2.91× speedup confirmed.
- **`scripts/yoloe_sam2_dinov2_module.py`** (formerly `auto_annotate.py`) — BUILT AND WORKING. Multi-class, batched, small-obj-aware. Pipeline A.
- **`scripts/sam3_dinov2_module.py`** — BUILT AND WORKING (integrated 2026-07-14). Canvas-composite few-shot, no crop-extraction step. Order: SAM3 proposals → combined-score gate (`--sam3-dino-thresh`, default 0.2, on `0.2*sam3_score + 0.8*dino_sim`) → voting (dups at IoU ≥ `--dup-iou-thresh` merge; `final_score = combined × voting_ref_groups / ref_groups`) → containment filter (2026-10-30). SAM3-unmasked refs dropped from bank (no mixed embeddings); masks box-local + bit-packed for scale. Pipeline B. DINOv2 scoring now masked-patch pooling (SAM3's own masks reused for proposals; ref crops box-prompted through SAM3 too) + max-sim vs proto bank, matching Pipeline A's approach — same small-object CLS fallback (`--small-obj-thresh`) (2026-07-20). Ref bbox padding (`--ref-box-padding`, default 0.01) added 2026-07-23 to fix fingertip/thin-extremity clipping from tight ref boxes.
- **`app.py`** — Gradio wizard wrapping both pipelines. Landing page picks Pipeline A or B. WORKING.
- **No-mask override + skip preview** (2026-07-23, both pipelines) — `--no-mask-classes` forces explicit class ids into small-object handling regardless of `--small-obj-thresh`. `--no-preview` skips preview image save (labels + summary.json still written, `preview_file: null`). Wired into app.py for both pipelines.

---

## Current scripts

| Script | What it does | Status |
|---|---|---|
| `scripts/yoloe_sam2_dinov2_module.py` | Pipeline A: YOLOe→SAM2→DINOv2→WBF→YOLO .txt | **ACTIVE — working** |
| `scripts/sam3_dinov2_module.py` | Pipeline B: SAM3 canvas-composite few-shot→DINOv2 masked-patch max-sim gate→containment/dup filter→YOLO .txt | **ACTIVE — working** |
| `app.py` | Gradio wizard UI — landing page picks Pipeline A or B | **ACTIVE — working** |
| `scripts/extract_crops_labelled.py` | Seed crop extraction with clustering (Pipeline A only) | Active |
| `scripts/extract_crops_varied.py` | Seed crop extraction without clustering | Active |
| `scripts/eval_map.py` | mAP/P/R/F1 evaluation vs ground truth | Active |
| `experiments/debug_yoloe_sam2_dino.py` | Pipeline A debug, 4-panel matplotlib viz | Active |
| `experiments/debug_sam3.py` | Pipeline B debug, 4-panel matplotlib viz per (ref, target) pair | Active |
| `experiments/test_yoloe_batch.py` | YOLOe batching benchmark (2.91× confirmed) | Reference |
| `experiments/debug_yoloe.py` | Pure YOLOe visual-prompt debug | Reference |
| `experiments/debug_yoloe_dinov2.py` | YOLOe+DINOv2, 3-panel viz, no SAM2 (predates `debug_yoloe_sam2_dino.py`) | Reference |
| `experiments/test_owlv2_dinov2.py` | OWLv2+DINOv2 — dead end | Dead end |
| `experiments/run_kshot.sh` | k-shot sweep (k=15/30/50/100), A + B, eval → `runs/kshot/` | Active |
| `experiments/summarize_kshot.py` | k-shot runs → markdown tables for README | Active |

---

## Hard rules

- **No text prompts anywhere** — pipeline is purely visual
- **Keep outlier clusters** — occluded/unusual views are valid
- **Monolith first** — keep each pipeline module (`scripts/yoloe_sam2_dinov2_module.py`, `scripts/sam3_dinov2_module.py`) as one script until design stabilizes
- **Human review via PIL preview** — no Label Studio or CVAT integration planned (optional via `--no-preview`, labels-only mode)
- **Device-independent code always** — never hardcode cache paths; code must run on any machine
- **VRAM rule** — never load two large models simultaneously (see below)
- **Embedding consistency** — refs and proposals for each class must use identical embedding method. If a class uses CLS mode, both proto bank and proposals use CLS. Never mix methods within a class.

---

## Key files

| File | Role | Status |
|---|---|---|
| `scripts/yoloe_sam2_dinov2_module.py` | Pipeline A production pipeline | **ACTIVE** |
| `scripts/sam3_dinov2_module.py` | Pipeline B production pipeline (canvas-composite few-shot + combined-score gate) | **ACTIVE — integrated into app.py; smoke-tested only, full mAP eval pending** |
| `app.py` | Gradio UI, both pipelines | **ACTIVE** |
| `scripts/extract_crops_labelled.py` | Seed crop extraction (clustered) | FINALIZED |
| `scripts/extract_crops_varied.py` | Seed crop extraction (all) | FINALIZED |
| `experiments/debug_yoloe_sam2_dino.py` | Pipeline A debug with 4-panel viz | ACTIVE |
| `experiments/debug_sam3.py` | Pipeline B debug, 4-panel viz per (ref, target) pair | ACTIVE |
| `requirements.txt` | Python deps (PyTorch CUDA 11.8, transformers, etc.) | Exists |

---

## Models

| Model | Role | ID / Package | Verdict |
|---|---|---|---|
| YOLOe | Visual-prompt proposal generation | `yoloe-11l-seg.pt` (ultralytics) | **CONFIRMED** — works on a large industrial object class + Construction-PPE at conf>=0.06 |
| SAM2 | Masked crop producer (not proposer) | `facebook/sam2.1-hiera-base-plus` | **CONFIRMED** for masking — failed as auto-proposer |
| DINOv2 | Embedding + cosine sim scoring | `facebook/dinov2-base` | **CONFIRMED** — masked patch pooling (normal) + CLS (small) |
| SAM3 | Region proposal (unprompted auto-mask) | `facebook/sam3` | Failed — same issue as SAM2 |
| SAM3 | Cross-image few-shot (canvas-composite exemplar, `scripts/sam3_dinov2_module.py`) | `facebook/sam3` | **CONFIRMED working end-to-end, integrated as Pipeline B** in `app.py` (2026-07-14). No native cross-image exemplar API (confirmed via SAM3 own docs/notebooks — strictly single-video/single-image, no few-shot) — ref + target composited onto one canvas, ref bbox remapped to canvas coords, box-only exemplar prompt (no text). Gated model, requires HF auth. Order: combined-score gate (`--sam3-dino-thresh`, default 0.2, on `0.2*sam3_score + 0.8*dino_sim`) first, then containment + duplicate suppression. SAM3 also box-prompted (single-image, own GT box) on ref crops to get masks for DINOv2 scoring — reuses the already-loaded SAM3 instead of a separate SAM2 phase, own-image box-prompt is a different mode from the failed unprompted auto-mask row above. All calibrated 2026-07-20; smoke-tested only, full mAP eval pending. |
| OWLv2 | Image-guided detection | `google/owlv2-base-patch16-ensemble` | Dead end — cannot detect large objects |

Cached at `~/.cache/huggingface/hub/`. YOLOe downloaded by ultralytics on first use.

---

## VRAM rule (CRITICAL — 8GB GPU)

**Never load two large models in VRAM at the same time.**

Always use this pattern:
1. Load model A → do work → `del model` + `torch.cuda.empty_cache()`
2. Load model B → do work → `del model` + `torch.cuda.empty_cache()`

Use `del` + `torch.cuda.empty_cache()` — not just `.cpu()` or `.to("cpu")`.

---

## Key findings from testing

1. **YOLOe WORKS** — confirmed on a large industrial object class and Construction-PPE at conf>=0.06.
2. **Batched YOLOe = 2.91× speedup** — `get_vpe` bakes VPE per stem, then `predict(source=[batch])` on all targets. Proposals identical to per-target loop.
3. **I/O bottleneck was the real YOLOe slowdown** — label file reads inside target loop caused 4–6× slowdown vs debug script. Fixed by pre-building `stem_prompts` dict before loops.
4. **Source-image context filtering is unnecessary for YOLOe** — dropped. Was designed for OWLv2's weakness.
5. **OWLv2 cannot detect large objects** — patch-based ViT, tested at 640/1008/1280px. Dead end.
6. **SAM2/SAM3 failed as proposal generators** — SAM2 repurposed as masked crop producer.
7. **Averaged DINOv2 prototypes dilute signal** — per-crop scoring, no averaging.
8. **Full-crop DINOv2 CLS too noisy** — 0.19–0.43 sims. Switched to masked patch pooling.
9. **Small objects break masked patch pooling** — cls1 (gloves, area≈0.007) got 0/N passing dino-thresh. 1–2 patches covered → high-variance embedding. Fixed: CLS on raw bbox crop.
10. **WBF nested box problem** — handled by post-WBF containment filter (intersection/min_area).
11. **Pipeline B had the same CLS-on-raw-crop noise as finding #8** — SAM3 gives masks but the pipeline only used them for the box (`mask_to_bbox`), never for DINOv2 scoring, so all proposals scored on raw-crop CLS regardless of background clutter. Fixed by reusing SAM3's mask directly (proposals) and box-prompting SAM3 on ref crops too (single-image, own GT box, no canvas) — same masked-patch pooling as Pipeline A, same small-obj CLS fallback. Also switched proto-bank sim from mean to max (best-matching ref), matching Pipeline A.

---

## Calibration values (current defaults, all tunable)

### Pipeline A (YOLOe/SAM2/DINOv2)

- YOLOe confidence: `0.06`
- YOLOe target batch size: `8`
- DINOv2 cosine similarity threshold (`--dino-thresh`): `0.65`
- Small object threshold (`--small-obj-thresh`): `0.01` (p90 w×h normalised)
- NMS IoU (`--nms-iou`): `0.45`
- WBF score gate (`--wbf-score`): `0.10`
- Result threshold (`--result-thresh`): `0.50`
- Containment filter (`--containment-thresh`): `0.70`
- WBF score formula: `0.3 × yoloe_conf + 0.7 × dino_sim`
- SAM2 mask padding: `0.05`
- SAM2 score min: `0.50`
- SAM2 area min: `0.10`
- Force no-mask classes (`--no-mask-classes`): none by default
- Skip preview (`--no-preview`): off by default

### Pipeline B (SAM3/DINOv2)

- SAM3 combined-score threshold (`--sam3-dino-thresh`): `0.2` (on `0.2*sam3_score + 0.8*dino_sim`)
- Small object threshold (`--small-obj-thresh`): `0.01` (p90 w×h normalised, same as Pipeline A)
- Containment filter (`--containment-thresh`): `0.85`
- Duplicate IoU (`--dup-iou-thresh`): `0.85`
- SAM3 score threshold (`--threshold`): `0.6`
- Max refs per class for SAM3 exemplars (`--max-refs-per-class`): `5`
- DINOv2 proto bank size (`--dino-proto-size`): `100`
- Phash dedup distance (`--phash-max-dist`): `4`
- Ref bbox padding (`--ref-box-padding`): `0.01` (fractional, clamped to image bounds)
- Force no-mask classes (`--no-mask-classes`): none by default
- Skip preview (`--no-preview`): off by default

---

## Known limitations

- **OWLv2** cannot detect large objects (patch-based ViT architecture)
- **SAM2** fail as proposal generators for large uniform regions
- **DINOv2 masked-patch pooling** breaks for tiny objects (<2% frame area) — use CLS mode
- **Threshold calibration** not validated across datasets — tune per project
