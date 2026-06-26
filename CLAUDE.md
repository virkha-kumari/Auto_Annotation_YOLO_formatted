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
| `docs/notes.md` | Pipeline design decisions, open questions, dataset overview |
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
  - Normal-size objects: masked patch pooling (mean pool tokens inside SAM2 mask, 16×16 grid)
  - Small objects (median bbox area < `--small-obj-thresh`): CLS token on raw bbox crop
  - Auto-detected per class at startup from source label statistics
- **WBF consolidated score** — `0.3 × yoloe_conf + 0.7 × dino_sim` per box before WBF fusion.
- **Pipeline VRAM order** — YOLOe (all targets, batched) → unload → SAM2 (refs + all target proposals) → unload → DINOv2 (embed + score all) → unload → WBF + output.
- **Batched YOLOe** — outer=stems, inner=target batches. `get_vpe` bakes VPE once per stem, `predict(source=[batch])` runs plain detection on all targets. 2.91× speedup confirmed.
- **`scripts/auto_annotate.py`** — BUILT AND WORKING. Multi-class, batched, small-obj-aware.
- **`app.py`** — Gradio 3-page wizard wrapping the full pipeline. WORKING.

### NOT DECIDED (under active experimentation)

- **Threshold calibration** — `--dino-thresh` (0.65), `--wbf-score` (0.10), `--result-thresh` (0.5), `--containment-thresh` (0.70), `--small-obj-thresh` (0.02) all need per-dataset tuning.
- **WBF score weights** — `0.3/0.7` split chosen, not yet validated across datasets.
- **mAP evaluation** — pipeline output vs ground truth on Construction-PPE train/ not yet run.

---

## Current scripts

| Script | What it does | Status |
|---|---|---|
| `scripts/auto_annotate.py` | Full pipeline: YOLOe→SAM2→DINOv2→WBF→YOLO .txt | **ACTIVE — working** |
| `app.py` | Gradio 3-page wizard UI | **ACTIVE — working** |
| `scripts/extract_crops_labelled.py` | Seed crop extraction with clustering | Active |
| `scripts/extract_crops_varied.py` | Seed crop extraction without clustering | Active |
| `test/debug_yoloe_sam2_dino.py` | Full pipeline debug, 4-panel matplotlib viz | Active |
| `test/test_yoloe_batch.py` | YOLOe batching benchmark (2.91× confirmed) | Reference |
| `utils/debug_yoloe.py` | Pure YOLOe visual-prompt debug | Reference |
| `utils/test_owlv2_dinov2.py` | OWLv2+DINOv2 — dead end | Dead end |

---

## Hard rules

- **No text prompts anywhere** — pipeline is purely visual
- **Keep outlier clusters** — occluded/unusual views are valid
- **Monolith first** — keep `auto_annotate.py` as one script until design stabilizes
- **Human review via PIL preview** — no Label Studio or CVAT integration planned
- **Device-independent code always** — never hardcode cache paths; code must run on any machine
- **VRAM rule** — never load two large models simultaneously (see below)
- **Embedding consistency** — refs and proposals for each class must use identical embedding method. If a class uses CLS mode, both proto bank and proposals use CLS. Never mix methods within a class.

---

## Key files

| File | Role | Status |
|---|---|---|
| `scripts/auto_annotate.py` | Main production pipeline | **ACTIVE** |
| `app.py` | Gradio UI | **ACTIVE** |
| `scripts/extract_crops_labelled.py` | Seed crop extraction (clustered) | FINALIZED |
| `scripts/extract_crops_varied.py` | Seed crop extraction (all) | FINALIZED |
| `test/debug_yoloe_sam2_dino.py` | Debug pipeline with 4-panel viz | ACTIVE |
| `requirements.txt` | Python deps (PyTorch CUDA 11.8, transformers, etc.) | Exists |

---

## Models

| Model | Role | ID / Package | Verdict |
|---|---|---|---|
| YOLOe | Visual-prompt proposal generation | `yoloe-11l-seg.pt` (ultralytics) | **CONFIRMED** — works on a large industrial object class + Construction-PPE at conf>=0.06 |
| SAM2 | Masked crop producer (not proposer) | `facebook/sam2.1-hiera-base-plus` | **CONFIRMED** for masking — failed as auto-proposer |
| DINOv2 | Embedding + cosine sim scoring | `facebook/dinov2-base` | **CONFIRMED** — masked patch pooling (normal) + CLS (small) |
| SAM3 | Region proposal | — | Failed — same issue as SAM2 |
| OWLv2 | Image-guided detection | `google/owlv2-base-patch16-ensemble` | Dead end — cannot detect large objects |

Cached at `C:/Users/Lenovo/.cache/huggingface/hub/`. YOLOe downloaded by ultralytics on first use.

---

## VRAM rule (CRITICAL — 8GB GPU)

**Never load two large models in VRAM at the same time.**

Always use this pattern:
1. Load model A → do work → `del model` + `torch.cuda.empty_cache()`
2. Load model B → do work → `del model` + `torch.cuda.empty_cache()`

Use `del` + `torch.cuda.empty_cache()` — not just `.cpu()` or `.to("cpu")`.

---

## Key findings from testing

1. **YOLOe WORKS** — confirmed on a large industrial object class (black panel) and Construction-PPE at conf>=0.06.
2. **Batched YOLOe = 2.91× speedup** — `get_vpe` bakes VPE per stem, then `predict(source=[batch])` on all targets. Proposals identical to per-target loop.
3. **I/O bottleneck was the real YOLOe slowdown** — label file reads inside target loop caused 4–6× slowdown vs debug script. Fixed by pre-building `stem_prompts` dict before loops.
4. **Source-image context filtering is unnecessary for YOLOe** — dropped. Was designed for OWLv2's weakness.
5. **OWLv2 cannot detect large objects** — patch-based ViT, tested at 640/1008/1280px. Dead end.
6. **SAM2/SAM3 failed as proposal generators** — SAM2 repurposed as masked crop producer.
7. **Averaged DINOv2 prototypes dilute signal** — per-crop scoring, no averaging.
8. **Full-crop DINOv2 CLS too noisy** — 0.19–0.43 sims. Switched to masked patch pooling.
9. **Small objects break masked patch pooling** — cls1 (gloves, area≈0.007) got 0/N passing dino-thresh. 1–2 patches covered → high-variance embedding. Fixed: CLS on raw bbox crop.
10. **WBF nested box problem** — handled by post-WBF containment filter (intersection/min_area).

---

## Calibration values (current defaults, all tunable)

- YOLOe confidence: `0.06`
- YOLOe target batch size: `8`
- DINOv2 cosine similarity threshold (`--dino-thresh`): `0.65`
- Small object threshold (`--small-obj-thresh`): `0.02` (p90 w×h normalised)
- NMS IoU (`--nms-iou`): `0.45`
- WBF score gate (`--wbf-score`): `0.10`
- Result threshold (`--result-thresh`): `0.50`
- Containment filter (`--containment-thresh`): `0.70`
- WBF score formula: `0.3 × yoloe_conf + 0.7 × dino_sim`
- SAM2 mask padding: `0.05`
- SAM2 score min: `0.50`
- SAM2 area min: `0.10`

---

## Known limitations

- **OWLv2** cannot detect large objects (patch-based ViT architecture)
- **SAM2/SAM3** fail as proposal generators for large uniform regions
- **DINOv2 masked-patch pooling** breaks for tiny objects (<2% frame area) — use CLS mode
- **Threshold calibration** not validated across datasets — tune per project
- **WBF 0.3/0.7 weights** not validated across datasets
