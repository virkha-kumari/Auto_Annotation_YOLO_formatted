# Auto-Annotation Pipeline — Log

Format:
- **ADDED** — new feature, script, capability
- **DROPPED** — removed approach, model, or file with reason
- **FINDING** — experimental result, benchmark, or insight

---

## 2026-07-09

### DROPPED — SAM2 as few-shot matcher

Considered extending `test/debug_sam2.py` into a SAM2-only few-shot script: ref crops → SAM2 auto-mask → DINOv2 embed/match → apply to test image. Dropped before implementation.

**Why dropped:**
- SAM2 has no native few-shot or class-conditioning capability — it only segments, doesn't recognize. Any "few-shot" behavior would come entirely from DINOv2 similarity matching bolted on top.
- YOLOe, already the pipeline's detector, has genuine native few-shot support (`get_vpe(refer_image)` bakes visual-prompt embeddings into the model, then detects the same visual concept on new images) — that's why YOLOe was incorporated as proposal generator in the first place, not SAM2. Re-deriving few-shot on SAM2 via DINOv2 would just duplicate what YOLOe already does natively, with a weaker signal.
- `test/debug_yoloe_sam2_dino.py` already covers ref→proposal→match: YOLOe few-shot detects, SAM2 refines mask, DINOv2 confirms similarity. No new capability from a SAM2-only variant.
- Redirecting effort to SAM3, which reportedly has native few-shot/visual-concept prompting similar to YOLOe's VPE — worth evaluating directly instead.

**Next:** Evaluate SAM3 few-shot/visual-prompt capability directly.

---

## 2026-06-26 (session 2)

### ADDED — Multi-class single-pass YOLOe (`scripts/auto_annotate.py`)

Rewrote Phase 1 to handle all classes in one pipeline pass instead of per-class loops:
- `--queries-dirs` + `--class-ids` (nargs="+") replace single `--class-id`/`--queries`
- `stem_prompts` pre-built before target loop: reads all label files once, builds `dict[stem → (src_path, visual_prompts)]` with mixed class bboxes per stem
- Inner loop iterates pre-built dict — zero disk I/O inside target loop
- Single subprocess call from `app.py` for all classes

**Performance:** 162 stems (union of cls0/1/2) → ~21-26s/img for 3 classes, vs ~9s for 1 class. Linear with stem count.

### FINDING — YOLOe call bottleneck was label file I/O, not GPU

Before fix: `resolve_class_bboxes_padded()` (opens source image + reads label `.txt`) called inside `for target in targets` loop → `N_targets × N_stems` disk reads → 40-70s/target.  
After fix: pre-build stem_prompts before loops → 9-11s/target (matches debug script).  
Root cause confirmed: pure I/O overhead, not GPU or YOLOe itself.

### FINDING — YOLOe batching across multiple refer_images is impossible

Investigated ultralytics source (`predict.py`, `model.py`). Findings:
- `refer_image` is processed via `get_vpe()` → `set_classes(names, vpe)` which **bakes VPE into model weights**
- `batch=1` hardcoded in predictor overrides during VP setup (line 399 `model.py`)
- Docs confirm: "For image-specific prompts, run images individually"
- Batching targets with same refer_image possible architecturally but `batch` param only works for directory/video/txt source, not list — confirmed no speedup

### FINDING — cls1 (gloves) 0% DINOv2 pass rate — small object problem

cls1 median bbox area = 0.007 (0.7% of image). At DINOv2's 16×16 patch grid:
- Each patch covers ~14×14px of 224px input
- Glove at 8% frame width → ~18px after resize → covers 1-2 patches only
- Masked-patch pooling of 1-2 tokens = high-variance embedding → cosine sim consistently <0.65
- Result: 0/N cls1 proposals pass dino-thresh on every target image

### ADDED — Small-object embedding mode (`scripts/auto_annotate.py`)

Per-class auto-detection of small objects via `median_bbox_area()` on source labels.  
If `median_area < --small-obj-thresh` (default 0.02):
- Phase 2a: skip SAM2, use raw bbox crop (`make_bbox_crop`) — no mask needed for tiny objects
- Phase 2b: skip SAM2, use raw bbox crop from target image
- Phase 3: use CLS token directly (`use_cls=True`) instead of masked-patch pooling

Consistency enforced: refs and proposals for same class always use identical embedding method — proto bank and proposal embeddings live in same subspace → cosine sim meaningful.

New helper: `make_bbox_crop(img_np, bbox)` — raw PIL crop, no mask, no resize.  
New helper: `median_bbox_area(labels_dir, cls_id)` — median w*h from all label files.  
`embed_masked_crops()` gains `use_cls: bool` param.

### ADDED — `test/test_yoloe_batch.py`

Test script comparing Method A (current: 1-target × 1-stem per call) vs Method B (get_vpe once per stem → plain detection on batched targets). Written to benchmark whether YOLOe supports true target batching after VPE bake-in. Method B turned out infeasible (batch param ignored for list source). Script kept for reference.

### ADDED — `app.py` major fixes

- Fixed duplicate `encoding` kwarg `SyntaxError` (line 147)
- Fixed `UnicodeEncodeError` cp1252 for `→` chars: all Popen calls get `env={**os.environ, "PYTHONIOENCODING": "utf-8"}`
- Added `source_images_dir_p2` + `source_labels_dir_p2` inputs on Page 2 (was using Page 1 images dir as labels dir — caused "Empty prototype bank" abort)
- Added skip-crops shortcut on Page 1: `skip_crops_dir` textbox + `skip_to_p2_btn` → sets `crops_dir_state`, navigates to Page 2
- `is_tqdm_line()` filter: suppresses tqdm progress bars from pipeline log (keeps phase/saved/error lines)
- `run_pipeline` now single subprocess call for ALL classes (no per-class loop)
- Fixed duplicate `crops_dir_state = gr.State("")` declaration (line 654 shadowed line 585)
- `summary.json` flat at output root (not per-cls subfolder)

### FINDING — Preview color legend (`app.py` / `save_preview`)

`COLORS = [(255,140,0),(0,200,255),(0,255,100),(255,80,80),(180,0,255)]`
- cls0 = orange, cls1 = cyan, cls2 = green, cls3 = red, cls4 = purple

---

## 2026-06-26

### ADDED — `scripts/auto_annotate.py` — production pipeline script

Full pipeline (no matplotlib): YOLOe → SAM2 → DINOv2 → WBF → containment filter → YOLO `.txt` output + PIL preview + `summary.json`.

- Same 4-phase VRAM-sequential logic as debug script
- `save_preview()` uses PIL ImageDraw (orange boxes, no matplotlib)
- Saves flat YOLO `.txt`: `output_dir/<target_stem>.txt`
- Saves `output_dir/<target_stem>_preview.jpg`
- Saves `output_dir/summary.json`: `{target_name: {n_proposals, n_wbf, n_final, label_file, preview_file, boxes}}`
- Key args: `--class-id` (required), `--result-thresh` (0.50), `--containment-thresh` (0.70)

### ADDED — `app.py` — Gradio 3-page wizard

3-page wizard UI wrapping the full pipeline:

- **Page 1:** Dataset setup + crop extraction. Browse buttons (tkinter `askdirectory`/`askopenfilename`) — no Gradio file upload. Calls `scripts/extract_crops_labelled.py` (unique crops) or `scripts/extract_crops_varied.py` (all crops). Streams subprocess stdout live.
- **Page 2:** Pipeline args + per-class run. Calls `scripts/auto_annotate.py` sequentially per selected class. All float thresholds as `gr.Number`, sliders only for discrete integer params (stride, min_hash_dist, batch_size, dino_batch_size).
- **Page 3:** Results gallery (reads `preview_file` from `summary.json`) + YOLO `.txt` download.
- Theme: `gr.themes.Neon()` (magenta/black)
- All subprocess Popen calls: `encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"}` — fixes cp1252 UnicodeEncodeError on Windows

### ADDED — Phase 1 YOLOe batch-targets optimisation (`scripts/auto_annotate.py`)

Was: `N_sources × N_targets` YOLOe forward passes (one call per source per target).  
Now: `N_sources` calls total — pass all target paths as list to `yoloe_model.predict(source=[...])`.  
Result: O(S×T) → O(S). On 50 source frames × 200 targets = ~200× fewer YOLOe forward passes.

`run_phase1_yoloe()` signature changed: takes `target_paths: list[Path]`, returns `dict[Path, list]` directly.

### ADDED — Phase 2b SAM2 batch-boxes optimisation (`scripts/auto_annotate.py`)

Was: N serial `predictor.predict(box=single_bbox)` calls per target.  
Now: stack all proposal boxes → `[N, 4]` array → single `predictor.predict(box=boxes_np)` call.  
SAM2 decoder runs once per target image regardless of proposal count. ~N× speedup on Phase 2b.

### ADDED — DINOv2 batch size bumped 16 → 32 (`scripts/auto_annotate.py`)

DINOv2-base (~340MB) leaves enough headroom on 8GB VRAM for batch 32. Halves number of DINOv2 forward passes.

### FINDING — Construction-PPE dataset confirmed for demo/eval

Dataset: Ultralytics Construction-PPE ([docs](https://docs.ultralytics.com/datasets/detect/construction-ppe))  
- 1,416 images, 11 classes: `helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots`  
- Native Ultralytics YOLO format — no conversion needed  
- Direct download: `https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip`  
- License: AGPL-3.0  

**Demo split:** `valid/` (143 images, human-labelled) → seed crops. `train/` (1,132 images) → unlabeled targets. Ground truth exists for train/ → enables mAP eval of pipeline output vs human labels.

---

## 2026-06-25

### ADDED — `test/debug_yoloe_sam2_dino.py` major refactor: correct VRAM sequencing, consolidated scoring, containment filter, 4-panel viz

**Pipeline order fixed (was broken — SAM2 was loading before YOLOe):**
- Phase 1: YOLOe loads once → ALL targets processed → YOLOe unloads
- Phase 2: SAM2 loads once → ref masked crops (Job A) + ALL target proposal masks (Job B) → SAM2 unloads
- Phase 3: DINOv2 loads once → proto bank built once → ALL targets scored → DINOv2 unloads
- Phase 4: WBF + containment filter + figures (no model in VRAM)
- Never two models in VRAM simultaneously — enforced correctly now

**Consolidated WBF scoring:**
- Was: WBF input = DINOv2 sim only
- Now: WBF input = `0.3 × yoloe_conf + 0.7 × dino_sim` (combined per-box before fusion)
- WBF then amplifies by vote count (how many source frames agreed on that location)
- Final score reflects detection confidence + appearance similarity + cross-frame consensus

**Containment filter (`--final-containment-thresh`, default 0.85):**
- Post-WBF: for each box pair, compute `intersection / min(area_i, area_j)`
- If ratio > threshold → nested boxes → drop lower-scoring one
- Catches WBF-surviving nested boxes that don't merge because IoU is low (different sizes)

**4-panel result figure:**
- Panel 1: raw proposals passing `--dino-thresh` (color by dino sim)
- Panel 2: all WBF output boxes (lime)
- Panel 3: WBF boxes ≥ `--result-panel3-thresh` (cyan)
- Panel 4: after containment filter (orange) — new, for debugging nested box removal

**Dead code removed:** `conf_color()`, `MASK_OVERLAY_COLORS`, `run_phase3_dinov2()` (inlined into main)

**YOLOe banner spam suppressed:** `ultralytics.utils.LOGGER.setLevel("WARNING")` before predict loop

---

## 2026-06-21

### ADDED — `test/debug_yoloe_sam2_dino.py` (YOLOe + SAM2 masked-crop refs + DINOv2 masked-patch scoring)

New 3-phase pipeline test script replacing `utils/debug_yoloe_dino.py`. Key evolution: SAM2 is now used for **reference crop refinement** (not proposal generation), and DINOv2 uses **masked patch pooling** instead of full-crop CLS embedding.

**Pipeline (sequential VRAM — 8GB rule):**
1. **Phase 1 — SAM2:** For each source image → predict SAM2 mask from label bbox → clean masked crop. Also: for target image → SAM2 predict mask per YOLOe proposal bbox (used later for DINOv2 embedding).
2. **Phase 2 — YOLOe:** Load after SAM2 unloaded. For each source group → `refer_image` + bboxes from label → proposals on target.
3. **Phase 3 — DINOv2:** Load after YOLOe unloaded. Embed ref masked crops → prototype bank. Embed proposal masked crops → cosine sim → threshold → WBF → figures saved to `output_results/`.

**DINOv2 masked patch pooling:**
- Extract patch tokens (`last_hidden_state[:, 1:, :]`) → reshape to 16×16 grid
- Resize mask to 16×16 → mean pool only tokens inside mask
- Fallback to CLS token if mask covers zero patch tokens
- Embeds object foreground only — background not included in similarity score

**Key design change vs previous scripts:**
- SAM2 no longer used for proposal generation (that failed) — used for masking reference crops
- DINOv2 scores masked crops, not full bbox crops — reduces background noise in embeddings

**Status at 2026-06-21:** YOLOe phase failing — `visual_prompts` passes `masks` key but YOLOe API requires `bboxes` key. Fix pending.

---

## 2026-04-14

### ADDED — `utils/debug_yoloe_dino.py` (YOLOe proposals + DINOv2 scoring, 3-panel)

New debug script adding DINOv2 proposal scoring on top of the confirmed YOLOe visual-prompt pipeline.

**3-panel visualization per source group:**
- Panel 1: `refer_image` with reference bboxes (cyan)
- Panel 2: YOLOe proposals — color coded by YOLOe confidence (green>=0.15 / yellow>=0.05 / red<0.05)
- Panel 3: Same proposals — color coded by DINOv2 cosine similarity vs query crops (green>=0.50 / yellow>=0.35 / red<0.35)

**DINOv2 scoring method:**
- Embed all query crops once → pool of per-crop embeddings
- For each YOLOe proposal: crop from target image → DINOv2 embed → max cosine sim against all query crop embeddings
- Per-crop max (not averaged prototype) — avoids prototype dilution finding from 2026-04-13

**VRAM sequence (8GB rule):**
1. YOLOe: load → all proposals → `del + empty_cache()`
2. DINOv2: load → embed query crops + proposal crops → `del + empty_cache()`

Saves figures to `output_results/` (no matplotlib `show()`). Allows side-by-side reading of YOLOe confidence vs DINOv2 similarity to calibrate the DINOv2 threshold.

---

## 2026-04-13

### FINDING — YOLOe visual-prompt detection WORKS on large uniform-color object class (cls2)

Tested `debug_yoloe.py` on an industrial dataset class 2 (large uniform-color region) with 50 query crops, conf>=0.06:
- **YOLOe consistently proposes correct regions** across multiple source images
- Proposals are well-localized on the black panel / car body area
- At conf>=0.15: fewer but accurate proposals. At conf>=0.06: more coverage, still correct.
- Works on the exact object class that SAM2, SAM3, and OWLv2 all failed on.

**YOLOe is confirmed as the proposal generator for the pipeline.**

### DROPPED — Source-image context filtering (DINOv2 source-sim)

The source-image similarity filter (embed source image + target image, threshold cosine sim) is **unnecessary** for YOLOe.

**Why dropped:**
- YOLOe's visual prompts (refer_image + bboxes) are sufficient — it doesn't need pre-filtering
- Source-sim filtering was adding complexity (extra DINOv2 load/unload cycle) without benefit
- YOLOe proposals are correct regardless of source-target scene similarity
- Was originally designed for OWLv2 which needed help — YOLOe doesn't

### DROPPED — `utils/test_yoloe_dinov2.py`

Deleted. Was over-engineered with source-sim filtering and DINOv2 scoring that obscured YOLOe's actual performance. Replaced by `debug_yoloe.py` for testing.

### ADDED — `utils/debug_yoloe.py` (pure YOLOe visual-prompt debug)

Minimal debug script — no DINOv2, no source filtering. For each source image:
- Resolves all class bboxes from YOLO label
- Runs YOLOe visual-prompt detection on target
- Shows side-by-side: source with ref bboxes | target with color-coded proposals

---

### ADDED — `utils/test_yoloe_dinov2.py` (YOLOe + DINOv2 few-shot detection) — DELETED

New test script replacing OWLv2 with YOLOe (Ultralytics YOLO11) for proposal generation.

**Why YOLOe over OWLv2:**
- YOLOe uses multi-scale feature pyramid (P3/P4/P5) — natively detects objects at all sizes
- OWLv2 is patch-based ViT — fundamentally cannot propose large bounding boxes
- YOLOe is ~10x faster (~20-50ms/image vs OWLv2 ~200-500ms)
- YOLOe visual prompt API: pass `refer_image` + `visual_prompts` (bboxes from source label) — cleaner than OWLv2's image-guided mode

**Architecture:**
1. Source-image context filter (DINOv2) — keep crops from similar scenes
2. Group crops by source image → run YOLOe once per unique source (all class bboxes as visual prompts)
3. DINOv2 score each proposal against query crop
4. Cross-source NMS

**VRAM sequence:** DINOv2 (filter) → del → YOLOe (proposals) → del → DINOv2 (scoring)

Model: `yoloe-11l-seg.pt` with `YOLOEVPSegPredictor`  
Requires: `pip install ultralytics`

---

### ADDED — Source-image context filtering (in both test scripts)

New feature: before sending query crops to the detector, compare each crop's raw source image to the target via DINOv2 cosine similarity. Only crops from visually similar scenes pass through.

**Why:** Sending diverse crops from very different scenes produces noisy proposals. Filtering by source-scene similarity keeps only relevant queries.

**Implementation:**
- Crop filename `<stem>_cls<id>_<idx>.png` → resolve `<stem>.{jpg,png,...}` in source-images dir
- DINOv2 embed source image + target image → cosine similarity
- Cache per source stem (avoid re-embedding same source for multiple crops)
- Default threshold: 0.75

---

### ADDED — Per-crop OWLv2 pipeline (in `test_owlv2_dinov2.py`)

Replaced batch-all-crops OWLv2 with per-crop-individually approach:
- Each crop runs OWLv2 independently → proposals scored against that specific crop (not averaged prototype)
- Removed `build_dino_prototype()` — averaging 25 embeddings diluted the fingerprint
- Cross-crop NMS merges overlapping detections

**Also:** OWLv2 target image always resized to 640px for proposal generation, crops taken from original resolution for DINOv2 scoring. Removed `--resize` flag (auto-resize is now built in).

---

### FINDING — OWLv2 fundamentally fails on large objects

Tested OWLv2 extensively on industrial dataset "large uniform-color region" (class 2) detection:
- **1280px target:** 244 proposals, best DINOv2 sim 0.699, 0 detections at threshold 0.6
- **1008px target:** same — best sim 0.699, only tiny boxes in corners
- **640px target:** worse — best sim 0.665, 4 detections but all tiny edge boxes
- **Per-crop pipeline:** 8 detections but ALL small boxes, none on the actual large object

**Root cause:** OWLv2 is a patch-based ViT. It treats the image as a grid of 16×16 tiles and matches tile-level textures. It physically cannot "see" or propose a bounding box around a large region that spans many tiles. Downscaling doesn't fix this — smaller image just means fewer tiles.

**Conclusion:** OWLv2 is unsuitable for large-object detection in industrial scenes. YOLOe (multi-scale feature pyramid) is the correct architecture for this.

---

### FINDING — DINOv2 source-image similarity ranges

On industrial dataset (class 2 — large uniform-color region), source-to-target DINOv2 cosine sims:
- All 25 crops: range 0.648–0.840
- Threshold 0.50: passes 25/25 (too loose)
- Threshold 0.75: passes 10/25 (good discrimination)
- Top sources: 0.84, 0.83, 0.83, 0.83, 0.80

Source-sim filtering works — same production line images cluster tightly. Threshold needs per-dataset tuning (0.75 is a good starting point for this domain).

---

### FINDING — Averaged DINOv2 prototype dilutes signal

When averaging 25 crop embeddings into a single prototype:
- Best proposal similarity: 0.699 (just under 0.7 threshold)
- Averaging diverse crops (different angles, lighting, occlusion levels) smears the fingerprint
- Per-crop scoring (each proposal vs individual crop embedding) gives better discrimination

**Decision:** Use per-crop scoring, not averaged prototypes, for the detection test scripts.

---

## 2026-04-12

### ADDED — `utils/extract_crops.py` FINALIZED

Extract diverse seed crops from YOLO-annotated datasets. Two-phase pipeline:
- Phase 1: scan images (stride sampling), crop annotated objects, perceptual-hash dedup
- Phase 2: DINOv2 embed all candidates → DBSCAN clustering → one medoid per cluster + farthest-point outlier sampling

**Features:**
- `--auto-tune` for DBSCAN eps via k-nearest-neighbors
- `--stride` + `--max-images` for temporal sampling
- `--min-hash-dist` for perceptual hash dedup (default 20)
- `--padding` for context around bbox
- `--max-output-crops` hard cap
- Temp dir on same filesystem for fast copy

Crop naming: `<image_stem>_cls<class_id>_<ann_idx>.png` — enables reverse resolution to source image.

---

## 2026-04-10

### DROPPED — OWLv2 as primary detector (initial attempt)

First attempt at OWLv2 (google/owlv2-base-patch16-ensemble) for image-guided few-shot detection.

**Reasons dropped from main pipeline:**
- ~7GB VRAM consumption for image-guided mode — kills dev loop
- Poor detection quality on industrial tools (factory domain shift from internet pretraining)
- Known upstream bug in `embed_image_query` (huggingface/transformers#39710) — picks background patches instead of object; manual workaround was fragile
- Text-prompt mode not applicable (we need purely visual, no text)
- Wrong tool for the job — open-vocabulary semantic model, not visual similarity matcher

**Note:** OWLv2 was later re-tested (2026-04-13) with per-crop pipeline + source filtering + multi-scale — still fails on large objects. See findings above.

**Replaced by:** SAM2 (auto mask generation) as region proposer + DINOv2 as scorer for main pipeline. YOLOe for test/exploration scripts.

---

### ADDED — `utils/auto_annotate.py` (planned)

Full few-shot auto-annotation pipeline (Stages 1–4):
- Stage 1: DINOv2 prototype bank via K-Means (all centroids kept, incl. outlier clusters)
- Stage 2: SAM2 auto mask generation (no prompts/text)
- Stage 3: DINOv2 cosine similarity scoring + NMS
- Stage 4: YOLO .txt output + matplotlib preview

VRAM strategy: models loaded/deleted sequentially (`del` + `torch.cuda.empty_cache()`).
Targets RTX 4060 Laptop 8GB — DINOv2 and SAM2 never in VRAM simultaneously.

---
