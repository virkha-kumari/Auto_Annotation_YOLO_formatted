# Few-Shot Visual Auto-Annotation for Object Detection

**Abstract.** Labelling detection datasets is the main cost of deploying object detectors in industrial settings. This project studies how far *purely visual* few-shot prompting (no text prompts, no training) can go in auto-annotating unlabelled images from *k* labelled instances per class. Two training-free pipelines built from frozen foundation models are compared: **(A)** YOLOe visual-prompt proposals → SAM2 masks → DINOv2 masked-patch verification → box fusion, and **(B)** SAM3 cross-image exemplar prompting via canvas compositing → DINOv2 verification → cross-exemplar voting. Both output YOLO-format labels and run on a single consumer GPU (8 GB VRAM) by loading one model at a time. Evaluation is on the Construction-PPE dataset (12 classes) with *k* ∈ {15, 20, 25, 30, 35, 40, 45, 50} labelled instances per class on a fixed held-out split.

---

## 1. Problem and research questions

Given *k* labelled boxes per class and a set of unlabelled images from the same domain, produce boxes + class ids for the unlabelled images, with no model training and no text prompts (industrial classes often have no useful name, so prompts must be visual).

- **RQ1** — Which training-free visual-prompt architecture gives the better annotation quality per labelled instance: detector-first (A) or segmenter-first (B)?
- **RQ2** — How does quality scale with *k*?
- **RQ3** — Which class properties (size, appearance variance, "absence" classes such as `no_helmet`) break visual few-shot prompting?

## 2. Contributions

1. **Two complete, training-free few-shot auto-annotation pipelines** that run under an 8 GB VRAM budget through strict sequential model loading.
2. **Cross-image few-shot with SAM3** by compositing reference and target on one canvas (after [FSS-SAM3](https://github.com/WongKinYiu/FSS-SAM3)), extended to multi-class, multi-exemplar prompting with DINOv2 verification and voting across exemplars.
3. **Masked-patch DINOv2 verification**: pooling only the patch tokens covered by a segmentation mask, with size-aware routing. Small classes use mean-token embeddings of the raw crop, and each class uses one embedding method for both references and proposals.
4. **A YOLOe batching scheme** (bake the visual-prompt embedding once per reference frame, then batch all targets): 2.91× faster with identical proposals.
5. **A documented record of negative results** (OWLv2, SAM2/SAM3 as unprompted proposers, averaged prototypes, full-crop CLS). See §6 and [docs/log.md](docs/log.md).

---

## 3. Method

Both pipelines load exactly one large model at a time (`del` + `torch.cuda.empty_cache()` between phases).

### Pipeline A — YOLOe → SAM2 → DINOv2 → WBF ([scripts/yoloe_sam2_dinov2_module.py](scripts/yoloe_sam2_dinov2_module.py))

```
Ref selection  phash dedup + DINOv2/DBSCAN diversity → k crops/class (scripts/extract_crops_labelled.py)
               each crop = exactly one labelled instance (crop name carries its label line index)
Phase 1 YOLOe  per reference frame: bake visual-prompt embedding (VPE) once → batch-predict all targets
Phase 2 SAM2   box-prompted masks for refs and proposals (normal-size classes only)
Phase 3 DINOv2 normal classes: mean of patch tokens covered by the SAM2 mask
               small classes: mean of all tokens on the raw bbox crop (no SAM2)
               score = max cosine similarity vs the class's reference bank → gate (--dino-thresh)
Phase 4 Fuse   combined = 0.3·yoloe_conf + 0.7·dino_sim → per-class WBF (score = cluster mean)
               → --result-thresh gate → containment filter → YOLO .txt
```

### Pipeline B — SAM3 canvas-composite → DINOv2 → voting ([scripts/sam3_dinov2_module.py](scripts/sam3_dinov2_module.py))

SAM3 supports image exemplars (box prompts), but only within the image being segmented; it has no "learn from image A, detect in image B" API. Each (reference, target) pair is therefore composited onto one 1008×1008 canvas (reference on top, target below), the reference box is remapped into canvas coordinates as a positive exemplar, and predictions are cropped back to the target half.

```
Phase 1 DINOv2 phash dedup + farthest-point sampling → k diverse refs/class (proto bank);
               first 5 = SAM3 exemplars (refs from the same frame share one canvas = one "ref group")
Phase 2 SAM3   box-prompt each proto ref on its own image → ref masks;
               canvas-composite every (ref group, target batch) → proposals + masks + scores
Phase 3 DINOv2 masked-patch (or small-class mean-token) embedding, max cosine sim vs proto bank
               combined = 0.2·sam3_score + 0.8·dino_sim → gate (--sam3-dino-thresh)
Phase 4 Vote   duplicates (IoU ≥ --dup-iou-thresh) merged; votes = #ref groups that found the object
               final = combined · votes / #ref groups → containment filter → YOLO .txt
```

### Shared design rules

- **Embedding consistency.** Within a class, references and proposals use the same embedding method. In Pipeline B, references SAM3 fails to mask are dropped from the bank rather than mixed in with a different embedding.
- **Small-object routing.** If the 90th-percentile normalised area of the *selected* reference boxes is below `--small-obj-thresh`, the class skips masking. A box that covers only 1–2 DINOv2 patches gives a high-variance masked embedding. `--no-mask-classes` forces this routing per class.
- **Patch alignment.** DINOv2 inputs are resized to 224×224 without center-cropping, so the mask maps exactly onto the 16×16 patch grid. Patches ≥ 50% covered by the mask are pooled.

---

## 4. Experimental protocol

| Item | Setting |
|---|---|
| Dataset | [Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/) (Ultralytics, 2026-11-03 release), 1,000 images, 12 classes: helmet, gloves, vest, boots, goggles, Person, no_helmet, no_goggle, no_gloves, no_boots, forklift, no_vest. Built with [scripts/prepare_ppe.py](scripts/prepare_ppe.py) |
| Labelled pool | `data/label/`: official train split, 800 images. References are drawn **only** from here |
| Test split | `data/test/`: official val split, 200 images. Never used as references |
| Shots | *k* ∈ {15, 20, 25, 30, 35, 40, 45, 50} labelled instances per class (capped by availability for rare classes, e.g. `no_boots`) |
| Reference selection | Same pool and same phash dedup (distance 4) for both pipelines. A: DBSCAN cluster medoids + farthest-point outliers; B: farthest-point sampling |
| Metrics | Per-class P/R/F1 @ IoU 0.50; AP@.50 (101-point); mAP@.50:.95 (COCO, 10 IoU thresholds). Classes with GT but no predictions count as AP = 0; classes with 0 GT are excluded. Predictions ranked by pipeline score (A: fused score, B: final voted score) |
| Timing | Per-phase wall-clock (`timing.json`, not cumulative) |
| Hardware | Development: RTX 4060 Laptop 8 GB. k-shot sweep: RTX A4000 16 GB |
| Script | [experiments/run_kshot.sh](experiments/run_kshot.sh) → [experiments/summarize_kshot.py](experiments/summarize_kshot.py) |

Thresholds were calibrated during development on Construction-PPE (history in [docs/log.md](docs/log.md)) and are held fixed across all *k*.

---

## 5. Results

Pending. To be filled from `runs/kshot/summary.md` after the 16 runs (2 pipelines × 8 values of *k*): mAP@.50, mAP@.50:.95, and per-phase time per run, plus per-class AP@.50.

---

## 6. Design history and negative results

Condensed; full chronology with dates and reasons in [docs/log.md](docs/log.md).

**Proposal generation: everything before YOLOe failed.**
- *SAM2 / SAM3 automatic mask generation*: no class conditioning, and large uniform-colour industrial objects have no boundary contrast. Zero useful proposals.
- *OWLv2 image-guided detection* (`google/owlv2-base-patch16-ensemble`): on a large object covering ~60% of the frame, every proposal was a small edge box at 640, 1008 and 1280 px (best DINOv2 similarity 0.699). The image-query embedding is built from individual patch embeddings, which did not produce proposals spanning large objects in any configuration tried.
- *YOLOe visual prompts* (`yoloe-11l-seg.pt`): a multi-scale feature pyramid with a `refer_image` + boxes API. Well-localised proposals at conf ≥ 0.06. Adopted.

**Verification embedding: three iterations.**
- *Full-crop CLS*: similarities 0.19–0.43; background dominates the token.
- *Masked-patch pooling* (SAM2 mask → patch tokens inside it): similarities 0.60–0.95 on matches. Adopted.
- *Small objects*: gloves (median area 0.7% of the frame) cover 1–2 patches, and 0/N proposals passed. Routed to mean-token embedding of the raw crop.

**Prototype bank.** Averaging reference embeddings into one prototype dropped the best-match similarity from ~0.70 to ~0.60. A per-instance bank with max similarity is kept, so an unusual pose only needs one matching reference.

**YOLOe speed.** (1) Label I/O inside the target loop caused a 4–6× slowdown; prompts are now pre-built once. (2) `predict(refer_image=…)` internally bakes the VPE via `set_classes()`, after which the model is a plain detector, so the VPE is baked once per reference frame and all targets are batched: **2.91× faster, identical proposals (242 vs 242)**. Different reference frames cannot share a batch because the VPE is global model state.

**Why a second pipeline.** Pipeline A's hard classes (gloves, boots, goggles) were not fixable by threshold tuning, and each iteration re-ran three heavy stages. SAM2 has no few-shot capability. SAM3 does (image exemplars), but only within one image, which the canvas composite works around.

**Pipeline B fixes along the way.** Boxes are computed per predicted mask, not on the merged canvas mask, which otherwise produced boxes spanning the whole image. bf16 required casting box prompts to the model dtype. The DINOv2 gate runs *before* containment so an oversized false box cannot suppress real ones. Reference boxes are padded 1% to stop fingertips and thin extremities being clipped.

---

## 7. Limitations

- **Single dataset.** Thresholds were calibrated on Construction-PPE; transfer to other domains is untested.
- **Score weights are empirical** (A: 0.3/0.7; B: 0.2/0.8) and not learned.
- **Canvas aspect ratio.** Pipeline B resizes reference and target to 1008×504 each regardless of aspect ratio, which distorts shapes and halves the target's vertical resolution. This likely hurts small objects.
- **Absence classes** (`no_*`) are structurally unsuited to exemplar matching: they are defined by a missing feature, so there is no coherent visual prototype to match.
- **Scaling.** Pipeline A's YOLOe phase costs `reference frames × ⌈targets / batch⌉` predict calls, so it grows with both the seed set and the target set. Pipeline B's SAM3 phase costs `ref groups × ⌈targets / batch⌉` forward passes. Both containment filters are O(n²) per class per image.

## 8. Future work

- Multi-label YOLOe: one pass per class with visual-prompt embeddings from all reference frames stacked, instead of one pass per reference frame — fewer passes, no cross-class suppression.
- Frozen YOLO11-L backbone with a few-shot head (ProtoNet-style or regularized linear) on ROI-pooled P3/P4/P5 features, trained on the *k* labelled crops, as a replacement for YOLOe's visual-prompt embedding — which reduces each class to one mean-pooled vector with no learned decision boundary.
- Negative exemplars (SAM3 supports positive/negative box labels) for look-alike and absence classes.
- Aspect-preserving (letterboxed) canvas composition for Pipeline B.
- Per-class thresholds and learned score weights from the labelled pool, using a validation split separate from the test split.
- Evaluation on additional domains, including industrial data.

---

## 9. Reproducing

### Install

```bash
# PyTorch CUDA 11.8 first
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

SAM3 (`facebook/sam3`) is a gated model: accept its licence on Hugging Face and log in (`hf auth login`). The other weights download on first use.

### Data layout

`data/label/` and `data/test/`: flat folders of images plus same-stem YOLO `.txt` labels. Construction-PPE:

```bash
curl -L -o construction-ppe.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
unzip -q construction-ppe.zip -d ppe_raw && python scripts/prepare_ppe.py --src ppe_raw --out data
```

### k-shot sweep (both pipelines)

```bash
bash experiments/run_kshot.sh          # writes runs/kshot/{A,B}_k{15,20,...,50}/ + summary.md
# overrides via env, e.g.:  KS="15 30" B_ARGS="--no-preview --batch-size 8" bash experiments/run_kshot.sh
```

### Single runs

```bash
# Pipeline A: extract k seed crops, then run
python scripts/extract_crops_labelled.py --images data/label --labels data/label --classes 0 1 2 \
    --output crops --stride 1 --min-hash-dist 4 --max-per-class-crops 30
python scripts/yoloe_sam2_dinov2_module.py --queries-dirs crops/cls0 crops/cls1 crops/cls2 --class-ids 0 1 2 \
    --targets-dir data/test --source-images data/label --labels data/label --output-dir out_A

# Pipeline B: no crop extraction step
python scripts/sam3_dinov2_module.py --refs-dir data/label --refs-labels data/label --class-ids 0 1 2 \
    --targets-dir data/test --output-dir out_B --dino-proto-size 30

# Evaluate (Pipeline B: point --preds at out_B/labels)
python scripts/eval_map.py --preds out_A --gt data/test --images data/test \
    --classes helmet gloves vest --out-json out_A/metrics.json
```

### UI

`python app.py` opens a Gradio wizard at `http://127.0.0.1:7860`. The landing page picks Pipeline A or B. Page 1 extracts crops (A only), page 2 runs the pipeline with streamed logs, and page 3 shows the preview gallery and lets you download the YOLO labels.

### Parameters

**Pipeline A**

| Arg | Default | Notes |
|---|---|---|
| `--yoloe-conf` | 0.06 | Min YOLOe confidence |
| `--nms-iou` | 0.45 | YOLOe NMS IoU; also the WBF IoU |
| `--yoloe-iou` | 0.90 | Cross-ref proposal dedup per class before SAM2 (1 = off) |
| `--yoloe-batch-size` | 8 | Targets per predict call after VPE bake |
| `--dino-thresh` | 0.65 | Min DINOv2 max-similarity |
| `--dino-batch-size` | 0 | 0 = auto (~70% of GPU VRAM) |
| `--small-obj-thresh` | 0.01 | p90 normalised area of selected refs below this → no masking |
| `--no-mask-classes` | — | Force class ids into no-mask routing |
| `--sam2-mask-padding` | 0.05 | Ref box padding before prompting |
| `--sam-score-min` / `--sam-area-min` | 0.50 / 0.10 | Ref mask quality gates |
| `--wbf-score` | 0.10 | Min fused score |
| `--result-thresh` | 0.50 | Gate for saved boxes |
| `--containment-thresh` | 0.70 | Intersection / smaller-area for nested-box removal |
| `--no-preview` | off | Labels + summary.json + timing.json only |

**Pipeline B**

| Arg | Default | Notes |
|---|---|---|
| `--dino-proto-size` | 100 | *k*: refs per class in the DINOv2 bank |
| `--max-refs-per-class` | 5 | SAM3 exemplars (first N of the bank) |
| `--dinov2-batch-size` | 0 | 0 = auto (~70% of GPU VRAM) |
| `--threshold` / `--mask-threshold` | 0.6 / 0.6 | SAM3 score / mask binarisation |
| `--sam3-dino-thresh` | 0.2 | Gate on `0.2·sam3 + 0.8·dino_sim` |
| `--dup-iou-thresh` | 0.85 | IoU at which boxes are the same object (merged, voted) |
| `--containment-thresh` | 0.85 | Nested-box removal |
| `--small-obj-thresh` / `--no-mask-classes` | 0.01 / — | As Pipeline A |
| `--ref-box-padding` | 0.01 | Fractional ref box padding |
| `--phash-max-dist` | 4 | Ref near-duplicate dedup (0 = off) |
| `--batch-size` | 8 | Targets per SAM3 forward |
| `--fp32` | off | Default bf16 on CUDA |
| `--no-preview` | off | Skip 3-panel figures and mask overlays |

### Outputs

```
Pipeline A: out/<stem>.txt, out/<stem>_preview.jpg, out/summary.json, out/timing.json
Pipeline B: out/labels/<stem>.txt, out/<stem>.png (raw | after gate | final), out/summary.json,
            out/timing.json, out/temp_refs/cls<id>/ (selected refs, mask overlays)
summary.json boxes — A: {xyxy, score}; B: {box, sam3_score, dino_sim, combined_score, votes, final_score}
```

---

## 10. Repository layout

| Path | Role |
|---|---|
| `scripts/yoloe_sam2_dinov2_module.py` | Pipeline A |
| `scripts/sam3_dinov2_module.py` | Pipeline B |
| `scripts/extract_crops_labelled.py` | k-crop selection (phash + DINOv2/DBSCAN) for Pipeline A |
| `scripts/extract_crops_varied.py` | All crops, no clustering |
| `scripts/eval_map.py` | P/R/F1, AP@.50, mAP@.50:.95 |
| `app.py` | Gradio UI for both pipelines |
| `experiments/run_kshot.sh`, `summarize_kshot.py` | k-shot sweep and result tables |
| `experiments/debug_*.py`, `test_*.py` | Development probes: per-stage visual debugging, YOLOe batching benchmark, OWLv2 dead end |
| `docs/log.md` | Chronological ADDED / DROPPED / FINDING log |
| `docs/plans.md` | Open tasks |

## References

- DINOv2: https://github.com/facebookresearch/dinov2
- SAM2: https://github.com/facebookresearch/segment-anything-2
- SAM3: https://arxiv.org/abs/2511.16719
- YOLOe: https://docs.ultralytics.com/models/yoloe/
- FSS-SAM3 (canvas-composite few-shot): https://github.com/WongKinYiu/FSS-SAM3
- Weighted Boxes Fusion: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
- Construction-PPE dataset: https://docs.ultralytics.com/datasets/detect/construction-ppe

## Collaboration

Open to collaboration — if you have ideas on improving the proposal quality, the scoring stage, feel free to reach out or open an issue.

If you hit a bug, a dataset where the pipeline behaves unexpectedly, or have a suggestion to improve any part of the pipeline — don't hesitate to open an issue. Every edge case is useful signal.

Made with love for CV-curious engineers who want to understand what's actually happening inside these models, not just run them.
