# Auto-Annotation Pipeline — Notes

## Goal

Few-shot auto-annotation across multiple industrial datasets.
Label ~10–15 instances per class manually → propagate bboxes to unlabeled images.
Final output: YOLO-format `.txt` annotations + PIL preview for human review.

---

## Pipeline Architecture (FINALIZED — `scripts/auto_annotate.py`)

```
Annotated dataset + YOLO labels
        ↓
scripts/extract_crops_labelled.py  (or extract_crops_varied.py)
  phash dedup → DINOv2 embed → DBSCAN cluster → medoid + outlier sampling
        ↓
seed crops: <stem>_cls<id>_<idx>.jpg  (naming enables source reverse-resolution)

        ↓  Phase 1 — YOLOe (load once, unload)

  Outer loop = stems (union of all class source frames)
  Per stem:
    get_vpe(refer_image) → bakes VPE into model weights (once per stem)
    predict(source=[batch of targets], batch_size=8) → proposals on all targets
    stem_local_to_cls mapping restores correct class ID from local bbox index
  → proposals_per_target: {target_path: [(box, conf, cls_idx, stem), ...]}

        ↓  Phase 2 — SAM2 (load once, unload)

  Small classes (median bbox area < --small-obj-thresh):
    Skip SAM2 → make_bbox_crop(img_np, bbox) → raw PIL crop
  Normal classes:
    set_image(src_img) → predict(box=bbox) → make_masked_crop(img_np, mask, bbox)

  Job A (refs): source frames → labeled bboxes → crops → proto bank input
  Job B (proposals): per target → all proposal bboxes batched in one SAM2 call

        ↓  Phase 3 — DINOv2 (load once, unload)

  Small classes (use_cls=True):  embed bbox crop → CLS token → L2-normalise
  Normal classes (use_cls=False): embed masked crop → masked patch pooling
    patch tokens [B, 256, 768] → mask resized to 16×16 → mean pool inside mask

  Proto bank per class: embed all ref crops → [N_refs, 768]
  Per target per class: embed proposal crops → [N_props, 768]
  sim_matrix = prop_embs @ bank.T → best_sim per proposal → filter by --dino-thresh

        ↓  Phase 4 — WBF + containment filter (no model)

  combined_score = 0.3 × yoloe_conf + 0.7 × dino_sim
  WBF: fuse overlapping proposals, weighted by combined_score + vote count
  Containment filter: for each pair, if intersection/min_area > thresh → drop lower score
  → YOLO .txt (all classes in one file) + preview .jpg + summary.json
```

---

## Ablation Log

### Proposal generation

| Approach | Verdict | Root cause of failure |
|---|---|---|
| SAM2 auto mask generation | ❌ Dead end | No proposals on large uniform regions. SAM2 needs texture contrast. |
| SAM3 auto mask generation (unprompted) | ❌ Dead end | Same as SAM2. |
| SAM3 canvas-composite few-shot (`test/debug_sam3.py`) | ✅ **Confirmed working end-to-end** — next: DINOv2 sanity-check score, then promotion | No native cross-image exemplar API, so ref+target composited onto one canvas, ref bbox remapped to canvas coords, box-only exemplar prompt, prediction cropped back to target. Supports: SAM3 native box+score visualization alongside mask-derived tight bbox, batched targets-per-ref forward passes (`--batch-size`), multi-class (`--class-ids`, explicit or auto-discover "all"), flat output naming, bf16 inference, DINOv2 diverse-ref selection. Confirmed on real 5-class dataset. **Next:** add DINOv2 cosine-sim sanity check per SAM3 proposal vs ref crop (same scoring pattern as YOLOe pipeline), then real accuracy pass, then promotion into `scripts/auto_annotate.py`. See `docs/log.md` 2026-07-10. |
| OWLv2 image-guided detection | ❌ Dead end | Patch-based ViT — tile-level texture matching. Cannot compose bbox larger than one tile. Tested at 640/1008/1280px. |
| YOLOe visual-prompt detection | ✅ Confirmed | Multi-scale FPN detects at all scales. Works at conf≥0.06. |

### DINOv2 embedding

| Approach | Sim range | Verdict |
|---|---|---|
| Full-crop CLS token | 0.19–0.43 | Background dominates. Same object looks different depending on framing. |
| Masked patch pooling (SAM2) | 0.60–0.95 | Foreground-only. Current method for normal-size objects. |
| CLS on raw bbox crop (small objs) | TBD | 1–2 patches covered by tiny obj → patch pooling is noise. CLS on bbox crop = object IS the crop. |

### YOLOe call structure

| Structure | Speed | Issue |
|---|---|---|
| outer=targets, inner=stems (old) | 40–70s/target | `resolve_class_bboxes_padded` inside target loop = `N_targets × N_stems` label reads |
| outer=targets, inner=stems (fixed I/O) | 9–26s/target | Pre-built stem_prompts. Still `N_targets × N_stems` calls total. |
| outer=stems, inner=target batches (current) | ~3s/target equivalent | `get_vpe` once per stem → batch predict. 2.91× speedup, identical proposals. |

Batching different refer_images is impossible — VPE is baked per-call into model weights. Only valid batch dimension = targets sharing the same stem.

### Source-image context filtering

Designed for OWLv2 (compare source scene to target via DINOv2 sim, filter noisy crops). Unnecessary for YOLOe — proposals are well-localized regardless of scene similarity. Dropped entirely.

---

## Open Questions

- [ ] **DINOv2 threshold calibration** — `--dino-thresh` 0.65 confirmed for helmet/vest. Needs tuning per dataset per class. Check `X/N passed dino-thresh` in logs.
- [ ] **WBF score weights** — `0.3×yoloe + 0.7×dino` chosen empirically. Not validated.
- [ ] **Small-object threshold** — `--small-obj-thresh` 0.02 auto-detects gloves (0.007) and helmets (0.012) as small. May be too aggressive for datasets where helmets are large.
- [ ] **mAP evaluation** — pipeline output vs Construction-PPE ground truth not yet run.
- [ ] **Scale to 5K targets** — ~3s/target × 162 stems × 625 chunks = needs profiling at scale.
- [ ] **Prototype bank size effect** — 200+ ref crops per class vs 20 — does more help or hurt?
- [ ] **SAM3 proposal sanity check** — add DINOv2 cosine-sim scoring per SAM3 canvas-composite proposal (vs ref crop), same pattern as YOLOe→SAM2→DINOv2. Needed before `test/debug_sam3.py` can be promoted into `scripts/auto_annotate.py`.

---

## Datasets in Scope

| Domain | Example classes | Notes |
|---|---|---|
| Assembly line tools | torque_gun_out/in, roller, metal_plate, hands, 30+ classes | Primary use case |
| Torque tool inspection | torque tools, wrenches, fittings correct/wrong, 50+ classes | High class count |
| Electronics inspection | board, screw, screw_holder, tape, case | SAM2 tested on cls0 (0.6–0.98 scores) |
| Plumbing assembly | screwdrivers, torque, connectors, hands | |
| Sheet cutting | sheets, hands, tips | |
| Large uniform object | black_region (single class) | Primary YOLOe test case — large obj on uniform bg |
| Construction-PPE | helmet, gloves, vest, boots, goggles + no-wear variants | Demo/eval dataset, 1,416 images, YOLO native format |

---

## Demo Dataset — Construction-PPE

Ultralytics Construction-PPE: 1,416 images, 11 classes.  
Download: `https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip`  
License: AGPL-3.0

Split for pipeline eval:
- `valid/` (143 images) → seed crops (human-annotated)
- `train/` (1,132 images) → unlabeled targets
- Ground truth exists for `train/` → mAP eval possible

---

## Links

- DINOv2: https://github.com/facebookresearch/dinov2
- SAM2: https://github.com/facebookresearch/segment-anything-2
- YOLOe: https://docs.ultralytics.com/models/yoloe/
- Construction-PPE dataset: https://docs.ultralytics.com/datasets/detect/construction-ppe
- OWLv2 (dead end): https://huggingface.co/google/owlv2-base-patch16-ensemble
