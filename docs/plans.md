# Plans — What's Next

Forward-looking only. History/ablation/design → README.md + log.md.

---

## Next up

- [ ] **Pipeline B precision via the DINOv2 gate** — gate currently rejects nothing (lowest kept 0.27 vs 0.2). Hold out part of `data/label` as validation, sweep `--sam3-dino-thresh` / a `dino_sim`-only gate, report P/R/F1 vs threshold. See [ablation.md §6](ablation.md#6-implications-and-next-experiments).
- [ ] **SAM3 exemplar selection** — typical (medoid) exemplars instead of FPS-first outliers; exemplars drawn from only k labelled instances (realistic few-shot).
- [ ] **Hybrid per class** — B for object classes, A for forklift / no_boots / no_goggle.
- [ ] **Containment/dup filter scaling** — O(n²) per class per target in both pipelines. Fine at current box counts; revisit if proposal counts grow.

## Research directions

- [ ] **Multi-label YOLOe** — per class, stack VPEs from every ref frame into one `set_classes` → one YOLOe pass per class over all targets (11 passes vs one per ref frame), slot → class map, then `--yoloe-iou` dedup. Removes cross-class suppression of current per-stem `agnostic_nms`. Validate vs current proposals + mAP at k=15 before adopting.
- [ ] **Frozen YOLO11-L backbone + few-shot head** — ROI-pooled P3/P4/P5 features per proposal, ProtoNet-style / regularized linear head trained on the k labelled crops per class. Candidate replacement for YOLOe VPE (one mean-pooled vector per class, no learned boundary — ceiling for look-alikes: gloves vs bare hands, goggles vs glasses). Open question: which layers carry the per-class signal.
- [ ] SAM3 negative exemplars (pos/neg box labels) for look-alike and absence (`no_*`) classes.
- [ ] Aspect-preserving (letterboxed) canvas for Pipeline B — current 1008×504 halves stretch non-uniformly.
- [ ] Per-class thresholds / learned score weights on a validation split separate from `data/test/`.
- [ ] Second dataset to test threshold transfer.

---
