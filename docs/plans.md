# Plans — What's Next

Forward-looking only. History/ablation/design → README.md + log.md.

---

## Next up

- [ ] **README mAP table stale — re-run Pipeline B eval** — measured under an old mean-sim/CLS-on-raw-crop/0.3-thresh config, before masked-patch DINOv2 scoring + max-sim + 0.2 thresh (calibrated, confirmed 2026-07-20) existed. Re-run `scripts/eval_map.py` on `data/test/`, update README.
- [ ] **Pipeline A re-run on `data/test/`** — needed for apples-to-apples mAP vs Pipeline B (currently different splits, see README).
- [ ] **DINOv2 threshold calibration per class (Pipeline A)** — `--dino-thresh` 0.65 confirmed helmet/vest only. Check `X/N passed dino-thresh` in logs per class.
- [ ] **WBF score weights** — `0.3×yoloe + 0.7×dino` empirical, not validated.
- [ ] **Scale to 5K+ targets** — needs profiling; current scaling bottleneck is YOLOe stem count (README "Scalability ceiling").
- [ ] **Prototype bank size** — 200+ ref crops/class vs 20 — unclear if more helps or hurts.
- [ ] **Containment/dup filter scaling** — both pipelines' filter (`filter_contained_boxes` in Pipeline A, `filter_containment_duplicates` in Pipeline B) is O(n²) per class per target. Fine at current box counts, undocumented as a scaling concern. Not urgent for Pipeline B right now — Pipeline A produces the larger proposal counts (no upstream box-count limiter like SAM3's exemplar filtering) — revisit if either pipeline's proposal counts grow much further.

## Research direction (longer-term, see README for full reasoning)

- [ ] Frozen YOLO backbone + few-shot head (ProtoNet-style) on ROI-pooled features — candidate replacement for YOLOe VPE's accuracy ceiling.

---
