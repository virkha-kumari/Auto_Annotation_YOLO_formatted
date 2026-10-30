# Plans — What's Next

Forward-looking only. History/ablation/design → README.md + log.md.

---

## Next up

- [ ] **k-shot sweep on 24 GB server** — `bash experiments/run_kshot.sh`: Pipeline A + B × k ∈ {15, 30, 50, 100} on `data/test/`. Paste `runs/kshot/summary.md` into README §5.1.
- [ ] **Sample previews for README** — one rerun per pipeline without `--no-preview` at the best k; replace the preliminary images in `docs/images/`.
- [ ] **Containment/dup filter scaling** — O(n²) per class per target in both pipelines. Fine at current box counts; revisit if proposal counts grow.

## Research directions

- [ ] SAM3 negative exemplars (pos/neg box labels) for look-alike and absence (`no_*`) classes.
- [ ] Aspect-preserving (letterboxed) canvas for Pipeline B — current 1008×504 halves stretch non-uniformly.
- [ ] Per-class thresholds / learned score weights on a validation split separate from `data/test/`.
- [ ] Second dataset to test threshold transfer.

---
