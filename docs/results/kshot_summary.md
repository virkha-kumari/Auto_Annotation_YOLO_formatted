### mAP vs k (labelled instances per class)

| Pipeline | k | mAP@.50 | mAP@.5:.95 | nPred | nGT | classes in mAP | total time | s/target |
|---|---|---|---|---|---|---|---|---|
| A | 15 | 0.517 | 0.351 | 2573 | 2961 | 12 | 33m51s | 10.2 |
| A | 20 | 0.527 | 0.355 | 2674 | 2961 | 12 | 43m17s | 13.0 |
| A | 25 | 0.529 | 0.354 | 2768 | 2961 | 12 | 49m35s | 14.9 |
| A | 30 | 0.535 | 0.357 | 2860 | 2961 | 12 | 56m08s | 16.8 |
| A | 35 | 0.542 | 0.358 | 2957 | 2961 | 12 | 63m16s | 19.0 |
| A | 40 | 0.532 | 0.352 | 3161 | 2961 | 12 | 69m42s | 20.9 |
| A | 45 | 0.531 | 0.349 | 3340 | 2961 | 12 | 74m48s | 22.4 |
| A | 50 | 0.532 | 0.350 | 3418 | 2961 | 12 | 80m08s | 24.0 |
| B | 15 | 0.582 | 0.381 | 5228 | 2961 | 12 | 64m58s | 19.5 |
| B | 20 | 0.585 | 0.382 | 5226 | 2961 | 12 | 65m15s | 19.6 |
| B | 25 | 0.586 | 0.382 | 5227 | 2961 | 12 | 65m16s | 19.6 |
| B | 30 | 0.587 | 0.383 | 5231 | 2961 | 12 | 65m58s | 19.8 |
| B | 35 | 0.587 | 0.382 | 5229 | 2961 | 12 | 65m30s | 19.6 |
| B | 40 | 0.587 | 0.383 | 5228 | 2961 | 12 | 66m01s | 19.8 |
| B | 45 | 0.587 | 0.383 | 5227 | 2961 | 12 | 65m55s | 19.8 |
| B | 50 | 0.586 | 0.382 | 5228 | 2961 | 12 | 66m08s | 19.8 |

### Per-phase time (not cumulative)

| Pipeline | k | phases |
|---|---|---|
| A | 15 | phase1_yoloe 27m38s, phase2_sam2 4m08s, phase3_dinov2 2m00s, phase4_wbf_save 0m03s |
| A | 20 | phase1_yoloe 36m18s, phase2_sam2 4m42s, phase3_dinov2 2m12s, phase4_wbf_save 0m03s |
| A | 25 | phase1_yoloe 41m52s, phase2_sam2 5m19s, phase3_dinov2 2m20s, phase4_wbf_save 0m03s |
| A | 30 | phase1_yoloe 47m53s, phase2_sam2 5m38s, phase3_dinov2 2m32s, phase4_wbf_save 0m03s |
| A | 35 | phase1_yoloe 54m36s, phase2_sam2 5m58s, phase3_dinov2 2m38s, phase4_wbf_save 0m03s |
| A | 40 | phase1_yoloe 59m47s, phase2_sam2 6m47s, phase3_dinov2 3m03s, phase4_wbf_save 0m03s |
| A | 45 | phase1_yoloe 65m00s, phase2_sam2 6m36s, phase3_dinov2 3m07s, phase4_wbf_save 0m03s |
| A | 50 | phase1_yoloe 69m51s, phase2_sam2 6m57s, phase3_dinov2 3m15s, phase4_wbf_save 0m03s |
| B | 15 | phase1_ref_select 8m30s, phase2_sam3 53m58s, phase3_dinov2 2m28s, phase4_vote_save 0m00s |
| B | 20 | phase1_ref_select 8m30s, phase2_sam3 54m14s, phase3_dinov2 2m29s, phase4_vote_save 0m00s |
| B | 25 | phase1_ref_select 8m31s, phase2_sam3 54m15s, phase3_dinov2 2m29s, phase4_vote_save 0m00s |
| B | 30 | phase1_ref_select 8m34s, phase2_sam3 54m51s, phase3_dinov2 2m32s, phase4_vote_save 0m00s |
| B | 35 | phase1_ref_select 8m29s, phase2_sam3 54m27s, phase3_dinov2 2m32s, phase4_vote_save 0m00s |
| B | 40 | phase1_ref_select 8m41s, phase2_sam3 54m45s, phase3_dinov2 2m34s, phase4_vote_save 0m00s |
| B | 45 | phase1_ref_select 8m30s, phase2_sam3 54m49s, phase3_dinov2 2m34s, phase4_vote_save 0m00s |
| B | 50 | phase1_ref_select 8m37s, phase2_sam3 54m55s, phase3_dinov2 2m35s, phase4_vote_save 0m00s |

### Per-class AP@.50

| Class | A k=15 | A k=20 | A k=25 | A k=30 | A k=35 | A k=40 | A k=45 | A k=50 | B k=15 | B k=20 | B k=25 | B k=30 | B k=35 | B k=40 | B k=45 | B k=50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| helmet | 0.742 | 0.762 | 0.771 | 0.771 | 0.771 | 0.771 | 0.781 | 0.781 | 0.780 | 0.780 | 0.779 | 0.780 | 0.779 | 0.779 | 0.779 | 0.779 |
| gloves | 0.603 | 0.614 | 0.623 | 0.633 | 0.643 | 0.653 | 0.662 | 0.662 | 0.818 | 0.817 | 0.818 | 0.820 | 0.818 | 0.817 | 0.817 | 0.817 |
| vest | 0.590 | 0.593 | 0.594 | 0.585 | 0.564 | 0.489 | 0.489 | 0.491 | 0.697 | 0.702 | 0.705 | 0.703 | 0.704 | 0.705 | 0.705 | 0.704 |
| boots | 0.497 | 0.494 | 0.508 | 0.502 | 0.523 | 0.531 | 0.528 | 0.536 | 0.651 | 0.648 | 0.649 | 0.647 | 0.629 | 0.637 | 0.638 | 0.635 |
| goggles | 0.910 | 0.919 | 0.914 | 0.914 | 0.914 | 0.922 | 0.900 | 0.912 | 0.920 | 0.921 | 0.921 | 0.921 | 0.921 | 0.919 | 0.920 | 0.920 |
| Person | 0.743 | 0.752 | 0.762 | 0.762 | 0.772 | 0.772 | 0.772 | 0.772 | 0.919 | 0.919 | 0.919 | 0.919 | 0.919 | 0.919 | 0.919 | 0.919 |
| no_helmet | 0.295 | 0.315 | 0.341 | 0.350 | 0.388 | 0.294 | 0.309 | 0.319 | 0.397 | 0.400 | 0.413 | 0.419 | 0.418 | 0.416 | 0.417 | 0.404 |
| no_goggle | 0.177 | 0.200 | 0.128 | 0.104 | 0.079 | 0.076 | 0.078 | 0.071 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| no_gloves | 0.242 | 0.266 | 0.288 | 0.305 | 0.302 | 0.302 | 0.310 | 0.320 | 0.462 | 0.486 | 0.486 | 0.484 | 0.487 | 0.481 | 0.479 | 0.476 |
| no_boots | 0.456 | 0.456 | 0.466 | 0.478 | 0.481 | 0.482 | 0.486 | 0.487 | 0.328 | 0.327 | 0.326 | 0.326 | 0.329 | 0.329 | 0.329 | 0.330 |
| forklift | 0.705 | 0.694 | 0.698 | 0.754 | 0.795 | 0.797 | 0.771 | 0.770 | 0.656 | 0.646 | 0.650 | 0.659 | 0.666 | 0.671 | 0.672 | 0.673 |
| no_vest | 0.247 | 0.257 | 0.250 | 0.266 | 0.271 | 0.291 | 0.284 | 0.262 | 0.363 | 0.368 | 0.368 | 0.370 | 0.372 | 0.371 | 0.370 | 0.371 |
