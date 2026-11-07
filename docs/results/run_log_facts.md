# Run log facts

Extracted from the per-run logs (`runs/kshot/<run>.log`, not tracked): reference counts, size routing, proposal dedup, DINOv2 gate pass counts, SAM3 exemplars and ref masking, auto batch size.

## A_k15

```
[crops] cls0: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[crops] cls1: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[crops] cls2: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[crops] cls3: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[crops] cls4: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[crops] cls5: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[crops] cls6: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[crops] cls7: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[crops] cls8: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[crops] cls9: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[crops] cls10: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[crops] cls11: 15 crop(s) → 15 ref instance(s) from 15 source frame(s)
[cls0] p90 bbox area=0.02160 → SAM2/masked-patch
[cls1] p90 bbox area=0.01471 → SAM2/masked-patch
[cls2] p90 bbox area=0.13291 → SAM2/masked-patch
[cls3] p90 bbox area=0.02361 → SAM2/masked-patch
[cls4] p90 bbox area=0.01422 → SAM2/masked-patch
[cls5] p90 bbox area=0.39696 → SAM2/masked-patch
[cls6] p90 bbox area=0.03341 → SAM2/masked-patch
[cls7] p90 bbox area=0.00232 → bbox-crop/mean-pool
[cls8] p90 bbox area=0.01855 → SAM2/masked-patch
[cls9] p90 bbox area=0.01879 → SAM2/masked-patch
[cls10] p90 bbox area=0.74148 → SAM2/masked-patch
[cls11] p90 bbox area=0.06445 → SAM2/masked-patch
[phase1] 35776 proposal(s) across 200 target(s)
[phase1] cross-stem dedup (IoU >= 0.9): 35776 → 8009 proposal(s)
[phase2a] cls0 [sam2-masked-patch]: 15 ref(s) (0 skipped)
[phase2a] cls1 [sam2-masked-patch]: 15 ref(s) (0 skipped)
[phase2a] cls2 [sam2-masked-patch]: 15 ref(s) (0 skipped)
[phase2a] cls3 [sam2-masked-patch]: 15 ref(s) (0 skipped)
[phase2a] cls4 [sam2-masked-patch]: 15 ref(s) (0 skipped)
[phase2a] cls5 [sam2-masked-patch]: 15 ref(s) (0 skipped)
[phase2a] cls6 [sam2-masked-patch]: 15 ref(s) (0 skipped)
[phase2a] cls7 [bbox-crop/mean-pool]: 15 ref(s) (0 skipped)
[phase2a] cls8 [sam2-masked-patch]: 15 ref(s) (0 skipped)
[phase2a] cls9 [sam2-masked-patch]: 15 ref(s) (0 skipped)
[phase2a] cls10 [sam2-masked-patch]: 15 ref(s) (0 skipped)
[phase2a] cls11 [sam2-masked-patch]: 15 ref(s) (0 skipped)
[dino] auto batch size 944 (~11.1 MiB/sample, 70% of 15.6 GiB)
[phase3] cls0: 243/304 passed dino-thresh
[phase3] cls1: 247/366 passed dino-thresh
[phase3] cls2: 543/761 passed dino-thresh
[phase3] cls3: 444/641 passed dino-thresh
[phase3] cls4: 238/432 passed dino-thresh
[phase3] cls5: 373/461 passed dino-thresh
[phase3] cls6: 167/910 passed dino-thresh
[phase3] cls7: 301/1062 passed dino-thresh
[phase3] cls8: 220/710 passed dino-thresh
[phase3] cls9: 389/627 passed dino-thresh
[phase3] cls10: 243/941 passed dino-thresh
[phase3] cls11: 89/794 passed dino-thresh
```

## A_k20

```
[crops] cls0: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[crops] cls1: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[crops] cls2: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[crops] cls3: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[crops] cls4: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[crops] cls5: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[crops] cls6: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[crops] cls7: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[crops] cls8: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[crops] cls9: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[crops] cls10: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[crops] cls11: 20 crop(s) → 20 ref instance(s) from 20 source frame(s)
[cls0] p90 bbox area=0.02213 → SAM2/masked-patch
[cls1] p90 bbox area=0.01471 → SAM2/masked-patch
[cls2] p90 bbox area=0.13291 → SAM2/masked-patch
[cls3] p90 bbox area=0.03130 → SAM2/masked-patch
[cls4] p90 bbox area=0.01422 → SAM2/masked-patch
[cls5] p90 bbox area=0.39696 → SAM2/masked-patch
[cls6] p90 bbox area=0.03341 → SAM2/masked-patch
[cls7] p90 bbox area=0.00232 → bbox-crop/mean-pool
[cls8] p90 bbox area=0.01855 → SAM2/masked-patch
[cls9] p90 bbox area=0.01879 → SAM2/masked-patch
[cls10] p90 bbox area=0.74148 → SAM2/masked-patch
[cls11] p90 bbox area=0.06445 → SAM2/masked-patch
[phase1] 45853 proposal(s) across 200 target(s)
[phase1] cross-stem dedup (IoU >= 0.9): 45853 → 8835 proposal(s)
[phase2a] cls0 [sam2-masked-patch]: 20 ref(s) (0 skipped)
[phase2a] cls1 [sam2-masked-patch]: 20 ref(s) (0 skipped)
[phase2a] cls2 [sam2-masked-patch]: 20 ref(s) (0 skipped)
[phase2a] cls3 [sam2-masked-patch]: 20 ref(s) (0 skipped)
[phase2a] cls4 [sam2-masked-patch]: 20 ref(s) (0 skipped)
[phase2a] cls5 [sam2-masked-patch]: 20 ref(s) (0 skipped)
[phase2a] cls6 [sam2-masked-patch]: 20 ref(s) (0 skipped)
[phase2a] cls7 [bbox-crop/mean-pool]: 20 ref(s) (0 skipped)
[phase2a] cls8 [sam2-masked-patch]: 20 ref(s) (0 skipped)
[phase2a] cls9 [sam2-masked-patch]: 20 ref(s) (0 skipped)
[phase2a] cls10 [sam2-masked-patch]: 20 ref(s) (0 skipped)
[phase2a] cls11 [sam2-masked-patch]: 20 ref(s) (0 skipped)
[dino] auto batch size 944 (~11.1 MiB/sample, 70% of 15.6 GiB)
[phase3] cls0: 255/354 passed dino-thresh
[phase3] cls1: 275/442 passed dino-thresh
[phase3] cls2: 586/807 passed dino-thresh
[phase3] cls3: 452/656 passed dino-thresh
[phase3] cls4: 260/527 passed dino-thresh
[phase3] cls5: 383/464 passed dino-thresh
[phase3] cls6: 188/992 passed dino-thresh
[phase3] cls7: 356/1135 passed dino-thresh
[phase3] cls8: 269/809 passed dino-thresh
[phase3] cls9: 399/858 passed dino-thresh
[phase3] cls10: 268/997 passed dino-thresh
[phase3] cls11: 99/794 passed dino-thresh
```

## A_k25

```
[crops] cls0: 25 crop(s) → 25 ref instance(s) from 25 source frame(s)
[crops] cls1: 25 crop(s) → 25 ref instance(s) from 25 source frame(s)
[crops] cls2: 25 crop(s) → 25 ref instance(s) from 25 source frame(s)
[crops] cls3: 25 crop(s) → 25 ref instance(s) from 25 source frame(s)
[crops] cls4: 25 crop(s) → 25 ref instance(s) from 25 source frame(s)
[crops] cls5: 25 crop(s) → 25 ref instance(s) from 25 source frame(s)
[crops] cls6: 25 crop(s) → 25 ref instance(s) from 25 source frame(s)
[crops] cls7: 25 crop(s) → 25 ref instance(s) from 25 source frame(s)
[crops] cls8: 25 crop(s) → 25 ref instance(s) from 24 source frame(s)
[crops] cls9: 25 crop(s) → 25 ref instance(s) from 25 source frame(s)
[crops] cls10: 25 crop(s) → 25 ref instance(s) from 22 source frame(s)
[crops] cls11: 25 crop(s) → 25 ref instance(s) from 24 source frame(s)
[cls0] p90 bbox area=0.02160 → SAM2/masked-patch
[cls1] p90 bbox area=0.01471 → SAM2/masked-patch
[cls2] p90 bbox area=0.11398 → SAM2/masked-patch
[cls3] p90 bbox area=0.02561 → SAM2/masked-patch
[cls4] p90 bbox area=0.01113 → SAM2/masked-patch
[cls5] p90 bbox area=0.40839 → SAM2/masked-patch
[cls6] p90 bbox area=0.02564 → SAM2/masked-patch
[cls7] p90 bbox area=0.00232 → bbox-crop/mean-pool
[cls8] p90 bbox area=0.01250 → SAM2/masked-patch
[cls9] p90 bbox area=0.01796 → SAM2/masked-patch
[cls10] p90 bbox area=0.43050 → SAM2/masked-patch
[cls11] p90 bbox area=0.07958 → SAM2/masked-patch
[phase1] 53411 proposal(s) across 200 target(s)
[phase1] cross-stem dedup (IoU >= 0.9): 53411 → 9553 proposal(s)
[phase2a] cls0 [sam2-masked-patch]: 25 ref(s) (0 skipped)
[phase2a] cls1 [sam2-masked-patch]: 25 ref(s) (0 skipped)
[phase2a] cls2 [sam2-masked-patch]: 25 ref(s) (0 skipped)
[phase2a] cls3 [sam2-masked-patch]: 25 ref(s) (0 skipped)
[phase2a] cls4 [sam2-masked-patch]: 25 ref(s) (0 skipped)
[phase2a] cls5 [sam2-masked-patch]: 25 ref(s) (0 skipped)
[phase2a] cls6 [sam2-masked-patch]: 25 ref(s) (0 skipped)
[phase2a] cls7 [bbox-crop/mean-pool]: 25 ref(s) (0 skipped)
[phase2a] cls8 [sam2-masked-patch]: 25 ref(s) (0 skipped)
[phase2a] cls9 [sam2-masked-patch]: 25 ref(s) (0 skipped)
[phase2a] cls10 [sam2-masked-patch]: 25 ref(s) (0 skipped)
[phase2a] cls11 [sam2-masked-patch]: 25 ref(s) (0 skipped)
[dino] auto batch size 944 (~11.1 MiB/sample, 70% of 15.6 GiB)
[phase3] cls0: 258/422 passed dino-thresh
[phase3] cls1: 292/491 passed dino-thresh
[phase3] cls2: 586/809 passed dino-thresh
[phase3] cls3: 461/682 passed dino-thresh
[phase3] cls4: 333/673 passed dino-thresh
[phase3] cls5: 391/473 passed dino-thresh
[phase3] cls6: 249/1175 passed dino-thresh
[phase3] cls7: 378/1198 passed dino-thresh
[phase3] cls8: 290/865 passed dino-thresh
[phase3] cls9: 410/894 passed dino-thresh
[phase3] cls10: 291/1050 passed dino-thresh
[phase3] cls11: 116/821 passed dino-thresh
```

## A_k30

```
[crops] cls0: 30 crop(s) → 30 ref instance(s) from 30 source frame(s)
[crops] cls1: 30 crop(s) → 30 ref instance(s) from 30 source frame(s)
[crops] cls2: 30 crop(s) → 30 ref instance(s) from 30 source frame(s)
[crops] cls3: 30 crop(s) → 30 ref instance(s) from 30 source frame(s)
[crops] cls4: 30 crop(s) → 30 ref instance(s) from 30 source frame(s)
[crops] cls5: 30 crop(s) → 30 ref instance(s) from 30 source frame(s)
[crops] cls6: 30 crop(s) → 30 ref instance(s) from 29 source frame(s)
[crops] cls7: 30 crop(s) → 30 ref instance(s) from 29 source frame(s)
[crops] cls8: 30 crop(s) → 30 ref instance(s) from 29 source frame(s)
[crops] cls9: 30 crop(s) → 30 ref instance(s) from 28 source frame(s)
[crops] cls10: 30 crop(s) → 30 ref instance(s) from 26 source frame(s)
[crops] cls11: 30 crop(s) → 30 ref instance(s) from 28 source frame(s)
[cls0] p90 bbox area=0.02160 → SAM2/masked-patch
[cls1] p90 bbox area=0.01471 → SAM2/masked-patch
[cls2] p90 bbox area=0.11398 → SAM2/masked-patch
[cls3] p90 bbox area=0.02561 → SAM2/masked-patch
[cls4] p90 bbox area=0.01113 → SAM2/masked-patch
[cls5] p90 bbox area=0.40839 → SAM2/masked-patch
[cls6] p90 bbox area=0.02986 → SAM2/masked-patch
[cls7] p90 bbox area=0.00235 → bbox-crop/mean-pool
[cls8] p90 bbox area=0.01250 → SAM2/masked-patch
[cls9] p90 bbox area=0.01796 → SAM2/masked-patch
[cls10] p90 bbox area=0.58154 → SAM2/masked-patch
[cls11] p90 bbox area=0.07958 → SAM2/masked-patch
[phase1] 61054 proposal(s) across 200 target(s)
[phase1] cross-stem dedup (IoU >= 0.9): 61054 → 10244 proposal(s)
[phase2a] cls0 [sam2-masked-patch]: 30 ref(s) (0 skipped)
[phase2a] cls1 [sam2-masked-patch]: 30 ref(s) (0 skipped)
[phase2a] cls2 [sam2-masked-patch]: 30 ref(s) (0 skipped)
[phase2a] cls3 [sam2-masked-patch]: 30 ref(s) (0 skipped)
[phase2a] cls4 [sam2-masked-patch]: 30 ref(s) (0 skipped)
[phase2a] cls5 [sam2-masked-patch]: 30 ref(s) (0 skipped)
[phase2a] cls6 [sam2-masked-patch]: 30 ref(s) (0 skipped)
[phase2a] cls7 [bbox-crop/mean-pool]: 30 ref(s) (0 skipped)
[phase2a] cls8 [sam2-masked-patch]: 30 ref(s) (0 skipped)
[phase2a] cls9 [sam2-masked-patch]: 30 ref(s) (0 skipped)
[phase2a] cls10 [sam2-masked-patch]: 30 ref(s) (0 skipped)
[phase2a] cls11 [sam2-masked-patch]: 30 ref(s) (0 skipped)
[dino] auto batch size 944 (~11.1 MiB/sample, 70% of 15.6 GiB)
[phase3] cls0: 260/456 passed dino-thresh
[phase3] cls1: 295/514 passed dino-thresh
[phase3] cls2: 600/838 passed dino-thresh
[phase3] cls3: 540/825 passed dino-thresh
[phase3] cls4: 348/698 passed dino-thresh
[phase3] cls5: 392/475 passed dino-thresh
[phase3] cls6: 287/1288 passed dino-thresh
[phase3] cls7: 451/1224 passed dino-thresh
[phase3] cls8: 321/979 passed dino-thresh
[phase3] cls9: 419/915 passed dino-thresh
[phase3] cls10: 303/1109 passed dino-thresh
[phase3] cls11: 152/923 passed dino-thresh
```

## A_k35

```
[crops] cls0: 35 crop(s) → 35 ref instance(s) from 34 source frame(s)
[crops] cls1: 35 crop(s) → 35 ref instance(s) from 35 source frame(s)
[crops] cls2: 35 crop(s) → 35 ref instance(s) from 35 source frame(s)
[crops] cls3: 35 crop(s) → 35 ref instance(s) from 35 source frame(s)
[crops] cls4: 35 crop(s) → 35 ref instance(s) from 35 source frame(s)
[crops] cls5: 35 crop(s) → 35 ref instance(s) from 34 source frame(s)
[crops] cls6: 35 crop(s) → 35 ref instance(s) from 33 source frame(s)
[crops] cls7: 35 crop(s) → 35 ref instance(s) from 34 source frame(s)
[crops] cls8: 35 crop(s) → 35 ref instance(s) from 34 source frame(s)
[crops] cls9: 35 crop(s) → 35 ref instance(s) from 32 source frame(s)
[crops] cls10: 35 crop(s) → 35 ref instance(s) from 30 source frame(s)
[crops] cls11: 35 crop(s) → 35 ref instance(s) from 32 source frame(s)
[cls0] p90 bbox area=0.01839 → SAM2/masked-patch
[cls1] p90 bbox area=0.01328 → SAM2/masked-patch
[cls2] p90 bbox area=0.11080 → SAM2/masked-patch
[cls3] p90 bbox area=0.02361 → SAM2/masked-patch
[cls4] p90 bbox area=0.01113 → SAM2/masked-patch
[cls5] p90 bbox area=0.39696 → SAM2/masked-patch
[cls6] p90 bbox area=0.02722 → SAM2/masked-patch
[cls7] p90 bbox area=0.00232 → bbox-crop/mean-pool
[cls8] p90 bbox area=0.01086 → SAM2/masked-patch
[cls9] p90 bbox area=0.01288 → SAM2/masked-patch
[cls10] p90 bbox area=0.58154 → SAM2/masked-patch
[cls11] p90 bbox area=0.07958 → SAM2/masked-patch
[phase1] 70082 proposal(s) across 200 target(s)
[phase1] cross-stem dedup (IoU >= 0.9): 70082 → 10630 proposal(s)
[phase2a] cls0 [sam2-masked-patch]: 35 ref(s) (0 skipped)
[phase2a] cls1 [sam2-masked-patch]: 35 ref(s) (0 skipped)
[phase2a] cls2 [sam2-masked-patch]: 35 ref(s) (0 skipped)
[phase2a] cls3 [sam2-masked-patch]: 35 ref(s) (0 skipped)
[phase2a] cls4 [sam2-masked-patch]: 35 ref(s) (0 skipped)
[phase2a] cls5 [sam2-masked-patch]: 35 ref(s) (0 skipped)
[phase2a] cls6 [sam2-masked-patch]: 35 ref(s) (0 skipped)
[phase2a] cls7 [bbox-crop/mean-pool]: 35 ref(s) (0 skipped)
[phase2a] cls8 [sam2-masked-patch]: 35 ref(s) (0 skipped)
[phase2a] cls9 [sam2-masked-patch]: 35 ref(s) (0 skipped)
[phase2a] cls10 [sam2-masked-patch]: 35 ref(s) (0 skipped)
[phase2a] cls11 [sam2-masked-patch]: 35 ref(s) (0 skipped)
[dino] auto batch size 944 (~11.1 MiB/sample, 70% of 15.6 GiB)
[phase3] cls0: 264/496 passed dino-thresh
[phase3] cls1: 301/558 passed dino-thresh
[phase3] cls2: 660/933 passed dino-thresh
[phase3] cls3: 567/898 passed dino-thresh
[phase3] cls4: 364/723 passed dino-thresh
[phase3] cls5: 393/478 passed dino-thresh
[phase3] cls6: 323/1315 passed dino-thresh
[phase3] cls7: 521/1264 passed dino-thresh
[phase3] cls8: 343/1006 passed dino-thresh
[phase3] cls9: 430/765 passed dino-thresh
[phase3] cls10: 330/1224 passed dino-thresh
[phase3] cls11: 159/970 passed dino-thresh
```

## A_k40

```
[crops] cls0: 40 crop(s) → 40 ref instance(s) from 38 source frame(s)
[crops] cls1: 40 crop(s) → 40 ref instance(s) from 40 source frame(s)
[crops] cls2: 40 crop(s) → 40 ref instance(s) from 39 source frame(s)
[crops] cls3: 40 crop(s) → 40 ref instance(s) from 40 source frame(s)
[crops] cls4: 40 crop(s) → 40 ref instance(s) from 39 source frame(s)
[crops] cls5: 40 crop(s) → 40 ref instance(s) from 39 source frame(s)
[crops] cls6: 40 crop(s) → 40 ref instance(s) from 38 source frame(s)
[crops] cls7: 40 crop(s) → 40 ref instance(s) from 38 source frame(s)
[crops] cls8: 40 crop(s) → 40 ref instance(s) from 39 source frame(s)
[crops] cls9: 40 crop(s) → 40 ref instance(s) from 37 source frame(s)
[crops] cls10: 40 crop(s) → 40 ref instance(s) from 34 source frame(s)
[crops] cls11: 40 crop(s) → 40 ref instance(s) from 37 source frame(s)
[cls0] p90 bbox area=0.01839 → SAM2/masked-patch
[cls1] p90 bbox area=0.01328 → SAM2/masked-patch
[cls2] p90 bbox area=0.11080 → SAM2/masked-patch
[cls3] p90 bbox area=0.02361 → SAM2/masked-patch
[cls4] p90 bbox area=0.01113 → SAM2/masked-patch
[cls5] p90 bbox area=0.39696 → SAM2/masked-patch
[cls6] p90 bbox area=0.02722 → SAM2/masked-patch
[cls7] p90 bbox area=0.00232 → bbox-crop/mean-pool
[cls8] p90 bbox area=0.01250 → SAM2/masked-patch
[cls9] p90 bbox area=0.01288 → SAM2/masked-patch
[cls10] p90 bbox area=0.58154 → SAM2/masked-patch
[cls11] p90 bbox area=0.08835 → SAM2/masked-patch
[phase1] 80995 proposal(s) across 200 target(s)
[phase1] cross-stem dedup (IoU >= 0.9): 80995 → 12320 proposal(s)
[phase2a] cls0 [sam2-masked-patch]: 40 ref(s) (0 skipped)
[phase2a] cls1 [sam2-masked-patch]: 40 ref(s) (0 skipped)
[phase2a] cls2 [sam2-masked-patch]: 40 ref(s) (0 skipped)
[phase2a] cls3 [sam2-masked-patch]: 40 ref(s) (0 skipped)
[phase2a] cls4 [sam2-masked-patch]: 40 ref(s) (0 skipped)
[phase2a] cls5 [sam2-masked-patch]: 40 ref(s) (0 skipped)
[phase2a] cls6 [sam2-masked-patch]: 40 ref(s) (0 skipped)
[phase2a] cls7 [bbox-crop/mean-pool]: 40 ref(s) (0 skipped)
[phase2a] cls8 [sam2-masked-patch]: 40 ref(s) (0 skipped)
[phase2a] cls9 [sam2-masked-patch]: 40 ref(s) (0 skipped)
[phase2a] cls10 [sam2-masked-patch]: 40 ref(s) (0 skipped)
[phase2a] cls11 [sam2-masked-patch]: 40 ref(s) (0 skipped)
[dino] auto batch size 944 (~11.1 MiB/sample, 70% of 15.6 GiB)
[phase3] cls0: 265/679 passed dino-thresh
[phase3] cls1: 307/581 passed dino-thresh
[phase3] cls2: 709/1020 passed dino-thresh
[phase3] cls3: 592/991 passed dino-thresh
[phase3] cls4: 372/842 passed dino-thresh
[phase3] cls5: 398/484 passed dino-thresh
[phase3] cls6: 566/1522 passed dino-thresh
[phase3] cls7: 618/1631 passed dino-thresh
[phase3] cls8: 347/1112 passed dino-thresh
[phase3] cls9: 486/854 passed dino-thresh
[phase3] cls10: 360/1401 passed dino-thresh
[phase3] cls11: 206/1203 passed dino-thresh
```

## A_k45

```
[crops] cls0: 45 crop(s) → 45 ref instance(s) from 43 source frame(s)
[crops] cls1: 45 crop(s) → 45 ref instance(s) from 45 source frame(s)
[crops] cls2: 45 crop(s) → 45 ref instance(s) from 43 source frame(s)
[crops] cls3: 45 crop(s) → 45 ref instance(s) from 45 source frame(s)
[crops] cls4: 45 crop(s) → 45 ref instance(s) from 44 source frame(s)
[crops] cls5: 45 crop(s) → 45 ref instance(s) from 43 source frame(s)
[crops] cls6: 45 crop(s) → 45 ref instance(s) from 43 source frame(s)
[crops] cls7: 45 crop(s) → 45 ref instance(s) from 43 source frame(s)
[crops] cls8: 45 crop(s) → 45 ref instance(s) from 42 source frame(s)
[crops] cls9: 45 crop(s) → 45 ref instance(s) from 41 source frame(s)
[crops] cls10: 45 crop(s) → 45 ref instance(s) from 38 source frame(s)
[crops] cls11: 45 crop(s) → 45 ref instance(s) from 42 source frame(s)
[cls0] p90 bbox area=0.01768 → SAM2/masked-patch
[cls1] p90 bbox area=0.01092 → SAM2/masked-patch
[cls2] p90 bbox area=0.09995 → SAM2/masked-patch
[cls3] p90 bbox area=0.02195 → SAM2/masked-patch
[cls4] p90 bbox area=0.00903 → bbox-crop/mean-pool
[cls5] p90 bbox area=0.37714 → SAM2/masked-patch
[cls6] p90 bbox area=0.02564 → SAM2/masked-patch
[cls7] p90 bbox area=0.00202 → bbox-crop/mean-pool
[cls8] p90 bbox area=0.01086 → SAM2/masked-patch
[cls9] p90 bbox area=0.01272 → SAM2/masked-patch
[cls10] p90 bbox area=0.58154 → SAM2/masked-patch
[cls11] p90 bbox area=0.08835 → SAM2/masked-patch
[phase1] 89188 proposal(s) across 200 target(s)
[phase1] cross-stem dedup (IoU >= 0.9): 89188 → 12590 proposal(s)
[phase2a] cls0 [sam2-masked-patch]: 45 ref(s) (0 skipped)
[phase2a] cls1 [sam2-masked-patch]: 45 ref(s) (0 skipped)
[phase2a] cls2 [sam2-masked-patch]: 45 ref(s) (0 skipped)
[phase2a] cls3 [sam2-masked-patch]: 45 ref(s) (0 skipped)
[phase2a] cls4 [bbox-crop/mean-pool]: 45 ref(s) (0 skipped)
[phase2a] cls5 [sam2-masked-patch]: 45 ref(s) (0 skipped)
[phase2a] cls6 [sam2-masked-patch]: 45 ref(s) (0 skipped)
[phase2a] cls7 [bbox-crop/mean-pool]: 45 ref(s) (0 skipped)
[phase2a] cls8 [sam2-masked-patch]: 45 ref(s) (0 skipped)
[phase2a] cls9 [sam2-masked-patch]: 45 ref(s) (0 skipped)
[phase2a] cls10 [sam2-masked-patch]: 45 ref(s) (0 skipped)
[phase2a] cls11 [sam2-masked-patch]: 45 ref(s) (0 skipped)
[dino] auto batch size 944 (~11.1 MiB/sample, 70% of 15.6 GiB)
[phase3] cls0: 267/683 passed dino-thresh
[phase3] cls1: 328/654 passed dino-thresh
[phase3] cls2: 712/1044 passed dino-thresh
[phase3] cls3: 622/1026 passed dino-thresh
[phase3] cls4: 732/856 passed dino-thresh
[phase3] cls5: 396/482 passed dino-thresh
[phase3] cls6: 576/1530 passed dino-thresh
[phase3] cls7: 627/1638 passed dino-thresh
[phase3] cls8: 354/1113 passed dino-thresh
[phase3] cls9: 499/846 passed dino-thresh
[phase3] cls10: 377/1475 passed dino-thresh
[phase3] cls11: 238/1243 passed dino-thresh
```

## A_k50

```
[crops] cls0: 50 crop(s) → 50 ref instance(s) from 48 source frame(s)
[crops] cls1: 50 crop(s) → 50 ref instance(s) from 50 source frame(s)
[crops] cls2: 50 crop(s) → 50 ref instance(s) from 48 source frame(s)
[crops] cls3: 50 crop(s) → 50 ref instance(s) from 49 source frame(s)
[crops] cls4: 50 crop(s) → 50 ref instance(s) from 48 source frame(s)
[crops] cls5: 50 crop(s) → 50 ref instance(s) from 48 source frame(s)
[crops] cls6: 50 crop(s) → 50 ref instance(s) from 48 source frame(s)
[crops] cls7: 50 crop(s) → 50 ref instance(s) from 48 source frame(s)
[crops] cls8: 50 crop(s) → 50 ref instance(s) from 45 source frame(s)
[crops] cls9: 50 crop(s) → 50 ref instance(s) from 45 source frame(s)
[crops] cls10: 50 crop(s) → 50 ref instance(s) from 42 source frame(s)
[crops] cls11: 50 crop(s) → 50 ref instance(s) from 47 source frame(s)
[cls0] p90 bbox area=0.01768 → SAM2/masked-patch
[cls1] p90 bbox area=0.01092 → SAM2/masked-patch
[cls2] p90 bbox area=0.11080 → SAM2/masked-patch
[cls3] p90 bbox area=0.02195 → SAM2/masked-patch
[cls4] p90 bbox area=0.00903 → bbox-crop/mean-pool
[cls5] p90 bbox area=0.39696 → SAM2/masked-patch
[cls6] p90 bbox area=0.02564 → SAM2/masked-patch
[cls7] p90 bbox area=0.00202 → bbox-crop/mean-pool
[cls8] p90 bbox area=0.01086 → SAM2/masked-patch
[cls9] p90 bbox area=0.01272 → SAM2/masked-patch
[cls10] p90 bbox area=0.58154 → SAM2/masked-patch
[cls11] p90 bbox area=0.08835 → SAM2/masked-patch
[phase1] 95658 proposal(s) across 200 target(s)
[phase1] cross-stem dedup (IoU >= 0.9): 95658 → 13214 proposal(s)
[phase2a] cls0 [sam2-masked-patch]: 50 ref(s) (0 skipped)
[phase2a] cls1 [sam2-masked-patch]: 50 ref(s) (0 skipped)
[phase2a] cls2 [sam2-masked-patch]: 50 ref(s) (0 skipped)
[phase2a] cls3 [sam2-masked-patch]: 50 ref(s) (0 skipped)
[phase2a] cls4 [bbox-crop/mean-pool]: 50 ref(s) (0 skipped)
[phase2a] cls5 [sam2-masked-patch]: 50 ref(s) (0 skipped)
[phase2a] cls6 [sam2-masked-patch]: 50 ref(s) (0 skipped)
[phase2a] cls7 [bbox-crop/mean-pool]: 50 ref(s) (0 skipped)
[phase2a] cls8 [sam2-masked-patch]: 50 ref(s) (0 skipped)
[phase2a] cls9 [sam2-masked-patch]: 50 ref(s) (0 skipped)
[phase2a] cls10 [sam2-masked-patch]: 50 ref(s) (0 skipped)
[phase2a] cls11 [sam2-masked-patch]: 50 ref(s) (0 skipped)
[dino] auto batch size 944 (~11.1 MiB/sample, 70% of 15.6 GiB)
[phase3] cls0: 269/695 passed dino-thresh
[phase3] cls1: 336/919 passed dino-thresh
[phase3] cls2: 713/1060 passed dino-thresh
[phase3] cls3: 626/1047 passed dino-thresh
[phase3] cls4: 771/902 passed dino-thresh
[phase3] cls5: 396/484 passed dino-thresh
[phase3] cls6: 604/1565 passed dino-thresh
[phase3] cls7: 676/1752 passed dino-thresh
[phase3] cls8: 359/1110 passed dino-thresh
[phase3] cls9: 505/885 passed dino-thresh
[phase3] cls10: 389/1498 passed dino-thresh
[phase3] cls11: 340/1297 passed dino-thresh
```

## B_k15

```
[refs] class 0: phash dedup 1282 -> 1279
[refs] class 1: phash dedup 1548 -> 1546
[dino] auto batch size 952 (~11.1 MiB/sample, 70% of 15.6 GiB)
[refs] class 0: 1279 -> 15 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-019_inst1, source-04d90135ba9a6511d789035fa9021f31_inst0, source-596bdd3a39761003f3913916c3a97c55_inst13, source-c567c20479dfc3b1087b8cccd8b2cde1_inst26, ultralytics-ppe-nov2026-557_inst18, source-7c9d94a7835cf299bd5aeddb27ded889_inst13, source-29a54f6e5e66462bbcbec90273e92a51_inst7, source-fbb201fd66f289e3ad4e6f1a71c2a282_inst3, ultralytics-ppe-nov2026-001_inst3, source-94e6c9c9971c3352b61271256f014daf_inst6 … (+5)
[refs] class 0: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 1: 1546 -> 15 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-355_inst6, source-6cec9c3de8d7970fe1dfe824f18573c2_inst22, ultralytics-ppe-nov2026-659_inst6, ultralytics-ppe-nov2026-220_inst1, source-8abad491606bbd2343029bf42b48208e_inst6, source-61d89c2415a4e91f0aded67312754811_inst20, source-fd87396c628d9e0ce1ac0177cfbf27e3_inst18, ultralytics-ppe-nov2026-607_inst1, source-3195fd14699725afdd2906e7293cc3c6_inst4, ultralytics-ppe-nov2026-555_inst5 … (+5)
[refs] class 1: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 2: 1476 -> 15 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-195_inst8, ultralytics-ppe-nov2026-003_inst3, source-dc39994159adfb8e9c6bdafba4841c4b_inst27, source-544a422efd96b918ee6696091f2fe9b4_inst6, ultralytics-ppe-nov2026-479_inst10, source-b1b1595542d8a8121fc095a8223157a2_inst6, source-6b6114dea7ecbe5997f1f0393824508b_inst3, source-473c9faf0e4f5803ad615ecc681b2a2e_inst11, ultralytics-ppe-nov2026-590_inst1, source-a28a006708943ec327e46b51d2fd3da4_inst9 … (+5)
[refs] class 2: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 3: 2003 -> 15 diverse ref(s) for DINOv2 proto bank: source-f4532be7ff670b1a99ec14fdae4b0557_inst25, ultralytics-ppe-nov2026-051_inst4, source-ba3aa5d1715142e5e962c3376ceca1a6_inst40, source-0e2a4d4463396c949ccdf3a3253e1552_inst11, source-baa51c2ac69f7c70e92d77f740b0f120_inst33, source-3403f252ac25e929eef3aa6978ad32b4_inst19, source-40884ba3867e4fb7a2dc3a778aefc18d_inst8, source-ef9a9cc38caf4d732fa9b1ac736d0c43_inst13, source-ba3aa5d1715142e5e962c3376ceca1a6_inst35, source-04d90135ba9a6511d789035fa9021f31_inst14 … (+5)
[refs] class 3: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 4: 612 -> 15 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst18, ultralytics-ppe-nov2026-033_inst14, ultralytics-ppe-nov2026-431_inst3, ultralytics-ppe-nov2026-278_inst11, ultralytics-ppe-nov2026-040_inst16, ultralytics-ppe-nov2026-618_inst5, ultralytics-ppe-nov2026-435_inst11, ultralytics-ppe-nov2026-229_inst6, ultralytics-ppe-nov2026-245_inst6, source-5832d306e381498099548ed6af72c5e6_inst19 … (+5)
[refs] class 4: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 5: 1962 -> 15 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-139_inst8, source-1b1d16e594d2940d1ce4dcc46b896830_inst1, ultralytics-ppe-nov2026-642_inst8, ultralytics-ppe-nov2026-519_inst8, source-47de682b0dee4fa2a0733219b8dbc82c_inst11, ultralytics-ppe-nov2026-607_inst6, source-3cf6bc936c8bbba28f8a203937897ae5_inst2, ultralytics-ppe-nov2026-431_inst0, ultralytics-ppe-nov2026-517_inst9, source-c01dfb2dbae89f8e5825648d9da4e842_inst3 … (+5)
[refs] class 5: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 6: 634 -> 15 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-435_inst8, source-32b77e88696e2ba988310ec62da94f16_inst48, ultralytics-ppe-nov2026-681_inst2, source-ed4844326ba70151720a9335b90cff72_inst2, ultralytics-ppe-nov2026-034_inst32, ultralytics-ppe-nov2026-190_inst2, source-ba3aa5d1715142e5e962c3376ceca1a6_inst17, source-3cf6bc936c8bbba28f8a203937897ae5_inst14, ultralytics-ppe-nov2026-302_inst7, source-d7438871aefee49d413dcede6cd5d626_inst7 … (+5)
[refs] class 6: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 7: 265 -> 15 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst31, source-c2733b5e92b8e910dc4688e6fd3c51fc_inst4, ultralytics-ppe-nov2026-505_inst7, ultralytics-ppe-nov2026-011_inst18, ultralytics-ppe-nov2026-185_inst7, source-d5d4d730d8fbb824c881beaa18b34d80_inst31, ultralytics-ppe-nov2026-489_inst4, source-0e59077ef44d89df66fa8b340b15aca0_inst22, ultralytics-ppe-nov2026-511_inst7, ultralytics-ppe-nov2026-467_inst5 … (+5)
[refs] class 7: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 8: 839 -> 15 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-454_inst4, source-c567c20479dfc3b1087b8cccd8b2cde1_inst28, ultralytics-ppe-nov2026-192_inst7, source-46bf35b89356017177174a7dd0b35186_inst28, source-efa761a319a37ba07e58ccd0bc1a4003_inst25, ultralytics-ppe-nov2026-617_inst11, source-c1a6a639b24165512bbfc8b97c5e0fb3_inst23, source-ba3aa5d1715142e5e962c3376ceca1a6_inst23, ultralytics-ppe-nov2026-040_inst24, source-29a54f6e5e66462bbcbec90273e92a51_inst21 … (+5)
[refs] class 8: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 9: 569 -> 15 diverse ref(s) for DINOv2 proto bank: source-e497834ebd715d706366646daeb2269e_inst7, ultralytics-ppe-nov2026-467_inst6, source-ba3aa5d1715142e5e962c3376ceca1a6_inst37, source-eabef86b2cecf0e9ffe06787e0298a38_inst29, source-2edf4fdf4d031044f1130ef13ef129cc_inst22, source-5aef8f03a221e1d0f7dc1df4501dda5e_inst27, source-780e7e97f694e83abcc1b08020b98069_inst23, ultralytics-ppe-nov2026-267_inst6, source-dc39994159adfb8e9c6bdafba4841c4b_inst33, ultralytics-ppe-nov2026-068_inst6 … (+5)
[refs] class 9: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 10: 275 -> 15 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-216_inst15, ultralytics-ppe-nov2026-064_inst6, ultralytics-ppe-nov2026-512_inst6, source-263d7b7ac85694a4522bfcb8c63666cd_inst13, ultralytics-ppe-nov2026-484_inst6, source-305deca284bbf639bcdcc00671e240aa_inst6, ultralytics-ppe-nov2026-201_inst9, source-305deca284bbf639bcdcc00671e240aa_inst5, source-8ead761447ba33e51dc962d76ef2e676_inst21, ultralytics-ppe-nov2026-345_inst9 … (+5)
[refs] class 10: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 11: 437 -> 15 diverse ref(s) for DINOv2 proto bank: source-f2c0cc19bce353d225ec84760d453285_inst19, ultralytics-ppe-nov2026-065_inst8, ultralytics-ppe-nov2026-011_inst19, ultralytics-ppe-nov2026-529_inst11, ultralytics-ppe-nov2026-631_inst9, source-47de682b0dee4fa2a0733219b8dbc82c_inst18, source-3cf6bc936c8bbba28f8a203937897ae5_inst6, source-8cbee74c2adeab1f66f1ebc0e625f48e_inst13, source-b389d7e14d9086cef980ae814b3c2bb1_inst7, ultralytics-ppe-nov2026-706_inst13 … (+5)
[refs] class 11: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[small-obj] classes [1, 4, 7, 8, 9] below --small-obj-thresh=0.01 — SAM3 masking skipped, DINOv2 mean-pools the raw crop
[refs] cls0: 13 ref(s) masked, 2 SAM3 miss(es) — dropped from proto bank
[refs] cls2: 15 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls3: 14 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls5: 15 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls6: 11 ref(s) masked, 4 SAM3 miss(es) — dropped from proto bank
[refs] cls10: 15 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls11: 15 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[dino] auto batch size 896 (~11.1 MiB/sample, 70% of 15.6 GiB)
[dino] cls0 proto bank: (13, 768)
[dino] cls1 proto bank: (15, 768)
[dino] cls2 proto bank: (15, 768)
[dino] cls3 proto bank: (14, 768)
[dino] cls4 proto bank: (15, 768)
[dino] cls5 proto bank: (15, 768)
[dino] cls6 proto bank: (11, 768)
[dino] cls7 proto bank: (15, 768)
[dino] cls8 proto bank: (15, 768)
[dino] cls9 proto bank: (15, 768)
[dino] cls10 proto bank: (15, 768)
[dino] cls11 proto bank: (15, 768)
```

## B_k20

```
[refs] class 0: phash dedup 1282 -> 1279
[refs] class 1: phash dedup 1548 -> 1546
[dino] auto batch size 952 (~11.1 MiB/sample, 70% of 15.6 GiB)
[refs] class 0: 1279 -> 20 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-019_inst1, source-04d90135ba9a6511d789035fa9021f31_inst0, source-596bdd3a39761003f3913916c3a97c55_inst13, source-c567c20479dfc3b1087b8cccd8b2cde1_inst26, ultralytics-ppe-nov2026-557_inst18, source-7c9d94a7835cf299bd5aeddb27ded889_inst13, source-29a54f6e5e66462bbcbec90273e92a51_inst7, source-fbb201fd66f289e3ad4e6f1a71c2a282_inst3, ultralytics-ppe-nov2026-001_inst3, source-94e6c9c9971c3352b61271256f014daf_inst6 … (+10)
[refs] class 0: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 1: 1546 -> 20 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-355_inst6, source-6cec9c3de8d7970fe1dfe824f18573c2_inst22, ultralytics-ppe-nov2026-659_inst6, ultralytics-ppe-nov2026-220_inst1, source-8abad491606bbd2343029bf42b48208e_inst6, source-61d89c2415a4e91f0aded67312754811_inst20, source-fd87396c628d9e0ce1ac0177cfbf27e3_inst18, ultralytics-ppe-nov2026-607_inst1, source-3195fd14699725afdd2906e7293cc3c6_inst4, ultralytics-ppe-nov2026-555_inst5 … (+10)
[refs] class 1: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 2: 1476 -> 20 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-195_inst8, ultralytics-ppe-nov2026-003_inst3, source-dc39994159adfb8e9c6bdafba4841c4b_inst27, source-544a422efd96b918ee6696091f2fe9b4_inst6, ultralytics-ppe-nov2026-479_inst10, source-b1b1595542d8a8121fc095a8223157a2_inst6, source-6b6114dea7ecbe5997f1f0393824508b_inst3, source-473c9faf0e4f5803ad615ecc681b2a2e_inst11, ultralytics-ppe-nov2026-590_inst1, source-a28a006708943ec327e46b51d2fd3da4_inst9 … (+10)
[refs] class 2: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 3: 2003 -> 20 diverse ref(s) for DINOv2 proto bank: source-f4532be7ff670b1a99ec14fdae4b0557_inst25, ultralytics-ppe-nov2026-051_inst4, source-ba3aa5d1715142e5e962c3376ceca1a6_inst40, source-0e2a4d4463396c949ccdf3a3253e1552_inst11, source-baa51c2ac69f7c70e92d77f740b0f120_inst33, source-3403f252ac25e929eef3aa6978ad32b4_inst19, source-40884ba3867e4fb7a2dc3a778aefc18d_inst8, source-ef9a9cc38caf4d732fa9b1ac736d0c43_inst13, source-ba3aa5d1715142e5e962c3376ceca1a6_inst35, source-04d90135ba9a6511d789035fa9021f31_inst14 … (+10)
[refs] class 3: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 4: 612 -> 20 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst18, ultralytics-ppe-nov2026-033_inst14, ultralytics-ppe-nov2026-431_inst3, ultralytics-ppe-nov2026-278_inst11, ultralytics-ppe-nov2026-040_inst16, ultralytics-ppe-nov2026-618_inst5, ultralytics-ppe-nov2026-435_inst11, ultralytics-ppe-nov2026-229_inst6, ultralytics-ppe-nov2026-245_inst6, source-5832d306e381498099548ed6af72c5e6_inst19 … (+10)
[refs] class 4: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 5: 1962 -> 20 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-139_inst8, source-1b1d16e594d2940d1ce4dcc46b896830_inst1, ultralytics-ppe-nov2026-642_inst8, ultralytics-ppe-nov2026-519_inst8, source-47de682b0dee4fa2a0733219b8dbc82c_inst11, ultralytics-ppe-nov2026-607_inst6, source-3cf6bc936c8bbba28f8a203937897ae5_inst2, ultralytics-ppe-nov2026-431_inst0, ultralytics-ppe-nov2026-517_inst9, source-c01dfb2dbae89f8e5825648d9da4e842_inst3 … (+10)
[refs] class 5: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 6: 634 -> 20 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-435_inst8, source-32b77e88696e2ba988310ec62da94f16_inst48, ultralytics-ppe-nov2026-681_inst2, source-ed4844326ba70151720a9335b90cff72_inst2, ultralytics-ppe-nov2026-034_inst32, ultralytics-ppe-nov2026-190_inst2, source-ba3aa5d1715142e5e962c3376ceca1a6_inst17, source-3cf6bc936c8bbba28f8a203937897ae5_inst14, ultralytics-ppe-nov2026-302_inst7, source-d7438871aefee49d413dcede6cd5d626_inst7 … (+10)
[refs] class 6: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 7: 265 -> 20 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst31, source-c2733b5e92b8e910dc4688e6fd3c51fc_inst4, ultralytics-ppe-nov2026-505_inst7, ultralytics-ppe-nov2026-011_inst18, ultralytics-ppe-nov2026-185_inst7, source-d5d4d730d8fbb824c881beaa18b34d80_inst31, ultralytics-ppe-nov2026-489_inst4, source-0e59077ef44d89df66fa8b340b15aca0_inst22, ultralytics-ppe-nov2026-511_inst7, ultralytics-ppe-nov2026-467_inst5 … (+10)
[refs] class 7: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 8: 839 -> 20 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-454_inst4, source-c567c20479dfc3b1087b8cccd8b2cde1_inst28, ultralytics-ppe-nov2026-192_inst7, source-46bf35b89356017177174a7dd0b35186_inst28, source-efa761a319a37ba07e58ccd0bc1a4003_inst25, ultralytics-ppe-nov2026-617_inst11, source-c1a6a639b24165512bbfc8b97c5e0fb3_inst23, source-ba3aa5d1715142e5e962c3376ceca1a6_inst23, ultralytics-ppe-nov2026-040_inst24, source-29a54f6e5e66462bbcbec90273e92a51_inst21 … (+10)
[refs] class 8: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 9: 569 -> 20 diverse ref(s) for DINOv2 proto bank: source-e497834ebd715d706366646daeb2269e_inst7, ultralytics-ppe-nov2026-467_inst6, source-ba3aa5d1715142e5e962c3376ceca1a6_inst37, source-eabef86b2cecf0e9ffe06787e0298a38_inst29, source-2edf4fdf4d031044f1130ef13ef129cc_inst22, source-5aef8f03a221e1d0f7dc1df4501dda5e_inst27, source-780e7e97f694e83abcc1b08020b98069_inst23, ultralytics-ppe-nov2026-267_inst6, source-dc39994159adfb8e9c6bdafba4841c4b_inst33, ultralytics-ppe-nov2026-068_inst6 … (+10)
[refs] class 9: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 10: 275 -> 20 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-216_inst15, ultralytics-ppe-nov2026-064_inst6, ultralytics-ppe-nov2026-512_inst6, source-263d7b7ac85694a4522bfcb8c63666cd_inst13, ultralytics-ppe-nov2026-484_inst6, source-305deca284bbf639bcdcc00671e240aa_inst6, ultralytics-ppe-nov2026-201_inst9, source-305deca284bbf639bcdcc00671e240aa_inst5, source-8ead761447ba33e51dc962d76ef2e676_inst21, ultralytics-ppe-nov2026-345_inst9 … (+10)
[refs] class 10: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 11: 437 -> 20 diverse ref(s) for DINOv2 proto bank: source-f2c0cc19bce353d225ec84760d453285_inst19, ultralytics-ppe-nov2026-065_inst8, ultralytics-ppe-nov2026-011_inst19, ultralytics-ppe-nov2026-529_inst11, ultralytics-ppe-nov2026-631_inst9, source-47de682b0dee4fa2a0733219b8dbc82c_inst18, source-3cf6bc936c8bbba28f8a203937897ae5_inst6, source-8cbee74c2adeab1f66f1ebc0e625f48e_inst13, source-b389d7e14d9086cef980ae814b3c2bb1_inst7, ultralytics-ppe-nov2026-706_inst13 … (+10)
[refs] class 11: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[small-obj] classes [1, 4, 7, 9] below --small-obj-thresh=0.01 — SAM3 masking skipped, DINOv2 mean-pools the raw crop
[refs] cls0: 17 ref(s) masked, 3 SAM3 miss(es) — dropped from proto bank
[refs] cls2: 20 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls3: 19 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls5: 20 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls6: 15 ref(s) masked, 5 SAM3 miss(es) — dropped from proto bank
[refs] cls8: 19 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls10: 20 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls11: 20 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[dino] auto batch size 896 (~11.1 MiB/sample, 70% of 15.6 GiB)
[dino] cls0 proto bank: (17, 768)
[dino] cls1 proto bank: (20, 768)
[dino] cls2 proto bank: (20, 768)
[dino] cls3 proto bank: (19, 768)
[dino] cls4 proto bank: (20, 768)
[dino] cls5 proto bank: (20, 768)
[dino] cls6 proto bank: (15, 768)
[dino] cls7 proto bank: (20, 768)
[dino] cls8 proto bank: (19, 768)
[dino] cls9 proto bank: (20, 768)
[dino] cls10 proto bank: (20, 768)
[dino] cls11 proto bank: (20, 768)
```

## B_k25

```
[refs] class 0: phash dedup 1282 -> 1279
[refs] class 1: phash dedup 1548 -> 1546
[dino] auto batch size 952 (~11.1 MiB/sample, 70% of 15.6 GiB)
[refs] class 0: 1279 -> 25 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-019_inst1, source-04d90135ba9a6511d789035fa9021f31_inst0, source-596bdd3a39761003f3913916c3a97c55_inst13, source-c567c20479dfc3b1087b8cccd8b2cde1_inst26, ultralytics-ppe-nov2026-557_inst18, source-7c9d94a7835cf299bd5aeddb27ded889_inst13, source-29a54f6e5e66462bbcbec90273e92a51_inst7, source-fbb201fd66f289e3ad4e6f1a71c2a282_inst3, ultralytics-ppe-nov2026-001_inst3, source-94e6c9c9971c3352b61271256f014daf_inst6 … (+15)
[refs] class 0: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 1: 1546 -> 25 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-355_inst6, source-6cec9c3de8d7970fe1dfe824f18573c2_inst22, ultralytics-ppe-nov2026-659_inst6, ultralytics-ppe-nov2026-220_inst1, source-8abad491606bbd2343029bf42b48208e_inst6, source-61d89c2415a4e91f0aded67312754811_inst20, source-fd87396c628d9e0ce1ac0177cfbf27e3_inst18, ultralytics-ppe-nov2026-607_inst1, source-3195fd14699725afdd2906e7293cc3c6_inst4, ultralytics-ppe-nov2026-555_inst5 … (+15)
[refs] class 1: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 2: 1476 -> 25 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-195_inst8, ultralytics-ppe-nov2026-003_inst3, source-dc39994159adfb8e9c6bdafba4841c4b_inst27, source-544a422efd96b918ee6696091f2fe9b4_inst6, ultralytics-ppe-nov2026-479_inst10, source-b1b1595542d8a8121fc095a8223157a2_inst6, source-6b6114dea7ecbe5997f1f0393824508b_inst3, source-473c9faf0e4f5803ad615ecc681b2a2e_inst11, ultralytics-ppe-nov2026-590_inst1, source-a28a006708943ec327e46b51d2fd3da4_inst9 … (+15)
[refs] class 2: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 3: 2003 -> 25 diverse ref(s) for DINOv2 proto bank: source-f4532be7ff670b1a99ec14fdae4b0557_inst25, ultralytics-ppe-nov2026-051_inst4, source-ba3aa5d1715142e5e962c3376ceca1a6_inst40, source-0e2a4d4463396c949ccdf3a3253e1552_inst11, source-baa51c2ac69f7c70e92d77f740b0f120_inst33, source-3403f252ac25e929eef3aa6978ad32b4_inst19, source-40884ba3867e4fb7a2dc3a778aefc18d_inst8, source-ef9a9cc38caf4d732fa9b1ac736d0c43_inst13, source-ba3aa5d1715142e5e962c3376ceca1a6_inst35, source-04d90135ba9a6511d789035fa9021f31_inst14 … (+15)
[refs] class 3: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 4: 612 -> 25 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst18, ultralytics-ppe-nov2026-033_inst14, ultralytics-ppe-nov2026-431_inst3, ultralytics-ppe-nov2026-278_inst11, ultralytics-ppe-nov2026-040_inst16, ultralytics-ppe-nov2026-618_inst5, ultralytics-ppe-nov2026-435_inst11, ultralytics-ppe-nov2026-229_inst6, ultralytics-ppe-nov2026-245_inst6, source-5832d306e381498099548ed6af72c5e6_inst19 … (+15)
[refs] class 4: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 5: 1962 -> 25 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-139_inst8, source-1b1d16e594d2940d1ce4dcc46b896830_inst1, ultralytics-ppe-nov2026-642_inst8, ultralytics-ppe-nov2026-519_inst8, source-47de682b0dee4fa2a0733219b8dbc82c_inst11, ultralytics-ppe-nov2026-607_inst6, source-3cf6bc936c8bbba28f8a203937897ae5_inst2, ultralytics-ppe-nov2026-431_inst0, ultralytics-ppe-nov2026-517_inst9, source-c01dfb2dbae89f8e5825648d9da4e842_inst3 … (+15)
[refs] class 5: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 6: 634 -> 25 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-435_inst8, source-32b77e88696e2ba988310ec62da94f16_inst48, ultralytics-ppe-nov2026-681_inst2, source-ed4844326ba70151720a9335b90cff72_inst2, ultralytics-ppe-nov2026-034_inst32, ultralytics-ppe-nov2026-190_inst2, source-ba3aa5d1715142e5e962c3376ceca1a6_inst17, source-3cf6bc936c8bbba28f8a203937897ae5_inst14, ultralytics-ppe-nov2026-302_inst7, source-d7438871aefee49d413dcede6cd5d626_inst7 … (+15)
[refs] class 6: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 7: 265 -> 25 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst31, source-c2733b5e92b8e910dc4688e6fd3c51fc_inst4, ultralytics-ppe-nov2026-505_inst7, ultralytics-ppe-nov2026-011_inst18, ultralytics-ppe-nov2026-185_inst7, source-d5d4d730d8fbb824c881beaa18b34d80_inst31, ultralytics-ppe-nov2026-489_inst4, source-0e59077ef44d89df66fa8b340b15aca0_inst22, ultralytics-ppe-nov2026-511_inst7, ultralytics-ppe-nov2026-467_inst5 … (+15)
[refs] class 7: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 8: 839 -> 25 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-454_inst4, source-c567c20479dfc3b1087b8cccd8b2cde1_inst28, ultralytics-ppe-nov2026-192_inst7, source-46bf35b89356017177174a7dd0b35186_inst28, source-efa761a319a37ba07e58ccd0bc1a4003_inst25, ultralytics-ppe-nov2026-617_inst11, source-c1a6a639b24165512bbfc8b97c5e0fb3_inst23, source-ba3aa5d1715142e5e962c3376ceca1a6_inst23, ultralytics-ppe-nov2026-040_inst24, source-29a54f6e5e66462bbcbec90273e92a51_inst21 … (+15)
[refs] class 8: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 9: 569 -> 25 diverse ref(s) for DINOv2 proto bank: source-e497834ebd715d706366646daeb2269e_inst7, ultralytics-ppe-nov2026-467_inst6, source-ba3aa5d1715142e5e962c3376ceca1a6_inst37, source-eabef86b2cecf0e9ffe06787e0298a38_inst29, source-2edf4fdf4d031044f1130ef13ef129cc_inst22, source-5aef8f03a221e1d0f7dc1df4501dda5e_inst27, source-780e7e97f694e83abcc1b08020b98069_inst23, ultralytics-ppe-nov2026-267_inst6, source-dc39994159adfb8e9c6bdafba4841c4b_inst33, ultralytics-ppe-nov2026-068_inst6 … (+15)
[refs] class 9: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 10: 275 -> 25 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-216_inst15, ultralytics-ppe-nov2026-064_inst6, ultralytics-ppe-nov2026-512_inst6, source-263d7b7ac85694a4522bfcb8c63666cd_inst13, ultralytics-ppe-nov2026-484_inst6, source-305deca284bbf639bcdcc00671e240aa_inst6, ultralytics-ppe-nov2026-201_inst9, source-305deca284bbf639bcdcc00671e240aa_inst5, source-8ead761447ba33e51dc962d76ef2e676_inst21, ultralytics-ppe-nov2026-345_inst9 … (+15)
[refs] class 10: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 11: 437 -> 25 diverse ref(s) for DINOv2 proto bank: source-f2c0cc19bce353d225ec84760d453285_inst19, ultralytics-ppe-nov2026-065_inst8, ultralytics-ppe-nov2026-011_inst19, ultralytics-ppe-nov2026-529_inst11, ultralytics-ppe-nov2026-631_inst9, source-47de682b0dee4fa2a0733219b8dbc82c_inst18, source-3cf6bc936c8bbba28f8a203937897ae5_inst6, source-8cbee74c2adeab1f66f1ebc0e625f48e_inst13, source-b389d7e14d9086cef980ae814b3c2bb1_inst7, ultralytics-ppe-nov2026-706_inst13 … (+15)
[refs] class 11: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[small-obj] classes [0, 1, 4, 7, 9] below --small-obj-thresh=0.01 — SAM3 masking skipped, DINOv2 mean-pools the raw crop
[refs] cls2: 25 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls3: 24 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls5: 25 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls6: 19 ref(s) masked, 6 SAM3 miss(es) — dropped from proto bank
[refs] cls8: 23 ref(s) masked, 2 SAM3 miss(es) — dropped from proto bank
[refs] cls10: 24 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls11: 24 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[dino] auto batch size 896 (~11.1 MiB/sample, 70% of 15.6 GiB)
[dino] cls0 proto bank: (25, 768)
[dino] cls1 proto bank: (25, 768)
[dino] cls2 proto bank: (25, 768)
[dino] cls3 proto bank: (24, 768)
[dino] cls4 proto bank: (25, 768)
[dino] cls5 proto bank: (25, 768)
[dino] cls6 proto bank: (19, 768)
[dino] cls7 proto bank: (25, 768)
[dino] cls8 proto bank: (23, 768)
[dino] cls9 proto bank: (25, 768)
[dino] cls10 proto bank: (24, 768)
[dino] cls11 proto bank: (24, 768)
```

## B_k30

```
[refs] class 0: phash dedup 1282 -> 1279
[refs] class 1: phash dedup 1548 -> 1546
[dino] auto batch size 952 (~11.1 MiB/sample, 70% of 15.6 GiB)
[refs] class 0: 1279 -> 30 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-019_inst1, source-04d90135ba9a6511d789035fa9021f31_inst0, source-596bdd3a39761003f3913916c3a97c55_inst13, source-c567c20479dfc3b1087b8cccd8b2cde1_inst26, ultralytics-ppe-nov2026-557_inst18, source-7c9d94a7835cf299bd5aeddb27ded889_inst13, source-29a54f6e5e66462bbcbec90273e92a51_inst7, source-fbb201fd66f289e3ad4e6f1a71c2a282_inst3, ultralytics-ppe-nov2026-001_inst3, source-94e6c9c9971c3352b61271256f014daf_inst6 … (+20)
[refs] class 0: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 1: 1546 -> 30 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-355_inst6, source-6cec9c3de8d7970fe1dfe824f18573c2_inst22, ultralytics-ppe-nov2026-659_inst6, ultralytics-ppe-nov2026-220_inst1, source-8abad491606bbd2343029bf42b48208e_inst6, source-61d89c2415a4e91f0aded67312754811_inst20, source-fd87396c628d9e0ce1ac0177cfbf27e3_inst18, ultralytics-ppe-nov2026-607_inst1, source-3195fd14699725afdd2906e7293cc3c6_inst4, ultralytics-ppe-nov2026-555_inst5 … (+20)
[refs] class 1: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 2: 1476 -> 30 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-195_inst8, ultralytics-ppe-nov2026-003_inst3, source-dc39994159adfb8e9c6bdafba4841c4b_inst27, source-544a422efd96b918ee6696091f2fe9b4_inst6, ultralytics-ppe-nov2026-479_inst10, source-b1b1595542d8a8121fc095a8223157a2_inst6, source-6b6114dea7ecbe5997f1f0393824508b_inst3, source-473c9faf0e4f5803ad615ecc681b2a2e_inst11, ultralytics-ppe-nov2026-590_inst1, source-a28a006708943ec327e46b51d2fd3da4_inst9 … (+20)
[refs] class 2: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 3: 2003 -> 30 diverse ref(s) for DINOv2 proto bank: source-f4532be7ff670b1a99ec14fdae4b0557_inst25, ultralytics-ppe-nov2026-051_inst4, source-ba3aa5d1715142e5e962c3376ceca1a6_inst40, source-0e2a4d4463396c949ccdf3a3253e1552_inst11, source-baa51c2ac69f7c70e92d77f740b0f120_inst33, source-3403f252ac25e929eef3aa6978ad32b4_inst19, source-40884ba3867e4fb7a2dc3a778aefc18d_inst8, source-ef9a9cc38caf4d732fa9b1ac736d0c43_inst13, source-ba3aa5d1715142e5e962c3376ceca1a6_inst35, source-04d90135ba9a6511d789035fa9021f31_inst14 … (+20)
[refs] class 3: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 4: 612 -> 30 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst18, ultralytics-ppe-nov2026-033_inst14, ultralytics-ppe-nov2026-431_inst3, ultralytics-ppe-nov2026-278_inst11, ultralytics-ppe-nov2026-040_inst16, ultralytics-ppe-nov2026-618_inst5, ultralytics-ppe-nov2026-435_inst11, ultralytics-ppe-nov2026-229_inst6, ultralytics-ppe-nov2026-245_inst6, source-5832d306e381498099548ed6af72c5e6_inst19 … (+20)
[refs] class 4: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 5: 1962 -> 30 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-139_inst8, source-1b1d16e594d2940d1ce4dcc46b896830_inst1, ultralytics-ppe-nov2026-642_inst8, ultralytics-ppe-nov2026-519_inst8, source-47de682b0dee4fa2a0733219b8dbc82c_inst11, ultralytics-ppe-nov2026-607_inst6, source-3cf6bc936c8bbba28f8a203937897ae5_inst2, ultralytics-ppe-nov2026-431_inst0, ultralytics-ppe-nov2026-517_inst9, source-c01dfb2dbae89f8e5825648d9da4e842_inst3 … (+20)
[refs] class 5: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 6: 634 -> 30 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-435_inst8, source-32b77e88696e2ba988310ec62da94f16_inst48, ultralytics-ppe-nov2026-681_inst2, source-ed4844326ba70151720a9335b90cff72_inst2, ultralytics-ppe-nov2026-034_inst32, ultralytics-ppe-nov2026-190_inst2, source-ba3aa5d1715142e5e962c3376ceca1a6_inst17, source-3cf6bc936c8bbba28f8a203937897ae5_inst14, ultralytics-ppe-nov2026-302_inst7, source-d7438871aefee49d413dcede6cd5d626_inst7 … (+20)
[refs] class 6: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 7: 265 -> 30 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst31, source-c2733b5e92b8e910dc4688e6fd3c51fc_inst4, ultralytics-ppe-nov2026-505_inst7, ultralytics-ppe-nov2026-011_inst18, ultralytics-ppe-nov2026-185_inst7, source-d5d4d730d8fbb824c881beaa18b34d80_inst31, ultralytics-ppe-nov2026-489_inst4, source-0e59077ef44d89df66fa8b340b15aca0_inst22, ultralytics-ppe-nov2026-511_inst7, ultralytics-ppe-nov2026-467_inst5 … (+20)
[refs] class 7: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 8: 839 -> 30 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-454_inst4, source-c567c20479dfc3b1087b8cccd8b2cde1_inst28, ultralytics-ppe-nov2026-192_inst7, source-46bf35b89356017177174a7dd0b35186_inst28, source-efa761a319a37ba07e58ccd0bc1a4003_inst25, ultralytics-ppe-nov2026-617_inst11, source-c1a6a639b24165512bbfc8b97c5e0fb3_inst23, source-ba3aa5d1715142e5e962c3376ceca1a6_inst23, ultralytics-ppe-nov2026-040_inst24, source-29a54f6e5e66462bbcbec90273e92a51_inst21 … (+20)
[refs] class 8: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 9: 569 -> 30 diverse ref(s) for DINOv2 proto bank: source-e497834ebd715d706366646daeb2269e_inst7, ultralytics-ppe-nov2026-467_inst6, source-ba3aa5d1715142e5e962c3376ceca1a6_inst37, source-eabef86b2cecf0e9ffe06787e0298a38_inst29, source-2edf4fdf4d031044f1130ef13ef129cc_inst22, source-5aef8f03a221e1d0f7dc1df4501dda5e_inst27, source-780e7e97f694e83abcc1b08020b98069_inst23, ultralytics-ppe-nov2026-267_inst6, source-dc39994159adfb8e9c6bdafba4841c4b_inst33, ultralytics-ppe-nov2026-068_inst6 … (+20)
[refs] class 9: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 10: 275 -> 30 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-216_inst15, ultralytics-ppe-nov2026-064_inst6, ultralytics-ppe-nov2026-512_inst6, source-263d7b7ac85694a4522bfcb8c63666cd_inst13, ultralytics-ppe-nov2026-484_inst6, source-305deca284bbf639bcdcc00671e240aa_inst6, ultralytics-ppe-nov2026-201_inst9, source-305deca284bbf639bcdcc00671e240aa_inst5, source-8ead761447ba33e51dc962d76ef2e676_inst21, ultralytics-ppe-nov2026-345_inst9 … (+20)
[refs] class 10: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 11: 437 -> 30 diverse ref(s) for DINOv2 proto bank: source-f2c0cc19bce353d225ec84760d453285_inst19, ultralytics-ppe-nov2026-065_inst8, ultralytics-ppe-nov2026-011_inst19, ultralytics-ppe-nov2026-529_inst11, ultralytics-ppe-nov2026-631_inst9, source-47de682b0dee4fa2a0733219b8dbc82c_inst18, source-3cf6bc936c8bbba28f8a203937897ae5_inst6, source-8cbee74c2adeab1f66f1ebc0e625f48e_inst13, source-b389d7e14d9086cef980ae814b3c2bb1_inst7, ultralytics-ppe-nov2026-706_inst13 … (+20)
[refs] class 11: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[small-obj] classes [4, 7, 9] below --small-obj-thresh=0.01 — SAM3 masking skipped, DINOv2 mean-pools the raw crop
[refs] cls0: 27 ref(s) masked, 3 SAM3 miss(es) — dropped from proto bank
[refs] cls1: 29 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls2: 29 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls3: 29 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls5: 30 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls6: 21 ref(s) masked, 9 SAM3 miss(es) — dropped from proto bank
[refs] cls8: 27 ref(s) masked, 3 SAM3 miss(es) — dropped from proto bank
[refs] cls10: 29 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls11: 29 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[dino] auto batch size 896 (~11.1 MiB/sample, 70% of 15.6 GiB)
[dino] cls0 proto bank: (27, 768)
[dino] cls1 proto bank: (29, 768)
[dino] cls2 proto bank: (29, 768)
[dino] cls3 proto bank: (29, 768)
[dino] cls4 proto bank: (30, 768)
[dino] cls5 proto bank: (30, 768)
[dino] cls6 proto bank: (21, 768)
[dino] cls7 proto bank: (30, 768)
[dino] cls8 proto bank: (27, 768)
[dino] cls9 proto bank: (30, 768)
[dino] cls10 proto bank: (29, 768)
[dino] cls11 proto bank: (29, 768)
```

## B_k35

```
[refs] class 0: phash dedup 1282 -> 1279
[refs] class 1: phash dedup 1548 -> 1546
[dino] auto batch size 952 (~11.1 MiB/sample, 70% of 15.6 GiB)
[refs] class 0: 1279 -> 35 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-019_inst1, source-04d90135ba9a6511d789035fa9021f31_inst0, source-596bdd3a39761003f3913916c3a97c55_inst13, source-c567c20479dfc3b1087b8cccd8b2cde1_inst26, ultralytics-ppe-nov2026-557_inst18, source-7c9d94a7835cf299bd5aeddb27ded889_inst13, source-29a54f6e5e66462bbcbec90273e92a51_inst7, source-fbb201fd66f289e3ad4e6f1a71c2a282_inst3, ultralytics-ppe-nov2026-001_inst3, source-94e6c9c9971c3352b61271256f014daf_inst6 … (+25)
[refs] class 0: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 1: 1546 -> 35 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-355_inst6, source-6cec9c3de8d7970fe1dfe824f18573c2_inst22, ultralytics-ppe-nov2026-659_inst6, ultralytics-ppe-nov2026-220_inst1, source-8abad491606bbd2343029bf42b48208e_inst6, source-61d89c2415a4e91f0aded67312754811_inst20, source-fd87396c628d9e0ce1ac0177cfbf27e3_inst18, ultralytics-ppe-nov2026-607_inst1, source-3195fd14699725afdd2906e7293cc3c6_inst4, ultralytics-ppe-nov2026-555_inst5 … (+25)
[refs] class 1: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 2: 1476 -> 35 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-195_inst8, ultralytics-ppe-nov2026-003_inst3, source-dc39994159adfb8e9c6bdafba4841c4b_inst27, source-544a422efd96b918ee6696091f2fe9b4_inst6, ultralytics-ppe-nov2026-479_inst10, source-b1b1595542d8a8121fc095a8223157a2_inst6, source-6b6114dea7ecbe5997f1f0393824508b_inst3, source-473c9faf0e4f5803ad615ecc681b2a2e_inst11, ultralytics-ppe-nov2026-590_inst1, source-a28a006708943ec327e46b51d2fd3da4_inst9 … (+25)
[refs] class 2: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 3: 2003 -> 35 diverse ref(s) for DINOv2 proto bank: source-f4532be7ff670b1a99ec14fdae4b0557_inst25, ultralytics-ppe-nov2026-051_inst4, source-ba3aa5d1715142e5e962c3376ceca1a6_inst40, source-0e2a4d4463396c949ccdf3a3253e1552_inst11, source-baa51c2ac69f7c70e92d77f740b0f120_inst33, source-3403f252ac25e929eef3aa6978ad32b4_inst19, source-40884ba3867e4fb7a2dc3a778aefc18d_inst8, source-ef9a9cc38caf4d732fa9b1ac736d0c43_inst13, source-ba3aa5d1715142e5e962c3376ceca1a6_inst35, source-04d90135ba9a6511d789035fa9021f31_inst14 … (+25)
[refs] class 3: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 4: 612 -> 35 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst18, ultralytics-ppe-nov2026-033_inst14, ultralytics-ppe-nov2026-431_inst3, ultralytics-ppe-nov2026-278_inst11, ultralytics-ppe-nov2026-040_inst16, ultralytics-ppe-nov2026-618_inst5, ultralytics-ppe-nov2026-435_inst11, ultralytics-ppe-nov2026-229_inst6, ultralytics-ppe-nov2026-245_inst6, source-5832d306e381498099548ed6af72c5e6_inst19 … (+25)
[refs] class 4: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 5: 1962 -> 35 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-139_inst8, source-1b1d16e594d2940d1ce4dcc46b896830_inst1, ultralytics-ppe-nov2026-642_inst8, ultralytics-ppe-nov2026-519_inst8, source-47de682b0dee4fa2a0733219b8dbc82c_inst11, ultralytics-ppe-nov2026-607_inst6, source-3cf6bc936c8bbba28f8a203937897ae5_inst2, ultralytics-ppe-nov2026-431_inst0, ultralytics-ppe-nov2026-517_inst9, source-c01dfb2dbae89f8e5825648d9da4e842_inst3 … (+25)
[refs] class 5: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 6: 634 -> 35 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-435_inst8, source-32b77e88696e2ba988310ec62da94f16_inst48, ultralytics-ppe-nov2026-681_inst2, source-ed4844326ba70151720a9335b90cff72_inst2, ultralytics-ppe-nov2026-034_inst32, ultralytics-ppe-nov2026-190_inst2, source-ba3aa5d1715142e5e962c3376ceca1a6_inst17, source-3cf6bc936c8bbba28f8a203937897ae5_inst14, ultralytics-ppe-nov2026-302_inst7, source-d7438871aefee49d413dcede6cd5d626_inst7 … (+25)
[refs] class 6: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 7: 265 -> 35 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst31, source-c2733b5e92b8e910dc4688e6fd3c51fc_inst4, ultralytics-ppe-nov2026-505_inst7, ultralytics-ppe-nov2026-011_inst18, ultralytics-ppe-nov2026-185_inst7, source-d5d4d730d8fbb824c881beaa18b34d80_inst31, ultralytics-ppe-nov2026-489_inst4, source-0e59077ef44d89df66fa8b340b15aca0_inst22, ultralytics-ppe-nov2026-511_inst7, ultralytics-ppe-nov2026-467_inst5 … (+25)
[refs] class 7: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 8: 839 -> 35 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-454_inst4, source-c567c20479dfc3b1087b8cccd8b2cde1_inst28, ultralytics-ppe-nov2026-192_inst7, source-46bf35b89356017177174a7dd0b35186_inst28, source-efa761a319a37ba07e58ccd0bc1a4003_inst25, ultralytics-ppe-nov2026-617_inst11, source-c1a6a639b24165512bbfc8b97c5e0fb3_inst23, source-ba3aa5d1715142e5e962c3376ceca1a6_inst23, ultralytics-ppe-nov2026-040_inst24, source-29a54f6e5e66462bbcbec90273e92a51_inst21 … (+25)
[refs] class 8: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 9: 569 -> 35 diverse ref(s) for DINOv2 proto bank: source-e497834ebd715d706366646daeb2269e_inst7, ultralytics-ppe-nov2026-467_inst6, source-ba3aa5d1715142e5e962c3376ceca1a6_inst37, source-eabef86b2cecf0e9ffe06787e0298a38_inst29, source-2edf4fdf4d031044f1130ef13ef129cc_inst22, source-5aef8f03a221e1d0f7dc1df4501dda5e_inst27, source-780e7e97f694e83abcc1b08020b98069_inst23, ultralytics-ppe-nov2026-267_inst6, source-dc39994159adfb8e9c6bdafba4841c4b_inst33, ultralytics-ppe-nov2026-068_inst6 … (+25)
[refs] class 9: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 10: 275 -> 35 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-216_inst15, ultralytics-ppe-nov2026-064_inst6, ultralytics-ppe-nov2026-512_inst6, source-263d7b7ac85694a4522bfcb8c63666cd_inst13, ultralytics-ppe-nov2026-484_inst6, source-305deca284bbf639bcdcc00671e240aa_inst6, ultralytics-ppe-nov2026-201_inst9, source-305deca284bbf639bcdcc00671e240aa_inst5, source-8ead761447ba33e51dc962d76ef2e676_inst21, ultralytics-ppe-nov2026-345_inst9 … (+25)
[refs] class 10: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 11: 437 -> 35 diverse ref(s) for DINOv2 proto bank: source-f2c0cc19bce353d225ec84760d453285_inst19, ultralytics-ppe-nov2026-065_inst8, ultralytics-ppe-nov2026-011_inst19, ultralytics-ppe-nov2026-529_inst11, ultralytics-ppe-nov2026-631_inst9, source-47de682b0dee4fa2a0733219b8dbc82c_inst18, source-3cf6bc936c8bbba28f8a203937897ae5_inst6, source-8cbee74c2adeab1f66f1ebc0e625f48e_inst13, source-b389d7e14d9086cef980ae814b3c2bb1_inst7, ultralytics-ppe-nov2026-706_inst13 … (+25)
[refs] class 11: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[small-obj] classes [0, 1, 4, 7, 9] below --small-obj-thresh=0.01 — SAM3 masking skipped, DINOv2 mean-pools the raw crop
[refs] cls2: 34 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls3: 34 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls5: 35 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls6: 26 ref(s) masked, 9 SAM3 miss(es) — dropped from proto bank
[refs] cls8: 31 ref(s) masked, 4 SAM3 miss(es) — dropped from proto bank
[refs] cls10: 34 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls11: 34 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[dino] auto batch size 896 (~11.1 MiB/sample, 70% of 15.6 GiB)
[dino] cls0 proto bank: (35, 768)
[dino] cls1 proto bank: (35, 768)
[dino] cls2 proto bank: (34, 768)
[dino] cls3 proto bank: (34, 768)
[dino] cls4 proto bank: (35, 768)
[dino] cls5 proto bank: (35, 768)
[dino] cls6 proto bank: (26, 768)
[dino] cls7 proto bank: (35, 768)
[dino] cls8 proto bank: (31, 768)
[dino] cls9 proto bank: (35, 768)
[dino] cls10 proto bank: (34, 768)
[dino] cls11 proto bank: (34, 768)
```

## B_k40

```
[refs] class 0: phash dedup 1282 -> 1279
[refs] class 1: phash dedup 1548 -> 1546
[dino] auto batch size 952 (~11.1 MiB/sample, 70% of 15.6 GiB)
[refs] class 0: 1279 -> 40 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-019_inst1, source-04d90135ba9a6511d789035fa9021f31_inst0, source-596bdd3a39761003f3913916c3a97c55_inst13, source-c567c20479dfc3b1087b8cccd8b2cde1_inst26, ultralytics-ppe-nov2026-557_inst18, source-7c9d94a7835cf299bd5aeddb27ded889_inst13, source-29a54f6e5e66462bbcbec90273e92a51_inst7, source-fbb201fd66f289e3ad4e6f1a71c2a282_inst3, ultralytics-ppe-nov2026-001_inst3, source-94e6c9c9971c3352b61271256f014daf_inst6 … (+30)
[refs] class 0: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 1: 1546 -> 40 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-355_inst6, source-6cec9c3de8d7970fe1dfe824f18573c2_inst22, ultralytics-ppe-nov2026-659_inst6, ultralytics-ppe-nov2026-220_inst1, source-8abad491606bbd2343029bf42b48208e_inst6, source-61d89c2415a4e91f0aded67312754811_inst20, source-fd87396c628d9e0ce1ac0177cfbf27e3_inst18, ultralytics-ppe-nov2026-607_inst1, source-3195fd14699725afdd2906e7293cc3c6_inst4, ultralytics-ppe-nov2026-555_inst5 … (+30)
[refs] class 1: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 2: 1476 -> 40 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-195_inst8, ultralytics-ppe-nov2026-003_inst3, source-dc39994159adfb8e9c6bdafba4841c4b_inst27, source-544a422efd96b918ee6696091f2fe9b4_inst6, ultralytics-ppe-nov2026-479_inst10, source-b1b1595542d8a8121fc095a8223157a2_inst6, source-6b6114dea7ecbe5997f1f0393824508b_inst3, source-473c9faf0e4f5803ad615ecc681b2a2e_inst11, ultralytics-ppe-nov2026-590_inst1, source-a28a006708943ec327e46b51d2fd3da4_inst9 … (+30)
[refs] class 2: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 3: 2003 -> 40 diverse ref(s) for DINOv2 proto bank: source-f4532be7ff670b1a99ec14fdae4b0557_inst25, ultralytics-ppe-nov2026-051_inst4, source-ba3aa5d1715142e5e962c3376ceca1a6_inst40, source-0e2a4d4463396c949ccdf3a3253e1552_inst11, source-baa51c2ac69f7c70e92d77f740b0f120_inst33, source-3403f252ac25e929eef3aa6978ad32b4_inst19, source-40884ba3867e4fb7a2dc3a778aefc18d_inst8, source-ef9a9cc38caf4d732fa9b1ac736d0c43_inst13, source-ba3aa5d1715142e5e962c3376ceca1a6_inst35, source-04d90135ba9a6511d789035fa9021f31_inst14 … (+30)
[refs] class 3: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 4: 612 -> 40 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst18, ultralytics-ppe-nov2026-033_inst14, ultralytics-ppe-nov2026-431_inst3, ultralytics-ppe-nov2026-278_inst11, ultralytics-ppe-nov2026-040_inst16, ultralytics-ppe-nov2026-618_inst5, ultralytics-ppe-nov2026-435_inst11, ultralytics-ppe-nov2026-229_inst6, ultralytics-ppe-nov2026-245_inst6, source-5832d306e381498099548ed6af72c5e6_inst19 … (+30)
[refs] class 4: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 5: 1962 -> 40 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-139_inst8, source-1b1d16e594d2940d1ce4dcc46b896830_inst1, ultralytics-ppe-nov2026-642_inst8, ultralytics-ppe-nov2026-519_inst8, source-47de682b0dee4fa2a0733219b8dbc82c_inst11, ultralytics-ppe-nov2026-607_inst6, source-3cf6bc936c8bbba28f8a203937897ae5_inst2, ultralytics-ppe-nov2026-431_inst0, ultralytics-ppe-nov2026-517_inst9, source-c01dfb2dbae89f8e5825648d9da4e842_inst3 … (+30)
[refs] class 5: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 6: 634 -> 40 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-435_inst8, source-32b77e88696e2ba988310ec62da94f16_inst48, ultralytics-ppe-nov2026-681_inst2, source-ed4844326ba70151720a9335b90cff72_inst2, ultralytics-ppe-nov2026-034_inst32, ultralytics-ppe-nov2026-190_inst2, source-ba3aa5d1715142e5e962c3376ceca1a6_inst17, source-3cf6bc936c8bbba28f8a203937897ae5_inst14, ultralytics-ppe-nov2026-302_inst7, source-d7438871aefee49d413dcede6cd5d626_inst7 … (+30)
[refs] class 6: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 7: 265 -> 40 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst31, source-c2733b5e92b8e910dc4688e6fd3c51fc_inst4, ultralytics-ppe-nov2026-505_inst7, ultralytics-ppe-nov2026-011_inst18, ultralytics-ppe-nov2026-185_inst7, source-d5d4d730d8fbb824c881beaa18b34d80_inst31, ultralytics-ppe-nov2026-489_inst4, source-0e59077ef44d89df66fa8b340b15aca0_inst22, ultralytics-ppe-nov2026-511_inst7, ultralytics-ppe-nov2026-467_inst5 … (+30)
[refs] class 7: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 8: 839 -> 40 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-454_inst4, source-c567c20479dfc3b1087b8cccd8b2cde1_inst28, ultralytics-ppe-nov2026-192_inst7, source-46bf35b89356017177174a7dd0b35186_inst28, source-efa761a319a37ba07e58ccd0bc1a4003_inst25, ultralytics-ppe-nov2026-617_inst11, source-c1a6a639b24165512bbfc8b97c5e0fb3_inst23, source-ba3aa5d1715142e5e962c3376ceca1a6_inst23, ultralytics-ppe-nov2026-040_inst24, source-29a54f6e5e66462bbcbec90273e92a51_inst21 … (+30)
[refs] class 8: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 9: 569 -> 40 diverse ref(s) for DINOv2 proto bank: source-e497834ebd715d706366646daeb2269e_inst7, ultralytics-ppe-nov2026-467_inst6, source-ba3aa5d1715142e5e962c3376ceca1a6_inst37, source-eabef86b2cecf0e9ffe06787e0298a38_inst29, source-2edf4fdf4d031044f1130ef13ef129cc_inst22, source-5aef8f03a221e1d0f7dc1df4501dda5e_inst27, source-780e7e97f694e83abcc1b08020b98069_inst23, ultralytics-ppe-nov2026-267_inst6, source-dc39994159adfb8e9c6bdafba4841c4b_inst33, ultralytics-ppe-nov2026-068_inst6 … (+30)
[refs] class 9: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 10: 275 -> 40 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-216_inst15, ultralytics-ppe-nov2026-064_inst6, ultralytics-ppe-nov2026-512_inst6, source-263d7b7ac85694a4522bfcb8c63666cd_inst13, ultralytics-ppe-nov2026-484_inst6, source-305deca284bbf639bcdcc00671e240aa_inst6, ultralytics-ppe-nov2026-201_inst9, source-305deca284bbf639bcdcc00671e240aa_inst5, source-8ead761447ba33e51dc962d76ef2e676_inst21, ultralytics-ppe-nov2026-345_inst9 … (+30)
[refs] class 10: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 11: 437 -> 40 diverse ref(s) for DINOv2 proto bank: source-f2c0cc19bce353d225ec84760d453285_inst19, ultralytics-ppe-nov2026-065_inst8, ultralytics-ppe-nov2026-011_inst19, ultralytics-ppe-nov2026-529_inst11, ultralytics-ppe-nov2026-631_inst9, source-47de682b0dee4fa2a0733219b8dbc82c_inst18, source-3cf6bc936c8bbba28f8a203937897ae5_inst6, source-8cbee74c2adeab1f66f1ebc0e625f48e_inst13, source-b389d7e14d9086cef980ae814b3c2bb1_inst7, ultralytics-ppe-nov2026-706_inst13 … (+30)
[refs] class 11: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[small-obj] classes [0, 1, 4, 7, 9] below --small-obj-thresh=0.01 — SAM3 masking skipped, DINOv2 mean-pools the raw crop
[refs] cls2: 39 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls3: 39 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls5: 40 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls6: 31 ref(s) masked, 9 SAM3 miss(es) — dropped from proto bank
[refs] cls8: 36 ref(s) masked, 4 SAM3 miss(es) — dropped from proto bank
[refs] cls10: 39 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls11: 39 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[dino] auto batch size 896 (~11.1 MiB/sample, 70% of 15.6 GiB)
[dino] cls0 proto bank: (40, 768)
[dino] cls1 proto bank: (40, 768)
[dino] cls2 proto bank: (39, 768)
[dino] cls3 proto bank: (39, 768)
[dino] cls4 proto bank: (40, 768)
[dino] cls5 proto bank: (40, 768)
[dino] cls6 proto bank: (31, 768)
[dino] cls7 proto bank: (40, 768)
[dino] cls8 proto bank: (36, 768)
[dino] cls9 proto bank: (40, 768)
[dino] cls10 proto bank: (39, 768)
[dino] cls11 proto bank: (39, 768)
```

## B_k45

```
[refs] class 0: phash dedup 1282 -> 1279
[refs] class 1: phash dedup 1548 -> 1546
[dino] auto batch size 952 (~11.1 MiB/sample, 70% of 15.6 GiB)
[refs] class 0: 1279 -> 45 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-019_inst1, source-04d90135ba9a6511d789035fa9021f31_inst0, source-596bdd3a39761003f3913916c3a97c55_inst13, source-c567c20479dfc3b1087b8cccd8b2cde1_inst26, ultralytics-ppe-nov2026-557_inst18, source-7c9d94a7835cf299bd5aeddb27ded889_inst13, source-29a54f6e5e66462bbcbec90273e92a51_inst7, source-fbb201fd66f289e3ad4e6f1a71c2a282_inst3, ultralytics-ppe-nov2026-001_inst3, source-94e6c9c9971c3352b61271256f014daf_inst6 … (+35)
[refs] class 0: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 1: 1546 -> 45 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-355_inst6, source-6cec9c3de8d7970fe1dfe824f18573c2_inst22, ultralytics-ppe-nov2026-659_inst6, ultralytics-ppe-nov2026-220_inst1, source-8abad491606bbd2343029bf42b48208e_inst6, source-61d89c2415a4e91f0aded67312754811_inst20, source-fd87396c628d9e0ce1ac0177cfbf27e3_inst18, ultralytics-ppe-nov2026-607_inst1, source-3195fd14699725afdd2906e7293cc3c6_inst4, ultralytics-ppe-nov2026-555_inst5 … (+35)
[refs] class 1: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 2: 1476 -> 45 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-195_inst8, ultralytics-ppe-nov2026-003_inst3, source-dc39994159adfb8e9c6bdafba4841c4b_inst27, source-544a422efd96b918ee6696091f2fe9b4_inst6, ultralytics-ppe-nov2026-479_inst10, source-b1b1595542d8a8121fc095a8223157a2_inst6, source-6b6114dea7ecbe5997f1f0393824508b_inst3, source-473c9faf0e4f5803ad615ecc681b2a2e_inst11, ultralytics-ppe-nov2026-590_inst1, source-a28a006708943ec327e46b51d2fd3da4_inst9 … (+35)
[refs] class 2: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 3: 2003 -> 45 diverse ref(s) for DINOv2 proto bank: source-f4532be7ff670b1a99ec14fdae4b0557_inst25, ultralytics-ppe-nov2026-051_inst4, source-ba3aa5d1715142e5e962c3376ceca1a6_inst40, source-0e2a4d4463396c949ccdf3a3253e1552_inst11, source-baa51c2ac69f7c70e92d77f740b0f120_inst33, source-3403f252ac25e929eef3aa6978ad32b4_inst19, source-40884ba3867e4fb7a2dc3a778aefc18d_inst8, source-ef9a9cc38caf4d732fa9b1ac736d0c43_inst13, source-ba3aa5d1715142e5e962c3376ceca1a6_inst35, source-04d90135ba9a6511d789035fa9021f31_inst14 … (+35)
[refs] class 3: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 4: 612 -> 45 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst18, ultralytics-ppe-nov2026-033_inst14, ultralytics-ppe-nov2026-431_inst3, ultralytics-ppe-nov2026-278_inst11, ultralytics-ppe-nov2026-040_inst16, ultralytics-ppe-nov2026-618_inst5, ultralytics-ppe-nov2026-435_inst11, ultralytics-ppe-nov2026-229_inst6, ultralytics-ppe-nov2026-245_inst6, source-5832d306e381498099548ed6af72c5e6_inst19 … (+35)
[refs] class 4: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 5: 1962 -> 45 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-139_inst8, source-1b1d16e594d2940d1ce4dcc46b896830_inst1, ultralytics-ppe-nov2026-642_inst8, ultralytics-ppe-nov2026-519_inst8, source-47de682b0dee4fa2a0733219b8dbc82c_inst11, ultralytics-ppe-nov2026-607_inst6, source-3cf6bc936c8bbba28f8a203937897ae5_inst2, ultralytics-ppe-nov2026-431_inst0, ultralytics-ppe-nov2026-517_inst9, source-c01dfb2dbae89f8e5825648d9da4e842_inst3 … (+35)
[refs] class 5: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 6: 634 -> 45 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-435_inst8, source-32b77e88696e2ba988310ec62da94f16_inst48, ultralytics-ppe-nov2026-681_inst2, source-ed4844326ba70151720a9335b90cff72_inst2, ultralytics-ppe-nov2026-034_inst32, ultralytics-ppe-nov2026-190_inst2, source-ba3aa5d1715142e5e962c3376ceca1a6_inst17, source-3cf6bc936c8bbba28f8a203937897ae5_inst14, ultralytics-ppe-nov2026-302_inst7, source-d7438871aefee49d413dcede6cd5d626_inst7 … (+35)
[refs] class 6: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 7: 265 -> 45 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst31, source-c2733b5e92b8e910dc4688e6fd3c51fc_inst4, ultralytics-ppe-nov2026-505_inst7, ultralytics-ppe-nov2026-011_inst18, ultralytics-ppe-nov2026-185_inst7, source-d5d4d730d8fbb824c881beaa18b34d80_inst31, ultralytics-ppe-nov2026-489_inst4, source-0e59077ef44d89df66fa8b340b15aca0_inst22, ultralytics-ppe-nov2026-511_inst7, ultralytics-ppe-nov2026-467_inst5 … (+35)
[refs] class 7: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 8: 839 -> 45 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-454_inst4, source-c567c20479dfc3b1087b8cccd8b2cde1_inst28, ultralytics-ppe-nov2026-192_inst7, source-46bf35b89356017177174a7dd0b35186_inst28, source-efa761a319a37ba07e58ccd0bc1a4003_inst25, ultralytics-ppe-nov2026-617_inst11, source-c1a6a639b24165512bbfc8b97c5e0fb3_inst23, source-ba3aa5d1715142e5e962c3376ceca1a6_inst23, ultralytics-ppe-nov2026-040_inst24, source-29a54f6e5e66462bbcbec90273e92a51_inst21 … (+35)
[refs] class 8: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 9: 569 -> 45 diverse ref(s) for DINOv2 proto bank: source-e497834ebd715d706366646daeb2269e_inst7, ultralytics-ppe-nov2026-467_inst6, source-ba3aa5d1715142e5e962c3376ceca1a6_inst37, source-eabef86b2cecf0e9ffe06787e0298a38_inst29, source-2edf4fdf4d031044f1130ef13ef129cc_inst22, source-5aef8f03a221e1d0f7dc1df4501dda5e_inst27, source-780e7e97f694e83abcc1b08020b98069_inst23, ultralytics-ppe-nov2026-267_inst6, source-dc39994159adfb8e9c6bdafba4841c4b_inst33, ultralytics-ppe-nov2026-068_inst6 … (+35)
[refs] class 9: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 10: 275 -> 45 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-216_inst15, ultralytics-ppe-nov2026-064_inst6, ultralytics-ppe-nov2026-512_inst6, source-263d7b7ac85694a4522bfcb8c63666cd_inst13, ultralytics-ppe-nov2026-484_inst6, source-305deca284bbf639bcdcc00671e240aa_inst6, ultralytics-ppe-nov2026-201_inst9, source-305deca284bbf639bcdcc00671e240aa_inst5, source-8ead761447ba33e51dc962d76ef2e676_inst21, ultralytics-ppe-nov2026-345_inst9 … (+35)
[refs] class 10: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 11: 437 -> 45 diverse ref(s) for DINOv2 proto bank: source-f2c0cc19bce353d225ec84760d453285_inst19, ultralytics-ppe-nov2026-065_inst8, ultralytics-ppe-nov2026-011_inst19, ultralytics-ppe-nov2026-529_inst11, ultralytics-ppe-nov2026-631_inst9, source-47de682b0dee4fa2a0733219b8dbc82c_inst18, source-3cf6bc936c8bbba28f8a203937897ae5_inst6, source-8cbee74c2adeab1f66f1ebc0e625f48e_inst13, source-b389d7e14d9086cef980ae814b3c2bb1_inst7, ultralytics-ppe-nov2026-706_inst13 … (+35)
[refs] class 11: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[small-obj] classes [0, 1, 4, 7, 9] below --small-obj-thresh=0.01 — SAM3 masking skipped, DINOv2 mean-pools the raw crop
[refs] cls2: 44 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls3: 44 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls5: 45 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls6: 35 ref(s) masked, 10 SAM3 miss(es) — dropped from proto bank
[refs] cls8: 41 ref(s) masked, 4 SAM3 miss(es) — dropped from proto bank
[refs] cls10: 44 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls11: 44 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[dino] auto batch size 896 (~11.1 MiB/sample, 70% of 15.6 GiB)
[dino] cls0 proto bank: (45, 768)
[dino] cls1 proto bank: (45, 768)
[dino] cls2 proto bank: (44, 768)
[dino] cls3 proto bank: (44, 768)
[dino] cls4 proto bank: (45, 768)
[dino] cls5 proto bank: (45, 768)
[dino] cls6 proto bank: (35, 768)
[dino] cls7 proto bank: (45, 768)
[dino] cls8 proto bank: (41, 768)
[dino] cls9 proto bank: (45, 768)
[dino] cls10 proto bank: (44, 768)
[dino] cls11 proto bank: (44, 768)
```

## B_k50

```
[refs] class 0: phash dedup 1282 -> 1279
[refs] class 1: phash dedup 1548 -> 1546
[dino] auto batch size 952 (~11.1 MiB/sample, 70% of 15.6 GiB)
[refs] class 0: 1279 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-019_inst1, source-04d90135ba9a6511d789035fa9021f31_inst0, source-596bdd3a39761003f3913916c3a97c55_inst13, source-c567c20479dfc3b1087b8cccd8b2cde1_inst26, ultralytics-ppe-nov2026-557_inst18, source-7c9d94a7835cf299bd5aeddb27ded889_inst13, source-29a54f6e5e66462bbcbec90273e92a51_inst7, source-fbb201fd66f289e3ad4e6f1a71c2a282_inst3, ultralytics-ppe-nov2026-001_inst3, source-94e6c9c9971c3352b61271256f014daf_inst6 … (+40)
[refs] class 0: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 1: 1546 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-355_inst6, source-6cec9c3de8d7970fe1dfe824f18573c2_inst22, ultralytics-ppe-nov2026-659_inst6, ultralytics-ppe-nov2026-220_inst1, source-8abad491606bbd2343029bf42b48208e_inst6, source-61d89c2415a4e91f0aded67312754811_inst20, source-fd87396c628d9e0ce1ac0177cfbf27e3_inst18, ultralytics-ppe-nov2026-607_inst1, source-3195fd14699725afdd2906e7293cc3c6_inst4, ultralytics-ppe-nov2026-555_inst5 … (+40)
[refs] class 1: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 2: 1476 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-195_inst8, ultralytics-ppe-nov2026-003_inst3, source-dc39994159adfb8e9c6bdafba4841c4b_inst27, source-544a422efd96b918ee6696091f2fe9b4_inst6, ultralytics-ppe-nov2026-479_inst10, source-b1b1595542d8a8121fc095a8223157a2_inst6, source-6b6114dea7ecbe5997f1f0393824508b_inst3, source-473c9faf0e4f5803ad615ecc681b2a2e_inst11, ultralytics-ppe-nov2026-590_inst1, source-a28a006708943ec327e46b51d2fd3da4_inst9 … (+40)
[refs] class 2: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 3: 2003 -> 50 diverse ref(s) for DINOv2 proto bank: source-f4532be7ff670b1a99ec14fdae4b0557_inst25, ultralytics-ppe-nov2026-051_inst4, source-ba3aa5d1715142e5e962c3376ceca1a6_inst40, source-0e2a4d4463396c949ccdf3a3253e1552_inst11, source-baa51c2ac69f7c70e92d77f740b0f120_inst33, source-3403f252ac25e929eef3aa6978ad32b4_inst19, source-40884ba3867e4fb7a2dc3a778aefc18d_inst8, source-ef9a9cc38caf4d732fa9b1ac736d0c43_inst13, source-ba3aa5d1715142e5e962c3376ceca1a6_inst35, source-04d90135ba9a6511d789035fa9021f31_inst14 … (+40)
[refs] class 3: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 4: 612 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst18, ultralytics-ppe-nov2026-033_inst14, ultralytics-ppe-nov2026-431_inst3, ultralytics-ppe-nov2026-278_inst11, ultralytics-ppe-nov2026-040_inst16, ultralytics-ppe-nov2026-618_inst5, ultralytics-ppe-nov2026-435_inst11, ultralytics-ppe-nov2026-229_inst6, ultralytics-ppe-nov2026-245_inst6, source-5832d306e381498099548ed6af72c5e6_inst19 … (+40)
[refs] class 4: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 5: 1962 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-139_inst8, source-1b1d16e594d2940d1ce4dcc46b896830_inst1, ultralytics-ppe-nov2026-642_inst8, ultralytics-ppe-nov2026-519_inst8, source-47de682b0dee4fa2a0733219b8dbc82c_inst11, ultralytics-ppe-nov2026-607_inst6, source-3cf6bc936c8bbba28f8a203937897ae5_inst2, ultralytics-ppe-nov2026-431_inst0, ultralytics-ppe-nov2026-517_inst9, source-c01dfb2dbae89f8e5825648d9da4e842_inst3 … (+40)
[refs] class 5: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 6: 634 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-435_inst8, source-32b77e88696e2ba988310ec62da94f16_inst48, ultralytics-ppe-nov2026-681_inst2, source-ed4844326ba70151720a9335b90cff72_inst2, ultralytics-ppe-nov2026-034_inst32, ultralytics-ppe-nov2026-190_inst2, source-ba3aa5d1715142e5e962c3376ceca1a6_inst17, source-3cf6bc936c8bbba28f8a203937897ae5_inst14, ultralytics-ppe-nov2026-302_inst7, source-d7438871aefee49d413dcede6cd5d626_inst7 … (+40)
[refs] class 6: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 7: 265 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst31, source-c2733b5e92b8e910dc4688e6fd3c51fc_inst4, ultralytics-ppe-nov2026-505_inst7, ultralytics-ppe-nov2026-011_inst18, ultralytics-ppe-nov2026-185_inst7, source-d5d4d730d8fbb824c881beaa18b34d80_inst31, ultralytics-ppe-nov2026-489_inst4, source-0e59077ef44d89df66fa8b340b15aca0_inst22, ultralytics-ppe-nov2026-511_inst7, ultralytics-ppe-nov2026-467_inst5 … (+40)
[refs] class 7: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 8: 839 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-454_inst4, source-c567c20479dfc3b1087b8cccd8b2cde1_inst28, ultralytics-ppe-nov2026-192_inst7, source-46bf35b89356017177174a7dd0b35186_inst28, source-efa761a319a37ba07e58ccd0bc1a4003_inst25, ultralytics-ppe-nov2026-617_inst11, source-c1a6a639b24165512bbfc8b97c5e0fb3_inst23, source-ba3aa5d1715142e5e962c3376ceca1a6_inst23, ultralytics-ppe-nov2026-040_inst24, source-29a54f6e5e66462bbcbec90273e92a51_inst21 … (+40)
[refs] class 8: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 9: 569 -> 50 diverse ref(s) for DINOv2 proto bank: source-e497834ebd715d706366646daeb2269e_inst7, ultralytics-ppe-nov2026-467_inst6, source-ba3aa5d1715142e5e962c3376ceca1a6_inst37, source-eabef86b2cecf0e9ffe06787e0298a38_inst29, source-2edf4fdf4d031044f1130ef13ef129cc_inst22, source-5aef8f03a221e1d0f7dc1df4501dda5e_inst27, source-780e7e97f694e83abcc1b08020b98069_inst23, ultralytics-ppe-nov2026-267_inst6, source-dc39994159adfb8e9c6bdafba4841c4b_inst33, ultralytics-ppe-nov2026-068_inst6 … (+40)
[refs] class 9: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 10: 275 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-216_inst15, ultralytics-ppe-nov2026-064_inst6, ultralytics-ppe-nov2026-512_inst6, source-263d7b7ac85694a4522bfcb8c63666cd_inst13, ultralytics-ppe-nov2026-484_inst6, source-305deca284bbf639bcdcc00671e240aa_inst6, ultralytics-ppe-nov2026-201_inst9, source-305deca284bbf639bcdcc00671e240aa_inst5, source-8ead761447ba33e51dc962d76ef2e676_inst21, ultralytics-ppe-nov2026-345_inst9 … (+40)
[refs] class 10: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 11: 437 -> 50 diverse ref(s) for DINOv2 proto bank: source-f2c0cc19bce353d225ec84760d453285_inst19, ultralytics-ppe-nov2026-065_inst8, ultralytics-ppe-nov2026-011_inst19, ultralytics-ppe-nov2026-529_inst11, ultralytics-ppe-nov2026-631_inst9, source-47de682b0dee4fa2a0733219b8dbc82c_inst18, source-3cf6bc936c8bbba28f8a203937897ae5_inst6, source-8cbee74c2adeab1f66f1ebc0e625f48e_inst13, source-b389d7e14d9086cef980ae814b3c2bb1_inst7, ultralytics-ppe-nov2026-706_inst13 … (+40)
[refs] class 11: 5 ref(s) for SAM3 exemplars (subset of proto bank)
[small-obj] classes [0, 1, 4, 7, 9] below --small-obj-thresh=0.01 — SAM3 masking skipped, DINOv2 mean-pools the raw crop
[refs] cls2: 49 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls3: 49 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls5: 50 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls6: 40 ref(s) masked, 10 SAM3 miss(es) — dropped from proto bank
[refs] cls8: 46 ref(s) masked, 4 SAM3 miss(es) — dropped from proto bank
[refs] cls10: 49 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls11: 48 ref(s) masked, 2 SAM3 miss(es) — dropped from proto bank
[dino] auto batch size 896 (~11.1 MiB/sample, 70% of 15.6 GiB)
[dino] cls0 proto bank: (50, 768)
[dino] cls1 proto bank: (50, 768)
[dino] cls2 proto bank: (49, 768)
[dino] cls3 proto bank: (49, 768)
[dino] cls4 proto bank: (50, 768)
[dino] cls5 proto bank: (50, 768)
[dino] cls6 proto bank: (40, 768)
[dino] cls7 proto bank: (50, 768)
[dino] cls8 proto bank: (46, 768)
[dino] cls9 proto bank: (50, 768)
[dino] cls10 proto bank: (49, 768)
[dino] cls11 proto bank: (48, 768)
```

## B_ex3_k50

```
[refs] class 0: phash dedup 1282 -> 1279
[refs] class 1: phash dedup 1548 -> 1546
[dino] auto batch size 952 (~11.1 MiB/sample, 70% of 15.6 GiB)
[refs] class 0: 1279 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-019_inst1, source-04d90135ba9a6511d789035fa9021f31_inst0, source-596bdd3a39761003f3913916c3a97c55_inst13, source-c567c20479dfc3b1087b8cccd8b2cde1_inst26, ultralytics-ppe-nov2026-557_inst18, source-7c9d94a7835cf299bd5aeddb27ded889_inst13, source-29a54f6e5e66462bbcbec90273e92a51_inst7, source-fbb201fd66f289e3ad4e6f1a71c2a282_inst3, ultralytics-ppe-nov2026-001_inst3, source-94e6c9c9971c3352b61271256f014daf_inst6 … (+40)
[refs] class 0: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 1: 1546 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-355_inst6, source-6cec9c3de8d7970fe1dfe824f18573c2_inst22, ultralytics-ppe-nov2026-659_inst6, ultralytics-ppe-nov2026-220_inst1, source-8abad491606bbd2343029bf42b48208e_inst6, source-61d89c2415a4e91f0aded67312754811_inst20, source-fd87396c628d9e0ce1ac0177cfbf27e3_inst18, ultralytics-ppe-nov2026-607_inst1, source-3195fd14699725afdd2906e7293cc3c6_inst4, ultralytics-ppe-nov2026-555_inst5 … (+40)
[refs] class 1: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 2: 1476 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-195_inst8, ultralytics-ppe-nov2026-003_inst3, source-dc39994159adfb8e9c6bdafba4841c4b_inst27, source-544a422efd96b918ee6696091f2fe9b4_inst6, ultralytics-ppe-nov2026-479_inst10, source-b1b1595542d8a8121fc095a8223157a2_inst6, source-6b6114dea7ecbe5997f1f0393824508b_inst3, source-473c9faf0e4f5803ad615ecc681b2a2e_inst11, ultralytics-ppe-nov2026-590_inst1, source-a28a006708943ec327e46b51d2fd3da4_inst9 … (+40)
[refs] class 2: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 3: 2003 -> 50 diverse ref(s) for DINOv2 proto bank: source-f4532be7ff670b1a99ec14fdae4b0557_inst25, ultralytics-ppe-nov2026-051_inst4, source-ba3aa5d1715142e5e962c3376ceca1a6_inst40, source-0e2a4d4463396c949ccdf3a3253e1552_inst11, source-baa51c2ac69f7c70e92d77f740b0f120_inst33, source-3403f252ac25e929eef3aa6978ad32b4_inst19, source-40884ba3867e4fb7a2dc3a778aefc18d_inst8, source-ef9a9cc38caf4d732fa9b1ac736d0c43_inst13, source-ba3aa5d1715142e5e962c3376ceca1a6_inst35, source-04d90135ba9a6511d789035fa9021f31_inst14 … (+40)
[refs] class 3: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 4: 612 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst18, ultralytics-ppe-nov2026-033_inst14, ultralytics-ppe-nov2026-431_inst3, ultralytics-ppe-nov2026-278_inst11, ultralytics-ppe-nov2026-040_inst16, ultralytics-ppe-nov2026-618_inst5, ultralytics-ppe-nov2026-435_inst11, ultralytics-ppe-nov2026-229_inst6, ultralytics-ppe-nov2026-245_inst6, source-5832d306e381498099548ed6af72c5e6_inst19 … (+40)
[refs] class 4: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 5: 1962 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-139_inst8, source-1b1d16e594d2940d1ce4dcc46b896830_inst1, ultralytics-ppe-nov2026-642_inst8, ultralytics-ppe-nov2026-519_inst8, source-47de682b0dee4fa2a0733219b8dbc82c_inst11, ultralytics-ppe-nov2026-607_inst6, source-3cf6bc936c8bbba28f8a203937897ae5_inst2, ultralytics-ppe-nov2026-431_inst0, ultralytics-ppe-nov2026-517_inst9, source-c01dfb2dbae89f8e5825648d9da4e842_inst3 … (+40)
[refs] class 5: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 6: 634 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-435_inst8, source-32b77e88696e2ba988310ec62da94f16_inst48, ultralytics-ppe-nov2026-681_inst2, source-ed4844326ba70151720a9335b90cff72_inst2, ultralytics-ppe-nov2026-034_inst32, ultralytics-ppe-nov2026-190_inst2, source-ba3aa5d1715142e5e962c3376ceca1a6_inst17, source-3cf6bc936c8bbba28f8a203937897ae5_inst14, ultralytics-ppe-nov2026-302_inst7, source-d7438871aefee49d413dcede6cd5d626_inst7 … (+40)
[refs] class 6: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 7: 265 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst31, source-c2733b5e92b8e910dc4688e6fd3c51fc_inst4, ultralytics-ppe-nov2026-505_inst7, ultralytics-ppe-nov2026-011_inst18, ultralytics-ppe-nov2026-185_inst7, source-d5d4d730d8fbb824c881beaa18b34d80_inst31, ultralytics-ppe-nov2026-489_inst4, source-0e59077ef44d89df66fa8b340b15aca0_inst22, ultralytics-ppe-nov2026-511_inst7, ultralytics-ppe-nov2026-467_inst5 … (+40)
[refs] class 7: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 8: 839 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-454_inst4, source-c567c20479dfc3b1087b8cccd8b2cde1_inst28, ultralytics-ppe-nov2026-192_inst7, source-46bf35b89356017177174a7dd0b35186_inst28, source-efa761a319a37ba07e58ccd0bc1a4003_inst25, ultralytics-ppe-nov2026-617_inst11, source-c1a6a639b24165512bbfc8b97c5e0fb3_inst23, source-ba3aa5d1715142e5e962c3376ceca1a6_inst23, ultralytics-ppe-nov2026-040_inst24, source-29a54f6e5e66462bbcbec90273e92a51_inst21 … (+40)
[refs] class 8: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 9: 569 -> 50 diverse ref(s) for DINOv2 proto bank: source-e497834ebd715d706366646daeb2269e_inst7, ultralytics-ppe-nov2026-467_inst6, source-ba3aa5d1715142e5e962c3376ceca1a6_inst37, source-eabef86b2cecf0e9ffe06787e0298a38_inst29, source-2edf4fdf4d031044f1130ef13ef129cc_inst22, source-5aef8f03a221e1d0f7dc1df4501dda5e_inst27, source-780e7e97f694e83abcc1b08020b98069_inst23, ultralytics-ppe-nov2026-267_inst6, source-dc39994159adfb8e9c6bdafba4841c4b_inst33, ultralytics-ppe-nov2026-068_inst6 … (+40)
[refs] class 9: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 10: 275 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-216_inst15, ultralytics-ppe-nov2026-064_inst6, ultralytics-ppe-nov2026-512_inst6, source-263d7b7ac85694a4522bfcb8c63666cd_inst13, ultralytics-ppe-nov2026-484_inst6, source-305deca284bbf639bcdcc00671e240aa_inst6, ultralytics-ppe-nov2026-201_inst9, source-305deca284bbf639bcdcc00671e240aa_inst5, source-8ead761447ba33e51dc962d76ef2e676_inst21, ultralytics-ppe-nov2026-345_inst9 … (+40)
[refs] class 10: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 11: 437 -> 50 diverse ref(s) for DINOv2 proto bank: source-f2c0cc19bce353d225ec84760d453285_inst19, ultralytics-ppe-nov2026-065_inst8, ultralytics-ppe-nov2026-011_inst19, ultralytics-ppe-nov2026-529_inst11, ultralytics-ppe-nov2026-631_inst9, source-47de682b0dee4fa2a0733219b8dbc82c_inst18, source-3cf6bc936c8bbba28f8a203937897ae5_inst6, source-8cbee74c2adeab1f66f1ebc0e625f48e_inst13, source-b389d7e14d9086cef980ae814b3c2bb1_inst7, ultralytics-ppe-nov2026-706_inst13 … (+40)
[refs] class 11: 3 ref(s) for SAM3 exemplars (subset of proto bank)
[small-obj] classes [0, 1, 4, 7, 9] below --small-obj-thresh=0.01 — SAM3 masking skipped, DINOv2 mean-pools the raw crop
[refs] cls2: 49 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls3: 49 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls5: 50 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls6: 40 ref(s) masked, 10 SAM3 miss(es) — dropped from proto bank
[refs] cls8: 46 ref(s) masked, 4 SAM3 miss(es) — dropped from proto bank
[refs] cls10: 49 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls11: 48 ref(s) masked, 2 SAM3 miss(es) — dropped from proto bank
[dino] auto batch size 888 (~11.1 MiB/sample, 70% of 15.6 GiB)
[dino] cls0 proto bank: (50, 768)
[dino] cls1 proto bank: (50, 768)
[dino] cls2 proto bank: (49, 768)
[dino] cls3 proto bank: (49, 768)
[dino] cls4 proto bank: (50, 768)
[dino] cls5 proto bank: (50, 768)
[dino] cls6 proto bank: (40, 768)
[dino] cls7 proto bank: (50, 768)
[dino] cls8 proto bank: (46, 768)
[dino] cls9 proto bank: (50, 768)
[dino] cls10 proto bank: (49, 768)
[dino] cls11 proto bank: (48, 768)
```
## B_ex10_k50

```
[refs] class 0: phash dedup 1282 -> 1279
[refs] class 1: phash dedup 1548 -> 1546
[dino] auto batch size 952 (~11.1 MiB/sample, 70% of 15.6 GiB)
[refs] class 0: 1279 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-019_inst1, source-04d90135ba9a6511d789035fa9021f31_inst0, source-596bdd3a39761003f3913916c3a97c55_inst13, source-c567c20479dfc3b1087b8cccd8b2cde1_inst26, ultralytics-ppe-nov2026-557_inst18, source-7c9d94a7835cf299bd5aeddb27ded889_inst13, source-29a54f6e5e66462bbcbec90273e92a51_inst7, source-fbb201fd66f289e3ad4e6f1a71c2a282_inst3, ultralytics-ppe-nov2026-001_inst3, source-94e6c9c9971c3352b61271256f014daf_inst6 … (+40)
[refs] class 0: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 1: 1546 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-355_inst6, source-6cec9c3de8d7970fe1dfe824f18573c2_inst22, ultralytics-ppe-nov2026-659_inst6, ultralytics-ppe-nov2026-220_inst1, source-8abad491606bbd2343029bf42b48208e_inst6, source-61d89c2415a4e91f0aded67312754811_inst20, source-fd87396c628d9e0ce1ac0177cfbf27e3_inst18, ultralytics-ppe-nov2026-607_inst1, source-3195fd14699725afdd2906e7293cc3c6_inst4, ultralytics-ppe-nov2026-555_inst5 … (+40)
[refs] class 1: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 2: 1476 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-195_inst8, ultralytics-ppe-nov2026-003_inst3, source-dc39994159adfb8e9c6bdafba4841c4b_inst27, source-544a422efd96b918ee6696091f2fe9b4_inst6, ultralytics-ppe-nov2026-479_inst10, source-b1b1595542d8a8121fc095a8223157a2_inst6, source-6b6114dea7ecbe5997f1f0393824508b_inst3, source-473c9faf0e4f5803ad615ecc681b2a2e_inst11, ultralytics-ppe-nov2026-590_inst1, source-a28a006708943ec327e46b51d2fd3da4_inst9 … (+40)
[refs] class 2: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 3: 2003 -> 50 diverse ref(s) for DINOv2 proto bank: source-f4532be7ff670b1a99ec14fdae4b0557_inst25, ultralytics-ppe-nov2026-051_inst4, source-ba3aa5d1715142e5e962c3376ceca1a6_inst40, source-0e2a4d4463396c949ccdf3a3253e1552_inst11, source-baa51c2ac69f7c70e92d77f740b0f120_inst33, source-3403f252ac25e929eef3aa6978ad32b4_inst19, source-40884ba3867e4fb7a2dc3a778aefc18d_inst8, source-ef9a9cc38caf4d732fa9b1ac736d0c43_inst13, source-ba3aa5d1715142e5e962c3376ceca1a6_inst35, source-04d90135ba9a6511d789035fa9021f31_inst14 … (+40)
[refs] class 3: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 4: 612 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst18, ultralytics-ppe-nov2026-033_inst14, ultralytics-ppe-nov2026-431_inst3, ultralytics-ppe-nov2026-278_inst11, ultralytics-ppe-nov2026-040_inst16, ultralytics-ppe-nov2026-618_inst5, ultralytics-ppe-nov2026-435_inst11, ultralytics-ppe-nov2026-229_inst6, ultralytics-ppe-nov2026-245_inst6, source-5832d306e381498099548ed6af72c5e6_inst19 … (+40)
[refs] class 4: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 5: 1962 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-139_inst8, source-1b1d16e594d2940d1ce4dcc46b896830_inst1, ultralytics-ppe-nov2026-642_inst8, ultralytics-ppe-nov2026-519_inst8, source-47de682b0dee4fa2a0733219b8dbc82c_inst11, ultralytics-ppe-nov2026-607_inst6, source-3cf6bc936c8bbba28f8a203937897ae5_inst2, ultralytics-ppe-nov2026-431_inst0, ultralytics-ppe-nov2026-517_inst9, source-c01dfb2dbae89f8e5825648d9da4e842_inst3 … (+40)
[refs] class 5: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 6: 634 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-435_inst8, source-32b77e88696e2ba988310ec62da94f16_inst48, ultralytics-ppe-nov2026-681_inst2, source-ed4844326ba70151720a9335b90cff72_inst2, ultralytics-ppe-nov2026-034_inst32, ultralytics-ppe-nov2026-190_inst2, source-ba3aa5d1715142e5e962c3376ceca1a6_inst17, source-3cf6bc936c8bbba28f8a203937897ae5_inst14, ultralytics-ppe-nov2026-302_inst7, source-d7438871aefee49d413dcede6cd5d626_inst7 … (+40)
[refs] class 6: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 7: 265 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-010_inst31, source-c2733b5e92b8e910dc4688e6fd3c51fc_inst4, ultralytics-ppe-nov2026-505_inst7, ultralytics-ppe-nov2026-011_inst18, ultralytics-ppe-nov2026-185_inst7, source-d5d4d730d8fbb824c881beaa18b34d80_inst31, ultralytics-ppe-nov2026-489_inst4, source-0e59077ef44d89df66fa8b340b15aca0_inst22, ultralytics-ppe-nov2026-511_inst7, ultralytics-ppe-nov2026-467_inst5 … (+40)
[refs] class 7: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 8: 839 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-454_inst4, source-c567c20479dfc3b1087b8cccd8b2cde1_inst28, ultralytics-ppe-nov2026-192_inst7, source-46bf35b89356017177174a7dd0b35186_inst28, source-efa761a319a37ba07e58ccd0bc1a4003_inst25, ultralytics-ppe-nov2026-617_inst11, source-c1a6a639b24165512bbfc8b97c5e0fb3_inst23, source-ba3aa5d1715142e5e962c3376ceca1a6_inst23, ultralytics-ppe-nov2026-040_inst24, source-29a54f6e5e66462bbcbec90273e92a51_inst21 … (+40)
[refs] class 8: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 9: 569 -> 50 diverse ref(s) for DINOv2 proto bank: source-e497834ebd715d706366646daeb2269e_inst7, ultralytics-ppe-nov2026-467_inst6, source-ba3aa5d1715142e5e962c3376ceca1a6_inst37, source-eabef86b2cecf0e9ffe06787e0298a38_inst29, source-2edf4fdf4d031044f1130ef13ef129cc_inst22, source-5aef8f03a221e1d0f7dc1df4501dda5e_inst27, source-780e7e97f694e83abcc1b08020b98069_inst23, ultralytics-ppe-nov2026-267_inst6, source-dc39994159adfb8e9c6bdafba4841c4b_inst33, ultralytics-ppe-nov2026-068_inst6 … (+40)
[refs] class 9: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 10: 275 -> 50 diverse ref(s) for DINOv2 proto bank: ultralytics-ppe-nov2026-216_inst15, ultralytics-ppe-nov2026-064_inst6, ultralytics-ppe-nov2026-512_inst6, source-263d7b7ac85694a4522bfcb8c63666cd_inst13, ultralytics-ppe-nov2026-484_inst6, source-305deca284bbf639bcdcc00671e240aa_inst6, ultralytics-ppe-nov2026-201_inst9, source-305deca284bbf639bcdcc00671e240aa_inst5, source-8ead761447ba33e51dc962d76ef2e676_inst21, ultralytics-ppe-nov2026-345_inst9 … (+40)
[refs] class 10: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[refs] class 11: 437 -> 50 diverse ref(s) for DINOv2 proto bank: source-f2c0cc19bce353d225ec84760d453285_inst19, ultralytics-ppe-nov2026-065_inst8, ultralytics-ppe-nov2026-011_inst19, ultralytics-ppe-nov2026-529_inst11, ultralytics-ppe-nov2026-631_inst9, source-47de682b0dee4fa2a0733219b8dbc82c_inst18, source-3cf6bc936c8bbba28f8a203937897ae5_inst6, source-8cbee74c2adeab1f66f1ebc0e625f48e_inst13, source-b389d7e14d9086cef980ae814b3c2bb1_inst7, ultralytics-ppe-nov2026-706_inst13 … (+40)
[refs] class 11: 10 ref(s) for SAM3 exemplars (subset of proto bank)
[small-obj] classes [0, 1, 4, 7, 9] below --small-obj-thresh=0.01 — SAM3 masking skipped, DINOv2 mean-pools the raw crop
[refs] cls2: 49 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls3: 49 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls5: 50 ref(s) masked, 0 SAM3 miss(es) — dropped from proto bank
[refs] cls6: 40 ref(s) masked, 10 SAM3 miss(es) — dropped from proto bank
[refs] cls8: 46 ref(s) masked, 4 SAM3 miss(es) — dropped from proto bank
[refs] cls10: 49 ref(s) masked, 1 SAM3 miss(es) — dropped from proto bank
[refs] cls11: 48 ref(s) masked, 2 SAM3 miss(es) — dropped from proto bank
[dino] auto batch size 904 (~11.1 MiB/sample, 70% of 15.6 GiB)
[dino] cls0 proto bank: (50, 768)
[dino] cls1 proto bank: (50, 768)
[dino] cls2 proto bank: (49, 768)
[dino] cls3 proto bank: (49, 768)
[dino] cls4 proto bank: (50, 768)
[dino] cls5 proto bank: (50, 768)
[dino] cls6 proto bank: (40, 768)
[dino] cls7 proto bank: (50, 768)
[dino] cls8 proto bank: (46, 768)
[dino] cls9 proto bank: (50, 768)
[dino] cls10 proto bank: (49, 768)
[dino] cls11 proto bank: (48, 768)
```

