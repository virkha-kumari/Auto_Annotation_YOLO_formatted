"""
Collect k-shot sweep results (experiments/run_kshot.sh) into markdown tables for the README.
Reads <runs>/{A,B}_k<K>/metrics.json + timing.json.

Usage:
    python experiments/summarize_kshot.py --runs runs/kshot --classes helmet gloves vest ...
"""

import argparse
import json
import re
from pathlib import Path

RUN_RE = re.compile(r"^([AB])_k(\d+)$")


def fmt_time(s: float | None) -> str:
    return "—" if s is None else f"{int(s // 60)}m{int(s % 60):02d}s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--classes", nargs="+", required=True)
    args = p.parse_args()

    runs = {}
    for d in Path(args.runs).iterdir():
        m = RUN_RE.match(d.name)
        if m and (d / "metrics.json").exists():
            timing = json.loads((d / "timing.json").read_text()) if (d / "timing.json").exists() else {}
            runs[(m.group(1), int(m.group(2)))] = (json.loads((d / "metrics.json").read_text()), timing)
    if not runs:
        print("no runs found"); return

    print("### mAP vs k (labelled instances per class)\n")
    print("| Pipeline | k | mAP@.50 | mAP@.5:.95 | nPred | nGT | classes in mAP | total time | s/target |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (pl, k), (m, t) in sorted(runs.items()):
        tot = t.get("total_s")
        per = f"{tot / t['n_targets']:.1f}" if tot and t.get("n_targets") else "—"
        print(f"| {pl} | {k} | {m['map50']:.3f} | {m['map50_95']:.3f} | {m['n_pred']} | {m['n_gt']} "
              f"| {m['n_classes_in_map']} | {fmt_time(tot)} | {per} |")

    print("\n### Per-phase time (not cumulative)\n")
    print("| Pipeline | k | phases |")
    print("|---|---|---|")
    for (pl, k), (_, t) in sorted(runs.items()):
        phases = ", ".join(f"{name.removesuffix('_s')} {fmt_time(v)}" for name, v in t.items()
                           if name.endswith("_s") and name != "total_s")
        print(f"| {pl} | {k} | {phases or '—'} |")

    print("\n### Per-class AP@.50\n")
    keys = sorted(runs)
    print("| Class | " + " | ".join(f"{pl} k={k}" for pl, k in keys) + " |")
    print("|---|" + "---|" * len(keys))
    for name in args.classes:
        cells = []
        for key in keys:
            c = runs[key][0]["per_class"].get(name)
            cells.append("—" if c is None else f"{c['ap50']:.3f}")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
