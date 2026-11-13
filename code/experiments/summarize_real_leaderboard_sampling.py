"""Summaries of real_leaderboard_sampling.py output: Figure 2 under (F) and (S), with e-Bonferroni.

Prints, per design, benchmark and subset, the percentage of true dominances certified at each
checkpoint (mean over orders with a 95% normal interval), and the fraction of orders with a false
statement at some look (exact 95% Clopper-Pearson interval). For design (F) it also checks that
ours, Holm and Romano-Wolf reproduce results/real_leaderboard/real_leaderboard_orders50.json.
Writes <result>_summary.json next to the input.

Usage:  python -m experiments.summarize_real_leaderboard_sampling ../results/real_leaderboard_sampling/fig2_orders50.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

METHODS = ("ours", "ebonf", "fixed_rw", "fixed_z", "fixed_n")
FRACS = ("0.05", "0.1", "0.25", "0.5", "0.75", "1.0")


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def summarize(runs: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in runs:
        groups[(r["design"], r["benchmark"], r["subset"])].append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        tp = rs[0]["true_pairs"]
        row = {"orders": len(rs), "true_pairs": tp, "certified_pct": {}, "ever_wrong": {}}
        for m in METHODS:
            row["certified_pct"][m] = {}
            for f in FRACS:
                v = np.array([100.0 * r[m][f]["true_certified"] / tp for r in rs])
                half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                row["certified_pct"][m][f] = (float(v.mean()), float(half))
            if m in ("ours", "ebonf"):
                k = sum(bool(r["ever_wrong"][m]) for r in rs)
            else:
                k = sum(bool(r[m]["1.0"]["ever_wrong_so_far"]) for r in rs)
            row["ever_wrong"][m] = (k, len(rs), *clopper_pearson(k, len(rs)))
        out["/".join(key)] = row
    return out


def check_reproduction(summary: dict, runs: list[dict], reference: Path) -> int:
    """Design (F) against the saved real-leaderboard aggregates (same seeds)."""
    ref = json.loads(reference.read_text())["results"]
    bad = 0
    names = {"ours": "ours", "fixed_n": "fixed_n", "fixed_z": "fixed_z", "fixed_rw": "fixed_rw"}
    groups = defaultdict(list)
    for r in runs:
        if r["design"] == "F":
            groups[f"{r['benchmark']}/{r['subset']}"].append(r)
    for key, rs in groups.items():
        for m, refname in names.items():
            for f in FRACS:
                mine = float(np.mean([r[m][f]["true_certified"] for r in rs]))
                theirs = ref[key][refname][f]["true_certified"]
                if abs(mine - theirs) > 1e-9:
                    bad += 1
                    print(f"  mismatch {key} {m} {f}: {mine} vs {theirs}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", type=Path)
    ap.add_argument("--reference", type=Path, default=Path("../results/real_leaderboard/real_leaderboard_orders50.json"))
    args = ap.parse_args()
    runs = json.loads(args.result.read_text())["runs"]
    summary = summarize(runs)
    bad = check_reproduction(summary, runs, args.reference)
    print(f"design (F) reproduces the Figure 2 aggregates: {'yes' if bad == 0 else f'NO, {bad} mismatches'}")
    for design in ("F", "S"):
        print(f"\n=== design {design}: % of true dominances certified (mean over orders), and runs with a false statement")
        for key, row in summary.items():
            if not key.startswith(design + "/"):
                continue
            print(f"{key[2:]:22s} n={row['orders']}  " + "  ".join(
                f"{m}: half {row['certified_pct'][m]['0.5'][0]:5.1f}±{row['certified_pct'][m]['0.5'][1]:.1f}"
                f" end {row['certified_pct'][m]['1.0'][0]:5.1f}" for m in ("ours", "ebonf", "fixed_rw", "fixed_z")))
            print(" " * 30 + "  ".join(f"{m} wrong {k}/{n}" for m, (k, n, lo, hi) in row["ever_wrong"].items()))
    out = args.result.with_name(args.result.stem + "_summary.json")
    out.write_text(json.dumps(summary, indent=1))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
