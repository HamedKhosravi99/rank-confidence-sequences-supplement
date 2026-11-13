"""Which pairs does Holm get wrong on all 395 models?

For every item order in which Holm, recomputed at every look, made a false statement
(monitored_full_leaderboard.py output), recompute the Holm sets at that order's first wrong look and
list the falsely certified pairs before transitive closure: the gap between the two models on the
full benchmark (in items, or in score units on TruthfulQA), whether they are exactly tied, whether
their score vectors are identical, the items seen, and the pair's z-statistic at that look.
Closure cannot create the first false statement, so these pairs are the cause of each error.
The paired variance is the exact one (holm_z_from_sums with x_seen), and a benchmark rerun with
it (results/monitored_full_leaderboard/exact) supplies that benchmark's error orders.

Usage:  python -m experiments.monitored_full_leaderboard_errors
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.real_leaderboard import BENCHMARKS, load
from experiments.monitored_full_leaderboard import FRACTIONS, holm_z_from_sums
from experiments.summarize_monitored_full_leaderboard import load_holm_events


def first_wrong_pairs(seed: int, benchmark: str, first_wrong: float, alpha: float, every_frac: float) -> dict:
    x_full, _ = load(benchmark)
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = sorted(set(int(t) for t in np.arange(every, n + 1, every)) | set(check) | {n})
    s, a, g = np.zeros(m), np.zeros(m), np.zeros((m, m))
    seen = 0
    for t in looks:
        chunk = x[seen:t]
        s += chunk.sum(axis=0)
        a += (chunk * chunk).sum(axis=0)
        g += chunk.T @ chunk
        seen = t
        bad = holm_z_from_sums(t, s, a, g, alpha, x_seen=x[:t]) & false_pair
        if bad.any():
            assert abs(t / n - first_wrong) < 1e-12, (benchmark, seed, t / n, first_wrong)
            pairs = []
            for i, j in zip(*np.nonzero(bad)):
                dfull = x_full[:, i] - x_full[:, j]
                dseen = x[:t, i] - x[:t, j]
                mean, sd = float(dseen.mean()), float(dseen.std(ddof=1))
                pairs.append({"i": int(i), "j": int(j),
                              "gap_full": float(n * (theta[j] - theta[i])),  # j's lead over i on the full benchmark
                              "tied": bool(theta[i] == theta[j]),
                              "identical": bool(np.all(dfull == 0)),
                              "items_differ_full": int((dfull != 0).sum()),
                              "seen_mean_diff": mean, "seen_sd": sd,
                              "z": math.sqrt(t) * mean / sd if sd > 0 else float("inf"),
                              "items_differ_seen": int((dseen != 0).sum()),
                              "seen_wins_of_worse": int((dseen > 0).sum()), "seen_wins_of_better": int((dseen < 0).sum())})
            return {"items_seen": t, "fraction": t / n, "pairs": pairs}
    raise AssertionError(f"no false statement found for {benchmark} seed {seed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("../results/monitored_full_leaderboard"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    h, _, source = load_holm_events(args.dir)
    out = []
    for b in sorted(h):
        seeds = np.random.SeedSequence([args.seed, BENCHMARKS.index(b)]).generate_state(max(h[b]) + 1)
        for o in sorted(h[b]):
            e = h[b][o]
            if not e["ever_wrong"]:
                continue
            rec = first_wrong_pairs(int(seeds[o]), b, e["first_wrong"], args.alpha, args.every_frac)
            rec.update({"benchmark": b, "order": o, "seed": int(seeds[o]), "wrong_looks": e["wrong_looks"],
                        "source": source[b]})
            out.append(rec)
    (args.dir / "holm395_errors.json").write_text(json.dumps(out, indent=1))
    by = defaultdict(list)
    for r in out:
        by[r["benchmark"]].append(r)
    for b, rs in by.items():
        pairs = [p for r in rs for p in r["pairs"]]
        print(f"{b}: {len(rs)} wrong orders, {len(pairs)} false pairs at the first wrong look; "
              f"tied {sum(p['tied'] for p in pairs)}, identical {sum(p['identical'] for p in pairs)}; "
              f"gap on the full benchmark (j ahead of i): {sorted(round(p['gap_full'], 3) for p in pairs)}; "
              f"items seen {sorted({r['items_seen'] for r in rs})}; "
              f"z {sorted(round(p['z'], 1) for p in pairs)}; items where the pair differs in the seen part "
              f"{sorted(p['items_differ_seen'] for p in pairs)}; wins of the worse model there "
              f"{sorted((p['seen_wins_of_worse'], p['seen_wins_of_better']) for p in pairs)}")


if __name__ == "__main__":
    main()
