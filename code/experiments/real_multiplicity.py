"""E3 on real leaderboard data: which multiplicity step buys power, on identical real wealths.

The simulation E3 (experiments/weighting.py, shortcut_vs_bonferroni.py) feeds the same pairwise
wealth paths to several certifiers, so that any difference is due to the multiplicity step alone.
This driver repeats that comparison on the Open LLM Leaderboard per-item data (the same
benchmarks, model subsets, random item orders and looks as experiments/real_leaderboard.py):

  e_bonferroni   threshold M(M-1)/alpha per pair, plus the free transitive closure
  shortcut       transitivity pooling (Algorithm 2)
  exact          restricted closed test over weak orders, arithmetic mean (Algorithm 1; M <= 8 only)
  exact_adjacent the same with adjacent-level weights (the ablation of the simulation E3; M <= 8)

Recorded per benchmark/subset, averaged over orders: true dominances certified at the checkpoint
fractions, whether any report was ever wrong, and a per-look comparison (number of looks at which
one certifier certified strictly more true pairs than another, and strictly fewer).

Usage:  python -m experiments.real_multiplicity --orders 50 --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.real_leaderboard import BENCHMARKS, FRACTIONS, SUBSETS, choose, leaderboard_score, load
from rcs.certify import BonferroniCertifier, ExactCertifier, ShortcutCertifier
from rcs.wealth import log_wealth_paths

PAIRS = (("shortcut", "e_bonferroni"), ("exact", "shortcut"), ("exact", "e_bonferroni"))


def _certifiers(m: int, alpha: float) -> dict:
    certs = {"e_bonferroni": BonferroniCertifier(m, alpha, close=True), "shortcut": ShortcutCertifier(m, alpha)}
    if m <= 8:
        certs["exact"] = ExactCertifier(m, alpha)
        certs["exact_adjacent"] = ExactCertifier(m, alpha, weighting="adjacent")
    return certs


def one_order(args: tuple) -> dict:
    seed, benchmark, subset, alpha, every_frac, bet = args
    scores, _ = load(benchmark)
    idx = choose(subset, leaderboard_score())
    x_full = scores[:, idx]
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = np.unique(np.concatenate([np.arange(every, n + 1, every), [n], list(check)]))

    paths = log_wealth_paths(x, population_size=n, bet=bet, at=looks)
    certs = _certifiers(m, alpha)
    out = {name: {"at": {}, "ever_wrong": False} for name in certs}
    certified = {name: np.zeros(looks.size, dtype=int) for name in certs}
    for i, t in enumerate(looks):
        for name, c in certs.items():
            d = c.update(paths[i])
            wrong = c.failed or bool((d & false_pair).any())
            out[name]["ever_wrong"] = out[name]["ever_wrong"] or wrong
            certified[name][i] = int((d & true_pair).sum())
            if int(t) in check:
                out[name]["at"][str(check[int(t)])] = certified[name][i]
    comp = {}
    for a, b in PAIRS:
        if a in certified and b in certified:
            comp[f"{a}>{b}"] = int((certified[a] > certified[b]).sum())
            comp[f"{a}<{b}"] = int((certified[a] < certified[b]).sum())
    return {"n_items": n, "n_models": m, "true_pairs": int(true_pair.sum()), "looks": int(looks.size),
            "certifiers": out, "per_look": comp}


def aggregate(runs: list[dict]) -> dict:
    agg = {k: runs[0][k] for k in ("n_items", "n_models", "true_pairs", "looks")}
    agg["orders"] = len(runs)
    agg["certifiers"] = {}
    for name in runs[0]["certifiers"]:
        rows = [r["certifiers"][name] for r in runs]
        agg["certifiers"][name] = {
            "true_certified": {f: float(np.mean([row["at"][f] for row in rows])) for f in rows[0]["at"]},
            "ever_wrong": float(np.mean([row["ever_wrong"] for row in rows])),
        }
    agg["per_look"] = {k: int(sum(r["per_look"][k] for r in runs)) for k in runs[0]["per_look"]}
    agg["per_look"]["looks_total"] = int(sum(r["looks"] for r in runs))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, default=50)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--bet", default="mixture")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("../results/real_multiplicity"))
    args = ap.parse_args()

    started = time.time()
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "results": {}}
    for b in args.benchmarks:
        for s in args.subsets:
            t0 = time.time()
            # the same seeds as experiments/real_leaderboard.py, so the item orders coincide
            seeds = np.random.SeedSequence([args.seed, BENCHMARKS.index(b), SUBSETS.index(s)]).generate_state(args.orders)
            jobs = [(int(sd), b, s, args.alpha, args.every_frac, args.bet) for sd in seeds]
            if args.workers > 1:
                with ProcessPoolExecutor(args.workers) as pool:
                    runs = list(pool.map(one_order, jobs, chunksize=max(1, args.orders // (args.workers * 4))))
            else:
                runs = [one_order(j) for j in jobs]
            result["results"][f"{b}/{s}"] = aggregate(runs)
            print(f"{b}/{s}: {time.time() - t0:.1f}s", flush=True)
    result["seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"real_multiplicity_orders{args.orders}.json"
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
