"""Figure 2 under both sampling models, with e-Bonferroni (Figures 2 and 3, Tables 10 and 11).

Under (S) the population is the empirical distribution of the benchmark's item score vectors:
items are drawn with replacement, whole score vectors at a time, so the dependence between models
on an item is kept. The estimand is each model's mean under that population, which is its mean on
the whole benchmark. Ours uses the (S) wealth (offset b = 0, ``population_size=None``), which is
valid under (S). The fixed-sample procedures are the same as in Figure 2, on the same item streams.

e-Bonferroni with transitive closure is computed from the same wealths as ours at every look,
under both designs.

Design (F) repeats real_leaderboard.py with its seeds and the same random-number calls, so ours,
Holm and Romano-Wolf reproduce real_leaderboard_orders50.json order by order, with e-Bonferroni
added. Design (S) uses its own seeds. Output is per order.

Usage:  python -m experiments.real_leaderboard_sampling --smoke
        python -m experiments.real_leaderboard_sampling --orders 50 --workers 3
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.real_leaderboard import (BENCHMARKS, FRACTIONS, SUBSETS, _fixed_n_at, _summary, choose,
                                          leaderboard_score, load, romano_wolf)
from rcs import RankConfidenceSequence
from rcs.certify import BonferroniCertifier, transitive_closure

DESIGNS = ("F", "S")


def one_order(args: tuple) -> dict:
    design, index, seed, benchmark, subset, alpha, every_frac = args
    scores, _ = load(benchmark)
    idx = choose(subset, leaderboard_score())
    x_full = scores[:, idx]
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    rng = np.random.default_rng(seed)
    if design == "F":
        x = x_full[rng.permutation(n)]
        population = n
    else:
        x = x_full[rng.integers(0, n, size=n)]
        population = None
    every = max(1, int(round(every_frac * n)))
    looks = np.unique(np.concatenate([np.arange(every, n + 1, every), [n]]))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = np.unique(np.concatenate([looks, list(check)]))
    method = "exact" if m <= 8 else "shortcut"

    out = {"design": design, "order": index, "seed": int(seed), "benchmark": benchmark, "subset": subset,
           "n_items": n, "n_models": m, "true_pairs": int(true_pair.sum()),
           "ours": {}, "ebonf": {}, "fixed_n": {}, "fixed_z": {}, "fixed_rw": {},
           "ever_wrong": {"ours": False, "ebonf": False}}
    seq = RankConfidenceSequence(m, alpha, population_size=population, method=method)
    ebonf = BonferroniCertifier(m, alpha)
    look_set = set(int(t) for t in looks)
    for t in range(n):
        seq.observe(x[t])
        if t + 1 not in look_set:
            continue
        rep = seq.report()
        d_e = ebonf.update(seq.wealth.log_wealth)
        for name, d, failed in (("ours", rep.dominance, rep.failed), ("ebonf", d_e, ebonf.failed)):
            wrong = failed or bool((d & false_pair).any())
            out["ever_wrong"][name] = out["ever_wrong"][name] or wrong
            if t + 1 in check:
                s = _summary(d, true_pair, false_pair)
                s["ever_wrong_so_far"] = out["ever_wrong"][name]
                out[name][str(check[t + 1])] = s
    fixed = _fixed_n_at(x, looks, alpha)
    for name in ("fixed_n", "fixed_z"):
        ever = False
        for i, t in enumerate(looks):
            d = transitive_closure(fixed[name][i])
            ever = ever or bool((d & false_pair).any())
            if int(t) in check:
                s = _summary(d, true_pair, false_pair)
                s["wrong_at_this_look"] = bool((d & false_pair).any())
                s["ever_wrong_so_far"] = ever
                out[name][str(check[int(t)])] = s
    ever = False
    for t in sorted(check):
        d = transitive_closure(romano_wolf(x[:t], alpha, rng))
        ever = ever or bool((d & false_pair).any())
        s = _summary(d, true_pair, false_pair)
        s["wrong_at_this_look"] = bool((d & false_pair).any())
        s["ever_wrong_so_far"] = ever
        out["fixed_rw"][str(check[t])] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, default=50)
    ap.add_argument("--designs", nargs="+", default=list(DESIGNS), choices=list(DESIGNS))
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--subsets", nargs="+", default=["top20", "spread20"], choices=list(SUBSETS))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../results/real_leaderboard_sampling"))
    args = ap.parse_args()
    if args.smoke:
        args.orders, args.benchmarks, args.subsets = 2, ["truthfulqa"], ["top20"]

    started = time.time()
    jobs = []
    for design in args.designs:
        for b in args.benchmarks:
            for s in args.subsets:
                key = [args.seed, BENCHMARKS.index(b), SUBSETS.index(s)] + ([] if design == "F" else [1])
                seeds = np.random.SeedSequence(key).generate_state(args.orders)
                jobs += [(design, i, int(sd), b, s, args.alpha, args.every_frac) for i, sd in enumerate(seeds)]
    size = {b: load(b)[0].shape[0] for b in args.benchmarks}
    jobs.sort(key=lambda j: -size[j[3]])
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ("fig2_smoke.json" if args.smoke else f"fig2_orders{args.orders}.json")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            runs = list(pool.map(one_order, jobs, chunksize=4))
    else:
        runs = [one_order(j) for j in jobs]
    path.write_text(json.dumps({"config": config, "runs": runs, "seconds": time.time() - started}))
    print("wrote", path, f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
