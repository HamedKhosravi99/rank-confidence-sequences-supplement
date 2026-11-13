"""How the spread of the abilities affects certification (Table 6 and the spread sentences of E2).

Twenty models evenly spaced through the top 20, 40, 80, 160 and all 395 ranks of the leaderboard
score (the ranking behind the paper's subsets). Span 20 is the top-20 subset and span 395 the
spread subset of Figure 2; their seeds are those of real_leaderboard.py, so those two rows
reproduce the paper. Spans 40, 80 and 160 get their own seeds. Ours (shortcut, mixture bet,
finite-benchmark offset) is checked every 1%.

Recorded per order: certified true dominances, tiers and width at the checkpoints, ever wrong,
and for every true pair the first look at which it is certified (censored at the end if never),
for the pair-level scale rho / (N Delta^2) computed on the whole benchmark.

Usage:  python -m experiments.ability_spread --smoke
        python -m experiments.ability_spread --orders 50 --workers 3
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.real_leaderboard import BENCHMARKS, FRACTIONS, SUBSETS, _summary, leaderboard_score, load
from rcs import RankConfidenceSequence

SPANS = (20, 40, 80, 160, 395)


def panel(span: int) -> np.ndarray:
    order = np.argsort(-leaderboard_score(), kind="stable")
    return order[np.linspace(0, span - 1, 20).round().astype(int)]


def seed_key(seed: int, benchmark: str, span: int) -> list[int]:
    if span == 20:
        return [seed, BENCHMARKS.index(benchmark), SUBSETS.index("top20")]
    if span == 395:
        return [seed, BENCHMARKS.index(benchmark), SUBSETS.index("spread20")]
    return [seed, BENCHMARKS.index(benchmark), 100 + span]


def one_order(args: tuple) -> dict:
    index, seed, benchmark, span, alpha, every_frac = args
    scores, _ = load(benchmark)
    x_full = scores[:, panel(span)]
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = set(int(t) for t in np.arange(every, n + 1, every)) | set(check) | {n}
    seq = RankConfidenceSequence(m, alpha, population_size=n, method="shortcut")
    first = np.full((m, m), np.nan)
    out = {"order": index, "seed": int(seed), "benchmark": benchmark, "span": span, "n_items": n,
           "true_pairs": int(true_pair.sum()), "ever_wrong": False, "checkpoints": {}}
    for t in range(n):
        seq.observe(x[t])
        if t + 1 in looks:
            rep = seq.report()
            out["ever_wrong"] = out["ever_wrong"] or rep.failed or bool((rep.dominance & false_pair).any())
            first[np.isnan(first) & rep.dominance] = (t + 1) / n
            if t + 1 in check:
                out["checkpoints"][str(check[t + 1])] = _summary(rep.dominance, true_pair, false_pair)
    out["first_certified"] = np.where(np.isnan(first), -1.0, first)[true_pair].round(4).tolist()
    return out


def pair_table(benchmark: str, span: int) -> dict:
    """Whole-benchmark Delta and rho = E[Z^2] for every true pair, in the order of first_certified."""
    scores, _ = load(benchmark)
    x = scores[:, panel(span)]
    theta = x.mean(axis=0)
    true_pair = theta[:, None] > theta[None, :]
    z2 = ((x[:, :, None] - x[:, None, :]) ** 2).mean(axis=0)
    gap = theta[:, None] - theta[None, :]
    return {"delta": gap[true_pair].tolist(), "rho": z2[true_pair].tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, default=50)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--spans", nargs="+", type=int, default=list(SPANS))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../results/ability_spread"))
    args = ap.parse_args()
    if args.smoke:
        args.orders, args.benchmarks, args.spans = 1, ["truthfulqa"], [20, 80]

    started = time.time()
    jobs = []
    for b in args.benchmarks:
        for span in args.spans:
            seeds = np.random.SeedSequence(seed_key(args.seed, b, span)).generate_state(args.orders)
            jobs += [(i, int(sd), b, span, args.alpha, args.every_frac) for i, sd in enumerate(seeds)]
    size = {b: load(b)[0].shape[0] for b in args.benchmarks}
    jobs.sort(key=lambda j: -size[j[2]])
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            runs = list(pool.map(one_order, jobs, chunksize=2))
    else:
        runs = [one_order(j) for j in jobs]
    pairs = {f"{b}/{span}": pair_table(b, span) for b in args.benchmarks for span in args.spans}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ("spread_smoke.json" if args.smoke else f"spread_orders{args.orders}.json")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    path.write_text(json.dumps({"config": config, "pairs": pairs, "runs": runs, "seconds": time.time() - started}))
    print("wrote", path, f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
