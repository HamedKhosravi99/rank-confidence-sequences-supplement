"""Which bet, on the real leaderboard (Table 16).

The bets of Appendix E.6 on the real top-20 and spread-20 subsets of Figure 2, on the same item
orders (real_leaderboard.py seeds), with the finite-benchmark offset and the shortcut certifier for
every bet, so only the bet differs:

  fixed@lam   each value of the default grid alone, lam in {0.03, 0.06, 0.12, 0.25, 0.5};
  mixture     the default five-point grid (the paper's method);
  dense       a uniform mixture over 50 values evenly spaced in [0.01, 0.5], a finite-grid
              approximation to a continuous mixture;
  agrapa      the plug-in bet of Remark B.3 (approximate GRAPA), capped at 0.75;
  ons         online Newton steps on [0, 1/2].

Every bet is predictable (it uses only past items), so validity is Lemma B.1 for all of them.
Recorded per order: certified true dominances, tiers and width at the checkpoints, whether any
report was ever wrong, and the cost of the top-3 goal with retirement.

Usage:  python -m experiments.bets_real --smoke
        python -m experiments.bets_real --orders 50 --workers 3
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.real_leaderboard import BENCHMARKS, FRACTIONS, SUBSETS, _summary, choose, leaderboard_score, load
from rcs import RankConfidenceSequence
from rcs.wealth import MIXTURE_GRID

FIXED = MIXTURE_GRID
DENSE_GRID = tuple(np.linspace(0.01, 0.5, 50))
VARIANTS = tuple(f"fixed@{lam}" for lam in FIXED) + ("mixture", "dense", "agrapa", "ons")


def make(variant: str, m: int, n: int, alpha: float, method: str) -> RankConfidenceSequence:
    if variant.startswith("fixed@"):
        return RankConfidenceSequence(m, alpha, population_size=n, method=method, bet="fixed",
                                      lam=float(variant.split("@")[1]))
    if variant == "dense":
        seq = RankConfidenceSequence(m, alpha, population_size=n, method=method, bet="mixture")
        bettor = seq.wealth._bettor  # same mixture update, on a denser grid, set before any item
        bettor._grid = np.asarray(DENSE_GRID)[:, None, None]
        bettor._grid_log_wealth = np.zeros((len(DENSE_GRID), m, m))
        return seq
    return RankConfidenceSequence(m, alpha, population_size=n, method=method, bet=variant)


def one_order(args: tuple) -> dict:
    index, seed, benchmark, subset, alpha, every_frac, k = args
    scores, _ = load(benchmark)
    x_full = scores[:, choose(subset, leaderboard_score())]
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]  # the first draw of real_leaderboard.py: the same orders
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = set(int(t) for t in np.arange(every, n + 1, every)) | set(check) | {n}
    out = {"order": index, "seed": int(seed), "benchmark": benchmark, "subset": subset, "n_items": n,
           "true_pairs": int(true_pair.sum()), "variants": {}}
    for v in VARIANTS:
        seq = make(v, m, n, alpha, "shortcut")
        rec = {"ever_wrong": False, "checkpoints": {}}
        for t in range(n):
            seq.observe(x[t])
            if t + 1 in looks:
                rep = seq.report()
                rec["ever_wrong"] = rec["ever_wrong"] or rep.failed or bool((rep.dominance & false_pair).any())
                if t + 1 in check:
                    rec["checkpoints"][str(check[t + 1])] = _summary(rep.dominance, true_pair, false_pair)
        seq = make(v, m, n, alpha, "shortcut")
        for t in range(n):
            if seq.active.sum() <= 1:
                break
            seq.observe(np.where(seq.active, x[t], np.nan))
            if (t + 1) % every == 0 or t + 1 == n:
                rep = seq.report()
                known = (rep.upper <= k) | (rep.lower > k)
                for j in np.flatnonzero(known & seq.active):
                    seq.retire(int(j))
        rep = seq.report()
        known = (rep.upper <= k) | (rep.lower > k)
        rec["retire_topk"] = {"cost": seq.evaluations / (m * n), "goal": bool(known.all()),
                              "settled_frac": float(known.mean())}
        out["variants"][v] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, default=50)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--subsets", nargs="+", default=["top20", "spread20"], choices=list(SUBSETS))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../results/bets_real"))
    args = ap.parse_args()
    if args.smoke:
        args.orders, args.benchmarks, args.subsets = 1, ["truthfulqa"], ["top20"]

    started = time.time()
    jobs = []
    for b in args.benchmarks:
        for s in args.subsets:
            seeds = np.random.SeedSequence([args.seed, BENCHMARKS.index(b), SUBSETS.index(s)]).generate_state(args.orders)
            jobs += [(i, int(sd), b, s, args.alpha, args.every_frac, args.k) for i, sd in enumerate(seeds)]
    size = {b: load(b)[0].shape[0] for b in args.benchmarks}
    jobs.sort(key=lambda j: -size[j[2]])
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            runs = list(pool.map(one_order, jobs, chunksize=2))
    else:
        runs = [one_order(j) for j in jobs]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ("bets_smoke.json" if args.smoke else f"bets_orders{args.orders}.json")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    path.write_text(json.dumps({"config": config, "variants": list(VARIANTS), "dense_grid": list(DENSE_GRID),
                                "runs": runs, "seconds": time.time() - started}))
    print("wrote", path, f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
