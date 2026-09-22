"""E3. Where does power come from? Compare certifiers that share the same pairwise wealths.

All certifiers see identical wealth paths, so differences are due to the multiplicity step alone:

  e_bonferroni     threshold M(M-1)/alpha per pair (+ free transitive closure)
  shortcut         transitivity pooling (Algorithm 2)
  exact_uniform    restricted closed test, arithmetic mean over T(W)         (paper default)
  exact_adjacent   restricted closed test, mean over same/adjacent-level pairs of T(W)
  exact_mixed      average of the two weightings

Reported per scenario: the mean number of TRUE dominances certified after n items, for several n,
and the probability that some report is ever wrong. Monitoring is every ``--every`` items.

Usage:  python -m experiments.weighting --reps 1000 --workers 10 --out ../results/weighting
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.peeking import simulate, true_theta
from rcs import BonferroniCertifier, ExactCertifier, ShortcutCertifier, log_wealth_paths

SCENARIOS = {
    "near_ties": [1.00, 0.98, 0.62, 0.60, 0.22, 0.20],
    "even_spread": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    "tied_leaders": [1.0, 1.0, 0.6, 0.6, 0.2, -0.2],
    "all_tied": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
}
METHODS = ("e_bonferroni", "shortcut", "exact_uniform", "exact_adjacent", "exact_mixed")


def _certifiers(m: int, alpha: float) -> dict:
    return {
        "e_bonferroni": BonferroniCertifier(m, alpha),
        "shortcut": ShortcutCertifier(m, alpha),
        "exact_uniform": ExactCertifier(m, alpha, weighting="uniform"),
        "exact_adjacent": ExactCertifier(m, alpha, weighting="adjacent"),
        "exact_mixed": ExactCertifier(m, alpha, weighting="mixed"),
    }


def one_replicate(args: tuple) -> dict:
    seed, abilities, n_max, every, checkpoints, alpha, lam, bet = args
    abilities = np.asarray(abilities, dtype=float)
    theta = true_theta(abilities)
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(len(theta), dtype=bool)
    x = simulate(np.random.default_rng(seed), abilities, n_max)
    times = np.arange(every, n_max + 1, every)
    paths = log_wealth_paths(x, bet=bet, lam=lam, at=times)
    out = {}
    for name, cert in _certifiers(x.shape[1], alpha).items():
        counts, wrong = {}, False
        for t, lw in zip(times, paths):
            d = cert.update(lw)
            wrong = wrong or bool((d & false_pair).any())
            if t in checkpoints:
                counts[int(t)] = int((d & true_pair).sum())
        out[name] = (counts, wrong)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--n-max", type=int, default=4000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bet", default="mixture", choices=["fixed", "agrapa", "ons", "mixture"])
    ap.add_argument("--lam", type=float, default=0.25, help="used only with --bet fixed")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--out", type=Path, default=Path("../results/weighting"))
    args = ap.parse_args()
    if any(c % args.every or c > args.n_max for c in args.checkpoints):
        raise SystemExit("checkpoints must be multiples of --every and at most --n-max")

    started = time.time()
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "scenarios": {}}
    for scenario in args.scenarios:
        t0 = time.time()
        abilities = SCENARIOS[scenario]
        seeds = np.random.SeedSequence([args.seed, sorted(SCENARIOS).index(scenario)]).generate_state(args.reps)
        jobs = [(int(s), abilities, args.n_max, args.every, tuple(args.checkpoints), args.alpha, args.lam, args.bet)
                for s in seeds]
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                reps = list(pool.map(one_replicate, jobs, chunksize=max(1, args.reps // (args.workers * 8))))
        else:
            reps = [one_replicate(j) for j in jobs]
        theta = true_theta(np.asarray(abilities, dtype=float))
        summary = {"abilities": abilities, "theta": theta.tolist(),
                   "true_pairs": int((theta[:, None] > theta[None, :]).sum()), "methods": {}}
        for name in METHODS:
            wrong = float(np.mean([r[name][1] for r in reps]))
            summary["methods"][name] = {
                "ever_wrong": wrong,
                "ever_wrong_se": float(np.sqrt(wrong * (1 - wrong) / args.reps)),
                "true_certified": {
                    str(c): {
                        "mean": float(np.mean([r[name][0][c] for r in reps])),
                        "se": float(np.std([r[name][0][c] for r in reps], ddof=1) / np.sqrt(args.reps)),
                    }
                    for c in args.checkpoints
                },
            }
        result["scenarios"][scenario] = summary
        print(f"{scenario}: {time.time() - t0:.1f}s", flush=True)
    result["seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"weighting_{args.bet}_reps{args.reps}_n{args.n_max}.json"
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
