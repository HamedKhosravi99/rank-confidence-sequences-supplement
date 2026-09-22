"""E3b. Does the choice of bet buy power? Same data, same certifier (exact, uniform weights).

  fixed      lam = 0.25 throughout (the default so far)
  agrapa50   plug-in Kelly bet mean(Y)/mean(Y^2), capped at 0.50
  agrapa75   the same, capped at 0.75
  ons        online Newton step on [0, 1/2]
  mixture    average of the wealths of fixed bets 0.03, 0.06, 0.12, 0.25, 0.5

A fixed bet has NEGATIVE growth when it exceeds about twice the log-optimal bet, which is roughly
gap / discordance rate: with lam = 0.25 and a gap of 0.02 the wealth tends to zero and the pair is
never certified. The ``close_race`` scenario (adjacent gaps of about 0.02, long horizon) tests this.

All bets are predictable, so validity is untouched (Lemma 1); only power can differ. Reported per
scenario: mean number of TRUE dominances certified after n items, and the probability that
some report is ever wrong.

Usage:  python -m experiments.bets --reps 1000 --workers 10 --out ../results/bets
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.peeking import simulate, true_theta
from experiments.weighting import SCENARIOS as _BASE
from rcs import ExactCertifier, log_wealth_paths

# order is fixed: a scenario's seed stream is its position in this tuple
SCENARIO_ORDER = ("near_ties", "even_spread", "tied_leaders", "all_tied", "close_race")
SCENARIOS = {**_BASE, "close_race": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]}

BET_CONFIGS = {
    "fixed": {"bet": "fixed", "lam": 0.25},
    "agrapa50": {"bet": "agrapa", "bet_cap": 0.50},
    "agrapa75": {"bet": "agrapa", "bet_cap": 0.75},
    "ons": {"bet": "ons"},
    "mixture": {"bet": "mixture"},
}


def one_replicate(args: tuple) -> dict:
    seed, abilities, n_max, every, checkpoints, alpha = args
    abilities = np.asarray(abilities, dtype=float)
    theta = true_theta(abilities)
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(len(theta), dtype=bool)
    x = simulate(np.random.default_rng(seed), abilities, n_max)
    times = np.arange(every, n_max + 1, every)
    out = {}
    for name, cfg in BET_CONFIGS.items():
        paths = log_wealth_paths(x, at=times, **cfg)
        cert = ExactCertifier(x.shape[1], alpha)
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
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--out", type=Path, default=Path("../results/bets"))
    args = ap.parse_args()
    if any(c % args.every or c > args.n_max for c in args.checkpoints):
        raise SystemExit("checkpoints must be multiples of --every and at most --n-max")

    started = time.time()
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "bets": BET_CONFIGS, "scenarios": {}}
    for scenario in args.scenarios:
        t0 = time.time()
        abilities = SCENARIOS[scenario]
        seeds = np.random.SeedSequence([args.seed, SCENARIO_ORDER.index(scenario)]).generate_state(args.reps)
        jobs = [(int(s), abilities, args.n_max, args.every, tuple(args.checkpoints), args.alpha) for s in seeds]
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                reps = list(pool.map(one_replicate, jobs, chunksize=max(1, args.reps // (args.workers * 8))))
        else:
            reps = [one_replicate(j) for j in jobs]
        theta = true_theta(np.asarray(abilities, dtype=float))
        summary = {"abilities": abilities, "theta": theta.tolist(),
                   "true_pairs": int((theta[:, None] > theta[None, :]).sum()), "methods": {}}
        for name in BET_CONFIGS:
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
    path = args.out / f"bets_{'-'.join(args.scenarios)}_reps{args.reps}_n{args.n_max}.json"
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
