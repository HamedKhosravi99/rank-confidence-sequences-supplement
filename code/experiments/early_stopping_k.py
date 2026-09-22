"""E5, cost against k: evaluation cost of certifying every model's top-k status, as k varies.

Same data, rules and procedures as experiments/early_stopping.py (finite benchmark, retirement
once a model's top-k status is certified), but the target k is swept over a grid while the
simulated benchmark is held fixed within each replicate, so the curves are comparable across k.

Recorded per setting, per k, per procedure: mean cost in model-item evaluations relative to a
full run, averaged over all replicates whether or not the goal was reached (with its standard
error), the same cost conditional on the goal being reached and on its not being reached
(diagnostics), the probability that the goal (every model's top-k status certified) was
reached, and the probability that some report was wrong. A retired model is charged nothing
after its retirement and is never reactivated (run_rule in early_stopping.py).

Usage:  python -m experiments.early_stopping_k --reps 200 --workers 8
        python -m experiments.plot_cost_k ../results/early_stopping/early_stopping_k_reps200_n5000.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.early_stopping import SETTING_ORDER, SETTINGS, run_rule
from experiments.peeking import simulate

K_GRID = {
    "spread6": [1, 2, 3, 4, 5],
    "board20": [1, 2, 3, 4, 5, 7, 10, 13, 16, 19],
    "tied6": [1, 2, 3, 4, 5],
}
RULE = "topk"


def one_replicate(args: tuple) -> dict:
    seed, abilities, ks, method, n_items, alpha, every, procedures = args
    rng = np.random.default_rng(seed)
    x = simulate(rng, np.asarray(abilities, dtype=float), n_items)
    theta = x.mean(axis=0)
    return {(proc, k): run_rule(x, theta, RULE, k, method, proc, alpha, every)
            for proc in procedures for k in ks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-items", type=int, default=5000)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bets", nargs="+", default=["mixture"])
    ap.add_argument("--pocock", nargs="*", default=["pocock", "pocock_bonf"])
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--settings", nargs="+", default=["spread6", "board20"], choices=list(SETTINGS))
    ap.add_argument("--out", type=Path, default=Path("../results/early_stopping"))
    args = ap.parse_args()

    started = time.time()
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "rule": RULE, "settings": {}}
    for name in args.settings:
        t0 = time.time()
        abilities, _, method = SETTINGS[name]
        ks = K_GRID[name]
        # the same seeds as experiments/early_stopping.py, so the simulated benchmarks coincide
        seeds = np.random.SeedSequence([args.seed, SETTING_ORDER.index(name)]).generate_state(args.reps)
        procedures = tuple(f"ours/{b}" for b in args.bets) + tuple(args.pocock)
        jobs = [(int(s), abilities, ks, method, args.n_items, args.alpha, args.every, procedures) for s in seeds]
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                reps = list(pool.map(one_replicate, jobs, chunksize=max(1, args.reps // (args.workers * 8))))
        else:
            reps = [one_replicate(j) for j in jobs]
        m = len(abilities)
        summary = {"abilities": abilities, "method": method, "n_models": m, "ks": ks, "procedures": {}}
        for proc in procedures:
            rows_by_k = []
            for k in ks:
                rows = [r[(proc, k)] for r in reps]
                cost = np.array([r["cost"] for r in rows])
                goal = np.array([r["goal_topk"] for r in rows], dtype=bool)
                rows_by_k.append({
                    "k": k,
                    "cost_mean": float(cost.mean()),  # over all runs, goal reached or not
                    "cost_se": float(cost.std(ddof=1) / np.sqrt(len(rows))),
                    # diagnostics: the same cost conditional on the goal being reached, and on not
                    "cost_goal_mean": float(cost[goal].mean()) if goal.any() else None,
                    "cost_nogoal_mean": float(cost[~goal].mean()) if (~goal).any() else None,
                    "n_goal": int(goal.sum()),
                    "goal_topk": float(np.mean([r["goal_topk"] for r in rows])),
                    "wrong": float(np.mean([r["wrong"] for r in rows])),
                    "items_seen_mean": float(np.mean([r["items_seen"] for r in rows])),
                })
            summary["procedures"][proc] = rows_by_k
        result["settings"][name] = summary
        print(f"{name}: {time.time() - t0:.1f}s", flush=True)
    result["seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"early_stopping_k_reps{args.reps}_n{args.n_items}.json"
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
