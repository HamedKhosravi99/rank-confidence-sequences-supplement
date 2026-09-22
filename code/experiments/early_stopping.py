"""E5. How much evaluation does it take to certify a leaderboard, if models can be retired early?

Setting: a fixed benchmark of N items (finite-benchmark estimand; the truth is the ranking by
full-benchmark accuracy), M models, items evaluated in a uniformly random order. Cost is counted
in model-item evaluations; a full run costs M * N. Every ``--every`` items the procedure reports,
and a rule decides which models stop being evaluated. Retirement times are stopping times, so
every report stays valid (Proposition 5); the experiment checks this and measures the saving.

Two goals and their rules:

  goal "top-k": certify, for every model, whether its rank is <= k.
    none           evaluate every model on every item
    stop_at_goal   no per-model retirement; stop everything once the goal is reached
    topk           retire a model as soon as its own top-k status is certified
    topk_safe      ... and, in addition, every pair between it and a model whose status is still
                   open is certified (so its retirement cannot block anybody else's certification)

  goal "full ranking": certify every pair.
    resolved       retire a model once every pair involving it is certified

Procedures: ours (RankConfidenceSequence, with the bets given by --bets) and the group-sequential
Pocock design of Arviv et al. (2026) on the same data and rules: K = 10 pre-specified looks, a
one-sided paired z-test per pair against the Pocock boundary, no multiplicity correction
("pocock") or a Bonferroni correction over the M(M-1) pairs ("pocock_bonf"); certifications are
permanent and closed under transitivity. The Pocock design is only valid at its pre-specified
looks and only asymptotically; "wrong" measures what that costs in practice.

Reported per procedure and rule: cost as a fraction of M * N averaged over all replicates
(goal reached or not), the same cost conditional on the rule's goal being reached and on its not
being reached (diagnostics), the fraction of runs in which each goal is reached by the end, the
number of true dominances certified, and how often some report was wrong. Retirement is
permanent for every procedure: ``retire`` clears the model's active flag, ``observe`` charges
only active models, and nothing sets the flag again.

Usage:  python -m experiments.early_stopping --reps 500 --workers 10 --out ../results/early_stopping
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.peeking import simulate
from rcs import RankConfidenceSequence
from rcs.baselines import PocockSequence

SETTINGS = {
    # name: (abilities, k, certification method)
    "spread6": (np.linspace(1.0, 0.0, 6).tolist(), 2, "exact"),
    "board20": (np.linspace(1.5, -0.5, 20).tolist(), 3, "shortcut"),
    "tied6": ([1.0, 1.0, 0.6, 0.4, 0.2, 0.0], 2, "exact"),  # two tied leaders: separating them is an error
}
SETTING_ORDER = ("spread6", "board20", "tied6")
RULES = ("none", "stop_at_goal", "topk", "topk_safe", "resolved")


def _to_retire(rule: str, report, k: int) -> np.ndarray:
    d = report.dominance
    m = d.shape[0]
    status_known = (report.upper <= k) | (report.lower > k)
    settled_pair = d | d.T | np.eye(m, dtype=bool)
    if rule == "none":
        return np.zeros(m, dtype=bool)
    if rule == "stop_at_goal":
        return np.full(m, bool(status_known.all()))
    if rule == "topk":
        return status_known
    if rule == "topk_safe":
        open_models = ~status_known
        return status_known & settled_pair[:, open_models].all(axis=1)
    if rule == "resolved":
        return settled_pair.all(axis=1)
    raise ValueError(rule)


def make_procedure(name: str, m: int, n: int, alpha: float, method: str, looks: int):
    """``name`` is "ours/<bet>", "pocock" or "pocock_bonf"."""
    if name.startswith("ours/"):
        return RankConfidenceSequence(m, alpha, population_size=n, method=method, bet=name.split("/", 1)[1])
    if name == "pocock":
        return PocockSequence(m, alpha, n, looks=looks)
    if name == "pocock_bonf":
        return PocockSequence(m, alpha, n, looks=looks, bonferroni=True)
    raise ValueError(name)


def run_rule(x: np.ndarray, theta: np.ndarray, rule: str, k: int, method: str, procedure: str,
             alpha: float, every: int, looks: int = 10) -> dict:
    n, m = x.shape
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    true_pair = theta[:, None] > theta[None, :]
    seq = make_procedure(procedure, m, n, alpha, method, looks)
    wrong = False
    for t in range(n):
        if seq.active.sum() <= 1:  # a lone model has nobody left to be compared with
            break
        seq.observe(np.where(seq.active, x[t], np.nan))  # retired models are not evaluated
        if (t + 1) % every and t + 1 < n:
            continue
        report = seq.report()
        wrong = wrong or bool((report.dominance & false_pair).any()) or report.failed
        for j in np.flatnonzero(_to_retire(rule, report, k) & seq.active):
            seq.retire(int(j))
    report = seq.report()
    d = report.dominance
    status_known = (report.upper <= k) | (report.lower > k)
    return {
        "cost": seq.evaluations / (m * n),
        "goal_topk": bool(status_known.all()),
        "goal_full": bool((d | d.T | np.eye(m, dtype=bool)).all()),
        "status_known": int(status_known.sum()),
        "true_certified": int((d & true_pair).sum()),
        "wrong": wrong,
        "items_seen": seq.t,
    }


def one_replicate(args: tuple) -> dict:
    seed, abilities, k, method, n_items, alpha, every, procedures = args
    rng = np.random.default_rng(seed)
    x = simulate(rng, np.asarray(abilities, dtype=float), n_items)  # rows are exchangeable: a random order
    theta = x.mean(axis=0)  # the finite-benchmark truth
    return {(proc, rule): run_rule(x, theta, rule, k, method, proc, alpha, every)
            for proc in procedures for rule in RULES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--n-items", type=int, default=5000)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bets", nargs="+", default=["mixture", "fixed"])
    ap.add_argument("--pocock", nargs="*", default=["pocock", "pocock_bonf"],
                    help="group-sequential competitors to include (pass nothing to skip)")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--out", type=Path, default=Path("../results/early_stopping"))
    args = ap.parse_args()

    started = time.time()
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "settings": {}}
    for name in args.settings:
        t0 = time.time()
        abilities, k, method = SETTINGS[name]
        seeds = np.random.SeedSequence([args.seed, SETTING_ORDER.index(name)]).generate_state(args.reps)
        procedures = tuple(f"ours/{b}" for b in args.bets) + tuple(args.pocock)
        jobs = [(int(s), abilities, k, method, args.n_items, args.alpha, args.every, procedures)
                for s in seeds]
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                reps = list(pool.map(one_replicate, jobs, chunksize=max(1, args.reps // (args.workers * 8))))
        else:
            reps = [one_replicate(j) for j in jobs]
        m = len(abilities)
        summary = {"abilities": abilities, "k": k, "method": method, "n_models": m,
                   "pairs": m * (m - 1) // 2, "rules": {}}
        for proc in procedures:
            for rule in RULES:
                rows = [r[(proc, rule)] for r in reps]
                cost = np.array([r["cost"] for r in rows])
                goal = np.array([r["goal_topk" if rule != "resolved" else "goal_full"] for r in rows], dtype=bool)
                summary["rules"][f"{proc}/{rule}"] = {
                    "cost_mean": float(cost.mean()),  # over all runs, goal reached or not
                    "cost_se": float(cost.std(ddof=1) / np.sqrt(len(rows))),
                    "cost_goal_mean": float(cost[goal].mean()) if goal.any() else None,
                    "cost_nogoal_mean": float(cost[~goal].mean()) if (~goal).any() else None,
                    "n_goal": int(goal.sum()),
                    "goal_topk": float(np.mean([r["goal_topk"] for r in rows])),
                    "goal_full": float(np.mean([r["goal_full"] for r in rows])),
                    "status_known_mean": float(np.mean([r["status_known"] for r in rows])),
                    "true_certified_mean": float(np.mean([r["true_certified"] for r in rows])),
                    "wrong": float(np.mean([r["wrong"] for r in rows])),
                    "items_seen_mean": float(np.mean([r["items_seen"] for r in rows])),
                }
        result["settings"][name] = summary
        print(f"{name}: {time.time() - t0:.1f}s", flush=True)
    result["seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"early_stopping_reps{args.reps}_n{args.n_items}.json"
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
