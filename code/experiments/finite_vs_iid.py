"""Does knowing that the benchmark is finite accelerate certification?

Three runs of the method on the very same item stream, identical in everything but the offset of
the betting update:

  finite    the finite-benchmark bound b_t of Proposition 1(a)                (the paper's method)
  baseline  max(0, b_t): valid for the same fixed-benchmark target, since a larger offset is still
            an upper bound on the conditional mean, but it never exploits a negative drift (the
            leader ahead so far); ours against it isolates what the finite benchmark buys in power
            with both procedures valid for the same target                    (primary comparison)
  iid       b_t = 0, the superpopulation update applied to the finite benchmark as if its items
            were an i.i.d. stream; NOT valid for the fixed-benchmark target when the leader is
            behind so far, so an algorithmic ablation only

Same order, models, differences, mixture bet, alpha, multiplicity procedure, closure, reporting
and retirement rule. Truth: the ranking by full-benchmark accuracy.

Recorded at checkpoint fractions of the benchmark, per order or replicate:
  true_frac     fraction of the true pairwise orderings certified
  width         mean rank-interval width (U - L + 1) over models
  tiers         number of certified tiers
and for the whole run: whether any report was wrong, the cost (fraction of M * N model-item
evaluations) and success of certifying every model's top-k status under the ``topk`` retirement
rule, and, per pair, the first look at which it was certified under each construction together
with its full-benchmark margin |theta_j - theta_l|.

Real data: the six Open LLM Leaderboard benchmarks, top-20 and spread-20 subsets, the 50 item
orders of experiments/real_leaderboard.py (same seeds). Simulation: six models, N = 5000 binary
items with shared difficulty, three separation regimes, 1000 replicates.

Usage:  python -m experiments.finite_vs_iid --orders 50 --reps 1000 --workers 8
        python -m experiments.finite_vs_iid --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.peeking import simulate, true_theta
from experiments.real_leaderboard import BENCHMARKS, SUBSETS, choose, leaderboard_score, load
from rcs import RankConfidenceSequence
from rcs.report import tiers

FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00)
CONSTRUCTIONS = ("finite", "baseline", "iid")
REAL_SUBSETS = ("top20", "spread20")
SIM_REGIMES = {  # abilities on the logistic scale of experiments/peeking.simulate
    "clear_gaps": np.linspace(1.5, 0.0, 6).tolist(),
    "moderate_gaps": np.linspace(1.0, 0.0, 6).tolist(),
    "near_ties": np.linspace(0.5, 0.25, 6).tolist(),
}
MARGIN_BINS = ((0.0, 0.01, "very close"), (0.01, 0.03, "moderate"), (0.03, 9.0, "well separated"))


def _wealth_kwargs(construction: str, n: int) -> dict:
    if construction == "finite":
        return {"population_size": n}
    if construction == "baseline":
        return {"population_size": n, "bound_floor": 0.0}
    if construction == "iid":
        return {"population_size": None}
    raise ValueError(construction)


def _monitor(x: np.ndarray, construction: str, looks: np.ndarray, check: dict, alpha: float,
             method: str, theta: np.ndarray) -> dict:
    """Monitor every look; return checkpoint summaries, ever-wrong, and first certification looks."""
    n, m = x.shape
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    seq = RankConfidenceSequence(m, alpha, method=method, **_wealth_kwargs(construction, n))
    look_set = set(int(t) for t in looks)
    first = np.full((m, m), -1, dtype=int)  # item count at first certification, -1 = never
    at, ever_wrong = {}, False
    for t in range(n):
        seq.observe(x[t])
        if t + 1 not in look_set:
            continue
        rep = seq.report()
        d = rep.dominance
        ever_wrong = ever_wrong or rep.failed or bool((d & false_pair).any())
        newly = d & (first < 0)
        first[newly] = t + 1
        if t + 1 in check:
            at[str(check[t + 1])] = {
                "true_frac": float((d & true_pair).sum() / max(1, true_pair.sum())),
                "width": float(np.mean(rep.upper - rep.lower + 1)),
                "tiers": int(tiers(d).max()) if not rep.failed else 0,
            }
    return {"at": at, "ever_wrong": ever_wrong, "first": first}


def _retire_topk(x: np.ndarray, construction: str, every: int, alpha: float, method: str, k: int,
                 theta: np.ndarray) -> dict:
    n, m = x.shape
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    seq = RankConfidenceSequence(m, alpha, method=method, **_wealth_kwargs(construction, n))
    wrong = False
    for t in range(n):
        if seq.wealth.active.sum() <= 1:
            break
        seq.observe(np.where(seq.wealth.active, x[t], np.nan))
        if (t + 1) % every == 0 or t + 1 == n:
            rep = seq.report()
            wrong = wrong or rep.failed or bool((rep.dominance & false_pair).any())
            known = (rep.upper <= k) | (rep.lower > k)
            for j in np.flatnonzero(known & seq.wealth.active):
                seq.retire(int(j))
    rep = seq.report()
    known = (rep.upper <= k) | (rep.lower > k)
    return {"cost": seq.evaluations / (m * n), "goal": bool(known.all()), "wrong": wrong, "items_seen": seq.t}


def _pairs(first: dict, theta: np.ndarray, n: int) -> list:
    """Per true pair: margin, first-certification fractions under both constructions."""
    m = len(theta)
    out = []
    for j in range(m):
        for l in range(m):
            if theta[j] > theta[l]:
                rec = {"margin": float(theta[j] - theta[l])}
                for c in CONSTRUCTIONS:
                    rec[c] = float(first[c][j, l] / n) if first[c][j, l] > 0 else None
                out.append(rec)
    return out


def one_real_order(args: tuple) -> dict:
    seed, benchmark, subset, k, alpha, every_frac = args
    scores, _ = load(benchmark)
    idx = choose(subset, leaderboard_score())
    x_full = scores[:, idx]
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = np.unique(np.concatenate([np.arange(every, n + 1, every), [n], list(check)]))
    method = "exact" if m <= 8 else "shortcut"
    mon = {c: _monitor(x, c, looks, check, alpha, method, theta) for c in CONSTRUCTIONS}
    ret = {c: _retire_topk(x, c, every, alpha, method, k, theta) for c in CONSTRUCTIONS}
    return {"n_items": n, "n_models": m, "true_pairs": int((theta[:, None] > theta[None, :]).sum()),
            "at": {c: mon[c]["at"] for c in CONSTRUCTIONS},
            "ever_wrong": {c: mon[c]["ever_wrong"] for c in CONSTRUCTIONS},
            "retire": ret, "pairs": _pairs({c: mon[c]["first"] for c in CONSTRUCTIONS}, theta, n)}


def one_sim_replicate(args: tuple) -> dict:
    seed, abilities, n, k, alpha, every = args
    x = simulate(np.random.default_rng(seed), np.asarray(abilities, dtype=float), n)
    theta = x.mean(axis=0)  # the finite-benchmark truth, as in E5
    m = x.shape[1]
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = np.unique(np.concatenate([np.arange(every, n + 1, every), [n], list(check)]))
    mon = {c: _monitor(x, c, looks, check, alpha, "exact", theta) for c in CONSTRUCTIONS}
    ret = {c: _retire_topk(x, c, every, alpha, "exact", k, theta) for c in CONSTRUCTIONS}
    return {"n_items": n, "n_models": m, "true_pairs": int((theta[:, None] > theta[None, :]).sum()),
            "at": {c: mon[c]["at"] for c in CONSTRUCTIONS},
            "ever_wrong": {c: mon[c]["ever_wrong"] for c in CONSTRUCTIONS},
            "retire": ret, "pairs": _pairs({c: mon[c]["first"] for c in CONSTRUCTIONS}, theta, n)}


def _aggregate(runs: list[dict]) -> dict:
    agg = {"n_runs": len(runs), "n_items": runs[0]["n_items"], "n_models": runs[0]["n_models"],
           "true_pairs": runs[0]["true_pairs"], "at": {}, "ever_wrong": {}, "retire": {}, "pairs": {}}
    for c in CONSTRUCTIONS:
        agg["at"][c] = {}
        for f in runs[0]["at"][c]:
            rows = [r["at"][c][f] for r in runs]
            agg["at"][c][f] = {q: {"mean": float(np.mean([row[q] for row in rows])),
                                   "se": float(np.std([row[q] for row in rows], ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else 0.0}
                               for q in rows[0]}
        agg["ever_wrong"][c] = float(np.mean([r["ever_wrong"][c] for r in runs]))
        rows = [r["retire"][c] for r in runs]
        agg["retire"][c] = {q: float(np.mean([row[q] for row in rows])) for q in rows[0]}
    # pair-level: certification-time savings by margin bin
    pairs = [p for r in runs for p in r["pairs"]]
    agg["pairs"] = {}
    for other in ("baseline", "iid"):
        bins = {}
        for lo, hi, name in MARGIN_BINS:
            sel = [p for p in pairs if lo <= p["margin"] < hi]
            both = [p for p in sel if p["finite"] is not None and p[other] is not None]
            saved = [p[other] - p["finite"] for p in both]
            bins[name] = {
                "n_pairs": len(sel),
                "certified_by_both": len(both),
                "only_finite": int(sum(p["finite"] is not None and p[other] is None for p in sel)),
                "only_other": int(sum(p[other] is not None and p["finite"] is None for p in sel)),
                "neither": int(sum(p["finite"] is None and p[other] is None for p in sel)),
                "median_saving": float(np.median(saved)) if saved else None,
                "mean_saving": float(np.mean(saved)) if saved else None,
                "q25_saving": float(np.percentile(saved, 25)) if saved else None,
                "q75_saving": float(np.percentile(saved, 75)) if saved else None,
                "median_first_finite": float(np.median([p["finite"] for p in both])) if both else None,
                "median_first_other": float(np.median([p[other] for p in both])) if both else None,
            }
        agg["pairs"][other] = bins
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, default=50)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--n-items", type=int, default=5000)
    ap.add_argument("--every-sim", type=int, default=25)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--real-seed", type=int, default=20260921, help="as in experiments/real_leaderboard.py")
    ap.add_argument("--sim-seed", type=int, default=20260922)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--subsets", nargs="+", default=list(REAL_SUBSETS), choices=list(REAL_SUBSETS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../results/finite_vs_iid"))
    args = ap.parse_args()
    if args.smoke:
        args.orders, args.reps, args.benchmarks = 2, 20, ["gsm8k", "truthfulqa"]

    def run_jobs(fn, jobs):
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                return list(pool.map(fn, jobs, chunksize=max(1, len(jobs) // (args.workers * 4))))
        return [fn(j) for j in jobs]

    started = time.time()
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "real": {}, "simulation": {}}
    for b in args.benchmarks:
        for s in args.subsets:
            t0 = time.time()
            seeds = np.random.SeedSequence([args.real_seed, BENCHMARKS.index(b), SUBSETS.index(s)]).generate_state(args.orders)
            runs = run_jobs(one_real_order, [(int(sd), b, s, args.k, args.alpha, args.every_frac) for sd in seeds])
            result["real"][f"{b}/{s}"] = _aggregate(runs)
            a = result["real"][f"{b}/{s}"]["at"]
            print(f"real {b}/{s}: {time.time() - t0:.0f}s; true frac at 0.75: finite {a['finite']['0.75']['true_frac']['mean']:.3f} "
                  f"baseline {a['baseline']['0.75']['true_frac']['mean']:.3f} iid {a['iid']['0.75']['true_frac']['mean']:.3f}", flush=True)
    for name, abilities in SIM_REGIMES.items():
        t0 = time.time()
        seeds = np.random.SeedSequence([args.sim_seed, list(SIM_REGIMES).index(name)]).generate_state(args.reps)
        runs = run_jobs(one_sim_replicate, [(int(sd), abilities, args.n_items, args.k, args.alpha, args.every_sim) for sd in seeds])
        summary = _aggregate(runs)
        summary["abilities"] = abilities
        summary["accuracies"] = true_theta(np.asarray(abilities, dtype=float)).tolist()
        result["simulation"][name] = summary
        a = summary["at"]
        print(f"simulation {name}: {time.time() - t0:.0f}s; true frac at 0.75: finite {a['finite']['0.75']['true_frac']['mean']:.3f} "
              f"baseline {a['baseline']['0.75']['true_frac']['mean']:.3f} iid {a['iid']['0.75']['true_frac']['mean']:.3f}", flush=True)
    result["seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else f"orders{args.orders}_reps{args.reps}"
    path = args.out / f"finite_vs_baseline_{tag}.json"
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
