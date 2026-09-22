"""E3a. Does weak-order inversion give tighter rank sets than counting?

On identical pairwise wealths, three rank statements are compared at every look:

  projection   the exact projection rank set R_j = {rank_j(W) : W surviving}   (Theorem 2(a))
  exact        the counting interval [L_j, U_j] read off the exact certified set D_t  (Theorem 2(b))
  e_bonferroni the counting interval read off e-Bonferroni with free transitive closure

projection-vs-exact isolates what retaining the surviving weak orders adds to counting;
exact-vs-e_bonferroni isolates what the weak-order multiplicity correction adds to Bonferroni.

Per look, averaged over models and then over replicates or item orders:
  size            mean number of feasible ranks per model (smaller is better)
  singleton       fraction of models whose rank is exactly identified
  tighter         fraction of models with |R_j| < U_j - L_j + 1   (projection strictly tighter)
  holes           mean (U_j - L_j + 1) - |R_j|, the ranks eliminated by projection alone
  noncontiguous   fraction of models whose R_j has a gap, i.e. max R_j - min R_j + 1 > |R_j|
and whether any report was ever wrong (the true rank outside R_j or outside [L_j, U_j]).

Simulation: the E3 design (six models, 4000 items monitored every 20, alpha 0.05, mixture bet,
near-tie and even-spread abilities). Real data: the eight top-scoring Open LLM Leaderboard
models on six benchmarks, the 50 item orders of experiments/real_leaderboard.py (same seeds).
Examples of non-contiguous rank sets on the real data are recorded with model names.

Usage:  python -m experiments.rank_sets --reps 2000 --orders 50 --workers 8
        python -m experiments.rank_sets --smoke            # 20 replicates, 2 orders, 2 benchmarks
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.peeking import simulate, true_theta
from experiments.real_leaderboard import BENCHMARKS, FRACTIONS, SUBSETS, choose, leaderboard_score, load
from experiments.weighting import SCENARIOS
from rcs import BonferroniCertifier, ExactCertifier, log_wealth_paths
from rcs.report import rank_intervals, rank_of

SIM_SCENARIOS = ("near_ties", "even_spread")
SIM_CHECKPOINTS = (500, 1000, 2000, 4000)
REAL_SUBSET = "top8"
METHODS = ("projection", "exact", "e_bonferroni")
METRICS = ("size", "singleton", "tighter", "holes", "noncontiguous")


def _look_metrics(exact: ExactCertifier, d_exact: np.ndarray, d_bonf: np.ndarray, true_rank: np.ndarray) -> dict:
    """Metrics for one look from the exact certifier state, its dominance and e-Bonferroni's."""
    rank_sets = exact.rank_sets()
    proj_size = np.array([r.size for r in rank_sets], dtype=float)
    lo, up = rank_intervals(d_exact)
    lo_b, up_b = rank_intervals(d_bonf)
    int_size = (up - lo + 1).astype(float)
    int_size_b = (up_b - lo_b + 1).astype(float)
    span = np.array([r.max() - r.min() + 1 for r in rank_sets], dtype=float)
    covered_proj = all(true_rank[j] in rank_sets[j] for j in range(len(rank_sets)))
    covered_int = bool(np.all((lo <= true_rank) & (true_rank <= up)))
    covered_b = bool(np.all((lo_b <= true_rank) & (true_rank <= up_b)))
    return {
        "size": {"projection": proj_size.mean(), "exact": int_size.mean(), "e_bonferroni": int_size_b.mean()},
        "singleton": {"projection": float(np.mean(proj_size == 1)), "exact": float(np.mean(int_size == 1)),
                      "e_bonferroni": float(np.mean(int_size_b == 1))},
        "tighter": {"projection": float(np.mean(proj_size < int_size))},
        "holes": {"projection": float(np.mean(int_size - proj_size))},
        "noncontiguous": {"projection": float(np.mean(span > proj_size))},
        "wrong": {"projection": not covered_proj, "exact": not covered_int, "e_bonferroni": not covered_b},
        "_sets": [(int(lo[j]), int(up[j]), [int(v) for v in rank_sets[j]]) for j in range(len(rank_sets))],
    }


def _run_path(paths: np.ndarray, m: int, alpha: float, true_rank: np.ndarray, want_examples: bool) -> tuple[list, dict, list]:
    exact = ExactCertifier(m, alpha)
    bonf = BonferroniCertifier(m, alpha, close=True)
    per_look, examples = [], []
    ever_wrong = {k: False for k in METHODS}
    for i, lw in enumerate(paths):
        d_exact = exact.update(lw)
        d_bonf = bonf.update(lw)
        met = _look_metrics(exact, d_exact, d_bonf, true_rank)
        for k in METHODS:
            ever_wrong[k] = ever_wrong[k] or met["wrong"][k] or (k == "projection" and exact.failed) or (k == "e_bonferroni" and bonf.failed)
        if want_examples:
            for j, (lo, up, rs) in enumerate(met["_sets"]):
                if rs[-1] - rs[0] + 1 > len(rs):  # a gap inside the projection set
                    examples.append({"look": i, "model": j, "interval": [lo, up], "rank_set": rs})
        met.pop("_sets")
        per_look.append(met)
    return per_look, ever_wrong, examples


def one_sim_replicate(args: tuple) -> dict:
    seed, abilities, n_max, every, alpha, bet = args
    abilities = np.asarray(abilities, dtype=float)
    theta = true_theta(abilities)
    true_rank = rank_of(theta)
    x = simulate(np.random.default_rng(seed), abilities, n_max)
    times = np.arange(every, n_max + 1, every)
    paths = log_wealth_paths(x, bet=bet, at=times)
    per_look, ever_wrong, _ = _run_path(paths, x.shape[1], alpha, true_rank, want_examples=False)
    return {"times": times.tolist(), "per_look": per_look, "ever_wrong": ever_wrong}


def one_real_order(args: tuple) -> dict:
    seed, benchmark, alpha, every_frac, bet = args
    scores, models = load(benchmark)
    idx = choose(REAL_SUBSET, leaderboard_score())
    x_full = scores[:, idx]
    n, m = x_full.shape
    true_rank = rank_of(x_full.mean(axis=0))
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = np.unique(np.concatenate([np.arange(every, n + 1, every), [n], list(check)]))
    paths = log_wealth_paths(x, population_size=n, bet=bet, at=looks)
    per_look, ever_wrong, examples = _run_path(paths, m, alpha, true_rank, want_examples=True)
    for e in examples:
        e["fraction"] = float(looks[e["look"]]) / n
        e["model"] = str(models[idx[e["model"]]])
    return {"n_items": n, "n_models": m, "looks": looks.tolist(), "fractions": (looks / n).tolist(),
            "per_look": per_look, "ever_wrong": ever_wrong, "examples": examples,
            "true_rank": true_rank.tolist(), "models": [str(models[i]) for i in idx]}


def _series(runs: list[dict], metric: str, method: str) -> tuple[list, list]:
    arr = np.array([[lk[metric][method] for lk in r["per_look"]] for r in runs], dtype=float)
    return arr.mean(axis=0).tolist(), (arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])).tolist()


def _aggregate(runs: list[dict], key_times: list, checkpoints: dict) -> dict:
    out = {"n_runs": len(runs), "times": key_times, "series": {}, "at": {}, "ever_wrong": {}}
    for metric in METRICS:
        out["series"][metric] = {}
        for method in METHODS:
            if method in runs[0]["per_look"][0][metric]:
                mean, se = _series(runs, metric, method)
                out["series"][metric][method] = {"mean": mean, "se": se}
    for label, i in checkpoints.items():
        out["at"][label] = {metric: {method: out["series"][metric][method]["mean"][i]
                                     for method in out["series"][metric]} for metric in METRICS}
    for method in METHODS:
        out["ever_wrong"][method] = float(np.mean([r["ever_wrong"][method] for r in runs]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--orders", type=int, default=50)
    ap.add_argument("--n-max", type=int, default=4000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bet", default="mixture")
    ap.add_argument("--seed", type=int, default=20260918, help="simulation seed, as in experiments/weighting.py")
    ap.add_argument("--real-seed", type=int, default=20260921, help="item-order seed, as in experiments/real_leaderboard.py")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../results/rank_sets"))
    args = ap.parse_args()
    if args.smoke:
        args.reps, args.orders, args.benchmarks = 20, 2, ["gsm8k", "arc"]

    started = time.time()
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "simulation": {}, "real": {}}

    def run_jobs(fn, jobs):
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                return list(pool.map(fn, jobs, chunksize=max(1, len(jobs) // (args.workers * 4))))
        return [fn(j) for j in jobs]

    # ---- simulation, the E3 design and seeds ----
    for scenario in SIM_SCENARIOS:
        t0 = time.time()
        abilities = SCENARIOS[scenario]
        seeds = np.random.SeedSequence([args.seed, sorted(SCENARIOS).index(scenario)]).generate_state(args.reps)
        jobs = [(int(s), abilities, args.n_max, args.every, args.alpha, args.bet) for s in seeds]
        reps = run_jobs(one_sim_replicate, jobs)
        times = reps[0]["times"]
        checkpoints = {str(c): times.index(c) for c in SIM_CHECKPOINTS if c in times}
        theta = true_theta(np.asarray(abilities, dtype=float))
        summary = _aggregate(reps, times, checkpoints)
        summary.update({"abilities": abilities, "theta": theta.tolist(), "true_rank": rank_of(theta).tolist()})
        result["simulation"][scenario] = summary
        print(f"simulation {scenario}: {time.time() - t0:.1f}s", flush=True)

    # ---- real data, the top-8 subset and the item orders of real_leaderboard.py ----
    for b in args.benchmarks:
        t0 = time.time()
        seeds = np.random.SeedSequence([args.real_seed, BENCHMARKS.index(b), SUBSETS.index(REAL_SUBSET)]).generate_state(args.orders)
        jobs = [(int(sd), b, args.alpha, args.every_frac, args.bet) for sd in seeds]
        runs = run_jobs(one_real_order, jobs)
        fractions = runs[0]["fractions"]
        checkpoints = {str(f): int(np.argmin(np.abs(np.array(fractions) - f))) for f in FRACTIONS}
        summary = _aggregate(runs, fractions, checkpoints)
        examples = [e for r in runs for e in r["examples"]]
        summary.update({"n_items": runs[0]["n_items"], "n_models": runs[0]["n_models"],
                        "models": runs[0]["models"], "true_rank": runs[0]["true_rank"],
                        "n_noncontiguous_examples": len(examples),
                        "orders_with_noncontiguous": int(sum(bool(r["examples"]) for r in runs)),
                        "examples": examples[:40]})
        result["real"][b] = summary
        print(f"real {b}/{REAL_SUBSET}: {time.time() - t0:.1f}s, {len(examples)} non-contiguous rank sets", flush=True)

    result["seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else f"reps{args.reps}_orders{args.orders}"
    path = args.out / f"rank_sets_{tag}.json"
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
