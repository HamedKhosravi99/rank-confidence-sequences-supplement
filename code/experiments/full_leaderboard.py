"""E6. The whole leaderboard: all 395 models, no selection, on the six real benchmarks.

E2 and E5 use twenty models chosen by their full-benchmark scores. That choice is valid under
the finite-benchmark estimand, because the score matrix is fixed before the random item order is
drawn, but it is not a procedure an operator could run prospectively without already knowing
which models deserve inclusion. This experiment removes the choice: the model set is every model
in the release, fixed before any randomness, and M = 395.

What changes with M. The multiplicity correction enters only through the threshold
M(M-1)/alpha, so the evidence a pair needs grows like 2 log M: from log(380/0.05) = 8.9 in
log-wealth at M = 20 to log(155,630/0.05) = 14.9 at M = 395, a factor 1.7 in items for the same
pair. The number of easy pairs grows quadratically, and for a small k most models are far from
the top k and retire at once.

Certification uses e-Bonferroni with transitive closure, which is O(M^2) per look and is a
certified set in its own right; by Proposition 4.4 it is contained in what the shortcut and the
exact test certify, so every count reported here is a lower bound on theirs (E3 measures the gap
at 0.01 pairs for M = 20, E3c at 1.4 of 190 for the exact test). The shortcut is available with
--method shortcut but costs an M^3 temporary (about 0.5 GB at M = 395).

Comparator. The fixed-sample Holm z-test is recomputed at the checkpoint fractions only. Its
paired statistics are built from column sums and the Gram matrix X'X rather than the (n, M, M)
tensor of pairwise differences, which would be 17.5 TB at M = 395.

Recorded per benchmark, per item order, at each checkpoint fraction: true dominances certified,
false ones, certified tiers, mean rank-interval width, and the fraction of models whose top-k
status is settled; over all looks, whether any report was ever wrong; and, in a second pass with
retirement, the evaluation cost of settling every model's top-k status.

Usage:  python -m experiments.full_leaderboard --smoke
        python -m experiments.full_leaderboard --orders 10 --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.early_stopping import _to_retire
from experiments.real_leaderboard import BENCHMARKS, load
from rcs import RankConfidenceSequence
from rcs.baselines import holm_dominance
from rcs.certify import transitive_closure
from rcs.report import rank_intervals, tiers

FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)


def holm_z_dominance(x: np.ndarray, alpha: float) -> np.ndarray:
    """Holm over the paired one-sided z-tests, without the (n, M, M) tensor of differences.

    For every ordered pair, the mean and the variance of the paired difference follow from the
    column sums, the column sums of squares and the Gram matrix:
    sum_t (x_tj - x_tl)^2 = A_j - 2 (X'X)_jl + A_l with A_j = sum_t x_tj^2.
    """
    n, m = x.shape
    s = x.sum(axis=0)
    a = (x * x).sum(axis=0)
    g = x.T @ x
    mean = (s[:, None] - s[None, :]) / n
    sq = a[:, None] - 2.0 * g + a[None, :]  # sum of squared differences, per pair
    var = np.maximum(sq - n * mean**2, 0.0) / max(n - 1, 1)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(sd > 0, np.sqrt(n) * mean / np.where(sd > 0, sd, 1.0), np.where(mean > 0, np.inf, 0.0))
    p = stats.norm.sf(stat)
    np.fill_diagonal(p, 1.0)
    return holm_dominance(p, alpha)


def _summary(d: np.ndarray, true_pair: np.ndarray, false_pair: np.ndarray, k: int, failed: bool) -> dict:
    lower, upper = rank_intervals(d)
    settled = (upper <= k) | (lower > k)
    return {"true_certified": int((d & true_pair).sum()),
            "false_certified": int((d & false_pair).sum()),
            "tiers": -1 if failed else int(tiers(d).max()),
            "width": float((upper - lower).mean()),
            "topk_settled": float(settled.mean())}


def one_order(args: tuple) -> dict:
    seed, benchmark, k, alpha, every_frac, method, with_holm = args
    x_full, _ = load(benchmark)
    n, m = x_full.shape
    theta = x_full.mean(axis=0)  # the finite-benchmark truth: accuracy on the whole benchmark
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = set(int(t) for t in np.arange(every, n + 1, every)) | set(check) | {n}

    out = {"n_items": n, "n_models": m, "true_pairs": int(true_pair.sum()),
           "tied_pairs": int(((theta[:, None] == theta[None, :]) & ~np.eye(m, dtype=bool)).sum() // 2),
           "ours": {}, "holm": {}, "ever_wrong": False, "seconds": {}}

    # monitored run: report at every look, record at the checkpoint fractions
    t0 = time.time()
    seq = RankConfidenceSequence(m, alpha, population_size=n, method=method, bet="mixture")
    for t in range(n):
        seq.observe(x[t])
        if t + 1 in looks:
            rep = seq.report()
            out["ever_wrong"] = out["ever_wrong"] or rep.failed or bool((rep.dominance & false_pair).any())
            if t + 1 in check:
                s = _summary(rep.dominance, true_pair, false_pair, k, rep.failed)
                s["ever_wrong_so_far"] = out["ever_wrong"]
                out["ours"][str(check[t + 1])] = s
    out["seconds"]["ours"] = time.time() - t0

    # fixed-sample Holm z-test, recomputed at the checkpoints as a fixed-sample user would
    if with_holm:
        t0 = time.time()
        ever = False
        for t in sorted(check):
            d = transitive_closure(holm_z_dominance(x[:t], alpha))  # closure is free for any FWER procedure
            ever = ever or bool((d & false_pair).any())
            s = _summary(d, true_pair, false_pair, k, False)
            s["wrong_at_this_look"] = bool((d & false_pair).any())
            s["ever_wrong_so_far"] = ever
            out["holm"][str(check[t])] = s
        out["seconds"]["holm"] = time.time() - t0

    # retirement: stop evaluating a model once its own top-k status is settled
    t0 = time.time()
    seq = RankConfidenceSequence(m, alpha, population_size=n, method=method, bet="mixture")
    wrong = False
    for t in range(n):
        if seq.active.sum() <= 1:
            break
        seq.observe(np.where(seq.active, x[t], np.nan))  # retired models are not evaluated
        if (t + 1) % every == 0 or t + 1 == n:
            rep = seq.report()
            wrong = wrong or rep.failed or bool((rep.dominance & false_pair).any())
            for j in np.flatnonzero(_to_retire("topk", rep, k) & seq.active):
                seq.retire(int(j))
    rep = seq.report()
    settled = (rep.upper <= k) | (rep.lower > k)
    out["retire_topk"] = {"cost": seq.evaluations / (m * n), "goal": bool(settled.all()),
                          "settled_frac": float(settled.mean()), "wrong": wrong, "items_seen": seq.t}
    out["seconds"]["retire"] = time.time() - t0
    return out


def aggregate(runs: list[dict]) -> dict:
    agg = {q: runs[0][q] for q in ("n_items", "n_models", "true_pairs", "tied_pairs")}
    agg["orders"] = len(runs)
    agg["ever_wrong"] = float(np.mean([r["ever_wrong"] for r in runs]))
    agg["seconds"] = {q: float(np.mean([r["seconds"][q] for r in runs])) for q in runs[0]["seconds"]}
    for key in ("ours", "holm"):
        agg[key] = {}
        for f in runs[0][key]:
            rows = [r[key][f] for r in runs]
            agg[key][f] = {q: float(np.mean([row[q] for row in rows])) for q in rows[0]}
    rows = [r["retire_topk"] for r in runs]
    agg["retire_topk"] = {q: float(np.mean([row[q] for row in rows])) for q in rows[0]}
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, default=10)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--method", default="bonferroni", choices=["bonferroni", "shortcut"])
    ap.add_argument("--no-holm", action="store_true", help="skip the fixed-sample comparator")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../results/full_leaderboard"))
    args = ap.parse_args()
    if args.smoke:
        args.orders, args.benchmarks, args.every_frac = 2, ["truthfulqa", "arc"], 0.1

    started = time.time()
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "benchmarks": {}}
    for benchmark in args.benchmarks:
        t0 = time.time()
        seeds = np.random.SeedSequence([args.seed, BENCHMARKS.index(benchmark)]).generate_state(args.orders)
        jobs = [(int(s), benchmark, args.k, args.alpha, args.every_frac, args.method, not args.no_holm) for s in seeds]
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                runs = list(pool.map(one_order, jobs))
        else:
            runs = [one_order(j) for j in jobs]
        agg = aggregate(runs)
        result["benchmarks"][benchmark] = agg
        half = agg["ours"]["0.5"]
        print(f"{benchmark}: {time.time() - t0:.0f}s; of {agg['true_pairs']:,} true orderings "
              f"({agg['tied_pairs']:,} tied pairs) ours certifies {half['true_certified']:,.0f} at half "
              f"({100*half['true_certified']/agg['true_pairs']:.0f}%), {half['tiers']:.1f} tiers, "
              f"top-{args.k} settled for {100*half['topk_settled']:.0f}% of models; "
              f"retirement cost {100*agg['retire_topk']['cost']:.1f}% of a full run, "
              f"goal in {100*agg['retire_topk']['goal']:.0f}% of orders; ever wrong {100*agg['ever_wrong']:.0f}%", flush=True)
    result["seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ("full_leaderboard_smoke.json" if args.smoke else f"full_leaderboard_orders{args.orders}.json")
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
