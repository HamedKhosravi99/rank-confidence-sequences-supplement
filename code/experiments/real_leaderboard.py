"""E2/E4/E5 on real leaderboard data (Open LLM Leaderboard per-item results via tinyBenchmarks).

The finite-benchmark estimand makes the truth exact: theta_j is model j's accuracy on the whole
benchmark, and every certified dominance can be checked against it. For each benchmark and model
subset we draw random item orders and, at looks every ``--every-frac`` of the benchmark, record

  ours          rank confidence sequence (mixture bet, shortcut or exact certification):
                certified true pairs, false certifications (ever), tiers, mean rank-interval width;
  fixed_n       Holm-corrected one-sided McNemar rank sets recomputed at every look, as a
                fixed-sample user would: the same quantities, with "wrong at this look" and
                "wrong at some look so far";
  fixed_z       the same with a paired z-test (valid for non-binary scores, CLT-based);
  fixed_rw      Romano-Wolf step-down with a centred multiplier bootstrap of the studentized
                max statistic (B draws), computed at the checkpoint fractions only (a bootstrap at
                every 1% look would cost days); "ever wrong" is over those checkpoints;
  retire_topk   ours with Step-4 retirement once a model's top-k status is certified:
                cost in model-item evaluations relative to a full run, and whether the goal was met.

Model subsets (by leaderboard score = mean accuracy over the six scenarios):
  top20    the 20 highest-scoring models (near-ties, many merges: the leaderboard top);
  spread20 20 models evenly spaced through the ranking (well separated);
  top8     the 8 highest-scoring models (exact certification).

Usage:  python -m experiments.real_leaderboard --orders 50 --workers 10
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rcs import RankConfidenceSequence
from rcs.baselines import holm_dominance
from rcs.certify import transitive_closure
from rcs.report import rank_intervals, tiers
from scipy import stats

DATA = Path("../data/processed")
BENCHMARKS = ("mmlu", "hellaswag", "arc", "gsm8k", "winogrande", "truthfulqa")
SUBSETS = ("top20", "spread20", "top8")
FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)


def load(benchmark: str):
    z = np.load(DATA / f"{benchmark}.npz")
    return z["scores"].astype(float), [str(m) for m in z["models"]]


def leaderboard_score() -> np.ndarray:
    return np.mean([load(b)[0].mean(axis=0) for b in BENCHMARKS], axis=0)


def choose(subset: str, score: np.ndarray) -> np.ndarray:
    order = np.argsort(-score, kind="stable")
    if subset == "top20":
        return order[:20]
    if subset == "top8":
        return order[:8]
    if subset == "spread20":
        return order[np.linspace(0, len(order) - 1, 20).round().astype(int)]
    raise ValueError(subset)


def _fixed_n_at(x: np.ndarray, times: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """Holm-McNemar and Holm-z dominance matrices at every look time, from cumulative sums."""
    z = x[:, :, None] - x[:, None, :]
    wins = np.cumsum(z > 0, axis=0)[times - 1]
    losses = np.cumsum(z < 0, axis=0)[times - 1]
    s1 = np.cumsum(z, axis=0)[times - 1]
    s2 = np.cumsum(z * z, axis=0)[times - 1]
    n = times[:, None, None].astype(float)
    eye = np.eye(x.shape[1], dtype=bool)
    p_mc = stats.binom.sf(wins - 1, wins + losses, 0.5)
    p_mc = np.where(wins + losses == 0, 1.0, p_mc)
    mean = s1 / n
    var = np.maximum(s2 - n * mean**2, 0.0) / np.maximum(n - 1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(var > 0, np.sqrt(n) * mean / np.sqrt(var), 0.0)
    p_z = stats.norm.sf(stat)
    out = {}
    for name, p in (("fixed_n", p_mc), ("fixed_z", p_z)):
        p = np.where(eye[None], 1.0, p)
        out[name] = np.stack([holm_dominance(p[i], alpha) for i in range(len(times))])
    return out


def romano_wolf(x: np.ndarray, alpha: float, rng: np.random.Generator, draws: int = 200) -> np.ndarray:
    """One-sided Romano-Wolf step-down over all ordered pairs, multiplier bootstrap of the
    studentized max statistic. D[j, l] = H_jl: theta_j <= theta_l rejected."""
    n, m = x.shape
    z = (x[:, :, None] - x[:, None, :]).reshape(n, m * m)
    mean = z.mean(axis=0)
    sd = z.std(axis=0, ddof=1) if n > 1 else np.zeros(m * m)
    ok = sd > 0
    stat = np.zeros(m * m)
    stat[ok] = np.sqrt(n) * mean[ok] / sd[ok]
    xi = rng.standard_normal((draws, n))
    boot = (xi @ (z[:, ok] - mean[ok])) / np.sqrt(n) / sd[ok]  # draws x active pairs, centred
    rejected = np.zeros(m * m, dtype=bool)
    active = ok.copy()
    while True:
        if not active.any():
            break
        crit = np.quantile(boot[:, active[ok]].max(axis=1), 1 - alpha)
        new = active & (stat > crit)
        if not new.any():
            break
        rejected |= new
        active &= ~new
    d = rejected.reshape(m, m)
    np.fill_diagonal(d, False)
    return d


def _summary(d: np.ndarray, true_pair: np.ndarray, false_pair: np.ndarray) -> dict:
    lower, upper = rank_intervals(d)
    return {
        "true_certified": int((d & true_pair).sum()),
        "false_certified": int((d & false_pair).sum()),
        "tiers": int(tiers(d).max()) if not np.any(np.diag(transitive_closure(d))) else -1,
        "width": float((upper - lower).mean()),
    }


def one_order(args: tuple) -> dict:
    seed, benchmark, subset, k, alpha, every_frac = args
    scores, _ = load(benchmark)
    idx = choose(subset, leaderboard_score())
    x_full = scores[:, idx]
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    every = max(1, int(round(every_frac * n)))
    looks = np.unique(np.concatenate([np.arange(every, n + 1, every), [n]]))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = np.unique(np.concatenate([looks, list(check)]))
    method = "exact" if m <= 8 else "shortcut"

    out = {"ours": {}, "fixed_n": {}, "fixed_z": {}, "fixed_rw": {}, "ours_ever_wrong": False}
    # ours, monitored at every look
    seq = RankConfidenceSequence(m, alpha, population_size=n, method=method)
    look_set = set(int(t) for t in looks)
    for t in range(n):
        seq.observe(x[t])
        if t + 1 in look_set:
            rep = seq.report()
            wrong = rep.failed or bool((rep.dominance & false_pair).any())
            out["ours_ever_wrong"] = out["ours_ever_wrong"] or wrong
            if t + 1 in check:
                s = _summary(rep.dominance, true_pair, false_pair)
                s["ever_wrong_so_far"] = out["ours_ever_wrong"]
                out["ours"][str(check[t + 1])] = s
    # fixed-n Holm-McNemar and Holm-z at the same looks; Romano-Wolf at the checkpoints
    fixed = _fixed_n_at(x, looks, alpha)
    for name in ("fixed_n", "fixed_z"):
        ever = False
        for i, t in enumerate(looks):
            d = transitive_closure(fixed[name][i])  # closure is free for any FWER procedure
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
    # retirement for the top-k question
    seq = RankConfidenceSequence(m, alpha, population_size=n, method=method)
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
    out["retire_topk"] = {"cost": seq.evaluations / (m * n), "goal": bool(known.all()),
                          "wrong": wrong, "items_seen": seq.t}
    out["n_items"], out["n_models"], out["true_pairs"] = n, m, int(true_pair.sum())
    out["tied_pairs"] = int(((theta[:, None] == theta[None, :]) & ~np.eye(m, dtype=bool)).sum() // 2)
    return out


def aggregate(runs: list[dict]) -> dict:
    agg = {"n_items": runs[0]["n_items"], "n_models": runs[0]["n_models"],
           "true_pairs": runs[0]["true_pairs"], "tied_pairs": runs[0]["tied_pairs"],
           "orders": len(runs), "ours": {}, "fixed_n": {}, "fixed_z": {}, "fixed_rw": {}, "retire_topk": {},
           "ours_ever_wrong": float(np.mean([r["ours_ever_wrong"] for r in runs]))}
    for key in ("ours", "fixed_n", "fixed_z", "fixed_rw"):
        for f in runs[0][key]:
            rows = [r[key][f] for r in runs]
            agg[key][f] = {q: float(np.mean([row[q] for row in rows])) for q in rows[0]}
    rows = [r["retire_topk"] for r in runs]
    agg["retire_topk"] = {q: float(np.mean([row[q] for row in rows])) for q in rows[0]}
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, default=50)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("../results/real_leaderboard"))
    args = ap.parse_args()

    started = time.time()
    score = leaderboard_score()
    _, models = load(BENCHMARKS[0])
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "subsets": {s: [models[i] for i in choose(s, score)] for s in args.subsets},
              "results": {}}
    for b in args.benchmarks:
        for s in args.subsets:
            t0 = time.time()
            seeds = np.random.SeedSequence([args.seed, BENCHMARKS.index(b), SUBSETS.index(s)]).generate_state(args.orders)
            jobs = [(int(sd), b, s, args.k, args.alpha, args.every_frac) for sd in seeds]
            if args.workers > 1:
                with ProcessPoolExecutor(args.workers) as pool:
                    runs = list(pool.map(one_order, jobs, chunksize=max(1, args.orders // (args.workers * 4))))
            else:
                runs = [one_order(j) for j in jobs]
            result["results"][f"{b}/{s}"] = aggregate(runs)
            print(f"{b}/{s}: {time.time() - t0:.1f}s", flush=True)
    result["seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"real_leaderboard_orders{args.orders}.json"
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
