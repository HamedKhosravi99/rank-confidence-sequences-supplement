"""E3c: the exact test beyond eight models, computed by integer programming (Algorithm 3).

On identical wealth paths (mixture bet, finite-benchmark bound), the exact certifier of
Algorithm 1 computed by the integer program of Algorithm 3 is compared with the transitivity
shortcut (Algorithm 2) and with e-Bonferroni:

  real      the six benchmarks of E2, top-20 and spread-20 subsets, random item orders, looks
            every 1% of the benchmark plus the checkpoint fractions: certified true orderings,
            mean rank-interval width, certified tiers, ever a false certification, and the cost
            of the integer program (solves, seconds);
  timing    simulated binary scores, evenly spread abilities, M = 10, 20, 30, 50: seconds and
            integer programs per run.

Usage:  python -m experiments.exact_ilp --orders 10 --workers 6
        python -m experiments.exact_ilp --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.real_leaderboard import BENCHMARKS, FRACTIONS, choose, leaderboard_score, load
from rcs import BonferroniCertifier, ExactILPCertifier, ShortcutCertifier, log_wealth_paths
from rcs.report import rank_intervals, tiers

SUBSETS = ("top20", "spread20")
CERTIFIERS = ("exact", "shortcut", "bonferroni")


def _make(name: str, m: int, alpha: float):
    return {"exact": ExactILPCertifier, "shortcut": ShortcutCertifier, "bonferroni": BonferroniCertifier}[name](m, alpha)


def _summary(d: np.ndarray, true_pair: np.ndarray, false_pair: np.ndarray, failed: bool) -> dict:
    lower, upper = rank_intervals(d)
    return {"true_certified": int((d & true_pair).sum()), "false_certified": int((d & false_pair).sum()),
            "tiers": -1 if failed else int(tiers(d).max()), "width": float((upper - lower).mean())}


def one_real_order(args: tuple) -> dict:
    seed, benchmark, subset, alpha, every_frac = args
    scores, _ = load(benchmark)
    x_full = scores[:, choose(subset, leaderboard_score())]
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = np.unique(np.concatenate([np.arange(every, n + 1, every), [n], list(check)])).astype(int)
    paths = log_wealth_paths(x, population_size=n, bet="mixture", at=looks)
    out = {"n_items": n, "n_models": m, "true_pairs": int(true_pair.sum()), "seconds": {}, "ilp_solves": 0}
    for name in CERTIFIERS:
        cert = _make(name, m, alpha)
        ever_wrong, t0 = False, time.time()
        out[name] = {}
        for t, lw in zip(looks, paths):
            d = cert.update(lw)
            ever_wrong = ever_wrong or cert.failed or bool((d & false_pair).any())
            if int(t) in check:
                s = _summary(d, true_pair, false_pair, cert.failed)
                s["ever_wrong_so_far"] = ever_wrong
                out[name][str(check[int(t)])] = s
        out["seconds"][name] = time.time() - t0
        if name == "exact":
            out["ilp_solves"] = cert.n_ilp_solves
            out["unresolved"] = cert.n_unresolved  # pairs the numerical tolerance left undecided
    return out


def one_timing(args: tuple) -> dict:
    seed, m, n, every, alpha = args
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.70, 0.50, m)
    x = (rng.random((n, m)) < theta[None, :]).astype(float)
    looks = np.arange(every, n + 1, every)
    paths = log_wealth_paths(x, population_size=n, bet="mixture", at=looks)
    theta_n = x.mean(axis=0)  # the finite-benchmark target: accuracies on these n items
    true_pair = theta_n[:, None] > theta_n[None, :]
    out = {"m": m, "n": n, "every": every, "true_pairs": int(true_pair.sum())}
    for name in ("exact", "shortcut"):
        cert = _make(name, m, alpha)
        t0 = time.time()
        for lw in paths:
            d = cert.update(lw)
        out[name] = {"seconds": time.time() - t0, "true_certified": int((d & true_pair).sum()),
                     "false_certified": int((d & ~true_pair).sum())}
        if name == "exact":
            out["ilp_solves"] = cert.n_ilp_solves
    return out


def aggregate_real(runs: list[dict]) -> dict:
    agg = {"n_items": runs[0]["n_items"], "n_models": runs[0]["n_models"], "true_pairs": runs[0]["true_pairs"],
           "orders": len(runs), "ilp_solves": float(np.mean([r["ilp_solves"] for r in runs])),
           "unresolved": float(np.mean([r.get("unresolved", 0) for r in runs])),
           "seconds": {k: float(np.mean([r["seconds"][k] for r in runs])) for k in CERTIFIERS}}
    for name in CERTIFIERS:
        agg[name] = {}
        for f in runs[0][name]:
            rows = [r[name][f] for r in runs]
            agg[name][f] = {q: float(np.mean([row[q] for row in rows])) for q in rows[0]}
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, default=10)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--timing-sizes", nargs="+", type=int, default=[10, 20, 30, 50])
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../results/exact_ilp"))
    args = ap.parse_args()
    if args.smoke:
        args.orders, args.benchmarks, args.subsets, args.timing_sizes = 2, ["gsm8k", "truthfulqa"], ["top20"], [10, 20]

    def run_jobs(fn, jobs):
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                return list(pool.map(fn, jobs))
        return [fn(j) for j in jobs]

    started = time.time()
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "real": {}, "timing": {}}
    for benchmark in args.benchmarks:
        for subset in args.subsets:
            t0 = time.time()
            seeds = np.random.SeedSequence([args.seed, BENCHMARKS.index(benchmark), SUBSETS.index(subset)]).generate_state(args.orders)
            runs = run_jobs(one_real_order, [(int(s), benchmark, subset, args.alpha, args.every_frac) for s in seeds])
            agg = aggregate_real(runs)
            result["real"][f"{benchmark}/{subset}"] = agg
            e, s = agg["exact"]["0.5"]["true_certified"], agg["shortcut"]["0.5"]["true_certified"]
            print(f"{benchmark}/{subset}: {time.time() - t0:.0f}s; at half the benchmark exact {e:.1f} vs shortcut {s:.1f} "
                  f"of {agg['true_pairs']} true orderings; {agg['ilp_solves']:.0f} ILP solves, {agg['seconds']['exact']:.1f}s per order", flush=True)
    for m in args.timing_sizes:
        n, every = (3000, 20) if not args.smoke else (600, 20)
        runs = run_jobs(one_timing, [(int(s), m, n, every, args.alpha) for s in np.random.SeedSequence([args.seed, m]).generate_state(3)])
        result["timing"][str(m)] = {"m": m, "n": n, "every": every, "true_pairs": runs[0]["true_pairs"],
                                    "seconds": float(np.mean([r["exact"]["seconds"] for r in runs])),
                                    "ilp_solves": float(np.mean([r["ilp_solves"] for r in runs])),
                                    "true_certified_exact": float(np.mean([r["exact"]["true_certified"] for r in runs])),
                                    "true_certified_shortcut": float(np.mean([r["shortcut"]["true_certified"] for r in runs])),
                                    "false_certified_exact": float(np.mean([r["exact"]["false_certified"] for r in runs]))}
        r = result["timing"][str(m)]
        print(f"timing M={m}: {r['seconds']:.1f}s per run of {n} items ({n // every} looks), {r['ilp_solves']:.0f} ILP solves; "
              f"true orderings certified exact {r['true_certified_exact']:.1f} vs shortcut {r['true_certified_shortcut']:.1f} of {r['true_pairs']}", flush=True)
    result["seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ("exact_ilp_smoke.json" if args.smoke else f"exact_ilp_orders{args.orders}.json")
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
