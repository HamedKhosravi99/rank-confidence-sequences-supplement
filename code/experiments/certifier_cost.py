"""Retirement cost and run time of the certifiers on real data (Tables 8 and 15).

retire  The paper's subsets (top 8, top 20, spread 20; six benchmarks; the 50 item orders of
        Section 5 E2). Three top-3 retirement passes per order, one per certifier: e-Bonferroni with
        closure, the shortcut (Algorithm 2) and the exact test (Algorithm 1 by enumeration for 8
        models, Algorithm 3 by integer programming for 20). Each certifier makes its own retirement
        decisions on the same item order, exactly as experiments/real_leaderboard.py retires. The
        pass with the paper's certifier (exact for 8 models, shortcut for 20) must reproduce the
        saved E2 run (results/real_leaderboard/real_leaderboard_orders50.json).
timing  The 8, 12, 20 and 30 highest-scoring models on the six benchmarks. The three certifiers on
        identical wealth paths at every 1% of the benchmark (the exact test by integer programming
        at every size, and also by enumeration at 8): wall-clock seconds per run, integer programs
        solved, true dominances certified and mean rank-interval width at the checkpoints. Table 15
        reports the exact test by integer programming and the shortcut, from a run with
        --workers 7, so up to seven runs at a time on one machine.

Usage:  python -m experiments.certifier_cost --part retire --orders 50 --workers 12
        python -m experiments.certifier_cost --part timing --orders 10 --workers 7
        python -m experiments.certifier_cost --part retire --orders 1 --benchmarks truthfulqa --smoke
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.real_leaderboard import BENCHMARKS, FRACTIONS, SUBSETS, choose, leaderboard_score, load
from rcs import (BonferroniCertifier, ExactCertifier, ExactILPCertifier, RankConfidenceSequence,
                 ShortcutCertifier, log_wealth_paths)
from rcs.report import rank_intervals

PANELS = ("top8", "top20", "spread20")
SIZES = (8, 12, 20, 30)


def exact_method(m: int) -> str:
    return "exact" if m <= 8 else "ilp"


def retire_order(args: tuple) -> dict:
    index, seed, benchmark, subset, k, alpha, every_frac = args
    scores, _ = load(benchmark)
    x_full = scores[:, choose(subset, leaderboard_score())]
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    every = max(1, int(round(every_frac * n)))
    out = {"order": index, "seed": int(seed), "benchmark": benchmark, "subset": subset, "n_items": n, "n_models": m}
    for name, method in (("bonferroni", "bonferroni"), ("shortcut", "shortcut"), ("exact", exact_method(m))):
        t0 = time.time()
        seq = RankConfidenceSequence(m, alpha, population_size=n, method=method)
        wrong = False
        for t in range(n):  # the retirement loop of experiments/real_leaderboard.py
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
        out[name] = {"method": method, "cost": seq.evaluations / (m * n), "goal": bool(known.all()),
                     "settled": float(known.mean()), "wrong": wrong, "items_seen": int(seq.t),
                     "seconds": time.time() - t0,
                     "ilp_solves": int(getattr(seq.certifier, "n_ilp_solves", 0))}
    return out


def timing_order(args: tuple) -> dict:
    index, seed, benchmark, size, alpha, every_frac = args
    scores, _ = load(benchmark)
    idx = np.argsort(-leaderboard_score(), kind="stable")[:size]
    x_full = scores[:, idx]
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    x = x_full[np.random.default_rng(seed).permutation(n)]
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = np.unique(np.concatenate([np.arange(every, n + 1, every), [n], list(check)])).astype(int)
    t0 = time.time()
    paths = log_wealth_paths(x, population_size=n, bet="mixture", at=looks)
    out = {"order": index, "seed": int(seed), "benchmark": benchmark, "size": size, "n_items": n,
           "n_looks": len(looks), "true_pairs": int(true_pair.sum()), "wealth_seconds": time.time() - t0}
    certs = {"bonferroni": BonferroniCertifier, "shortcut": ShortcutCertifier, "exact_ilp": ExactILPCertifier}
    if m <= 8:
        certs["exact_enum"] = ExactCertifier
    for name, cls in certs.items():
        cert = cls(m, alpha)
        rec = {"checkpoints": {}, "wrong": False}
        t0 = time.time()
        for t, lw in zip(looks, paths):
            d = cert.update(lw)
            rec["wrong"] = rec["wrong"] or cert.failed or bool((d & false_pair).any())
            if int(t) in check:
                lower, upper = rank_intervals(d)
                rec["checkpoints"][str(check[int(t)])] = {"true_certified": int((d & true_pair).sum()),
                                                          "width": float((upper - lower).mean())}
        rec["seconds"] = time.time() - t0
        rec["ilp_solves"] = int(getattr(cert, "n_ilp_solves", 0))
        rec["unresolved"] = int(getattr(cert, "n_unresolved", 0))
        out[name] = rec
    return out


def check_retire_against_paper(runs: list[dict], saved: Path) -> list[str]:
    paper = json.loads(saved.read_text())["results"]
    problems = []
    for key, agg in paper.items():
        b, s = key.split("/")
        rs = [r for r in runs if r["benchmark"] == b and r["subset"] == s and r["order"] < 50]
        if len(rs) < 50:
            continue
        name = "exact" if rs[0]["n_models"] <= 8 else "shortcut"  # the paper's certifier for this panel
        for q in ("cost", "goal", "wrong", "items_seen"):
            mine = float(np.mean([r[name][q] for r in rs]))
            if abs(mine - agg["retire_topk"][q]) > 1e-12:
                problems.append(f"{key} {name} {q}: {mine} vs {agg['retire_topk'][q]}")
    return problems


def summarize_retire(runs: list[dict]) -> dict:
    table = {}
    for s in PANELS:
        for b in BENCHMARKS:
            rs = [r for r in runs if r["benchmark"] == b and r["subset"] == s]
            if not rs:
                continue
            row = {"orders": len(rs)}
            for name in ("bonferroni", "shortcut", "exact"):
                row[name] = {q: float(np.mean([r[name][q] for r in rs]))
                             for q in ("cost", "goal", "settled", "wrong", "items_seen", "seconds", "ilp_solves")}
            row["exact_minus_bonferroni_cost"] = [r["exact"]["cost"] - r["bonferroni"]["cost"] for r in rs]
            table[f"{b}/{s}"] = row
    return table


def summarize_timing(runs: list[dict]) -> dict:
    table = {}
    for size in SIZES:
        for b in BENCHMARKS:
            rs = [r for r in runs if r["benchmark"] == b and r["size"] == size]
            if not rs:
                continue
            row = {"orders": len(rs), "true_pairs": rs[0]["true_pairs"], "n_looks": rs[0]["n_looks"],
                   "wealth_seconds": float(np.mean([r["wealth_seconds"] for r in rs]))}
            for name in ("bonferroni", "shortcut", "exact_ilp", "exact_enum"):
                if name not in rs[0]:
                    continue
                row[name] = {"seconds": float(np.mean([r[name]["seconds"] for r in rs])),
                             "ilp_solves": float(np.mean([r[name]["ilp_solves"] for r in rs])),
                             "unresolved": float(np.mean([r[name]["unresolved"] for r in rs])),
                             "wrong": float(np.mean([r[name]["wrong"] for r in rs])),
                             "checkpoints": {f: {q: float(np.mean([r[name]["checkpoints"][f][q] for r in rs]))
                                                 for q in ("true_certified", "width")}
                                             for f in rs[0][name]["checkpoints"]}}
            table[f"{b}/top{size}"] = row
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("retire", "timing"), required=True)
    ap.add_argument("--orders", type=int, default=50)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--panels", nargs="+", default=list(PANELS), choices=list(PANELS))
    ap.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260921, help="E2's seed, so retire uses the paper's orders")
    ap.add_argument("--timing-seed", type=int, default=20261110)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--paper", type=Path, default=Path("../results/real_leaderboard/real_leaderboard_orders50.json"))
    ap.add_argument("--out", type=Path, default=Path("../results/certifier_cost"))
    args = ap.parse_args()

    started = time.time()
    jobs = []
    if args.part == "retire":
        for b in args.benchmarks:
            for s in args.panels:
                seeds = np.random.SeedSequence([args.seed, BENCHMARKS.index(b), SUBSETS.index(s)]).generate_state(args.orders)
                jobs += [(i, int(seeds[i]), b, s, args.k, args.alpha, args.every_frac) for i in range(args.orders)]
        fn = retire_order
    else:
        for b in args.benchmarks:
            for size in args.sizes:
                seeds = np.random.SeedSequence([args.timing_seed, BENCHMARKS.index(b), size]).generate_state(args.orders)
                jobs += [(i, int(seeds[i]), b, size, args.alpha, args.every_frac) for i in range(args.orders)]
        fn = timing_order
    jobs.sort(key=lambda j: -j[3] if args.part == "timing" else 0)  # longest (30 models) first
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else f"orders{args.orders}"
    path = args.out / f"certifier_cost_{args.part}_{tag}.json"
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["processor"] = platform.processor() or platform.machine()
    runs = []
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        futures = [pool.submit(fn, j) for j in jobs]
        for fut in as_completed(futures):
            runs.append(fut.result())
            if len(runs) % 50 == 0:
                print(f"{len(runs)} of {len(jobs)} done, {time.time() - started:.0f}s", flush=True)
    out = {"config": config, "runs": runs, "seconds": time.time() - started}
    if args.part == "retire":
        out["table"] = summarize_retire(runs)
        out["paper_check"] = check_retire_against_paper(runs, args.paper)
        print("paper check (the paper's certifier against the saved E2 run):",
              "all equal" if not out["paper_check"] else out["paper_check"][:10])
        for key, row in out["table"].items():
            print(f"{key:<20} cost bonf {100 * row['bonferroni']['cost']:7.3f}  shortcut {100 * row['shortcut']['cost']:7.3f}  "
                  f"exact {100 * row['exact']['cost']:7.3f}  goal {row['bonferroni']['goal']:.2f}/{row['shortcut']['goal']:.2f}/"
                  f"{row['exact']['goal']:.2f}  s {row['bonferroni']['seconds']:.1f}/{row['shortcut']['seconds']:.1f}/{row['exact']['seconds']:.1f}")
    else:
        out["table"] = summarize_timing(runs)
        for key, row in out["table"].items():
            cells = "  ".join(f"{n} {row[n]['seconds']:.2f}s w{row[n]['checkpoints']['0.5']['width']:.2f}"
                              for n in ("bonferroni", "shortcut", "exact_ilp", "exact_enum") if n in row)
            print(f"{key:<18} looks {row['n_looks']}  wealth {row['wealth_seconds']:.1f}s  {cells}")
    path.write_text(json.dumps(out))
    print("wrote", path, f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
