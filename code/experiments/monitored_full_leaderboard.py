"""All 395 models, monitored, with per-order output (Table 3, Table 8 and Appendix E.6).

Fixed-sample rank sets on all 395 models, recomputed at every look: Holm over the M(M-1) one-sided
paired z-tests, closed under transitivity, is recomputed at every 1% of the benchmark, as in
Table 9, and also once at the halfway look alone as the single-look control. A run is wrong if
any look certifies a false dominance, which is the error event of Appendix E.

One monitored wealth path per item order feeds two certifiers on identical wealths, the shortcut
(the paper's method) and e-Bonferroni with transitive closure. Two retirement passes, one per
certifier, give the matched top-k cost: each certifier makes its own retirement decisions on the
same item order.

Seeds follow full_leaderboard.py, so order i here is order i of that script (the seed sequence
is prefix-consistent). ``--start`` skips orders that an earlier run already covered, and
``--holm-only`` records Holm's error event alone, for many orders.

The paired variance is recomputed from the differences for near-identical models (see
holm_z_from_sums). The first runs (full395_orders0-49.json, holm395_orders50-499/500-999.json) used
the running sums alone; on the binary benchmarks that is exact, and TruthfulQA's Holm events were
rerun with the fix (``--out ../results/monitored_full_leaderboard/exact``).

Usage:  python -m experiments.monitored_full_leaderboard --smoke
        python -m experiments.monitored_full_leaderboard --orders 50 --workers 7
        python -m experiments.monitored_full_leaderboard --holm-only --start 50 --orders 450 --workers 3
        python -m experiments.monitored_full_leaderboard --holm-only --start 500 --orders 500 --workers 3
        python -m experiments.monitored_full_leaderboard --holm-only --orders 1000 --benchmarks truthfulqa --out ../results/monitored_full_leaderboard/exact
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.early_stopping import _to_retire
from experiments.real_leaderboard import BENCHMARKS, load
from rcs import RankConfidenceSequence
from rcs.baselines import holm_dominance
from rcs.certify import BonferroniCertifier, ShortcutCertifier, transitive_closure
from rcs.report import rank_intervals, tiers

FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
CERTIFIERS = {"shortcut": ShortcutCertifier, "bonferroni": BonferroniCertifier}


def holm_z_from_sums(n: int, s: np.ndarray, a: np.ndarray, g: np.ndarray, alpha: float,
                     x_seen: np.ndarray | None = None, tol: float = 1e-6) -> np.ndarray:
    """Holm over the paired one-sided z-tests from running sums (same formulas as
    full_leaderboard.holm_z_dominance, which recomputes them from x[:n]).

    With continuous scores (TruthfulQA), two near-identical models can differ by about 1e-10 on
    an item. Their sum of squared differences, A_j - 2 (X'X)_jl + A_l, is then lost to rounding,
    the variance comes out as 0 and the zero-variance rule turns any positive rounding residue in
    the mean into an infinite statistic. Given ``x_seen`` (the items seen, x[:n]), every pair whose
    sum of squared differences is below ``tol`` is recomputed from the differences themselves.
    Binary scores have exact sums, so for them the result is unchanged.
    """
    mean = (s[:, None] - s[None, :]) / n
    sq = a[:, None] - 2.0 * g + a[None, :]
    var = np.maximum(sq - n * mean**2, 0.0) / max(n - 1, 1)
    if x_seen is not None:
        ii, jj = np.nonzero(np.triu(sq < tol, 1))
        if len(ii):
            dd = x_seen[:, ii] - x_seen[:, jj]
            mean[ii, jj] = dd.mean(axis=0)
            mean[jj, ii] = -mean[ii, jj]
            var[ii, jj] = var[jj, ii] = dd.var(axis=0, ddof=1) if n > 1 else 0.0
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
    index, seed, benchmark, k, alpha, every_frac = args
    t_start = time.time()
    x_full, _ = load(benchmark)
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = sorted(set(int(t) for t in np.arange(every, n + 1, every)) | set(check) | {n})
    half = int(round(0.5 * n))
    out = {"order": index, "seed": int(seed), "benchmark": benchmark, "n_items": n, "n_models": m,
           "n_looks": len(looks), "seconds": {}}

    # monitored pass: one wealth path, two certifiers on identical wealths
    t0 = time.time()
    seq = RankConfidenceSequence(m, alpha, population_size=n, method="bonferroni", bet="mixture")
    certs = {name: cls(m, alpha) for name, cls in CERTIFIERS.items()}
    mon = {name: {"ever_wrong": False, "first_wrong": None, "checkpoints": {}} for name in certs}
    look_set = set(looks)
    for t in range(n):
        seq.observe(x[t])
        if t + 1 not in look_set:
            continue
        for name, cert in certs.items():
            d = cert.update(seq.wealth.log_wealth)
            wrong = cert.failed or bool((d & false_pair).any())
            rec = mon[name]
            if wrong and not rec["ever_wrong"]:
                rec["first_wrong"] = (t + 1) / n
            rec["ever_wrong"] = rec["ever_wrong"] or wrong
            if t + 1 in check:
                s = _summary(d, true_pair, false_pair, k, cert.failed)
                s["ever_wrong_so_far"] = rec["ever_wrong"]
                rec["checkpoints"][str(check[t + 1])] = s
    out["monitored"] = mon
    out["seconds"]["monitored"] = time.time() - t0

    # Holm with the paired z-test, recomputed at every look; the halfway look alone is the control
    t0 = time.time()
    s_sum, a_sum, g_sum = np.zeros(m), np.zeros(m), np.zeros((m, m))
    seen = 0
    holm = {"ever_wrong": False, "first_wrong": None, "wrong_looks": 0, "max_false_at_a_look": 0,
            "false_pairs_ever": 0, "half_only_wrong": None, "checkpoints": {}}
    false_union = np.zeros((m, m), dtype=bool)
    for t in looks:
        chunk = x[seen:t]
        s_sum += chunk.sum(axis=0)
        a_sum += (chunk * chunk).sum(axis=0)
        g_sum += chunk.T @ chunk
        seen = t
        d = transitive_closure(holm_z_from_sums(t, s_sum, a_sum, g_sum, alpha, x_seen=x[:t]))
        bad = d & false_pair
        wrong = bool(bad.any())
        if wrong:
            if not holm["ever_wrong"]:
                holm["first_wrong"] = t / n
            holm["wrong_looks"] += 1
            holm["max_false_at_a_look"] = max(holm["max_false_at_a_look"], int(bad.sum()))
            false_union |= bad
        holm["ever_wrong"] = holm["ever_wrong"] or wrong
        if t == half:
            holm["half_only_wrong"] = wrong
        if t in check:
            s = _summary(d, true_pair, false_pair, k, False)
            s["wrong_at_this_look"] = wrong
            s["ever_wrong_so_far"] = holm["ever_wrong"]
            holm["checkpoints"][str(check[t])] = s
    holm["false_pairs_ever"] = int(false_union.sum())
    out["holm"] = holm
    out["seconds"]["holm"] = time.time() - t0

    # retirement: each certifier retires models on its own reports, same item order
    out["retire_topk"] = {}
    for name in CERTIFIERS:
        t0 = time.time()
        seq = RankConfidenceSequence(m, alpha, population_size=n, method=name, bet="mixture")
        wrong = False
        for t in range(n):
            if seq.active.sum() <= 1:
                break
            seq.observe(np.where(seq.active, x[t], np.nan))
            if (t + 1) % every == 0 or t + 1 == n:
                rep = seq.report()
                wrong = wrong or rep.failed or bool((rep.dominance & false_pair).any())
                for j in np.flatnonzero(_to_retire("topk", rep, k) & seq.active):
                    seq.retire(int(j))
        rep = seq.report()
        settled = (rep.upper <= k) | (rep.lower > k)
        out["retire_topk"][name] = {"cost": seq.evaluations / (m * n), "goal": bool(settled.all()),
                                    "settled_frac": float(settled.mean()), "wrong": wrong,
                                    "items_seen": int(seq.t)}
        out["seconds"][f"retire_{name}"] = time.time() - t0
    out["seconds"]["total"] = time.time() - t_start
    return out


def holm_only_order(args: tuple) -> dict:
    """Holm's error event alone, for many more orders. The closure is skipped: it adds a false
    dominance only when one is already certified, so the event is unchanged."""
    index, seed, benchmark, k, alpha, every_frac = args
    x_full, _ = load(benchmark)
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = sorted(set(int(t) for t in np.arange(every, n + 1, every)) | set(check) | {n})
    half = int(round(0.5 * n))
    s_sum, a_sum, g_sum = np.zeros(m), np.zeros(m), np.zeros((m, m))
    seen = 0
    rec = {"order": index, "seed": int(seed), "benchmark": benchmark, "n_looks": len(looks),
           "ever_wrong": False, "first_wrong": None, "wrong_looks": 0, "half_only_wrong": None}
    for t in looks:
        chunk = x[seen:t]
        s_sum += chunk.sum(axis=0)
        a_sum += (chunk * chunk).sum(axis=0)
        g_sum += chunk.T @ chunk
        seen = t
        wrong = bool((holm_z_from_sums(t, s_sum, a_sum, g_sum, alpha, x_seen=x[:t]) & false_pair).any())
        if wrong:
            if not rec["ever_wrong"]:
                rec["first_wrong"] = t / n
            rec["wrong_looks"] += 1
        rec["ever_wrong"] = rec["ever_wrong"] or wrong
        if t == half:
            rec["half_only_wrong"] = wrong
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holm-only", action="store_true", help="Holm's error event only, for many orders")
    ap.add_argument("--orders", type=int, default=50)
    ap.add_argument("--start", type=int, default=0, help="index of the first order (skip earlier ones)")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../results/monitored_full_leaderboard"))
    args = ap.parse_args()
    if args.smoke:
        args.orders, args.benchmarks, args.every_frac = 1, ["truthfulqa"], 0.1

    started = time.time()
    jobs = []
    for b in args.benchmarks:
        seeds = np.random.SeedSequence([args.seed, BENCHMARKS.index(b)]).generate_state(args.start + args.orders)
        jobs += [(args.start + i, int(seeds[args.start + i]), b, args.k, args.alpha, args.every_frac)
                 for i in range(args.orders)]
    # round-robin over benchmarks (order 0 of each, then order 1, ...): if the run is stopped
    # early, every benchmark has about the same number of finished orders
    jobs.sort(key=lambda j: (j[0], args.benchmarks.index(j[2])))
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else f"orders{args.start}-{args.start + args.orders - 1}"
    path = args.out / f"{'holm395' if args.holm_only else 'full395'}_{tag}.json"
    fn = holm_only_order if args.holm_only else one_order
    runs = []
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(fn, j) for j in jobs]
            for fut in as_completed(futures):  # saved as each order finishes
                r = fut.result()
                runs.append(r)
                if args.holm_only:
                    if len(runs) % 50 == 0:  # saved as it goes, so a cutoff keeps every finished order
                        print(f"{len(runs)} orders done", flush=True)
                        path.write_text(json.dumps({"config": {k: (str(v) if isinstance(v, Path) else v)
                                                               for k, v in vars(args).items()}, "runs": runs}))
                    continue
                print(f"{r['benchmark']} order {r['order']}: {r['seconds']['total']:.0f}s; ours wrong "
                      f"{r['monitored']['shortcut']['ever_wrong']}, Holm wrong {r['holm']['ever_wrong']} "
                      f"(half only {r['holm']['half_only_wrong']})", flush=True)
                path.write_text(json.dumps({"config": {k: (str(v) if isinstance(v, Path) else v)
                                                       for k, v in vars(args).items()}, "runs": runs}))
    else:
        for j in jobs:
            runs.append(fn(j))
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "runs": runs, "seconds": time.time() - started}
    path.write_text(json.dumps(result))
    print("wrote", path, f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
