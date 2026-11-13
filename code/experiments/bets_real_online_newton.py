"""A data-driven bet inside the grid mixture (the online Newton mixture of Table 16).

Online Newton steps (ONS) choose lambda from the data. Here ONS becomes one more component of the
wealth-weighted mixture, next to the five grid values, with prior weight 1/6 (uniform over six
components) or 1/2. The mixture wealth is the prior-weighted average of the component wealths, so
it is again a product of predictable bets in [0, 1/2] (Remark B.3) and valid by Lemma B.1. Against
the default grid it loses at most log(1/(1 - w)) in log-wealth for ONS weight w: log(6/5) with
w = 1/6 and log 2 with w = 1/2.

Same subsets, item orders, offset, certifier (shortcut) and top-3 retirement as bets_real.py.
The plain mixture and plain ONS are rerun as references and must reproduce the saved bet run.

Usage:  python -m experiments.bets_real_online_newton --orders 1 --benchmarks truthfulqa --out /tmp/x   (smoke)
        python -m experiments.bets_real_online_newton --orders 50 --workers 12
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.real_leaderboard import BENCHMARKS, FRACTIONS, SUBSETS, _summary, choose, leaderboard_score, load
from rcs import RankConfidenceSequence
from rcs.wealth import _ONS_RATE, MIXTURE_GRID, _Bettor

PANELS = ("top20", "spread20")
HYBRIDS = {"mix+ons(1/6)": 1.0 / 6.0, "mix+ons(1/2)": 0.5}
VARIANTS = ("mixture", "ons") + tuple(HYBRIDS)


class HybridBettor(_Bettor):
    """The five-point grid and an ONS bettor, mixed by wealth with prior mass ``prior_ons`` on ONS."""

    def __init__(self, shape: tuple[int, int], prior_ons: float) -> None:
        super().__init__(shape, "mixture", 0.5, 0.75)
        self._log_prior_grid = np.log((1.0 - prior_ons) / len(MIXTURE_GRID))
        self._log_prior_ons = np.log(prior_ons)
        self._ons_log_wealth = np.zeros(shape)

    def next(self) -> np.ndarray:
        lg = self._grid_log_wealth + self._log_prior_grid
        lo = self._ons_log_wealth + self._log_prior_ons
        top = np.maximum(lg.max(axis=0), lo)
        wg, wo = np.exp(lg - top), np.exp(lo - top)
        return ((wg * self._grid).sum(axis=0) + wo * self._ons_lam) / (wg.sum(axis=0) + wo)

    def observe(self, y: np.ndarray, used, live: np.ndarray) -> None:
        self._grid_log_wealth[:, live] += np.log1p(self._grid * y)[:, live]
        own = self._ons_lam.copy()  # the ONS component's own bet, not the mixed one that was placed
        self._ons_log_wealth[live] += np.log1p(own * y)[live]
        grad = -y / (1.0 + own * y)
        self._ons_a[live] += grad[live] ** 2
        step = own - _ONS_RATE * grad / self._ons_a
        self._ons_lam[live] = np.clip(step, 0.0, 0.5)[live]


def make(variant: str, m: int, n: int, alpha: float) -> RankConfidenceSequence:
    if variant in HYBRIDS:
        seq = RankConfidenceSequence(m, alpha, population_size=n, method="shortcut", bet="mixture")
        seq.wealth._bettor = HybridBettor((m, m), HYBRIDS[variant])
        return seq
    return RankConfidenceSequence(m, alpha, population_size=n, method="shortcut", bet=variant)


def one_order(args: tuple) -> dict:
    index, seed, benchmark, subset, alpha, every_frac, k = args
    scores, _ = load(benchmark)
    x_full = scores[:, choose(subset, leaderboard_score())]
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    x = x_full[np.random.default_rng(seed).permutation(n)]
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = set(int(t) for t in np.arange(every, n + 1, every)) | set(check) | {n}
    out = {"order": index, "seed": int(seed), "benchmark": benchmark, "subset": subset, "n_items": n,
           "true_pairs": int(true_pair.sum()), "variants": {}}
    for v in VARIANTS:
        seq = make(v, m, n, alpha)
        rec = {"ever_wrong": False, "checkpoints": {}}
        for t in range(n):
            seq.observe(x[t])
            if t + 1 in looks:
                rep = seq.report()
                rec["ever_wrong"] = rec["ever_wrong"] or rep.failed or bool((rep.dominance & false_pair).any())
                if t + 1 in check:
                    rec["checkpoints"][str(check[t + 1])] = _summary(rep.dominance, true_pair, false_pair)
        seq = make(v, m, n, alpha)
        for t in range(n):
            if seq.active.sum() <= 1:
                break
            seq.observe(np.where(seq.active, x[t], np.nan))
            if (t + 1) % every == 0 or t + 1 == n:
                rep = seq.report()
                known = (rep.upper <= k) | (rep.lower > k)
                for j in np.flatnonzero(known & seq.active):
                    seq.retire(int(j))
        rep = seq.report()
        known = (rep.upper <= k) | (rep.lower > k)
        rec["retire_topk"] = {"cost": seq.evaluations / (m * n), "goal": bool(known.all()),
                              "settled_frac": float(known.mean())}
        out["variants"][v] = rec
    return out


def check_references(runs: list[dict], saved: Path) -> list[str]:
    """The plain mixture and plain ONS must reproduce the saved bet run order by order."""
    old = {(r["benchmark"], r["subset"], r["order"]): r for r in json.loads(saved.read_text())["runs"]}
    problems = []
    for r in runs:
        o = old.get((r["benchmark"], r["subset"], r["order"]))
        if o is None:
            continue
        for v in ("mixture", "ons"):
            if r["variants"][v]["checkpoints"] != o["variants"][v]["checkpoints"]:
                problems.append(f"{r['benchmark']}/{r['subset']} order {r['order']} {v}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, default=50)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--saved", type=Path, default=Path("../results/bets_real/bets_orders50.json"))
    ap.add_argument("--out", type=Path, default=Path("../results/bets_real_online_newton"))
    args = ap.parse_args()
    started = time.time()
    jobs = []
    for b in args.benchmarks:
        for s in PANELS:
            seeds = np.random.SeedSequence([args.seed, BENCHMARKS.index(b), SUBSETS.index(s)]).generate_state(args.orders)
            jobs += [(i, int(seeds[i]), b, s, args.alpha, args.every_frac, args.k) for i in range(args.orders)]
    runs = []
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        for fut in as_completed([pool.submit(one_order, j) for j in jobs]):
            runs.append(fut.result())
            if len(runs) % 50 == 0:
                print(f"{len(runs)} of {len(jobs)} done, {time.time() - started:.0f}s", flush=True)
    problems = check_references(runs, args.saved) if args.saved.exists() else ["saved run not found"]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bets_hybrid_orders{args.orders}.json"
    path.write_text(json.dumps({"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                                "variants": list(VARIANTS), "hybrids": HYBRIDS, "reference_check": problems,
                                "runs": runs, "seconds": time.time() - started}))
    print("reference check (mixture and ONS against the saved bet run):", "all equal" if not problems else problems[:10])
    for b in args.benchmarks:
        for s in PANELS:
            rs = [r for r in runs if r["benchmark"] == b and r["subset"] == s]
            pct = lambda v, c: 100 * np.mean([r["variants"][v]["checkpoints"][c]["true_certified"] / r["true_pairs"] for r in rs])
            print(f"{b:<11}{s:<9}" + "  ".join(f"{v} half {pct(v, '0.5') - pct('mixture', '0.5'):+.2f} end {pct(v, '1.0') - pct('mixture', '1.0'):+.2f}" for v in VARIANTS[1:]))
    print("wrote", path, f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
