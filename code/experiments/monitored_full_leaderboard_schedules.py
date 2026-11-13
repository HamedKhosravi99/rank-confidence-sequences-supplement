"""Holm on all 395 models under different look schedules (the all-395 column of Table 3).

Is Holm's error under monitoring a small-sample effect of the earliest looks, or does it persist
once those looks are dropped? For the same 1,000 item orders per benchmark as
monitored_full_leaderboard.py, this counts the orders in which Holm certifies a false dominance under

  * looks every 1% starting at 1% (the monitored run itself), 5%, 10%, 25% or 50%;
  * one pre-planned look at 1%, 5%, 10%, 25% or 50%.

Every schedule uses looks from the same grid (every 1% of the benchmark plus the 5%, 10%, 25%, 50%,
75% and 100% checkpoints), so an order with no wrong look on the full grid is wrong under none of
them. Only the orders that erred on the full grid (holm395_errors.json, 88 orders) are therefore
recomputed, look by look, with the same seeds and the exact paired variance. Each recomputation must
reproduce the saved first wrong look and number of wrong looks, and a few orders without error per
benchmark are recomputed as a check that they stay error-free.

Usage:  python -m experiments.monitored_full_leaderboard_schedules --workers 6
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.real_leaderboard import BENCHMARKS, load
from experiments.monitored_full_leaderboard import FRACTIONS, holm_z_from_sums
from experiments.summarize_monitored_full_leaderboard import load_holm_events

STARTS = (0.01, 0.05, 0.10, 0.25, 0.50)
SCHEDULES = [f"every 1% from {f}" for f in STARTS] + [f"one look at {f}" for f in STARTS]


def look_grid(n: int, every_frac: float) -> tuple[list[int], dict[float, int]]:
    """The monitored run's looks, and the look used for each start fraction (1% is the first look)."""
    every = max(1, int(round(every_frac * n)))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = sorted(set(int(t) for t in np.arange(every, n + 1, every)) | set(check) | {n})
    at = {f: (every if f == 0.01 else int(round(f * n))) for f in STARTS}
    assert all(t in looks for t in at.values())
    return looks, at


def wrong_looks(args: tuple) -> dict:
    benchmark, order, seed, alpha, every_frac = args
    x_full, _ = load(benchmark)
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    x = x_full[np.random.default_rng(seed).permutation(n)]
    looks, at = look_grid(n, every_frac)
    s, a, g = np.zeros(m), np.zeros(m), np.zeros((m, m))
    seen, wrong = 0, []
    for t in looks:
        chunk = x[seen:t]
        s += chunk.sum(axis=0)
        a += (chunk * chunk).sum(axis=0)
        g += chunk.T @ chunk
        seen = t
        bad = holm_z_from_sums(t, s, a, g, alpha, x_seen=x[:t]) & false_pair
        if bad.any():
            wrong.append({"items": t, "fraction": t / n, "false_pairs": int(bad.sum())})
    return {"benchmark": benchmark, "order": order, "seed": seed, "n_items": n, "n_looks": len(looks),
            "start_items": {str(f): t for f, t in at.items()}, "wrong": wrong}


def schedule_counts(rec: dict) -> dict:
    """Is this order wrong under each schedule?"""
    items = [w["items"] for w in rec["wrong"]]
    out = {}
    for f, t0 in rec["start_items"].items():
        out[f"every 1% from {f}"] = any(t >= t0 for t in items)
        out[f"one look at {f}"] = t0 in items
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("../results/monitored_full_leaderboard"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--controls", type=int, default=3, help="error-free orders per benchmark to recheck")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    started = time.time()

    events, _, source = load_holm_events(args.dir)
    errors = json.loads((args.dir / "holm395_errors.json").read_text())
    orders = {b: sorted(events[b]) for b in events}
    for b, os_ in orders.items():
        assert os_ == list(range(len(os_))), f"{b}: orders are not 0..{len(os_) - 1}"
    seeds = {b: np.random.SeedSequence([args.seed, BENCHMARKS.index(b)]).generate_state(len(orders[b]))
             for b in orders}
    wrong_set = {(b, o) for b in events for o, e in events[b].items() if e["ever_wrong"]}
    assert wrong_set == {(r["benchmark"], r["order"]) for r in errors}, "error records and events disagree"

    jobs = []
    for r in errors:
        assert int(seeds[r["benchmark"]][r["order"]]) == r["seed"], (r["benchmark"], r["order"])
        jobs.append((r["benchmark"], r["order"], r["seed"], args.alpha, args.every_frac))
    controls = []
    for b in orders:
        clean = [o for o in orders[b] if not events[b][o]["ever_wrong"]][:args.controls]
        controls += [(b, o, int(seeds[b][o]), args.alpha, args.every_frac) for o in clean]
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        recs = list(pool.map(wrong_looks, jobs + controls))
    err_recs, ctl_recs = recs[:len(jobs)], recs[len(jobs):]

    # reproduction checks against the saved runs
    by_key = {(r["benchmark"], r["order"]): r for r in errors}
    for rec in err_recs:
        saved = by_key[(rec["benchmark"], rec["order"])]
        assert rec["wrong"], f"{rec['benchmark']} order {rec['order']}: no wrong look on recomputation"
        assert rec["wrong"][0]["items"] == saved["items_seen"], (rec["benchmark"], rec["order"])
        assert len(rec["wrong"]) == saved["wrong_looks"], (rec["benchmark"], rec["order"])
        ev = events[rec["benchmark"]][rec["order"]]
        assert abs(rec["wrong"][0]["fraction"] - ev["first_wrong"]) < 1e-12
        assert rec["n_looks"] == ev["n_looks"]
        half = rec["start_items"]["0.5"]
        assert (half in [w["items"] for w in rec["wrong"]]) == bool(ev["half_only_wrong"])
    for rec in ctl_recs:
        assert not rec["wrong"], f"{rec['benchmark']} order {rec['order']}: control order erred"

    # counts per benchmark, over all orders (orders without error are wrong under no schedule)
    table = {}
    for b in BENCHMARKS:
        if b not in orders:
            continue
        rs = [r for r in err_recs if r["benchmark"] == b]
        n_items = load(b)[0].shape[0]
        looks, at = look_grid(n_items, args.every_frac)
        row = {"orders": len(orders[b]), "n_items": n_items,
               "start_items": {str(f): t for f, t in at.items()},
               "looks_from": {str(f): sum(t >= t0 for t in looks) for f, t0 in at.items()}}
        counts = defaultdict(int)
        for r in rs:
            for k, v in schedule_counts(r).items():
                counts[k] += int(v)
        row["wrong_orders"] = {k: counts[k] for k in SCHEDULES}
        table[b] = row

    out = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "source": source, "table": table, "orders": err_recs, "controls_checked": len(ctl_recs),
           "seconds": time.time() - started}
    path = args.dir / "holm395_schedules.json"
    path.write_text(json.dumps(out, indent=1))

    print(f"{'schedule':<24}" + "".join(f"{b:>12}" for b in table) + f"{'total':>8}")
    for k in SCHEDULES:
        vals = [table[b]["wrong_orders"].get(k, 0) for b in table]
        print(f"{k:<24}" + "".join(f"{v:>12}" for v in vals) + f"{sum(vals):>8}")
    print(f"{'items at 5% / 10%':<24}" + "".join(
        f"{table[b]['start_items']['0.05']:>6}/{table[b]['start_items']['0.1']:<5}" for b in table))
    print(f"reproduced {len(err_recs)} error orders and {len(ctl_recs)} error-free controls; "
          f"wrote {path} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
