"""Checks of statements in Section 5 and Appendix E, read from saved results.

1. Ability spread (E2): pair-and-order cases certified by halfway below x = 0.01 and above x = 0.3,
   with x = rho / (N Delta^2) as in summarize_ability_spread.py.
2. All 395 models (E.1): how many of Holm's wrong orders first went wrong at the first look, and
   whether any recorded falsely ordered pair is tied.
3. All 395 models: the orders with more than one wrong look are recomputed look by look (same seeds,
   same Holm routine as monitored_full_leaderboard_schedules.py) to check that no falsely ordered
   pair is tied at any wrong look. The errors file records pairs only at each order's first wrong
   look.

Usage:  python -m experiments.real_leaderboard_checks --workers 6
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.monitored_full_leaderboard import holm_z_from_sums
from experiments.monitored_full_leaderboard_schedules import look_grid
from experiments.real_leaderboard import load

RES = Path("../results")


def spread_check() -> dict:
    d = json.loads((RES / "ability_spread" / "spread_orders50.json").read_text())
    g = defaultdict(list)
    for r in d["runs"]:
        g[(r["benchmark"], r["span"])].append(r)
    out = defaultdict(lambda: {"below_0.01": [0, 0], "above_0.3": [0, 0], "pairs_above_0.3_certified": 0,
                               "pairs_above_0.3": 0, "max_x_certified": 0.0})
    for (b, s), rs in g.items():
        info = d["pairs"][f"{b}/{s}"]
        x = np.array(info["rho"]) / (rs[0]["n_items"] * np.array(info["delta"]) ** 2)
        first = np.array([r["first_certified"] for r in rs])  # orders x pairs, -1 = never
        byhalf = (first >= 0) & (first <= 0.5 + 1e-9)
        lo, hi = x < 0.01, x > 0.3
        o = out[b]
        o["below_0.01"][0] += int(byhalf[:, lo].sum()); o["below_0.01"][1] += int(lo.sum() * len(rs))
        o["above_0.3"][0] += int(byhalf[:, hi].sum()); o["above_0.3"][1] += int(hi.sum() * len(rs))
        o["pairs_above_0.3"] += int(hi.sum())
        hit = byhalf[:, hi].any(axis=0)
        o["pairs_above_0.3_certified"] += int(hit.sum())
        if hit.any():
            o["max_x_certified"] = max(o["max_x_certified"], float(x[hi][hit].max()))
    return dict(out)


def first_look_check() -> dict:
    d = json.loads((RES / "monitored_full_leaderboard" / "holm395_errors.json").read_text())
    first = [r for r in d if r["fraction"] <= 0.0105]
    return {"orders": len(d), "wrong_at_first_look": len(first),
            "recorded_false_pairs": sum(len(r["pairs"]) for r in d),
            "tied_among_recorded": sum(p["tied"] for r in d for p in r["pairs"]),
            "orders_with_several_wrong_looks": sum(r["wrong_looks"] > 1 for r in d)}


def tied_at_wrong_looks(args: tuple) -> dict:
    benchmark, order, seed = args
    x_full, _ = load(benchmark)
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    tied = (np.abs(theta[:, None] - theta[None, :]) < 1e-9) & ~np.eye(m, dtype=bool)
    x = x_full[np.random.default_rng(seed).permutation(n)]
    looks, _ = look_grid(n, 0.01)
    s, a, gm = np.zeros(m), np.zeros(m), np.zeros((m, m))
    seen, wrong = 0, []
    for t in looks:
        chunk = x[seen:t]
        s += chunk.sum(axis=0); a += (chunk * chunk).sum(axis=0); gm += chunk.T @ chunk
        seen = t
        bad = holm_z_from_sums(t, s, a, gm, 0.05, x_seen=x[:t]) & false_pair
        if bad.any():
            wrong.append({"items": t, "false_pairs": int(bad.sum()), "tied": int((bad & tied).sum())})
    return {"benchmark": benchmark, "order": order, "wrong_looks": wrong}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=RES / "real_leaderboard_checks")
    args = ap.parse_args()
    report = {"spread": spread_check(), "first_look": first_look_check()}
    errs = json.loads((RES / "monitored_full_leaderboard" / "holm395_errors.json").read_text())
    jobs = [(r["benchmark"], r["order"], r["seed"]) for r in errs if r["wrong_looks"] > 1]
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        report["several_wrong_looks"] = list(pool.map(tied_at_wrong_looks, jobs))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "real_leaderboard_checks.json"
    path.write_text(json.dumps(report, indent=1))
    sp = report["spread"]
    print("spread, below 0.01 certified by half:", sum(v["below_0.01"][0] for v in sp.values()), "of",
          sum(v["below_0.01"][1] for v in sp.values()))
    print("spread, above 0.3 certified by half:", sum(v["above_0.3"][0] for v in sp.values()), "of",
          sum(v["above_0.3"][1] for v in sp.values()), "| pairs:", sum(v["pairs_above_0.3_certified"] for v in sp.values()),
          "of", sum(v["pairs_above_0.3"] for v in sp.values()))
    print("first look:", report["first_look"])
    print("tied at any wrong look of the multi-look orders:",
          sum(w["tied"] for o in report["several_wrong_looks"] for w in o["wrong_looks"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
