"""Summaries of close_pairs.py output: close real pairs (Table 2, and the real pairs of Table 4).

Pair categories, by the gap between the two models' total scores on the whole benchmark:
  tied        equal totals, score vectors not identical (a test can wrongly name a winner);
  identical   identical score vectors (no test can name a winner; left out of Table 2);
  near        totals more than 0 and at most 5 apart (items on the binary benchmarks, score-sum
              units on TruthfulQA), the column "unequal, within 5 points" of Table 2;
  far         more than 5 apart.
For each category, schedule and test: the mean over pairs of each pair's error rate across item
orders. All pairs share each item order, so the 95% interval resamples item orders (cluster
bootstrap). Windows: the fraction of windows with any false dominance, with the same interval.

Usage:  python -m experiments.summarize_close_pairs ../results/close_pairs/pairs_*_orders0-199.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TESTS = ("z", "mcnemar", "ours")
BOOT = 2000


def categories(meta: dict) -> dict[str, np.ndarray]:
    gap = np.array(meta["gap"])
    ident = np.array(meta["identical"])
    return {"tied": (gap == 0) & ~ident, "identical": ident, "near": (gap > 0) & (gap <= 5), "far": gap > 5}


def indicator(runs: list[dict], getter, size: int) -> np.ndarray:
    m = np.zeros((len(runs), size), dtype=bool)
    for i, r in enumerate(runs):
        idx = getter(r)
        if idx:
            m[i, idx] = True
    return m


def mean_ci(per_order: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean over orders of a per-order quantity, with a 95% bootstrap interval over orders."""
    n = per_order.size
    boot = per_order[rng.integers(0, n, size=(BOOT, n))].mean(axis=1)
    return float(per_order.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarize(path: Path, rng: np.random.Generator) -> dict:
    d = json.loads(path.read_text())
    meta, runs = d["meta"], d["runs"]
    runs = sorted(runs, key=lambda r: r["order"])
    n_pairs = len(meta["pairs"])
    cats = categories(meta)
    out = {"benchmark": runs[0]["benchmark"], "orders": len(runs), "binary": meta["binary"],
           "n_items": meta["n_items"], "counts": {c: int(v.sum()) for c, v in cats.items()},
           "pairs": {}, "single_look": {}, "windows": {}}
    tests = [t for t in TESTS if meta["binary"] or t != "mcnemar"]
    for sched in list(meta["schedules"]) + ["every_item"]:
        out["pairs"][sched] = {}
        for t in tests if sched != "every_item" else ["ours"]:
            e = indicator(runs, lambda r: r["pairs"][sched][t], n_pairs)
            out["pairs"][sched][t] = {}
            for c, mask in cats.items():
                if mask.sum() == 0:
                    continue
                per_order = e[:, mask].mean(axis=1)
                mean, lo, hi = mean_ci(per_order, rng)
                per_pair = e[:, mask].mean(axis=0)
                out["pairs"][sched][t][c] = {"mean": mean, "lo": lo, "hi": hi,
                                             "max_pair": float(per_pair.max()),
                                             "pairs_over_5pct": int((per_pair > 0.05).sum())}
    # single-look control: each look of the 100-look schedule on its own, tied pairs
    looks = meta["schedules"]["equal100"]
    for t in tests:
        rates = []
        for look in looks:
            e = indicator(runs, lambda r: r["single_look"][t][str(look)], n_pairs)
            rates.append(float(e[:, cats["tied"]].mean()) if cats["tied"].any() else float("nan"))
        out["single_look"][t] = {"looks": looks, "tied_rate": rates}
    if "windows" in runs[0]:
        for w, n_win in meta["windows"].items():
            out["windows"][w] = {}
            for sched in ("equal100", "half"):
                out["windows"][w][sched] = {}
                for t in tests:
                    e = indicator(runs, lambda r: r["windows"][w][sched][t], n_win)
                    mean, lo, hi = mean_ci(e.mean(axis=1), rng)
                    out["windows"][w][sched][t] = {"mean": mean, "lo": lo, "hi": hi, "n_windows": n_win}
    return out


def pct(x: dict) -> str:
    return f"{100*x['mean']:.1f} [{100*x['lo']:.1f}, {100*x['hi']:.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("../results/close_pairs/pairs_summary.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(20261108)
    summ = [summarize(p, rng) for p in args.results]
    args.out.write_text(json.dumps(summ, indent=1))
    for s in summ:
        print(f"\n=== {s['benchmark']} ({s['orders']} orders, N={s['n_items']}) pairs by category: {s['counts']}")
        for c in ("tied", "near", "far"):
            if c not in s["pairs"]["equal100"]["z"]:
                continue
            row = [f"{t} 100 looks {pct(s['pairs']['equal100'][t][c])}, half {pct(s['pairs']['half'][t][c])}"
                   for t in s["pairs"]["equal100"]]
            print(f"  {c:5s}: " + " | ".join(row) + f" | ours every item {pct(s['pairs']['every_item']['ours'][c])}")
        if "tied" in s["pairs"]["equal100"]["z"]:
            print("  per-pair max (100 looks): " + ", ".join(f"{t}: {100*s['pairs']['equal100'][t]['tied']['max_pair']:.0f}%"
                                                        for t in s["pairs"]["equal100"]))
            sl = s["single_look"]["z"]["tied_rate"]
            print(f"  single-look z, tied pairs: first five looks {[round(100*v, 1) for v in sl[:5]]}, max after 10% "
                  f"{100*max(sl[10:]):.1f}%, at half {100*sl[49]:.1f}%")
            for sched in ("equal5", "equal10", "equal20", "equal50", "early10", "late10", "early20", "late20"):
                print(f"  {sched:8s} tied z {pct(s['pairs'][sched]['z']['tied'])}" +
                      (f"  mcnemar {pct(s['pairs'][sched]['mcnemar']['tied'])}" if "mcnemar" in s["pairs"][sched] else ""))
        for w, d in s["windows"].items():
            print(f"  windows of {w} ({d['equal100']['z']['n_windows']}): " + " | ".join(
                f"{t} 100 looks {pct(d['equal100'][t])}, half {pct(d['half'][t])}" for t in d["equal100"]))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
