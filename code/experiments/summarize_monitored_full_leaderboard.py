"""Summaries of monitored_full_leaderboard.py output (all 395 models).

Holm with paired z-tests over all 395 models, closed under transitivity and recomputed at every look
(every 1% of the benchmark plus the six checkpoints), against the same Holm sets at the halfway look
alone, and against ours (shortcut) and e-Bonferroni monitored at the same looks. A run is wrong if
any look certifies a false dominance. Rates use each benchmark's longest complete prefix of item
orders (orders 0..K-1), with exact Clopper-Pearson 95% intervals.

Top-3 retirement cost of ours against e-Bonferroni on the same item orders, each certifier making
its own retirement decisions; differences are paired by order.

Table 1 with e-Bonferroni: true dominances certified at half and at the end, and tiers at half, for
ours and e-Bonferroni on all 395 models, from the same monitored wealths.

Reproduction checks: ours, Holm at the six checkpoints and the top-3 retirement must match
full_leaderboard_orders50.json (method shortcut) once orders 0-49 are complete; e-Bonferroni is
checked against full_leaderboard_orders10.json (method bonferroni) on orders 0-9 when that file is
present. Writes full395_summary.json next to the runs and prints the tables.

Usage:  python -m experiments.summarize_monitored_full_leaderboard
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

NAMES = {"mmlu": "MMLU", "hellaswag": "HellaSwag", "gsm8k": "GSM8K", "winogrande": "WinoGrande",
         "arc": "ARC-Challenge", "truthfulqa": "TruthfulQA"}
ORDER = ("mmlu", "hellaswag", "gsm8k", "winogrande", "arc", "truthfulqa")
FIELDS = ("true_certified", "false_certified", "tiers", "width", "topk_settled")


def clopper_pearson(x: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = 1.0 - level
    lo = 0.0 if x == 0 else float(stats.beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - a / 2, x + 1, n - x))
    return lo, hi


def prefix(orders) -> int:
    have, k = set(orders), 0
    while k in have:
        k += 1
    return k


def pct_ci(x: int, n: int) -> str:
    lo, hi = clopper_pearson(x, n)
    return f"{x} ({100 * x / n:.1f}%; {100 * lo:.1f} to {100 * hi:.1f})"


EVENT_FIELDS = ("ever_wrong", "first_wrong", "wrong_looks", "half_only_wrong", "n_looks")


def load_holm_events(d: Path) -> tuple[dict, dict, dict]:
    """Holm's error event per benchmark and order, from the full run (orders 0-49) and the
    Holm-only runs. A benchmark rerun with the exact paired variance (``d / "exact"``) replaces
    the first runs; the first runs' events are returned as ``raw`` for the record."""
    full = json.loads((d / "full395_orders0-49.json").read_text())["runs"]
    h = defaultdict(dict)
    for p in sorted(d.glob("holm395_orders*.json")):
        for r in json.loads(p.read_text())["runs"]:
            h[r["benchmark"]][r["order"]] = {q: r[q] for q in EVENT_FIELDS}
    for r in full:
        assert r["order"] not in h[r["benchmark"]], f"order {r['order']} of {r['benchmark']} is in both runs"
        h[r["benchmark"]][r["order"]] = {q: r["holm"][q] for q in EVENT_FIELDS if q != "n_looks"} | {"n_looks": r["n_looks"]}
    raw, source = {}, {b: "first runs" for b in h}
    for p in sorted((d / "exact").glob("holm395_orders*.json")):
        exact = defaultdict(dict)
        for r in json.loads(p.read_text())["runs"]:
            exact[r["benchmark"]][r["order"]] = {q: r[q] for q in EVENT_FIELDS}
        for b, ev in exact.items():
            raw[b] = h[b]
            h[b] = ev
            source[b] = f"exact rerun ({p.name})"
    return h, raw, source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("../results/monitored_full_leaderboard"))
    ap.add_argument("--reference50", type=Path, default=Path("../results/full_leaderboard/full_leaderboard_orders50.json"))
    ap.add_argument("--reference10", type=Path, default=Path("../results/full_leaderboard/full_leaderboard_orders10.json"))
    args = ap.parse_args()

    full = json.loads((args.dir / "full395_orders0-49.json").read_text())["runs"]
    ref50 = json.loads(args.reference50.read_text())["benchmarks"]
    ref10 = json.loads(args.reference10.read_text())["benchmarks"] if args.reference10.exists() else None

    g = defaultdict(dict)  # benchmark -> order -> full record
    for r in full:
        g[r["benchmark"]][r["order"]] = r
    h, raw, source = load_holm_events(args.dir)  # benchmark -> order -> Holm event record

    # reproduction checks against the saved full-leaderboard runs (same seeds, prefix-consistent)
    checks = {}
    for b in ORDER:
        k_full = prefix(g[b])
        msgs = []
        if k_full >= 10 and ref10 is not None:
            rs = [g[b][o] for o in range(10)]
            bad = 0
            for f, ref in ref10[b]["ours"].items():
                for q in FIELDS:
                    mine = np.mean([r["monitored"]["bonferroni"]["checkpoints"][f][q] for r in rs])
                    bad += abs(mine - ref[q]) > 1e-6 * max(1.0, abs(ref[q]))
            for q in ("cost", "goal", "settled_frac", "items_seen"):
                mine = np.mean([r["retire_topk"]["bonferroni"][q] for r in rs])
                ref = ref10[b]["retire_topk"][q]
                bad += abs(mine - ref) > 1e-6 * max(1.0, abs(ref))
            msgs.append(f"e-Bonferroni, orders 0-9 vs full_leaderboard_orders10: {'match' if bad == 0 else f'{bad} MISMATCHES'}")
        if k_full >= 50:
            rs = [g[b][o] for o in range(50)]
            bad = 0
            for key, src in (("ours", lambda r: r["monitored"]["shortcut"]["checkpoints"]), ("holm", lambda r: r["holm"]["checkpoints"])):
                for f, ref in ref50[b][key].items():
                    for q in FIELDS:
                        mine = np.mean([src(r)[f][q] for r in rs])
                        bad += abs(mine - ref[q]) > 1e-6 * max(1.0, abs(ref[q]))
            for q in ("cost", "goal", "settled_frac", "items_seen"):
                mine = np.mean([r["retire_topk"]["shortcut"][q] for r in rs])
                ref = ref50[b]["retire_topk"][q]
                bad += abs(mine - ref) > 1e-6 * max(1.0, abs(ref))
            msgs.append(f"ours, Holm checkpoints and top-3 retirement, orders 0-49 vs full_leaderboard_orders50: "
                        f"{'match' if bad == 0 else f'{bad} MISMATCHES'}")
        checks[b] = msgs or [f"only {k_full} complete orders, no check yet"]

    summary = {"benchmarks": {}}
    for b in ORDER:
        kh = prefix(h[b])
        hs = [h[b][o] for o in range(kh)]
        kf = prefix(g[b])
        fs = [g[b][o] for o in range(kf)]
        first = [x["first_wrong"] for x in hs if x["ever_wrong"]]
        n_items = ref50[b]["n_items"]
        first_look = round(0.01 * n_items) / n_items
        tp = ref50[b]["true_pairs"]
        s = {"n_items": n_items, "true_pairs": tp, "n_looks": sorted({x["n_looks"] for x in hs}),
             "holm_orders": kh, "holm_wrong": sum(x["ever_wrong"] for x in hs),
             "holm_wrong_first_look": sum(abs(f - first_look) < 1e-9 for f in first),
             "holm_first_wrong": sorted(first),
             "holm_wrong_looks": sorted(x["wrong_looks"] for x in hs if x["ever_wrong"]),
             "holm_half_only_wrong": sum(bool(x["half_only_wrong"]) for x in hs),
             "holm_source": source.get(b, "first runs"),
             "full_orders": kf,
             "ours_wrong": sum(r["monitored"]["shortcut"]["ever_wrong"] for r in fs),
             "ebonf_wrong": sum(r["monitored"]["bonferroni"]["ever_wrong"] for r in fs),
             "retire_wrong": {c: sum(r["retire_topk"][c]["wrong"] for r in fs) for c in ("shortcut", "bonferroni")}}
        if kf:
            cost = {c: np.array([100 * r["retire_topk"][c]["cost"] for r in fs]) for c in ("shortcut", "bonferroni")}
            diff = cost["bonferroni"] - cost["shortcut"]
            hw = 1.96 * diff.std(ddof=1) / np.sqrt(kf) if kf > 1 else float("nan")
            s["retire"] = {c: {"cost_pct": float(cost[c].mean()),
                               "goal": float(np.mean([r["retire_topk"][c]["goal"] for r in fs])),
                               "settled_frac": float(np.mean([r["retire_topk"][c]["settled_frac"] for r in fs])),
                               "items_seen": float(np.mean([r["retire_topk"][c]["items_seen"] for r in fs]))}
                           for c in ("shortcut", "bonferroni")}
            s["retire"]["diff_pct"] = {"mean": float(diff.mean()), "halfwidth": float(hw),
                                       "max_abs": float(np.abs(diff).max()),
                                       "ebonf_higher": int((diff > 1e-12).sum()), "equal": int((np.abs(diff) <= 1e-12).sum()),
                                       "ebonf_lower": int((diff < -1e-12).sum())}
            t1 = {}
            for c in ("shortcut", "bonferroni"):
                cp = lambda f, q: np.array([r["monitored"][c]["checkpoints"][f][q] for r in fs])
                t1[c] = {"half_pct": float(100 * cp("0.5", "true_certified").mean() / tp),
                         "end_pct": float(100 * cp("1.0", "true_certified").mean() / tp),
                         "tiers_half": float(cp("0.5", "tiers").mean()),
                         "width_half": float(cp("0.5", "width").mean())}
            d_half = np.array([r["monitored"]["bonferroni"]["checkpoints"]["0.5"]["true_certified"]
                               - r["monitored"]["shortcut"]["checkpoints"]["0.5"]["true_certified"] for r in fs])
            d_end = np.array([r["monitored"]["bonferroni"]["checkpoints"]["1.0"]["true_certified"]
                              - r["monitored"]["shortcut"]["checkpoints"]["1.0"]["true_certified"] for r in fs])
            t1["diff_pairs"] = {"half_mean": float(d_half.mean()), "half_min": int(d_half.min()),
                                "end_mean": float(d_end.mean()), "end_min": int(d_end.min()), "max_positive": int(max(d_half.max(), d_end.max()))}
            s["table1"] = t1
        if b in raw:
            kr = prefix(raw[b])
            s["holm_first_runs"] = {"orders": kr, "wrong": sum(raw[b][o]["ever_wrong"] for o in range(kr)),
                                    "half_only_wrong": sum(bool(raw[b][o]["half_only_wrong"]) for o in range(kr))}
        summary["benchmarks"][b] = s
    summary["checks"] = checks

    for b in ORDER:
        for m in checks[b]:
            print(f"{NAMES[b]}: {m}")
        print(f"{NAMES[b]}: Holm events from {summary['benchmarks'][b]['holm_source']}"
              + (f"; first runs: {summary['benchmarks'][b]['holm_first_runs']}" if b in raw else ""))

    B = summary["benchmarks"]
    lines = ["**All 395 models: item orders with a false dominance at some look (count, % and exact 95% "
             "interval).** Holm over the 155,630 ordered pairs with paired $z$-tests, closed under transitivity. "
             "*Every look*: recomputed at every 1% of the benchmark and at the six checkpoints. *Half only*: the same "
             "sets at the halfway look alone. Ours and e-Bonferroni are monitored at the same looks. Orders are each "
             "benchmark's first $K$ orders, with the seeds of Table 1.\n",
             "| | " + " | ".join(NAMES[b] for b in ORDER) + " |", "|---|" + "---:|" * len(ORDER),
             "| Items $N$ (first look) | " + " | ".join(f"{B[b]['n_items']:,} ({round(0.01 * B[b]['n_items'])})" for b in ORDER) + " |",
             "| Looks | " + " | ".join("–".join(str(v) for v in B[b]["n_looks"]) for b in ORDER) + " |",
             "| Holm orders $K$ | " + " | ".join(str(B[b]["holm_orders"]) for b in ORDER) + " |",
             "| Holm, every look | " + " | ".join(pct_ci(B[b]["holm_wrong"], B[b]["holm_orders"]) for b in ORDER) + " |",
             "| of which first wrong at the first look | " + " | ".join(str(B[b]["holm_wrong_first_look"]) for b in ORDER) + " |",
             "| Holm, half only | " + " | ".join(pct_ci(B[b]["holm_half_only_wrong"], B[b]["holm_orders"]) for b in ORDER) + " |",
             "| Ours and e-Bonferroni orders | " + " | ".join(str(B[b]["full_orders"]) for b in ORDER) + " |",
             "| Ours, every look | " + " | ".join(pct_ci(B[b]["ours_wrong"], B[b]["full_orders"]) for b in ORDER) + " |",
             "| e-Bonferroni, every look | " + " | ".join(pct_ci(B[b]["ebonf_wrong"], B[b]["full_orders"]) for b in ORDER) + " |",
             ""]
    lines += ["**Top-3 retirement on all 395 models: ours against e-Bonferroni on the same pairwise e-processes, "
              "item orders and retirement rule, each certifier making its own retirement decisions.** Cost in % of a full "
              "evaluation ($395N$ model-item evaluations); difference e-Bonferroni minus ours, paired by order.\n",
              "| | " + " | ".join(NAMES[b] for b in ORDER) + " |", "|---|" + "---:|" * len(ORDER),
              "| Orders | " + " | ".join(str(B[b]["full_orders"]) for b in ORDER) + " |",
              "| Cost, ours | " + " | ".join(f"{B[b]['retire']['shortcut']['cost_pct']:.3f}" for b in ORDER) + " |",
              "| Cost, e-Bonferroni | " + " | ".join(f"{B[b]['retire']['bonferroni']['cost_pct']:.3f}" for b in ORDER) + " |",
              "| Difference (95% interval) | " + " | ".join(f"{B[b]['retire']['diff_pct']['mean']:+.4f} ± {B[b]['retire']['diff_pct']['halfwidth']:.4f}" for b in ORDER) + " |",
              "| Largest difference in one order | " + " | ".join(f"{B[b]['retire']['diff_pct']['max_abs']:.3f}" for b in ORDER) + " |",
              "| Orders e-Bonferroni higher / equal / lower | " + " | ".join(f"{B[b]['retire']['diff_pct']['ebonf_higher']} / {B[b]['retire']['diff_pct']['equal']} / {B[b]['retire']['diff_pct']['ebonf_lower']}" for b in ORDER) + " |",
              "| Goal reached, ours / e-Bonferroni (% of orders) | " + " | ".join(f"{100 * B[b]['retire']['shortcut']['goal']:.0f} / {100 * B[b]['retire']['bonferroni']['goal']:.0f}" for b in ORDER) + " |",
              "| Models settled, ours / e-Bonferroni (%) | " + " | ".join(f"{100 * B[b]['retire']['shortcut']['settled_frac']:.1f} / {100 * B[b]['retire']['bonferroni']['settled_frac']:.1f}" for b in ORDER) + " |",
              "| Orders with a false statement, ours / e-Bonferroni | " + " | ".join(f"{B[b]['retire_wrong']['shortcut']} / {B[b]['retire_wrong']['bonferroni']}" for b in ORDER) + " |",
              ""]
    lines += ["**Table 1 on all 395 models, with e-Bonferroni.** True dominances certified, in % of those "
              "that exist, and mean tiers at half; both certifiers on the same monitored wealths, every 1%.\n",
              "| | Ours: half | end | tiers | e-Bonferroni: half | end | tiers | Pairs ours certifies and e-Bonferroni does not: half / end |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b in ORDER:
        t = B[b]["table1"]
        lines.append(f"| {NAMES[b]} | {t['shortcut']['half_pct']:.1f} | {t['shortcut']['end_pct']:.1f} | {t['shortcut']['tiers_half']:.1f} | "
                     f"{t['bonferroni']['half_pct']:.1f} | {t['bonferroni']['end_pct']:.1f} | {t['bonferroni']['tiers_half']:.1f} | "
                     f"{0.0 - t['diff_pairs']['half_mean'] + 0.0:.2f} / {0.0 - t['diff_pairs']['end_mean'] + 0.0:.2f} |")
        if t["diff_pairs"]["max_positive"] > 0:
            print(f"WARNING: e-Bonferroni certified more than ours in some order of {b} (Prop. B.8 violated)")
    lo, hi = min(B[b]["full_orders"] for b in ORDER), max(B[b]["full_orders"] for b in ORDER)
    lines.append(f"\nMean over {lo if lo == hi else f'{lo} to {hi}'} item orders. "
                 "The shortcut never certifies less than e-Bonferroni on the same wealths (Prop. B.8).")
    md = "\n".join(lines) + "\n"
    (args.dir / "full395_summary.json").write_text(json.dumps(summary, indent=1))
    print(md)


if __name__ == "__main__":
    main()
