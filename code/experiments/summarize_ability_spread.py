"""Summaries of ability_spread.py output (Table 6).

First table: for twenty models evenly spaced through the top 20, 40, 80, 160 and 395 ranks, the
percentage of true dominances certified at half and at the end, mean tiers at half, and orders
with a false statement (50 item orders). Spans 20 and 395 are checked against Figure 2.

Second table: every true pair of every panel, by its difficulty x = rho / (N Delta^2), where Delta
is the gap in mean score on the whole benchmark, rho = E[Z^2] the mean squared score difference,
and N the number of items. Under i.i.d. sampling Prop. B.4 puts the items needed at a multiple of
rho / Delta^2 (up to a log factor), so x is that scale as a fraction of the benchmark. On the
fixed benchmark it is descriptive only. Shown: the share of (pair, item order) cases certified by
a quarter, by half and by the end, in bins of x; pairs never certified count as not certified.

Usage:  python -m experiments.summarize_ability_spread ../results/ability_spread/spread_orders50.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

NAMES = {"mmlu": "MMLU", "hellaswag": "HellaSwag", "gsm8k": "GSM8K", "winogrande": "WinoGrande",
         "arc": "ARC-Challenge", "truthfulqa": "TruthfulQA"}
ORDER = ("mmlu", "hellaswag", "gsm8k", "winogrande", "arc", "truthfulqa")
SPANS = (20, 40, 80, 160, 395)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", type=Path)
    ap.add_argument("--reference", type=Path, default=Path("../results/real_leaderboard/real_leaderboard_orders50.json"))
    args = ap.parse_args()
    d = json.loads(args.result.read_text())
    runs, pairs = d["runs"], d["pairs"]
    g = defaultdict(list)
    for r in runs:
        g[(r["benchmark"], r["span"])].append(r)

    ref = json.loads(args.reference.read_text())["results"]
    bad = 0
    for b in ORDER:
        for span, sub in ((20, "top20"), (395, "spread20")):
            for f in ("0.05", "0.1", "0.25", "0.5", "0.75", "1.0"):
                mine = np.mean([r["checkpoints"][f]["true_certified"] for r in g[(b, span)]])
                if abs(mine - ref[f"{b}/{sub}"]["ours"][f]["true_certified"]) > 1e-9:
                    bad += 1
    print(f"spans 20 and 395 reproduce Figure 2: {'yes' if bad == 0 else f'NO ({bad} mismatches)'}")

    lines = [f"**Twenty models evenly spaced through the top 20 to all 395 ranks: % of true dominances certified "
             f"(ours, fixed benchmark, monitored every 1%), mean over {len(g[('mmlu', 20)])} item orders. "
             "Span 20 is the top-20 panel and span 395 the spread panel of Figure 2.**\n",
             "| Benchmark | " + " | ".join(f"span {s}: half / end" for s in SPANS) + " | tiers at half, span 20 → 395 |",
             "|---|" + "---:|" * len(SPANS) + "---|"]
    wrong = 0
    for b in ORDER:
        cells, tiers = [], []
        for s in SPANS:
            rs = g[(b, s)]
            tp = rs[0]["true_pairs"]
            h = np.mean([100 * r["checkpoints"]["0.5"]["true_certified"] / tp for r in rs])
            e = np.mean([100 * r["checkpoints"]["1.0"]["true_certified"] / tp for r in rs])
            cells.append(f"{h:.1f} / {e:.1f}")
            tiers.append(f"{np.mean([r['checkpoints']['0.5']['tiers'] for r in rs]):.1f}")
            wrong += sum(bool(r["ever_wrong"]) for r in rs)
        lines.append(f"| {NAMES[b]} | " + " | ".join(cells) + " | " + " → ".join(tiers) + " |")
    lines.append(f"\nOrders with a false statement at some look: {wrong} of {len(runs)}.")

    # pair-level difficulty
    xs, cert = [], {u: [] for u in (0.25, 0.5, 1.0)}
    per_bench = defaultdict(lambda: ([], {u: [] for u in (0.25, 0.5, 1.0)}))
    for (b, s), rs in g.items():
        info = pairs[f"{b}/{s}"]
        n = rs[0]["n_items"]
        x = np.array(info["rho"]) / (n * np.array(info["delta"]) ** 2)
        first = np.array([r["first_certified"] for r in rs])  # orders x pairs, -1 = never
        for u in cert:
            share = ((first >= 0) & (first <= u + 1e-9)).mean(axis=0)
            cert[u].append(share)
            per_bench[b][1][u].append(share)
        xs.append(x)
        per_bench[b][0].append(x)
    x = np.concatenate(xs)
    edges = np.logspace(np.log10(x.min()) - 1e-9, np.log10(x.max()) + 1e-9, 13)
    mids = np.sqrt(edges[:-1] * edges[1:])
    which = np.clip(np.digitize(x, edges) - 1, 0, len(mids) - 1)
    lines.append("\n**True pairs by difficulty x = ρ/(NΔ²), all panels and benchmarks pooled: share certified, in %.**\n")
    lines.append("| x (bin centre) | pairs | by a quarter | by half | by the end |")
    lines.append("|---:|---:|---:|---:|---:|")
    shares = {u: np.concatenate(cert[u]) for u in cert}
    for k, m in enumerate(mids):
        sel = which == k
        if sel.sum() == 0:
            continue
        lines.append(f"| {m:.3g} | {int(sel.sum())} | " + " | ".join(f"{100*shares[u][sel].mean():.0f}" for u in (0.25, 0.5, 1.0)) + " |")
    md = "\n".join(lines) + "\n"
    print(md)



if __name__ == "__main__":
    main()
