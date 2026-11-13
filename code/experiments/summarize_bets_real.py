"""Summaries of bets_real.py output (Table 16).

For each benchmark and subset (top 20, spread 20), each bet: percentage of true dominances
certified at half and at the end (mean over the same 50 item orders), the top-3 retirement cost,
and orders with a false statement. Differences from the default mixture are paired by item order,
with 95% normal intervals. The default mixture is checked against Figure 2. Table 16 gives, for
each bet, the range of these differences over the two subsets and six benchmarks.

Usage:  python -m experiments.summarize_bets_real ../results/bets_real/bets_orders50.json
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
LABEL = {"fixed@0.03": "0.03", "fixed@0.06": "0.06", "fixed@0.12": "0.12", "fixed@0.25": "0.25",
         "fixed@0.5": "0.5", "mixture": "default mixture", "dense": "50-point mixture",
         "agrapa": "plug-in", "ons": "online Newton"}


def pct(rs, v, f):
    tp = rs[0]["true_pairs"]
    return np.array([100.0 * r["variants"][v]["checkpoints"][f]["true_certified"] / tp for r in rs])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", type=Path)
    ap.add_argument("--reference", type=Path, default=Path("../results/real_leaderboard/real_leaderboard_orders50.json"))
    args = ap.parse_args()
    d = json.loads(args.result.read_text())
    variants = d["variants"]
    g = defaultdict(list)
    for r in d["runs"]:
        g[(r["benchmark"], r["subset"])].append(r)
    for k in g:
        g[k].sort(key=lambda r: r["order"])

    ref = json.loads(args.reference.read_text())["results"]
    bad = 0
    for (b, s), rs in g.items():
        for f in ("0.05", "0.1", "0.25", "0.5", "0.75", "1.0"):
            mine = np.mean([r["variants"]["mixture"]["checkpoints"][f]["true_certified"] for r in rs])
            bad += abs(mine - ref[f"{b}/{s}"]["ours"][f]["true_certified"]) > 1e-9
        mine = np.mean([r["variants"]["mixture"]["retire_topk"]["cost"] for r in rs])
        bad += abs(mine - ref[f"{b}/{s}"]["retire_topk"]["cost"]) > 1e-9
    print(f"default mixture reproduces Figure 2 and its top-3 cost: {'yes' if bad == 0 else f'NO ({bad})'}")

    wrong = sum(bool(r["variants"][v]["ever_wrong"]) for rs in g.values() for r in rs for v in variants)
    n_orders = len(g[("mmlu", "top20")])
    lines = []
    for f, fname in (("0.5", "half"), ("1.0", "the end")):
        for subset, slabel in (("top20", "twenty highest-scoring models"), ("spread20", "twenty spread models")):
            lines.append(f"**% of true dominances certified at {fname}, {slabel}, mean over {n_orders} item orders "
                         "(same orders and shortcut certifier for every bet; fixed benchmark).**\n")
            lines.append("| Bet | " + " | ".join(NAMES[b] for b in ORDER) + " |")
            lines.append("|---|" + "---:|" * len(ORDER))
            for v in variants:
                cells = []
                for b in ORDER:
                    rs = g[(b, subset)]
                    m = pct(rs, v, f).mean()
                    if v == "mixture":
                        cells.append(f"**{m:.1f}**")
                    else:
                        diff = pct(rs, v, f) - pct(rs, "mixture", f)
                        h = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
                        cells.append(f"{m:.1f} ({diff.mean():+.1f}±{h:.1f})")
                lines.append(f"| {LABEL[v]} | " + " | ".join(cells) + " |")
            lines.append("")
    lines.append(f"**Top-3 retirement cost, % of a full evaluation, mean over {n_orders} orders (twenty highest-scoring / "
                 "twenty spread models).**\n")
    lines.append("| Bet | " + " | ".join(NAMES[b] for b in ORDER) + " |")
    lines.append("|---|" + "---:|" * len(ORDER))
    for v in variants:
        cells = []
        for b in ORDER:
            c = [100 * np.mean([r["variants"][v]["retire_topk"]["cost"] for r in g[(b, s)]]) for s in ("top20", "spread20")]
            cells.append(f"{c[0]:.1f} / {c[1]:.1f}")
        lines.append(f"| {LABEL[v]} | " + " | ".join(cells) + " |")
    lines.append(f"\nOrders with a false statement at some look, over all bets and settings: {wrong} of "
                 f"{sum(len(rs) for rs in g.values()) * len(variants)}. Differences in parentheses are paired by item "
                 "order (bet minus default mixture, 95% interval).")
    md = "\n".join(lines) + "\n"
    print(md)


if __name__ == "__main__":
    main()
