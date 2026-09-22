"""Plain-text summary of a dependence result file: validity per regime, efficiency at 0.50 and
0.75, retirement cost, and the mechanism (Var(X_j - X_l) against median certification time).

Usage:  python -m experiments.summarize_dependence ../results/dependence/dependence_full.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    args = ap.parse_args()
    res = json.loads(args.result.read_text())
    print(f"run time {res['seconds'] / 60:.0f} min; regimes: " + "; ".join(f"{k}: rho={v['rho']}, a={v['a']}" for k, v in res["regimes"].items()))
    for config, s in res["results"].items():
        print(f"\n===== {config}: accuracies {[round(v, 3) for v in s['accuracies']]}, {s['true_pairs']} true pairs, {s['replicates']} replicates =====")
        print(f"{'regime':12s} | anytime error (2 s.e.) | mean within-item corr | true frac 0.50 / 0.75 / 1.00 | width 0.50 / 0.75 | tiers 0.50 / 0.75 | top-3 resolved 0.50 / 0.75 | top-3 cost, goal")
        for regime, g in s["regimes"].items():
            corr = np.array(g["corr"]); off = corr[~np.eye(corr.shape[0], dtype=bool)]
            a = g["at"]
            def v(f, q): return a[f][q]["mean"]
            ret = f"{100*g['retire']['cost']:.0f}%, {100*g['retire']['goal']:.0f}%" if "retire" in g else "n/a"
            print(f"{regime:12s} | {100*g['ever_wrong']:.2f}% ({100*2*g['ever_wrong_se']:.2f}) | {off.mean():+.2f} [{off.min():+.2f},{off.max():+.2f}] | "
                  f"{100*v('0.5','true_frac'):.0f}% / {100*v('0.75','true_frac'):.0f}% / {100*v('1.0','true_frac'):.0f}% | "
                  f"{v('0.5','width'):.2f} / {v('0.75','width'):.2f} | {v('0.5','tiers'):.2f} / {v('0.75','tiers'):.2f} | "
                  f"{100*v('0.5','topk_resolved'):.0f}% / {100*v('0.75','topk_resolved'):.0f}% | {ret}")
        if s["true_pairs"]:
            print("  mechanism, per true pair: Var(X_j - X_l) and median first-certification fraction (never-certified share), by regime")
            pairs0 = s["regimes"]["independent"]["pairs"]
            for i, p in enumerate(pairs0):
                cells = []
                for regime, g in s["regimes"].items():
                    q = g["pairs"][i]
                    med = "never" if q["median_first_frac"] is None else f"{100*q['median_first_frac']:.0f}%"
                    cells.append(f"{regime}: var {q['var_diff']:.3f}, first {med} ({100*(1-q['certified_frac']):.0f}% never)")
                print(f"    pair {p['pair']} margin {p['margin']:.3f}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
