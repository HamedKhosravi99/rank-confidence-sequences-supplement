"""Figures 2 and 3: the real leaderboard on a finite benchmark with e-Bonferroni, and under i.i.d. sampling.

real_leaderboard: Figure 2, drawn from the fixed-benchmark runs (design F) of
real_leaderboard_sampling.py, with e-Bonferroni of Gu et al. Design F repeats real_leaderboard.py
order by order, so ours, Holm and Romano-Wolf are the curves of real_leaderboard_orders50.json.
Before drawing, the script checks every plotted value and both halfway tiers against
results/real_leaderboard/real_leaderboard_orders50.json and stops if any of them differ.
iid_top20: Figure 3, the same figure under i.i.d. sampling (design S), with the halfway tiers of
the S runs.

Twenty highest-scoring models; means over the 50 item orders. Layout, size, fonts, labels, tier
annotations and the Holm and Romano-Wolf lines follow plot_real_leaderboard.py. e-Bonferroni
nearly coincides with ours (within 0.04 points), so ours is drawn as a wider translucent line and
e-Bonferroni as a thin line on top.

Usage:  python -m experiments.plot_real_leaderboard_sampling ../results/real_leaderboard_sampling/fig2_orders50.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Copied from plot_real_leaderboard.py.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
TITLES = {"mmlu": "MMLU (14,042 items)", "hellaswag": "HellaSwag (10,042)", "gsm8k": "GSM8K (1,319)",
          "winogrande": "WinoGrande (1,267)", "arc": "ARC-Challenge (1,172)", "truthfulqa": "TruthfulQA (817)"}
ORDER = ("mmlu", "hellaswag", "gsm8k", "winogrande", "arc", "truthfulqa")
SUBSET = "top20"

# The paper's lines (plot_real_leaderboard.py), with ours made wider and translucent, and e-Bonferroni added.
LABELS = {"ours": "rank confidence sequence (monitored)",
          "ebonf": "e-Bonferroni of Gu et al. (monitored)",
          "fixed_z": "fixed-sample Holm $z$-test sets, one look",
          "fixed_rw": "fixed-sample Romano–Wolf step-down sets, one look"}
STYLE = {"ours": dict(color="#2a78d6", marker="o", markersize=5.6, markeredgewidth=0.8, linewidth=4.0, alpha=0.45,
                      solid_capstyle="round", solid_joinstyle="round"),
         "ebonf": dict(color="#008300", marker="x", markersize=3.0, markeredgewidth=0.9, linewidth=0.9),
         "fixed_z": dict(color="#eda100", marker="D", markersize=3.8, linewidth=1.5, linestyle="--",
                         markerfacecolor="white"),
         "fixed_rw": dict(color="#e87ba4", marker="v", markersize=3.8, linewidth=1.5, linestyle=":",
                          markerfacecolor="white")}
DRAW_ORDER = ("ours", "fixed_z", "fixed_rw", "ebonf")     # the paper's order, e-Bonferroni last (on top)
LEGEND_ORDER = ("ours", "ebonf", "fixed_z", "fixed_rw")   # two columns: monitored | one look
FIGURES = {"F": "real_leaderboard", "S": "iid_top20"}


def aggregate(runs: list, design: str) -> dict:
    """Mean over item orders of certified true dominances and tiers, per benchmark, method and fraction,
    in the layout of results[...] in real_leaderboard_orders50.json."""
    groups = defaultdict(list)
    for r in runs:
        if r["design"] == design and r["subset"] == SUBSET:
            groups[r["benchmark"]].append(r)
    res = {}
    for b, rs in groups.items():
        rs.sort(key=lambda r: r["order"])
        (tot,) = {r["true_pairs"] for r in rs}
        res[b] = {"true_pairs": tot, "orders": len(rs)}
        for m in LABELS:
            res[b][m] = {f: {q: float(np.mean([r[m][f][q] for r in rs])) for q in ("true_certified", "tiers")}
                         for f in rs[0][m]}
    return res


def check_against_paper(res_f: dict, paper_path: Path) -> None:
    """Ours, Holm-z and Romano-Wolf of design F must equal the paper's Figure 2 data at every plotted
    fraction, and the halfway tiers of ours and Romano-Wolf must match too; otherwise stop."""
    paper = json.loads(paper_path.read_text())["results"]
    worst_curve = worst_tier = 0.0
    for b in ORDER:
        p, r = paper[f"{b}/{SUBSET}"], res_f[b]
        if (p["true_pairs"], p["orders"]) != (r["true_pairs"], r["orders"]):
            raise SystemExit(f"{b}: true pairs or number of orders differ from the paper's run")
        for m in ("ours", "fixed_z", "fixed_rw"):
            for f in p[m]:
                diff = abs(p[m][f]["true_certified"] - r[m][f]["true_certified"]) / p["true_pairs"]
                worst_curve = max(worst_curve, diff)
        for m in ("ours", "fixed_rw"):
            worst_tier = max(worst_tier, abs(p[m]["0.5"]["tiers"] - r[m]["0.5"]["tiers"]))
    print(f"design F against {paper_path.name}: largest difference {worst_curve:.3g} (fraction certified), "
          f"{worst_tier:.3g} (tiers at one half)")
    if worst_curve > 1e-9 or worst_tier > 1e-9:
        raise SystemExit("design F does not reproduce Figure 2; not drawing")


def figure(res: dict, out: Path) -> None:
    """plot_real_leaderboard.py's main(), with e-Bonferroni added and the legend reordered."""
    benches = [b for b in ORDER if b in res]
    plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.edgecolor": MUTED,
                         "axes.linewidth": 0.8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.6), sharey=True)
    for ax, b in zip(axes.ravel(), benches):
        r = res[b]
        fr = sorted(float(f) for f in r["ours"])
        tot = r["true_pairs"]
        for m in DRAW_ORDER:
            ax.plot(fr, [r[m][str(f)]["true_certified"] / tot for f in fr], label=LABELS[m], **STYLE[m])
        ax.set_xscale("log")
        ticks = [f for f in fr if f in (0.05, 0.25, 1.0)] or fr   # a readable subset of the log axis
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{f:g}" for f in ticks], fontsize=7.5)
        ax.minorticks_off()
        ax.set_ylim(0, 1.45)  # headroom for the two-line tier annotation
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(TITLES.get(b, b), fontsize=8.5, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        t_ours = r["ours"]["0.5"]["tiers"]
        t_rw = r["fixed_rw"]["0.5"]["tiers"]
        ax.text(0.03, 0.97, f"tiers at one half:\n{t_ours:.1f} ours vs {t_rw:.1f}",
                transform=ax.transAxes, fontsize=7, color=INK, va="top", linespacing=1.25)
    fig.supylabel("fraction of true dominances certified", fontsize=8.5, color=MUTED, x=0.012)
    fig.text(0.5, 0.168, "fraction of the benchmark evaluated (log scale)", ha="center",
             fontsize=8.5, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    pick = [labels.index(LABELS[m]) for m in LEGEND_ORDER]
    fig.legend([handles[i] for i in pick], [labels[i] for i in pick], loc="lower center", ncol=2,
               frameon=False, fontsize=7.5, columnspacing=1.4, handlelength=2.4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.02, 0.19, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf"), "and .png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", type=Path, help="real_leaderboard_sampling.py output (fig2_orders50.json)")
    ap.add_argument("--paper", type=Path, default=Path("../results/real_leaderboard/real_leaderboard_orders50.json"),
                    help="the data of the paper's Figure 2, for the design F check")
    ap.add_argument("--outdir", type=Path, default=Path("../manuscript/figures"))
    args = ap.parse_args()
    runs = json.loads(args.result.read_text())["runs"]
    res = {design: aggregate(runs, design) for design in FIGURES}
    check_against_paper(res["F"], args.paper)
    for design, name in FIGURES.items():
        figure(res[design], args.outdir / name)
    for design, name in FIGURES.items():
        print(f"\n{name} (design {design}, {SUBSET}): tiers at one half, ours vs Romano-Wolf "
              f"(e-Bonferroni); largest |ours - e-Bonferroni| over fractions, in points")
        for b in ORDER:
            r = res[design][b]
            gap = max(abs(r["ours"][f]["true_certified"] - r["ebonf"][f]["true_certified"]) for f in r["ours"])
            print(f"  {b:11s} {r['ours']['0.5']['tiers']:.2f} vs {r['fixed_rw']['0.5']['tiers']:.2f} "
                  f"({r['ebonf']['0.5']['tiers']:.2f});  {100 * gap / r['true_pairs']:.3f}")


if __name__ == "__main__":
    main()
