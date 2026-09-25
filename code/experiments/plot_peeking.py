"""Figure for E1: probability that some report is wrong, against the number of looks.

Small multiples (one panel per scenario, one shared y-axis). Identity is never carried by colour
alone: fixed-sample procedures are dashed with open markers, e-process procedures are solid with
filled markers, each series has its own marker, and the two groups are labelled on the plot.
Colours are the first five categorical slots of the validated default palette
(validate_palette.js, light mode, white surface: all checks pass).

Usage:  python -m experiments.plot_peeking ../results/peeking/peeking_mixture_reps5000_n2000.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = {  # name: (label, colour, marker, fixed-sample?)
    "ours_exact": ("Ours, exact", "#2a78d6", "o", False),
    "ours_shortcut": ("Ours, shortcut", "#eb6834", "s", False),
    "e_bonferroni": ("e-Bonferroni", "#1baf7a", "^", False),
    "holm_mcnemar": ("Fixed-sample Holm, exact McNemar", "#eda100", "D", True),
    "holm_ztest": ("Fixed-sample Holm, $z$-test", "#e87ba4", "v", True),
}
TITLES = {
    "all_tied": "All six models tied",
    "tied_leaders": "Tied pairs at ranks 1 and 3",
    "near_ties": "Near ties, gaps of 0.004",
}
ORDER = ("all_tied", "tied_leaders", "near_ties")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--metric", default="ever_wrong", choices=["ever_wrong", "stop_wrong"])
    ap.add_argument("--out", type=Path, default=Path("../manuscript/figures/peeking.pdf"))
    args = ap.parse_args()
    res = json.loads(args.result.read_text())
    alpha = res["config"]["alpha"]
    scenarios = [s for s in ORDER if s in res["scenarios"]]

    plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.edgecolor": MUTED,
                         "axes.linewidth": 0.8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.75, 2.4), sharey=True)
    for panel, (ax, name) in enumerate(zip(axes, scenarios)):
        methods = res["scenarios"][name]["methods"]
        for key, (label, colour, marker, fixed) in SERIES.items():
            rows = methods[key]
            x = [r["looks"] for r in rows]
            y = [r[args.metric] for r in rows]
            se = [r[args.metric + "_se"] for r in rows]
            ax.fill_between(x, [a - 2 * b for a, b in zip(y, se)], [a + 2 * b for a, b in zip(y, se)],
                            color=colour, alpha=0.15, linewidth=0)
            ax.plot(x, y, color=colour, linewidth=1.5, linestyle="--" if fixed else "-",
                    marker=marker, markersize=4.2, markerfacecolor="white" if fixed else colour,
                    markeredgewidth=1.0, label=label, clip_on=False, zorder=3)
        ax.axhline(alpha, color=INK, linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 20, 100])          # a readable subset; the axis is logarithmic
        ax.set_xticklabels(["1", "5", "20", "100"])
        ax.minorticks_off()
        ax.set_title(f"({chr(97 + panel)}) " + TITLES.get(name, name),
                     fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=2)
        ax.tick_params(axis="x", labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("P(some report is wrong)", color=MUTED)
    axes[0].set_ylim(0, 0.65)  # headroom for the key above the highest series
    axes[1].text(0.04, 0.97, f"nominal level $\\alpha$ = {alpha:g}",
                 transform=axes[1].transAxes, fontsize=7.5, color=INK, va="top", ha="left")
    # direct group labels in ink (text never wears the series colour), placed clear of the marks
    axes[0].text(0.04, 0.97, "dashed, open: fixed-sample\nsolid, filled: e-process",
                 transform=axes[0].transAxes, fontsize=7.5, color=INK, va="top", linespacing=1.3)
    fig.text(0.5, 0.16, "number of looks at the leaderboard (log scale)", ha="center",
             fontsize=8.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               handlelength=2.6, columnspacing=1.4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
