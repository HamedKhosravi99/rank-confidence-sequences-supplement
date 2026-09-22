"""Figure for E2: how much of a real leaderboard is distinguishable, as the benchmark is evaluated.

One panel per benchmark; x = fraction of the benchmark evaluated (log scale); y = certified true
orderings as a fraction of all true orderings, for our rank confidence sequence (monitored
throughout), fixed-sample Holm rank sets with a paired z-test, and fixed-sample Romano-Wolf step-down rank
sets (each a single look at that fraction). Tier counts are printed as a table. Colours: validated
default palette, slots 1, 4 and 5 (line style and marker as secondary encoding).

Usage:  python -m experiments.plot_real_leaderboard ../results/real_leaderboard/real_leaderboard_orders50.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
TITLES = {"mmlu": "MMLU (14,042 items)", "hellaswag": "HellaSwag (10,042)", "gsm8k": "GSM8K (1,319)",
          "winogrande": "Winogrande (1,267)", "arc": "ARC-Challenge (1,172)", "truthfulqa": "TruthfulQA (817)"}
ORDER = ("mmlu", "hellaswag", "gsm8k", "winogrande", "arc", "truthfulqa")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--subset", default="top20")
    ap.add_argument("--out", type=Path, default=Path("../manuscript/figures/real_leaderboard.pdf"))
    args = ap.parse_args()
    res = json.loads(args.result.read_text())["results"]
    benches = [b for b in ORDER if f"{b}/{args.subset}" in res]

    plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.edgecolor": MUTED,
                         "axes.linewidth": 0.8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.6), sharey=True)
    for ax, b in zip(axes.ravel(), benches):
        r = res[f"{b}/{args.subset}"]
        fr = sorted(float(f) for f in r["ours"])
        tot = r["true_pairs"]
        ours = [r["ours"][str(f)]["true_certified"] / tot for f in fr]
        fixed_z = [r["fixed_z"][str(f)]["true_certified"] / tot for f in fr]
        fixed_rw = [r["fixed_rw"][str(f)]["true_certified"] / tot for f in fr]
        ax.plot(fr, ours, color="#2a78d6", marker="o", markersize=3.8, linewidth=1.5,
                label="rank confidence sequence (monitored)")
        ax.plot(fr, fixed_z, color="#eda100", marker="D", markersize=3.8, linewidth=1.5, linestyle="--",
                markerfacecolor="white", label="fixed-sample Holm $z$-test sets, one look")
        ax.plot(fr, fixed_rw, color="#e87ba4", marker="v", markersize=3.8, linewidth=1.5, linestyle=":",
                markerfacecolor="white", label="fixed-sample Romano–Wolf step-down sets, one look")
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
    fig.supylabel("certified true orderings", fontsize=8.5, color=MUTED, x=0.012)
    fig.text(0.5, 0.168, "fraction of the benchmark evaluated (log scale)", ha="center",
             fontsize=8.5, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7.5, columnspacing=1.4,
               handlelength=2.4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.02, 0.19, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print("wrote", args.out)

    print(f"\n{args.subset}: tiers (ours / Holm-McNemar / Holm-z / Romano-Wolf) at each fraction of the benchmark")
    for b in benches:
        r = res[f"{b}/{args.subset}"]
        row = "  ".join(f"{f}: {r['ours'][f]['tiers']:.1f}/{r['fixed_n'][f]['tiers']:.1f}/{r['fixed_z'][f]['tiers']:.1f}/{r['fixed_rw'][f]['tiers']:.1f}"
                        for f in sorted(r["ours"], key=float))
        ew = "/".join(f"{r[k]['1.0']['ever_wrong_so_far']:.2f}" for k in ("fixed_n", "fixed_z", "fixed_rw"))
        print(f"{b:11s} ties={r['tied_pairs']:2d} ours-ever-wrong={r['ours_ever_wrong']:.2f} fixed-ever-wrong={ew} | {row}")
        print(f"{'':11s} retire top-k: cost {r['retire_topk']['cost']:.2f}, goal {r['retire_topk']['goal']:.2f}, wrong {r['retire_topk']['wrong']:.2f}")


if __name__ == "__main__":
    main()
