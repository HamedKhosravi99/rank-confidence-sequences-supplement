"""Exact ties among all 395 models, per benchmark (the tie column of Table 7).

Ties are exact equalities of the whole-benchmark score sums. For the five binary benchmarks the
sums are integers, so equality is exact; for TruthfulQA (scores in [0, 1]) the sums are compared
as computed in float64 from the stored scores, with no tolerance. Reported per benchmark:

  tied_pairs          unordered pairs with equal means (all pairs, not only adjacent ones);
  tied_models         distinct models in at least one tie: under the all-pairs retirement rule
                      these never retire (Corollary B.15);
  tie_groups          number of groups of two or more models sharing a mean, and their sizes;
  identical_vectors   pairs whose per-item scores are identical, a subset of the tied pairs;
  adjacent_*          the same counts for the 394 adjacent pairs of the ranking;
  boundary            the gap between the 3rd and 4th models (score-sum units: items for binary).

Usage:  python -m experiments.exact_ties
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.real_leaderboard import BENCHMARKS, load


def ties(benchmark: str, k: int = 3) -> dict:
    x, models = load(benchmark)
    n, m = x.shape
    binary = bool(np.all((x == 0) | (x == 1)))
    total = x.sum(axis=0)
    if binary:
        total = np.rint(total).astype(np.int64)
    eq = total[:, None] == total[None, :]
    np.fill_diagonal(eq, False)
    tied_pairs = int(eq.sum() // 2)
    tied_models = int(eq.any(axis=1).sum())
    groups = Counter(total.tolist())
    sizes = sorted((c for c in groups.values() if c > 1), reverse=True)
    iu = np.triu_indices(m, 1)
    tied_idx = [(i, j) for i, j in zip(*iu) if eq[i, j]]
    identical = sum(1 for i, j in tied_idx if np.array_equal(x[:, i], x[:, j]))
    order = np.argsort(-total, kind="stable")
    adj_gap = np.abs(np.diff(total[order].astype(float)))
    adj_identical = sum(1 for a, b in zip(order[:-1], order[1:]) if np.array_equal(x[:, a], x[:, b]))
    srt = total[order]
    return {"n_items": n, "n_models": m, "binary": binary,
            "tied_pairs": tied_pairs, "tied_models": tied_models,
            "tie_groups": len(sizes), "largest_groups": sizes[:5],
            "identical_vectors": identical,
            "adjacent_tied": int((adj_gap == 0).sum()), "adjacent_identical": adj_identical,
            "adjacent_gap_1to5": int(((adj_gap > 0) & (adj_gap <= 5)).sum()),
            "adjacent_gap_median": float(np.median(adj_gap)),
            "boundary": {"kth": float(srt[k - 1]), "k_plus_1th": float(srt[k]),
                         "gap": float(srt[k - 1] - srt[k]),
                         "tied_at_boundary": bool(srt[k - 1] == srt[k]),
                         "models": [models[order[k - 1]], models[order[k]]]}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("../results/exact_ties"))
    args = ap.parse_args()
    result = {b: ties(b, args.k) for b in BENCHMARKS}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ties.json").write_text(json.dumps(result, indent=1))
    for b, r in result.items():
        print(f"{b:11s} N={r['n_items']:6d} tied pairs {r['tied_pairs']:4d}, tied models {r['tied_models']:3d}, "
              f"groups {r['tie_groups']:3d} (largest {r['largest_groups'][:3]}), identical {r['identical_vectors']:3d} | "
              f"adjacent: tied {r['adjacent_tied']:3d}, identical {r['adjacent_identical']:2d}, gap 1-5 {r['adjacent_gap_1to5']:3d}, "
              f"median gap {r['adjacent_gap_median']:.2f} | 3rd-4th gap {r['boundary']['gap']:.3f}")


if __name__ == "__main__":
    main()
