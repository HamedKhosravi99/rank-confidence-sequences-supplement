"""False-winner probability of a monitored tied pair under a look schedule (the approximation of Table 4).

For a tied pair (equal means, non-degenerate differences) the uncorrected paired z-statistic after
a fraction u of the items behaves for large N like B(u)/sqrt(u) on a fixed benchmark sampled
without replacement (B a Brownian bridge) and like W(u)/sqrt(u) under i.i.d. sampling (W a
Brownian motion). Holm over the pair's two one-sided tests claims a winner when |z| >= 1.96. The
probability that some look in a schedule makes a false claim therefore depends only on the look
fractions. This is a large-sample approximation, computed by Monte Carlo with exact Gaussian
increments at the look fractions and 2 million paths per cell, which brings the standard error
below 0.04 points. It says nothing about non-tied pairs and is least accurate at very early looks.
One look at half has the closed forms 2 Phi(-1.96 sqrt 2) = 0.557% under (F) and 5.0% under (S).
Under (S) the values agree with Armitage, McPherson and Rowe (1969).

Usage:  python -m experiments.look_schedules
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

Z = 1.959964


def schedules() -> dict[str, np.ndarray]:
    out = {"half": np.array([0.5])}
    for k in (5, 10, 20, 50, 100):
        out[f"equal{k}"] = np.arange(1, k + 1) / k
    for k in (10, 20):
        geo = 0.01 ** ((k - np.arange(1, k + 1)) / (k - 1))
        out[f"early{k}"] = geo
        out[f"late{k}"] = np.sort(1.01 - geo)
    return out


def crossing(looks: np.ndarray, paths: int, rng: np.random.Generator, bridge: bool, chunk: int = 200_000) -> float:
    grid = np.unique(np.concatenate([looks, [1.0]]))
    idx = np.searchsorted(grid, looks)
    sd = np.sqrt(np.diff(np.concatenate([[0.0], grid])))
    hits = 0
    for _ in range(paths // chunk):
        w = np.cumsum(rng.standard_normal((chunk, grid.size)) * sd, axis=1)
        if bridge:
            w = w - grid[None, :] * w[:, -1:]
        hits += int((np.abs(w[:, idx]) / np.sqrt(looks)[None, :] >= Z).any(axis=1).sum())
    return hits / paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--out", type=Path, default=Path("../results/look_schedules"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    res = {}
    for name, looks in schedules().items():
        f = crossing(looks, args.paths, rng, bridge=True)
        s = crossing(looks, args.paths, rng, bridge=False)
        res[name] = {"looks": len(looks), "first": float(looks[0]), "fixed_benchmark": f, "iid": s,
                     "mc_se": float(np.sqrt(max(f, s) * (1 - max(f, s)) / args.paths))}
        print(f"{name:8s} {len(looks):3d} looks from {looks[0]:.3f}: fixed benchmark {100*f:6.2f}%   i.i.d. {100*s:6.2f}%")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bridge_paths{args.paths}.json"
    path.write_text(json.dumps({"config": {k: str(v) for k, v in vars(args).items()}, "schedules": res}, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
