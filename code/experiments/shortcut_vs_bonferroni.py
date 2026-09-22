"""Does transitivity pooling (Algorithm 2) ever certify before e-Bonferroni with free closure?

Both certifiers see identical wealth paths and are compared look by look. Reported: the number of
runs, and of looks, at which their certified sets differ. (The shortcut can only be ahead.)

Usage:  python -m experiments.shortcut_vs_bonferroni --runs-per-scenario 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.peeking import simulate
from experiments.weighting import SCENARIOS
from rcs import BonferroniCertifier, ShortcutCertifier, log_wealth_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-per-scenario", type=int, default=150)
    ap.add_argument("--n-max", type=int, default=4000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--bet", default="mixture", choices=["fixed", "agrapa", "ons", "mixture"])
    ap.add_argument("--workers", type=int, default=1, help="accepted for the cluster wrapper; unused")
    ap.add_argument("--out", type=Path, default=Path("../results/weighting"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    times = np.arange(args.every, args.n_max + 1, args.every)
    runs = runs_differ = looks = looks_differ = behind = 0
    for abilities in SCENARIOS.values():
        a = np.asarray(abilities, dtype=float)
        for _ in range(args.runs_per_scenario):
            paths = log_wealth_paths(simulate(rng, a, args.n_max), at=times, bet=args.bet)
            sc, bf = ShortcutCertifier(len(a), args.alpha), BonferroniCertifier(len(a), args.alpha)
            differ = False
            for lw in paths:
                d_sc, d_bf = sc.update(lw), bf.update(lw)
                looks += 1
                behind += bool(np.any(d_bf & ~d_sc))
                if not np.array_equal(d_sc, d_bf):
                    looks_differ += 1
                    differ = True
            runs += 1
            runs_differ += differ
    result = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "runs": runs, "runs_with_a_difference": runs_differ,
        "looks": looks, "looks_with_a_difference": looks_differ,
        "looks_where_shortcut_is_behind": behind,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"shortcut_vs_bonferroni_{args.bet}.json"
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
