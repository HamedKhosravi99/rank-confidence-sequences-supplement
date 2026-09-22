"""How tight is Theorem 1? Error rate when all models are tied, monitored after every item.

With every model tied, the true weak order is the all-tied order, every certified dominance is
false, and each pairwise wealth is an exact martingale, so Ville's inequality is close to an
equality. We report the frequency with which (a) the true weak order is ever rejected and (b) a
report is ever wrong; Theorem 1 bounds both by alpha. (a) is also recomputed directly from the
pairwise wealths, independently of the certifier.

Usage:  python -m experiments.tightness --runs 4000 --out ../results/tightness
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rcs import ExactCertifier, log_wealth_paths, weak_orders
from rcs.report import rank_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=4000)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--models", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--accuracy", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=1, help="accepted for the cluster wrapper; unused")
    ap.add_argument("--out", type=Path, default=Path("../results/tightness"))
    args = ap.parse_args()

    m, n, alpha = args.models, args.n, args.alpha
    rng = np.random.default_rng(args.seed)
    theta = np.full(m, args.accuracy)
    ranks = rank_of(theta)
    orders = weak_orders(m)
    true_idx = int(np.flatnonzero((orders == 0).all(axis=1))[0])
    off = ~np.eye(m, dtype=bool)

    started = time.time()
    rejected = rejected_direct = wrong = 0
    for _ in range(args.runs):
        x = (rng.random((n, m)) < theta).astype(float)
        paths = log_wealth_paths(x, lam=args.lam)
        cert = ExactCertifier(m, alpha)
        bad = False
        for t in range(n):
            d = cert.update(paths[t])
            if not bad and (cert.failed or d.any() or any(r not in s for r, s in zip(ranks, cert.rank_sets()))):
                bad = True
        rejected += not cert.surviving[true_idx]
        rejected_direct += np.exp(paths[:, off]).mean(axis=1).max() >= 1 / alpha
        wrong += bad

    def rate(k: int) -> dict:
        p = k / args.runs
        return {"rate": p, "se": float(np.sqrt(p * (1 - p) / args.runs))}

    result = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "true_order_rejected": rate(rejected),
        "true_order_rejected_recomputed": rate(int(rejected_direct)),
        "report_ever_wrong": rate(wrong),
        "seconds": time.time() - started,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"tightness_m{m}_alpha{alpha:g}_runs{args.runs}.json"
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
