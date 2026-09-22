"""E1. What monitoring does to fixed-sample rank sets, and what it does to ours.

An analyst evaluates M models on a stream of items and looks at the leaderboard K times, equally
spaced, up to n_max items. At every look each procedure issues a report (certified dominances and
the rank intervals they imply). We record, per procedure and per K:

  ever_wrong  -- some report at some look contains a false statement (a certified dominance
                 j > l with theta_j <= theta_l, or equivalently a rank interval missing the truth);
  stop_wrong  -- the analyst stops at the first look where a unique best model is declared
                 (some U_j = 1), or at n_max otherwise, and that final report is wrong;
  certified   -- number of TRUE dominances certified at the last look (the price of validity).

Fixed-sample procedures (rcs.baselines) are recomputed from scratch at every look, which is how
they are used in practice. Ours are run on the same K looks. K = 1 is the single pre-planned
analysis at which the fixed-sample procedures are valid.

Data: item i has difficulty d_i ~ N(0, 1), and model j answers correctly with probability
sigmoid(a_j - d_i), independently given d_i. Models are therefore positively dependent through
shared items. theta_j = E sigmoid(a_j - d) is computed by Gauss-Hermite quadrature; equal
abilities give exactly equal theta.

Usage:  python -m experiments.peeking --reps 5000 --workers 10 --out ../results/peeking
The bet is the mixture by default (--bet fixed reproduces the earlier run).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from rcs import BonferroniCertifier, ExactCertifier, ShortcutCertifier, log_wealth_paths
from rcs.baselines import holm_dominance
from rcs.report import rank_intervals

SCENARIOS = {
    # ability vectors a_j; ties in a are exact ties in theta
    "tied_leaders": [1.0, 1.0, 0.6, 0.6, 0.2, -0.2],
    "all_tied": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "near_ties": [1.00, 0.98, 0.62, 0.60, 0.22, 0.20],
}
LOOKS = (1, 2, 5, 10, 20, 50, 100, 200)
METHODS = ("holm_mcnemar", "holm_ztest", "ours_exact", "ours_shortcut", "e_bonferroni")


def true_theta(abilities: np.ndarray) -> np.ndarray:
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    p = 1.0 / (1.0 + np.exp(-(abilities[:, None] - nodes[None, :])))
    return (p * weights).sum(axis=1) / weights.sum()


def simulate(rng: np.random.Generator, abilities: np.ndarray, n: int) -> np.ndarray:
    d = rng.standard_normal((n, 1))
    p = 1.0 / (1.0 + np.exp(-(abilities[None, :] - d)))
    return (rng.random(p.shape) < p).astype(float)


def _fixed_n_reports(x: np.ndarray, times: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """Holm-corrected dominance matrices at every look time, from cumulative counts."""
    z = x[:, :, None] - x[:, None, :]
    wins = np.cumsum(z > 0, axis=0)[times - 1]
    losses = np.cumsum(z < 0, axis=0)[times - 1]
    s1 = np.cumsum(z, axis=0)[times - 1]
    s2 = np.cumsum(z * z, axis=0)[times - 1]
    n = times[:, None, None].astype(float)
    m = x.shape[1]
    eye = np.eye(m, dtype=bool)

    p_mc = stats.binom.sf(wins - 1, wins + losses, 0.5)
    p_mc = np.where(wins + losses == 0, 1.0, p_mc)
    mean = s1 / n
    var = np.maximum(s2 - n * mean**2, 0.0) / np.maximum(n - 1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(var > 0, np.sqrt(n) * mean / np.sqrt(var), 0.0)
    p_z = stats.norm.sf(stat)
    out = {}
    for name, p in (("holm_mcnemar", p_mc), ("holm_ztest", p_z)):
        p = np.where(eye[None], 1.0, p)
        out[name] = np.stack([holm_dominance(p[i], alpha) for i in range(len(times))])
    return out


def _ours_reports(paths: np.ndarray, alpha: float, m: int) -> dict[str, np.ndarray]:
    certs = {
        "ours_exact": ExactCertifier(m, alpha),
        "ours_shortcut": ShortcutCertifier(m, alpha),
        "e_bonferroni": BonferroniCertifier(m, alpha),
    }
    return {name: np.stack([c.update(lw) for lw in paths]) for name, c in certs.items()}


def _summarise(reports: np.ndarray, false_pair: np.ndarray, true_pair: np.ndarray) -> tuple[bool, bool, int]:
    wrong = (reports & false_pair[None]).any(axis=(1, 2))
    declared = np.array([(rank_intervals(d)[1] == 1).any() for d in reports])
    stop = int(np.argmax(declared)) if declared.any() else len(reports) - 1
    return bool(wrong.any()), bool(wrong[stop]), int((reports[-1] & true_pair).sum())


def one_replicate(args: tuple) -> dict:
    seed, abilities, n_max, alpha, lam, bet = args
    abilities = np.asarray(abilities, dtype=float)
    theta = true_theta(abilities)
    false_pair = theta[:, None] <= theta[None, :]
    np.fill_diagonal(false_pair, False)
    true_pair = theta[:, None] > theta[None, :]
    rng = np.random.default_rng(seed)
    x = simulate(rng, abilities, n_max)
    m = x.shape[1]

    finest = np.arange(1, max(LOOKS) + 1) * (n_max // max(LOOKS))
    fixed = _fixed_n_reports(x, finest, alpha)
    paths_all = log_wealth_paths(x, bet=bet, lam=lam, at=finest)

    out = {}
    for k in LOOKS:
        step = max(LOOKS) // k
        idx = np.arange(step - 1, max(LOOKS), step)  # positions of this schedule in the finest grid
        for name, reports in fixed.items():
            out[(name, k)] = _summarise(reports[idx], false_pair, true_pair)
        for name, reports in _ours_reports(paths_all[idx], alpha, m).items():
            out[(name, k)] = _summarise(reports, false_pair, true_pair)
    return out


def run(scenario: str, reps: int, n_max: int, alpha: float, lam: float, bet: str, seed: int, workers: int) -> dict:
    abilities = SCENARIOS[scenario]
    seeds = np.random.SeedSequence([seed, sorted(SCENARIOS).index(scenario)]).generate_state(reps)
    jobs = [(int(s), abilities, n_max, alpha, lam, bet) for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(one_replicate, jobs, chunksize=max(1, reps // (workers * 8))))
    else:
        results = [one_replicate(j) for j in jobs]

    theta = true_theta(np.asarray(abilities, dtype=float))
    summary = {"abilities": abilities, "theta": theta.tolist(), "methods": {}}
    n_true = int((theta[:, None] > theta[None, :]).sum())
    for name in METHODS:
        rows = []
        for k in LOOKS:
            arr = np.array([r[(name, k)] for r in results], dtype=float)
            ever, stop, cert = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
            rows.append(
                {
                    "looks": k,
                    "ever_wrong": ever,
                    "ever_wrong_se": float(np.sqrt(ever * (1 - ever) / reps)),
                    "stop_wrong": stop,
                    "stop_wrong_se": float(np.sqrt(stop * (1 - stop) / reps)),
                    "true_certified_at_end": cert,
                    "true_pairs": n_true,
                }
            )
        summary["methods"][name] = rows
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--n-max", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bet", default="mixture", choices=["fixed", "agrapa", "ons", "mixture"])
    ap.add_argument("--lam", type=float, default=0.25, help="used only with --bet fixed")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--out", type=Path, default=Path("../results/peeking"))
    args = ap.parse_args()
    if args.n_max % max(LOOKS):
        raise SystemExit(f"--n-max must be a multiple of {max(LOOKS)}")

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "looks": list(LOOKS),
        "scenarios": {},
    }
    for scenario in args.scenarios:
        t0 = time.time()
        result["scenarios"][scenario] = run(
            scenario, args.reps, args.n_max, args.alpha, args.lam, args.bet, args.seed, args.workers
        )
        print(f"{scenario}: {time.time() - t0:.1f}s", flush=True)
    result["seconds"] = time.time() - started
    path = args.out / f"peeking_{args.bet}_reps{args.reps}_n{args.n_max}.json"
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
