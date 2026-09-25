"""Cross-model dependence stress test: validity and efficiency under arbitrary within-item dependence.

Six models, N = 4000 i.i.d. items, the E3 ability configurations, monitoring every 20 items,
alpha 0.05, the mixture bet and the exact certifier; only the dependence of the models' scores
on the same item varies, with every model's marginal accuracy held fixed exactly. Gaussian
copula: Z_tj = a_j sqrt(rho) G_t + sqrt(1 - rho) eps_tj with G_t, eps_tj i.i.d. N(0, 1), and
X_tj = 1{Z_tj <= Phi^{-1}(theta_j)}, so P(X_tj = 1) = theta_j for every rho and a. Regimes:

  independent   rho = 0
  moderate      rho = 0.5, a = 1
  strong        rho = 0.9, a = 1
  mixed         rho = 0.8, a = (1, 1, 1, -1, -1, -1): strong positive dependence within two groups,
                negative across them

Recorded per configuration and regime: the anytime family-wise false-statement rate (some false
dominance certified at any look, or no weak order surviving); at fractions 0.10, 0.25, 0.50,
0.75, 1.00 the fraction of true dominances certified, the mean rank-interval width (U - L + 1), the number of
certified tiers and the fraction of models with resolved top-3 status; the first certification
look of every true pair; the empirical within-item correlation matrix and Var(X_j - X_l); and,
for the power configurations, the cost of certifying every model's top-3 status under the
retirement rule of E5. The same downstream code runs in every regime.

Usage:  python -m experiments.dependence --check                  # sanity checks 1-3
        python -m experiments.dependence --smoke --workers 4
        python -m experiments.dependence --workers 9              # 5000 / 2000 replicates
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import NormalDist

import numpy as np

from experiments.early_stopping import _to_retire
from experiments.peeking import true_theta
from experiments.weighting import SCENARIOS
from rcs import ExactCertifier, RankConfidenceSequence, log_wealth_paths
from rcs.report import rank_intervals, tiers

REGIMES = {  # name: (rho, a)
    "independent": (0.0, [1, 1, 1, 1, 1, 1]),
    "moderate": (0.5, [1, 1, 1, 1, 1, 1]),
    "strong": (0.9, [1, 1, 1, 1, 1, 1]),
    "mixed": (0.8, [1, 1, 1, -1, -1, -1]),
}
CONFIGS = {  # name: (marginal accuracies = the E3 configuration's true accuracies, replicates, retirement?)
    "all_tied": ("all_tied", 5000, False),
    "tied_leaders": ("tied_leaders", 5000, False),
    "near_ties": ("near_ties", 2000, True),
    "even_spread": ("even_spread", 2000, True),
}
FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
K = 3


def accuracies(config: str) -> np.ndarray:
    return true_theta(np.asarray(SCENARIOS[CONFIGS[config][0]], dtype=float))


def gaussian_copula(rng: np.random.Generator, theta: np.ndarray, rho: float, a: np.ndarray, n: int) -> np.ndarray:
    """Binary scores with marginals theta and within-item dependence set by (rho, a)."""
    g = rng.standard_normal((n, 1))
    eps = rng.standard_normal((n, theta.size))
    z = a[None, :] * np.sqrt(rho) * g + np.sqrt(1.0 - rho) * eps
    q = np.array([NormalDist().inv_cdf(float(p)) for p in theta])
    return (z <= q[None, :]).astype(float)


def one_replicate(args: tuple) -> dict:
    seed, config, regime, n, every, alpha, retire = args
    theta = accuracies(config)
    rho, a = REGIMES[regime]
    rng = np.random.default_rng(seed)
    x = gaussian_copula(rng, theta, rho, np.asarray(a, dtype=float), n)
    m = theta.size
    true_pair = theta[:, None] > theta[None, :]
    false_pair = (theta[:, None] <= theta[None, :]) & ~np.eye(m, dtype=bool)
    times = np.arange(every, n + 1, every)
    check = {int(round(f * n)): f for f in FRACTIONS}
    paths = log_wealth_paths(x, bet="mixture", at=times)  # superpopulation bound, as in E3
    cert = ExactCertifier(m, alpha)
    first = np.full((m, m), -1, dtype=int)
    at, wrong = {}, False
    for t, lw in zip(times, paths):
        d = cert.update(lw)
        wrong = wrong or cert.failed or bool((d & false_pair).any())
        first[d & (first < 0)] = int(t)
        if int(t) in check:
            lo, up = rank_intervals(d)
            at[str(check[int(t)])] = {
                "true_frac": float((d & true_pair).sum() / max(1, true_pair.sum())),
                "width": float(np.mean(up - lo + 1)),
                "tiers": int(tiers(d).max()) if not cert.failed else 0,
                "topk_resolved": float(np.mean((up <= K) | (lo > K))),
            }
    out = {"wrong": wrong, "at": at,
           "first": [[int(first[j, l]) for l in range(m)] for j in range(m)],
           "corr": np.corrcoef(x.T).tolist(),
           "var_diff": [[float(np.var(x[:, j] - x[:, l])) for l in range(m)] for j in range(m)],
           "means": x.mean(axis=0).tolist()}
    if retire:
        seq = RankConfidenceSequence(m, alpha, population_size=None, method="exact", bet="mixture")
        rwrong = False
        for t in range(n):
            if seq.wealth.active.sum() <= 1:
                break
            seq.observe(np.where(seq.wealth.active, x[t], np.nan))
            if (t + 1) % every == 0 or t + 1 == n:
                rep = seq.report()
                rwrong = rwrong or rep.failed or bool((rep.dominance & false_pair).any())
                for j in np.flatnonzero(_to_retire("topk", rep, K) & seq.wealth.active):
                    seq.retire(int(j))
        rep = seq.report()
        known = (rep.upper <= K) | (rep.lower > K)
        out["retire"] = {"cost": seq.evaluations / (m * n), "goal": bool(known.all()), "wrong": rwrong}
    return out


def aggregate(runs: list[dict], theta: np.ndarray, n: int) -> dict:
    m = theta.size
    r = len(runs)
    wrong = float(np.mean([x["wrong"] for x in runs]))
    agg = {"n_runs": r, "ever_wrong": wrong, "ever_wrong_se": float(np.sqrt(wrong * (1 - wrong) / r)), "at": {}}
    for f in runs[0]["at"]:
        rows = [x["at"][f] for x in runs]
        agg["at"][f] = {q: {"mean": float(np.mean([row[q] for row in rows])),
                            "se": float(np.std([row[q] for row in rows], ddof=1) / np.sqrt(r))} for q in rows[0]}
    if "retire" in runs[0]:
        rows = [x["retire"] for x in runs]
        agg["retire"] = {q: float(np.mean([row[q] for row in rows])) for q in rows[0]}
    agg["corr"] = np.mean([x["corr"] for x in runs], axis=0).tolist()
    agg["means"] = np.mean([x["means"] for x in runs], axis=0).tolist()
    var_diff = np.mean([x["var_diff"] for x in runs], axis=0)
    firsts = np.array([x["first"] for x in runs], dtype=float)  # (r, m, m), -1 = never
    pairs = []
    for j in range(m):
        for l in range(m):
            if theta[j] > theta[l]:
                f = firsts[:, j, l]
                cert = f[f > 0]
                pairs.append({"pair": [j, l], "margin": float(theta[j] - theta[l]), "var_diff": float(var_diff[j, l]),
                              "certified_frac": float(len(cert) / r),
                              "median_first_frac": float(np.median(cert) / n) if len(cert) else None})
    agg["pairs"] = pairs
    return agg


def sanity_checks(n: int = 200_000, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    for config in CONFIGS:
        theta = accuracies(config)
        print(f"== {config}: target accuracies {np.round(theta, 4).tolist()}")
        for regime, (rho, a) in REGIMES.items():
            x = gaussian_copula(rng, theta, rho, np.asarray(a, dtype=float), n)
            means = x.mean(axis=0)
            corr = np.corrcoef(x.T)
            lag1 = [float(np.corrcoef(x[:-1, j], x[1:, j])[0, 1]) for j in range(theta.size)]
            off = corr[~np.eye(theta.size, dtype=bool)]
            print(f"   {regime:12s} max |mean - theta| = {np.abs(means - theta).max():.4f} (2 s.e. about {2*np.sqrt(0.25/n):.4f}); "
                  f"within-item corr: min {off.min():+.3f} max {off.max():+.3f}; max |lag-1 autocorr| = {max(abs(v) for v in lag1):.4f}")
            if config == "near_ties":
                print("      corr matrix:\n" + "\n".join("      " + " ".join(f"{v:+.2f}" for v in row) for row in corr))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-items", type=int, default=4000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--reps-scale", type=float, default=1.0, help="multiply the replicate counts (smoke: 0.01)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true", help="run the sanity checks and exit")
    ap.add_argument("--out", type=Path, default=Path("../results/dependence"))
    args = ap.parse_args()
    if args.check:
        sanity_checks()
        return
    if args.smoke:
        args.reps_scale = 0.01

    def run_jobs(jobs):
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                return list(pool.map(one_replicate, jobs, chunksize=max(1, len(jobs) // (args.workers * 4))))
        return [one_replicate(j) for j in jobs]

    started = time.time()
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "regimes": {k: {"rho": v[0], "a": v[1]} for k, v in REGIMES.items()}, "results": {}}
    for config in args.configs:
        scenario, reps, retire = CONFIGS[config]
        reps = max(2, int(round(reps * args.reps_scale)))
        theta = accuracies(config)
        result["results"][config] = {"accuracies": theta.tolist(), "true_pairs": int((theta[:, None] > theta[None, :]).sum()),
                                     "replicates": reps, "regimes": {}}
        for regime in REGIMES:
            t0 = time.time()
            seeds = np.random.SeedSequence([args.seed, list(CONFIGS).index(config), list(REGIMES).index(regime)]).generate_state(reps)
            runs = run_jobs([(int(s), config, regime, args.n_items, args.every, args.alpha, retire) for s in seeds])
            agg = aggregate(runs, theta, args.n_items)
            result["results"][config]["regimes"][regime] = agg
            a = agg["at"]["0.5"]["true_frac"]["mean"] if "0.5" in agg["at"] else float("nan")
            print(f"{config}/{regime}: {time.time() - t0:.0f}s, {reps} reps; ever wrong {agg['ever_wrong']:.4f}; "
                  f"true frac at 0.50 {a:.3f}" + (f"; top-3 cost {agg['retire']['cost']:.3f}" if "retire" in agg else ""), flush=True)
    result["seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ("dependence_smoke.json" if args.smoke else "dependence_full.json")
    path.write_text(json.dumps(result, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
