"""Numerical check of Proposition B.4 (growth, consistency and certification time under (S)).

1. The elementary inequality log(1+u) >= u - u^2 on [-1/2, 1/2] and c_lam <= 2.2 lam on the grid.
2. Strong law: log E^{(k)}_t / t -> mu(lambda_k) for every grid bet, one pair, binary scores.
3. Certification times of the paper's procedure (M = 6, mixture bet, alpha = 0.05, even-spread
   abilities of E3) against the predicted e-Bonferroni crossing time L / mu* and the
   high-probability bound t*(eps) of part (c); the exact binary growth of part (b) is used.

Not cited in the paper; it checks that the proposition's rates describe the implemented method.

Usage (from code/):  OMP_NUM_THREADS=1 python -m experiments.power_check --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rcs import BonferroniCertifier, ExactCertifier, ShortcutCertifier, log_wealth_paths
from rcs.wealth import MIXTURE_GRID

GRID = tuple(MIXTURE_GRID)
K = len(GRID)


def c_of(lam: float) -> float:
    return math.log((1 + lam) / (1 - lam))


def x0(lam: float) -> float:
    """Binary positive-growth threshold of Proposition B.4(b): mu(lam) > 0 iff Delta/rho > x0(lam)."""
    return math.log(1 / (1 - lam * lam)) / c_of(lam)


def mu_binary(lam: float, tj: float, tl: float) -> float:
    """Exact E log(1 + lam Z) for independent binary scores with accuracies tj > tl."""
    rho, delta = rho_binary(tj, tl), tj - tl
    return 0.5 * rho * c_of(lam) * (delta / rho - x0(lam))


def rho_binary(tj: float, tl: float) -> float:
    return tj * (1 - tl) + tl * (1 - tj)


def predictions(theta: np.ndarray, M: int, alpha: float, eps: float) -> list[dict]:
    out = []
    L = math.log(K * M * (M - 1) / alpha)
    L0 = math.log(M * (M - 1) / alpha)
    for j in range(M):
        for l in range(M):
            if theta[j] > theta[l]:
                mus = {lam: mu_binary(lam, theta[j], theta[l]) for lam in GRID}
                lam_star, mu_star = max(mus.items(), key=lambda kv: kv[1])
                delta, rho = theta[j] - theta[l], rho_binary(theta[j], theta[l])
                c = c_of(lam_star)
                v = 4 * lam_star**2 * rho  # the variance bound of part (c)
                a = max(mu_star**2 / (2 * c * c), mu_star**2 / (8 * v + 4 / 3 * c * mu_star)) if mu_star > 0 else 0.0
                t_hp = max(2 * L / mu_star, math.log(1 / eps) / a) if mu_star > 0 else math.inf
                out.append({"pair": (j, l), "Delta": delta, "rho": rho, "lam_star": lam_star, "mu_star": mu_star,
                            "t_pred": L / mu_star if mu_star > 0 else math.inf,
                            "t_pred_noK": L0 / mu_star if mu_star > 0 else math.inf, "t_hp": t_hp})
    return out


def one_replicate(args: tuple) -> dict:
    seed, theta, n, every, alpha = args
    theta = np.asarray(theta)
    M = theta.size
    rng = np.random.default_rng(seed)
    x = (rng.random((n, M)) < theta[None, :]).astype(float)
    times = np.arange(every, n + 1, every)
    paths = log_wealth_paths(x, bet="mixture", at=times)
    certs = {"bonf": BonferroniCertifier(M, alpha, close=False), "shortcut": ShortcutCertifier(M, alpha),
             "exact": ExactCertifier(M, alpha)}
    first = {k: np.full((M, M), -1, dtype=int) for k in certs}
    for t, lw in zip(times, paths):
        for k, cert in certs.items():
            d = cert.update(lw)
            first[k][d & (first[k] < 0)] = int(t)
    return {k: v.tolist() for k, v in first.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("../results/power/power_check.json"))
    args = ap.parse_args()

    u = np.linspace(-0.5, 0.5, 100001)
    print(f"1. min of log(1+u) - (u - u^2) on [-1/2,1/2] = {(np.log1p(u) - (u - u**2)).min():.3e} (>= 0); "
          f"max c_lam / lam on the grid = {max(c_of(l) / l for l in GRID):.3f} (<= 2.2); "
          f"x0(lam_min) = {x0(min(GRID)):.5f}")

    tj, tl, T = 0.622, 0.582, 20000
    rng = np.random.default_rng(1)
    rows = np.array([[np.log1p(lam * ((rng.random(T) < tj).astype(float) - (rng.random(T) < tl).astype(float))).sum() / T
                      for lam in GRID] for _ in range(5)])
    print(f"2. strong law, pair ({tj}, {tl}), T={T}: empirical growth vs mu(lambda):")
    for i, lam in enumerate(GRID):
        print(f"   lam={lam}: {rows[:, i].mean():+.5f} (sd {rows[:, i].std():.5f})  mu={mu_binary(lam, tj, tl):+.5f}")

    theta = np.array([0.697, 0.66, 0.622, 0.582, 0.541, 0.5])  # even spread, as in E3
    M = theta.size
    preds = predictions(theta, M, args.alpha, args.eps)
    seeds = np.random.SeedSequence(20260919).generate_state(args.reps)
    jobs = [(int(s), theta.tolist(), args.n, args.every, args.alpha) for s in seeds]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            runs = list(pool.map(one_replicate, jobs, chunksize=4))
    else:
        runs = [one_replicate(j) for j in jobs]
    print(f"\n3. M={M}, alpha={args.alpha}, mixture bet, N={args.n}, every {args.every}, {args.reps} reps")
    print(f"{'pair':7s} {'Delta':>6s} {'rho':>5s} {'lam*':>5s} {'mu*':>8s} | {'L/mu*':>7s} {'L0/mu*':>7s} | "
          f"{'med bonf':>8s} {'med sc':>7s} {'med exact':>9s} | {'t*(eps)':>8s} {'P[bonf<=t*]':>11s} {'not by N':>8s}")
    summary = []
    for p in preds:
        j, l = p["pair"]
        meds, fr, never = {}, {}, {}
        for k in ("bonf", "shortcut", "exact"):
            f = np.array([r[k][j][l] for r in runs], dtype=float)
            cert = f[f > 0]
            meds[k] = float(np.median(cert)) if len(cert) else math.nan
            never[k] = float(np.mean(f < 0))
            fr[k] = float(np.mean((f > 0) & (f <= p["t_hp"]))) if math.isfinite(p["t_hp"]) else math.nan
        print(f"({j},{l})  {p['Delta']:6.3f} {p['rho']:5.2f} {p['lam_star']:5.2f} {p['mu_star']:8.5f} | "
              f"{p['t_pred']:7.0f} {p['t_pred_noK']:7.0f} | {meds['bonf']:8.0f} {meds['shortcut']:7.0f} {meds['exact']:9.0f} | "
              f"{p['t_hp']:8.0f} {fr['bonf']:11.3f} {never['bonf']:8.3f}")
        summary.append({**p, "median": meds, "frac_by_t_hp": fr, "not_certified_by_n": never})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                                    "theta": theta.tolist(), "pairs": summary}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
