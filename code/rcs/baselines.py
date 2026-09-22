"""Fixed-sample competitors, used to show what happens when they are monitored.

Both follow the construction used for leaderboard rank intervals (Holm-corrected one-sided pairwise
tests, rank interval by counting rejections; Al Mohamad et al. 2022, Neuhof and Benjamini 2026).
Each is valid when applied ONCE at a sample size fixed in advance. They differ in the pairwise test:

  * ``"mcnemar"`` -- exact one-sided conditional binomial test on discordant items: finite-sample
    valid for binary scores, so any error inflation under monitoring is due to monitoring alone;
  * ``"ztest"``   -- one-sided paired z-test (the usual CLT interval).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def pairwise_pvalues(scores: np.ndarray, test: str = "mcnemar") -> np.ndarray:
    """p[j, l] for H_jl: theta_j <= theta_l from the first ``len(scores)`` items. Diagonal = 1."""
    x = np.asarray(scores, dtype=float)
    n, m = x.shape
    z = x[:, :, None] - x[:, None, :]
    if test == "mcnemar":
        wins = (z > 0).sum(axis=0)
        losses = (z < 0).sum(axis=0)
        p = stats.binom.sf(wins - 1, wins + losses, 0.5)  # P(Bin(n_d, 1/2) >= wins)
        p = np.where(wins + losses == 0, 1.0, p)
    elif test == "ztest":
        mean = z.mean(axis=0)
        sd = z.std(axis=0, ddof=1) if n > 1 else np.zeros((m, m))
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = np.sqrt(n) * mean / sd
        stat = np.where(sd > 0, stat, 0.0)
        p = stats.norm.sf(stat)
    else:
        raise ValueError(f"unknown test {test!r}")
    np.fill_diagonal(p, 1.0)
    return p


def holm_dominance(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Holm step-down over the M(M-1) one-sided pairwise hypotheses. D[j, l] = H_jl rejected."""
    p = np.asarray(pvalues, dtype=float)
    m = p.shape[0]
    off = ~np.eye(m, dtype=bool)
    flat = p[off]
    order = np.argsort(flat, kind="stable")
    k = flat.size
    thresholds = alpha / (k - np.arange(k))
    passed = flat[order] <= thresholds
    n_reject = k if passed.all() else int(np.argmin(passed))
    rejected = np.zeros(k, dtype=bool)
    rejected[order[:n_reject]] = True
    d = np.zeros((m, m), dtype=bool)
    d[off] = rejected
    return d


def fixed_n_dominance(scores: np.ndarray, alpha: float, test: str = "mcnemar") -> np.ndarray:
    """The fixed-sample procedure applied to the items seen so far."""
    return holm_dominance(pairwise_pvalues(scores, test), alpha)


# ---------------------------------------------------------------------------------------------
# Group-sequential (Pocock) competitor, as used for early stopping of pairwise evaluation by
# Arviv et al. (2026): K pre-specified equally spaced looks, one boundary c(K, alpha) for the
# standardized statistic at every look, no multiplicity correction across pairs (optionally a
# Bonferroni correction). Asymptotic (normal approximation) and valid only for the pre-specified
# looks; this is the baseline, not our method.
# ---------------------------------------------------------------------------------------------

_POCOCK_CACHE: dict = {}


def pocock_constant(looks: int, alpha: float, *, draws: int = 2_000_000, seed: int = 0) -> float:
    """One-sided Pocock boundary c with P(max_k Z_k >= c) = alpha for K equally spaced looks,
    where Z_k are the standardized partial sums of a Gaussian random walk. Monte Carlo, cached."""
    key = (looks, round(alpha, 12), draws, seed)
    if key not in _POCOCK_CACHE:
        rng = np.random.default_rng(seed)
        maxes = np.full(draws, -np.inf)
        chunk = 200_000
        for start in range(0, draws, chunk):
            size = min(chunk, draws - start)
            walk = np.cumsum(rng.standard_normal((size, looks)), axis=1)
            z = walk / np.sqrt(np.arange(1, looks + 1))
            maxes[start:start + size] = z.max(axis=1)
        _POCOCK_CACHE[key] = float(np.quantile(maxes, 1 - alpha))
    return _POCOCK_CACHE[key]


class PocockSequence:
    """Drop-in stand-in for RankConfidenceSequence with the same observe/report/retire interface.

    Certifies j > l at one of the K looks once the paired z-statistic exceeds the Pocock boundary
    at level ``alpha`` (or ``alpha / (M(M-1))`` with ``bonferroni=True``); certifications are
    permanent and closed under transitivity.
    """

    def __init__(self, n_models: int, alpha: float, n_items: int, *, looks: int = 10,
                 bonferroni: bool = False) -> None:
        from .certify import has_cycle, transitive_closure  # local import to avoid a cycle
        from .report import Report, rank_intervals, tiers

        self._closure, self._has_cycle = transitive_closure, has_cycle
        self._Report, self._rank_intervals, self._tiers = Report, rank_intervals, tiers
        self.n_models, self.alpha, self.n_items, self.looks = n_models, alpha, n_items, looks
        level = alpha / (n_models * (n_models - 1)) if bonferroni else alpha
        self.boundary = pocock_constant(looks, level)
        self.look_times = set(int(round(n_items * k / looks)) for k in range(1, looks + 1))
        self.t = 0
        self.evaluations = 0
        self.active = np.ones(n_models, dtype=bool)
        self._s1 = np.zeros((n_models, n_models))
        self._s2 = np.zeros((n_models, n_models))
        self._cnt = np.zeros((n_models, n_models))
        self._d = np.zeros((n_models, n_models), dtype=bool)

    def retire(self, model: int) -> None:
        self.active[model] = False

    def observe(self, scores: np.ndarray) -> None:
        x = np.asarray(scores, dtype=float)
        live = self.active
        self.t += 1
        self.evaluations += int(live.sum())
        pair_live = np.outer(live, live)
        np.fill_diagonal(pair_live, False)
        z = np.where(pair_live, x[:, None] - x[None, :], 0.0)
        self._s1[pair_live] += z[pair_live]
        self._s2[pair_live] += z[pair_live] ** 2
        self._cnt[pair_live] += 1

    def report(self):
        if self.t in self.look_times:
            n = np.maximum(self._cnt, 1.0)
            mean = self._s1 / n
            var = np.maximum(self._s2 - n * mean**2, 0.0) / np.maximum(n - 1.0, 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                stat = np.where(var > 0, np.sqrt(n) * mean / np.sqrt(var),
                                np.where(mean > 0, np.inf, 0.0))
            hit = (stat >= self.boundary) & (self._cnt > 1)
            np.fill_diagonal(hit, False)
            self._d = self._closure(self._d | hit)
        d = self._d.copy()
        failed = self._has_cycle(d)
        lower, upper = self._rank_intervals(d)
        return self._Report(t=self.t, dominance=d, lower=lower, upper=upper,
                            tier=None if failed else self._tiers(d), failed=failed,
                            active=self.active.copy())
