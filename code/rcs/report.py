"""Step 3 of the method: read rank intervals, top-k sets and tiers off a certified set D.

``D[j, l] = True`` means model j is certified better than model l. Ranks are 1-based, 1 = best.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rank_intervals(dominance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(L, U): L_j = 1 + #{l certified better than j}, U_j = M - #{l certified worse than j}."""
    d = np.asarray(dominance, dtype=bool)
    m = d.shape[0]
    lower = 1 + d.sum(axis=0)
    upper = m - d.sum(axis=1)
    return lower, upper


def top_k_status(dominance: np.ndarray, k: int) -> np.ndarray:
    """Per model: +1 certified inside the top k, -1 certified outside, 0 unresolved."""
    lower, upper = rank_intervals(dominance)
    status = np.zeros(lower.shape[0], dtype=int)
    status[upper <= k] = 1
    status[lower > k] = -1
    return status


def tiers(dominance: np.ndarray) -> np.ndarray:
    """Layer tiers: tier(j) = 1 + length of the longest chain of certified dominators of j.

    Requires an acyclic relation; raises ``ValueError`` on a cycle, which certifies that an error
    has occurred (probability at most alpha).
    """
    d = np.asarray(dominance, dtype=bool)
    m = d.shape[0]
    tier = np.ones(m, dtype=int)
    # Longest path by relaxation: a DAG on m nodes converges within m rounds.
    for _ in range(m + 1):
        above = np.where(d, tier[:, None], 0).max(axis=0)  # highest tier among dominators, 0 if none
        new = 1 + above
        if np.array_equal(new, tier):
            return tier
        tier = new
    raise ValueError("dominance relation contains a cycle")


def rank_of(theta: np.ndarray) -> np.ndarray:
    """True ranks R_j = 1 + #{l: theta_l > theta_j}."""
    theta = np.asarray(theta, dtype=float)
    return 1 + (theta[None, :] > theta[:, None]).sum(axis=1)


def false_dominances(dominance: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Mask of certified pairs (j, l) that are false, i.e. theta_j <= theta_l."""
    theta = np.asarray(theta, dtype=float)
    return np.asarray(dominance, dtype=bool) & (theta[:, None] <= theta[None, :])


@dataclass
class Report:
    """What the user sees after an update."""

    t: int
    dominance: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    tier: np.ndarray | None
    rank_sets: list[np.ndarray] | None = None
    failed: bool = False
    n_surviving_orders: int | None = None
    active: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    def top_k(self, k: int) -> np.ndarray:
        return top_k_status(self.dominance, k)

    def covers(self, theta: np.ndarray) -> bool:
        """True iff every reported statement is correct for the parameter ``theta``."""
        if self.failed:
            return False
        ranks = rank_of(theta)
        if false_dominances(self.dominance, theta).any():
            return False
        if np.any(ranks < self.lower) or np.any(ranks > self.upper):
            return False
        if self.rank_sets is not None:
            return all(r in s for r, s in zip(ranks, self.rank_sets))
        return True
