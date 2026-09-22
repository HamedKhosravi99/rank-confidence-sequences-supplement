"""Orderings with ties (weak orders) of M models, as level vectors.

A weak order assigns model j a level v[j] in {0, ..., k-1}, using every level, for some k <= M.
A higher level is better; equal levels are ties. The number of weak orders is the ordered Bell
(Fubini) number: 1, 3, 13, 75, 541, 4683, 47293, 545835 for M = 1..8.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np

FUBINI = (1, 1, 3, 13, 75, 541, 4683, 47293, 545835)
MAX_MODELS_EXACT = 8


@lru_cache(maxsize=None)
def weak_orders(n_models: int) -> np.ndarray:
    """All weak orders of ``n_models`` models, shape (n_orders, n_models), dtype int8."""
    if not 1 <= n_models <= MAX_MODELS_EXACT:
        raise ValueError(f"exact enumeration supports 1..{MAX_MODELS_EXACT} models")
    # Build recursively: insert the last model into a weak order of the others, either into an
    # existing level (a tie) or into a new level at any of the k+1 positions.
    if n_models == 1:
        out = np.zeros((1, 1), dtype=np.int8)
        out.setflags(write=False)
        return out
    prev = weak_orders(n_models - 1)
    rows = []
    for v in prev:
        k = int(v.max()) + 1
        for level in range(k):  # tie with an existing level
            rows.append(np.append(v, level))
        for pos in range(k + 1):  # new level inserted at position pos
            shifted = np.where(v >= pos, v + 1, v)
            rows.append(np.append(shifted, pos))
    out = np.array(rows, dtype=np.int8)
    assert out.shape[0] == FUBINI[n_models]
    out.setflags(write=False)
    return out


@lru_cache(maxsize=None)
def true_pair_masks(n_models: int) -> np.ndarray:
    """T(v) for every weak order v: mask[i, j, l] is True iff j != l and v_i[j] <= v_i[l].

    These are the dominance hypotheses H_jl: theta_j <= theta_l that are true under order i.
    """
    v = weak_orders(n_models).astype(np.int16)
    mask = v[:, :, None] <= v[:, None, :]
    diag = np.arange(n_models)
    mask[:, diag, diag] = False
    mask.setflags(write=False)
    return mask


@lru_cache(maxsize=None)
def order_ranks(n_models: int) -> np.ndarray:
    """rank[i, j] = 1 + number of models strictly above model j in weak order i."""
    v = weak_orders(n_models).astype(np.int16)
    ranks = 1 + (v[:, None, :] > v[:, :, None]).sum(axis=2)
    ranks.setflags(write=False)
    return ranks


def weak_order_of(theta: np.ndarray) -> np.ndarray:
    """The level vector induced by a parameter vector (dense ranks of theta, 0 = worst)."""
    theta = np.asarray(theta, dtype=float)
    _, inverse = np.unique(theta, return_inverse=True)
    return inverse.astype(np.int8)


def brute_force_weak_orders(n_models: int) -> set[tuple[int, ...]]:
    """Reference enumeration by filtering all level assignments; used only in tests."""
    out = set()
    for v in product(range(n_models), repeat=n_models):
        if sorted(set(v)) == list(range(max(v) + 1)):
            out.add(v)
    return out
