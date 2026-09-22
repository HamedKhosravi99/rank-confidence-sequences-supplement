"""Step 2 of the method: turn pairwise log-wealth into certified dominances.

``D[j, l] = True`` means "model j is better than model l" has been certified, i.e. the null
H_jl: theta_j <= theta_l has been rejected. Certification is permanent.

Four certifiers share one interface, ``update(log_wealth) -> D``:

  * :class:`ExactCertifier`     -- the restricted closed test over weak orders (Algorithm 1);
  * :class:`ShortcutCertifier`  -- transitivity pooling (Algorithm 2), any number of models;
  * :class:`BonferroniCertifier`-- e-Bonferroni over the M(M-1) ordered pairs, a baseline;
  * :class:`ExactILPCertifier`  -- Algorithm 1's set for any number of models, computed exactly
                                   by integer programming instead of enumeration (Algorithm 3).

Numerical note. Every decision compares an average or a sum of wealths with a threshold of at most
``M(M-1)/alpha``. Capping each wealth at that value cannot change any decision: a wealth at the cap
already forces every sum containing it over its threshold. Capping removes all overflow concerns,
so the arithmetic below is done on the natural scale.
"""

from __future__ import annotations

import numpy as np

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, vstack

from .weak_orders import MAX_MODELS_EXACT, order_ranks, true_pair_masks, weak_orders


def _capped_wealth(log_wealth: np.ndarray, alpha: float, factor: float | None = None) -> np.ndarray:
    """Wealth on the natural scale, capped at ``factor / alpha`` (default factor M(M-1)).

    ``factor`` must be at least 1 / (smallest positive weight used by the caller), so that a
    single capped wealth still pushes any weighted sum containing it over ``1 / alpha``.
    """
    m = log_wealth.shape[0]
    cap = np.log((m * (m - 1) if factor is None else factor) / alpha) + 1e-9
    wealth = np.exp(np.minimum(log_wealth, cap))
    np.fill_diagonal(wealth, 0.0)
    return wealth


_POOL_BYTES = 64 << 20  # working-set budget for the transitivity-pooling minimum


def _check(log_wealth: np.ndarray, n_models: int) -> np.ndarray:
    lw = np.asarray(log_wealth, dtype=float)
    if lw.shape != (n_models, n_models):
        raise ValueError(f"log_wealth must have shape ({n_models}, {n_models})")
    return lw


def transitive_closure(dominance: np.ndarray) -> np.ndarray:
    """Boolean transitive closure (Warshall). Free of error by the free-closure lemma."""
    d = np.array(dominance, dtype=bool)
    for k in range(d.shape[0]):
        d |= d[:, k, None] & d[None, k, :]
    return d


def has_cycle(dominance: np.ndarray) -> bool:
    """True iff the dominance relation contains a directed cycle (certifies that an error occurred)."""
    return bool(np.any(np.diag(transitive_closure(dominance))))


WEIGHTINGS = ("uniform", "adjacent", "mixed")


def order_weights(n_models: int, weighting: str = "uniform") -> np.ndarray:
    """Deterministic weights w[i, j*M + l] on T(v_i), each row summing to one.

    Theorem 1 holds for any such weights. The choices differ only in power:

      * ``"uniform"``  -- the arithmetic mean over all of T(v); the paper's default.
      * ``"adjacent"`` -- uniform over the pairs of T(v) in the same or in adjacent levels of v.
        If T(v) contains a false hypothesis then so does this subset (otherwise transitivity would
        make all of T(v) true), so nothing that can reject v is discarded, and a strict order is
        diluted over M - 1 pairs instead of M(M-1)/2.
      * ``"mixed"``    -- the average of the two.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    v = weak_orders(n_models).astype(np.int16)
    true_pairs = true_pair_masks(n_models)
    gap = v[:, None, :] - v[:, :, None]  # gap[i, j, l] = v_l - v_j
    adjacent = true_pairs & (gap <= 1)

    def normalise(mask: np.ndarray) -> np.ndarray:
        flat = mask.reshape(mask.shape[0], -1).astype(float)
        return flat / flat.sum(axis=1, keepdims=True)

    if weighting == "uniform":
        return normalise(true_pairs)
    if weighting == "adjacent":
        return normalise(adjacent)
    return 0.5 * normalise(true_pairs) + 0.5 * normalise(adjacent)


class ExactCertifier:
    """Restricted closed test: one weighted average of wealths per weak order, rejected at 1/alpha."""

    def __init__(self, n_models: int, alpha: float, *, weighting: str = "uniform") -> None:
        if not 2 <= n_models <= MAX_MODELS_EXACT:
            raise ValueError(f"exact certification supports 2..{MAX_MODELS_EXACT} models")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.n_models = n_models
        self.alpha = alpha
        masks = true_pair_masks(n_models)
        self._masks = masks.reshape(masks.shape[0], -1)
        self.weighting = weighting
        weights = order_weights(n_models, weighting)
        self._cap_factor = 1.0 / weights[weights > 0].min()
        # Single precision at M = 8 keeps the weight matrix near 140 MB.
        self._weights = weights.astype(np.float32 if n_models >= 8 else np.float64)
        self._ranks = order_ranks(n_models)
        self.surviving = np.ones(masks.shape[0], dtype=bool)

    @property
    def failed(self) -> bool:
        """No weak order survives: an error has occurred (probability at most alpha)."""
        return not self.surviving.any()

    def update(self, log_wealth: np.ndarray) -> np.ndarray:
        wealth = _capped_wealth(_check(log_wealth, self.n_models), self.alpha, self._cap_factor).ravel()
        idx = np.flatnonzero(self.surviving)
        if idx.size:
            means = self._weights[idx] @ wealth.astype(self._weights.dtype)
            self.surviving[idx[means >= 1.0 / self.alpha]] = False
        return self.dominance()

    def dominance(self) -> np.ndarray:
        """D[j, l] iff every surviving weak order ranks j strictly above l."""
        m = self.n_models
        possible = self._masks[self.surviving].any(axis=0).reshape(m, m)  # some survivor has v_j <= v_l
        d = ~possible
        np.fill_diagonal(d, False)
        return d

    def rank_sets(self) -> list[np.ndarray]:
        """Projection rank set of each model: its ranks across the surviving weak orders."""
        ranks = self._ranks[self.surviving]
        return [np.unique(ranks[:, j]) for j in range(self.n_models)]


class ShortcutCertifier:
    """Transitivity pooling: certify j > l once
    ``E_jl + sum_k min(E_jk, E_kl) >= M(M-1)/alpha``; then close under transitivity.
    """

    def __init__(self, n_models: int, alpha: float) -> None:
        if n_models < 2:
            raise ValueError("need at least two models")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.n_models = n_models
        self.alpha = alpha
        self._d = np.zeros((n_models, n_models), dtype=bool)

    @property
    def failed(self) -> bool:
        return has_cycle(self._d)

    def pooled(self, log_wealth: np.ndarray) -> np.ndarray:
        w = _capped_wealth(_check(log_wealth, self.n_models), self.alpha)
        # sum over k of min(E[j, k], E[k, l]); the zero diagonal removes k = j and k = l.
        # Done in blocks of rows: the full (M, M, M) minimum is 0.5 GB at M = 395.
        m = self.n_models
        rows = max(1, min(m, int(_POOL_BYTES // (8 * m * m))))
        through = np.empty((m, m))
        for start in range(0, m, rows):
            stop = min(start + rows, m)
            through[start:stop] = np.minimum(w[start:stop, :, None], w[None, :, :]).sum(axis=1)
        return w + through

    def update(self, log_wealth: np.ndarray) -> np.ndarray:
        m = self.n_models
        hit = self.pooled(log_wealth) >= m * (m - 1) / self.alpha
        np.fill_diagonal(hit, False)
        self._d = transitive_closure(self._d | hit)
        return self.dominance()

    def dominance(self) -> np.ndarray:
        return self._d.copy()


class BonferroniCertifier:
    """e-Bonferroni baseline: certify j > l once ``E_jl >= M(M-1)/alpha``."""

    def __init__(self, n_models: int, alpha: float, *, close: bool = True) -> None:
        if n_models < 2:
            raise ValueError("need at least two models")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.n_models = n_models
        self.alpha = alpha
        self.close = close
        self._d = np.zeros((n_models, n_models), dtype=bool)

    @property
    def failed(self) -> bool:
        return has_cycle(self._d)

    def update(self, log_wealth: np.ndarray) -> np.ndarray:
        m = self.n_models
        lw = _check(log_wealth, m)
        hit = lw >= np.log(m * (m - 1) / self.alpha)
        np.fill_diagonal(hit, False)
        self._d |= hit
        if self.close:
            self._d = transitive_closure(self._d)
        return self.dominance()

    def dominance(self) -> np.ndarray:
        return self._d.copy()


class ExactILPCertifier:
    """Algorithm 3: the restricted closed test of Algorithm 1 for any number of models, computed
    exactly by integer programming instead of enumeration.

    A weak order w is encoded by x in {0, 1}^{M(M-1)} with ``x_ab = 1`` iff a <=_w b, i.e. iff
    (a, b) is in T(w). The 0/1 vectors satisfying totality (``x_ab + x_ba >= 1``) and transitivity
    (``x_ab + x_bc - x_ac <= 1``) are exactly the weak orders. Order w survives call time s iff its
    mean wealth is below 1/alpha iff ``g_s(x) := sum_ab x_ab (E_s[a, b] - 1/alpha) < 0``. Pair
    (j, l) is certified at time t iff every weak order with ``x_jl = 1`` has ``g_s(x) >= 0`` at
    some call time s <= t, i.e. iff ``min_x max_{s <= t} g_s(x) >= 0``: a min-max integer program
    (one continuous variable z >= g_s(x) for each s, minimize z). Its optimum is compared with 0,
    so the strict inequality needs no margin: in exact arithmetic the certifier returns the set
    of Algorithm 1 at every call time, running maxima included (Proposition on exactness in
    Appendix A). A *call time* is one invocation of ``update``, so the running maximum here is
    over exactly the times over which :class:`ExactCertifier` takes its own.

    The numerics are conservative, never the other way. A pair is certified only when the
    solver's own lower (dual) bound on the optimum exceeds ``tol``; a minimizer is accepted as a
    witness only when it survives every call time by a margin, ``max_s g_s < -tol``; and when
    neither holds, because the optimum sits within ``tol`` of zero, the pair is left unresolved
    at this call and tried again at the next one (``n_unresolved`` counts these). Certifying a
    subset of Algorithm 1's set is always valid; certifying a superset would not be.

    Integer programs are solved only when needed. A pool of surviving *witness* orders is kept;
    while a live witness has ``x_jl = 1`` the pair (j, l) is known to be uncertified. When a
    pair is uncovered, a program looks for a witness, with times added lazily: solve the min-max
    over a subset S of the call times; a lower bound above tol certifies the pair (the maximum
    over all times is at least the maximum over S); otherwise verify the minimizer against all
    times, add a violated time to S, repeat. The transitivity-pooling bound is tried first,
    since whatever it certifies the exact test certifies. Deciding certification is coNP-hard in
    the worst case (Appendix A); the instances of this paper (M <= 20) solve in milliseconds.
    """

    _UNRESOLVED = object()  # neither certified nor witnessed: the optimum is within tol of zero

    def __init__(self, n_models: int, alpha: float, *, tol_rel: float = 1e-9) -> None:
        if n_models < 2:
            raise ValueError("need at least two models")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        m = n_models
        self.n_models = m
        self.alpha = alpha
        self._c = 1.0 / alpha
        # g_s has M(M-1) terms of order 1/alpha, so the tolerance is relative to that scale
        self._tol = tol_rel * m * (m - 1) * self._c
        self._pairs = np.array([(a, b) for a in range(m) for b in range(m) if a != b], dtype=int)
        self._n = m * (m - 1)
        self._index = -np.ones((m, m), dtype=int)
        self._index[self._pairs[:, 0], self._pairs[:, 1]] = np.arange(self._n)
        self._flat = self._pairs[:, 0] * m + self._pairs[:, 1]  # positions in a flattened M x M matrix
        self._static, self._static_lb, self._static_ub = self._build_static()
        self._hist = np.empty((256, self._n))  # capped wealth at every call time, grown by doubling
        self._t = -1
        self._d = np.zeros((m, m), dtype=bool)
        self._witnesses = np.zeros((0, self._n), dtype=bool)
        self._times: list[int] = []  # call times that have ever rejected a candidate witness
        self.n_ilp_solves = 0
        self.n_unresolved = 0  # pairs left undecided by the numerical tolerance

    # -- construction ---------------------------------------------------------------------------
    def _build_static(self):
        """Totality and transitivity constraints as one sparse matrix with row bounds."""
        m, idx = self.n_models, self._index
        rows, cols, vals, lb, ub = [], [], [], [], []
        r = 0
        for a in range(m):
            for b in range(a + 1, m):
                rows += [r, r]
                cols += [idx[a, b], idx[b, a]]
                vals += [1.0, 1.0]
                lb.append(1.0)
                ub.append(np.inf)
                r += 1
        for a in range(m):
            for b in range(m):
                if b == a:
                    continue
                for c in range(m):
                    if c == a or c == b:
                        continue
                    rows += [r, r, r]
                    cols += [idx[a, b], idx[b, c], idx[a, c]]
                    vals += [1.0, 1.0, -1.0]
                    lb.append(-np.inf)
                    ub.append(1.0)
                    r += 1
        A = csr_matrix((vals, (rows, cols)), shape=(r, self._n))
        return A, np.array(lb), np.array(ub)

    # -- interface ------------------------------------------------------------------------------
    @property
    def failed(self) -> bool:
        """No weak order survives: an error has occurred (probability at most alpha)."""
        return has_cycle(self._d)

    def dominance(self) -> np.ndarray:
        return self._d.copy()

    def update(self, log_wealth: np.ndarray) -> np.ndarray:
        m = self.n_models
        e = _capped_wealth(_check(log_wealth, m), self.alpha).ravel()[self._flat]
        self._t += 1
        if self._t >= self._hist.shape[0]:
            self._hist = np.vstack([self._hist, np.empty_like(self._hist)])
        self._hist[self._t] = e
        # witnesses whose mean reaches 1/alpha are rejected for good
        if self._witnesses.shape[0]:
            alive = self._witnesses @ (e - self._c) < 0.0
            self._witnesses = self._witnesses[alive]
        covered = self._witnesses.any(axis=0) if self._witnesses.shape[0] else np.zeros(self._n, dtype=bool)
        certified = self._d[self._pairs[:, 0], self._pairs[:, 1]]
        pooled = None
        # smallest wealth first: its witness is found easily and covers many pairs
        for p in np.argsort(e, kind="stable"):
            if certified[p] or covered[p]:
                continue
            if pooled is None:
                pooled = ShortcutCertifier(m, self.alpha).pooled(log_wealth).ravel()[self._flat]
            if pooled[p] >= m * (m - 1) * self._c:  # Proposition (transitivity pooling)
                self._d[self._pairs[p, 0], self._pairs[p, 1]] = True
                continue
            witness = self._find_witness(int(p))
            if witness is None:
                self._d[self._pairs[p, 0], self._pairs[p, 1]] = True
            elif witness is self._UNRESOLVED:
                self.n_unresolved += 1  # left uncertified at this call; retried at the next one
            else:
                self._witnesses = np.vstack([self._witnesses, witness[None, :]])
                covered |= witness
        return self.dominance()

    # -- the integer program --------------------------------------------------------------------
    def _find_witness(self, p: int):
        """``None`` if pair p is certified, a surviving weak order if it is not, ``_UNRESOLVED``
        if the solver cannot separate the optimum from zero by more than the tolerance."""
        t = self._t
        times = list(dict.fromkeys(self._times + [t]))
        while True:
            x, bound = self._solve(p, times)  # minimizer and a certified lower bound on the optimum
            if bound > self._tol:
                return None  # every order with this pair is rejected at some time in ``times``
            g = self._hist[: t + 1] @ x - self._c * x.sum()  # g_s(x) at every call time; survival iff < 0
            bad = np.flatnonzero(g >= -self._tol)  # times this order fails, or fails to clear by a margin
            if bad.size == 0:
                return x  # survives every call time with room to spare: a witness
            s = int(bad[np.argmax(g[bad])])
            if s in times:
                # the incumbent fails at a constrained time yet the lower bound is not above the
                # tolerance: the optimum is within tol of zero, so decide nothing here
                return self._UNRESOLVED
            times.append(s)
            self._times.append(s)

    def _solve(self, p: int, times: list[int]) -> tuple[np.ndarray, float]:
        """Minimize z subject to z >= g_s(x) for s in ``times``, x a weak order with x_p = 1.

        Returns the minimizer, rebuilt as a weak order, and the solver's lower (dual) bound on
        the optimum, which is what certification is decided on.
        """
        c, n, k = self._c, self._n, len(times)
        rows = self._hist[times] - c  # g_s(x) = rows[s] @ x
        A = vstack([hstack([self._static, csr_matrix((self._static.shape[0], 1))]),
                    hstack([csr_matrix(-rows), csr_matrix(np.ones((k, 1)))])], format="csr")  # z - g_s(x) >= 0
        lb = np.concatenate([self._static_lb, np.zeros(k)])
        ub = np.concatenate([self._static_ub, np.full(k, np.inf)])
        lower = np.concatenate([np.zeros(n), [-np.inf]])
        upper = np.concatenate([np.ones(n), [np.inf]])
        lower[p] = 1.0
        obj = np.zeros(n + 1)
        obj[n] = 1.0
        res = milp(c=obj, constraints=LinearConstraint(A, lb, ub), integrality=np.concatenate([np.ones(n), [0]]),
                   bounds=Bounds(lower, upper), options={"mip_rel_gap": 0.0})
        self.n_ilp_solves += 1
        if res.status != 0 or res.x is None:
            raise RuntimeError(f"integer program did not solve: {res.message}")
        x = np.round(res.x[:n]).astype(bool)
        # rebuild the order from its levels, so the returned vector is a weak order exactly
        m, idx = self.n_models, self._index
        xm = np.zeros((m, m), dtype=bool)
        xm[self._pairs[:, 0], self._pairs[:, 1]] = x
        strictly_below = (xm.T & ~xm).sum(axis=1)  # number of b with b <_w a
        levels = np.unique(strictly_below, return_inverse=True)[1]
        mask = levels[self._pairs[:, 0]] <= levels[self._pairs[:, 1]]
        if not mask[p]:
            raise RuntimeError("integer program returned an order without the required pair")
        bound = res.mip_dual_bound if res.mip_dual_bound is not None else -np.inf
        return mask, float(bound)
