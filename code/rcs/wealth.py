"""Step 1 of the method: betting wealth for every ordered pair of models.

For the ordered pair (j, l) the null is H_jl: theta_j <= theta_l, the paired difference on item t
is Z = x_tj - x_tl in [-1, 1], and the wealth is updated by

    E_jl <- E_jl * (1 + lam_t * Y_t),      Y_t = (Z_t - b_t) / (1 + b_t)  >=  -1,

where b_t is a predictable upper bound on E[Z_t | past] under H_jl, so E[Y_t | past] <= 0:

  * superpopulation (i.i.d. items, ``population_size=None``):  b = 0;
  * finite benchmark of N items revealed in uniformly random order:
        b = max(-1 + delta, min(1, -S / (N - t + 1))),   S = sum of Z over items already seen.

The finite-benchmark bound is the sampling-without-replacement construction of Waudby-Smith and
Ramdas (NeurIPS 2020) applied to each pairwise difference, clipped at ``-1 + delta`` so that every
factor is finite and positive (Proposition 1(a) of the paper). All logs are natural; wealth is stored as log-wealth.

``bound_floor`` (ablation only): replace the finite-benchmark bound by ``max(bound_floor, b)``.
Any floor keeps the update valid for the fixed-benchmark target, since a larger b is still an
upper bound on the conditional mean; ``bound_floor=0`` is the validity-matched baseline that
never exploits a negative drift (a leader ahead so far), so ours against it isolates what the
finite benchmark buys in power. ``None`` (default) is the paper's method.

Bets. Validity needs only that lam_t is in [0, 1] and is computed from the past (Lemma 1), so the
bet can be tuned for power. Each pair has its own bet, a function of that pair's past Y's:

  * ``"fixed"``  -- lam_t = lam (default 0.25);
  * ``"agrapa"`` -- plug-in Kelly bet, lam_t = mean(Y) / mean(Y^2) over the past, clipped to
    [0, bet_cap] (approximate GRAPA of Waudby-Smith and Ramdas, 2024). For Z in {-1, 0, 1} and
    b = 0 the log-optimal constant bet is exactly E[Y] / E[Y^2], which this estimates;
  * ``"ons"``    -- online Newton step on the log-wealth (Cutkosky and Orabona, 2018), on [0, 1/2];
  * ``"mixture"``-- the average of the wealths of several fixed bets (``MIXTURE_GRID``). An average
    of wealth processes is the wealth of one predictable bet, the wealth-weighted mean of the
    grid, so it fits the same update. It has no learning phase and cannot lose more than
    log(len(grid)) against the best bet on the grid.

Two interfaces compute the same numbers (the tests check this):

  * :class:`PairwiseWealth` -- online, one item at a time, supports retiring models;
  * :func:`log_wealth_paths` -- a whole score matrix at once, for simulations.
"""

from __future__ import annotations

import numpy as np

DEFAULT_LAMBDA = 0.25
DEFAULT_DELTA = 0.01
DEFAULT_BET_CAP = 0.75
BETS = ("fixed", "agrapa", "ons", "mixture")
MIXTURE_GRID = (0.03, 0.06, 0.12, 0.25, 0.5)
_ONS_RATE = 2.0 / (2.0 - np.log(3.0))


def _check_params(lam: float, delta: float, bet: str, bet_cap: float) -> None:
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must lie strictly between 0 and 1")
    if not 0.0 < delta <= 1.0:
        raise ValueError("delta must lie in (0, 1]")
    if bet not in BETS:
        raise ValueError(f"bet must be one of {BETS}")
    if not 0.0 < bet_cap < 1.0:
        raise ValueError("bet_cap must lie strictly between 0 and 1")


def _bound(sum_before: np.ndarray, remaining: np.ndarray | float, delta: float,
           floor: float | None = None) -> np.ndarray:
    """Finite-benchmark bound b_t from the running sum before item t and the items remaining."""
    b = np.clip(-sum_before / remaining, -1.0 + delta, 1.0)
    return b if floor is None else np.maximum(b, floor)


class _Bettor:
    """Predictable bets for an (M, M) array of pairs. ``next()`` uses only what was observed."""

    def __init__(self, shape: tuple[int, int], bet: str, lam: float, bet_cap: float) -> None:
        self.bet, self.lam, self.cap = bet, lam, bet_cap
        self._sum_y = np.zeros(shape)
        self._sum_y2 = np.ones(shape)  # one pseudo-observation with Y^2 = 1: the first bet is 0
        self._ons_lam = np.zeros(shape)
        self._ons_a = np.ones(shape)
        self._grid = np.asarray(MIXTURE_GRID)[:, None, None]
        self._grid_log_wealth = np.zeros((len(MIXTURE_GRID), *shape))

    def next(self) -> np.ndarray | float:
        if self.bet == "fixed":
            return self.lam
        if self.bet == "agrapa":
            return np.clip(self._sum_y / self._sum_y2, 0.0, self.cap)
        if self.bet == "mixture":
            w = np.exp(self._grid_log_wealth - self._grid_log_wealth.max(axis=0))
            return (w * self._grid).sum(axis=0) / w.sum(axis=0)
        return self._ons_lam

    def observe(self, y: np.ndarray, used: np.ndarray | float, live: np.ndarray) -> None:
        """Record this item's Y for the pairs in ``live``; ``used`` is the bet that was placed."""
        if self.bet == "agrapa":
            self._sum_y[live] += y[live]
            self._sum_y2[live] += y[live] ** 2
        elif self.bet == "mixture":
            self._grid_log_wealth[:, live] += np.log1p(self._grid * y)[:, live]
        elif self.bet == "ons":
            grad = -y / (1.0 + used * y)
            self._ons_a[live] += grad[live] ** 2
            step = self._ons_lam - _ONS_RATE * grad / self._ons_a
            self._ons_lam[live] = np.clip(step, 0.0, 0.5)[live]


class PairwiseWealth:
    """Online log-wealth for all ordered pairs of ``n_models`` models.

    ``log_wealth[j, l]`` is evidence against H_jl, i.e. evidence that model j is better than
    model l. The diagonal is unused and kept at ``-inf`` (wealth 0).
    """

    def __init__(
        self,
        n_models: int,
        *,
        population_size: int | None = None,
        bet: str = "fixed",
        lam: float = DEFAULT_LAMBDA,
        bet_cap: float = DEFAULT_BET_CAP,
        delta: float = DEFAULT_DELTA,
        bound_floor: float | None = None,
    ) -> None:
        if n_models < 2:
            raise ValueError("need at least two models")
        _check_params(lam, delta, bet, bet_cap)
        if population_size is not None and population_size < 1:
            raise ValueError("population_size must be positive")
        if bound_floor is not None and not -1.0 < bound_floor <= 1.0:
            raise ValueError("bound_floor must lie in (-1, 1]")
        self.n_models = n_models
        self.population_size = population_size
        self.delta = delta
        self.bound_floor = bound_floor
        self.t = 0
        self.log_wealth = np.zeros((n_models, n_models))
        np.fill_diagonal(self.log_wealth, -np.inf)
        self._sum = np.zeros((n_models, n_models))
        self._bettor = _Bettor((n_models, n_models), bet, lam, bet_cap)
        self.active = np.ones(n_models, dtype=bool)
        self.evaluations = 0  # model-item evaluations consumed so far

    def retire(self, model: int) -> None:
        """Stop evaluating ``model``; every pair involving it is frozen from now on.

        Under the finite-benchmark design a retired model cannot be brought back, because the
        bound needs the differences on every revealed item.
        """
        self.active[model] = False

    def update(self, scores: np.ndarray) -> np.ndarray:
        """Process one item. ``scores[j]`` in [0, 1]; entries of retired models are ignored."""
        x = np.asarray(scores, dtype=float)
        if x.shape != (self.n_models,):
            raise ValueError(f"scores must have shape ({self.n_models},)")
        live = self.active
        if np.any(~np.isfinite(x[live])) or np.any(x[live] < 0.0) or np.any(x[live] > 1.0):
            raise ValueError("scores of active models must be finite and lie in [0, 1]")
        self.t += 1
        if self.population_size is not None and self.t > self.population_size:
            raise ValueError("more items than population_size")
        self.evaluations += int(live.sum())

        pair_live = np.outer(live, live)
        np.fill_diagonal(pair_live, False)
        z = np.where(pair_live, x[:, None] - x[None, :], 0.0)
        if self.population_size is None:
            y = z
        else:
            b = _bound(self._sum, self.population_size - self.t + 1, self.delta, self.bound_floor)
            y = (z - b) / (1.0 + b)
        lam = self._bettor.next()
        self.log_wealth[pair_live] += np.log1p((lam * y)[pair_live])
        self._sum[pair_live] += z[pair_live]
        self._bettor.observe(y, lam, pair_live)
        return self.log_wealth


def log_wealth_paths(
    scores: np.ndarray,
    *,
    population_size: int | None = None,
    bet: str = "fixed",
    lam: float = DEFAULT_LAMBDA,
    bet_cap: float = DEFAULT_BET_CAP,
    delta: float = DEFAULT_DELTA,
    at: np.ndarray | None = None,
    bound_floor: float | None = None,
) -> np.ndarray:
    """Log-wealth of every ordered pair after each item.

    ``scores`` has shape (n_items, n_models), rows in the order of evaluation. Returns an array of
    shape (len(at), n_models, n_models) holding the log-wealth after ``at[i]`` items (1-based
    counts, increasing); ``at=None`` returns every time point. No retirement. A fixed bet is
    computed in one vectorised pass; adaptive bets replay :class:`PairwiseWealth` item by item.
    """
    x = np.asarray(scores, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("scores must have shape (n_items, n_models) with n_models >= 2")
    if np.any(~np.isfinite(x)) or np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError("scores must be finite and lie in [0, 1]")
    _check_params(lam, delta, bet, bet_cap)
    n_items, n_models = x.shape
    if population_size is not None and population_size < n_items:
        raise ValueError("population_size cannot be smaller than the number of items revealed")
    times = np.arange(1, n_items + 1) if at is None else np.asarray(at, dtype=int)
    if np.any(times < 1) or np.any(times > n_items) or np.any(np.diff(times) <= 0):
        raise ValueError("look times must be increasing and lie in 1..n_items")

    if bet == "fixed":
        z = x[:, :, None] - x[:, None, :]
        if population_size is None:
            y = z
        else:
            sum_before = np.cumsum(z, axis=0) - z
            remaining = (population_size - np.arange(n_items)).astype(float)[:, None, None]
            b = _bound(sum_before, remaining, delta, bound_floor)
            y = (z - b) / (1.0 + b)
        paths = np.cumsum(np.log1p(lam * y), axis=0)
        diag = np.arange(n_models)
        paths[:, diag, diag] = -np.inf
        return paths[times - 1]

    wealth = PairwiseWealth(n_models, population_size=population_size, bet=bet, lam=lam,
                            bet_cap=bet_cap, delta=delta, bound_floor=bound_floor)
    out = np.empty((times.size, n_models, n_models))
    wanted = {int(t): i for i, t in enumerate(times)}
    for t in range(1, int(times[-1]) + 1):
        lw = wealth.update(x[t - 1])
        if t in wanted:
            out[wanted[t]] = lw
    return out
