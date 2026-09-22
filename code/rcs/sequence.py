"""The whole method in one object: update -> certify -> report, with optional retirement."""

from __future__ import annotations

import numpy as np

from .certify import BonferroniCertifier, ExactCertifier, ExactILPCertifier, ShortcutCertifier
from .report import Report, rank_intervals, tiers
from .weak_orders import MAX_MODELS_EXACT
from .wealth import DEFAULT_BET_CAP, DEFAULT_DELTA, DEFAULT_LAMBDA, PairwiseWealth

_CERTIFIERS = {
    "exact": ExactCertifier,
    "ilp": ExactILPCertifier,
    "shortcut": ShortcutCertifier,
    "bonferroni": BonferroniCertifier,
}


class RankConfidenceSequence:
    """Anytime-valid rank sets, top-k sets and tiers for ``n_models`` models.

    Parameters
    ----------
    n_models, alpha :
        Fixed before any data are seen. With probability at least ``1 - alpha`` no report ever
        contains a false statement.
    population_size :
        ``None`` for an open-ended stream of i.i.d. items. For a fixed benchmark, the number of
        items N; the caller must feed items in a uniformly random order drawn in advance.
    bet :
        ``"mixture"`` (default: average of the wealths of five fixed bets), ``"fixed"`` (constant
        ``lam``), ``"agrapa"`` (plug-in Kelly, capped at ``bet_cap``) or ``"ons"``. Affects power
        only; validity holds for every choice.
    method :
        ``"exact"`` (enumeration, at most 8 models), ``"ilp"`` (the same set for any number of
        models, by integer programming), ``"shortcut"``, ``"bonferroni"``, or ``"auto"``
        (enumeration up to 8 models, otherwise the integer program).
    """

    def __init__(
        self,
        n_models: int,
        alpha: float = 0.05,
        *,
        population_size: int | None = None,
        method: str = "auto",
        bet: str = "mixture",
        lam: float = DEFAULT_LAMBDA,
        bet_cap: float = DEFAULT_BET_CAP,
        delta: float = DEFAULT_DELTA,
        bound_floor: float | None = None,
    ) -> None:
        if method == "auto":
            method = "exact" if n_models <= MAX_MODELS_EXACT else "ilp"
        if method not in _CERTIFIERS:
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.alpha = alpha
        self.wealth = PairwiseWealth(n_models, population_size=population_size, bet=bet, lam=lam,
                                     bet_cap=bet_cap, delta=delta, bound_floor=bound_floor)
        self.certifier = _CERTIFIERS[method](n_models, alpha)

    @property
    def t(self) -> int:
        return self.wealth.t

    @property
    def evaluations(self) -> int:
        """Model-item evaluations consumed so far (retired models cost nothing)."""
        return self.wealth.evaluations

    @property
    def active(self) -> np.ndarray:
        """Mask of models still being evaluated."""
        return self.wealth.active

    def retire(self, model: int) -> None:
        """Stop evaluating a model. Any rule based on past reports is allowed."""
        self.wealth.retire(model)

    def update(self, scores: np.ndarray) -> Report:
        """Feed one item's scores (length ``n_models``, values in [0, 1]) and get the report."""
        self.observe(scores)
        return self.report()

    def observe(self, scores: np.ndarray) -> None:
        """Feed one item without issuing a report. Reporting only at some times is always valid."""
        self.wealth.update(scores)

    def report(self) -> Report:
        """Certify on the evidence so far and report. May be called at any time, any number of times."""
        return self._report(self.certifier.update(self.wealth.log_wealth))

    def _report(self, dominance: np.ndarray) -> Report:
        failed = self.certifier.failed
        lower, upper = rank_intervals(dominance)
        rank_sets = None
        n_surviving = None
        if isinstance(self.certifier, ExactCertifier):
            n_surviving = int(self.certifier.surviving.sum())
            if not failed:
                rank_sets = self.certifier.rank_sets()
        return Report(
            t=self.t,
            dominance=dominance,
            lower=lower,
            upper=upper,
            tier=None if failed else tiers(dominance),
            rank_sets=rank_sets,
            failed=failed,
            n_surviving_orders=n_surviving,
            active=self.wealth.active.copy(),
        )
