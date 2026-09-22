"""Step 2: certification. Each test names the statement in the paper it checks."""

from itertools import combinations

import numpy as np
import pytest

from rcs import (
    BonferroniCertifier,
    ExactCertifier,
    ShortcutCertifier,
    has_cycle,
    rank_intervals,
    transitive_closure,
    true_pair_masks,
    weak_orders,
)
from rcs.weak_orders import FUBINI, brute_force_weak_orders


def _log_wealth(pairs: dict, m: int, default: float = 1e-3) -> np.ndarray:
    lw = np.full((m, m), np.log(default))
    for (j, l), value in pairs.items():
        lw[j, l] = np.log(value) if value > 0 else -np.inf
    np.fill_diagonal(lw, -np.inf)
    return lw


def _random_log_wealth(rng, m, scale=4.0):
    lw = rng.normal(0.0, scale, size=(m, m))
    np.fill_diagonal(lw, -np.inf)
    return lw


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_weak_order_counts_are_fubini_numbers(m):
    orders = weak_orders(m)
    assert orders.shape == (FUBINI[m], m)
    assert len({tuple(v) for v in orders}) == FUBINI[m]
    if m <= 5:
        assert {tuple(int(a) for a in v) for v in orders} == brute_force_weak_orders(m)


def test_remark_restriction_counts():
    """Remark: 13 of 63 intersections remain for M = 3, 75 of 4095 for M = 4."""
    for m, n_full in [(3, 63), (4, 4095)]:
        masks = true_pair_masks(m).reshape(FUBINI[m], -1)
        assert len({tuple(row) for row in masks}) == FUBINI[m]  # T(W) determines W
        assert 2 ** (m * (m - 1)) - 1 == n_full
        sizes = masks.sum(axis=1)
        assert sizes.min() == m * (m - 1) // 2 and sizes.max() == m * (m - 1)


def _full_closure_rejects(lw, pair, alpha):
    m = lw.shape[0]
    wealth = np.exp(lw)
    all_pairs = [(j, l) for j in range(m) for l in range(m) if j != l]
    for r in range(1, len(all_pairs) + 1):
        for subset in combinations(all_pairs, r):
            if pair in subset and np.mean([wealth[p] for p in subset]) < 1 / alpha:
                return False
    return True


@pytest.mark.parametrize(
    "a_times_alpha, b, restricted, full",
    [(4.0, 0.5, True, False), (3.9, 0.5, False, False), (4.5, 0.1, True, False), (6.5, 0.1, True, True)],
)
def test_example_three_models(a_times_alpha, b, restricted, full):
    """Example (M = 3): restricted test needs a >= 4/alpha - 3 eps; full closure needs more."""
    alpha, eps = 0.05, 1e-3
    lw = _log_wealth({(0, 2): 1e6, (0, 1): a_times_alpha / alpha, (1, 2): b}, 3, default=eps)
    d = ExactCertifier(3, alpha).update(lw)
    assert bool(d[0, 1]) is restricted
    assert _full_closure_rejects(lw, (0, 1), alpha) is full
    orders = weak_orders(3)
    with_h01 = orders[orders[:, 0] <= orders[:, 1]]
    assert len(with_h01) == 8 and (with_h01[:, 0] <= with_h01[:, 2]).sum() == 6


@pytest.mark.parametrize("m", [3, 4, 5])
def test_structure_and_inclusions_on_random_wealth(m):
    """Exact D is a strict partial order; Bonferroni <= shortcut <= exact (Props 2, 4)."""
    rng = np.random.default_rng(m)
    alpha = 0.1
    for _ in range(300):
        lw = _random_log_wealth(rng, m)
        exact = ExactCertifier(m, alpha)
        d = exact.update(lw)
        if exact.failed:
            assert d.sum() == m * (m - 1)
            continue
        assert not has_cycle(d)
        assert np.array_equal(transitive_closure(d), d)
        sc = ShortcutCertifier(m, alpha).update(lw)
        bf = BonferroniCertifier(m, alpha).update(lw)
        assert not np.any(sc & ~d), "shortcut certified something the exact test did not"
        assert not np.any(bf & ~sc), "Bonferroni certified something the shortcut did not"


def test_full_closure_inside_restricted_for_three_models():
    """Proposition (restriction never hurts), M = 3, by enumerating all 63 subsets."""
    rng = np.random.default_rng(7)
    alpha = 0.1
    for _ in range(150):
        lw = _random_log_wealth(rng, 3, scale=3.0)
        d = ExactCertifier(3, alpha).update(lw)
        for j in range(3):
            for l in range(3):
                if j != l and _full_closure_rejects(lw, (j, l), alpha):
                    assert d[j, l]


def test_projection_rank_sets_inside_counting_intervals_and_strict_example():
    """Theorem (rank duality)(c), with the strict example from the appendix."""
    alpha = 0.1
    lw = _log_wealth({(0, 1): 10, (0, 2): 90, (1, 0): 0.2, (2, 1): 21, (1, 2): 0, (2, 0): 0}, 3)
    exact = ExactCertifier(3, alpha)
    d = exact.update(lw)
    assert exact.surviving.sum() == 4
    assert {(int(j), int(l)) for j, l in zip(*np.nonzero(d))} == {(0, 2)}
    lower, upper = rank_intervals(d)
    assert (lower[0], upper[0]) == (1, 2)
    assert exact.rank_sets()[0].tolist() == [1]

    rng = np.random.default_rng(3)
    for _ in range(300):
        exact = ExactCertifier(4, alpha)
        d = exact.update(_random_log_wealth(rng, 4))
        if exact.failed:
            continue
        lower, upper = rank_intervals(d)
        for j, s in enumerate(exact.rank_sets()):
            assert lower[j] <= s.min() and s.max() <= upper[j]


def test_certification_is_permanent():
    alpha = 0.05
    strong = _log_wealth({(0, 1): 1e9}, 2, default=1e-6)
    weak = _log_wealth({}, 2, default=1.0)
    for cls in (ExactCertifier, ShortcutCertifier, BonferroniCertifier):
        c = cls(2, alpha)
        assert c.update(strong)[0, 1]
        assert c.update(weak)[0, 1]


def test_huge_wealth_does_not_overflow():
    lw = np.full((5, 5), -3000.0)
    lw[np.triu_indices(5, 1)] = 3000.0
    np.fill_diagonal(lw, -np.inf)
    for cls in (ExactCertifier, ShortcutCertifier, BonferroniCertifier):
        d = cls(5, 0.05).update(lw)
        assert np.array_equal(d, np.triu(np.ones((5, 5), dtype=bool), 1))


@pytest.mark.parametrize("weighting", ["uniform", "adjacent", "mixed"])
@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_order_weights_are_valid_for_theorem_one(weighting, m):
    """Rows are probability vectors supported inside T(v): all Theorem 1 needs."""
    from rcs.certify import order_weights

    w = order_weights(m, weighting)
    masks = true_pair_masks(m).reshape(FUBINI[m], -1)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert np.all(w >= 0) and not np.any((w > 0) & ~masks)


def test_adjacent_weights_keep_a_false_pair_whenever_one_exists():
    """If T(v) contains a false hypothesis under theta, so does its adjacent-level subset."""
    from rcs.certify import order_weights

    rng = np.random.default_rng(4)
    m = 5
    support = order_weights(m, "adjacent").reshape(FUBINI[m], m, m) > 0
    masks = true_pair_masks(m)
    for _ in range(50):
        theta = rng.integers(0, 3, size=m).astype(float)  # ties are likely
        false_pair = theta[:, None] > theta[None, :]  # H_jl false iff theta_j > theta_l
        has_false = (masks & false_pair[None]).any(axis=(1, 2))
        has_false_adjacent = (support & false_pair[None]).any(axis=(1, 2))
        assert np.array_equal(has_false, has_false_adjacent)


@pytest.mark.parametrize("weighting", ["adjacent", "mixed"])
def test_alternative_weightings_give_partial_orders_and_survive_overflow(weighting):
    rng = np.random.default_rng(9)
    for _ in range(200):
        c = ExactCertifier(4, 0.1, weighting=weighting)
        d = c.update(_random_log_wealth(rng, 4))
        if not c.failed:
            assert not has_cycle(d) and np.array_equal(transitive_closure(d), d)
    lw = np.full((5, 5), -3000.0)
    lw[np.triu_indices(5, 1)] = 3000.0
    np.fill_diagonal(lw, -np.inf)
    d = ExactCertifier(5, 0.05, weighting=weighting).update(lw)
    assert np.array_equal(d, np.triu(np.ones((5, 5), dtype=bool), 1))


# ---- Algorithm 3: exact certification by integer programming ---------------------------------

def _random_walk_paths(rng, m, n_steps, drift_scale=0.6, noise=0.7):
    """Log-wealth paths with a random true order: false nulls drift up, true nulls drift down."""
    theta = rng.random(m)
    drift = np.where(theta[:, None] > theta[None, :], drift_scale, -0.3) * rng.uniform(0.3, 1.5, size=(m, m))
    lw = np.zeros((m, m))
    for _ in range(n_steps):
        lw = lw + drift + rng.normal(0.0, noise, size=(m, m))
        out = lw.copy()
        np.fill_diagonal(out, -np.inf)
        yield out


@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_ilp_certifier_matches_enumeration_along_paths(m):
    """Algorithm 3 returns Algorithm 1's set at every call time, running maxima included."""
    from rcs import ExactILPCertifier

    rng = np.random.default_rng(100 + m)
    alpha = 0.1
    n_paths, n_steps = (12, 60) if m <= 5 else (4, 40)
    for _ in range(n_paths):
        exact, ilp = ExactCertifier(m, alpha), ExactILPCertifier(m, alpha)
        for lw in _random_walk_paths(rng, m, n_steps):
            d_exact = exact.update(lw)
            d_ilp = ilp.update(lw)
            assert np.array_equal(d_exact, d_ilp)
            assert exact.failed == ilp.failed
            if exact.failed:
                break


def test_ilp_certifier_matches_enumeration_on_simulated_scores():
    """Same check on wealths produced by the actual update (mixture bet, finite benchmark)."""
    from rcs import ExactILPCertifier, log_wealth_paths

    rng = np.random.default_rng(5)
    m, n, alpha = 5, 400, 0.05
    theta = np.array([0.72, 0.66, 0.62, 0.6, 0.5])
    x = (rng.random((n, m)) < theta[None, :]).astype(float)
    paths = log_wealth_paths(x, population_size=n, bet="mixture")
    exact, ilp = ExactCertifier(m, alpha), ExactILPCertifier(m, alpha)
    for lw in paths:
        assert np.array_equal(exact.update(lw), ilp.update(lw))
    assert exact.dominance().sum() > 0


def test_ilp_certifier_reports_failure_like_enumeration():
    """Contradictory evidence rejects every weak order: both certifiers report failure."""
    from rcs import ExactILPCertifier

    alpha = 0.1
    lw = _log_wealth({(0, 1): 1e6, (1, 0): 1e6, (1, 2): 1e6, (2, 1): 1e6}, 3, default=1e-6)
    exact, ilp = ExactCertifier(3, alpha), ExactILPCertifier(3, alpha)
    d1, d2 = exact.update(lw), ilp.update(lw)
    assert exact.failed and ilp.failed and np.array_equal(d1, d2)


def test_ilp_certifier_decides_on_the_solver_lower_bound_and_is_conservative():
    """Certification uses the dual bound, so a suboptimal incumbent cannot certify a live pair."""
    from rcs import ExactILPCertifier

    rng = np.random.default_rng(11)
    m, alpha = 5, 0.1
    ilp = ExactILPCertifier(m, alpha)
    seen = []
    solve = ilp._solve

    def spy(p, times):
        x, bound = solve(p, times)
        g = ilp._hist[times] @ x - ilp._c * x.sum()
        seen.append((float(bound), float(g.max())))
        return x, bound

    ilp._solve = spy
    exact = ExactCertifier(m, alpha)
    for lw in _random_walk_paths(rng, m, 40):
        assert np.array_equal(exact.update(lw), ilp.update(lw))
        if exact.failed:
            break
    assert seen, "no integer program was solved"
    # with a zero MIP gap the bound matches the incumbent, and it never exceeds it
    assert all(bound <= incumbent + 1e-6 for bound, incumbent in seen)
    assert ilp.n_unresolved == 0


def test_ilp_certifier_leaves_a_boundary_pair_unresolved_rather_than_certifying():
    """A weak order whose mean sits exactly at 1/alpha is not certified against."""
    from rcs import ExactILPCertifier

    m, alpha = 3, 0.1
    ilp = ExactILPCertifier(m, alpha)
    # every order containing (0, 1) has mean exactly 1/alpha: the optimum of the program is 0
    lw = np.full((m, m), np.log(1.0 / alpha))
    np.fill_diagonal(lw, -np.inf)
    d = ilp.update(lw)
    assert not d.any(), "a pair at the threshold must not be certified"
    assert ilp.n_unresolved > 0


def test_shortcut_pooling_is_blocked_but_unchanged():
    """Blocking the (M, M, M) minimum over rows does not change the pooled statistic."""
    rng = np.random.default_rng(23)
    for m in (3, 5, 12, 40):
        lw = _random_log_wealth(rng, m)
        sc = ShortcutCertifier(m, 0.05)
        w = np.exp(np.minimum(lw, np.log(m * (m - 1) / 0.05) + 1e-9))
        np.fill_diagonal(w, 0.0)
        reference = w + np.minimum(w[:, :, None], w[None, :, :]).sum(axis=1)  # one shot, no blocking
        assert np.allclose(sc.pooled(lw), reference, rtol=0, atol=0)
