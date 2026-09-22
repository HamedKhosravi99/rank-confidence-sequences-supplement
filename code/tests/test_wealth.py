"""Step 1: pairwise wealth. Includes an exact check of Proposition 1(a)."""

from itertools import permutations

import numpy as np
import pytest

from rcs import PairwiseWealth, log_wealth_paths


def _random_scores(rng, n, m, binary=True):
    return (rng.random((n, m)) < 0.6).astype(float) if binary else rng.random((n, m))


@pytest.mark.parametrize("population", [None, 40, 25])
@pytest.mark.parametrize("binary", [True, False])
def test_online_matches_vectorised(population, binary):
    rng = np.random.default_rng(0)
    n, m = 25, 4
    x = _random_scores(rng, n, m, binary)
    paths = log_wealth_paths(x, population_size=population, lam=0.3)
    w = PairwiseWealth(m, population_size=population, lam=0.3)
    for t in range(n):
        lw = w.update(x[t])
        off = ~np.eye(m, dtype=bool)
        np.testing.assert_allclose(lw[off], paths[t][off], rtol=1e-12, atol=1e-12)


def test_wealth_stays_positive_and_directional():
    x = np.tile([1.0, 0.0], (50, 1))  # model 0 always right, model 1 always wrong
    lw = log_wealth_paths(x, lam=0.25)[-1]
    assert lw[0, 1] > 10 and lw[1, 0] < -10 and np.isfinite(lw[1, 0])


@pytest.mark.parametrize(
    "z",
    [
        [1, -1, 0, -1, 1],  # sum 0: boundary of the null
        [1, 1, -1, -1, -1],  # sum -1
        [0, 0, 0, 0, 0],
        [1, -1, -1, -1, -1],
        [0.5, -0.25, -0.25, 0.75, -1.0],  # non-binary, sum -0.25
    ],
)
@pytest.mark.parametrize("lam", [0.25, 0.9])
def test_finite_population_supermartingale_exact(z, lam):
    """Under H: sum z <= 0, E[E_t] <= 1 for every t, averaging over ALL item orders exactly."""
    z = np.asarray(z, dtype=float)
    assert z.sum() <= 0
    n = z.size
    # two models whose score difference is z: x0 = (1 + z)/2, x1 = (1 - z)/2 gives x0 - x1 = z.
    total = np.zeros(n)
    count = 0
    for perm in permutations(range(n)):
        zz = z[list(perm)]
        x = np.column_stack([(1 + zz) / 2, (1 - zz) / 2])
        total += np.exp(log_wealth_paths(x, population_size=n, lam=lam)[:, 0, 1])
        count += 1
    expected = total / count
    assert np.all(expected <= 1 + 1e-12), expected
    # and the conditional property: expected wealth is non-increasing in t
    assert np.all(np.diff(np.concatenate([[1.0], expected])) <= 1e-12)


def test_finite_population_prefix_uses_full_population_size():
    rng = np.random.default_rng(1)
    x = _random_scores(rng, 30, 3)
    full = log_wealth_paths(x, population_size=30)
    prefix = log_wealth_paths(x[:10], population_size=30)
    np.testing.assert_allclose(prefix[-1][~np.eye(3, dtype=bool)], full[9][~np.eye(3, dtype=bool)])


def test_retirement_freezes_pairs():
    rng = np.random.default_rng(2)
    x = _random_scores(rng, 20, 3)
    w = PairwiseWealth(3)
    for t in range(10):
        w.update(x[t])
    frozen = w.log_wealth.copy()
    w.retire(2)
    for t in range(10, 20):
        row = x[t].copy()
        row[2] = np.nan  # a retired model's score is never read
        w.update(row)
    for j, l in [(0, 2), (2, 0), (1, 2), (2, 1)]:
        assert w.log_wealth[j, l] == frozen[j, l]
    assert w.log_wealth[0, 1] != frozen[0, 1]


def test_input_validation():
    with pytest.raises(ValueError):
        PairwiseWealth(3, lam=1.0)
    with pytest.raises(ValueError):
        PairwiseWealth(3).update(np.array([0.0, 2.0, 1.0]))
    with pytest.raises(ValueError):
        log_wealth_paths(np.zeros((5, 3)), population_size=4)


@pytest.mark.parametrize("bet", ["agrapa", "ons", "mixture"])
@pytest.mark.parametrize("population", [None, 30])
def test_adaptive_bets_online_matches_paths_and_stay_in_range(bet, population):
    rng = np.random.default_rng(3)
    x = _random_scores(rng, 30, 3)
    paths = log_wealth_paths(x, population_size=population, bet=bet)
    w = PairwiseWealth(3, population_size=population, bet=bet)
    off = ~np.eye(3, dtype=bool)
    for t in range(30):
        lam = w._bettor.next()
        assert np.all(lam >= 0) and np.all(lam <= 0.75)
        np.testing.assert_allclose(w.update(x[t])[off], paths[t][off], rtol=1e-12)
    assert np.all(np.isfinite(paths[-1][off]))


@pytest.mark.parametrize("bet", ["agrapa", "ons", "mixture"])
def test_adaptive_bets_are_predictable(bet):
    """The bet placed on item t must not depend on item t: changing item t leaves lam_t unchanged."""
    rng = np.random.default_rng(4)
    x = _random_scores(rng, 12, 3)
    a, b = PairwiseWealth(3, bet=bet), PairwiseWealth(3, bet=bet)
    for t in range(11):
        a.update(x[t]); b.update(x[t])
    np.testing.assert_array_equal(np.asarray(a._bettor.next()), np.asarray(b._bettor.next()))
    a.update(np.array([1.0, 0.0, 0.0])); b.update(np.array([0.0, 1.0, 1.0]))
    assert not np.array_equal(np.asarray(a._bettor.next()), np.asarray(b._bettor.next()))


@pytest.mark.parametrize("bet", ["agrapa", "ons", "mixture"])
@pytest.mark.parametrize("z", [[1, -1, 0, -1, 1], [1, 1, -1, -1, -1], [0.5, -0.25, -0.25, 0.75, -1.0]])
def test_adaptive_bets_keep_the_finite_population_supermartingale_exact(bet, z):
    """Proposition 1(a) with data-dependent bets: E[E_t] <= 1, averaged over all item orders."""
    z = np.asarray(z, dtype=float)
    n = z.size
    total = np.zeros(n)
    for perm in permutations(range(n)):
        zz = z[list(perm)]
        x = np.column_stack([(1 + zz) / 2, (1 - zz) / 2])
        total += np.exp(log_wealth_paths(x, population_size=n, bet=bet)[:, 0, 1])
    expected = total / 120
    assert np.all(expected <= 1 + 1e-12), expected


def test_adaptive_bets_grow_faster_than_the_fixed_bet_on_a_clear_gap():
    rng = np.random.default_rng(5)
    x = (rng.random((3000, 2)) < np.array([0.75, 0.55])).astype(float)
    final = {bet: log_wealth_paths(x, bet=bet, at=np.array([3000]))[0, 0, 1] for bet in ("fixed", "agrapa", "ons")}
    assert final["agrapa"] > final["fixed"] and final["ons"] > 0


def test_evaluation_counter_respects_retirement():
    w = PairwiseWealth(4)
    w.update(np.zeros(4)); w.retire(1); w.update(np.zeros(4)); w.update(np.zeros(4))
    assert w.evaluations == 4 + 3 + 3


def test_mixture_bet_equals_the_average_of_fixed_bet_wealths():
    from rcs.wealth import MIXTURE_GRID

    rng = np.random.default_rng(6)
    x = _random_scores(rng, 200, 3)
    off = ~np.eye(3, dtype=bool)
    for population in (None, 200):
        mix = log_wealth_paths(x, population_size=population, bet="mixture")[-1][off]
        each = np.stack([log_wealth_paths(x, population_size=population, lam=g)[-1][off] for g in MIXTURE_GRID])
        np.testing.assert_allclose(mix, np.logaddexp.reduce(each, axis=0) - np.log(len(MIXTURE_GRID)), rtol=1e-9)
