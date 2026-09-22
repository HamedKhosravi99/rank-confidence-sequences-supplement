"""Step 3 (report), the end-to-end object, the baselines, and a Monte Carlo validity check."""

import numpy as np
import pytest

from rcs import RankConfidenceSequence, rank_intervals, rank_of, tiers, top_k_status
from rcs.baselines import fixed_n_dominance, holm_dominance, pairwise_pvalues


def _d(m, pairs):
    d = np.zeros((m, m), dtype=bool)
    for j, l in pairs:
        d[j, l] = True
    return d


def test_rank_of_with_ties():
    assert rank_of([0.9, 0.9, 0.5, 0.1]).tolist() == [1, 1, 3, 4]


def test_tier_examples_from_the_paper():
    # D = {(a, b)} on {a, b, c}: tiers {a, c} and {b}; c is comparable to neither.
    assert tiers(_d(3, [(0, 1)])).tolist() == [1, 2, 1]
    # a chain 0 > 1 > 2 > 3 (transitively closed) plus an isolated model
    chain = _d(5, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    assert tiers(chain).tolist() == [1, 2, 3, 4, 1]
    # tiers do not need a closed relation: longest chain, not number of dominators
    assert tiers(_d(4, [(0, 3), (1, 3), (2, 3)])).tolist() == [1, 1, 1, 2]


def test_tier_is_at_most_lower_rank_bound_when_closed():
    rng = np.random.default_rng(0)
    for _ in range(200):
        theta = rng.random(6)
        d = (theta[:, None] > theta[None, :]) & (rng.random((6, 6)) < 0.4)
        from rcs import transitive_closure

        d = transitive_closure(d)
        lower, _ = rank_intervals(d)
        assert np.all(tiers(d) <= lower) and np.all(lower <= rank_of(theta))


def test_tiers_detect_cycle():
    with pytest.raises(ValueError):
        tiers(_d(3, [(0, 1), (1, 2), (2, 0)]))


def test_top_k_status():
    d = _d(4, [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)])
    assert top_k_status(d, 1).tolist() == [1, -1, -1, -1]
    assert top_k_status(d, 2).tolist() == [1, 0, 0, -1]


def test_holm_matches_definition():
    p = np.array([[1.0, 0.001, 0.02], [0.9, 1.0, 0.004], [0.99, 0.5, 1.0]])
    d = holm_dominance(p, 0.05)  # sorted: .001<=.05/6, .004<=.05/5, .02>.05/4 -> stop
    assert {(int(j), int(l)) for j, l in zip(*np.nonzero(d))} == {(0, 1), (1, 2)}


def test_mcnemar_pvalue_is_exact_binomial_tail():
    x = np.array([[1, 0]] * 8 + [[0, 1]] * 2 + [[1, 1]] * 5, dtype=float)
    p = pairwise_pvalues(x, "mcnemar")
    assert p[0, 1] == pytest.approx(56 / 1024)  # P(Bin(10, 1/2) >= 8)
    assert fixed_n_dominance(x, 0.05).sum() == 0


def _stream(rng, theta, n):
    """Correlated binary scores: a shared item effect makes all models succeed or fail together."""
    u = rng.random((n, 1))
    return (u < np.asarray(theta)[None, :]).astype(float)


@pytest.mark.parametrize("method", ["exact", "shortcut"])
def test_end_to_end_separates_clear_gaps(method):
    rng = np.random.default_rng(5)
    theta = [0.9, 0.6, 0.3]
    seq = RankConfidenceSequence(3, alpha=0.05, method=method)
    report = None
    for row in _stream(rng, theta, 600):
        report = seq.update(row)
    assert report.covers(theta)
    assert report.lower.tolist() == [1, 2, 3] and report.upper.tolist() == [1, 2, 3]
    assert report.tier.tolist() == [1, 2, 3]


def test_monte_carlo_validity_under_the_hardest_null():
    """All models tied and perfectly dependent items: any certification at any time is an error.

    Theorem 1 bounds the probability by alpha = 0.2; with 300 runs the count should stay well
    under 0.2 * 300 + 4 standard deviations = 88. (The test is conservative in practice.)
    """
    rng = np.random.default_rng(11)
    alpha, runs, errors = 0.2, 300, 0
    theta = np.array([0.5, 0.5, 0.5, 0.5])
    for _ in range(runs):
        x = (rng.random((400, 4)) < theta).astype(float)
        seq = RankConfidenceSequence(4, alpha=alpha, method="exact", lam=0.5)
        ever_wrong = False
        for row in x:
            if not seq.update(row).covers(theta):
                ever_wrong = True
                break
        errors += ever_wrong
    assert errors <= 88, errors


def test_finite_benchmark_end_to_end_covers_true_benchmark_ranks():
    rng = np.random.default_rng(12)
    n, m = 500, 5
    benchmark = (rng.random((n, m)) < np.array([0.85, 0.8, 0.6, 0.6, 0.3])).astype(float)
    theta = benchmark.mean(axis=0)
    order = rng.permutation(n)
    seq = RankConfidenceSequence(m, alpha=0.05, population_size=n)
    for i in order:
        assert seq.update(benchmark[i]).covers(theta)
