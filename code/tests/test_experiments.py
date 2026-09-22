"""The experiment drivers use fast paths; check them against the reference implementations."""

import numpy as np
import pytest

from experiments.peeking import LOOKS, _fixed_n_reports, one_replicate, simulate, true_theta
from rcs.baselines import fixed_n_dominance


def test_true_theta_respects_ties_and_order():
    theta = true_theta(np.array([1.0, 1.0, 0.2, -0.5]))
    assert theta[0] == theta[1] and theta[1] > theta[2] > theta[3]
    # Monte Carlo agreement with the quadrature value
    rng = np.random.default_rng(0)
    x = simulate(rng, np.array([1.0, 1.0, 0.2, -0.5]), 200_000)
    np.testing.assert_allclose(x.mean(axis=0), theta, atol=5e-3)


def test_fast_fixed_n_reports_match_reference():
    rng = np.random.default_rng(1)
    x = simulate(rng, np.array([1.0, 0.7, 0.7, 0.1]), 400)
    times = np.array([20, 100, 250, 400])
    fast = _fixed_n_reports(x, times, alpha=0.1)
    for i, t in enumerate(times):
        for name, test in (("holm_mcnemar", "mcnemar"), ("holm_ztest", "ztest")):
            assert np.array_equal(fast[name][i], fixed_n_dominance(x[:t], 0.1, test)), (name, t)


def test_one_replicate_shape_and_single_look_consistency():
    out = one_replicate((123, [0.8, 0.8, 0.0], 400, 0.05, 0.25, "mixture"))
    assert set(k for _, k in out) == set(LOOKS)
    for (name, k), (ever, stop, cert) in out.items():
        assert isinstance(ever, bool) and isinstance(stop, bool) and 0 <= cert <= 2
        if k == 1:
            assert ever == stop  # one look: the only report is the final report


def _report_from(pairs, m, k):
    from rcs.report import Report, rank_intervals, tiers

    d = np.zeros((m, m), dtype=bool)
    for j, l in pairs:
        d[j, l] = True
    lower, upper = rank_intervals(d)
    return Report(t=0, dominance=d, lower=lower, upper=upper, tier=tiers(d))


def test_retirement_rules():
    from experiments.early_stopping import _to_retire

    # 4 models, k = 1. Model 0 beats 2 and 3; model 1 beats 2 and 3; 0 versus 1 is open.
    rep = _report_from([(0, 2), (0, 3), (1, 2), (1, 3)], 4, k=1)
    assert _to_retire("none", rep, 1).tolist() == [False] * 4
    assert _to_retire("stop_at_goal", rep, 1).tolist() == [False] * 4  # status of 0 and 1 is open
    assert _to_retire("topk", rep, 1).tolist() == [False, False, True, True]  # 2 and 3 are certified out
    assert _to_retire("topk_safe", rep, 1).tolist() == [False, False, True, True]  # and settled against 0, 1
    assert _to_retire("resolved", rep, 1).tolist() == [False] * 4  # pair (2, 3) is still open

    # Model 3 is certified out of the top 1 by model 0 alone, but its pair with open model 1 is
    # unsettled: the naive rule retires it, the safe rule keeps it.
    rep = _report_from([(0, 3)], 4, k=1)
    assert _to_retire("topk", rep, 1).tolist() == [False, False, False, True]
    assert _to_retire("topk_safe", rep, 1).tolist() == [False] * 4


def test_early_stopping_run_saves_cost_without_errors():
    from experiments.early_stopping import run_rule

    rng = np.random.default_rng(8)
    x = simulate(rng, np.array([1.5, 0.5, -0.5, -1.5]), 1500)
    theta = x.mean(axis=0)
    full = run_rule(x, theta, "none", 1, "exact", "ours/mixture", 0.05, 25)
    early = run_rule(x, theta, "topk", 1, "exact", "ours/mixture", 0.05, 25)
    pocock = run_rule(x, theta, "topk", 1, "exact", "pocock_bonf", 0.05, 25)
    assert pocock["cost"] < 1.0 and pocock["goal_topk"]
    assert full["cost"] == 1.0 and not full["wrong"] and full["goal_topk"]
    assert early["cost"] < 0.5 and not early["wrong"] and early["goal_topk"]


def test_real_leaderboard_helpers():
    import pytest

    from experiments.real_leaderboard import DATA, _fixed_n_at, choose, load

    score = np.array([0.9, 0.1, 0.5, 0.7, 0.3])
    assert choose("top8", score).tolist() == [0, 3, 2, 4, 1]
    rng = np.random.default_rng(2)
    x = simulate(rng, np.array([1.0, 0.5, 0.0]), 300)
    fast = _fixed_n_at(x, np.array([50, 300]), 0.1)
    for i, t in enumerate((50, 300)):
        assert np.array_equal(fast["fixed_n"][i], fixed_n_dominance(x[:t], 0.1, "mcnemar"))
        assert np.array_equal(fast["fixed_z"][i], fixed_n_dominance(x[:t], 0.1, "ztest"))
    if not (DATA / "arc.npz").exists():
        pytest.skip("processed leaderboard data not present")
    scores, models = load("arc")
    assert scores.shape == (1172, 395) and len(models) == 395
    assert scores.min() >= 0 and scores.max() <= 1


def test_romano_wolf_rejects_clear_gaps_and_respects_the_null():
    from experiments.real_leaderboard import romano_wolf

    rng = np.random.default_rng(5)
    x = simulate(rng, np.array([1.5, 0.0, -1.5]), 400)
    d = romano_wolf(x, 0.05, rng)
    assert d[0, 1] and d[1, 2] and d[0, 2] and not d[1, 0] and not d[2, 0]
    # global null, four tied models: family-wise error near or below alpha
    errors = 0
    for _ in range(200):
        x = (rng.random((200, 4)) < 0.5).astype(float)
        errors += romano_wolf(x, 0.1, rng).any()
    assert errors <= 0.1 * 200 + 3 * np.sqrt(0.1 * 0.9 * 200)  # <= 33


def test_pocock_constant_and_sequence():
    from rcs.baselines import PocockSequence, pocock_constant

    assert abs(pocock_constant(1, 0.05) - 1.645) < 0.01  # one look: the usual one-sided z
    assert pocock_constant(10, 0.05) > pocock_constant(5, 0.05) > pocock_constant(1, 0.05)
    rng = np.random.default_rng(7)
    x = simulate(rng, np.array([1.5, 0.0, -1.5]), 1000)
    seq = PocockSequence(3, 0.05, 1000, looks=10)
    for row in x:
        seq.observe(row)
    d = seq.report().dominance
    assert d[0, 1] and d[1, 2] and d[0, 2] and not d[1, 0]
    # no certification between the pre-specified looks
    seq = PocockSequence(3, 0.05, 1000, looks=10)
    for row in x[:99]:
        seq.observe(row)
    assert not seq.report().dominance.any()


def test_real_multiplicity_invariants_on_one_order():
    """E3 on real data: on identical wealths, e-Bonferroni <= shortcut in true pairs, and no false pair."""
    from experiments.real_leaderboard import DATA
    from experiments.real_multiplicity import aggregate, one_order

    if not (DATA / "arc.npz").exists():
        pytest.skip("processed leaderboard data not present (run from code/ after the data step)")
    run = one_order((7, "arc", "top20", 0.05, 0.10, "mixture"))
    c = run["certifiers"]
    assert set(c) == {"e_bonferroni", "shortcut"}  # 20 models: no exact test
    for f in run["certifiers"]["shortcut"]["at"]:
        assert c["e_bonferroni"]["at"][f] <= c["shortcut"]["at"][f] <= run["true_pairs"]
    assert not c["e_bonferroni"]["ever_wrong"] and not c["shortcut"]["ever_wrong"]
    assert run["per_look"]["shortcut<e_bonferroni"] == 0
    agg = aggregate([run, run])
    assert agg["orders"] == 2 and agg["per_look"]["looks_total"] == 2 * run["looks"]
    assert agg["certifiers"]["shortcut"]["true_certified"]["1.0"] == run["certifiers"]["shortcut"]["at"]["1.0"]


def test_rank_sets_invariants_on_one_replicate():
    """E3a: projection set inside the exact counting interval, which is inside e-Bonferroni's."""
    import numpy as np
    from experiments.rank_sets import one_sim_replicate
    from experiments.weighting import SCENARIOS

    out = one_sim_replicate((7, SCENARIOS["even_spread"], 600, 20, 0.05, "mixture"))
    for look in out["per_look"]:
        assert look["size"]["projection"] <= look["size"]["exact"] + 1e-12
        assert look["size"]["exact"] <= look["size"]["e_bonferroni"] + 1e-12
        assert 0.0 <= look["tighter"]["projection"] <= 1.0
        assert look["holes"]["projection"] >= -1e-12
        assert abs(look["holes"]["projection"] - (look["size"]["exact"] - look["size"]["projection"])) < 1e-9
    assert set(out["ever_wrong"]) == {"projection", "exact", "e_bonferroni"}


def test_finite_and_iid_wealth_agree_at_item_one_and_for_a_huge_population():
    """Finite-benchmark ablation: b_1 = 0 makes the two bounds agree after item 1, and a huge
    population reproduces the i.i.d. path."""
    import numpy as np
    from experiments.peeking import simulate
    from rcs import log_wealth_paths

    x = simulate(np.random.default_rng(1), np.linspace(1.0, 0.0, 6), 300)
    off = ~np.eye(6, dtype=bool)
    fin = log_wealth_paths(x, population_size=len(x), bet="mixture")
    iid = log_wealth_paths(x, population_size=None, bet="mixture")
    huge = log_wealth_paths(x, population_size=10**9, bet="mixture")
    assert np.allclose(fin[0][off], iid[0][off])
    assert np.abs(fin[:, off] - iid[:, off]).max() > 0.1  # the finite bound does engage later
    assert np.abs(huge[:, off] - iid[:, off]).max() < 1e-4


def test_bound_floor_baseline_is_max_of_zero_and_finite_bound():
    """Finite-benchmark ablation: bound_floor=0 gives b = max(0, b_finite) at every item and pair."""
    import numpy as np
    from experiments.peeking import simulate
    from rcs.wealth import _bound

    x = simulate(np.random.default_rng(3), np.linspace(1.0, 0.0, 6), 400)
    n = len(x)
    z = x[:, :, None] - x[:, None, :]
    before = np.cumsum(z, axis=0) - z
    remaining = (n - np.arange(n)).astype(float)[:, None, None]
    b_finite = _bound(before, remaining, 0.01)
    b_base = _bound(before, remaining, 0.01, 0.0)
    assert np.allclose(b_base, np.maximum(b_finite, 0.0))
    assert (b_base >= b_finite).all()  # a larger offset is still a valid upper bound


def test_gaussian_copula_preserves_marginals_and_sets_dependence():
    """Dependence stress test: marginals fixed across regimes, correlation sign set by a and rho."""
    import numpy as np
    from experiments.dependence import REGIMES, accuracies, gaussian_copula

    theta = accuracies("even_spread")
    rng = np.random.default_rng(0)
    for regime, (rho, a) in REGIMES.items():
        x = gaussian_copula(rng, theta, rho, np.asarray(a, dtype=float), 60_000)
        assert np.abs(x.mean(axis=0) - theta).max() < 0.01, regime
        c = np.corrcoef(x.T)
        if regime == "independent":
            assert np.abs(c[~np.eye(6, dtype=bool)]).max() < 0.03
        if regime == "strong":
            assert c[0, 1] > 0.5
        if regime == "mixed":
            assert c[0, 1] > 0.4 and c[0, 3] < -0.3
