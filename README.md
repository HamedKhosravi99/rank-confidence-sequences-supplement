# Rank Confidence Sequences: Anytime-Valid Leaderboards

Reference implementation, experiment drivers and result files for the paper.

The method is in `code/rcs/`. Every number, table and figure in the paper is produced by a driver
in `code/experiments/` and stored as JSON under `results/`. The table below maps each one.

## Install and test

```bash
cd code
python -m venv ../.venv && source ../.venv/bin/activate
pip install -e ".[dev,experiments]"
python -m pytest -q
```

The test suite checks the paper's claims numerically, not only the code: that the
finite-benchmark wealth has expectation at most one when averaged exactly over all item orders,
that e-Bonferroni is contained in the shortcut and the shortcut in the exact test, that the full
closure is contained in the exact test at three models by enumerating all 63 subsets, the counts
of weak orders, and the three-model strictness example. The suite also imports
`experiments/rank_sets.py`, which has no result file of its own, and `experiments/finite_vs_iid.py`,
to check the rank-set inclusions of the duality theorem and the behaviour of the finite-benchmark
offset.

Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` before any run; parallelism is
over replicates, not inside NumPy.

## Data

The real-leaderboard experiments use publicly released per-item results for 395 Open LLM
Leaderboard models, as distributed with tinyBenchmarks. `code/data/prepare_tinybenchmarks.py`
downloads them, checks the hash against `data/processed/manifest.json` and writes one `.npz`
score matrix per benchmark. The `.npz` files are not in this repository because they are large
and reproducible from that script.

```bash
cd code && python -m data.prepare_tinybenchmarks
```

## Where each result comes from

Every driver writes JSON to `results/<name>/`. Re-running a driver with the same flags and seed
reproduces its file. Summary scripts read the JSON and print the tables, and plotting scripts read
the JSON and write the figure. All commands run from `code/`, as `python -m experiments.<driver>`.

### Main text

| Paper | Driver | Result file |
|---|---|---|
| Figure 1, peeking | `experiments/peeking.py`, plotted by `experiments/plot_peeking.py` | `results/peeking/peeking_mixture_reps5000_n2000.json` |
| Table 1, top twenty against all 395 | `experiments/real_leaderboard.py` and `experiments/full_leaderboard.py` | `results/real_leaderboard/real_leaderboard_orders50.json`, `results/full_leaderboard/full_leaderboard_orders50.json` |
| Table 2, close real pairs | `experiments/close_pairs.py`, summarized by `experiments/summarize_close_pairs.py` | `results/close_pairs/pairs_<benchmark>_orders0-199.json` (six files), `results/close_pairs/pairs_summary.json` |
| E2, twenty models spaced through the top 40, 80 or 160 ranks | `experiments/ability_spread.py` | `results/ability_spread/spread_orders50.json` |
| Figure 2, real leaderboard with e-Bonferroni | `experiments/real_leaderboard_sampling.py` (design F), plotted by `experiments/plot_real_leaderboard_sampling.py` | `results/real_leaderboard_sampling/fig2_orders50.json`; design F repeats `results/real_leaderboard/real_leaderboard_orders50.json` order by order, and the plotting script checks this before drawing |

### Appendix E

| Paper | Driver | Result file |
|---|---|---|
| E.1, monitoring breaks fixed-sample sets | `experiments/peeking.py` | `results/peeking/peeking_mixture_reps5000_n2000.json`, `results/peeking/peeking_fixed_reps5000_n2000.json` |
| E.1, Table 3, monitoring on the three subsets | `experiments/monitored_subsets.py` | `results/monitored_subsets/panels_orders0-999.json` |
| E.1, Table 3, monitoring on all 395 models | `experiments/monitored_full_leaderboard.py`, with `experiments/monitored_full_leaderboard_errors.py` and `experiments/monitored_full_leaderboard_schedules.py`, summarized by `experiments/summarize_monitored_full_leaderboard.py` | `results/monitored_full_leaderboard/`: `full395_orders0-49.json`, `holm395_orders50-499.json`, `holm395_orders500-999.json`, `exact/holm395_orders0-999.json`, `holm395_errors.json`, `holm395_schedules.json`, `full395_summary.json` |
| E.1, first looks and tied pairs on all 395 models | `experiments/real_leaderboard_checks.py` | `results/real_leaderboard_checks/real_leaderboard_checks.json` |
| E.1, Table 4, number and timing of looks | `experiments/look_schedules.py` (approximation) and `experiments/close_pairs.py` (real pairs) | `results/look_schedules/bridge_paths2000000.json`, `results/close_pairs/pairs_summary.json` |
| E.2, top twenty, twenty spread and top eight | `experiments/real_leaderboard.py` | `results/real_leaderboard/real_leaderboard_orders50.json` |
| E.2, Table 5, whole leaderboard | `experiments/full_leaderboard.py` | `results/full_leaderboard/full_leaderboard_orders50.json` |
| E.2, Table 6, spread of the abilities | `experiments/ability_spread.py`, summarized by `experiments/summarize_ability_spread.py`; the pair-level scale by `experiments/real_leaderboard_checks.py` | `results/ability_spread/spread_orders50.json`, `results/real_leaderboard_checks/real_leaderboard_checks.json` |
| E.3, simulated retirement rules and the Pocock comparison | `experiments/early_stopping.py` | `results/early_stopping/early_stopping_reps300_n5000.json` |
| E.3, cost against `k`, with the Pocock design across `k` | `experiments/early_stopping_k.py` | `results/early_stopping/early_stopping_k_reps200_n5000.json`, `results/early_stopping/diagnostics/early_stopping_k_reps200_n5000_conditional.json` |
| E.3, retirement on the real leaderboard, twenty models | `experiments/real_leaderboard.py` | `results/real_leaderboard/real_leaderboard_orders50.json` |
| E.3, Table 7, cost of retirement on all 395, with the tie counts | `experiments/full_leaderboard.py`; ties by `experiments/exact_ties.py` | `results/full_leaderboard/full_leaderboard_orders50.json`, `results/exact_ties/ties.json` |
| E.3, Table 8, retirement cost against e-Bonferroni | `experiments/certifier_cost.py --part retire`; all 395 models by `experiments/monitored_full_leaderboard.py` | `results/certifier_cost/certifier_cost_retire_orders50.json`, `results/monitored_full_leaderboard/full395_summary.json` |
| E.4, price of anytime validity in simulation, one look at 2000 items | `experiments/peeking.py` | `results/peeking/peeking_mixture_reps5000_n2000.json` |
| E.4, Table 9, price of anytime validity on the real leaderboard | `experiments/real_leaderboard.py` | `results/real_leaderboard/real_leaderboard_orders50.json` |
| E.4, Figure 3 and Table 10, i.i.d. sampling and the finite-benchmark offset | `experiments/real_leaderboard_sampling.py`, summarized by `experiments/summarize_real_leaderboard_sampling.py` and plotted by `experiments/plot_real_leaderboard_sampling.py`; the column with offset max{0, b_t} by `experiments/finite_vs_iid.py` | `results/real_leaderboard_sampling/fig2_orders50.json`, `results/real_leaderboard_sampling/fig2_orders50_summary.json`, `results/finite_vs_iid/finite_vs_baseline_orders50_reps1000.json` |
| E.4, Table 11, width of the rank intervals | `experiments/real_leaderboard_sampling.py` | `results/real_leaderboard_sampling/fig2_orders50.json` |
| E.5, Table 12, cross-model dependence | `experiments/dependence.py` | `results/dependence/dependence_full.json` |
| E.6, Table 13, multiplicity correction on real wealths | `experiments/real_multiplicity.py` | `results/real_multiplicity/real_multiplicity_orders50.json`; the looks ahead of and behind e-Bonferroni for the adjacent-level weights (831 and 15,052) are in `results/real_multiplicity/real_multiplicity_top8_orders50.json` |
| E.6, the shortcut and e-Bonferroni on real data, both sampling models and all 395 models | `experiments/real_leaderboard_sampling.py`, `experiments/monitored_full_leaderboard.py` | `results/real_leaderboard_sampling/fig2_orders50_summary.json`, `results/monitored_full_leaderboard/full395_summary.json` |
| E.6, the same comparison in simulation | `experiments/weighting.py` | `results/weighting/weighting_mixture_reps2000_n4000.json`, `results/weighting/weighting_fixed_reps2000_n4000.json` |
| E.6, shortcut against e-Bonferroni, per look | `experiments/shortcut_vs_bonferroni.py` | `results/weighting/shortcut_vs_bonferroni_mixture.json`, `results/weighting/shortcut_vs_bonferroni_fixed.json` |
| E.6, Table 14, exact test by integer programming, and the shortcut's extra width at every checkpoint | `experiments/exact_ilp.py` | `results/exact_ilp/exact_ilp_orders50.json` |
| E.6, Table 15, run time for 8 to 30 models | `experiments/certifier_cost.py --part timing` | `results/certifier_cost/certifier_cost_timing_orders10.json` |
| E.6, the bet matters more than the correction | `experiments/bets.py` | `results/bets/bets_near_ties-even_spread-tied_leaders-all_tied_reps1000_n4000.json`, `results/bets/bets_close_race_reps300_n20000.json` |
| E.6, Table 16, bets on the real leaderboard | `experiments/bets_real.py` and `experiments/bets_real_online_newton.py`, summarized by `experiments/summarize_bets_real.py` | `results/bets_real/bets_orders50.json`, `results/bets_real_online_newton/bets_hybrid_orders50.json` |
| Opening paragraph (Implementation), tightness of the validity bound | `experiments/tightness.py` | `results/tightness/tightness_m4_alpha0.2_runs4000.json` |

### Not reported in the paper

`experiments/power_check.py` writes `results/power/power_check.json`, a numerical check of the growth
and certification-time rates of Proposition B.4 against the implemented method.

`experiments/full_leaderboard.py --method bonferroni --orders 10` writes
`results/full_leaderboard/full_leaderboard_orders10.json`, the reference against which
`experiments/summarize_monitored_full_leaderboard.py` checks the e-Bonferroni runs on all 395 models.

## Layout

```
code/rcs/          the method: wealths, weak orders, certifiers, reporting
code/experiments/  one driver per experiment, plus summary and plotting scripts
code/tests/        test suite
code/data/         tinyBenchmarks download and preparation
code/scripts/      Slurm job wrapper for the long runs
results/           one JSON per run, as cited in the table above
data/processed/    manifest with the expected hashes
```

## Long runs

Most drivers finish in minutes on a laptop. The 395-model runs and the larger replication counts
were submitted to a Slurm cluster with `code/scripts/slurm/run_experiment.sbatch`, which runs the
test suite first and then the driver. Timings reported in the paper were measured on one core. The
times of Table 15 are wall-clock times from a run with `--workers 7`, so up to seven runs went at a
time on one machine.
