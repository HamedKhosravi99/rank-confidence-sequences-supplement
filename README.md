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
of weak orders, and the three-model strictness example. Two drivers, `experiments/rank_sets.py`
and `experiments/finite_vs_iid.py`, have no result file of their own; the suite imports them to
check the rank-set inclusions of the duality theorem and the behaviour of the finite-benchmark
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
reproduces its file. Plotting scripts read the JSON and write the figure.

### Main text

| Paper | Driver | Result file |
|---|---|---|
| Figure 1, peeking | `experiments/peeking.py`, plotted by `experiments/plot_peeking.py` | `results/peeking/peeking_mixture_reps5000_n2000.json` |
| Table 1, top twenty against all 395 | `experiments/real_leaderboard.py` and `experiments/full_leaderboard.py` | `results/real_leaderboard/real_leaderboard_orders50.json`, `results/full_leaderboard/full_leaderboard_orders50.json` |
| Figure 2, real leaderboard | `experiments/real_leaderboard.py`, plotted by `experiments/plot_real_leaderboard.py` | `results/real_leaderboard/real_leaderboard_orders50.json` |

### Appendix E

| Paper | Driver | Result file |
|---|---|---|
| E.1, monitoring breaks fixed-sample sets | `experiments/peeking.py` | `results/peeking/peeking_mixture_reps5000_n2000.json`, `results/peeking/peeking_fixed_reps5000_n2000.json` |
| E.2, Table 2, whole leaderboard | `experiments/full_leaderboard.py` | `results/full_leaderboard/full_leaderboard_orders50.json` |
| E.3, Table 3, cost of retirement on all 395 | `experiments/full_leaderboard.py` | `results/full_leaderboard/full_leaderboard_orders50.json` |
| E.3, simulated retirement rules | `experiments/early_stopping.py` | `results/early_stopping/early_stopping_reps300_n5000.json` |
| E.3, cost against `k` and the Pocock comparison | `experiments/early_stopping_k.py` | `results/early_stopping/early_stopping_k_reps200_n5000.json`, `results/early_stopping/diagnostics/early_stopping_k_reps200_n5000_conditional.json` |
| E.4, Table 4, price of anytime validity | `experiments/real_leaderboard.py` | `results/real_leaderboard/real_leaderboard_orders50.json` |
| E.5, Table 5, cross-model dependence | `experiments/dependence.py` | `results/dependence/dependence_full.json` |
| E.6, Table 6, multiplicity correction on real wealths | `experiments/real_multiplicity.py` | `results/real_multiplicity/real_multiplicity_orders50.json` |
| E.6, the same comparison in simulation | `experiments/weighting.py` | `results/weighting/weighting_mixture_reps2000_n4000.json`, `results/weighting/weighting_fixed_reps2000_n4000.json` |
| E.6, shortcut against e-Bonferroni, per look | `experiments/shortcut_vs_bonferroni.py` | `results/weighting/shortcut_vs_bonferroni_mixture.json`, `results/weighting/shortcut_vs_bonferroni_fixed.json` |
| E.6, Table 7, exact test by integer programming | `experiments/exact_ilp.py` | `results/exact_ilp/exact_ilp_orders50.json` |
| E.6, the bet matters more than the correction | `experiments/bets.py` | `results/bets/bets_near_ties-even_spread-tied_leaders-all_tied_reps1000_n4000.json`, `results/bets/bets_close_race_reps300_n20000.json` |
| Implementation note, tightness of the validity bound | `experiments/tightness.py` | `results/tightness/tightness_m4_alpha0.2_runs4000.json` |

### Appendix B

| Paper | Driver | Result file |
|---|---|---|
| Numerical check of the growth and certification-time proposition | `experiments/power_check.py` | `results/power/power_check.json` |

## Layout

```
code/rcs/          the method: wealths, weak orders, certifiers, reporting
code/experiments/  one driver per experiment, plus the two plotting scripts
code/tests/        test suite
code/data/         tinyBenchmarks download and preparation
code/scripts/      Slurm job wrapper for the long runs
results/           one JSON per run, as cited in the table above
data/processed/    manifest with the expected hashes
```

## Long runs

Most drivers finish in minutes on a laptop. The 395-model runs and the larger replication counts
were submitted to a Slurm cluster with `code/scripts/slurm/run_experiment.sbatch`, which runs the
test suite first and then the driver. Timings reported in the paper were measured on one core.
