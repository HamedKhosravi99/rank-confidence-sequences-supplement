#!/bin/bash
# Generic Slurm job for any experiment driver in code/experiments.
# Anything expected to run longer than about thirty minutes was submitted this way.
#
# Submit from the repository root, e.g.
#   sbatch code/scripts/slurm/run_experiment.sbatch peeking --reps 20000 --n-max 4000
# Everything after the script name is passed to `python -m experiments.<name>`.
#
# Set --account and --partition to whatever your site requires; they are left out here.
#
#SBATCH --job-name=rcs
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch $0 <experiment> [driver arguments]" >&2
    exit 2
fi
EXPERIMENT="$1"; shift

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT/code"
mkdir -p "$REPO_ROOT/runs"

module load anaconda3 2>/dev/null || module load python 2>/dev/null || true
if [[ ! -d "$REPO_ROOT/.venv" ]]; then
    python -m venv "$REPO_ROOT/.venv"
    "$REPO_ROOT/.venv/bin/pip" install --quiet --upgrade pip
    "$REPO_ROOT/.venv/bin/pip" install --quiet -e ".[dev,experiments]"
fi
source "$REPO_ROOT/.venv/bin/activate"

# keep numpy single-threaded; parallelism is over replicates
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "node: $(hostname)   cpus: ${SLURM_CPUS_PER_TASK:-1}"
python -m pytest -q -x                       # refuse to burn hours on broken code
python -m "experiments.${EXPERIMENT}" --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
