#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=medium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=aml-baselines
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Baseline comparison runner for Mahti.
# Mirrors the environment that worked for the Transformer training job
# (production_transformer_13092025.py) but without GPU — none of the four
# baselines (logreg, RF, LSC17, mean-expression) need GPU.
#
# Adjust account / partition if your project allocations differ.
# Run with: sbatch baselines25042026.sbatch
#
# IMPORTANT: this script must be in the same directory as baselines25042026.py
# (which itself must be next to production_transformer_13092025.py so the
#  /scratch data paths resolve identically).

set -euo pipefail

mkdir -p logs

echo "===================================================================="
echo "AML baseline comparison on Mahti — start: $(date)"
echo "Host:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-N/A}"
echo "Account: ${SLURM_JOB_ACCOUNT:-N/A}"
echo "Partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "CPUs:    ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Mem:     ${SLURM_MEM_PER_NODE:-N/A}"
echo "===================================================================="

# Match the environment used by production_transformer_13092025.py
module --force purge
module load python-data/3.10-24.04

# Activate the same venv used for the Transformer training
source ../.venv_amlphase2/bin/activate

# Reproducibility / threading
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Sanity checks
python -V
which python
python -c "import sklearn, numpy, pandas, scipy; print(f'sklearn={sklearn.__version__} numpy={numpy.__version__} pandas={pandas.__version__} scipy={scipy.__version__}')"

if [ ! -f "baselines25042026.py" ]; then
    echo "ERROR: baselines25042026.py not found in $(pwd)"
    exit 1
fi
if [ ! -f "production_transformer_13092025.py" ]; then
    echo "WARNING: production_transformer_13092025.py not in $(pwd) — data paths may not resolve"
fi

# Run
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting baseline comparison..."
srun python -u baselines25042026.py

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
echo "Outputs:"
ls -la baseline_comparison.csv baseline_per_fold.csv baseline_predictions.npz 2>/dev/null || true

echo "===================================================================="
echo "Job complete: $(date)"
echo "===================================================================="
