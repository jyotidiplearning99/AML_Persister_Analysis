#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=medium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=aml-baselines-logo
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Donor-level (StratifiedGroupKFold) cross-validation for the four baselines.
# Same environment as baselines.sbatch (no GPU needed).
# Wall time: typically 1-3 hours.

set -euo pipefail

mkdir -p logs

echo "===================================================================="
echo "AML baselines - donor-level LOGO cross-validation"
echo "Start: $(date)"
echo "Host:  $(hostname)"
echo "Job:   ${SLURM_JOB_ID:-N/A}"
echo "===================================================================="

module --force purge
module load python-data/3.10-24.04
source ../.venv_amlphase2/bin/activate

export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python -V
python -c "import sklearn, scipy; print(f'sklearn={sklearn.__version__} scipy={scipy.__version__}')"

if [ ! -f "baselines_logo_25042026.py" ]; then
    echo "ERROR: baselines_logo_25042026.py not found in $(pwd)"
    exit 1
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting LOGO cross-validation..."
srun python -u baselines_logo_25042026.py

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
ls -la logo_baseline_comparison.csv logo_baseline_per_fold.csv 2>/dev/null || true
echo "===================================================================="
