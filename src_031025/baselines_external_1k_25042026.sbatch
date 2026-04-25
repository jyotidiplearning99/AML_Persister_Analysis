#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=medium
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=aml-baselines-fair-1k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Fair-comparison baselines on the 1,000-gene panel — apples-to-apples vs Transformer.
# Wall time: typically 30-60 min.

set -euo pipefail
mkdir -p logs

echo "===================================================================="
echo "AML baselines - FAIR 1k-panel BeatAML comparison"
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

if [ ! -f "baselines_external_1k_25042026.py" ]; then
    echo "ERROR: baselines_external_1k_25042026.py not found in $(pwd)"
    exit 1
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting FAIR 1k-panel baseline run..."
srun python -u baselines_external_1k_25042026.py

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
ls -la fair_baseline_internal.csv \
       fair_baseline_external_per_drug.csv \
       fair_baseline_external_aggregate.csv 2>/dev/null || true
echo "===================================================================="
