#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=aml-persister-transformer-LOGO
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Donor-level (StratifiedGroupKFold) Transformer retraining.
#
# This sbatch is a copy of the canonical Transformer training sbatch
# pointed at the LOGO-patched .py file. The architecture, dropout,
# attention heads, optimizer, threshold safety bounds, and all other
# training settings remain IDENTICAL to the canonical run — the only
# difference is the cross-validation strategy (3-fold StratifiedGroupKFold
# with donor groups, instead of 5-fold StratifiedKFold over cells).
#
# REQUIRED before submitting:
#   1. The patched .py file exists in the same directory:
#        production_transformer_LOGO_25042026.py
#   2. n_folds inside that file has been changed from 5 to 3
#      (5-fold cannot satisfy 6 positive / 3 negative donor split).
#   3. Output filenames inside the .py have been renamed with `_logo`
#      suffixes so this run does not overwrite the canonical results.
#   4. Output directories exist:
#        mkdir -p logs results/models_logo results/metadata_logo \
#                 results/predictions_logo results/plots_logo
#
# Wall time: ~6-8 hours on a single A100.

set -e

# Setup directories
mkdir -p logs temp_data results/models_logo results/metadata_logo \
         results/predictions_logo results/plots_logo
export TMPDIR="$PWD/temp_data"

# Load modules — same as canonical Transformer training
module --force purge
module load tensorflow/2.13

# Environment optimizations — identical to canonical run
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Same threshold target as canonical
export TARGET_RECALL=0.80

# Print environment
echo "===================================================================="
echo "AML Transformer LOGO retraining (donor-level StratifiedGroupKFold)"
echo "Start:    $(date)"
echo "Host:     $(hostname)"
echo "Job ID:   $SLURM_JOB_ID"
echo "Account:  ${SLURM_JOB_ACCOUNT}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "===================================================================="

# Check GPU
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Verify the patched script is present
PATCHED_PY="production_transformer_LOGO_25042026.py"
if [ ! -f "$PATCHED_PY" ]; then
    echo "ERROR: $PATCHED_PY not found in $(pwd)"
    echo "Run the patch script first:"
    echo "  python transformer_logo_patch_v2.py \\"
    echo "    --in ../production_transformer_13092025.py \\"
    echo "    --out $PATCHED_PY"
    exit 1
fi

# Sanity-check that key edits are present
if ! grep -q "StratifiedGroupKFold" "$PATCHED_PY"; then
    echo "ERROR: $PATCHED_PY does not contain StratifiedGroupKFold."
    echo "       Was the patch applied?"
    exit 1
fi
if ! grep -q "groups=sids" "$PATCHED_PY"; then
    echo "ERROR: $PATCHED_PY does not contain groups=sids."
    echo "       Was the patch applied?"
    exit 1
fi

# Sanity-check n_folds was reduced to 3
if grep -E "n_folds\s*=\s*5" "$PATCHED_PY" > /dev/null; then
    echo "WARNING: $PATCHED_PY still has 'n_folds = 5'."
    echo "         StratifiedGroupKFold(n_splits=5) cannot be satisfied with 9 donors."
    echo "         Change n_folds to 3 before running, or expect a ValueError at fold time."
    echo "         Continuing anyway — if it fails, this is why."
fi

# Sanity-check output files were renamed (warn but don't block)
if grep -q "results/models[^_]" "$PATCHED_PY"; then
    echo "WARNING: $PATCHED_PY still writes to results/models/ (canonical path)."
    echo "         The LOGO run may overwrite canonical results."
    echo "         Consider sed-renaming to results/models_logo/."
fi

# Run training
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting LOGO Transformer retraining..."
srun python -u "$PATCHED_PY" || {
    echo "ERROR: Training failed with exit code $?"
    exit 1
}

# Organize outputs (same convention as canonical sbatch but with _logo paths)
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Organizing LOGO outputs..."

for ext in keras pkl h5; do
    mv *_logo.$ext results/models_logo/ 2>/dev/null || true
done
mv *_predictions_logo.csv results/predictions_logo/ 2>/dev/null || true
mv *_logo.json results/metadata_logo/ 2>/dev/null || true
mv *_logo.txt results/metadata_logo/ 2>/dev/null || true
mv fold_metrics_logo.csv results/metadata_logo/ 2>/dev/null || true
mv training_history_logo.csv results/metadata_logo/ 2>/dev/null || true
mv *_logo.png results/plots_logo/ 2>/dev/null || true
mv *_logo.pdf results/plots_logo/ 2>/dev/null || true

# Summary
cat > results/job_summary_logo.txt <<EOF
LOGO Retraining Summary
=======================
Job ID:    $SLURM_JOB_ID
Node:      $(hostname)
Start:     $SLURM_JOB_START_TIME
End:       $(date)

Models:
$(ls -la results/models_logo/ 2>/dev/null | head -20)

Metadata:
$(ls -la results/metadata_logo/ 2>/dev/null | head -20)
EOF

# Clean tmp
rm -rf temp_data

echo "===================================================================="
echo "Completed: $(date)"
echo "Send back to Claude:"
echo "  - results/metadata_logo/metadata_logo.json"
echo "  - results/metadata_logo/fold_metrics_logo.csv"
echo "  - logs/aml-persister-transformer-LOGO-${SLURM_JOB_ID}.out"
echo "===================================================================="
