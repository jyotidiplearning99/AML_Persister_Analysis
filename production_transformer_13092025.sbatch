#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00              # Increased to 8h for safety
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=aml-persister-transformer
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL        # Email notifications
#SBATCH --mail-user=your.email@domain.com

# Exit on any error
set -e

# Setup directories
mkdir -p logs temp_data results/{models,predictions,plots,metadata}
export TMPDIR="$PWD/temp_data"

# Load modules
module --force purge
module load tensorflow/2.13    # Specify version for reproducibility

# Environment optimizations
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1      # Real-time output

# Set analysis parameters
export TARGET_RECALL=0.80

# Print system info
echo "========================================="
echo "AML-Enhanced Persister Classifier (Transformer)"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="

# Check GPU availability
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Check if script exists
if [ ! -f "production_transformer_13092025.py" ]; then
    echo "ERROR: production_transformer_13092025.py not found!"
    exit 1
fi

# Run training with error handling
echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..."
srun python -u production_transformer_13092025.py || {
    echo "ERROR: Training failed with exit code $?"
    exit 1
}

# Organize outputs
echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] Organizing outputs..."

# Move model files
for ext in keras pkl h5; do
    mv *.$ext results/models/ 2>/dev/null || true
done

# Move prediction files (AML samples)
mv *_predictions.csv results/predictions/ 2>/dev/null || true
mv AML_*_predictions.csv results/predictions/ 2>/dev/null || true

# Move metadata and logs
mv *.json results/metadata/ 2>/dev/null || true
mv *.txt results/metadata/ 2>/dev/null || true
mv *.log results/logs/ 2>/dev/null || true
mv training_history.csv results/metadata/ 2>/dev/null || true

# Move plots if any
mv *.png results/plots/ 2>/dev/null || true
mv *.pdf results/plots/ 2>/dev/null || true

# Create summary report
cat > results/job_summary.txt <<EOF
Job Summary
===========
Job ID: $SLURM_JOB_ID
Node: $(hostname)
Start: $SLURM_JOB_START_TIME
End: $(date)
Exit Code: $?

Output Files:
$(ls -la results/models/ 2>/dev/null | head -20)

Predictions:
$(ls -la results/predictions/ 2>/dev/null | head -20)
EOF

# Check for key outputs
if [ ! -f "results/models/aml_persister_transformer.keras" ]; then
    echo "WARNING: Main model file not found!"
fi

if [ ! -f "results/metadata/model_metadata.json" ]; then
    echo "WARNING: Metadata file not found!"
fi

# Count prediction files
n_preds=$(ls results/predictions/*.csv 2>/dev/null | wc -l)
echo "Generated $n_preds prediction files"

# Clean temporary files
rm -rf temp_data

echo "========================================="
echo "Completed: $(date)"
echo "Results in: $PWD/results/"
echo "========================================="
