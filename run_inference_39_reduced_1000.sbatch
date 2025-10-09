#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=persister_1000_39
#SBATCH --output=logs/inference_1000_39_%j.out
#SBATCH --error=logs/inference_1000_39_%j.err

module purge
module load tensorflow/2.13

cd /scratch/project_2010376/JDs_Project/AML_Persister_Analysis
mkdir -p logs

echo "========================================="
echo "Persister Inference - 1000 Gene Reduced Model"
echo "39 New AML Samples"
echo "Started: $(date)"
echo "Job ID:  $SLURM_JOB_ID"
echo "========================================="

# Environment setup
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Paths
MODEL_DIR="./reduced_model_distilled"
OUT_DIR="./results/inference_39_new_reduced_1000"
AML_ROOT="/scratch/project_2010751/AML_scRNA/cellranger_matrices"
MANIFEST="./manifest_39_new_aml_samples.csv"

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

echo "Model directory: $MODEL_DIR"
echo "Output directory: $OUT_DIR"
echo "Processing 39 AML samples from: $AML_ROOT"

# Run inference with the reduced model
srun python3 -u inference_persister_v3_masked.py \
    --model-dir "$MODEL_DIR" \
    --out-dir "$OUT_DIR" \
    --use-reduced \
    --aml-root "$AML_ROOT" \
    --max-cells 20000 \
    --batch-size 8192

echo "========================================="
echo "Finished: $(date)"
echo "Results saved to: $OUT_DIR"
echo "========================================="

# Quick summary
echo ""
echo "Quick Summary:"
if [ -f "$OUT_DIR/inference_results_summary.csv" ]; then
    python3 -c "
import pandas as pd
df = pd.read_csv('$OUT_DIR/inference_results_summary.csv')
print(f'Total samples processed: {len(df)}')
print(f'Average persister %: {df[\"persister_pct\"].mean():.1f}%')
print(f'Total cells analyzed: {df[\"cells\"].sum():,}')
"
fi
