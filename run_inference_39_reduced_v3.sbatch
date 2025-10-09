#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=persister_1k_39
#SBATCH --output=logs/inference_1k_39_%j.out
#SBATCH --error=logs/inference_1k_39_%j.err

module purge
module load tensorflow/2.13

cd /scratch/project_2010376/JDs_Project/AML_Persister_Analysis
mkdir -p logs

echo "========================================="
echo "Persister Inference - 1000 Gene Reduced Model"
echo "39 AML Samples (Auto-discovery)"
echo "Started: $(date)"
echo "========================================="

export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

# Paths
MODEL_DIR="./reduced_model_distilled"
OUT_DIR="./results/inference_39_reduced_1k_v3"
AML_ROOT="/scratch/project_2010751/AML_scRNA"  # Will auto-discover cellranger_matrices

echo "Model directory: $MODEL_DIR"
echo "Output directory: $OUT_DIR"
echo "AML data root: $AML_ROOT"
echo "Using script: src/inference_persister_v3_masked.py"

# Run inference with the v3_masked script
srun python3 -u src/inference_persister_v3_masked.py \
    --model-dir "$MODEL_DIR" \
    --out-dir "$OUT_DIR" \
    --aml-root "$AML_ROOT" \
    --use-reduced \
    --max-cells 20000 \
    --batch-size 8192

echo "========================================="
echo "Finished: $(date)"
echo "Results saved to: $OUT_DIR"
echo "========================================="

# Quick summary
if [ -f "$OUT_DIR/inference_results_summary.csv" ]; then
    echo ""
    echo "Quick Summary:"
    python3 -c "
import pandas as pd
df = pd.read_csv('$OUT_DIR/inference_results_summary.csv')
print(f'Total samples processed: {len(df)}')
print(f'Average persister %: {df[\"persister_pct\"].mean():.1f}%')
print(f'Range: {df[\"persister_pct\"].min():.1f}% - {df[\"persister_pct\"].max():.1f}%')
print(f'Total cells analyzed: {df[\"cells\"].sum():,}')
"
fi
