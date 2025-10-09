#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=persister_1k_thr031
#SBATCH --output=logs/inference_1k_thr031_%j.out
#SBATCH --error=logs/inference_1k_thr031_%j.err

module purge
module load tensorflow/2.13

cd /scratch/project_2010376/JDs_Project/AML_Persister_Analysis
mkdir -p logs

echo "========================================="
echo "Persister Inference - 1000 Gene Model"
echo "Using SAME threshold as 13k model: 0.31"
echo "39 AML Samples"
echo "Started: $(date)"
echo "========================================="

export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

# Run with threshold 0.31 (same as 13k model)
srun python3 -u src/inference_persister_v3_masked.py \
    --model-dir ./reduced_model_distilled \
    --out-dir ./results/inference_39_reduced_1k_thr031 \
    --aml-root /scratch/project_2010751/AML_scRNA \
    --use-reduced \
    --max-cells 20000 \
    --batch-size 8192 \
    --threshold 0.60

echo "========================================="
echo "Finished: $(date)"
echo "Results saved to: ./results/inference_39_reduced_1k_thr031"
echo "========================================="

# Quick comparison
if [ -f "./results/inference_39_reduced_1k_thr031/inference_results_summary.csv" ]; then
    echo ""
    echo "Quick Summary (threshold=0.31):"
    python3 -c "
import pandas as pd
df = pd.read_csv('./results/inference_39_reduced_1k_thr031/inference_results_summary.csv')
print(f'Total samples: {len(df)}')
print(f'Average persister %: {df[\"persister_pct\"].mean():.1f}%')
print(f'Range: {df[\"persister_pct\"].min():.1f}% - {df[\"persister_pct\"].max():.1f}%')
"
fi
