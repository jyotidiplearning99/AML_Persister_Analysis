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
echo "Persister Inference - 1000 Gene Model"
echo "39 AML Samples (Manifest-based)"
echo "Started: $(date)"
echo "========================================="

export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

# Convert manifest to proper format for the script
python3 -c "
import pandas as pd

# Read manifest
df = pd.read_csv('manifest_39_new_aml_samples.csv')
print(f'Processing {len(df)} samples from manifest')

# Save in the format expected by the script
df.to_csv('manifest_39_formatted.csv', index=False)
"

# Run inference with manifest
srun python3 -u inference_persister_transformer_14092025.py \
    --model-dir ./reduced_model_distilled \
    --out-dir ./results/inference_39_reduced_1k \
    --manifest manifest_39_formatted.csv \
    --no-discover \
    --max-cells 20000 \
    --threshold 0.5

echo "========================================="
echo "Finished: $(date)"
echo "========================================="
