#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=persister_39_new
#SBATCH --output=logs/inference_39_new_%j.out
#SBATCH --error=logs/inference_39_new_%j.err

module purge
module load tensorflow/2.13

cd /scratch/project_2010376/JDs_Project/AML_Persister_Analysis
mkdir -p logs

echo "========================================="
echo "Persister Inference - 39 New AML Samples"
echo "Started: $(date)"
echo "Job ID:  $SLURM_JOB_ID"
echo "========================================="

# Set environment
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

# Run inference
srun python3 -u inference_persister_transformer_14092025.py \
    --model-dir ./results/models \
    --out-dir ./results/inference_39_new_aml_samples \
    --manifest manifest_39_new_aml_samples.csv \
    --no-discover \
    --max-cells 20000

echo "========================================="
echo "Finished: $(date)"
echo "Results saved to: ./results/inference_39_new_aml_samples/"
echo "========================================="
