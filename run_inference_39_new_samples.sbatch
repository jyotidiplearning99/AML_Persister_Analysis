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

echo "Starting inference for 39 new AML samples at $(date)"
echo "Model: Using existing trained model"
echo "Data: From /scratch/project_2010751/AML_scRNA/cellranger_matrices"

python3 inference_persister_transformer_14092025.py \
    --model-dir ./results/models \
    --out-dir ./results/inference_39_new_aml_samples \
    --manifest manifest_39_aml_samples.csv \
    --no-discover \
    --max-cells 20000

echo "Finished at $(date)"
echo "Results saved to: ./results/inference_39_new_aml_samples/"
