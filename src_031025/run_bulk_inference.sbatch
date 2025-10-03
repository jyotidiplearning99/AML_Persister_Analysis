#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=bulk_predictions
#SBATCH --output=logs/bulk_pred_%j.out
#SBATCH --error=logs/bulk_pred_%j.err

# Load environment
module load python-data/3.10-24.04

# Change to project directory
cd /scratch/project_2010376/JDs_Project/AML_Persister_Analysis

# Activate virtual environment
source .venv_amlphase2/bin/activate

echo "========================================="
echo "Running Bulk RNA-seq Predictions"
echo "Started: $(date)"
echo "========================================="

# Create output directories if needed
mkdir -p results/bulk_BeatAML/predictions
mkdir -p results/bulk_TCGA/predictions

# BeatAML - Reduced model (1k genes)
echo "Processing BeatAML with 1k model..."
python src/analysis/bulk_inference.py \
    --model-dir reduced_model_distilled \
    --input-csv results/bulk_BeatAML/processed/expression.csv \
    --output results/bulk_BeatAML/predictions/predictions_reduced_1k.csv \
    --use-reduced

# TCGA - Reduced model (1k genes)
echo "Processing TCGA with 1k model..."
python src/analysis/bulk_inference.py \
    --model-dir reduced_model_distilled \
    --input-csv results/bulk_TCGA/processed/expression.csv \
    --output results/bulk_TCGA/predictions/predictions_reduced_1k.csv \
    --use-reduced

# Optional: Also run with full 13k model if it exists
if [ -f "results/models/final_model.h5" ]; then
    echo "Processing with 13k model..."
    
    python src/analysis/bulk_inference.py \
        --model-dir results/models \
        --input-csv results/bulk_BeatAML/processed/expression.csv \
        --output results/bulk_BeatAML/predictions/predictions_full_13k.csv \
        --use-full
    
    python src/analysis/bulk_inference.py \
        --model-dir results/models \
        --input-csv results/bulk_TCGA/processed/expression.csv \
        --output results/bulk_TCGA/predictions/predictions_full_13k.csv \
        --use-full
fi

echo "========================================="
echo "Completed: $(date)"
echo "========================================="
