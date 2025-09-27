#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gene-reduction
#SBATCH --output=logs/gene-reduction-%j.out
#SBATCH --error=logs/gene-reduction-%j.err

module load tensorflow/2.13
cd /scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src
python3 train_gene_reduction.py
