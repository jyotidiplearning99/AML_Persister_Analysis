#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=small
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=depmap-refine
#SBATCH --output=logs/depmap-%j.out

set -euo pipefail
mkdir -p logs

module load python-data
export PYTHONUNBUFFERED=1

cd /scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src

python3 depmap_gene_refinement.py