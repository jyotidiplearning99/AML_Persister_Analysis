#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=small
#SBATCH --time=2:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=make-gse200997-h5ad
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

cd /scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025

source /scratch/project_2010376/JDs_Project/AML_Persister_Analysis/.venv_amlphase2/bin/activate

python make_GSE200997_h5ad.py

ls -lh GSE200997.h5ad