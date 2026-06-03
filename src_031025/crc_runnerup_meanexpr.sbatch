#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=small
#SBATCH --time=1:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=crc-runnerup
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

cd /scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025

source /scratch/project_2010376/JDs_Project/AML_Persister_Analysis/.venv_amlphase2/bin/activate

python crc_runnerup_persample.py \
  --genes-file /scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt \
  --matrix GSE200997_crc/GSE200997_GEO_processed_CRC_10X_raw_UMI_count_matrix.csv.gz \
  --annot GSE200997_crc/GSE200997_GEO_processed_CRC_10X_cell_annotation.csv.gz \
  --condition-col Condition \
  --sample-col samples