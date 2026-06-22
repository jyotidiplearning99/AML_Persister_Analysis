#!/bin/bash
#SBATCH --job-name=aml_agev3
#SBATCH --account=project_2010376
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/aml_agev3_%j.out
#SBATCH --error=logs/aml_agev3_%j.err

set -euo pipefail

WORKDIR=/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025
SCRIPT=${WORKDIR}/aml_hybrid_analysis_v3_nature_ageing.py
CTRL=${WORKDIR}/data/GSE146590/AML1566_Ctrl
ARAC=${WORKDIR}/data/GSE146590/AML1566_AraC
OUT=${WORKDIR}/results/GSE146590_v3_nature_ageing_$(date +%Y%m%d_%H%M%S)

mkdir -p ${WORKDIR}/logs ${OUT}
cd ${WORKDIR}

module purge || true
module load python-data || true

source /projappl/project_2010376/venvs/scanpy_aml/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python ${SCRIPT} \
  --gse146590_ctrl ${CTRL} \
  --gse146590_arac ${ARAC} \
  --output_dir ${OUT}

echo "Done. Results in: ${OUT}"
