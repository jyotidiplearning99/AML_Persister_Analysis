#!/bin/bash
#SBATCH --job-name=aml_ageing
#SBATCH --account=project_2010376
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/aml_ageing_%j.out
#SBATCH --error=logs/aml_ageing_%j.err

set -euo pipefail

WORKDIR=/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025
SCRIPT=${WORKDIR}/aml_hybrid_analysis_patched.py
CTRL=${WORKDIR}/data/GSE146590/AML1566_Ctrl
ARAC=${WORKDIR}/data/GSE146590/AML1566_AraC
OUT=${WORKDIR}/results/GSE146590_$(date +%Y%m%d_%H%M%S)

mkdir -p ${WORKDIR}/logs ${OUT}
cd ${WORKDIR}

module purge
module load python-data || true

# Use the venv you created on the login node.
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
