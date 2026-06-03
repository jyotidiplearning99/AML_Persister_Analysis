#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=small
#SBATCH --time=4:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=prepare-gse176078-brca
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

module --force purge
module load python-data

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

WORKDIR="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "======================================================"
echo "Preparing GSE176078 BRCA PersisterAI upload file"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Workdir: $WORKDIR"
echo "======================================================"

python prepare_gse176078_brca.py \
  --n_cells 20000 \
  --seed 42 \
  --outdir "$WORKDIR/output"

echo "======================================================"
echo "Done: $(date)"
echo "Output files:"
find "$WORKDIR/output" -maxdepth 2 -type f | sort
echo "======================================================"