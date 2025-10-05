#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=aml_score_scRNA
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/external_scRNA

module --force purge
module load tensorflow/2.13  # only for env; script doesn't use TF

PY=python
SCRIPT=apply_score_on_scRNA_roots.py

AML_ROOT="/scratch/project_2010751/AML_scRNA_decrypted"
HEALTHY_ROOT="/scratch/project_2010751/GSE120221_RAW"
OUTDIR="results/external_scRNA"

$PY "$SCRIPT" \
  --aml-root "$AML_ROOT" \
  --healthy-root "$HEALTHY_ROOT" \
  --outdir "$OUTDIR" \
  --max-cells 50000 \
  --write-cells
