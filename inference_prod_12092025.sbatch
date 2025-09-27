#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=persister_infer
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@domain.com

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs results/inference

# Clean env and load the TF module that works on Mahti
module --force purge
module load tensorflow/2.13

# Show versions
which python
python - <<'PY'
import sys, numpy, pandas, sklearn, tensorflow as tf
print("PYTHON:", sys.executable)
print("NUMPY :", getattr(numpy, "__version__", "missing"))
print("PANDAS:", getattr(pandas, "__version__", "missing"))
print("SKLEARN:", getattr(sklearn, "__version__", "missing"))
print("TF    :", tf.__version__)
PY

# Runtime tweaks
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Persister Transformer — Inference"
echo "Started: $(date)"
echo "Node:    $(hostname)"
echo "Job ID:  $SLURM_JOB_ID"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

SCRIPT="inference_persister_transformer_14092025.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: $SCRIPT not found in $PWD"
  exit 1
fi

MODEL_DIR="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/models"
OUT_DIR="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/inference"
AML_ROOT="/scratch/project_2010751/AML_scRNA_decrypted"
HEALTHY_ROOT="/scratch/project_2010751/GSE120221_RAW"

# If you want to force a gene file explicitly (optional). Otherwise script auto-finds in ../metadata.
GENES_FILE=""   # e.g. /scratch/project_2010376/.../results/metadata/common_genes.txt

MAX_CELLS=20000
THRESH_OVERRIDE=""  # e.g. 0.31 to enforce a specific threshold

cmd=( srun python -u "$SCRIPT"
  --model-dir "$MODEL_DIR"
  --out-dir "$OUT_DIR"
  --aml-root "$AML_ROOT"
  --healthy-root "$HEALTHY_ROOT"
  --max-cells "$MAX_CELLS"
)

# optional args
[[ -n "$GENES_FILE" ]]      && cmd+=( --genes-file "$GENES_FILE" )
[[ -n "$THRESH_OVERRIDE" ]] && cmd+=( --threshold "$THRESH_OVERRIDE" )
# If you have a manifest, add it here:
# cmd+=( --manifest /path/to/manifest_all.csv --no-discover )

printf 'Running: %q ' "${cmd[@]}"; echo
"${cmd[@]}"

echo "========================================="
echo "Finished: $(date)"
echo "Outputs in: $OUT_DIR/"
echo "========================================="
