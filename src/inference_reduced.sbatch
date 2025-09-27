#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=persister-reduced
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src
mkdir -p logs

# ---------------------------
# Modules / env
# ---------------------------
module --force purge
module load tensorflow/2.13

export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# ---------------------------
# Paths & params
# ---------------------------
MODEL_DIR="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled"
GENES_FILE="${MODEL_DIR}/selected_genes.txt"
OUT_DIR="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/inference_reduced"

# Keep ALL cells; control memory via batching
MAX_CELLS=-1
BATCH_SIZE=8192
THRESHOLD="0.551"

# Discovery roots
DATA_AML_DIR="/scratch/project_2010751/AML_scRNA_decrypted"
DATA_HEALTHY_DIR="/scratch/project_2010751/GSE120221_RAW"

mkdir -p "$OUT_DIR"

ts(){ date +"%Y-%m-%d %H:%M:%S,%3N"; }

echo "========================================="
echo "Persister Inference - REDUCED Model"
echo "Started: $(date)"
echo "Output dir: $OUT_DIR"
command -v nvidia-smi >/dev/null && nvidia-smi || true
python - <<'PY' || true
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
echo "========================================="

# ---- Discovery previews (optional) ----
AML_COUNT=$(
  find "$DATA_AML_DIR" -type f -path "*/filtered_feature_bc_matrix/matrix.mtx.gz" 2>/dev/null \
  | sed 's#/filtered_feature_bc_matrix/.*##' | sort -u | wc -l
)
HEALTHY_TRIPLET_COUNT=$(
  ls "$DATA_HEALTHY_DIR"/GSM*_matrix_*.mtx.gz 2>/dev/null | wc -l
)

printf "%s | INFO | [DISCOVER] AML datasets seen: %d under %s\n"  "$(ts)" "$AML_COUNT" "$DATA_AML_DIR"
printf "%s | INFO | [DISCOVER] Healthy triplets seen: %d under %s\n" "$(ts)" "$HEALTHY_TRIPLET_COUNT" "$DATA_HEALTHY_DIR"

# ---- Sanity: required inputs exist ----
[[ -d "$MODEL_DIR" ]] || { echo "Model dir not found: $MODEL_DIR"; exit 2; }
[[ -f "$GENES_FILE" ]] || { echo "Genes file not found: $GENES_FILE"; exit 2; }
[[ -d "$DATA_AML_DIR" ]] || { echo "AML root not found: $DATA_AML_DIR"; exit 2; }
[[ -d "$DATA_HEALTHY_DIR" ]] || { echo "Healthy root not found: $DATA_HEALTHY_DIR"; exit 2; }

# ---- Run inference ----
srun python -u inference_reduced_gene.py \
  --model-dir "$MODEL_DIR" \
  --genes-file "$GENES_FILE" \
  --threshold "$THRESHOLD" \
  --out-dir "$OUT_DIR" \
  --max-cells "$MAX_CELLS" \
  --batch-size "$BATCH_SIZE" \
  --use-reduced \
  --aml-root "$DATA_AML_DIR" \
  --healthy-root "$DATA_HEALTHY_DIR"

# ---- Post-run summary ----
SUMMARY="$OUT_DIR/inference_results_summary.csv"
if [[ -f "$SUMMARY" ]]; then
  TOTAL_SAMPLES=$( (tail -n +2 "$SUMMARY" || true) | grep -c . || true )
  echo "$(ts) | INFO | Summary shows $TOTAL_SAMPLES samples in $SUMMARY"
else
  echo "$(ts) | WARNING | Summary CSV not found at $SUMMARY"
fi

echo "========================================="
echo "Finished: $(date)"
echo "Output: $OUT_DIR"
echo "========================================="
