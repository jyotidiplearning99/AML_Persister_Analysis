#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=persister-masked
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ts(){ date +"%Y-%m-%d %H:%M:%S,%3N"; }

# ---------------------------
# Paths
# ---------------------------
SRC_DIR="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src"
MODEL_DIR="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled"
AML_DIR="/scratch/project_2010751/AML_scRNA_decrypted"
HEALTHY_DIR="/scratch/project_2010751/GSE120221_RAW"

MASK_TOP100="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/depmap_refined/top100.txt"
MASK_TOP30="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/depmap_refined/top30.txt"

# Output roots
OUT_BASE="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results"
OUT_BASELINE="${OUT_BASE}/inference_reduced_baseline"
OUT_TOP100="${OUT_BASE}/inference_reduced_top100"
OUT_TOP30="${OUT_BASE}/inference_reduced_top30"

mkdir -p "$SRC_DIR" "$OUT_BASELINE" "$OUT_TOP100" "$OUT_TOP30" logs
cd "$SRC_DIR"

# ---------------------------
# Modules / env
# ---------------------------
module --force purge
module load tensorflow/2.13

export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Persister Inference - REDUCED Model (masked)"
echo "Started: $(date)"
command -v nvidia-smi >/dev/null && nvidia-smi || true
python - <<'PY' || true
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
echo "========================================="

# ---------------------------
# Params
# ---------------------------
BATCH_SIZE=8192
MAX_CELLS=-1     # use all cells
# Healthy-based calibration settings
TARGET_FPR="0.05"
CLIP_MIN="0.25"
CLIP_MAX="0.99"

# ---------------------------
# Sanity
# ---------------------------
[[ -f "${SRC_DIR}/inference_persister_v3_masked.py" ]] || { echo "Missing: inference_persister_v3_masked.py"; exit 2; }
[[ -d "$MODEL_DIR" ]] || { echo "Missing model dir: $MODEL_DIR"; exit 2; }
[[ -d "$AML_DIR" ]] || { echo "Missing AML root: $AML_DIR"; exit 2; }
[[ -d "$HEALTHY_DIR" ]] || { echo "Missing Healthy root: $HEALTHY_DIR"; exit 2; }

# Quick discovery preview
AML_COUNT=$(find "$AML_DIR" -type f -path "*/filtered_feature_bc_matrix/matrix.mtx*" 2>/dev/null | sed 's#/filtered_feature_bc_matrix/.*##' | sort -u | wc -l)
HEALTHY_TRIPLET_COUNT=$(ls "$HEALTHY_DIR"/GSM*_matrix_*.mtx.gz 2>/dev/null | wc -l || true)
printf "%s | INFO | AML datasets: %d | Healthy triplets: %d\n" "$(ts)" "$AML_COUNT" "$HEALTHY_TRIPLET_COUNT"

run_one () {
  local OUT_DIR="$1"
  local MASK_FILE="$2"   # "" for baseline
  local MASK_STAGE="pre_cpm"

  mkdir -p "$OUT_DIR"
  echo "$(ts) | INFO | Running -> OUT=${OUT_DIR} MASK=${MASK_FILE:-<none>} STAGE=${MASK_STAGE}"

  # Stage A: Calibrate on healthy with the SAME mask
  echo "$(ts) | INFO | Calibrating on healthy (FPR=${TARGET_FPR}, clip=[${CLIP_MIN},${CLIP_MAX}])"
  if [[ -n "${MASK_FILE}" ]]; then
    srun python -u inference_persister_v3_masked.py \
      --model-dir "$MODEL_DIR" \
      --out-dir   "$OUT_DIR" \
      --healthy-root "$HEALTHY_DIR" \
      --use-reduced \
      --max-cells "$MAX_CELLS" \
      --batch-size "$BATCH_SIZE" \
      --feature-mask "$MASK_FILE" \
      --mask-stage "$MASK_STAGE" \
      --calibrate-healthy \
      --target-fpr "$TARGET_FPR" \
      --clip-min "$CLIP_MIN" \
      --clip-max "$CLIP_MAX" \
      --save-threshold-to-modeldir
  else
    srun python -u inference_persister_v3_masked.py \
      --model-dir "$MODEL_DIR" \
      --out-dir   "$OUT_DIR" \
      --healthy-root "$HEALTHY_DIR" \
      --use-reduced \
      --max-cells "$MAX_CELLS" \
      --batch-size "$BATCH_SIZE" \
      --calibrate-healthy \
      --target-fpr "$TARGET_FPR" \
      --clip-min "$CLIP_MIN" \
      --clip-max "$CLIP_MAX" \
      --save-threshold-to-modeldir
  fi

  # Stage B: Full inference (AML + Healthy) using saved/loaded threshold
  echo "$(ts) | INFO | Full inference (AML + Healthy)"
  if [[ -n "${MASK_FILE}" ]]; then
    srun python -u inference_persister_v3_masked.py \
      --model-dir "$MODEL_DIR" \
      --out-dir   "$OUT_DIR" \
      --aml-root "$AML_DIR" \
      --healthy-root "$HEALTHY_DIR" \
      --use-reduced \
      --max-cells "$MAX_CELLS" \
      --batch-size "$BATCH_SIZE" \
      --feature-mask "$MASK_FILE" \
      --mask-stage "$MASK_STAGE"
  else
    srun python -u inference_persister_v3_masked.py \
      --model-dir "$MODEL_DIR" \
      --out-dir   "$OUT_DIR" \
      --aml-root "$AML_DIR" \
      --healthy-root "$HEALTHY_DIR" \
      --use-reduced \
      --max-cells "$MAX_CELLS" \
      --batch-size "$BATCH_SIZE"
  fi

  # Summary ping
  local SUMMARY="$OUT_DIR/inference_results_summary.csv"
  if [[ -f "$SUMMARY" ]]; then
    local ROWS
    ROWS=$( (tail -n +2 "$SUMMARY" || true) | grep -c . || true )
    echo "$(ts) | INFO | Summary rows: $ROWS at $SUMMARY"
  else
    echo "$(ts) | WARNING | Summary CSV not found at $SUMMARY"
  fi
}

# ---------------------------
# RUNS
# ---------------------------
# 1) Baseline (no mask)
run_one "$OUT_BASELINE" ""

# 2) Top-100 mask
[[ -f "$MASK_TOP100" ]] && run_one "$OUT_TOP100" "$MASK_TOP100" || echo "WARN: missing $MASK_TOP100"

# 3) Top-30 mask
[[ -f "$MASK_TOP30"  ]] && run_one "$OUT_TOP30" "$MASK_TOP30"   || echo "WARN: missing $MASK_TOP30"

echo "========================================="
echo "Finished: $(date)"
echo "Outputs: $OUT_BASELINE | $OUT_TOP100 | $OUT_TOP30"
echo "========================================="
