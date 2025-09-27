#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=persister-v3-fpr30
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------------------------
# Paths
# ---------------------------
SRC_DIR="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src"
MODEL_DIR="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled"
OUT_DIR="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/inference_reduced"

# Data roots
DATA_AML_DIR="/scratch/project_2010751/AML_scRNA_decrypted"
DATA_HEALTHY_DIR="/scratch/project_2010751/GSE120221_RAW"

# Script (v3)
SCRIPT="${SRC_DIR}/inference_reduced_v2.py"

# ---------------------------
# Controls
# ---------------------------
# Use fewer cells per GSM for calibration to balance cohorts
MAX_CELLS_CALIB=3000      # per-GSM cap during calibration (set -1 to use all)
MAX_CELLS_RUN=-1          # use all cells for the final run
BATCH_SIZE=8192

TARGET_FPR="0.30"         # match ~30% healthy positives like the 13k results
CLIP_MIN="0.20"
CLIP_MAX="0.85"           # avoid brittle 0.90–0.99 caps

# Toggle to skip calibration stage (set to 1 to skip)
SKIP_CALIB=${SKIP_CALIB:-0}

# ---------------------------
# Env / modules
# ---------------------------
mkdir -p "$SRC_DIR" "$OUT_DIR" logs
cd "$SRC_DIR"

module --force purge
module load tensorflow/2.13

export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Persister Inference - REDUCED Model v3"
echo "Started: $(date)"
echo "Output dir: $OUT_DIR"
command -v nvidia-smi >/dev/null && nvidia-smi || true
python - <<'PY' || true
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
echo "========================================="

# ---------------------------
# Sanity checks
# ---------------------------
[[ -f "$SCRIPT" ]] || { echo "Missing script: $SCRIPT"; exit 2; }
[[ -d "$MODEL_DIR" ]] || { echo "Missing model dir: $MODEL_DIR"; exit 2; }
[[ -d "$DATA_HEALTHY_DIR" ]] || { echo "Missing Healthy root: $DATA_HEALTHY_DIR"; exit 2; }
[[ -d "$DATA_AML_DIR" ]] || { echo "Missing AML root: $DATA_AML_DIR"; exit 2; }

# ---------------------------
# Stage A: Calibrate on healthy only (writes threshold to model dir)
# ---------------------------
if [[ "$SKIP_CALIB" -eq 0 ]]; then
  echo "$(date +'%F %T,%3N') | INFO | Calibrating threshold from healthy (FPR=$TARGET_FPR, clip=[$CLIP_MIN,$CLIP_MAX])"
  srun python -u "$SCRIPT" \
    --model-dir "$MODEL_DIR" \
    --out-dir "$OUT_DIR" \
    --healthy-root "$DATA_HEALTHY_DIR" \
    --use-reduced \
    --max-cells "$MAX_CELLS_CALIB" \
    --batch-size "$BATCH_SIZE" \
    --calibrate-healthy \
    --target-fpr "$TARGET_FPR" \
    --clip-min "$CLIP_MIN" \
    --clip-max "$CLIP_MAX" \
    --save-threshold-to-modeldir
else
  echo "$(date +'%F %T,%3N') | INFO | Skipping calibration stage (SKIP_CALIB=1)."
fi

# ---------------------------
# Stage B: Full inference (AML + healthy) using saved/loaded threshold
# ---------------------------
echo "$(date +'%F %T,%3N') | INFO | Running full inference (AML + Healthy) with saved/loaded threshold"
srun python -u "$SCRIPT" \
  --model-dir "$MODEL_DIR" \
  --out-dir "$OUT_DIR" \
  --aml-root "$DATA_AML_DIR" \
  --healthy-root "$DATA_HEALTHY_DIR" \
  --use-reduced \
  --max-cells "$MAX_CELLS_RUN" \
  --batch-size "$BATCH_SIZE"

# ---------------------------
# Post-run summary
# ---------------------------
SUMMARY="$OUT_DIR/inference_results_summary.csv"
if [[ -f "$SUMMARY" ]]; then
  ROWS=$( (tail -n +2 "$SUMMARY" || true) | grep -c . || true )
  echo "$(date +'%F %T,%3N') | INFO | Summary rows (samples): $ROWS at $SUMMARY"
else
  echo "$(date +'%F %T,%3N') | WARNING | Summary CSV not found at $SUMMARY"
fi

echo "========================================="
echo "Finished: $(date)"
echo "Output: $OUT_DIR"
echo "========================================="
