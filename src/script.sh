# update your .sh to call v5 with joint_target and WITHOUT saving threshold:
srun python -u "$SCRIPT" \
  --model-dir "$MODEL_DIR" \
  --genes-file "$GENES_FILE" \
  --out-dir "$OUT_DIR" \
  --use-reduced \
  --aml-root "$DATA_AML_DIR" \
  --healthy-root "$DATA_HEALTHY_DIR" \
  --calib joint_target --aml-target-pct 0.65 --target-fpr 0.05 \
  --clip-min 0.25 --clip-max 0.90
