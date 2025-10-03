
#!/usr/bin/env python3
"""
Bulk RNA-seq inference for persister model - FINAL COMPLETE VERSION
With RAW space alignment, neutral imputation, and PC analysis
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
import argparse
import warnings
import os
import sys
import hashlib
import json

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def run_bulk_inference(model_dir, input_csv, output_path, use_reduced=True, 
                      threshold_override=None, global_shift_correct=False, 
                      align_gene_medians=False, match_variance=False):
    
    print(f"\n{'='*60}")
    print("BULK RNA-SEQ PERSISTER INFERENCE - COMPLETE VERSION")
    print(f"{'='*60}")
    
    model_dir = Path(model_dir)
    
    # Determine paths
    if use_reduced:
        model_path = model_dir / 'model_reduced.h5'
        scaler_path = model_dir / 'scaler_reduced.pkl'
        pca_path = model_dir / 'pca_reduced.pkl'
        threshold_path = model_dir / 'threshold_reduced.pkl'
        genes_path = model_dir / 'selected_genes.txt'
        print("Using REDUCED model configuration")
    else:
        model_path = model_dir / 'final_model.h5'
        scaler_path = model_dir / 'scaler.pkl'
        pca_path = model_dir / 'pca.pkl'
        threshold_path = model_dir / 'threshold.pkl'
        genes_path = model_dir / 'common_genes.txt'
        print("Using FULL model configuration")
    
    # Verify files exist
    print("\nVerifying required files...")
    required_files = {
        "Model": model_path,
        "Scaler": scaler_path,
        "PCA": pca_path,
        "Gene list": genes_path
    }
    
    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
        print(f"  ✓ {name}: {path.name}")
    
    # Load gene list
    with open(genes_path) as f:
        model_genes = [line.strip().upper() for line in f if line.strip()]
    print(f"\nModel expects {len(model_genes)} genes")
    
    # Gene list fingerprint for integrity
    gene_fingerprint = hashlib.sha1('\n'.join(model_genes).encode()).hexdigest()[:12]
    print(f"Gene list fingerprint: {gene_fingerprint}")
    
    # Load model
    print(f"\nLoading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print(f"Model input shape: {model.input_shape}")
    
    # Load scaler - CRITICAL for alignment and imputation
    print("\nLoading preprocessing components...")
    scaler = joblib.load(scaler_path)
    print(f"  ✓ Scaler loaded: {scaler.__class__.__name__}")
    
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'var_'):
        print(f"  Scaler mean range: [{scaler.mean_.min():.2f}, {scaler.mean_.max():.2f}]")
        print(f"  Scaler var range: [{scaler.var_.min():.3f}, {scaler.var_.max():.3f}]")
    
    # Load PCA
    pca = joblib.load(pca_path)
    n_components = getattr(pca, 'n_components_', getattr(pca, 'n_components', 'Unknown'))
    print(f"  ✓ PCA loaded: n_components={n_components}")
    
    if hasattr(pca, "explained_variance_"):
        ev = pca.explained_variance_
        print(f"  PCA explained_variance_ range: [{ev.min():.4f}, {ev.max():.4f}]")
    
    # Test neutral baseline
    print("\nTesting neutral baseline (z≈0)...")
    try:
        X0_raw = scaler.mean_.reshape(1, -1).astype("float32")
        X0_scaled = scaler.transform(X0_raw)
        X0_pca = pca.transform(X0_scaled).astype("float32")
        p0 = float(model.predict(X0_pca, verbose=0).flatten()[0])
        print(f"  Neutral baseline probability: {p0:.3f}")
    except Exception as e:
        print(f"  Neutral baseline check failed: {e}")
        p0 = None
    
    # Load threshold (safer version)
    if threshold_override is not None:
        threshold = float(np.clip(threshold_override, 0.05, 0.95))
        print(f"\nUsing override threshold: {threshold:.3f}")
    elif threshold_path.exists():
        try:
            t = float(joblib.load(threshold_path))
            if not (0.10 <= t <= 0.90):
                print(f"Loaded threshold {t:.3f} looks suspicious; using 0.31")
                t = 0.31
            threshold = t
        except:
            threshold = 0.31
    else:
        threshold = 0.31
    print(f"Threshold: {threshold:.3f}")
    
    # Load expression data
    print(f"\nLoading expression data...")
    expr = pd.read_csv(input_csv, index_col=0)
    
    if expr.shape[1] > expr.shape[0]:
        expr = expr.T
    
    print(f"Expression shape: {expr.shape[0]} genes × {expr.shape[1]} samples")
    
    # Clean gene names
    expr.index = [str(g).strip().upper() for g in expr.index]
    
    # Data quality check
    data_min = expr.min().min()
    data_max = expr.max().max()
    print(f"Data range: [{data_min:.2f}, {data_max:.2f}]")
    
    if data_min < -2:
        raise ValueError("Data appears Z-SCORED! Use expression_for_model.csv")
    
    # Align to model genes
    expr_aligned = expr.reindex(model_genes)  # Keep NaNs
    
    # Identify missing genes (NaN or all-zero)
    missing_by_nan = expr_aligned.isna().all(axis=1)
    missing_by_allzero = (expr_aligned.min(axis=1) == 0) & (expr_aligned.max(axis=1) == 0)
    missing_gene_rows = expr_aligned.index[missing_by_nan | missing_by_allzero]
    rows_present = ~(missing_by_nan | missing_by_allzero)
    
    n_present_true = rows_present.sum()
    cohort_coverage_pct = 100 * n_present_true / len(model_genes)
    
    print(f"\nGene coverage: {n_present_true}/{len(model_genes)} ({cohort_coverage_pct:.1f}%)")
    
    # NEUTRAL IMPUTATION for missing genes
    if hasattr(scaler, 'mean_'):
        training_means = pd.Series(scaler.mean_, index=model_genes)
        
        if len(missing_gene_rows) > 0:
            print(f"  Imputing {len(missing_gene_rows)} missing genes with training means")
            fill_block = np.repeat(
                training_means.loc[missing_gene_rows].values.reshape(-1, 1),
                expr_aligned.shape[1],
                axis=1
            )
            expr_aligned.loc[missing_gene_rows] = fill_block
        
        # Fill sporadic NaNs
        if expr_aligned.isna().any().any():
            expr_aligned = expr_aligned.apply(
                lambda row: row.fillna(training_means[row.name]), axis=1
            )
    else:
        expr_aligned = expr_aligned.fillna(0.0)
    
    # ====================================================================
    # CRITICAL FIX: RAW-SPACE ALIGNMENT BEFORE SCALING
    # ====================================================================
    alignment_applied = False
    global_shift_value = 0.0
    
    if global_shift_correct or align_gene_medians:
        print("\n>>> APPLYING RAW-SPACE ALIGNMENT <<<")
        alignment_applied = True
        
        # Training means in RAW log2 space
        train_mean = pd.Series(scaler.mean_, index=model_genes)
        
        # Use ONLY present genes for shift estimation
        cohort_median = expr_aligned.loc[rows_present].median(axis=1)
        
        # Option A: Robust global shift correction
        if global_shift_correct:
            raw_delta = (cohort_median - train_mean.loc[cohort_median.index]).dropna()
            if len(raw_delta) > 0:
                # Robust estimation: trim outliers
                q5, q50, q95 = np.quantile(raw_delta.values, [0.05, 0.5, 0.95])
                print(f"  Delta quantiles: 5%={q5:.2f}, 50%={q50:.2f}, 95%={q95:.2f}")
                
                trim = raw_delta[(raw_delta > q5) & (raw_delta < q95)]
                s = float(np.median(trim.values))
                global_shift_value = s
                print(f"  Global shift (trimmed median): {s:.3f} log2 units")
                
                # Apply ONLY to present genes
                print(f"  Subtracting {s:.3f} from present genes...")
                expr_aligned.loc[rows_present] = expr_aligned.loc[rows_present] - s
            
        # Option B: Per-gene median alignment
        if align_gene_medians:
            per_gene_shift = (cohort_median - train_mean.loc[cohort_median.index])
            per_gene_shift = per_gene_shift.reindex(expr_aligned.index).fillna(0.0)
            n_adj = int((per_gene_shift != 0).sum())
            print(f"  Aligning medians for {n_adj} present genes")
            expr_aligned.loc[rows_present] = expr_aligned.loc[rows_present].sub(
                per_gene_shift[rows_present], axis=0
            )
        
        # RE-NEUTRALIZE imputed genes AFTER alignment
        if len(missing_gene_rows) > 0:
            expr_aligned.loc[missing_gene_rows] = np.repeat(
                training_means.loc[missing_gene_rows].values.reshape(-1, 1),
                expr_aligned.shape[1],
                axis=1
            )
        
        print("  ✓ RAW alignment complete (imputed genes kept neutral)")
    
    # Optional: Gentle variance matching
    if match_variance and rows_present.sum() > 0:
        print("\n>>> APPLYING VARIANCE MATCHING <<<")
        if hasattr(scaler, "scale_"):
            train_std = pd.Series(scaler.scale_, index=model_genes)
        elif hasattr(scaler, "var_"):
            train_std = pd.Series(np.sqrt(scaler.var_), index=model_genes)
        else:
            train_std = None
        
        if train_std is not None:
            cohort_std = expr_aligned.loc[rows_present].std(axis=1, ddof=0).replace(0, np.nan)
            ratio = (train_std / cohort_std).reindex(expr_aligned.index)
            ratio = ratio.clip(lower=0.5, upper=2.0).fillna(1.0)  # Gentle caps
            
            # Center, scale, recenter
            current_median = expr_aligned.loc[rows_present].median(axis=1)
            expr_aligned.loc[rows_present] = (
                expr_aligned.loc[rows_present].sub(current_median, axis=0)
                .mul(ratio.loc[rows_present], axis=0)
                .add(current_median, axis=0)
            )
            print("  ✓ Applied gentle variance matching (0.5-2.0× cap)")
    
    # BATCH PREDICTION
    print(f"\nProcessing {expr_aligned.shape[1]} samples...")
    
    X = expr_aligned.T.values.astype('float32')
    
    # Apply preprocessing
    print("  Applying scaler and PCA...")
    X_scaled = scaler.transform(X)
    
    # Check z-score distribution AFTER alignment
    frac_hi = (X_scaled > 2).mean(axis=1)
    frac_lo = (X_scaled < -2).mean(axis=1)
    print(f"  Post-alignment z-scores: {frac_hi.mean():.1%} genes with z>+2, {frac_lo.mean():.1%} with z<-2")
    
    if frac_hi.mean() < 0.15 and frac_lo.mean() < 0.15:
        print("    ✓ Dataset shift corrected successfully!")
    else:
        print("    ⚠ Still seeing some shift - may need stronger alignment")
    
    X_pca = pca.transform(X_scaled).astype('float32')
    
    # Predict
    print("  Running inference...")
    probs = model.predict(X_pca, batch_size=256, verbose=0).flatten()
    
    # Calculate logits
    eps = 1e-7
    probs_clipped = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    
    # PC correlation analysis
    pc_logit_corr = pd.Series(
        np.corrcoef(X_pca.T, logits)[range(X_pca.shape[1]), X_pca.shape[1]],
        index=[f"PC{i+1}" for i in range(X_pca.shape[1])]
    ).sort_values(key=np.abs, ascending=False)
    
    print("\nTop PCs driving the logit:")
    print(pc_logit_corr.head(10).to_string())
    
    # Top genes for strongest PC
    if hasattr(pca, "components_"):
        top_pc = int(pc_logit_corr.index[0].split("PC")[1]) - 1
        comp = pca.components_[top_pc]
        top_idx = np.argsort(np.abs(comp))[-15:][::-1]
        print(f"\nTop genes for {pc_logit_corr.index[0]} (abs loadings):")
        for g, w in [(model_genes[i], comp[i]) for i in top_idx[:10]]:
            print(f"  {g:15s} {w:+.4f}")
    
    # Build results
    per_sample_coverage = pd.Series(cohort_coverage_pct, index=expr_aligned.columns)
    
    df_results = pd.DataFrame({
        'sample_id': expr_aligned.columns,
        'persister_probability': probs,
        'decision_logit': logits,
        'prediction': np.where(probs >= threshold, 'Persister', 'Non-Persister'),
        'threshold_used': threshold,
        'gene_coverage_pct': per_sample_coverage.values
    })
    
    # Add stratification
    print("\nAdding stratification...")
    scores = logits
    jitter = np.random.RandomState(42).randn(len(scores)) * 1e-6
    scores_jittered = scores + jitter
    
    # Deciles
    try:
        df_results["decile"] = pd.qcut(
            scores_jittered, q=10,
            labels=[f"D{i}" for i in range(1, 11)],
            duplicates="drop"
        )
        print(f"  Created {df_results['decile'].nunique()} deciles")
    except:
        df_results["decile"] = None
    
    # Tertiles
    try:
        df_results["tertile"] = pd.qcut(
            scores_jittered, q=[0, 1/3, 2/3, 1],
            labels=["Low", "Mid", "High"],
            duplicates="drop"
        )
        print(f"  Created {df_results['tertile'].nunique()} tertiles")
    except:
        df_results["tertile"] = pd.cut(scores_jittered, bins=2, labels=["Low", "High"])
    
    # Relative classification (top 30%)
    top_30_cutoff = np.percentile(scores_jittered, 70)
    df_results["top_30_percent"] = scores >= top_30_cutoff
    
    # Save results
    df_results.to_csv(output_path, index=False)
    
    # Save audit log
    audit_log = {
        "gene_fingerprint": gene_fingerprint,
        "global_shift_applied": global_shift_correct,
        "global_shift_value": float(global_shift_value),
        "gene_median_alignment": align_gene_medians,
        "variance_matching": match_variance,
        "z_score_frac_high": float(frac_hi.mean()),
        "z_score_frac_low": float(frac_lo.mean()),
        "neutral_baseline": float(p0) if p0 else None
    }
    
    audit_path = output_path.replace('.csv', '_audit.json')
    with open(audit_path, 'w') as f:
        json.dump(audit_log, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(df_results)}")
    
    # Percentiles
    percentiles = [1, 25, 50, 75, 99]
    prob_pcts = np.percentile(probs, percentiles)
    
    print(f"\nProbability percentiles:")
    for p, v in zip(percentiles, prob_pcts):
        marker = " ← median" if p == 50 else ""
        print(f"  {p:3d}th: {v:.3f}{marker}")
    
    # Check if still saturated
    median_prob = prob_pcts[2]
    if median_prob > 0.95:
        print("\n⚠ Still saturated HIGH - try stronger alignment or use tertiles")
    elif median_prob < 0.05:
        print("\n⚠ Saturated LOW - may have overcorrected")
    else:
        print("\n✓ Distribution looks reasonable after alignment!")
    
    # Classification summary
    n_persisters = (df_results['prediction'] == 'Persister').sum()
    print(f"\nClassification (threshold={threshold:.3f}):")
    print(f"  Persisters: {n_persisters}/{len(df_results)} ({100*n_persisters/len(df_results):.1f}%)")
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"✓ Audit log saved to: {audit_path}")
    
    if alignment_applied:
        print("✓ RAW-space alignment was applied")
    else:
        print("⚠ No alignment applied - consider using --global-shift-correct")
    
    return df_results

def main():
    parser = argparse.ArgumentParser(
        description='Bulk RNA-seq persister inference with complete fixes')
    
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--input-csv', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--use-reduced', action='store_true')
    parser.add_argument('--use-full', action='store_true')
    parser.add_argument('--threshold', type=float, default=None)
    
    # RAW space alignment options
    parser.add_argument('--global-shift-correct', action='store_true',
                       help='Estimate robust global shift and subtract')
    parser.add_argument('--align-gene-medians', action='store_true',
                       help='Align per-gene medians to training')
    parser.add_argument('--match-variance', action='store_true',
                       help='Gently match per-gene variance')
    
    args = parser.parse_args()
    
    use_reduced = not args.use_full
    
    try:
        results = run_bulk_inference(
            model_dir=args.model_dir,
            input_csv=args.input_csv,
            output_path=args.output,
            use_reduced=use_reduced,
            threshold_override=args.threshold,
            global_shift_correct=args.global_shift_correct,
            align_gene_medians=args.align_gene_medians,
            match_variance=args.match_variance
        )
        
        print("\n" + "="*60)
        print("INFERENCE COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

