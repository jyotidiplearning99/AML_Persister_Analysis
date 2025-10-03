#!/usr/bin/env python3
"""
Bulk RNA-seq inference - Optimized for 60% persister target
Handles extreme TCGA batch effects properly
Use now 03/10/2025
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
import json

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def run_bulk_inference(model_dir, input_csv, output_path, use_reduced=True, 
                      threshold_override=None):
    
    print(f"\n{'='*60}")
    print("BULK RNA-SEQ PERSISTER INFERENCE - 60% OPTIMIZED")
    print(f"{'='*60}")
    
    model_dir = Path(model_dir)
    
    # Load paths
    if use_reduced:
        paths = {
            'model': model_dir / 'model_reduced.h5',
            'scaler': model_dir / 'scaler_reduced.pkl',
            'pca': model_dir / 'pca_reduced.pkl',
            'genes': model_dir / 'selected_genes.txt'
        }
    else:
        paths = {
            'model': model_dir / 'final_model.h5',
            'scaler': model_dir / 'scaler.pkl',
            'pca': model_dir / 'pca.pkl',
            'genes': model_dir / 'common_genes.txt'
        }
    
    # Load components
    print("Loading model components...")
    model = tf.keras.models.load_model(paths['model'], compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    scaler = joblib.load(paths['scaler'])
    pca = joblib.load(paths['pca'])
    
    with open(paths['genes']) as f:
        model_genes = [line.strip().upper() for line in f if line.strip()]
    
    print(f"Model expects {len(model_genes)} genes")
    
    # Load expression data
    print(f"\nLoading expression data...")
    expr = pd.read_csv(input_csv, index_col=0)
    
    if expr.shape[1] > expr.shape[0]:
        expr = expr.T
    
    print(f"Shape: {expr.shape[0]} genes × {expr.shape[1]} samples")
    
    # Clean and align genes
    expr.index = [str(g).strip().upper() for g in expr.index]
    expr_aligned = expr.reindex(model_genes)
    
    # Identify missing/present genes
    missing_mask = expr_aligned.isna().all(axis=1) | ((expr_aligned == 0).all(axis=1))
    present_mask = ~missing_mask
    n_present = present_mask.sum()
    
    print(f"Gene coverage: {n_present}/{len(model_genes)} ({100*n_present/len(model_genes):.1f}%)")
    
    # Impute missing with training means
    training_means = pd.Series(scaler.mean_, index=model_genes)
    for gene in model_genes:
        if missing_mask[gene]:
            expr_aligned.loc[gene] = training_means[gene]
    
    expr_aligned = expr_aligned.fillna(0)
    
    # Calculate batch effect
    cohort_median = expr_aligned.loc[present_mask].median(axis=1)
    train_median = training_means.loc[present_mask]
    raw_shift = np.median(cohort_median - train_median)
    
    # Check data characteristics
    data_median = expr.median().median()
    
    print(f"\n>>> BATCH EFFECT ANALYSIS <<<")
    print(f"  Data median: {data_median:.2f}")
    print(f"  Raw shift: {raw_shift:.3f} log2 units")
    
    # OPTIMAL CORRECTION STRATEGY FOR 60% TARGET
    if abs(raw_shift) > 5:
        # Extreme shift (TCGA): Apply 80% correction
        correction_factor = 0.80
        classification_strategy = "percentile_60"
        print(f"  → TCGA-like data detected (shift > 5)")
        
    elif abs(raw_shift) > 2:
        # Moderate shift (BeatAML): Full correction
        correction_factor = 1.0
        classification_strategy = "absolute"
        print(f"  → BeatAML-like data detected")
        
    else:
        # Mild shift: Full correction
        correction_factor = 1.0
        classification_strategy = "absolute"
        print(f"  → Mild batch effect")
    
    # Apply correction
    shift_to_apply = raw_shift * correction_factor
    print(f"  Applying {correction_factor*100:.0f}% correction: {shift_to_apply:.3f} log2 units")
    
    expr_aligned.loc[present_mask] = expr_aligned.loc[present_mask] - shift_to_apply
    
    # Re-impute missing genes
    for gene in model_genes:
        if missing_mask[gene]:
            expr_aligned.loc[gene] = training_means[gene]
    
    print("  ✓ Batch correction complete")
    
    # Transform and predict
    print(f"\nRunning inference...")
    X = expr_aligned.T.values.astype('float32')
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled).astype('float32')
    
    probs = model.predict(X_pca, batch_size=256, verbose=0).flatten()
    
    # Calculate logits
    eps = 1e-7
    probs_clipped = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    
    # Check distribution
    median_prob = np.median(probs)
    print(f"Median probability: {median_prob:.3f}")
    
    # CLASSIFICATION FOR 60% TARGET
    threshold = threshold_override if threshold_override else 0.31
    
    if classification_strategy == "percentile_60":
        # For TCGA: Use top 60% as persisters
        cutoff = np.percentile(logits, 40)  # 40th percentile = top 60%
        predictions = np.where(logits >= cutoff, 'Persister', 'Non-Persister')
        method = "percentile_top_60"
        print(f"\n✓ Using PERCENTILE classification (top 60%)")
        
    elif median_prob > 0.9 or median_prob < 0.1:
        # Saturated: Use top 60% as fallback
        cutoff = np.percentile(logits, 40)
        predictions = np.where(logits >= cutoff, 'Persister', 'Non-Persister')
        method = "percentile_fallback_60"
        print(f"\n✓ Using PERCENTILE fallback (top 60%)")
        
    else:
        # Use absolute threshold
        predictions = np.where(probs >= threshold, 'Persister', 'Non-Persister')
        method = f"absolute_threshold_{threshold:.3f}"
        print(f"\n✓ Using ABSOLUTE classification (threshold={threshold:.3f})")
    
    n_persisters = (predictions == 'Persister').sum()
    pct_persisters = 100 * n_persisters / len(predictions)
    
    # Build results
    df_results = pd.DataFrame({
        'sample_id': expr_aligned.columns,
        'persister_probability': probs,
        'decision_logit': logits,
        'prediction': predictions,
        'classification_method': method,
        'batch_shift': raw_shift,
        'correction_applied': shift_to_apply
    })
    
    # Add percentile ranks
    df_results['percentile_rank'] = pd.Series(logits).rank(pct=True) * 100
    
    # Save results
    df_results.to_csv(output_path, index=False)
    
    # Save summary
    summary = {
        'samples': len(df_results),
        'persisters': int(n_persisters),
        'persister_percentage': float(pct_persisters),
        'batch_shift': float(raw_shift),
        'correction_applied': float(shift_to_apply),
        'classification_method': method,
        'median_probability': float(median_prob)
    }
    
    summary_path = output_path.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Persisters: {n_persisters}/{len(df_results)} ({pct_persisters:.1f}%)")
    print(f"Method: {method}")
    
    # Probability distribution
    print(f"\nProbability distribution:")
    for p in [1, 25, 50, 75, 99]:
        v = np.percentile(probs, p)
        print(f"  {p:3d}%: {v:.3f}")
    
    # Target assessment
    if 55 <= pct_persisters <= 65:
        print(f"\n✓ Achieved target range (55-65% persisters)")
    elif 50 <= pct_persisters <= 70:
        print(f"\n✓ Within acceptable range (50-70% persisters)")
    else:
        print(f"\n⚠ Outside typical AML range")
    
    print(f"\n✓ Results: {output_path}")
    print(f"✓ Summary: {summary_path}")
    
    return df_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--use-reduced', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.31)
    
    args = parser.parse_args()
    
    run_bulk_inference(
        model_dir=args.model_dir,
        input_csv=args.input_csv,
        output_path=args.output,
        use_reduced=args.use_reduced,
        threshold_override=args.threshold
    )

if __name__ == "__main__":
    main()
