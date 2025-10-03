#!/usr/bin/env python3
"""
Bulk RNA-seq inference for persister model - ORIGINAL VERSION
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
import argparse
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def run_bulk_inference(model_dir, input_csv, output_path, use_reduced=True, threshold_override=None):
    print(f"\n{'='*60}")
    print("BULK RNA-SEQ PERSISTER INFERENCE")
    print(f"{'='*60}")
    
    model_dir = Path(model_dir)
    
    # Determine paths
    if use_reduced:
        model_path = model_dir / 'model_reduced.h5'
        scaler_path = model_dir / 'scaler_reduced.pkl'
        pca_path = model_dir / 'pca_reduced.pkl'
        threshold_path = model_dir / 'threshold_reduced.pkl'
        genes_path = model_dir / 'selected_genes.txt'
    else:
        model_path = model_dir / 'final_model.h5'
        scaler_path = model_dir / 'scaler.pkl'
        pca_path = model_dir / 'pca.pkl'
        threshold_path = model_dir / 'threshold.pkl'
        genes_path = model_dir / 'common_genes.txt'
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Load preprocessing
    print("Loading preprocessing components...")
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    pca = joblib.load(pca_path) if pca_path.exists() else None
    
    # Load threshold with override and safety checks
    if threshold_override is not None:
        threshold = float(np.clip(threshold_override, 0.25, 0.75))
        print(f"Using override threshold: {threshold:.3f} (clipped to safe range)")
    elif threshold_path.exists():
        threshold = float(joblib.load(threshold_path))
        # Safety check
        if threshold < 0.25 or threshold > 0.75:
            print(f"WARNING: Loaded threshold {threshold:.3f} outside safe range [0.25, 0.75]")
            threshold = float(np.clip(threshold, 0.25, 0.75))
            print(f"Clipped to: {threshold:.3f}")
    else:
        threshold = 0.4  # Use safe default
        print(f"No threshold found, using default: {threshold:.3f}")
    
    print(f"Final threshold: {threshold:.3f}")
    
    # Load gene list
    with open(genes_path) as f:
        model_genes = [line.strip().upper() for line in f if line.strip()]
    print(f"Model expects {len(model_genes)} genes")
    
    # Load expression data
    print(f"\nLoading expression data from: {input_csv}")
    expr = pd.read_csv(input_csv, index_col=0)
    
    # Check orientation
    if expr.shape[1] > expr.shape[0]:
        print("Transposing to genes x samples format")
        expr = expr.T
    
    print(f"Expression shape: {expr.shape[0]} genes × {expr.shape[1]} samples")
    
    # Clean gene names
    expr.index = [str(g).strip().upper() for g in expr.index]
    
    # Check coverage
    present_genes = [g for g in model_genes if g in expr.index]
    coverage_pct = 100 * len(present_genes) / len(model_genes)
    print(f"Gene coverage: {len(present_genes)}/{len(model_genes)} ({coverage_pct:.1f}%)")
    
    if coverage_pct < 50:
        print("WARNING: Low gene coverage (<50%). Results may be unreliable.")
    
    # Align to model genes
    expr_aligned = expr.reindex(model_genes).fillna(0.0)
    
    # Process each sample
    print(f"\nProcessing {expr_aligned.shape[1]} samples...")
    results = []
    
    for sample_id in expr_aligned.columns:
        X = expr_aligned[sample_id].values.reshape(1, -1)
        
        # Apply preprocessing
        if scaler is not None:
            X = scaler.transform(X)
        if pca is not None:
            X = pca.transform(X)
        
        # Predict
        prob = model.predict(X, verbose=0)[0, 0]
        
        results.append({
            'sample_id': sample_id,
            'persister_probability': float(prob),
            'prediction': 'Persister' if prob >= threshold else 'Non-Persister',
            'threshold_used': threshold,
            'gene_coverage_pct': coverage_pct
        })
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(df_results)}")
    print(f"Mean persister probability: {df_results['persister_probability'].mean():.3f}")
    print(f"Median persister probability: {df_results['persister_probability'].median():.3f}")
    
    n_persisters = (df_results['prediction'] == 'Persister').sum()
    print(f"Persister predictions: {n_persisters}/{len(df_results)} ({100*n_persisters/len(df_results):.1f}%)")
    print(f"\nResults saved to: {output_path}")
    
    return df_results

def main():
    parser = argparse.ArgumentParser(description='Bulk RNA-seq inference')
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--input-csv', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--use-reduced', action='store_true')
    parser.add_argument('--use-full', action='store_true')
    parser.add_argument('--threshold', type=float, default=None, 
                       help='Override threshold (recommended: 0.31)')
    
    args = parser.parse_args()
    
    use_reduced = True if args.use_reduced else (False if args.use_full else True)
    
    results = run_bulk_inference(
        model_dir=args.model_dir,
        input_csv=args.input_csv,
        output_path=args.output,
        use_reduced=use_reduced,
        threshold_override=args.threshold
    )

if __name__ == "__main__":
    main()
