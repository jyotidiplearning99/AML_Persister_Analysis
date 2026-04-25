#!/usr/bin/env python3
"""
Quick test: Run inference on MOLM-13 and MV4-11
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
import scanpy as sc
import joblib
from scipy import sparse

# Config
MODEL_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled")
SCRNA_DIR = Path("/scratch/project_2010376/JDs_Project/scrna_data")
OUTPUT_DIR = Path("/scratch/project_2010376/JDs_Project/scrna_persister_scores")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load model
model = tf.keras.models.load_model(MODEL_DIR / "final_model.h5", compile=False)

with open(MODEL_DIR / "selected_genes.txt") as f:
    TRAINING_GENES = [line.strip().upper() for line in f if line.strip()]

scaler = joblib.load(MODEL_DIR / "scaler.pkl")
pca = joblib.load(MODEL_DIR / "pca.pkl")

print(f"✓ Model loaded: {len(TRAINING_GENES)} genes → {pca.n_components_} PCA components")

# Functions
def clean_gene_names(names):
    return [str(g).strip().upper().rsplit(".", 1)[0] for g in names]

def scrna_cpm_log1p(X):
    X = np.asarray(X, dtype=np.float32)
    X = np.maximum(X, 0.0)
    lib = X.sum(axis=1, keepdims=True)
    np.maximum(lib, 1.0, out=lib)
    X = (X / lib) * 1e4
    return np.log1p(X).astype(np.float32)

def align_genes(X, genes, training_genes):
    genes_h = clean_gene_names(genes)
    gene2idx = {g: i for i, g in enumerate(genes_h)}
    out = np.zeros((X.shape[0], len(training_genes)), dtype=np.float32)
    found = 0
    for j, g in enumerate(training_genes):
        i = gene2idx.get(g)
        if i is not None:
            out[:, j] = X[:, i]
            found += 1
    return out, found

def predict_h5(h5_file, cell_line_name, max_cells=10000):
    """Process H5 file"""
    print(f"\nProcessing {cell_line_name} from {h5_file.name}...")
    
    adata = sc.read_10x_h5(h5_file)
    
    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=SEED)
    
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    X = X.astype(np.float32)
    genes = [str(g) for g in adata.var_names]
    
    # Align
    X_aligned, found = align_genes(X, genes, TRAINING_GENES)
    print(f"  Genes: {found}/{len(TRAINING_GENES)} ({100*found/len(TRAINING_GENES):.1f}%)")
    
    # Preprocess
    X_processed = scrna_cpm_log1p(X_aligned)
    X_processed = scaler.transform(X_processed)
    X_processed = pca.transform(X_processed).astype(np.float32)
    
    # Predict
    probs = model.predict(X_processed, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    
    persister_pct = 100.0 * preds.sum() / len(preds)
    
    print(f"  ✓ {cell_line_name}: {persister_pct:.1f}% persister ({preds.sum()}/{len(preds)} cells)")
    
    return {
        'cell_line': cell_line_name,
        'dataset': h5_file.stem,
        'n_cells': len(preds),
        'n_persister': int(preds.sum()),
        'persister_pct': persister_pct,
        'mean_prob': float(np.mean(probs)),
        'genes_matched': found
    }

# Find and process H5 files
print("="*80)
print("BATCH scRNA-seq INFERENCE")
print("="*80)

results = []

# Look for H5 files in extracted directories
for cell_line in ['MOLM13', 'MV411', 'THP1']:
    h5_files = list(SCRNA_DIR.glob(f"**/{cell_line}*/*filtered*.h5"))
    
    if not h5_files:
        h5_files = list(SCRNA_DIR.glob(f"**/*{cell_line}*.h5"))
    
    if h5_files:
        result = predict_h5(h5_files[0], cell_line.replace('411', '4-11'))
        results.append(result)
    else:
        print(f"\n✗ {cell_line}: No H5 file found")

if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / 'scrna_persister_scores.csv', index=False)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    for _, row in df_results.iterrows():
        print(f"{row['cell_line']:15s}: {row['persister_pct']:6.1f}% ({row['n_cells']} cells)")
    
    print(f"\n✓ Saved: {OUTPUT_DIR / 'scrna_persister_scores.csv'}")
else:
    print("\n✗ No datasets processed")
