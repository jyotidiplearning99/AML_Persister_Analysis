#!/usr/bin/env python3
"""
Process MV4-11 scRNA-seq data (GSE228326 Control sample)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
import joblib
from scipy.io import mmread
import gzip

# Config
MODEL_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled")
SCRNA_DIR = Path("/scratch/project_2010376/JDs_Project/scrna_data")
OUTPUT_DIR = Path("/scratch/project_2010376/JDs_Project/scrna_persister_scores")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("="*80)
print("MV4-11 scRNA-seq INFERENCE")
print("="*80)

# Load model
print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_DIR / "final_model.h5", compile=False)

with open(MODEL_DIR / "selected_genes.txt") as f:
    TRAINING_GENES = [line.strip().upper() for line in f if line.strip()]

scaler = joblib.load(MODEL_DIR / "scaler.pkl")
pca = joblib.load(MODEL_DIR / "pca.pkl")

print(f"✓ Model: {len(TRAINING_GENES)} genes → {pca.n_components_} components")

# Helper functions
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

# Load MTX files
print("\nLoading MTX data...")

matrix_file = SCRNA_DIR / "GSM7118799_DF68555_Control_matrix.mtx.gz"
features_file = SCRNA_DIR / "GSM7118799_DF68555_Control_features.tsv.gz"
barcodes_file = SCRNA_DIR / "GSM7118799_DF68555_Control_barcodes.tsv.gz"

# Verify files exist
if not matrix_file.exists():
    print(f"✗ Matrix file not found: {matrix_file}")
    exit(1)

if not features_file.exists():
    print(f"✗ Features file not found: {features_file}")
    exit(1)

if not barcodes_file.exists():
    print(f"✗ Barcodes file not found: {barcodes_file}")
    exit(1)

# Load matrix
print("  Loading matrix...")
with gzip.open(matrix_file, 'rt') as f:
    X = mmread(f).T.tocsr()

# Load genes
print("  Loading features...")
features_df = pd.read_csv(features_file, sep='\t', header=None)
genes = features_df[1].tolist() if features_df.shape[1] >= 2 else features_df[0].tolist()

# Load barcodes
print("  Loading barcodes...")
with gzip.open(barcodes_file, 'rt') as f:
    barcodes = [line.strip() for line in f]

print(f"✓ Loaded: {X.shape[0]} cells × {X.shape[1]} genes")

# Convert to dense
X = X.toarray().astype(np.float32)

# Subsample if too large
max_cells = 10000
if X.shape[0] > max_cells:
    print(f"  Subsampling to {max_cells} cells...")
    indices = np.random.choice(X.shape[0], max_cells, replace=False)
    X = X[indices]
    barcodes = [barcodes[i] for i in indices]
    print(f"  Using {len(barcodes)} cells")

# Align genes
print("\nAligning genes...")
X_aligned, found = align_genes(X, genes, TRAINING_GENES)
match_pct = 100.0 * found / len(TRAINING_GENES)
print(f"✓ Matched: {found}/{len(TRAINING_GENES)} ({match_pct:.1f}%)")

# Preprocess
print("\nPreprocessing...")
X_proc = scrna_cpm_log1p(X_aligned)
print(f"  After CPM+log1p: shape={X_proc.shape}")

X_proc = scaler.transform(X_proc)
print(f"  After scaling: shape={X_proc.shape}")

X_proc = pca.transform(X_proc).astype(np.float32)
print(f"  After PCA: shape={X_proc.shape}")

# Predict
print("\nRunning inference...")
probs = model.predict(X_proc, batch_size=512, verbose=1).ravel()
preds = (probs >= 0.5).astype(int)

# Results
pers_count = int(preds.sum())
pers_pct = 100.0 * pers_count / len(preds)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nMV4-11 (GSE228326 Control):")
print(f"  Total cells: {len(preds):,}")
print(f"  Persister: {pers_count:,} ({pers_pct:.1f}%)")
print(f"  Non-persister: {len(preds) - pers_count:,} ({100 - pers_pct:.1f}%)")
print(f"  Mean probability: {np.mean(probs):.3f}")
print(f"  Genes matched: {found}/{len(TRAINING_GENES)} ({match_pct:.1f}%)")

# Save
result = pd.DataFrame({
    'cell_line': ['MV4-11'],
    'persister_pct': [pers_pct],
    'n_cells': [len(preds)],
    'n_persister': [pers_count],
    'mean_prob': [float(np.mean(probs))],
    'genes_matched': [found]
})

result.to_csv(OUTPUT_DIR / 'mv411_persister_score.csv', index=False)
print(f"\n✓ Saved: {OUTPUT_DIR / 'mv411_persister_score.csv'}")

print("\n" + "="*80)
print("This is REAL scRNA-seq data - persister % should be realistic (30-70%)")
print("Not 100% like the bulk RNA-seq results!")
print("="*80)
