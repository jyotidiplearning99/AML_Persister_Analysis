#!/usr/bin/env python3
"""
Prepare GSE74246 for persister prediction
"""

import numpy as np
import pandas as pd
import gzip
from pathlib import Path

# Configuration
MODEL_GENES = "/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt"
INPUT_FILE = "/scratch/project_2010751/Public_Datasets/GEO_Datasets/GSE74246_RNAseq_All_Counts.txt.gz"
OUT_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/GSE74246")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading GSE74246...")
with gzip.open(INPUT_FILE, 'rt') as f:
    expr = pd.read_csv(f, sep='\t', index_col=0)

print(f"Loaded: {expr.shape}")

# Apply log2(x+1) transformation
print("Applying log2(x+1) transformation...")
expr = np.log2(expr + 1)

# Clean gene names
expr.index = [str(g).upper() for g in expr.index]

# Load model genes
with open(MODEL_GENES) as f:
    model_genes = [line.strip().upper() for line in f if line.strip()]

print(f"Model expects {len(model_genes)} genes")

# Check coverage
present = [g for g in model_genes if g in expr.index]
print(f"Found {len(present)}/{len(model_genes)} genes ({100*len(present)/len(model_genes):.1f}%)")

# Align to model genes
expr_aligned = expr.reindex(model_genes).fillna(0)

# Save for model
expr_aligned.to_csv(OUT_DIR / "expression_for_model.csv")
print(f"Saved: {expr_aligned.shape}")
