#!/usr/bin/env python3
"""
Prepare TCGA-LAML data for persister analysis
"""

import pandas as pd
import numpy as np
import os

# Configuration
GENE_LIST = "/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt"
OUT_DIR = "/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/bulk_TCGA/processed"

# Find TCGA file
TCGA_PATHS = [
    "/scratch/project_2010751/Public_Datasets/TCGA_LAML/TCGA_LAML_HiSeqV2.tsv",
    "/scratch/project_2010751/Public_Datasets/TCGA_LAML/HiSeqV2.gz",
    "/scratch/project_2010751/Public_Datasets/TCGA_LAML/tcga_laml_rnaseq_fpkm-uq.tsv",
]

os.makedirs(OUT_DIR, exist_ok=True)

tcga_file = None
for path in TCGA_PATHS:
    if os.path.exists(path):
        tcga_file = path
        break

if tcga_file is None:
    print("ERROR: No TCGA file found")
    exit(1)

print(f"Loading from: {tcga_file}")

# Load data
if tcga_file.endswith('.gz'):
    expr = pd.read_csv(tcga_file, sep="\t", index_col=0, compression='gzip')
else:
    expr = pd.read_csv(tcga_file, sep="\t", index_col=0)

# Orient correctly
if expr.shape[1] > expr.shape[0]:
    expr = expr.T

# Clean gene names
expr.index = expr.index.astype(str).str.upper()
if '|' in str(expr.index[0]):
    expr.index = [str(x).split('|')[0] for x in expr.index]
expr.index = expr.index.str.replace(r'\.\d+$', '', regex=True)

# Handle duplicates
if expr.index.duplicated().any():
    expr = expr.groupby(level=0).mean()

# Check scale and transform
expr = expr.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
data_max = expr.max().max()

if data_max > 100:
    print("Applying log2(x+1) transformation")
    expr = np.log2(expr.clip(lower=0) + 1)
else:
    print("Data appears already log-transformed")

# Load gene list
genes_1k = pd.read_csv(GENE_LIST, header=None).iloc[:,0].astype(str).str.upper()
genes_1k = genes_1k.drop_duplicates().tolist()

# Find available genes
genes_available = [g for g in genes_1k if g in expr.index]
print(f"Found {len(genes_available)}/{len(genes_1k)} genes")

# Subset and add missing as zeros
expr_subset = expr.loc[genes_available]
missing_genes = [g for g in genes_1k if g not in genes_available]

if missing_genes:
    
    expr_subset = expr_subset

expr_1k = expr_subset.reindex(genes_1k).fillna(0)

# Save log2 only for model
expr_1k.to_csv(f"{OUT_DIR}/expression_for_model.csv")
print(f"Saved expression_for_model.csv: {expr_1k.shape}")

# Create metadata
meta_out = pd.DataFrame({
    "sample_id": expr_1k.columns.astype(str),
    "cohort": "TCGA-LAML"
})

meta_out.to_csv(f"{OUT_DIR}/metadata.csv", index=False)
print(f"Saved metadata: {meta_out.shape}")
