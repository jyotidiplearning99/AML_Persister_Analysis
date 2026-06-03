#!/usr/bin/env python3

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse
from pathlib import Path

base = Path("GSE200997_crc")

matrix_path = base / "GSE200997_GEO_processed_CRC_10X_raw_UMI_count_matrix.csv.gz"
ann_path = base / "GSE200997_GEO_processed_CRC_10X_cell_annotation.csv.gz"
out_path = Path("GSE200997.h5ad")

print("Reading annotation...")
obs = pd.read_csv(ann_path)
print("Annotation shape:", obs.shape)
print("Annotation columns:", obs.columns.tolist())

print("Reading count matrix...")
mat = pd.read_csv(matrix_path)
print("Matrix shape:", mat.shape)
print("First columns:", mat.columns[:5].tolist())

# Detect cell ID column
cell_col = mat.columns[0]
print("Using matrix cell ID column:", cell_col)

mat[cell_col] = mat[cell_col].astype(str)
mat = mat.set_index(cell_col)

# Detect annotation cell ID column
possible_cell_cols = ["Unnamed: 0", "cell", "Cell", "cell_id", "Cell_ID", "barcode", "Barcode"]
ann_cell_col = None

for c in possible_cell_cols:
    if c in obs.columns:
        ann_cell_col = c
        break

if ann_cell_col is None:
    ann_cell_col = obs.columns[0]

print("Using annotation cell ID column:", ann_cell_col)

obs[ann_cell_col] = obs[ann_cell_col].astype(str)
obs = obs.set_index(ann_cell_col)

# Align common cells
common = mat.index.intersection(obs.index)
print("Common cells:", len(common))

if len(common) == 0:
    raise RuntimeError("No common cell IDs between matrix and annotation.")

mat = mat.loc[common]
obs = obs.loc[common]

# Convert counts to sparse matrix
print("Converting matrix to numeric...")
mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0)

print("Converting to sparse CSR...")
X = sparse.csr_matrix(mat.values.astype(np.float32))

var = pd.DataFrame(index=mat.columns.astype(str))

adata = ad.AnnData(X=X, obs=obs.copy(), var=var)

print("Final AnnData:", adata)

print("Writing:", out_path)
adata.write_h5ad(out_path, compression="gzip")

print("Done.")
print("Saved:", out_path)