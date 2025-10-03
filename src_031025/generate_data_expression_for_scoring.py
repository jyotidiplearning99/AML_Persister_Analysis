#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create pseudo-bulk expression from single-cell data for module scoring
Fixed: Handle duplicate gene names after uppercase conversion
"""

import json
import gzip
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
import argparse

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------- utils ---------------------------

def _mmread_any(path: Path):
    """Read MatrixMarket (.mtx or .mtx.gz) with scipy.io.mmread."""
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return mmread(fh)
    return mmread(path)


def _read_table_any(path: Path, sep: str = "\t", header=None) -> pd.DataFrame:
    """Read TSV/CSV (optionally gzipped) into DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if str(path).endswith(".gz"):
        return pd.read_csv(path, sep=sep, header=header, compression="gzip")
    return pd.read_csv(path, sep=sep, header=header)


def _series_from_mean(mtx, axis=1) -> np.ndarray:
    """Mean along axis for dense or sparse matrix."""
    if sparse.issparse(mtx):
        return np.array(mtx.mean(axis=axis)).ravel()
    mtx = np.asarray(mtx)
    return mtx.mean(axis=axis)


def _handle_duplicate_genes(gene_names: np.ndarray, expr_values: np.ndarray, sample_name: str) -> pd.Series:
    """
    Handle duplicate gene names by aggregating (mean) expression values.
    """
    # Create DataFrame to handle duplicates
    df = pd.DataFrame({'gene': gene_names, 'expr': expr_values})
    
    # Check for duplicates
    n_dups = df['gene'].duplicated().sum()
    if n_dups > 0:
        logging.info(f"  [{sample_name}] Found {n_dups} duplicate genes after uppercase conversion, aggregating by mean")
        # Aggregate by mean for duplicate genes
        df = df.groupby('gene', as_index=False)['expr'].mean()
    
    # Return as Series
    return pd.Series(df['expr'].values, index=df['gene'].values, name=sample_name)


# ---------------------- loaders / aggregators ----------------------

def load_and_aggregate_10x(matrix_dir: Path, sample_name: str, max_cells: Optional[int] = None) -> pd.Series:
    """Load 10x filtered_feature_bc_matrix and produce mean expression per gene."""
    logging.info(f"[10x] {sample_name}: {matrix_dir}")

    # Matrix
    mtx_path = None
    for cand in ("matrix.mtx.gz", "matrix.mtx"):
        p = matrix_dir / cand
        if p.exists():
            mtx_path = p
            break
    if mtx_path is None:
        logging.error(f"[10x] Missing matrix file under {matrix_dir}")
        return pd.Series(dtype=float, name=sample_name)

    # Features
    features_path = None
    for cand in ("features.tsv.gz", "genes.tsv.gz", "features.tsv", "genes.tsv"):
        p = matrix_dir / cand
        if p.exists():
            features_path = p
            break
    if features_path is None:
        logging.error(f"[10x] Missing features/genes file under {matrix_dir}")
        return pd.Series(dtype=float, name=sample_name)

    # Read
    mtx = _mmread_any(mtx_path)
    features = _read_table_any(features_path, sep="\t", header=None)

    # Gene symbols preferred (col 1), otherwise IDs (col 0)
    if features.shape[1] >= 2:
        gene_names = features[1].astype(str).str.upper().values
    else:
        gene_names = features[0].astype(str).str.upper().values

    # Subsample cells if needed (columns are cells for MM format)
    if sparse.issparse(mtx):
        mtx = mtx.tocsr()
    else:
        mtx = np.asarray(mtx)

    n_cells = mtx.shape[1]
    if max_cells and n_cells > max_cells:
        rng = np.random.default_rng(42)
        keep_idx = rng.choice(n_cells, max_cells, replace=False)
        mtx = mtx[:, keep_idx]
        n_cells = mtx.shape[1]

    mean_expr = _series_from_mean(mtx, axis=1)
    
    # Handle duplicates
    result = _handle_duplicate_genes(gene_names, mean_expr, sample_name)
    
    logging.info(f"[10x] {sample_name}: {len(result)} unique genes × {n_cells} cells")
    return result


def load_and_aggregate_loose_triplet(matrix_file: Path, sample_name: str, max_cells: Optional[int] = None) -> pd.Series:
    """Load GSE120221-style loose triplet files and aggregate per gene."""
    logging.info(f"[LooseTriplet] {sample_name}: {matrix_file}")

    stem = matrix_file.name
    if "_matrix_" not in stem:
        logging.error(f"[LooseTriplet] Cannot parse filename: {matrix_file}")
        return pd.Series(dtype=float, name=sample_name)

    prefix, suffix = stem.split("_matrix_")
    suffix = suffix.replace(".mtx.gz", "").replace(".mtx", "")
    genes_file = matrix_file.parent / f"{prefix}_genes_{suffix}.tsv.gz"
    if not genes_file.exists():
        genes_file = matrix_file.parent / f"{prefix}_genes_{suffix}.tsv"

    if not genes_file.exists():
        logging.error(f"[LooseTriplet] Genes file not found: {genes_file}")
        return pd.Series(dtype=float, name=sample_name)

    mtx = _mmread_any(matrix_file)
    genes_df = _read_table_any(genes_file, sep="\t", header=None)

    if genes_df.shape[1] >= 2:
        gene_names = genes_df[1].astype(str).str.upper().values
    else:
        gene_names = genes_df[0].astype(str).str.upper().values

    if sparse.issparse(mtx):
        mtx = mtx.tocsr()
    else:
        mtx = np.asarray(mtx)

    n_cells = mtx.shape[1]
    if max_cells and n_cells > max_cells:
        rng = np.random.default_rng(42)
        keep_idx = rng.choice(n_cells, max_cells, replace=False)
        mtx = mtx[:, keep_idx]
        n_cells = mtx.shape[1]

    mean_expr = _series_from_mean(mtx, axis=1)
    
    # Handle duplicates
    result = _handle_duplicate_genes(gene_names, mean_expr, sample_name)
    
    logging.info(f"[LooseTriplet] {sample_name}: {len(result)} unique genes × {n_cells} cells")
    return result


def load_and_aggregate_dense_csv(csv_file: Path, sample_name: str, max_cells: Optional[int] = None) -> pd.Series:
    """Load dense CSV and aggregate per gene."""
    logging.info(f"[DenseCSV] {sample_name}: {csv_file}")

    df = pd.read_csv(csv_file, index_col=0)

    # Heuristic: if many rows with gene-like names, genes are rows
    if df.shape[0] > df.shape[1] and df.index.astype(str).str.match(r"^[A-Za-z][A-Za-z0-9_-]+").mean() > 0.8:
        # genes x cells
        if max_cells and df.shape[1] > max_cells:
            rng = np.random.default_rng(42)
            keep_cols = rng.choice(df.columns, max_cells, replace=False)
            df = df.loc[:, keep_cols]
        mean_expr = df.mean(axis=1).values
        gene_names = df.index.astype(str).str.upper().values
        n_cells = df.shape[1]
    else:
        # cells x genes
        if max_cells and df.shape[0] > max_cells:
            rng = np.random.default_rng(42)
            keep_rows = rng.choice(df.index, max_cells, replace=False)
            df = df.loc[keep_rows]
        mean_expr = df.mean(axis=0).values
        gene_names = df.columns.astype(str).str.upper().values
        n_cells = df.shape[0]

    # Handle duplicates
    result = _handle_duplicate_genes(gene_names, mean_expr, sample_name)
    
    logging.info(f"[DenseCSV] {sample_name}: {len(result)} unique genes × {n_cells} cells")
    return result


# ---------------------- main builder ----------------------

def create_bulk_expression_matrix(output_dir: Path, max_cells_per_sample: int = 20000):
    """
    Create pseudo-bulk expression matrix from available samples.
    Edit the sample lists below to point at your real data locations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_data = []
    sample_metadata = []

    # 1) AML (10x format)
    aml_samples = [
        {"path": Path("/scratch/project_2010376/scRNAseq/FH_5897_2/filtered_feature_bc_matrix"),
         "name": "FH_5897_2", "condition": "AML"},
        {"path": Path("/scratch/project_2010376/scRNAseq/FH_6333_2/filtered_feature_bc_matrix"),
         "name": "FH_6333_2", "condition": "AML"},
    ]

    aml_root = Path("/scratch/project_2010751/AML_scRNA_decrypted")
    if aml_root.exists():
        for sample_dir in aml_root.glob("*/filtered_feature_bc_matrix"):
            if sample_dir.is_dir():
                sample_name = sample_dir.parent.name
                if sample_name not in {"FH_5897_2", "FH_6333_2"}:
                    aml_samples.append({"path": sample_dir, "name": sample_name, "condition": "AML"})

    for s in aml_samples:
        if s["path"].exists():
            try:
                expr = load_and_aggregate_10x(s["path"], s["name"], max_cells_per_sample)
                if not expr.empty:
                    samples_data.append(expr)
                    sample_metadata.append({"sample_id": s["name"], "condition": s["condition"], "data_type": "10x"})
            except Exception as e:
                logging.warning(f"[10x] Failed for {s['name']}: {e}")
        else:
            logging.info(f"[10x] Skipping (not found): {s['path']}")

    # 2) Healthy (loose triplets, e.g., GSE120221)
    healthy_root = Path("/scratch/project_2010751/GSE120221_RAW")
    if healthy_root.exists():
        for matrix_file in sorted(healthy_root.glob("*_matrix_*.mtx.gz")):
            sample_name = matrix_file.name.split("_matrix_")[0]
            try:
                expr = load_and_aggregate_loose_triplet(matrix_file, sample_name, max_cells_per_sample)
                if not expr.empty:
                    samples_data.append(expr)
                    sample_metadata.append({"sample_id": sample_name, "condition": "Healthy", "data_type": "loose_triplet"})
            except Exception as e:
                logging.warning(f"[LooseTriplet] Failed for {sample_name}: {e}")
    else:
        logging.info(f"[LooseTriplet] Skipping (root not found): {healthy_root}")

    # 3) Dense CSV examples (e.g., GSE123902)
    gsm_samples = [
        {"path": Path("/scratch/project_2010751/GSE123902_RAW/GSM3516666_MSK_LX675_NORMAL_dense.csv"),
         "name": "GSM3516666_NORMAL", "condition": "Healthy"},
        {"path": Path("/scratch/project_2010751/GSE123902_RAW/GSM3516665_MSK_LX675_PRIMARY_TUMOUR_dense.csv"),
         "name": "GSM3516665_PRIMARY", "condition": "AML"},
        {"path": Path("/scratch/project_2010751/GSE123902_RAW/GSM3516667_MSK_LX676_PRIMARY_TUMOUR_dense.csv"),
         "name": "GSM3516667_PRIMARY", "condition": "AML"},
        {"path": Path("/scratch/project_2010751/GSE123902_RAW/GSM3516664_MSK_LX666_METASTASIS_dense.csv"),
         "name": "GSM3516664_METASTASIS", "condition": "AML"},
        {"path": Path("/scratch/project_2010751/GSE123902_RAW/GSM3516668_MSK_LX255B_METASTASIS_dense.csv"),
         "name": "GSM3516668_METASTASIS", "condition": "AML"},
    ]

    for s in gsm_samples:
        if s["path"].exists():
            try:
                expr = load_and_aggregate_dense_csv(s["path"], s["name"], max_cells_per_sample)
                if not expr.empty:
                    samples_data.append(expr)
                    sample_metadata.append({"sample_id": s["name"], "condition": s["condition"], "data_type": "dense_csv"})
            except Exception as e:
                logging.warning(f"[DenseCSV] Failed for {s['name']}: {e}")
        else:
            logging.info(f"[DenseCSV] Skipping (not found): {s['path']}")

    # Bail out if nothing loaded
    if not samples_data:
        logging.error("No samples were successfully loaded. Please point the script at real data locations.")
        return None, None

    # Combine
    logging.info(f"Combining {len(samples_data)} samples into expression matrix...")
    expr_df = pd.concat(samples_data, axis=1, join="outer").fillna(0.0)
    expr_df = expr_df.loc[(expr_df != 0).any(axis=1)]  # drop all-zero genes

    metadata_df = pd.DataFrame(sample_metadata).set_index("sample_id")
    expr_df = expr_df[metadata_df.index]  # align column order

    logging.info(f"Final matrix: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
    logging.info(f"Conditions: {metadata_df['condition'].value_counts().to_dict()}")

    # CPM normalize
    logging.info("CPM normalization...")
    col_sums = expr_df.sum(axis=0).replace(0, 1)
    expr_df_cpm = expr_df.divide(col_sums, axis=1) * 1e6

    # log1p
    logging.info("log1p transform...")
    expr_df_log = np.log1p(expr_df_cpm)

    # Save
    expr_file = output_dir / "pseudobulk_expression.csv"
    expr_log_file = output_dir / "pseudobulk_expression_log1p.csv"
    metadata_file = output_dir / "sample_metadata.csv"

    logging.info(f"Saving CPM to {expr_file}")
    expr_df_cpm.to_csv(expr_file)

    logging.info(f"Saving log1p to {expr_log_file}")
    expr_df_log.to_csv(expr_log_file)

    logging.info(f"Saving metadata to {metadata_file}")
    metadata_df.to_csv(metadata_file)

    # Summary
    summary = {
        "n_samples": int(expr_df.shape[1]),
        "n_genes": int(expr_df.shape[0]),
        "samples_per_condition": {k: int(v) for k, v in metadata_df["condition"].value_counts().to_dict().items()},
        "data_types": {k: int(v) for k, v in metadata_df["data_type"].value_counts().to_dict().items()},
        "top_expressed_genes": expr_df_cpm.mean(axis=1).nlargest(20).index.tolist(),
        "files_created": [str(expr_file), str(expr_log_file), str(metadata_file)],
    }
    summary_file = output_dir / "data_generation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("PSEUDO-BULK DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Samples: {summary['n_samples']}")
    print(f"Genes: {summary['n_genes']}")
    print(f"Conditions: {summary['samples_per_condition']}")
    print("\nFiles created:")
    print(f"  Expression (CPM):   {expr_file}")
    print(f"  Expression (log1p): {expr_log_file}")
    print(f"  Metadata:           {metadata_file}")
    print(f"  Summary:            {summary_file}")
    print("\n" + "=" * 60)
    print("NEXT: Module scoring with morphogenesis genes")
    print("=" * 60)
    print("python module_score_analysis.py \\")
    print(f"  --expression {expr_log_file} \\")
    print(f"  --metadata   {metadata_file} \\")
    print("  --inline-module 'GO_morphogenesis:AGT,AREG,BMP2,CAMSAP3,CD44,CELSR1,CSF1,CTNNB1,DDR1,DEAF1,EFNB2,EGFR,EPHA2,EPHA4,FERMT1,FERMT2,FOLR1,FOXQ1,FZD6,HNF1B,HOXB7,IFT57,INTU,IRX2,IRX3,KDF1,KDM5B,KDR,KLF4,LAMA5,LGR4,LRG1,LRP5,LZTS2,MDK,MESP1,MET,MYC,MYO9A,NAGLU,NKX2-1,NPHP1,PRICKLE1,PTEN,RPGRIP1L,SDC4,SNAI2,SOX9,SYNE4,TCTN1,TGFB1I1,TGFB2,TNC,WNT4,WNT7B,YAP1' \\")
    print("  --method pca \\")
    print(f"  --outdir {output_dir}/module_results")

    return expr_df_log, metadata_df


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-bulk expression data for module scoring")
    parser.add_argument("--output-dir", type=Path, default=Path("./pseudobulk_data"),
                        help="Output directory for expression and metadata files")
    parser.add_argument("--max-cells", type=int, default=20000,
                        help="Maximum cells per sample (for memory management)")
    args = parser.parse_args()

    expr_df, metadata_df = create_bulk_expression_matrix(output_dir=args.output_dir,
                                                         max_cells_per_sample=args.max_cells)
    if expr_df is not None:
        print("\nSuccess! Module scoring inputs were generated.")
        print(f"\nYou have {metadata_df['condition'].value_counts()['AML']} AML and "
              f"{metadata_df['condition'].value_counts()['Healthy']} Healthy samples")
    else:
        print("\nError: No samples were loaded. See the log for which paths were missing.")


if __name__ == "__main__":
    main()
