#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Inference for Reduced Persister Model (1000 genes)
Supports both full (13,369 genes) and reduced (1,000 genes) models
- Handles 10x folders (*/filtered_feature_bc_matrix) and dense CSVs
- Handles GSE120221 "flat triplets" (matrix.mtx.gz, genes.tsv.gz, barcodes.tsv.gz in one directory)
"""

import os
import sys
import gzip
import logging
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
import tensorflow as tf
import joblib

# -------------------------
# Setup
# -------------------------
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("inference")

# -------------------------
# Utilities
# -------------------------
def clean_gene_names(names: List[str]) -> List[str]:
    return [str(g).strip().upper().rsplit(".", 1)[0] for g in names]

def scrna_cpm_log1p(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.maximum(X, 0.0)
    lib = X.sum(axis=1, keepdims=True)
    np.maximum(lib, 1.0, out=lib)
    X = (X / lib) * 1e4
    return np.log1p(X).astype(np.float32)

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")

# -------------------------
# Data Loaders
# -------------------------
def load_dense_csv(path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load CSV file with auto-orientation detection"""
    df = pd.read_csv(path, index_col=0, low_memory=False)
    idx = pd.Index(df.index.astype(str))
    looks_like_genes = (len(idx) > 100) and (pd.Series(idx).str.match(r"^[A-Za-z0-9\-._]+$").mean() > 0.9)

    if looks_like_genes and df.shape[0] > df.shape[1]:
        X = df.T.values.astype(np.float32, copy=False)
        genes = clean_gene_names(df.index.tolist())
        cells = df.columns.astype(str).tolist()
    else:
        X = df.values.astype(np.float32, copy=False)
        genes = clean_gene_names(df.columns.astype(str).tolist())
        cells = df.index.astype(str).tolist()

    return X, genes, cells

def _read_tsv_lines(p: Path) -> List[str]:
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as f:
        return [line.rstrip("\n") for line in f]

def load_10x_dir(matrix_dir: Path, max_cells: Optional[int] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load 10x Genomics directory (has matrix.mtx[.gz], features/genes.tsv[.gz], barcodes.tsv[.gz])"""
    # Find matrix
    mtx = None
    for cand in ["matrix.mtx.gz", "matrix.mtx"]:
        p = matrix_dir / cand
        if p.exists():
            mtx = mmread(str(p))
            break
    if mtx is None:
        raise FileNotFoundError(f"No matrix.mtx(.gz) found in {matrix_dir}")

    # Features/genes
    feat = None
    for cand in ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"]:
        p = matrix_dir / cand
        if p.exists():
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rt") as f:
                feat = [line.strip().split("\t") for line in f]
            break
    if feat is None:
        raise FileNotFoundError(f"No features.tsv/genes.tsv found in {matrix_dir}")

    # Barcodes
    barcodes = None
    for cand in ["barcodes.tsv.gz", "barcodes.tsv"]:
        p = matrix_dir / cand
        if p.exists():
            barcodes = _read_tsv_lines(p)
            break
    if barcodes is None:
        barcodes = [f"CELL_{i}" for i in range(mtx.shape[1])]

    # Gene names
    genes = [str(r[1]).upper() if len(r) >= 2 else str(r[0]).upper() for r in feat]

    # Matrix to dense cells x genes
    if sparse.isspmatrix(mtx):
        mtx = mtx.tocsr()  # genes x cells
        if (max_cells is not None) and (max_cells > 0) and (mtx.shape[1] > max_cells):
            rng = np.random.default_rng(SEED)
            keep = np.sort(rng.choice(mtx.shape[1], max_cells, replace=False))
            mtx = mtx[:, keep]
            barcodes = [barcodes[i] for i in keep]
        X = mtx.T.astype(np.float32).toarray()
    else:
        M = np.asarray(mtx)
        if (max_cells is not None) and (max_cells > 0) and (M.shape[1] > max_cells):
            rng = np.random.default_rng(SEED)
            keep = np.sort(rng.choice(M.shape[1], max_cells, replace=False))
            M = M[:, keep]
            barcodes = [barcodes[i] for i in keep]
        X = M.T.astype(np.float32)

    return X, clean_gene_names(genes), barcodes

def load_10x_flat(matrix_path: Path, genes_path: Path, barcodes_path: Path,
                  max_cells: Optional[int] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load a 'flat triplet' (each file is in the same folder), e.g. GSE120221:
      - GSMxxxxxxx_matrix_*.mtx.gz
      - GSMxxxxxxx_genes_*.tsv.gz
      - GSMxxxxxxx_barcodes_*.tsv.gz
    """
    mtx = mmread(str(matrix_path))

    # Genes
    opener = gzip.open if str(genes_path).endswith(".gz") else open
    with opener(genes_path, "rt") as f:
        feat = [line.strip().split("\t") for line in f]
    genes = [str(r[1]).upper() if len(r) >= 2 else str(r[0]).upper() for r in feat]
    genes = clean_gene_names(genes)

    # Barcodes
    barcodes = _read_tsv_lines(barcodes_path)
    if not barcodes:
        barcodes = [f"CELL_{i}" for i in range(mtx.shape[1])]

    # To dense cells x genes
    if sparse.isspmatrix(mtx):
        mtx = mtx.tocsr()  # genes x cells
        if (max_cells is not None) and (max_cells > 0) and (mtx.shape[1] > max_cells):
            rng = np.random.default_rng(SEED)
            keep = np.sort(rng.choice(mtx.shape[1], max_cells, replace=False))
            mtx = mtx[:, keep]
            barcodes = [barcodes[i] for i in keep]
        X = mtx.T.astype(np.float32).toarray()
    else:
        M = np.asarray(mtx)
        if (max_cells is not None) and (max_cells > 0) and (M.shape[1] > max_cells):
            rng = np.random.default_rng(SEED)
            keep = np.sort(rng.choice(M.shape[1], max_cells, replace=False))
            M = M[:, keep]
            barcodes = [barcodes[i] for i in keep]
        X = M.T.astype(np.float32)

    return X, genes, barcodes

# -------------------------
# Model Inference Class
# -------------------------
class PersisterInference:
    def __init__(self, model_dir: Path, genes_file: Optional[Path] = None,
                 threshold_override: Optional[float] = None, use_reduced: bool = True):
        self.model_dir = model_dir
        self.use_reduced = use_reduced

        # Find and load model
        model_path = self._find_model_file(model_dir)
        self.model = self._load_model(model_path)

        # Load preprocessing components
        if use_reduced:
            self.scaler = self._load_optional(model_dir / "scaler_reduced.pkl", "scaler")
            self.pca = self._load_optional(model_dir / "pca_reduced.pkl", "pca")
            self.threshold = self._load_threshold(model_dir / "threshold_reduced.pkl", threshold_override)
        else:
            self.scaler = self._load_optional(model_dir / "scaler.pkl", "scaler")
            self.pca = self._load_optional(model_dir / "pca.pkl", "pca")
            self.threshold = self._load_threshold(model_dir / "threshold.pkl", threshold_override)

        self.resid = self._load_optional(model_dir / "residualizer.pkl", "residualizer")

        # Load gene list
        self.training_genes = self._load_training_genes(model_dir, genes_file)

        log.info(f"[MODEL] Type: {'REDUCED (1000 genes)' if use_reduced else 'FULL (13369 genes)'}")
        log.info(f"[MODEL] Genes: {len(self.training_genes)}")
        log.info(f"[MODEL] PCA components: {getattr(self.pca, 'n_components_', None)}")
        log.info(f"[MODEL] Threshold: {self.threshold:.3f}")

    def _find_model_file(self, model_dir: Path) -> Path:
        """Find the model file (supports both reduced and full naming)"""
        candidates = [
            "model_reduced.h5",  # Reduced model
            "final_model.h5",    # Full model
            "best_model.keras",
            "aml_persister_transformer.keras",
        ]
        for name in candidates:
            p = model_dir / name
            if p.exists():
                return p
        raise FileNotFoundError(f"No model found in {model_dir}")

    def _load_model(self, p: Path):
        log.info(f"[MODEL] Loading {p}")
        m = tf.keras.models.load_model(p, compile=False)
        m.compile(optimizer="adam", loss="binary_crossentropy")
        return m

    def _load_optional(self, p: Path, kind: str):
        if p.exists():
            try:
                obj = joblib.load(p)
                log.info(f"[LOAD] {kind} loaded from {p}")
                return obj
            except Exception as e:
                log.warning(f"[LOAD] Could not load {kind}: {e}")
                return None
        return None

    def _load_training_genes(self, model_dir: Path, override: Optional[Path]) -> List[str]:
        """Load gene list with support for reduced model"""
        if override and override.exists():
            with open(override) as f:
                genes = [line.strip().upper() for line in f if line.strip()]
            log.info(f"[GENES] Loaded {len(genes)} genes from {override}")
            return genes

        # Try different gene file names
        for fname in ["selected_genes.txt", "genes_training_order.txt", "common_genes.txt"]:
            p = model_dir / fname
            if p.exists():
                with open(p) as f:
                    genes = [line.strip().upper() for line in f if line.strip()]
                log.info(f"[GENES] Loaded {len(genes)} genes from {p}")
                return genes

        # Try metadata directory
        meta = model_dir.parent / "metadata"
        for fname in ["common_genes.txt", "genes_training_order.txt"]:
            p = meta / fname
            if p.exists():
                with open(p) as f:
                    genes = [line.strip().upper() for line in f if line.strip()]
                log.info(f"[GENES] Loaded {len(genes)} genes from {p}")
                return genes

        raise FileNotFoundError("No gene file found")

    def _load_threshold(self, p: Path, override: Optional[float]) -> float:
        if override is not None:
            log.info(f"[THRESH] Using override: {override:.3f}")
            return float(override)
        if p.exists():
            try:
                thr = joblib.load(p)
                t = float(thr)
                log.info(f"[THRESH] Loaded threshold: {t:.3f}")
                return t
            except Exception as e:
                log.warning(f"[THRESH] Failed to load: {e}")
        return 0.5

    def process_sample(self, sample_source: Any, sample_name: str,
                       max_cells: int = 20000, batch_size: int = 8192) -> Tuple[pd.DataFrame, dict]:
        """
        Process a single sample (batched to control RAM)
        sample_source can be:
          - Path to a 10x folder
          - Path to a dense CSV
          - Tuple[Path, Path, Path] for 'flat triplet' (matrix, genes, barcodes)
        """
        # Load data
        if isinstance(sample_source, tuple) and len(sample_source) == 3:
            matrix_p, genes_p, barcodes_p = sample_source
            X, genes, cell_ids = load_10x_flat(matrix_p, genes_p, barcodes_p, max_cells)
            pretty_name = Path(matrix_p).name
        else:
            sample_path = Path(sample_source)
            if sample_path.is_dir():
                X, genes, cell_ids = load_10x_dir(sample_path, max_cells)
                pretty_name = sample_path.name
            else:
                X, genes, cell_ids = load_dense_csv(sample_path)
                pretty_name = sample_path.name

        n, g = X.shape
        log.info(f"[LOAD] {sample_name}: {n:,} cells × {g:,} genes")

        # Align
        gene2idx = {ge: i for i, ge in enumerate(genes)}
        idx_map = np.array([gene2idx.get(ge, -1) for ge in self.training_genes], dtype=np.int32)
        matched = int((idx_map >= 0).sum())
        log.info(f"[ALIGN] Gene coverage: {matched}/{len(self.training_genes)} ({matched/len(self.training_genes)*100:.1f}%)")

        # Predict in batches
        probs_all = np.empty(n, dtype=np.float32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = X[start:end, :]

            Xb_aligned = np.zeros((end - start, len(self.training_genes)), dtype=np.float32)
            valid_cols = idx_map >= 0
            if valid_cols.any():
                Xb_aligned[:, valid_cols] = Xb[:, idx_map[valid_cols]]

            # Normalize + preprocess
            Xb_norm = scrna_cpm_log1p(Xb_aligned)
            if self.scaler is not None:
                Xb_norm = self.scaler.transform(Xb_norm)
            if self.pca is not None:
                Xb_norm = self.pca.transform(Xb_norm)

            # Predict
            probs_all[start:end] = self.model.predict(Xb_norm, verbose=0).ravel()

        preds = (probs_all >= self.threshold).astype(int)
        pos = int(preds.sum())
        neg = n - pos
        pos_pct = (pos / n * 100.0) if n else 0.0
        log.info(f"[RESULT] {sample_name}: {pos} persisters ({pos_pct:.1f}%), {neg} non-persisters")

        df_out = pd.DataFrame({
            "cell_id": cell_ids[:n],
            "prob_persister": probs_all,
            "pred_label": np.where(preds == 1, "Persister", "Non-Persister")
        })

        summary = {
            "sample": sample_name,
            "threshold": float(self.threshold),
            "cells": n,
            "persister_count": pos,
            "non_persister_count": neg,
            "persister_pct": float(pos_pct),
            "mean_prob": float(np.mean(probs_all)),
            "std_prob": float(np.std(probs_all))
        }

        return df_out, summary

# -------------------------
# Sample Discovery
# -------------------------
def build_sample_list(aml_root: Path, healthy_root: Path) -> List[Tuple[Any, str]]:
    """
    Build list of samples from:
      - AML 10x under aml_root (*/filtered_feature_bc_matrix)
      - Healthy GSE120221 under healthy_root (flat triplets)
      - GSE123902 CSVs (the GSM35166xx series)
    Returns list of tuples: (sample_source, label)
      sample_source is Path (dir or csv) or (matrix_path, genes_path, barcodes_path) for flat triplets
    """
    samples: List[Tuple[Any, str]] = []

    # AML 10x (recursive)
    if aml_root.exists():
        mtx_dirs = sorted(set(
            Path(p).parent for p in
            map(Path,
                os.popen(
                    f"find '{aml_root}' -type f -path '*/filtered_feature_bc_matrix/matrix.mtx*' 2>/dev/null"
                ).read().splitlines()
            )
        ))
        for d in mtx_dirs:
            # Give a readable label based on parent chain
            label = d.parent.name if d.parent.name != "outs" else d.parent.parent.name
            samples.append((d, label))

    # Healthy (GSE120221) flat triplets (files in one directory)
    if healthy_root.exists():
        groups: Dict[Tuple[str, str], Dict[str, Path]] = defaultdict(dict)

        # Files are named like:
        #   GSM3396161_matrix_A.mtx.gz
        #   GSM3396161_genes_A.tsv.gz
        #   GSM3396161_barcodes_A.tsv.gz
        for p in healthy_root.iterdir():
            name = p.name
            m = re.match(r'^(GSM\d+)_(barcodes|genes|matrix)_([A-Za-z0-9]+)\.(?:mtx|tsv)\.gz$', name)
            if not m:
                continue
            gsm, kind, tag = m.groups()
            groups[(gsm, tag)][kind] = p

        log.info(f"[DISCOVER] Healthy GSM-tag groups formed: {len(groups)}")

        for (gsm, tag), dct in sorted(groups.items()):
            # Require exactly the three sides below
            needed = {'matrix', 'genes', 'barcodes'}
            if needed <= dct.keys():
                label = f"{gsm}_{tag}_HEALTHY"
                samples.append(((dct['matrix'], dct['genes'], dct['barcodes']), label))
            else:
                missing = sorted(needed - set(dct.keys()))
                log.warning(f"[SKIP] {gsm}_{tag}: missing {missing}")

    # GSE123902 CSVs (non-healthy)
    gse123902 = Path("/scratch/project_2010751/GSE123902_RAW")
    gse_csvs = [
        ("GSM3516664_MSK_LX666_METASTASIS_dense.csv", "GSM3516664_METASTASIS"),
        ("GSM3516665_MSK_LX675_PRIMARY_TUMOUR_dense.csv", "GSM3516665_PRIMARY"),
        ("GSM3516666_MSK_LX675_NORMAL_dense.csv", "GSM3516666_NORMAL"),
        ("GSM3516667_MSK_LX676_PRIMARY_TUMOUR_dense.csv", "GSM3516667_PRIMARY"),
        ("GSM3516668_MSK_LX255B_METASTASIS_dense.csv", "GSM3516668_METASTASIS"),
        ("GSM3516671_MSK_LX681_METASTASIS_dense.csv", "GSM3516671_METASTASIS"),
    ]
    for fname, label in gse_csvs:
        p = gse123902 / fname
        if p.exists():
            samples.append((p, label))

    return samples

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Persister Inference Pipeline")
    parser.add_argument("--model-dir", type=Path, required=True, help="Model directory")
    parser.add_argument("--genes-file", type=Path, help="Path to gene list")
    parser.add_argument("--threshold", type=float, help="Override threshold")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--max-cells", type=int, default=20000, help="Max cells per sample (-1 or 0 = all)")
    parser.add_argument("--batch-size", type=int, default=8192, help="Cells per prediction batch")
    parser.add_argument("--use-reduced", action="store_true", help="Use reduced model")
    parser.add_argument("--use-full", action="store_true", help="Use full model")
    parser.add_argument("--aml-root", type=Path, default=Path("/scratch/project_2010751/AML_scRNA_decrypted"),
                        help="Root containing AML 10x datasets")
    parser.add_argument("--healthy-root", type=Path, default=Path("/scratch/project_2010751/GSE120221_RAW"),
                        help="Root containing GSE120221 datasets")

    args = parser.parse_args()

    # Which model?
    use_reduced = True
    if args.use_full:
        use_reduced = False
    elif args.use_reduced:
        use_reduced = True

    # Paths
    model_dir = args.model_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Quick discovery counts (informational)
    aml_seen = int(os.popen(
        f"find '{args.aml_root}' -type f -path '*/filtered_feature_bc_matrix/matrix.mtx*' 2>/dev/null "
        "| sed 's#/filtered_feature_bc_matrix/.*##' | sort -u | wc -l"
    ).read().strip() or "0")
    healthy_seen = len([1 for p in Path(args.healthy_root).glob("GSM*_matrix_*.mtx.gz")])
    log.info(f"[DISCOVER] AML datasets seen: {aml_seen} under {args.aml_root}")
    log.info(f"[DISCOVER] Healthy triplets seen: {healthy_seen} under {args.healthy_root}")

    log.info(f"Starting inference at {pd.Timestamp.now()}")
    log.info(f"Model directory: {model_dir}")
    log.info(f"Output directory: {out_dir}")

    # Initialize inference engine
    infer = PersisterInference(
        model_dir,
        genes_file=args.genes_file,
        threshold_override=args.threshold,
        use_reduced=use_reduced
    )

    # Build sample list
    samples = build_sample_list(args.aml_root.resolve(), args.healthy_root.resolve())
    log.info(f"[DISCOVER] Total samples to process: {len(samples)}")

    # Process samples
    all_summaries = []
    for sample_source, sample_name in samples:
        # Path existence checks
        try:
            if isinstance(sample_source, tuple):
                ok = all(Path(p).exists() for p in sample_source)
            else:
                ok = Path(sample_source).exists()
            if not ok:
                log.warning(f"[SKIP] {sample_name}: path does not exist ({sample_source})")
                continue
        except Exception as e:
            log.warning(f"[SKIP] {sample_name}: bad path ({sample_source}): {e}")
            continue

        try:
            df_cells, summary = infer.process_sample(
                sample_source, sample_name,
                max_cells=args.max_cells,
                batch_size=args.batch_size
            )
            all_summaries.append(summary)

            # Save predictions
            if not df_cells.empty:
                df_cells.to_csv(out_dir / f"{sanitize(sample_name)}_predictions.csv", index=False)

        except Exception as e:
            log.error(f"[ERROR] {sample_name}: {e}")

    # Save summary
    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        df_summary.to_csv(out_dir / "inference_results_summary.csv", index=False)

        # Print summary statistics
        log.info("\n" + "="*80)
        log.info("INFERENCE COMPLETE")
        log.info("="*80)
        log.info(f"Processed: {len(all_summaries)} samples")
        log.info(f"Total cells: {df_summary['cells'].sum():,}")
        log.info(f"Mean persister %: {df_summary['persister_pct'].mean():.1f}%")
        log.info(f"Results saved to: {out_dir}")
    else:
        log.warning("No summaries produced; check dataset discovery and paths.")

if __name__ == "__main__":
    main()
