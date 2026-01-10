#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL: AML Persister Inference with H5 Support & Proper Patient ID Extraction
"""

import os, sys, gzip, json, logging, warnings, re, argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
import tensorflow as tf
import joblib

try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("inference")

# ============================================================================
# Helper Classes
# ============================================================================

class GroupResidualizer:
    def __init__(self):
        self.sid_to_col = None
        self.B = None
    def _design(self, sids, fit=False):
        sids = np.asarray(sids)
        if fit:
            uniq = sorted(np.unique(sids))
            self.sid_to_col = {sid: i+1 for i, sid in enumerate(uniq)}
        n = len(sids)
        G = 1 + (0 if self.sid_to_col is None else len(self.sid_to_col))
        Z = np.zeros((n, G), dtype=np.float32)
        Z[:, 0] = 1.0
        if self.sid_to_col:
            for i, sid in enumerate(sids):
                j = self.sid_to_col.get(sid, None)
                if j is not None:
                    Z[i, j] = 1.0
        return Z
    def fit(self, X, sids):
        X = np.asarray(X, dtype=np.float32)
        Z = self._design(sids, fit=True)
        ZTZ_inv = np.linalg.pinv(Z.T @ Z)
        self.B = ZTZ_inv @ (Z.T @ X)
        return self
    def transform(self, X, sids):
        X = np.asarray(X, dtype=np.float32)
        Z = self._design(sids, fit=False)
        return X - Z @ self.B

def clean_gene_names(names: List[str]) -> List[str]:
    return [str(g).strip().upper().rsplit(".", 1)[0] for g in names]

def scrna_cpm_log1p(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.maximum(X, 0.0)
    lib = X.sum(axis=1, keepdims=True)
    np.maximum(lib, 1.0, out=lib)
    X = (X / lib) * 1e4
    return np.log1p(X).astype(np.float32)

# ============================================================================
# FIXED: H5 Discovery & Patient ID Extraction
# ============================================================================

def extract_patient_id_from_h5_path(h5_path: Path) -> str:
    """
    Extract patient ID from H5 file path.
    
    Example:
    .../FH_6088_3/outs/filtered_feature_bc_matrix.h5 → FH_6088_3
    .../FHRB_1886_6/outs/filtered_feature_bc_matrix.h5 → FHRB_1886_6
    """
    SKIP = {"outs", "filtered_feature_bc_matrix", "count"}
    
    # Walk up from H5 file
    for part in h5_path.parts[::-1]:
        if part in SKIP or part.endswith('.h5'):
            continue
        # Check if looks like patient ID
        if re.match(r'^(FH|FHRB|BERG)_\d+', part):
            return part
    
    # Fallback
    return h5_path.parent.name

def discover_h5_files(root: Path) -> List[Tuple[Path, str]]:
    """Find all H5 files and extract patient IDs"""
    samples = []
    if not root.exists():
        log.warning(f"Root not found: {root}")
        return samples
    
    for h5_file in root.rglob("filtered_feature_bc_matrix.h5"):
        if h5_file.is_file():
            patient_id = extract_patient_id_from_h5_path(h5_file)
            samples.append((h5_file, patient_id))
            log.info(f"  H5: {patient_id} at {h5_file.relative_to(root)}")
    
    return samples

# ============================================================================
# Data Loading
# ============================================================================

def load_h5_file(h5_path: Path, max_cells: Optional[int] = None):
    if not HAS_SCANPY:
        raise ImportError("scanpy required for H5. Install: pip install scanpy")
    
    adata = sc.read_10x_h5(h5_path)
    
    if max_cells and adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=SEED)
    
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    X = X.astype(np.float32)
    
    genes = [str(g).upper() for g in adata.var_names]
    cell_ids = [str(c) for c in adata.obs_names]
    
    return X, clean_gene_names(genes), cell_ids

def merge_duplicate_columns(X: np.ndarray, genes: List[str]):
    genes_arr = np.asarray(genes)
    uniq, inv = np.unique(genes_arr, return_inverse=True)
    if len(uniq) == len(genes_arr):
        return X, genes, 0
    out = np.zeros((X.shape[0], len(uniq)), dtype=X.dtype)
    for j in range(X.shape[1]):
        out[:, inv[j]] += X[:, j]
    return out, uniq.tolist(), int(len(genes_arr) - len(uniq))

def align_to_training_genes(X, genes_raw, training_genes, id2name):
    genes_h = clean_gene_names(genes_raw)
    Xm, genes_u, n_merged = merge_duplicate_columns(X, genes_h)
    gene2idx = {g: i for i, g in enumerate(genes_u)}
    out = np.zeros((Xm.shape[0], len(training_genes)), dtype=np.float32)
    found = 0
    for j, g in enumerate(training_genes):
        i = gene2idx.get(g)
        if i is not None:
            out[:, j] = Xm[:, i]
            found += 1
    return out, found, n_merged

# ============================================================================
# Inference Class
# ============================================================================

class PersisterInference:
    def __init__(self, model_dir: Path, genes_file: Path, threshold: float = 0.5):
        model_path = model_dir / "final_model.h5"
        if not model_path.exists():
            model_path = model_dir / "final_model.keras"
        
        self.model = tf.keras.models.load_model(model_path, compile=False)
        log.info(f"[MODEL] Loaded from {model_path}")
        
        self.scaler = self._load_opt(model_dir / "scaler.pkl")
        self.pca = self._load_opt(model_dir / "pca.pkl")
        self.resid = self._load_opt(model_dir / "residualizer.pkl")
        
        with open(genes_file) as f:
            self.training_genes = [line.strip().upper() for line in f if line.strip()]
        log.info(f"[GENES] Loaded {len(self.training_genes)} genes")
        
        self.id2name = {}
        self.threshold = threshold
    
    def _load_opt(self, p: Path):
        if p.exists():
            try:
                return joblib.load(p)
            except:
                pass
        return None
    
    def _preprocess(self, X):
        X = scrna_cpm_log1p(X)
        if self.resid and getattr(self.resid, "B", None) is not None:
            try:
                b0 = np.asarray(self.resid.B)[0]
                if b0.shape[0] == X.shape[1]:
                    X = X - b0
            except:
                pass
        if self.scaler:
            X = self.scaler.transform(X)
        if self.pca:
            X = self.pca.transform(X).astype(np.float32)
        return X
    
    def predict_sample(self, h5_path: Path, sample_name: str):
        X, genes, cell_ids = load_h5_file(h5_path, max_cells=ARGS.max_cells)
        log.info(f"[LOAD] {sample_name}: cells={X.shape[0]:,} genes={X.shape[1]:,}")
        
        X_aligned, found, n_merged = align_to_training_genes(X, genes, self.training_genes, self.id2name)
        pct = 100.0 * found / len(self.training_genes)
        log.info(f"[ALIGN] {sample_name}: matched={found}/{len(self.training_genes)} ({pct:.1f}%)")
        
        Xp = self._preprocess(X_aligned)
        probs = self.model.predict(Xp, verbose=0).ravel().astype(np.float32)
        preds = (probs >= self.threshold).astype(np.int32)
        
        total = len(preds)
        pos = int(preds.sum())
        pos_pct = (pos / total * 100.0) if total else 0.0
        
        log.info(f"[RESULT] {sample_name}: Persister {pos} ({pos_pct:.1f}%)")
        
        df_out = pd.DataFrame({
            "cell_id": list(map(str, cell_ids[:total])),
            "prob_persister": probs,
            "pred_label": np.where(preds == 1, "Persister", "Non-Persister")
        })
        
        summary = {
            "sample": sample_name,
            "cells": total,
            "persister_count": pos,
            "persister_pct": float(pos_pct),
            "mean_prob": float(np.mean(probs)),
            "std_prob": float(np.std(probs))
        }
        
        return df_out, summary

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--genes-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--aml-root", type=Path, required=True)
    ap.add_argument("--max-cells", type=int, default=20000)
    return ap.parse_args()

ARGS = None

def main():
    global ARGS
    ARGS = parse_args()
    
    out_dir = ARGS.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"\n{'='*80}")
    log.info(f"AML PERSISTER INFERENCE (H5 FIXED)")
    log.info(f"{'='*80}")
    log.info(f"Output: {out_dir}")
    
    if not HAS_SCANPY:
        log.error("scanpy not installed! Run: pip install scanpy")
        sys.exit(1)
    
    infer = PersisterInference(ARGS.model_dir, ARGS.genes_file)
    
    # Discover H5 files
    log.info(f"\nDiscovering H5 files in: {ARGS.aml_root}")
    samples = discover_h5_files(ARGS.aml_root)
    
    if not samples:
        log.error("No H5 files found!")
        sys.exit(1)
    
    log.info(f"\n✓ Found {len(samples)} H5 files")
    
    all_summaries = []
    for h5_path, sname in samples:
        log.info(f"\n{'='*70}")
        log.info(f"Processing: {sname}")
        try:
            df_cells, summary = infer.predict_sample(h5_path, sname)
            all_summaries.append(summary)
            df_cells.to_csv(out_dir / f"{sname}_predictions.csv", index=False)
        except Exception as e:
            log.error(f"[ERROR] {sname}: {e}")
    
    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv(out_dir / "inference_results_summary.csv", index=False)
    log.info(f"\n✓ Summary: {out_dir / 'inference_results_summary.csv'}")

if __name__ == "__main__":
    main()
