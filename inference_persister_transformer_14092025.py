#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Inference for Persister Cell Classifier
- Matches training pipeline (CPM→log1p → optional Residualizer (intercept-only) → StandardScaler → PCA → Keras model)
- Aligns to saved training gene order (auto-locate in models/ or ../metadata/ or via --genes-file)
- Supports:
    1) 10x folders: .../filtered_feature_bc_matrix/{matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz}
    2) Loose triplets (GSE120221 healthy): *_matrix_*.mtx.gz + *_genes_*.tsv.gz + *_barcodes_*.tsv.gz
    3) Dense CSV (cells x genes OR genes x cells; auto-detect)
- Merges duplicate symbols (sum) before alignment
"""

import os
import sys
import gzip
import json
import logging
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import re
import argparse

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
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("inference.log")]
)
log = logging.getLogger("inference")


# -------------------------
# Needed so residualizer.pkl can be unpickled
# -------------------------
class GroupResidualizer:
    def __init__(self):
        self.sid_to_col = None
        self.B = None  # [G x p]

    def _design(self, sids, fit=False):
        sids = np.asarray(sids)
        if fit:
            uniq = sorted(np.unique(sids))
            self.sid_to_col = {sid: i+1 for i, sid in enumerate(uniq)}  # +1 for intercept
        n = len(sids)
        G = 1 + (0 if self.sid_to_col is None else len(self.sid_to_col))
        Z = np.zeros((n, G), dtype=np.float32)
        Z[:, 0] = 1.0  # intercept
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
        self.B = ZTZ_inv @ (Z.T @ X)  # [G x p]
        return self

    def transform(self, X, sids):
        X = np.asarray(X, dtype=np.float32)
        Z = self._design(sids, fit=False)
        return X - Z @ self.B


# -------------------------
# Utilities (match training)
# -------------------------
def clean_gene_names(names: List[str]) -> List[str]:
    return [str(g).strip().upper().rsplit(".", 1)[0] for g in names]

def _read_tsv(path: Path) -> List[List[str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [line.rstrip("\n").split("\t") for line in f]

def _read_barcodes_file(p: Path) -> List[str]:
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as f:
        return [line.strip() for line in f]

def mapping_from_10x_features(tenx_dir: Path) -> Dict[str, str]:
    candidates = list(tenx_dir.glob("features.tsv*")) or list(tenx_dir.glob("genes.tsv*"))
    if not candidates:
        return {}
    rows = _read_tsv(candidates[0])
    m = {}
    for r in rows:
        gid = (r[0] if len(r) > 0 else "").upper().rsplit(".", 1)[0]
        gnm = (r[1] if len(r) > 1 else "").upper()
        if gid and gnm and gid.startswith("ENSG"):
            m[gid] = gnm
    return m

def load_id2name(model_dir: Path) -> Dict[str, str]:
    p = model_dir / "id2name.json"
    if p.exists():
        with open(p) as f:
            m = json.load(f)
        log.info(f"[MAP] Loaded ENSG→SYMBOL mapping from {p}")
        return {k.upper().rsplit(".", 1)[0]: v.upper() for k, v in m.items()}
    # fallback: try a couple of 10x dirs
    for tenx in [
        Path("/scratch/project_2010376/scRNAseq/FH_5897_2/filtered_feature_bc_matrix"),
        Path("/scratch/project_2010376/scRNAseq/FH_6333_2/filtered_feature_bc_matrix"),
    ]:
        if tenx.exists():
            m = mapping_from_10x_features(tenx)
            if m:
                log.info(f"[MAP] Built ENSG→SYMBOL mapping from {tenx}")
                return m
    log.warning("[MAP] No ENSG→SYMBOL mapping found; using symbols as-is.")
    return {}

def harmonize_to_symbols(genes: List[str], id2name: Dict[str, str]) -> List[str]:
    out = []
    for g in genes:
        g2 = g.upper().rsplit(".", 1)[0]
        if g2.startswith("ENSG"):
            g2 = id2name.get(g2, None)
        out.append(g2)
    return [g for g in out if g is not None]

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
# Loaders
# -------------------------
def load_dense_csv(path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
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

def load_10x_dir(matrix_dir: Path, max_cells: Optional[int] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    # matrix
    mtx = None
    for cand in ["matrix.mtx.gz", "matrix.mtx"]:
        p = matrix_dir / cand
        if p.exists():
            mtx = mmread(str(p))
            break
    if mtx is None:
        raise FileNotFoundError(f"No matrix.mtx(.gz) found in {matrix_dir}")
    # features
    feat = None
    for cand in ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"]:
        p = matrix_dir / cand
        if p.exists():
            feat = _read_tsv(p)
            break
    if feat is None:
        raise FileNotFoundError(f"No features.tsv/genes.tsv found in {matrix_dir}")
    # barcodes
    barcodes = None
    for cand in ["barcodes.tsv.gz", "barcodes.tsv"]:
        p = matrix_dir / cand
        if p.exists():
            barcodes = _read_barcodes_file(p)
            break
    if barcodes is None:
        barcodes = [f"CELL_{i}" for i in range(mtx.shape[1])]

    # genes: prefer symbol column if present
    genes = [str(r[1]).upper() if len(r) >= 2 else str(r[0]).upper() for r in feat]

    # subsample BEFORE densifying
    if sparse.isspmatrix(mtx):
        mtx = mtx.tocsr()  # genes x cells
        if max_cells and mtx.shape[1] > max_cells:
            rng = np.random.default_rng(SEED)
            keep = np.sort(rng.choice(mtx.shape[1], max_cells, replace=False))
            mtx = mtx[:, keep]
            barcodes = [barcodes[i] for i in keep]
        X = mtx.T.astype(np.float32).toarray()
    else:
        M = np.asarray(mtx)
        if max_cells and M.shape[1] > max_cells:
            rng = np.random.default_rng(SEED)
            keep = np.sort(rng.choice(M.shape[1], max_cells, replace=False))
            M = M[:, keep]
            barcodes = [barcodes[i] for i in keep]
        X = M.T.astype(np.float32)

    return X, clean_gene_names(genes), barcodes

def load_loose_triplet_matrix_file(matrix_file: Path, max_cells: Optional[int] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load GSE120221-style loose triplets:
      GSMxxxx_matrix_*.mtx.gz + GSMxxxx_genes_*.tsv.gz + GSMxxxx_barcodes_*.tsv.gz
    """
    stem = matrix_file.name  # e.g., GSM3396161_matrix_A.mtx.gz
    m = re.match(r"^(.*)_matrix_(.*)\.mtx(\.gz)?$", stem)
    if not m:
        raise ValueError(f"Matrix filename not recognized: {matrix_file}")
    prefix, suff = m.group(1), m.group(2)

    genes_file = matrix_file.with_name(f"{prefix}_genes_{suff}.tsv.gz")
    if not genes_file.exists():
        genes_file = matrix_file.with_name(f"{prefix}_genes_{suff}.tsv")
    barcodes_file = matrix_file.with_name(f"{prefix}_barcodes_{suff}.tsv.gz")
    if not barcodes_file.exists():
        barcodes_file = matrix_file.with_name(f"{prefix}_barcodes_{suff}.tsv")
    if not genes_file.exists() or not barcodes_file.exists():
        raise FileNotFoundError(f"Missing genes/barcodes for {matrix_file}")

    mtx = mmread(str(matrix_file))
    if sparse.isspmatrix(mtx):
        mtx = mtx.tocsr()
        if max_cells and mtx.shape[1] > max_cells:
            rng = np.random.default_rng(SEED)
            keep = np.sort(rng.choice(mtx.shape[1], max_cells, replace=False))
            mtx = mtx[:, keep]
            barcodes = _read_barcodes_file(barcodes_file)
            barcodes = [barcodes[i] for i in keep]
        else:
            barcodes = _read_barcodes_file(barcodes_file)
        X = mtx.T.astype(np.float32).toarray()
    else:
        M = np.asarray(mtx)
        if max_cells and M.shape[1] > max_cells:
            rng = np.random.default_rng(SEED)
            keep = np.sort(rng.choice(M.shape[1], max_cells, replace=False))
            M = M[:, keep]
            barcodes = _read_barcodes_file(barcodes_file)
            barcodes = [barcodes[i] for i in keep]
        else:
            barcodes = _read_barcodes_file(barcodes_file)
        X = M.T.astype(np.float32)

    rows = _read_tsv(genes_file)
    genes = [str(r[1]).upper() if len(r) >= 2 else str(r[0]).upper() for r in rows]
    return X, clean_gene_names(genes), barcodes


# -------------------------
# Duplicate-merge helper
# -------------------------
def merge_duplicate_columns_dense(X: np.ndarray, genes: List[str]) -> Tuple[np.ndarray, List[str], int]:
    genes_arr = np.asarray(genes)
    uniq, inv = np.unique(genes_arr, return_inverse=True)
    if len(uniq) == len(genes_arr):
        return X, genes, 0
    out = np.zeros((X.shape[0], len(uniq)), dtype=X.dtype)
    for j in range(X.shape[1]):
        out[:, inv[j]] += X[:, j]
    n_merged = int(len(genes_arr) - len(uniq))
    return out, uniq.tolist(), n_merged


# -------------------------
# Alignment & preprocessing
# -------------------------
def align_to_training_genes(
    X: np.ndarray,
    genes_raw: List[str],
    training_genes: List[str],
    id2name: Dict[str, str]
) -> Tuple[np.ndarray, int, int]:
    genes_h = harmonize_to_symbols(genes_raw, id2name) if id2name else clean_gene_names(genes_raw)
    Xm, genes_u, n_merged = merge_duplicate_columns_dense(X, genes_h)
    gene2idx = {g: i for i, g in enumerate(genes_u)}
    out = np.zeros((Xm.shape[0], len(training_genes)), dtype=np.float32)
    found = 0
    for j, g in enumerate(training_genes):
        i = gene2idx.get(g)
        if i is not None:
            out[:, j] = Xm[:, i]
            found += 1
    return out, found, n_merged


# -------------------------
# Inference core
# -------------------------
def find_model_file(model_dir: Path) -> Path:
    candidates = [
        "final_model.h5",
        "best_model.keras",
        "aml_persister_transformer.keras",
        "final_model.keras",
    ]
    for name in candidates:
        p = model_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No model found in {model_dir} (tried: {', '.join(candidates)})")

class PersisterInference:
    def __init__(self, model_dir: Path, genes_file: Optional[Path] = None, threshold_override: Optional[float] = None):
        self.model_dir = model_dir
        model_path = find_model_file(model_dir)
        self.model = self._load_model(model_path)

        # artifacts
        self.scaler = self._load_optional(model_dir / "scaler.pkl", kind="scaler")
        self.pca    = self._load_optional(model_dir / "pca.pkl",    kind="pca")
        self.resid  = self._load_optional(model_dir / "residualizer.pkl", kind="residualizer")

        self.training_genes = self._load_training_genes(model_dir, genes_file)
        self.id2name = load_id2name(model_dir)
        self.threshold = self._load_threshold(model_dir / "threshold.pkl", threshold_override)

        log.info(f"[ARTIFACTS] genes={len(self.training_genes)} | "
                 f"pca={getattr(self.pca, 'n_components_', None)} | "
                 f"threshold={self.threshold:.2f}")

        if self.pca is not None and hasattr(self.model, "input_shape"):
            in_dim = int(self.model.input_shape[-1])
            if int(self.pca.n_components_) != in_dim:
                raise ValueError(f"PCA components ({self.pca.n_components_}) != model input dim ({in_dim})")

    def _load_model(self, p: Path):
        log.info(f"[MODEL] Loading {p}")
        m = tf.keras.models.load_model(p, compile=False)
        m.compile(optimizer="adam", loss="binary_crossentropy")
        log.info(f"[MODEL] Input shape: {m.input_shape}")
        return m

    def _load_optional(self, p: Path, kind: str):
        if p.exists():
            try:
                obj = joblib.load(p)
                log.info(f"[LOAD] {kind} loaded from {p}")
                return obj
            except Exception as e:
                log.warning(f"[LOAD] could not load {kind} from {p}: {e}. Proceeding without it.")
                return None
        log.warning(f"[LOAD] {kind} not found at {p}")
        return None

    def _load_training_genes(self, model_dir: Path, override: Optional[Path]) -> List[str]:
        # 1) explicit override
        if override:
            override = override.resolve()
            if override.exists():
                with open(override) as f:
                    genes = [line.strip().upper() for line in f if line.strip()]
                log.info(f"[GENES] Loaded {len(genes)} genes from --genes-file {override}")
                return genes
            else:
                raise FileNotFoundError(f"--genes-file not found: {override}")
        # 2) next to the model
        for fname in ["genes_training_order.txt", "common_genes.txt", "gene_names.txt"]:
            p = model_dir / fname
            if p.exists():
                with open(p) as f:
                    genes = [line.strip().upper() for line in f if line.strip()]
                log.info(f"[GENES] Loaded {len(genes)} genes from {p}")
                return genes
        # 3) try sibling metadata dir
        meta = model_dir.parent / "metadata"
        for fname in ["genes_training_order.txt", "common_genes.txt", "gene_names.txt"]:
            p = meta / fname
            if p.exists():
                with open(p) as f:
                    genes = [line.strip().upper() for line in f if line.strip()]
                log.info(f"[GENES] Loaded {len(genes)} genes from {p}")
                return genes
        raise FileNotFoundError(
            "Training gene order file not found. "
            "Place 'genes_training_order.txt' (or 'common_genes.txt') in model-dir or model-dir/../metadata, "
            "or provide --genes-file."
        )

    def _load_threshold(self, p: Path, override: Optional[float]) -> float:
        if override is not None:
            log.info(f"[THRESH] Using CLI override threshold={override:.3f}")
            return float(override)
        if p.exists():
            try:
                thr = joblib.load(p)
                t = float(thr)
                log.info(f"[THRESH] Loaded threshold={t:.3f} from {p}")
                return t
            except Exception as e:
                log.warning(f"[THRESH] Failed to load {p}: {e}; using default 0.5")
        else:
            log.warning(f"[THRESH] {p} not found; using default 0.5")
        return 0.5

    def _preprocess(self, X_cellsxgenes: np.ndarray) -> np.ndarray:
        X = scrna_cpm_log1p(X_cellsxgenes)
        # intercept-only residual correction (optional)
        if getattr(self, "resid", None) is not None and getattr(self.resid, "B", None) is not None:
            try:
                b0 = np.asarray(self.resid.B)[0]
                if b0.shape[0] == X.shape[1]:
                    X = X - b0
            except Exception as e:
                log.warning(f"[RESID] Skipping intercept correction: {e}")
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X).astype(np.float32)
        return X

    def predict_matrix(self, X_cellsxgenes: np.ndarray) -> np.ndarray:
        Xp = self._preprocess(X_cellsxgenes)
        probs = self.model.predict(Xp, verbose=0).ravel().astype(np.float32)
        return probs

    def process_sample(self, sample_path: Path, sample_name: str) -> Tuple[pd.DataFrame, dict]:
        # loader selection
        if sample_path.is_dir():
            X, genes, cell_ids = load_10x_dir(sample_path, max_cells=ARGS.max_cells)
        elif sample_path.suffix in [".csv", ".tsv"]:
            X, genes, cell_ids = load_dense_csv(sample_path)
        elif sample_path.suffixes[-2:] == [".mtx", ".gz"] or sample_path.suffix == ".mtx":
            X, genes, cell_ids = load_loose_triplet_matrix_file(sample_path, max_cells=ARGS.max_cells)
        else:
            # best-effort: try loose triplet by pattern
            X, genes, cell_ids = load_loose_triplet_matrix_file(sample_path, max_cells=ARGS.max_cells)

        log.info(f"[LOAD] {sample_name}: cells={X.shape[0]:,} genes={X.shape[1]:,}")

        # align (with duplicate merge)
        X_aligned, found, n_merged = align_to_training_genes(X, genes, self.training_genes, self.id2name)
        pct = 100.0 * found / max(1, len(self.training_genes))
        msg_m = f", merged_duplicates={n_merged}" if n_merged > 0 else ""
        log.info(f"[ALIGN] {sample_name}: matched={found}/{len(self.training_genes)} genes ({pct:.1f}%)" + msg_m)

        # predict
        probs = self.predict_matrix(X_aligned)
        preds = (probs >= self.threshold).astype(np.int32)
        total = len(preds)
        pos = int(preds.sum())
        neg = int(total - pos)
        pos_pct = (pos / total * 100.0) if total else 0.0
        neg_pct = (neg / total * 100.0) if total else 0.0

        if total > 0:
            qs = np.quantile(probs, [0.1, 0.5, 0.9, 0.95, 0.99])
            log.info(f"[SCORES] {sample_name}: q10={qs[0]:.3f} q50={qs[1]:.3f} q90={qs[2]:.3f} q95={qs[3]:.3f} q99={qs[4]:.3f}")

        df_out = pd.DataFrame({
            "cell_id": list(map(str, cell_ids[:total])),
            "prob_persister": probs,
            "pred_label": np.where(preds == 1, "Persister", "Non-Persister")
        })

        summary = {
            "sample": sample_name,
            "threshold": float(self.threshold),
            "cells": int(total),
            "merged_duplicates": int(n_merged),
            "matched_genes": int(found),
            "total_training_genes": int(len(self.training_genes)),
            "persister_count": pos,
            "non_persister_count": neg,
            "persister_pct": float(pos_pct),
            "non_persister_pct": float(neg_pct),
            "mean_prob": float(np.mean(probs)) if total else 0.0,
            "std_prob": float(np.std(probs)) if total else 0.0
        }
        log.info(f"[RESULT] {sample_name}: Persister {pos} ({pos_pct:.1f}%) | "
                 f"Non-Persister {neg} ({neg_pct:.1f}%) | thr={self.threshold:.2f}")

        return df_out, summary


# -------------------------
# Discovery
# -------------------------
def discover_10x_dirs(root: Path) -> List[Tuple[Path, str]]:
    """
    Find all .../filtered_feature_bc_matrix folders.
    Sample name is derived by climbing up from that folder until the directory
    name is NOT one of {'filtered_feature_bc_matrix','outs','count'}.
    """
    out: List[Tuple[Path, str]] = []
    if not root.exists():
        return out
    SKIP = {"filtered_feature_bc_matrix", "outs", "count"}
    for p in root.rglob("filtered_feature_bc_matrix"):
        if not p.is_dir():
            continue
        parent = p.parent
        while parent.name in SKIP and parent.parent != parent:
            parent = parent.parent
        name = parent.name
        out.append((p, name))
    return out

def discover_healthy_loose_triplets(root: Path) -> List[Tuple[Path, str]]:
    """
    Find GSE120221-style *_matrix_*.mtx.gz files and use each as a sample.
    Sample name = prefix before '_matrix_' (e.g., GSM3396161_HEALTHY).
    """
    out: List[Tuple[Path, str]] = []
    if not root.exists():
        return out
    for mtx in sorted(root.glob("*_matrix_*.mtx.gz")):
        m = re.match(r"^(.*)_matrix_.*\.mtx(\.gz)?$", mtx.name)
        if not m:
            continue
        prefix = m.group(1)  # "GSM3396161"
        out.append((mtx, f"{prefix}_HEALTHY"))
    return out

def build_default_samples() -> List[Tuple[Path, str]]:
    gsm_base = Path("/scratch/project_2010751/GSE123902_RAW")
    fixed: List[Tuple[Path, str]] = [
        (gsm_base / "GSM3516666_MSK_LX675_NORMAL_dense.csv", "GSM3516666_NORMAL"),
        (gsm_base / "GSM3516665_MSK_LX675_PRIMARY_TUMOUR_dense.csv", "GSM3516665_PRIMARY"),
        (gsm_base / "GSM3516667_MSK_LX676_PRIMARY_TUMOUR_dense.csv", "GSM3516667_PRIMARY"),
        (gsm_base / "GSM3516664_MSK_LX666_METASTASIS_dense.csv", "GSM3516664_METASTASIS"),
        (gsm_base / "GSM3516668_MSK_LX255B_METASTASIS_dense.csv", "GSM3516668_METASTASIS"),
        (Path("/scratch/project_2010376/scRNAseq/FH_5897_2/filtered_feature_bc_matrix"), "FH_5897_2"),
        (Path("/scratch/project_2010376/scRNAseq/FH_6333_2/filtered_feature_bc_matrix"), "FH_6333_2"),
    ]
    return fixed


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Persister inference runner")
    ap.add_argument("--model-dir",   type=Path, required=True, help="Directory containing model + artifacts")
    ap.add_argument("--genes-file",  type=Path, default=None,  help="Explicit path to training gene list")
    ap.add_argument("--threshold",   type=float, default=None, help="Override threshold (e.g., 0.31)")
    ap.add_argument("--out-dir",     type=Path, required=True, help="Where to write predictions/summary")
    ap.add_argument("--aml-root",    type=Path, default=Path("/scratch/project_2010751/AML_scRNA_decrypted"),
                    help="Root to discover AML 10x 'filtered_feature_bc_matrix' folders")
    ap.add_argument("--healthy-root",type=Path, default=Path("/scratch/project_2010751/GSE120221_RAW"),
                    help="Root with GSE120221 loose triplets")
    ap.add_argument("--no-discover", action="store_true", help="Disable auto-discovery under roots")
    ap.add_argument("--manifest",    type=Path, default=None, help="Optional CSV with columns: path,sample_name")
    ap.add_argument("--max-cells",   type=int, default=20000, help="Cap cells/sample before densifying (0=all)")
    return ap.parse_args()

# Make args global in this module so loader can see max-cells
ARGS = None

def main():
    global ARGS
    ARGS = parse_args()

    model_dir = ARGS.model_dir.resolve()
    out_dir = ARGS.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting inference at {pd.Timestamp.now(tz=None)}")
    log.info(f"MODEL DIR: {model_dir}")
    log.info(f"OUT DIR  : {out_dir}")

    infer = PersisterInference(model_dir, genes_file=ARGS.genes_file, threshold_override=ARGS.threshold)

    # Build sample list
    samples: List[Tuple[Path, str]] = []
    samples.extend(build_default_samples())

    if not ARGS.no_discover:
        discovered_aml = discover_10x_dirs(ARGS.aml_root)
        if discovered_aml:
            log.info(f"[DISCOVER] Found {len(discovered_aml)} AML 10x datasets under {ARGS.aml_root}")
            samples.extend(discovered_aml)

        discovered_healthy = discover_healthy_loose_triplets(ARGS.healthy_root)
        if discovered_healthy:
            log.info(f"[DISCOVER] Found {len(discovered_healthy)} healthy GSE120221 triplets under {ARGS.healthy_root}")
            samples.extend(discovered_healthy)

    if ARGS.manifest and ARGS.manifest.exists():
        try:
            dfm = pd.read_csv(ARGS.manifest)
            for _, r in dfm.iterrows():
                p = Path(str(r["path"])).expanduser()
                n = str(r.get("sample_name", p.name))
                samples.append((p, n))
            log.info(f"[MANIFEST] Added {len(dfm)} entries from {ARGS.manifest}")
        except Exception as e:
            log.warning(f"[MANIFEST] Failed to read {ARGS.manifest}: {e}")

    # Deduplicate by path; sanitize names for file outputs
    uniq = {}
    for spath, sname in samples:
        uniq[str(spath)] = (spath, sanitize(sname))
    samples = list(uniq.values())

    out_summary_path = out_dir / "inference_results_summary.csv"
    all_summaries = []
    for spath, sname in samples:
        log.info("\n" + "=" * 70)
        log.info(f"Processing {sname}")
        log.info("=" * 70)
        spath = spath.resolve()
        if not spath.exists():
            log.warning(f"[SKIP] Missing: {spath}")
            all_summaries.append({
                "sample": sname, "threshold": float(infer.threshold),
                "cells": 0, "merged_duplicates": 0, "matched_genes": 0,
                "total_training_genes": len(infer.training_genes),
                "persister_count": 0, "non_persister_count": 0,
                "persister_pct": 0.0, "non_persister_pct": 0.0,
                "mean_prob": 0.0, "std_prob": 0.0
            })
            continue

        try:
            df_cells, summary = infer.process_sample(spath, sname)
            all_summaries.append(summary)
            if not df_cells.empty:
                df_cells.to_csv(out_dir / f"{sname}_predictions.csv", index=False)
        except Exception as e:
            log.exception(f"[ERROR] {sname}: {e}")
            all_summaries.append({
                "sample": sname, "threshold": float(infer.threshold),
                "cells": 0, "merged_duplicates": 0, "matched_genes": 0,
                "total_training_genes": len(infer.training_genes),
                "persister_count": 0, "non_persister_count": 0,
                "persister_pct": 0.0, "non_persister_pct": 0.0,
                "mean_prob": 0.0, "std_prob": 0.0
            })

    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv(out_summary_path, index=False)
    log.info("\n" + "=" * 80)
    log.info("INFERENCE SUMMARY")
    log.info("=" * 80)
    log.info(f"Saved summary to: {out_summary_path}")
    if not df_sum.empty:
        total_cells = int(df_sum["cells"].sum())
        log.info(f"Total samples: {len(df_sum)} | Total cells: {total_cells:,}")
        valid = df_sum[df_sum["cells"] > 0]
        if not valid.empty:
            log.info(f"Avg persister %: {valid['persister_pct'].mean():.1f}% | "
                     f"Avg mean prob: {valid['mean_prob'].mean():.3f}")
    log.info(f"Inference completed at {pd.Timestamp.now(tz=None).strftime('%a %b %d %H:%M:%S %Y')}")

if __name__ == "__main__":
    main()
