#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persister Inference (1000-gene reduced) — v3 + Feature Mask
- Robust 10x loader (filters Gene Expression, auto-detects symbol column)
- Healthy (GSE120221) grouping per GSM -> one label: GSMxxxxxxx_HEALTHY
- AML 10x naming from meaningful ancestor directory
- Coverage guard (abort on 0 matched genes)
- Optional healthy-based threshold calibration (target FPR)
- NEW: Feature-mask ablation (keep a subset of genes; others zeroed or imputed)

USAGE SNIPPETS
--------------
# Baseline (no mask)
python inference_persister_v3_masked.py \
  --model-dir /.../reduced_model_distilled \
  --out-dir   /.../results/inference_reduced \
  --aml-root /.../AML_scRNA_decrypted \
  --healthy-root /.../GSE120221_RAW \
  --use-reduced

# Ablation with Top-100 genes (pre-CPM masking; recommended default)
python inference_persister_v3_masked.py \
  --model-dir /.../reduced_model_distilled \
  --out-dir   /.../results/inference_reduced_top100 \
  --aml-root /.../AML_scRNA_decrypted \
  --healthy-root /.../GSE120221_RAW \
  --use-reduced \
  --feature-mask /.../depmap_refined/top100.txt \
  --mask-stage pre_cpm \
  --calibrate-healthy --target-fpr 0.05 --save-threshold-to-modeldir
"""

import os, re, sys, gzip, logging, warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from scipy.io import mmread
from scipy import sparse

# -------------------------
# Setup & logging
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
# Utils
# -------------------------
ENSEMBL_PAT = re.compile(r"^ENSG\d+", re.IGNORECASE)

def clean_gene_names(names: List[str]) -> List[str]:
    out = []
    for g in names:
        s = str(g).strip()
        if ENSEMBL_PAT.match(s):
            s = s.split(".")[0]
        out.append(s.upper())
    return out

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
# Feature parsing helpers
# -------------------------
def parse_features_tsv(path: Path) -> Tuple[List[str], Optional[List[str]], Optional[List[str]]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    c0, c1, c2 = [], [], []
    with opener(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            if len(parts) >= 1: c0.append(parts[0])
            if len(parts) >= 2: c1.append(parts[1])
            if len(parts) >= 3: c2.append(parts[2])
    if not c2: c2 = None
    if not c1: c1 = None
    return c0, c1, c2

def choose_symbol_column(col0: List[str], col1: Optional[List[str]]) -> int:
    def ensg_frac(lst):
        if not lst: return 1.0
        n = len(lst)
        if n == 0: return 1.0
        return sum(1 for x in lst if ENSEMBL_PAT.match(str(x))) / n
    f0 = ensg_frac(col0)
    f1 = ensg_frac(col1) if col1 is not None else 1.0
    return 0 if f0 < f1 else 1

# -------------------------
# Loaders
# -------------------------
def load_10x_dir(matrix_dir: Path, max_cells: Optional[int] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    # matrix
    mtx = None
    for cand in ["matrix.mtx.gz", "matrix.mtx"]:
        p = matrix_dir / cand
        if p.exists():
            mtx = mmread(str(p))
            break
    if mtx is None:
        raise FileNotFoundError(f"No matrix.mtx(.gz) in {matrix_dir}")

    # features
    feat_path = None
    for cand in ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"]:
        p = matrix_dir / cand
        if p.exists():
            feat_path = p
            break
    if feat_path is None:
        raise FileNotFoundError(f"No features/genes file in {matrix_dir}")

    c0, c1, c2 = parse_features_tsv(feat_path)

    # Filter Gene Expression
    keep_idx = list(range(len(c0)))
    if c2 is not None:
        keep_idx = [i for i, t in enumerate(c2) if str(t).strip().lower() == "gene expression"]

    col_choice = choose_symbol_column([c0[i] for i in keep_idx], [c1[i] for i in keep_idx] if c1 else None)
    raw_symbols = [ (c0 if col_choice==0 else c1)[i] for i in keep_idx ]
    genes = clean_gene_names(raw_symbols)

    # barcodes
    barcodes = None
    for cand in ["barcodes.tsv.gz", "barcodes.tsv"]:
        p = matrix_dir / cand
        if p.exists():
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rt") as f:
                barcodes = [line.strip() for line in f]
            break
    if barcodes is None:
        n_cells = mtx.shape[1] if sparse.isspmatrix(mtx) else np.asarray(mtx).shape[1]
        barcodes = [f"CELL_{i}" for i in range(n_cells)]

    # convert
    if sparse.isspmatrix(mtx):
        mtx = mtx.tocsr()  # genes x cells
        mtx = mtx[keep_idx, :]
        if max_cells and max_cells > 0 and mtx.shape[1] > max_cells:
            keep = np.sort(np.random.default_rng(SEED).choice(mtx.shape[1], max_cells, replace=False))
            mtx = mtx[:, keep]
            barcodes = [barcodes[i] for i in keep]
        X = mtx.T.astype(np.float32).toarray()
    else:
        M = np.asarray(mtx)
        M = M[keep_idx, :]
        if max_cells and max_cells > 0 and M.shape[1] > max_cells:
            keep = np.sort(np.random.default_rng(SEED).choice(M.shape[1], max_cells, replace=False))
            M = M[:, keep]
            barcodes = [barcodes[i] for i in keep]
        X = M.T.astype(np.float32)

    return X, genes, barcodes

def discover_aml_10x(aml_root: Path) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    if not aml_root or not aml_root.exists():
        return out
    paths = os.popen(
        f"find '{aml_root}' -type f -path '*/filtered_feature_bc_matrix/matrix.mtx*' 2>/dev/null"
    ).read().splitlines()
    dirs = sorted(set(Path(p).parent for p in map(Path, paths)))

    def meaningful_label(d: Path) -> str:
        generic = {"filtered_feature_bc_matrix", "count", "outs"}
        cur = d
        for _ in range(6):
            if cur.name not in generic and cur.name:
                name = cur.name
                break
            cur = cur.parent
        else:
            name = d.parent.name
        return name

    for d in dirs:
        label = meaningful_label(d)
        out.append((d, label))
    log.info(f"[DISCOVER] AML datasets seen: {len(out)} under {aml_root}")
    return out

def discover_healthy_gse120221_grouped(healthy_root: Path) -> List[Tuple[List[Path], str]]:
    out: List[Tuple[List[Path], str]] = []
    if not healthy_root or not healthy_root.exists():
        return out

    bar_pat = re.compile(r"^(GSM\d+)_barcodes_([A-Za-z0-9]+)\.tsv\.gz$")
    gen_pat = re.compile(r"^(GSM\d+)_genes_([A-Za-z0-9]+)\.tsv\.gz$")
    mtx_pat = re.compile(r"^(GSM\d+)_matrix_([A-Za-z0-9]+)\.mtx\.gz$")

    by_gsm: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for fname in os.listdir(healthy_root):
        m = bar_pat.match(fname)
        if m:
            gsm, tag = m.groups()
            by_gsm.setdefault(gsm, {}).setdefault(tag, {})["barcodes"] = healthy_root / fname
            continue
        m = gen_pat.match(fname)
        if m:
            gsm, tag = m.groups()
            by_gsm.setdefault(gsm, {}).setdefault(tag, {})["genes"] = healthy_root / fname
            continue
        m = mtx_pat.match(fname)
        if m:
            gsm, tag = m.groups()
            by_gsm.setdefault(gsm, {}).setdefault(tag, {})["matrix"] = healthy_root / fname
            continue

    for gsm, tagmap in sorted(by_gsm.items()):
        triplets = []
        for tag, files in tagmap.items():
            if {"barcodes","genes","matrix"} <= files.keys():
                triplets.append(files["matrix"])
                triplets.append(files["genes"])
                triplets.append(files["barcodes"])
        if triplets:
            out.append((triplets, f"{gsm}_HEALTHY"))

    log.info(f"[DISCOVER] Healthy GSM groups: {len(out)}")
    return out

def load_gse120221_gsm_merged(triplet_paths: List[Path], max_cells: Optional[int] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    bar_pat = re.compile(r"^(GSM\d+)_barcodes_([A-Za-z0-9]+)\.tsv\.gz$")
    gen_pat = re.compile(r"^(GSM\d+)_genes_([A-Za-z0-9]+)\.tsv\.gz$")
    mtx_pat = re.compile(r"^(GSM\d+)_matrix_([A-Za-z0-9]+)\.mtx\.gz$")

    per_tag: Dict[str, Dict[str, Path]] = {}
    for p in triplet_paths:
        fname = p.name
        m = bar_pat.match(fname) or gen_pat.match(fname) or mtx_pat.match(fname)
        if not m:
            continue
        gsm, tag = m.groups()
        if fname.startswith(gsm + "_barcodes_"):
            per_tag.setdefault(tag, {})["barcodes"] = p
        elif fname.startswith(gsm + "_genes_"):
            per_tag.setdefault(tag, {})["genes"] = p
        elif fname.startswith(gsm + "_matrix_"):
            per_tag.setdefault(tag, {})["matrix"] = p

    X_all, genes_ref, bar_all = [], None, []
    for tag, files in sorted(per_tag.items()):
        if {"barcodes","genes","matrix"} - files.keys():
            continue
        mtx = mmread(str(files["matrix"]))

        c0, c1, c2 = parse_features_tsv(files["genes"])
        keep_idx = list(range(len(c0)))
        if c2:
            keep_idx = [i for i, t in enumerate(c2) if str(t).strip().lower() == "gene expression"]
        col_choice = choose_symbol_column([c0[i] for i in keep_idx], [c1[i] for i in keep_idx] if c1 else None)
        raw_symbols = [ (c0 if col_choice==0 else c1)[i] for i in keep_idx ]
        genes = clean_gene_names(raw_symbols)

        opener = gzip.open if str(files["barcodes"]).endswith(".gz") else open
        with opener(files["barcodes"], "rt") as f:
            barcodes = [line.strip() for line in f]

        if sparse.isspmatrix(mtx):
            mtx = mtx.tocsr()[keep_idx, :]
            X = mtx.T.astype(np.float32).toarray()
        else:
            M = np.asarray(mtx)[keep_idx, :]
            X = M.T.astype(np.float32)

        if genes_ref is None:
            genes_ref = genes
        else:
            common = {g: i for i, g in enumerate(genes)}
            idx = [common.get(g, -1) for g in genes_ref]
            valid = np.array(idx) >= 0
            if not np.all(valid):
                new_ref = [g for g, ok in zip(genes_ref, valid) if ok]
                X = X[:, np.array([common[g] for g in new_ref], dtype=int)]
                for k in range(len(X_all)):
                    Xi = X_all[k]
                    if Xi.shape[1] != len(genes_ref):
                        continue
                    keep = np.array([g in common for g in genes_ref])
                    X_all[k] = Xi[:, keep]
                genes_ref = new_ref

        X_all.append(X)
        bar_all.extend(barcodes[:X.shape[0]])

    if not X_all or genes_ref is None:
        raise RuntimeError("No complete healthy triplets to merge for this GSM.")

    X_merged = np.vstack(X_all)
    return X_merged, genes_ref, bar_all

# -------------------------
# Model wrapper
# -------------------------
class PersisterInference:
    def __init__(self, model_dir: Path, genes_file: Optional[Path], threshold_override: Optional[float],
                 use_reduced: bool = True, feature_mask_genes: Optional[List[str]] = None,
                 mask_stage: str = "pre_cpm"):
        self.model_dir = model_dir
        self.use_reduced = use_reduced

        # model
        self.model = tf.keras.models.load_model(self._find_model_file(model_dir), compile=False)
        self.model.compile(optimizer="adam", loss="binary_crossentropy")

        # preprocessing
        if use_reduced:
            self.scaler = self._load_obj(model_dir / "scaler_reduced.pkl", "scaler")
            self.pca    = self._load_obj(model_dir / "pca_reduced.pkl", "pca")
            self.thr_path = model_dir / "threshold_reduced.pkl"
        else:
            self.scaler = self._load_obj(model_dir / "scaler.pkl", "scaler")
            self.pca    = self._load_obj(model_dir / "pca.pkl", "pca")
            self.thr_path = model_dir / "threshold.pkl"

        # genes (training order)
        self.training_genes = self._load_training_genes(model_dir, genes_file)

        # threshold
        self.threshold = None
        if threshold_override is not None:
            self.threshold = float(threshold_override)
            log.info(f"[THRESH] Using override: {self.threshold:.3f}")
        elif self.thr_path.exists():
            try:
                self.threshold = float(joblib.load(self.thr_path))
                log.info(f"[THRESH] Loaded threshold: {self.threshold:.3f}")
            except Exception as e:
                log.warning(f"[THRESH] Failed to load threshold: {e}")

        # feature mask
        self.feature_mask = None
        self.mask_stage = mask_stage
        if feature_mask_genes:
            keep_set = {g.upper() for g in feature_mask_genes}
            self.feature_mask = np.array([g in keep_set for g in self.training_genes], dtype=bool)
            log.info(f"[MASK] Stage={self.mask_stage}; keeping {int(self.feature_mask.sum())}/{len(self.training_genes)} genes")

        log.info(f"[MODEL] Type: {'REDUCED (1000 genes)' if use_reduced else 'FULL'}")
        log.info(f"[MODEL] Genes: {len(self.training_genes)}")
        log.info(f"[MODEL] PCA components: {getattr(self.pca, 'n_components_', None)}")
        if self.threshold is not None:
            log.info(f"[MODEL] Threshold: {self.threshold:.3f}")

    def _find_model_file(self, model_dir: Path) -> Path:
        for fname in ["model_reduced.h5", "final_model.h5", "best_model.keras", "aml_persister_transformer.keras"]:
            p = model_dir / fname
            if p.exists():
                return p
        raise FileNotFoundError(f"No model file found in {model_dir}")

    def _load_obj(self, p: Path, name: str):
        if p.exists():
            try:
                obj = joblib.load(p)
                log.info(f"[LOAD] {name} loaded from {p}")
                return obj
            except Exception as e:
                log.warning(f"[LOAD] Could not load {name}: {e}")
        return None

    def _load_training_genes(self, model_dir: Path, override: Optional[Path]) -> List[str]:
        if override and override.exists():
            with open(override) as f:
                return [line.strip().upper() for line in f if line.strip()]
        for fname in ["selected_genes.txt", "genes_training_order.txt", "common_genes.txt"]:
            p = model_dir / fname
            if p.exists():
                with open(p) as f:
                    return [line.strip().upper() for line in f if line.strip()]
        meta = model_dir.parent / "metadata"
        for fname in ["common_genes.txt", "genes_training_order.txt"]:
            p = meta / fname
            if p.exists():
                with open(p) as f:
                    return [line.strip().upper() for line in f if line.strip()]
        raise FileNotFoundError("No gene list found")

    def align_and_preprocess(self, X: np.ndarray, genes: List[str]) -> Tuple[np.ndarray, int]:
        """Align to training gene order and apply masking + CPM-log1p + scaler + PCA."""
        gene2idx = {g: i for i, g in enumerate(genes)}
        idx_map = np.array([gene2idx.get(g, -1) for g in self.training_genes], dtype=np.int32)
        valid = idx_map >= 0
        coverage = int(valid.sum())

        if coverage == 0:
            raise RuntimeError("Matched 0 / {} training genes — check feature naming (symbol vs Ensembl).".format(len(self.training_genes)))

        # Build aligned matrix in training order
        X_al = np.zeros((X.shape[0], len(self.training_genes)), dtype=np.float32)
        if valid.any():
            X_al[:, valid] = X[:, idx_map[valid]]

        # --- Feature-mask handling ---
        mask = self.feature_mask
        stage = (self.mask_stage or "pre_cpm").lower()

        if mask is not None:
            if stage == "pre_cpm":
                # zero out non-kept genes BEFORE CPM
                X_al[:, ~mask] = 0.0
            # CPM-log1p normalization
            Xn = scrna_cpm_log1p(X_al)
            if mask is not None and stage in ("post_cpm", "impute_scaler_mean"):
                if stage == "post_cpm":
                    Xn[:, ~mask] = 0.0
                elif stage == "impute_scaler_mean":
                    # requires scaler mean_; fallback to zero if unavailable
                    if hasattr(self.scaler, "mean_") and self.scaler.mean_ is not None:
                        # scaler.mean_ is in feature space (training order)
                        mu = np.asarray(self.scaler.mean_, dtype=np.float32)
                        Xn[:, ~mask] = mu[~mask]
                    else:
                        log.warning("[MASK] impute_scaler_mean requested but scaler.mean_ missing; zeroing instead.")
                        Xn[:, ~mask] = 0.0
        else:
            # no mask: standard path
            Xn = scrna_cpm_log1p(X_al)

        # scaler, PCA
        if self.scaler is not None:
            Xn = self.scaler.transform(Xn)
        if self.pca is not None:
            Xn = self.pca.transform(Xn)
        return Xn, coverage

    def predict_probs(self, Xn: np.ndarray, batch: int = 8192) -> np.ndarray:
        probs = np.empty(Xn.shape[0], dtype=np.float32)
        for s in range(0, Xn.shape[0], batch):
            e = min(s + batch, Xn.shape[0])
            probs[s:e] = self.model.predict(Xn[s:e], verbose=0).ravel()
        return probs

# -------------------------
# Threshold calibration (healthy-based)
# -------------------------
def calibrate_threshold_healthy(all_probs: np.ndarray, target_fpr: float, clip_bounds: Tuple[float, float]=(0.25,0.75)) -> float:
    target_fpr = float(np.clip(target_fpr, 0.001, 0.5))
    thr = float(np.quantile(all_probs, 1.0 - target_fpr))
    thr = float(np.clip(thr, clip_bounds[0], clip_bounds[1]))
    log.info(f"[CALIBRATE] Healthy-based threshold @ FPR={target_fpr:.3f} -> {thr:.3f} (clipped to {clip_bounds})")
    return thr

# -------------------------
# Discovery wrapper
# -------------------------
def build_sample_list(aml_root: Optional[Path], healthy_root: Optional[Path]) -> List[Tuple[str, object, str]]:
    samples: List[Tuple[str, object, str]] = []
    if aml_root:
        for d, label in discover_aml_10x(aml_root):
            samples.append(("AML_10X", d, label))
    if healthy_root:
        for files, label in discover_healthy_gse120221_grouped(healthy_root):
            samples.append(("HEALTHY_GSM", files, label))
    log.info(f"[DISCOVER] Total samples to process: {len(samples)}")
    return samples

# -------------------------
# CLI
# -------------------------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Persister Inference (Reduced, 1000 genes) with robust feature handling, healthy calibration, and feature-mask ablation")
    ap.add_argument("--model-dir",   type=Path, required=True)
    ap.add_argument("--out-dir",     type=Path, required=True)
    ap.add_argument("--genes-file",  type=Path, help="Override gene list (default: selected_genes.txt)")
    ap.add_argument("--threshold",   type=float, help="Override threshold (disables calibration/loading)")
    ap.add_argument("--use-reduced", action="store_true", help="Use reduced model (default)")
    ap.add_argument("--use-full",    action="store_true", help="Use full model")
    ap.add_argument("--max-cells",   type=int, default=-1)
    ap.add_argument("--batch-size",  type=int, default=8192)

    ap.add_argument("--aml-root",      type=Path, help="Root with AML 10x datasets")
    ap.add_argument("--healthy-root",  type=Path, help="Root with GSE120221 triplets")

    # Calibration
    ap.add_argument("--calibrate-healthy", action="store_true", help="Calibrate threshold from healthy cohort only")
    ap.add_argument("--target-fpr",   type=float, default=0.05, help="Target FPR on healthy (default 0.05)")
    ap.add_argument("--clip-min",     type=float, default=0.25)
    ap.add_argument("--clip-max",     type=float, default=0.75)
    ap.add_argument("--save-threshold-to-modeldir", action="store_true",
                    help="If set, write calibrated threshold back to model_dir/threshold_reduced.pkl (or threshold.pkl for full)")

    # NEW: Feature-mask ablation
    ap.add_argument("--feature-mask", type=Path,
                    help="Path to text file with ONE SYMBOL PER LINE to KEEP; others will be zeroed/imputed")
    ap.add_argument("--mask-stage", choices=["pre_cpm", "post_cpm", "impute_scaler_mean"],
                    default="pre_cpm", help="When to apply the mask (default: pre_cpm)")

    return ap.parse_args()

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # Which model type?
    use_reduced = True
    if args.use_full:
        use_reduced = False
    elif args.use_reduced:
        use_reduced = True

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting inference at {pd.Timestamp.now()}")
    log.info(f"Model directory: {args.model_dir}")
    log.info(f"Output directory: {out_dir}")

    # Load mask file (optional)
    mask_genes: Optional[List[str]] = None
    if args.feature_mask:
        if not args.feature_mask.exists():
            log.error(f"[MASK] File not found: {args.feature_mask}")
            sys.exit(2)
        with open(args.feature_mask) as f:
            mask_genes = [line.strip() for line in f if line.strip()]
        log.info(f"[MASK] Loaded {len(mask_genes)} genes from {args.feature_mask}")

    infer = PersisterInference(
        model_dir=args.model_dir.resolve(),
        genes_file=args.genes_file,
        threshold_override=args.threshold,
        use_reduced=use_reduced,
        feature_mask_genes=mask_genes,
        mask_stage=args.mask_stage
    )

    # Discover
    samples = build_sample_list(args.aml_root.resolve() if args.aml_root else None,
                                args.healthy_root.resolve() if args.healthy_root else None)

    # Healthy calibration (optional)
    calibrated = False
    if args.calibrate_healthy or (infer.threshold is None and args.healthy_root):
        healthy_probs_all = []
        for kind, payload, label in samples:
            if kind != "HEALTHY_GSM":
                continue
            X, genes, cell_ids = load_gse120221_gsm_merged(payload, max_cells=None if args.max_cells <= 0 else args.max_cells)
            log.info(f"[LOAD] {label}: {X.shape[0]:,} cells × {X.shape[1]:,} genes")
            Xn, cov = infer.align_and_preprocess(X, genes)
            log.info(f"[ALIGN] Gene coverage: {cov}/{len(infer.training_genes)} ({cov/len(infer.training_genes)*100:.1f}%)")
            probs = infer.predict_probs(Xn, batch=args.batch_size)
            healthy_probs_all.append(probs)

        if healthy_probs_all:
            all_probs = np.concatenate(healthy_probs_all, axis=0)
            infer.threshold = calibrate_threshold_healthy(
                all_probs,
                target_fpr=args.target_fpr,
                clip_bounds=(args.clip_min, args.clip_max)
            )
            calibrated = True
            if args.save_threshold_to_modeldir:
                joblib.dump(float(infer.threshold), infer.thr_path)
                log.info(f"[CALIBRATE] Saved calibrated threshold to {infer.thr_path}")
        else:
            log.warning("[CALIBRATE] No healthy samples found; skipping calibration.")

    if infer.threshold is None:
        log.warning("[THRESH] No threshold available; falling back to 0.5")
        infer.threshold = 0.5

    # Run full inference
    summaries = []
    for kind, payload, label in samples:
        try:
            if kind == "AML_10X":
                X, genes, cell_ids = load_10x_dir(payload, max_cells=None if args.max_cells <= 0 else args.max_cells)
            elif kind == "HEALTHY_GSM":
                X, genes, cell_ids = load_gse120221_gsm_merged(payload, max_cells=None if args.max_cells <= 0 else args.max_cells)
            else:
                continue

            log.info(f"[LOAD] {label}: {X.shape[0]:,} cells × {X.shape[1]:,} genes")
            Xn, cov = infer.align_and_preprocess(X, genes)
            log.info(f"[ALIGN] Gene coverage: {cov}/{len(infer.training_genes)} ({cov/len(infer.training_genes)*100:.1f}%)")
            probs = infer.predict_probs(Xn, batch=args.batch_size)
            preds = (probs >= infer.threshold).astype(np.int32)
            pos = int(preds.sum())
            n = int(preds.size)
            pos_pct = (pos / n * 100.0) if n else 0.0
            log.info(f"[RESULT] {label}: {pos} persisters ({pos_pct:.1f}%), {n - pos} non-persisters")

            df_cells = pd.DataFrame({
                "cell_id": cell_ids[:n],
                "prob_persister": probs,
                "pred_label": np.where(preds == 1, "Persister", "Non-Persister")
            })
            df_cells.to_csv(out_dir / f"{sanitize(label)}_predictions.csv", index=False)

            summaries.append({
                "sample": label,
                "cells": n,
                "persister_count": pos,
                "non_persister_count": n - pos,
                "persister_pct": float(pos_pct),
                "mean_prob": float(np.mean(probs)),
                "std_prob": float(np.std(probs)),
                "threshold_used": float(infer.threshold),
                "coverage": cov,
                "mask_stage": infer.mask_stage,
                "kept_genes": int(infer.feature_mask.sum()) if infer.feature_mask is not None else len(infer.training_genes)
            })
        except Exception as e:
            log.error(f"[ERROR] {label}: {e}")

    if summaries:
        df_sum = pd.DataFrame(summaries)
        df_sum.to_csv(out_dir / "inference_results_summary.csv", index=False)
        log.info("\n" + "="*80)
        log.info("INFERENCE COMPLETE")
        log.info("="*80)
        log.info(f"Processed: {len(df_sum)} samples")
        log.info(f"Total cells: {df_sum['cells'].sum():,}")
        log.info(f"Mean persister %: {df_sum['persister_pct'].mean():.1f}%")
        log.info(f"Results saved to: {out_dir}")
        if calibrated:
            log.info(f"Threshold was calibrated from healthy (target FPR={args.target_fpr}).")
    else:
        log.warning("No summaries produced; check discovery and paths.")

if __name__ == "__main__":
    main()
