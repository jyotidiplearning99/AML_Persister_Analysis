#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline comparison for the AML drug-tolerance Transformer.

Compares four baselines against the same training corpus, train/test split,
5-fold CV, threshold selection, and held-out evaluation as the Transformer
(production_transformer_13092025.py).

Baselines:
  1. Logistic Regression (L2)         — on 100 PCs, identical features to the Transformer
  2. Random Forest (300 trees)        — on 100 PCs, identical features to the Transformer
  3. LSC17 score (Ng et al. 2016)     — fixed gene-signature score on raw log-CPM expression
  4. Mean-expression baseline         — mean(z-scored 1,000-gene panel)

Outputs:
  - baseline_comparison.csv           — per-model 5-fold CV + held-out test metrics
  - baseline_per_fold.csv             — full per-fold AUROC for each model
  - baseline_predictions.npz          — out-of-fold and test predictions for each model

Usage on Puhti (or local):
    python baselines.py

Place this file in the same directory as production_transformer_13092025.py.
It reuses the same data-loading functions and paths.
"""

import os
import sys
import time
import json
import warnings
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- 
# Reproducibility (match training script settings)
# -----------------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# sklearn / numerical
# -----------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef, precision_score
)
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------------------------------------------------------
# Reuse the EXACT data loaders / preprocessors from the training script
# This is critical: any divergence introduces a confound that a reviewer will
# question. We import from the existing training file.
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent

# Direct copy of helpers from production_transformer_13092025.py
# (kept inline so this script is self-contained and reviewable)
# -----------------------------------------------------------------------------
import re
import gzip
from scipy.io import mmread

def clean_gene_names(names):
    return [re.sub(r"\.\d+$", "", str(g).strip().upper()) for g in names]

def coalesce_duplicate_genes(df, how="sum"):
    df = df.copy()
    df.columns = clean_gene_names(df.columns)
    gb = df.T.groupby(level=0)
    return gb.sum().T if how == "sum" else gb.mean().T

def _read_tsv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [line.rstrip("\n").split("\t") for line in f]

def scrna_cpm_log1p(X):
    X = np.asarray(X, dtype=np.float32)
    X = np.maximum(X, 0.0)
    lib = X.sum(axis=1, keepdims=True)
    np.maximum(lib, 1.0, out=lib)
    X = (X / lib) * 1e4
    return np.log1p(X).astype(np.float32)

def load_10x_dir(matrix_dir: Path):
    mtx = None
    for cand in ["matrix.mtx.gz", "matrix.mtx"]:
        p = matrix_dir / cand
        if p.exists():
            mtx = mmread(str(p)); break
    if mtx is None:
        raise FileNotFoundError(f"No matrix in {matrix_dir}")
    feat = None
    for cand in ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"]:
        p = matrix_dir / cand
        if p.exists():
            feat = _read_tsv(p); break
    if feat is None:
        raise FileNotFoundError(f"No features in {matrix_dir}")
    genes = []
    for r in feat:
        if len(r) >= 2: genes.append(r[1])
        elif len(r) >= 1: genes.append(r[0])
        else: genes.append("UNKNOWN")
    genes = clean_gene_names(genes)
    if hasattr(mtx, "tocsr"):
        mtx = mtx.tocsr().T
    else:
        mtx = np.asarray(mtx).T
    arr = mtx.toarray() if hasattr(mtx, "toarray") else np.asarray(mtx)
    df = pd.DataFrame(arr, columns=genes)
    return coalesce_duplicate_genes(df, how="sum")

def detect_orientation_and_load_csv(path):
    df = None
    for sep in ['\t', ',', None]:
        try:
            df = pd.read_csv(path, index_col=0, sep=sep, engine='python' if sep is None else 'c')
            if df is not None and df.shape[0] > 0 and df.shape[1] > 0:
                break
        except: continue
    if df is None:
        raise ValueError(f"Failed to read {path}")
    df = df.select_dtypes(include=[np.number])
    idx_looks_like_genes = (
        len(df.index) > 100 and
        pd.Series(df.index.astype(str)).str.match(r"^[A-Z]").mean() > 0.5
    )
    if idx_looks_like_genes and df.shape[0] > df.shape[1]:
        df = df.T
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return coalesce_duplicate_genes(df, how="sum")

# -----------------------------------------------------------------------------
# Training-script CONSTANTS (must match)
# -----------------------------------------------------------------------------
FORCE_THRESHOLD_RANGE = True
MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.75
DEFAULT_THRESHOLD = 0.4
TARGET_RECALL = 0.80

GSM_BASE = Path("/scratch/project_2010751/GSE123902_RAW")
AML_ROOT = Path("/scratch/project_2010751/AML_scRNA_decrypted")
N_FOLDS = 5
TEST_SIZE = 0.20

# -----------------------------------------------------------------------------
# Threshold selection — IDENTICAL to find_stable_threshold() in training script
# -----------------------------------------------------------------------------
def find_stable_threshold(y_true, y_prob, target_recall=0.8):
    y_true = np.asarray(y_true)
    y_prob = np.clip(y_prob, 0.001, 0.999)
    thresholds = np.percentile(y_prob, np.arange(10, 91, 5))
    thresholds = np.unique(thresholds)
    best_f1 = -1.0
    best_threshold = DEFAULT_THRESHOLD
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
        if recall < target_recall * 0.9: continue
        if precision < 0.3: continue
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    if FORCE_THRESHOLD_RANGE:
        best_threshold = float(np.clip(best_threshold, MIN_THRESHOLD, MAX_THRESHOLD))
    return best_threshold

# -----------------------------------------------------------------------------
# LSC17 weights — Ng et al. Nature 2016 (Suppl. Table 18)
# Higher score = more LSC-like
# -----------------------------------------------------------------------------
LSC17_WEIGHTS = {
    'DNMT3B':    0.0874,
    'ZBTB46':   -0.0347,
    'NYNRIN':    0.00865,
    'ARHGAP22': -0.0138,
    'LAPTM4B':   0.00582,
    'MMRN1':     0.0258,
    'DPYSL3':    0.0284,
    'KIAA0125':  0.0196,
    'CDK6':     -0.0704,
    'CPXM1':    -0.0258,
    'SOCS2':     0.0271,
    'SMIM24':   -0.0226,    # also published as 'C10orf140'
    'EMP1':      0.0146,
    'NGFRAP1':   0.0465,
    'CD34':      0.0338,
    'AKR1C3':   -0.0402,
    'GPR56':     0.0501,    # also published as 'ADGRG1'
}
LSC17_GENES = list(LSC17_WEIGHTS.keys())

# -----------------------------------------------------------------------------
# Data loading — replicates main() in production_transformer_13092025.py
# -----------------------------------------------------------------------------
def load_training_corpus():
    """Load the same 9-dataset pooled corpus used to train the Transformer."""
    all_data = []
    
    metastasis_files = [
        (GSM_BASE / "GSM3516664_MSK_LX666_METASTASIS_dense.csv", 1, "META_664"),
        (GSM_BASE / "GSM3516668_MSK_LX255B_METASTASIS_dense.csv", 1, "META_668"),
        (GSM_BASE / "GSM3516671_MSK_LX681_METASTASIS_dense.csv", 1, "META_671"),
    ]
    normal_files = [
        (GSM_BASE / "GSM3516666_MSK_LX675_NORMAL_dense.csv", 0, "NORM_666"),
        (GSM_BASE / "GSM3516665_MSK_LX675_PRIMARY_TUMOUR_dense.csv", 0, "PRIM_665"),
        (GSM_BASE / "GSM3516667_MSK_LX676_PRIMARY_TUMOUR_dense.csv", 0, "PRIM_667"),
    ]
    
    print("[DATA] Loading training corpus...", flush=True)
    for filepath, label, sid in metastasis_files + normal_files:
        if filepath.exists():
            try:
                df = detect_orientation_and_load_csv(filepath)
                df_norm = pd.DataFrame(scrna_cpm_log1p(df.values),
                                       columns=df.columns, index=df.index)
                if len(df_norm) > 5000:
                    idx = np.random.choice(len(df_norm), 5000, replace=False)
                    df_norm = df_norm.iloc[idx]
                all_data.append((df_norm, label, sid))
                print(f"  {sid}: {df_norm.shape}", flush=True)
            except Exception as e:
                print(f"  Failed {sid}: {e}", flush=True)
    
    if AML_ROOT.exists():
        aml_dirs = list(AML_ROOT.rglob("filtered_feature_bc_matrix"))[:3]
        for matrix_dir in aml_dirs:
            try:
                name = matrix_dir.parent.parent.name if "outs" in str(matrix_dir) else matrix_dir.parent.name
                sid = f"AML_{name}"
                df = load_10x_dir(matrix_dir)
                df_norm = pd.DataFrame(scrna_cpm_log1p(df.values),
                                       columns=df.columns, index=df.index)
                if len(df_norm) > 5000:
                    idx = np.random.choice(len(df_norm), 5000, replace=False)
                    df_norm = df_norm.iloc[idx]
                all_data.append((df_norm, 1, sid))
                print(f"  {sid}: {df_norm.shape}", flush=True)
            except Exception as e:
                print(f"  Failed AML: {e}", flush=True)
    
    if len(all_data) < 2:
        raise ValueError("Need at least 2 datasets")
    
    # Common gene index (intersection)
    gene_sets = [set(df.columns) for df, _, _ in all_data]
    common_genes = set.intersection(*gene_sets)
    if len(common_genes) < 50:
        gene_counter = Counter()
        for gs in gene_sets:
            gene_counter.update(gs)
        min_presence = max(2, len(gene_sets) // 2)
        common_genes = {g for g, c in gene_counter.items() if c >= min_presence}
    common_genes = sorted(common_genes)
    
    # Combine
    X_list, y_list, sid_list = [], [], []
    for df, label, sid in all_data:
        df_aligned = df.reindex(columns=common_genes).fillna(0.0)
        X_list.append(df_aligned.values)
        y_list.extend([label] * len(df))
        sid_list.extend([sid] * len(df))
    
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    sids = np.array(sid_list)
    
    print(f"[DATA] Combined: X={X.shape}, classes={np.bincount(y)}", flush=True)
    return X, y, sids, common_genes

# -----------------------------------------------------------------------------
# Metric helper — IDENTICAL set of metrics to the Transformer test report
# -----------------------------------------------------------------------------
def evaluate_predictions(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'auc_roc':            float(roc_auc_score(y_true, y_prob)),
        'auc_pr':             float(average_precision_score(y_true, y_prob)),
        'balanced_accuracy':  float(balanced_accuracy_score(y_true, y_pred)),
        'mcc':                float(matthews_corrcoef(y_true, y_pred)),
        'recall':             float(recall_score(y_true, y_pred, zero_division=0)),
        'precision':          float(precision_score(y_true, y_pred, zero_division=0)),
        'f1':                 float(f1_score(y_true, y_pred, zero_division=0)),
        'tn': int(cm[0,0]) if cm.shape == (2,2) else 0,
        'fp': int(cm[0,1]) if cm.shape == (2,2) else 0,
        'fn': int(cm[1,0]) if cm.shape == (2,2) else 0,
        'tp': int(cm[1,1]) if cm.shape == (2,2) else 0,
    }

# -----------------------------------------------------------------------------
# Baseline 1+2: classifiers on 100 PCs (matches Transformer's input space)
# -----------------------------------------------------------------------------
def cv_classifier_on_pcs(name, model_factory, X_train, y_train, X_test, y_test, n_pcs=100):
    """5-fold CV + held-out test for a classifier fed with 100-PC features."""
    print(f"\n[CV] {name}", flush=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []
    oof_probs = np.zeros(len(y_train), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
        scaler = StandardScaler().fit(X_train[tr])
        Xtr_s = scaler.transform(X_train[tr])
        Xva_s = scaler.transform(X_train[va])
        n_comp = min(n_pcs, Xtr_s.shape[1] - 1, Xtr_s.shape[0] - 1)
        if n_comp < 2: n_comp = 2
        pca = PCA(n_components=n_comp, random_state=SEED).fit(Xtr_s)
        Xtr_p, Xva_p = pca.transform(Xtr_s), pca.transform(Xva_s)
        clf = model_factory()
        clf.fit(Xtr_p, y_train[tr])
        probs = clf.predict_proba(Xva_p)[:, 1]
        oof_probs[va] = probs
        fauc = roc_auc_score(y_train[va], probs)
        fold_aucs.append(fauc)
        print(f"  fold {fold}: AUROC = {fauc:.4f}", flush=True)

    threshold = find_stable_threshold(y_train, oof_probs, target_recall=TARGET_RECALL)
    print(f"  threshold (OOF, target_recall={TARGET_RECALL}, range [{MIN_THRESHOLD},{MAX_THRESHOLD}]): {threshold:.3f}",
          flush=True)

    # Held-out test (refit on full train)
    scaler_full = StandardScaler().fit(X_train)
    Xs = scaler_full.transform(X_train)
    Xts = scaler_full.transform(X_test)
    n_comp = min(n_pcs, Xs.shape[1] - 1, Xs.shape[0] - 1)
    pca_full = PCA(n_components=n_comp, random_state=SEED).fit(Xs)
    Xp, Xtp = pca_full.transform(Xs), pca_full.transform(Xts)
    clf_full = model_factory()
    clf_full.fit(Xp, y_train)
    test_probs = clf_full.predict_proba(Xtp)[:, 1]

    test_metrics = evaluate_predictions(y_test, test_probs, threshold)
    print(f"  held-out test: AUROC={test_metrics['auc_roc']:.4f}, "
          f"F1={test_metrics['f1']:.3f}, MCC={test_metrics['mcc']:.3f}", flush=True)

    return {
        'model': name,
        'cv_fold_aucs': [float(a) for a in fold_aucs],
        'cv_mean_auroc': float(np.mean(fold_aucs)),
        'cv_std_auroc':  float(np.std(fold_aucs)),
        'threshold': threshold,
        **{f'test_{k}': v for k, v in test_metrics.items()},
        'oof_probs': oof_probs,
        'test_probs': test_probs,
    }

# -----------------------------------------------------------------------------
# Baseline 3: LSC17 score (Ng et al. 2016)
# -----------------------------------------------------------------------------
def score_lsc17(X_logcpm, gene_index_map):
    """
    Compute LSC17 score per cell.
    X_logcpm: cells × genes (log1p-CPM)
    gene_index_map: dict from gene symbol -> column index
    Returns: probabilities mapped from raw score by min-max normalisation
    """
    score = np.zeros(X_logcpm.shape[0], dtype=np.float32)
    found = []
    missing = []
    for g, w in LSC17_WEIGHTS.items():
        idx = gene_index_map.get(g)
        if idx is not None:
            score += w * X_logcpm[:, idx]
            found.append(g)
        else:
            missing.append(g)
    print(f"[LSC17] Found {len(found)}/{len(LSC17_WEIGHTS)} genes; missing: {missing}", flush=True)
    # Convert raw score to a probability-like number via min-max in [0.001, 0.999]
    # so that find_stable_threshold can operate on it. Note LSC17 is not calibrated;
    # the threshold-selection is matched to the Transformer's protocol.
    smin, smax = float(score.min()), float(score.max())
    if smax > smin:
        prob = (score - smin) / (smax - smin)
    else:
        prob = np.full_like(score, 0.5)
    return np.clip(prob, 0.001, 0.999), len(found), missing

def cv_lsc17(X_train_raw_logcpm, y_train, X_test_raw_logcpm, y_test, gene_index_map):
    """LSC17: not a trained classifier; produce score on full train + test directly."""
    print(f"\n[CV] LSC17 score (Ng et al. 2016)", flush=True)
    # OOF: LSC17 is deterministic, so OOF == train score
    train_score, n_found, missing = score_lsc17(X_train_raw_logcpm, gene_index_map)
    test_score, _, _              = score_lsc17(X_test_raw_logcpm,  gene_index_map)

    # CV is symbolic only — same score in every fold; we still report fold AUROCs for fairness.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []
    for fold, (tr, va) in enumerate(skf.split(X_train_raw_logcpm, y_train), 1):
        fauc = roc_auc_score(y_train[va], train_score[va])
        fold_aucs.append(fauc)
        print(f"  fold {fold}: AUROC = {fauc:.4f}", flush=True)

    threshold = find_stable_threshold(y_train, train_score, target_recall=TARGET_RECALL)
    print(f"  threshold: {threshold:.3f}", flush=True)
    test_metrics = evaluate_predictions(y_test, test_score, threshold)
    print(f"  held-out test: AUROC={test_metrics['auc_roc']:.4f}, "
          f"F1={test_metrics['f1']:.3f}", flush=True)

    return {
        'model': f'LSC17 (Ng 2016) [{n_found}/17 genes found]',
        'cv_fold_aucs': [float(a) for a in fold_aucs],
        'cv_mean_auroc': float(np.mean(fold_aucs)),
        'cv_std_auroc':  float(np.std(fold_aucs)),
        'threshold': threshold,
        'genes_found': n_found,
        'genes_missing': missing,
        **{f'test_{k}': v for k, v in test_metrics.items()},
        'oof_probs': train_score,
        'test_probs': test_score,
    }

# -----------------------------------------------------------------------------
# Baseline 4: mean-expression of the 1,000-gene panel
# -----------------------------------------------------------------------------
def cv_mean_expression(X_train_raw_logcpm, y_train, X_test_raw_logcpm, y_test):
    """Mean of z-scored panel as a single 'score'. No model fit beyond scaling."""
    print(f"\n[CV] Mean-expression baseline (whole training panel)", flush=True)
    scaler = StandardScaler().fit(X_train_raw_logcpm)
    Xtr_s = scaler.transform(X_train_raw_logcpm)
    Xte_s = scaler.transform(X_test_raw_logcpm)
    train_score = Xtr_s.mean(axis=1)
    test_score  = Xte_s.mean(axis=1)
    # Map to [0.001, 0.999] for threshold selection compatibility
    def to_prob(s):
        smin, smax = float(s.min()), float(s.max())
        if smax > smin:
            return np.clip((s - smin) / (smax - smin), 0.001, 0.999)
        return np.full_like(s, 0.5)
    train_prob = to_prob(train_score)
    test_prob  = to_prob(test_score)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []
    for fold, (tr, va) in enumerate(skf.split(X_train_raw_logcpm, y_train), 1):
        fauc = roc_auc_score(y_train[va], train_prob[va])
        fold_aucs.append(fauc)
        print(f"  fold {fold}: AUROC = {fauc:.4f}", flush=True)

    threshold = find_stable_threshold(y_train, train_prob, target_recall=TARGET_RECALL)
    print(f"  threshold: {threshold:.3f}", flush=True)
    test_metrics = evaluate_predictions(y_test, test_prob, threshold)
    print(f"  held-out test: AUROC={test_metrics['auc_roc']:.4f}, F1={test_metrics['f1']:.3f}",
          flush=True)

    return {
        'model': 'Mean-expression (whole training panel)',
        'cv_fold_aucs': [float(a) for a in fold_aucs],
        'cv_mean_auroc': float(np.mean(fold_aucs)),
        'cv_std_auroc':  float(np.std(fold_aucs)),
        'threshold': threshold,
        **{f'test_{k}': v for k, v in test_metrics.items()},
        'oof_probs': train_prob,
        'test_probs': test_prob,
    }

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("="*70)
    print(f"Baseline comparison vs Transformer | start {time.ctime()}")
    print("="*70)

    # 1) Load identical training corpus as the Transformer
    X, y, sids, gene_list = load_training_corpus()
    gene_index_map = {g: i for i, g in enumerate(gene_list)}
    print(f"[INFO] Common gene index: {len(gene_list)} genes")

    # 2) Identical train/test split (seed 42, stratified, 20% test)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"[SPLIT] Train: {len(y_train)} (classes {np.bincount(y_train)}), "
          f"Test: {len(y_test)} (classes {np.bincount(y_test)})")

    results = []

    # ---- 1. Logistic regression (L2) on 100 PCs ----
    res_lr = cv_classifier_on_pcs(
        'Logistic Regression (L2, balanced)',
        lambda: LogisticRegression(C=1.0, max_iter=2000,
                                    class_weight='balanced',
                                    solver='lbfgs', random_state=SEED),
        X_train, y_train, X_test, y_test, n_pcs=100,
    )
    results.append(res_lr)

    # ---- 2. Random forest on 100 PCs ----
    res_rf = cv_classifier_on_pcs(
        'Random Forest (300 trees, balanced)',
        lambda: RandomForestClassifier(n_estimators=300, max_depth=None,
                                        class_weight='balanced',
                                        random_state=SEED, n_jobs=-1),
        X_train, y_train, X_test, y_test, n_pcs=100,
    )
    results.append(res_rf)

    # ---- 3. LSC17 score on raw log-CPM ----
    res_lsc = cv_lsc17(X_train, y_train, X_test, y_test, gene_index_map)
    results.append(res_lsc)

    # ---- 4. Mean-expression baseline ----
    res_mean = cv_mean_expression(X_train, y_train, X_test, y_test)
    results.append(res_mean)

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    summary_rows = []
    perfold_rows = []
    pred_arrays = {}
    for r in results:
        row = {k: v for k, v in r.items() if k not in ('cv_fold_aucs', 'oof_probs', 'test_probs')}
        summary_rows.append(row)
        for f, a in enumerate(r['cv_fold_aucs'], 1):
            perfold_rows.append({'model': r['model'], 'fold': f, 'auroc': a})
        pred_arrays[f"{r['model']}_oof"]  = r['oof_probs']
        pred_arrays[f"{r['model']}_test"] = r['test_probs']

    pd.DataFrame(summary_rows).to_csv("baseline_comparison.csv", index=False)
    pd.DataFrame(perfold_rows).to_csv("baseline_per_fold.csv", index=False)
    np.savez("baseline_predictions.npz", y_train=y_train, y_test=y_test, **pred_arrays)

    # -------------------------------------------------------------------------
    # Print final summary table
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("BASELINE COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<46s} | {'CV mean':>9s} | {'Test AUC':>9s} | {'F1':>6s} | {'MCC':>6s}")
    print("-"*88)
    for r in results:
        print(f"{r['model'][:46]:<46s} | "
              f"{r['cv_mean_auroc']:.3f}±{r['cv_std_auroc']:.3f}".ljust(11) + " | " +
              f"{r['test_auc_roc']:>9.3f} | {r['test_f1']:>6.3f} | {r['test_mcc']:>+6.3f}")
    print("-"*88)
    print("\nReference (Transformer, from canonical metadata.json):")
    print(f"{'Feature-Token Transformer (canonical)':<46s} |  0.936±0.014 | "
          f"{0.941:>9.3f} | {0.895:>6.3f} | {0.696:>+6.3f}")
    print("="*70)
    print(f"Saved: baseline_comparison.csv, baseline_per_fold.csv, baseline_predictions.npz")
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()