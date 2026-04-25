#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAIR baseline comparison on the 1,000-gene reduced panel.

Why this exists
---------------
The previous `baselines_external.py` showed:

  Aggregate Spearman r on BeatAML drug-AUC (3,164 patient-drug pairs):
    Transformer (canonical):    +0.482
    Mean-expression:            +0.238
    LSC17:                      +0.128
    Random Forest:              -0.083
    Logistic Regression:        -0.005

But that comparison was UNFAIR for the LR/RF baselines:
    - The Transformer used the 1,000-gene reduced "student" panel, where
      BeatAML had ~99% gene coverage.
    - The LR/RF baselines used the 13,369-gene full training index, where
      BeatAML matched only 877 genes (6.6% coverage).

So the apparent "LR fails to transfer" result might be a gene-coverage
artifact, not a real transferability failure.

This script re-runs the LR/RF/LSC17/mean baselines using the SAME 1,000-gene
panel as the Transformer, ensuring like-for-like comparison.

What it does
------------
1. Loads the 1,000-gene panel from the archive
   (gene_reduction_model_aware/selected_genes_model_aware.txt OR
    reduced_model_distilled/selected_genes.txt — first one found).
2. Loads the same 9-dataset training corpus, but restricts to the 1k panel.
3. Trains LR / RF on the panel (also reports CV/test, comparable to baselines.py).
4. Computes LSC17 and mean-expression scores (panel-restricted).
5. Loads BeatAML expression, restricts to the 1k panel (high coverage).
6. Scores BeatAML with each baseline.
7. Computes per-drug Spearman correlations and aggregate r (3,164 pairs).
8. Reports side-by-side with the canonical Transformer (loaded from existing
   predictions_final.csv).

Outputs:
    fair_baseline_internal.csv         — internal CV/test on 1k panel
    fair_baseline_external_per_drug.csv  — per-drug Spearman r per baseline
    fair_baseline_external_aggregate.csv — aggregate Spearman r per baseline

Usage on Mahti:
    sbatch baselines_external_1k.sbatch
"""

import os, sys, time, warnings, random, re, gzip
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, precision_score, recall_score
from scipy.stats import spearmanr
from scipy.io import mmread

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
GSM_BASE = Path("/scratch/project_2010751/GSE123902_RAW")
AML_ROOT = Path("/scratch/project_2010751/AML_scRNA_decrypted")
PROJECT_ROOT = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis")

# Try both gene-list locations (first found wins)
GENE_LIST_CANDIDATES = [
    PROJECT_ROOT / "gene_reduction_model_aware" / "selected_genes_model_aware.txt",
    PROJECT_ROOT / "reduced_model_distilled" / "selected_genes.txt",
]

BEATAML_EXPR_PATH = PROJECT_ROOT / "results" / "bulk_BeatAML" / "processed" / "expression.csv"
BEATAML_DRUG_PATH = PROJECT_ROOT / "data" / "beataml_drugs" / "processed" / "beataml_key_drugs_auc.csv"
TRANSFORMER_PRED_PATH = PROJECT_ROOT / "results" / "bulk_BeatAML" / "predictions_final.csv"

N_FOLDS = 5
N_PCS = 100
TEST_SIZE = 0.20
TARGET_RECALL = 0.80
MIN_THRESHOLD, MAX_THRESHOLD = 0.25, 0.75
DEFAULT_THRESHOLD = 0.4

KEY_DRUGS = ['Venetoclax', 'Cytarabine', 'Daunorubicin', 'Idarubicin',
             'Midostaurin', 'Gilteritinib', 'Quizartinib']

LSC17_WEIGHTS = {
    'DNMT3B': 0.0874, 'ZBTB46': -0.0347, 'NYNRIN': 0.00865, 'ARHGAP22': -0.0138,
    'LAPTM4B': 0.00582, 'MMRN1': 0.0258, 'DPYSL3': 0.0284, 'KIAA0125': 0.0196,
    'CDK6': -0.0704, 'CPXM1': -0.0258, 'SOCS2': 0.0271, 'SMIM24': -0.0226,
    'EMP1': 0.0146, 'NGFRAP1': 0.0465, 'CD34': 0.0338, 'AKR1C3': -0.0402,
    'GPR56': 0.0501,
}

# -----------------------------------------------------------------------------
# Data helpers (same as baselines.py)
# -----------------------------------------------------------------------------
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
    if mtx is None: raise FileNotFoundError(f"No matrix in {matrix_dir}")
    feat = None
    for cand in ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"]:
        p = matrix_dir / cand
        if p.exists():
            feat = _read_tsv(p); break
    if feat is None: raise FileNotFoundError(f"No features in {matrix_dir}")
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
    if df is None: raise ValueError(f"Failed to read {path}")
    df = df.select_dtypes(include=[np.number])
    idx_looks_like_genes = (
        len(df.index) > 100 and
        pd.Series(df.index.astype(str)).str.match(r"^[A-Z]").mean() > 0.5
    )
    if idx_looks_like_genes and df.shape[0] > df.shape[1]:
        df = df.T
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return coalesce_duplicate_genes(df, how="sum")

def find_stable_threshold(y_true, y_prob, target_recall=0.8):
    y_true = np.asarray(y_true)
    y_prob = np.clip(y_prob, 0.001, 0.999)
    thresholds = np.unique(np.percentile(y_prob, np.arange(10, 91, 5)))
    best_f1, best_t = -1.0, DEFAULT_THRESHOLD
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if len(np.unique(y_pred)) < 2: continue
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        if rec < target_recall * 0.9: continue
        if prec < 0.3: continue
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(np.clip(best_t, MIN_THRESHOLD, MAX_THRESHOLD))

# -----------------------------------------------------------------------------
# 1) Load the 1,000-gene panel
# -----------------------------------------------------------------------------
def load_gene_panel():
    for path in GENE_LIST_CANDIDATES:
        if path.exists():
            with open(path) as f:
                genes = [line.strip().upper() for line in f if line.strip()]
            print(f"[PANEL] Loaded {len(genes)} genes from {path}", flush=True)
            return genes, path
    raise FileNotFoundError(
        f"No gene panel file found. Tried: {[str(p) for p in GENE_LIST_CANDIDATES]}")

# -----------------------------------------------------------------------------
# 2) Load training corpus restricted to the 1k panel
# -----------------------------------------------------------------------------
def load_training_corpus_panel(panel_genes):
    """Same as baselines.py but reindexes to panel_genes only."""
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

    # Restrict each donor's expression to the 1k panel; missing genes → 0.0
    panel_set = list(panel_genes)
    X_list, y_list, sid_list = [], [], []
    for df, label, sid in all_data:
        df_aligned = df.reindex(columns=panel_set).fillna(0.0)
        X_list.append(df_aligned.values.astype(np.float32))
        y_list.extend([label] * len(df))
        sid_list.extend([sid] * len(df))
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    sids = np.array(sid_list)
    # Coverage report
    n_present = sum(1 for g in panel_set if any(g in df.columns for df, _, _ in all_data))
    print(f"[DATA] Combined on 1k panel: X={X.shape}, classes={np.bincount(y)}, "
          f"coverage {n_present}/{len(panel_set)} genes present in ≥1 donor", flush=True)
    return X, y, sids, panel_set

# -----------------------------------------------------------------------------
# 3) Internal CV/test on the 1k panel — same protocol as baselines.py
# -----------------------------------------------------------------------------
def evaluate_predictions(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'auc_roc':   float(roc_auc_score(y_true, y_prob)),
        'mcc':       float(matthews_corrcoef(y_true, y_pred)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
    }

def cv_classifier_on_pcs(name, model_factory, X_train, y_train, X_test, y_test, n_pcs=N_PCS):
    print(f"\n[CV-1k] {name}", flush=True)
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
        clf = model_factory()
        clf.fit(pca.transform(Xtr_s), y_train[tr])
        probs = clf.predict_proba(pca.transform(Xva_s))[:, 1]
        oof_probs[va] = probs
        fauc = roc_auc_score(y_train[va], probs)
        fold_aucs.append(fauc)
        print(f"  fold {fold}: AUROC={fauc:.4f}", flush=True)
    threshold = find_stable_threshold(y_train, oof_probs, target_recall=TARGET_RECALL)
    # held-out test
    scaler = StandardScaler().fit(X_train)
    n_comp = min(n_pcs, X_train.shape[1] - 1, X_train.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED).fit(scaler.transform(X_train))
    clf = model_factory()
    clf.fit(pca.transform(scaler.transform(X_train)), y_train)
    test_probs = clf.predict_proba(pca.transform(scaler.transform(X_test)))[:, 1]
    test_metrics = evaluate_predictions(y_test, test_probs, threshold)
    print(f"  threshold: {threshold:.3f}; test AUROC={test_metrics['auc_roc']:.4f}, "
          f"F1={test_metrics['f1']:.3f}, MCC={test_metrics['mcc']:+.3f}", flush=True)
    return {
        'model': name,
        'cv_mean_auroc': float(np.mean(fold_aucs)),
        'cv_std_auroc':  float(np.std(fold_aucs)),
        'threshold': threshold,
        **{f'test_{k}': v for k, v in test_metrics.items()},
    }

# -----------------------------------------------------------------------------
# 4) BeatAML loaders
# -----------------------------------------------------------------------------
def load_beataml_expression(panel_genes):
    print(f"\n[BeatAML] Loading expression from: {BEATAML_EXPR_PATH}", flush=True)
    expr = pd.read_csv(BEATAML_EXPR_PATH, index_col=0)
    print(f"[BeatAML] Raw shape: {expr.shape}", flush=True)
    idx_looks_like_genes = (len(expr.index) > 100 and
                            pd.Series(expr.index.astype(str)).str.match(r"^[A-Z]").mean() > 0.5)
    if idx_looks_like_genes and expr.shape[0] > expr.shape[1]:
        expr = expr.T
        print(f"[BeatAML] Transposed -> {expr.shape}", flush=True)
    expr.columns = clean_gene_names(expr.columns)
    expr = coalesce_duplicate_genes(expr, how="sum")
    sample_ids = expr.index.astype(str).tolist()
    expr_norm = pd.DataFrame(scrna_cpm_log1p(expr.values),
                             columns=expr.columns, index=sample_ids)
    expr_aligned = expr_norm.reindex(columns=panel_genes).fillna(0.0)
    n_found = (expr_norm.columns.isin(panel_genes)).sum()
    coverage = 100.0 * n_found / len(panel_genes)
    print(f"[BeatAML] Aligned to 1k panel: {n_found}/{len(panel_genes)} matched "
          f"({coverage:.1f}% coverage), shape={expr_aligned.shape}", flush=True)
    return expr_aligned, sample_ids

def load_beataml_drug_auc():
    print(f"\n[BeatAML] Loading drug AUC from: {BEATAML_DRUG_PATH}", flush=True)
    return pd.read_csv(BEATAML_DRUG_PATH)

# -----------------------------------------------------------------------------
# 5) Score wrappers — fit on full training (1k panel), score BeatAML
# -----------------------------------------------------------------------------
def fit_score_on_pcs(name, model_factory, X_train, y_train, X_beataml, n_pcs=N_PCS):
    print(f"\n[Score] {name}", flush=True)
    scaler = StandardScaler().fit(X_train)
    n_comp = min(n_pcs, X_train.shape[1] - 1, X_train.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED).fit(scaler.transform(X_train))
    clf = model_factory()
    clf.fit(pca.transform(scaler.transform(X_train)), y_train)
    scores = clf.predict_proba(pca.transform(scaler.transform(X_beataml)))[:, 1]
    print(f"  range=[{scores.min():.3f}, {scores.max():.3f}], "
          f"mean={scores.mean():.3f}, median={np.median(scores):.3f}", flush=True)
    return scores

def score_lsc17_on_panel(X_df_panel):
    """LSC17 on whatever LSC17 genes are in the 1k panel (or BeatAML alignment)."""
    score = np.zeros(X_df_panel.shape[0], dtype=np.float32)
    found, missing = [], []
    for g, w in LSC17_WEIGHTS.items():
        if g in X_df_panel.columns:
            score += w * X_df_panel[g].values.astype(np.float32)
            found.append(g)
        else:
            missing.append(g)
    print(f"[LSC17] Found {len(found)}/17; missing: {missing}", flush=True)
    smin, smax = float(score.min()), float(score.max())
    return (np.clip((score - smin) / (smax - smin), 0.001, 0.999)
            if smax > smin else np.full_like(score, 0.5))

def score_mean_expression(X_train, X_beataml):
    scaler = StandardScaler().fit(X_train)
    s = scaler.transform(X_beataml).mean(axis=1)
    smin, smax = float(s.min()), float(s.max())
    return (np.clip((s - smin) / (smax - smin), 0.001, 0.999)
            if smax > smin else np.full_like(s, 0.5))

def load_transformer_predictions():
    if not TRANSFORMER_PRED_PATH.exists():
        print(f"[Transformer] WARNING: {TRANSFORMER_PRED_PATH} not found", flush=True)
        return None, None
    df = pd.read_csv(TRANSFORMER_PRED_PATH)
    print(f"[Transformer] Loaded {len(df)} reference predictions", flush=True)
    return df['sample_id'].astype(str).values, df['persister_probability'].values.astype(np.float32)

# -----------------------------------------------------------------------------
# 6) Drug-AUC correlation
# -----------------------------------------------------------------------------
def compute_drug_correlations(name, sample_ids, scores, drug_df, key_drugs=KEY_DRUGS):
    score_df = pd.DataFrame({'sample_id': sample_ids, 'score': scores})
    sample_col = next((c for c in ['sample_id','sample','Sample','sampleId','patient_id','Patient_ID']
                        if c in drug_df.columns), None)
    if sample_col is None:
        raise ValueError(f"No sample-id column in drug_df. Available: {drug_df.columns.tolist()}")
    drug_cols_map = {}
    for drug in key_drugs:
        for col in drug_df.columns:
            if col != sample_col and drug.lower() in col.lower():
                drug_cols_map[drug] = col; break
    merged = pd.merge(score_df, drug_df, left_on='sample_id', right_on=sample_col, how='inner')
    print(f"\n[{name}] merged n={len(merged)}", flush=True)
    rows, agg_xy = [], []
    for drug, col in drug_cols_map.items():
        sub = merged[['score', col]].dropna()
        if len(sub) < 10: continue
        r, p = spearmanr(sub['score'], sub[col])
        rows.append({'model': name, 'drug': drug, 'auc_column': col,
                     'n_samples': int(len(sub)),
                     'spearman_r': float(r), 'p_value': float(p)})
        agg_xy.extend(zip(sub['score'].values, sub[col].values))
        print(f"  {drug:<14s}: r={r:+.3f}, p={p:.2e}, n={len(sub)}", flush=True)
    if agg_xy:
        xs, ys = zip(*agg_xy)
        ar, ap = spearmanr(xs, ys)
        agg = {'model': name, 'n_pairs': len(agg_xy),
               'aggregate_spearman_r': float(ar), 'aggregate_p_value': float(ap)}
    else:
        agg = {'model': name, 'n_pairs': 0,
               'aggregate_spearman_r': float('nan'), 'aggregate_p_value': float('nan')}
    return rows, agg

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("="*70)
    print(f"FAIR baseline comparison on 1,000-gene panel | start {time.ctime()}")
    print("="*70)

    # Load gene panel
    panel, panel_path = load_gene_panel()

    # Load training corpus on panel
    X, y, sids, panel_genes = load_training_corpus_panel(panel)

    # Internal split (same as baselines.py: seed 42, stratified, 20% test)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\n[SPLIT] Train: {len(y_train)} (classes {np.bincount(y_train)}), "
          f"Test: {len(y_test)} (classes {np.bincount(y_test)})", flush=True)

    # ------------------------------------------------------------------
    # Internal CV/test on 1k panel
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("INTERNAL CV/TEST on 1k panel (sanity check vs baselines.py 13k panel)")
    print("="*70)
    internal = []
    internal.append(cv_classifier_on_pcs(
        'Logistic Regression (L2, balanced) [1k panel]',
        lambda: LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                                    solver='lbfgs', random_state=SEED),
        X_train, y_train, X_test, y_test))
    internal.append(cv_classifier_on_pcs(
        'Random Forest (300 trees, balanced) [1k panel]',
        lambda: RandomForestClassifier(n_estimators=300, max_depth=None,
                                        class_weight='balanced',
                                        random_state=SEED, n_jobs=-1),
        X_train, y_train, X_test, y_test))
    pd.DataFrame(internal).to_csv("fair_baseline_internal.csv", index=False)

    # ------------------------------------------------------------------
    # External BeatAML transfer on 1k panel
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("EXTERNAL BeatAML drug-AUC transfer on 1k panel")
    print("="*70)
    X_beataml_df, beataml_sids = load_beataml_expression(panel_genes)
    X_beataml = X_beataml_df.values.astype(np.float32)
    drug_df = load_beataml_drug_auc()

    # Score BeatAML with each baseline (fit on FULL training, panel-restricted)
    lr_scores = fit_score_on_pcs(
        '[1k] LR',
        lambda: LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                                    solver='lbfgs', random_state=SEED),
        X, y, X_beataml)
    rf_scores = fit_score_on_pcs(
        '[1k] RF',
        lambda: RandomForestClassifier(n_estimators=300, max_depth=None,
                                        class_weight='balanced',
                                        random_state=SEED, n_jobs=-1),
        X, y, X_beataml)
    lsc_scores = score_lsc17_on_panel(X_beataml_df)
    mean_scores = score_mean_expression(X, X_beataml)
    transformer_sids, transformer_scores = load_transformer_predictions()

    # Per-drug + aggregate correlations
    all_rows, all_agg = [], []
    for name, sids_arr, scores_arr in [
        ('Logistic Regression [1k panel]', np.array(beataml_sids), lr_scores),
        ('Random Forest [1k panel]',       np.array(beataml_sids), rf_scores),
        ('LSC17 [1k panel scope]',         np.array(beataml_sids), lsc_scores),
        ('Mean-expression [1k panel]',     np.array(beataml_sids), mean_scores),
    ]:
        rows, agg = compute_drug_correlations(name, sids_arr, scores_arr, drug_df)
        all_rows.extend(rows); all_agg.append(agg)
    if transformer_sids is not None:
        rows, agg = compute_drug_correlations('Transformer (canonical, 1k panel)',
                                               transformer_sids, transformer_scores, drug_df)
        all_rows.extend(rows); all_agg.append(agg)

    pd.DataFrame(all_rows).to_csv("fair_baseline_external_per_drug.csv", index=False)
    pd.DataFrame(all_agg).to_csv("fair_baseline_external_aggregate.csv", index=False)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("AGGREGATE SUMMARY (BeatAML drug-AUC, 1k panel)")
    print("="*70)
    print(f"{'Model':<40s} | {'n pairs':>8s} | {'r':>9s} | {'p-value':>10s}")
    print("-"*78)
    for agg in all_agg:
        print(f"{agg['model'][:40]:<40s} | {agg['n_pairs']:>8d} | "
              f"{agg['aggregate_spearman_r']:>+9.3f} | {agg['aggregate_p_value']:>10.2e}")
    print("-"*78)
    print(f"\nReference (13k panel from baselines_external.py):")
    print(f"  Transformer:  +0.482 (3164 pairs, p=4.1e-184)")
    print(f"  LR (13k):     -0.005")
    print(f"  RF (13k):     -0.083")
    print(f"  LSC17 (13k):  +0.128")
    print(f"  Mean (13k):   +0.238")
    print(f"\nThe key question: does LR/RF transfer improve when given proper gene")
    print(f"coverage on the 1k panel that the Transformer used?")
    print(f"\nSaved: fair_baseline_internal.csv, fair_baseline_external_per_drug.csv, "
          f"fair_baseline_external_aggregate.csv")
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
