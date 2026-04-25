#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External BeatAML drug-AUC comparison for the four baselines and the Transformer.

This script answers the question: do the simpler baselines also produce the
BeatAML drug-AUC correlation (Spearman r ≈ 0.83) that we attribute to the
Transformer in the manuscript?

Pipeline:
  1. Train each baseline on the IDENTICAL pooled training corpus (9-dataset)
     using the same train/test split as baselines.py.
  2. Score the BeatAML expression matrix with each fitted baseline.
  3. Compute per-drug Spearman correlation between each baseline's score
     and the per-patient ex-vivo drug AUC, across the same 7 AML drugs as
     the manuscript (Venetoclax, Cytarabine, Daunorubicin, Idarubicin,
     Midostaurin, Gilteritinib, Quizartinib).
  4. Compute aggregate Spearman correlation across the full patient × drug
     matrix (the headline r=0.83 number in the manuscript).
  5. Report Transformer score on the same data for comparison (loaded from
     the canonical predictions_reduced_1k.csv).

Outputs:
  - external_beataml_per_drug.csv   — per-baseline × per-drug Spearman r and p
  - external_beataml_aggregate.csv  — per-baseline aggregate r across all 7 drugs
  - external_beataml_summary.txt    — human-readable comparison table

Usage on Mahti:
    sbatch baselines_external.sbatch     (or run interactively)

Place this file in the same directory as baselines.py / production_transformer_13092025.py
on Mahti so the data paths resolve identically.
"""

import os, sys, time, json, warnings, random, re, gzip
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Reproducibility (match baselines.py / training)
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
from scipy.io import mmread

# -----------------------------------------------------------------------------
# Configuration — match training script and baselines.py
# -----------------------------------------------------------------------------
GSM_BASE = Path("/scratch/project_2010751/GSE123902_RAW")
AML_ROOT = Path("/scratch/project_2010751/AML_scRNA_decrypted")
PROJECT_ROOT = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis")

# BeatAML data paths (verified against existing inference scripts)
BEATAML_EXPR_PATH = PROJECT_ROOT / "results" / "bulk_BeatAML" / "processed" / "expression.csv"
BEATAML_DRUG_PATH = PROJECT_ROOT / "data" / "beataml_drugs" / "processed" / "beataml_key_drugs_auc.csv"

# Canonical Transformer predictions on BeatAML (for reference comparison)
TRANSFORMER_PRED_PATH = PROJECT_ROOT / "results" / "bulk_BeatAML" / "predictions_final.csv"

TEST_SIZE = 0.20
N_PCS = 100

# Seven AML-relevant drugs from the manuscript
KEY_DRUGS = ['Venetoclax', 'Cytarabine', 'Daunorubicin', 'Idarubicin',
             'Midostaurin', 'Gilteritinib', 'Quizartinib']

# -----------------------------------------------------------------------------
# Data loaders — copies of helpers from baselines.py (to make this self-contained)
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
# LSC17 weights (Ng et al. Nature 2016)
# -----------------------------------------------------------------------------
LSC17_WEIGHTS = {
    'DNMT3B': 0.0874, 'ZBTB46': -0.0347, 'NYNRIN': 0.00865, 'ARHGAP22': -0.0138,
    'LAPTM4B': 0.00582, 'MMRN1': 0.0258, 'DPYSL3': 0.0284, 'KIAA0125': 0.0196,
    'CDK6': -0.0704, 'CPXM1': -0.0258, 'SOCS2': 0.0271, 'SMIM24': -0.0226,
    'EMP1': 0.0146, 'NGFRAP1': 0.0465, 'CD34': 0.0338, 'AKR1C3': -0.0402,
    'GPR56': 0.0501,
}

# -----------------------------------------------------------------------------
# Training corpus loader — IDENTICAL to baselines.py
# -----------------------------------------------------------------------------
def load_training_corpus():
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

    gene_sets = [set(df.columns) for df, _, _ in all_data]
    common_genes = set.intersection(*gene_sets)
    if len(common_genes) < 50:
        gene_counter = Counter()
        for gs in gene_sets:
            gene_counter.update(gs)
        min_presence = max(2, len(gene_sets) // 2)
        common_genes = {g for g, c in gene_counter.items() if c >= min_presence}
    common_genes = sorted(common_genes)

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
# BeatAML loaders
# -----------------------------------------------------------------------------
def load_beataml_expression(common_genes_train):
    """Load BeatAML expression matrix and align to training gene set."""
    print(f"\n[BeatAML] Loading expression from: {BEATAML_EXPR_PATH}", flush=True)
    if not BEATAML_EXPR_PATH.exists():
        raise FileNotFoundError(f"BeatAML expression not found: {BEATAML_EXPR_PATH}")
    expr = pd.read_csv(BEATAML_EXPR_PATH, index_col=0)
    print(f"[BeatAML] Raw shape: {expr.shape}", flush=True)

    # Auto-orient: rows should be cells/samples, cols should be genes
    idx_looks_like_genes = (len(expr.index) > 100 and
                            pd.Series(expr.index.astype(str)).str.match(r"^[A-Z]").mean() > 0.5)
    if idx_looks_like_genes and expr.shape[0] > expr.shape[1]:
        expr = expr.T
        print(f"[BeatAML] Transposed -> {expr.shape}", flush=True)

    # Clean gene names, dedupe
    expr.columns = clean_gene_names(expr.columns)
    expr = coalesce_duplicate_genes(expr, how="sum")

    # Apply CPM-log1p (same preprocessing as training)
    sample_ids = expr.index.astype(str).tolist()
    expr_norm = pd.DataFrame(scrna_cpm_log1p(expr.values),
                             columns=expr.columns, index=sample_ids)

    # Align to training gene index
    expr_aligned = expr_norm.reindex(columns=common_genes_train).fillna(0.0)
    n_found = (expr_norm.columns.isin(common_genes_train)).sum()
    coverage = 100.0 * n_found / len(common_genes_train)
    print(f"[BeatAML] Aligned to {len(common_genes_train)} training genes: "
          f"{n_found} matched ({coverage:.1f}% coverage), shape={expr_aligned.shape}",
          flush=True)
    return expr_aligned, sample_ids

def load_beataml_drug_auc():
    """Load BeatAML drug AUC matrix."""
    print(f"\n[BeatAML] Loading drug AUC from: {BEATAML_DRUG_PATH}", flush=True)
    if not BEATAML_DRUG_PATH.exists():
        raise FileNotFoundError(f"BeatAML drug AUC not found: {BEATAML_DRUG_PATH}")
    drug = pd.read_csv(BEATAML_DRUG_PATH)
    print(f"[BeatAML] Drug data shape: {drug.shape}, columns: {drug.columns.tolist()[:8]}...",
          flush=True)
    return drug

# -----------------------------------------------------------------------------
# Score wrappers — each returns a probability/score per BeatAML sample
# -----------------------------------------------------------------------------
def fit_lr_and_score_beataml(X_train, y_train, X_beataml, n_pcs=100):
    """Train LR on full training set, then score BeatAML."""
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    n_comp = min(n_pcs, Xs.shape[1] - 1, Xs.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED).fit(Xs)
    Xp = pca.transform(Xs)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                             solver='lbfgs', random_state=SEED)
    clf.fit(Xp, y_train)
    Xb_s = scaler.transform(X_beataml)
    Xb_p = pca.transform(Xb_s)
    return clf.predict_proba(Xb_p)[:, 1]

def fit_rf_and_score_beataml(X_train, y_train, X_beataml, n_pcs=100):
    """Train RF on full training set, then score BeatAML."""
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    n_comp = min(n_pcs, Xs.shape[1] - 1, Xs.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED).fit(Xs)
    Xp = pca.transform(Xs)
    clf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                  class_weight='balanced',
                                  random_state=SEED, n_jobs=-1)
    clf.fit(Xp, y_train)
    Xb_s = scaler.transform(X_beataml)
    Xb_p = pca.transform(Xb_s)
    return clf.predict_proba(Xb_p)[:, 1]

def score_lsc17_on_beataml(X_beataml_df):
    """Compute LSC17 score on BeatAML."""
    score = np.zeros(X_beataml_df.shape[0], dtype=np.float32)
    found, missing = [], []
    for g, w in LSC17_WEIGHTS.items():
        if g in X_beataml_df.columns:
            score += w * X_beataml_df[g].values.astype(np.float32)
            found.append(g)
        else:
            missing.append(g)
    print(f"[LSC17 BeatAML] Found {len(found)}/17 genes; missing: {missing}", flush=True)
    smin, smax = float(score.min()), float(score.max())
    if smax > smin:
        prob = (score - smin) / (smax - smin)
    else:
        prob = np.full_like(score, 0.5)
    return np.clip(prob, 0.001, 0.999)

def score_mean_expression_on_beataml(X_train, X_beataml):
    """Mean-expression baseline: mean of z-scored panel."""
    scaler = StandardScaler().fit(X_train)
    Xb_s = scaler.transform(X_beataml)
    score = Xb_s.mean(axis=1)
    smin, smax = float(score.min()), float(score.max())
    if smax > smin:
        return np.clip((score - smin) / (smax - smin), 0.001, 0.999)
    return np.full_like(score, 0.5)

def load_transformer_predictions_on_beataml():
    """Load canonical Transformer predictions on BeatAML for reference."""
    if not TRANSFORMER_PRED_PATH.exists():
        print(f"[Transformer] WARNING: {TRANSFORMER_PRED_PATH} not found; "
              f"will skip Transformer reference comparison", flush=True)
        return None, None
    df = pd.read_csv(TRANSFORMER_PRED_PATH)
    print(f"[Transformer] Loaded {len(df)} reference predictions", flush=True)
    return df['sample_id'].astype(str).values, df['persister_probability'].values.astype(np.float32)

# -----------------------------------------------------------------------------
# Drug-AUC correlation analysis
# -----------------------------------------------------------------------------
def compute_drug_correlations(model_name, sample_ids, scores, drug_df, key_drugs=KEY_DRUGS):
    """
    Compute per-drug Spearman correlation between model scores and BeatAML drug AUCs.
    Returns: list of dicts (one per drug), plus an aggregate correlation across all drugs.
    """
    # Build score dataframe indexed by sample_id
    score_df = pd.DataFrame({'sample_id': sample_ids, 'score': scores})

    # Detect column with sample IDs in drug_df
    sample_col = None
    for cand in ['sample_id', 'sample', 'Sample', 'sampleId', 'patient_id', 'Patient_ID']:
        if cand in drug_df.columns:
            sample_col = cand
            break
    if sample_col is None:
        raise ValueError(f"No sample-id column in drug_df. Available: {drug_df.columns.tolist()}")

    # Detect drug AUC columns. They might have variants: 'venetoclax', 'venetoclax_AUC',
    # 'AUC_venetoclax', etc. Try case-insensitive match.
    drug_cols_map = {}
    drug_cols_all = [c for c in drug_df.columns if c != sample_col]
    for drug in key_drugs:
        for col in drug_cols_all:
            cl = col.lower()
            if drug.lower() in cl:
                drug_cols_map[drug] = col
                break

    print(f"\n[{model_name}] Drug column map: {drug_cols_map}", flush=True)

    # Merge
    merged = pd.merge(score_df, drug_df, left_on='sample_id', right_on=sample_col, how='inner')
    print(f"[{model_name}] Merged samples for drug analysis: n={len(merged)}", flush=True)

    rows = []
    aggregate_xy = []  # for full pooled correlation
    for drug, col in drug_cols_map.items():
        sub = merged[['score', col]].dropna()
        if len(sub) < 10:
            print(f"[{model_name}] {drug}: insufficient overlap (n={len(sub)})", flush=True)
            continue
        r, p = spearmanr(sub['score'], sub[col])
        rows.append({
            'model': model_name, 'drug': drug, 'auc_column': col,
            'n_samples': int(len(sub)),
            'spearman_r': float(r), 'p_value': float(p),
        })
        aggregate_xy.extend([(s, a) for s, a in zip(sub['score'].values, sub[col].values)])
        print(f"[{model_name}] {drug:<14s}: r={r:+.3f}, p={p:.2e}, n={len(sub)}", flush=True)

    # Aggregate across all (sample, drug) pairs
    if aggregate_xy:
        xs, ys = zip(*aggregate_xy)
        agg_r, agg_p = spearmanr(xs, ys)
        agg = {
            'model': model_name,
            'n_pairs': len(aggregate_xy),
            'aggregate_spearman_r': float(agg_r),
            'aggregate_p_value': float(agg_p),
        }
    else:
        agg = {'model': model_name, 'n_pairs': 0,
               'aggregate_spearman_r': float('nan'),
               'aggregate_p_value': float('nan')}

    return rows, agg

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("="*70)
    print(f"External BeatAML drug-AUC comparison | start {time.ctime()}")
    print("="*70)

    # 1) Load training corpus
    X, y, sids, gene_list = load_training_corpus()

    # 2) Same train/test split as baselines.py — but here we fit on FULL data
    # for transfer. We don't need a held-out test split for the BeatAML transfer
    # analysis. Use full train for the LR/RF/mean fits.
    print(f"\n[INFO] Fitting baselines on full training corpus (n={len(y)} cells)", flush=True)

    # 3) Load BeatAML expression and drug AUC
    X_beataml_df, beataml_sids = load_beataml_expression(gene_list)
    X_beataml = X_beataml_df.values.astype(np.float32)
    drug_df = load_beataml_drug_auc()

    # 4) Score BeatAML with each baseline
    print("\n" + "="*70)
    print("Scoring BeatAML with each baseline")
    print("="*70)

    print("\n[1/4] Logistic Regression (L2)", flush=True)
    lr_scores = fit_lr_and_score_beataml(X, y, X_beataml)
    print(f"  Score range: [{lr_scores.min():.3f}, {lr_scores.max():.3f}], "
          f"mean={lr_scores.mean():.3f}, median={np.median(lr_scores):.3f}", flush=True)

    print("\n[2/4] Random Forest (300 trees)", flush=True)
    rf_scores = fit_rf_and_score_beataml(X, y, X_beataml)
    print(f"  Score range: [{rf_scores.min():.3f}, {rf_scores.max():.3f}], "
          f"mean={rf_scores.mean():.3f}, median={np.median(rf_scores):.3f}", flush=True)

    print("\n[3/4] LSC17 (Ng et al. 2016)", flush=True)
    lsc_scores = score_lsc17_on_beataml(X_beataml_df)
    print(f"  Score range: [{lsc_scores.min():.3f}, {lsc_scores.max():.3f}], "
          f"mean={lsc_scores.mean():.3f}, median={np.median(lsc_scores):.3f}", flush=True)

    print("\n[4/4] Mean-expression baseline", flush=True)
    mean_scores = score_mean_expression_on_beataml(X, X_beataml)
    print(f"  Score range: [{mean_scores.min():.3f}, {mean_scores.max():.3f}], "
          f"mean={mean_scores.mean():.3f}, median={np.median(mean_scores):.3f}", flush=True)

    # 5) Reference: Transformer predictions on BeatAML (canonical)
    print("\n[Reference] Loading canonical Transformer predictions on BeatAML", flush=True)
    transformer_sids, transformer_scores = load_transformer_predictions_on_beataml()

    # 6) Per-drug correlations
    print("\n" + "="*70)
    print("Per-drug Spearman correlations with ex-vivo drug AUC")
    print("="*70)

    all_rows = []
    all_aggregates = []

    for name, sids_arr, scores_arr in [
        ('Logistic Regression', np.array(beataml_sids), lr_scores),
        ('Random Forest',       np.array(beataml_sids), rf_scores),
        ('LSC17',               np.array(beataml_sids), lsc_scores),
        ('Mean-expression',     np.array(beataml_sids), mean_scores),
    ]:
        rows, agg = compute_drug_correlations(name, sids_arr, scores_arr, drug_df)
        all_rows.extend(rows)
        all_aggregates.append(agg)

    if transformer_sids is not None:
        rows, agg = compute_drug_correlations('Transformer (canonical)',
                                               transformer_sids, transformer_scores, drug_df)
        all_rows.extend(rows)
        all_aggregates.append(agg)

    # 7) Save
    pd.DataFrame(all_rows).to_csv("external_beataml_per_drug.csv", index=False)
    pd.DataFrame(all_aggregates).to_csv("external_beataml_aggregate.csv", index=False)

    # 8) Human-readable summary
    print("\n" + "="*70)
    print("AGGREGATE SUMMARY (across all 7 drugs × n patients)")
    print("="*70)
    print(f"{'Model':<28s} | {'n pairs':>8s} | {'Spearman r':>11s} | {'p-value':>10s}")
    print("-"*68)
    for agg in all_aggregates:
        print(f"{agg['model'][:28]:<28s} | {agg['n_pairs']:>8d} | "
              f"{agg['aggregate_spearman_r']:>+11.3f} | {agg['aggregate_p_value']:>10.2e}")
    print("-"*68)
    print(f"\nManuscript reference (canonical Transformer): r=0.83, p≈2.8e-102")
    print(f"\nSaved:")
    print(f"  external_beataml_per_drug.csv")
    print(f"  external_beataml_aggregate.csv")
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
