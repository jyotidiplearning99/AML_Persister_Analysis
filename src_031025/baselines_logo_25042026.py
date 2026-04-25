#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Donor-level cross-validation for the four baselines.

This script answers the question: do the baseline performance numbers
(LR AUROC 0.999, RF 0.998, Transformer 0.94) survive donor-level cross-
validation, where cells from the same donor cannot appear in both train
and validation folds?

If the cell-level numbers are inflated by within-donor cell similarity,
all methods will collapse together. If the numbers are real biology,
they will mostly hold.

Pipeline:
  1. Load the same pooled training corpus (9 donors).
  2. Build sample/donor labels (the `sids` array).
  3. Run StratifiedGroupKFold with `groups=sids` for each baseline.
  4. Report per-fold AUROC + aggregate mean ± SD per baseline.
  5. Compare to the cell-level numbers from baselines.py.

Note on Transformer: re-running the Transformer under LOGO requires
TensorFlow and would take several hours per re-run. This script does NOT
re-run the Transformer — it reports the cell-level Transformer numbers
from the canonical metadata.json for reference. To get a Transformer-LOGO
number, run the existing production_transformer_13092025.py with
StratifiedGroupKFold substituted in (separate task, ~1 day of compute).

Outputs:
  - logo_baseline_comparison.csv   — per-baseline mean ± SD AUROC
  - logo_baseline_per_fold.csv     — per-baseline × per-fold AUROC

Usage on Mahti:
    sbatch baselines_logo.sbatch     (or run interactively)

Place this file in the same directory as baselines.py /
production_transformer_13092025.py.
"""

import os, sys, time, json, warnings, random, re, gzip
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from scipy.io import mmread

GSM_BASE = Path("/scratch/project_2010751/GSE123902_RAW")
AML_ROOT = Path("/scratch/project_2010751/AML_scRNA_decrypted")

N_FOLDS = 5
N_PCS = 100

# -----------------------------------------------------------------------------
# Helpers — copies from baselines.py
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

LSC17_WEIGHTS = {
    'DNMT3B': 0.0874, 'ZBTB46': -0.0347, 'NYNRIN': 0.00865, 'ARHGAP22': -0.0138,
    'LAPTM4B': 0.00582, 'MMRN1': 0.0258, 'DPYSL3': 0.0284, 'KIAA0125': 0.0196,
    'CDK6': -0.0704, 'CPXM1': -0.0258, 'SOCS2': 0.0271, 'SMIM24': -0.0226,
    'EMP1': 0.0146, 'NGFRAP1': 0.0465, 'CD34': 0.0338, 'AKR1C3': -0.0402,
    'GPR56': 0.0501,
}

def load_training_corpus():
    """Same as baselines.py — pooled 9-dataset corpus with per-cell sample/donor labels."""
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
    print(f"[DATA] Donors / samples: {len(np.unique(sids))} unique", flush=True)
    print(f"[DATA] Per-donor class composition:")
    for s in np.unique(sids):
        cls = y[sids == s]
        print(f"  {s}: n={len(cls)}, class={cls[0]}", flush=True)
    return X, y, sids, common_genes

# -----------------------------------------------------------------------------
# CV evaluators
# -----------------------------------------------------------------------------
def stratified_group_cv(name, model_factory, X, y, groups, n_pcs=100, n_folds=N_FOLDS,
                        feature_mode='pca'):
    """
    StratifiedGroupKFold CV with the same fold structure for every method.
    feature_mode: 'pca' for LR/RF, 'raw' for LSC17/mean-expression.
    """
    print(f"\n[LOGO-CV] {name}", flush=True)
    print(f"  feature_mode={feature_mode}, n_folds={n_folds}, n_pcs={n_pcs}", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    fold_aucs = []
    fold_records = []

    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups), 1):
        train_donors = np.unique(groups[tr])
        val_donors = np.unique(groups[va])
        # Sanity: no donor leakage
        leak = set(train_donors) & set(val_donors)
        assert not leak, f"Donor leakage: {leak}"

        train_class_balance = np.bincount(y[tr], minlength=2)
        val_class_balance = np.bincount(y[va], minlength=2)
        # Skip if val has only one class
        if len(np.unique(y[va])) < 2:
            print(f"  fold {fold}: SKIPPED (val has only one class; donors={val_donors}, "
                  f"class balance={val_class_balance})", flush=True)
            continue

        if feature_mode == 'pca':
            scaler = StandardScaler().fit(X[tr])
            Xtr_s, Xva_s = scaler.transform(X[tr]), scaler.transform(X[va])
            n_comp = min(n_pcs, Xtr_s.shape[1] - 1, Xtr_s.shape[0] - 1)
            if n_comp < 2: n_comp = 2
            pca = PCA(n_components=n_comp, random_state=SEED).fit(Xtr_s)
            Xtr_f, Xva_f = pca.transform(Xtr_s), pca.transform(Xva_s)
            clf = model_factory()
            clf.fit(Xtr_f, y[tr])
            probs = clf.predict_proba(Xva_f)[:, 1]
        elif feature_mode == 'lsc17':
            # Need column index map. Pass via closure: model_factory returns score directly
            probs = model_factory(X[va], gene_index_map=X_aux_global)
        elif feature_mode == 'mean':
            scaler = StandardScaler().fit(X[tr])
            Xva_s = scaler.transform(X[va])
            score = Xva_s.mean(axis=1)
            smin, smax = float(score.min()), float(score.max())
            probs = (score - smin) / (smax - smin) if smax > smin else np.full_like(score, 0.5)
            probs = np.clip(probs, 0.001, 0.999)
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        try:
            auc = roc_auc_score(y[va], probs)
            fold_aucs.append(auc)
            fold_records.append({
                'model': name, 'fold': fold, 'auroc': float(auc),
                'train_donors': list(map(str, train_donors)),
                'val_donors':   list(map(str, val_donors)),
                'val_n_cells':  int(len(y[va])),
                'val_class_neg': int(val_class_balance[0]),
                'val_class_pos': int(val_class_balance[1]),
            })
            print(f"  fold {fold}: AUROC={auc:.4f}, val donors={list(val_donors)}, "
                  f"n_val={len(y[va])}, classes={val_class_balance}", flush=True)
        except Exception as e:
            print(f"  fold {fold}: FAILED ({e})", flush=True)

    summary = {
        'model': name,
        'logo_n_folds_evaluated': len(fold_aucs),
        'logo_mean_auroc': float(np.mean(fold_aucs)) if fold_aucs else float('nan'),
        'logo_std_auroc':  float(np.std(fold_aucs)) if fold_aucs else float('nan'),
        'logo_min_auroc':  float(np.min(fold_aucs)) if fold_aucs else float('nan'),
        'logo_max_auroc':  float(np.max(fold_aucs)) if fold_aucs else float('nan'),
    }
    print(f"  [LOGO summary] mean AUROC = {summary['logo_mean_auroc']:.3f} ± "
          f"{summary['logo_std_auroc']:.3f} (range {summary['logo_min_auroc']:.3f}–"
          f"{summary['logo_max_auroc']:.3f}) across {summary['logo_n_folds_evaluated']} folds",
          flush=True)
    return summary, fold_records

# Global gene-index map needed by lsc17 in fold-loop; populated in main()
X_aux_global = None

def lsc17_fold_score(X_fold, gene_index_map):
    """Apply LSC17 weights to a single fold's expression matrix."""
    score = np.zeros(X_fold.shape[0], dtype=np.float32)
    for g, w in LSC17_WEIGHTS.items():
        idx = gene_index_map.get(g)
        if idx is not None:
            score += w * X_fold[:, idx]
    smin, smax = float(score.min()), float(score.max())
    if smax > smin:
        return np.clip((score - smin) / (smax - smin), 0.001, 0.999)
    return np.full_like(score, 0.5)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    global X_aux_global
    print("="*70)
    print(f"Donor-level (StratifiedGroupKFold) cross-validation | start {time.ctime()}")
    print("="*70)

    X, y, sids, gene_list = load_training_corpus()
    gene_index_map = {g: i for i, g in enumerate(gene_list)}
    X_aux_global = gene_index_map

    # Class-balance per donor — needed to sanity-check that LOGO is even feasible
    print(f"\n[INFO] Total donors: {len(np.unique(sids))}")
    print(f"[INFO] Donors per class:")
    pos_donors = np.unique(sids[y == 1])
    neg_donors = np.unique(sids[y == 0])
    print(f"  positive class donors (n={len(pos_donors)}): {list(pos_donors)}")
    print(f"  negative class donors (n={len(neg_donors)}): {list(neg_donors)}")

    if len(pos_donors) < N_FOLDS or len(neg_donors) < N_FOLDS:
        n_feasible = min(len(pos_donors), len(neg_donors))
        print(f"\n[WARN] Cannot run {N_FOLDS}-fold StratifiedGroupKFold; fewer donors than folds.")
        print(f"       Reducing to {n_feasible}-fold.", flush=True)
        n_folds_use = n_feasible
    else:
        n_folds_use = N_FOLDS

    # ---- Run all four baselines under StratifiedGroupKFold ----
    all_summaries = []
    all_fold_records = []

    s, f = stratified_group_cv(
        'Logistic Regression (L2, balanced)',
        lambda: LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                                    solver='lbfgs', random_state=SEED),
        X, y, sids, n_pcs=N_PCS, n_folds=n_folds_use, feature_mode='pca')
    all_summaries.append(s); all_fold_records.extend(f)

    s, f = stratified_group_cv(
        'Random Forest (300 trees, balanced)',
        lambda: RandomForestClassifier(n_estimators=300, max_depth=None,
                                        class_weight='balanced',
                                        random_state=SEED, n_jobs=-1),
        X, y, sids, n_pcs=N_PCS, n_folds=n_folds_use, feature_mode='pca')
    all_summaries.append(s); all_fold_records.extend(f)

    # LSC17 — feature_mode='lsc17' uses the closure to produce score on val raw
    # Precompute LSC17 score for the entire X (since it's deterministic) to avoid recompute
    print("\n[LOGO-CV] LSC17 (Ng et al. 2016)", flush=True)
    full_lsc_score = lsc17_fold_score(X, gene_index_map)
    print(f"  feature_mode=lsc17 (deterministic; score precomputed for whole X)", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=n_folds_use)
    lsc_fold_aucs = []
    for fold, (tr, va) in enumerate(sgkf.split(X, y, sids), 1):
        if len(np.unique(y[va])) < 2:
            val_donors = np.unique(sids[va])
            print(f"  fold {fold}: SKIPPED (val has only one class; donors={val_donors})", flush=True)
            continue
        try:
            auc = roc_auc_score(y[va], full_lsc_score[va])
            lsc_fold_aucs.append(auc)
            val_donors = np.unique(sids[va])
            print(f"  fold {fold}: AUROC={auc:.4f}, val donors={list(val_donors)}", flush=True)
            all_fold_records.append({
                'model': 'LSC17 (Ng 2016)', 'fold': fold, 'auroc': float(auc),
                'train_donors': list(map(str, np.unique(sids[tr]))),
                'val_donors':   list(map(str, val_donors)),
                'val_n_cells':  int(len(y[va])),
                'val_class_neg': int(np.sum(y[va] == 0)),
                'val_class_pos': int(np.sum(y[va] == 1)),
            })
        except Exception as e:
            print(f"  fold {fold}: FAILED ({e})", flush=True)
    all_summaries.append({
        'model': 'LSC17 (Ng 2016)',
        'logo_n_folds_evaluated': len(lsc_fold_aucs),
        'logo_mean_auroc': float(np.mean(lsc_fold_aucs)) if lsc_fold_aucs else float('nan'),
        'logo_std_auroc':  float(np.std(lsc_fold_aucs)) if lsc_fold_aucs else float('nan'),
        'logo_min_auroc':  float(np.min(lsc_fold_aucs)) if lsc_fold_aucs else float('nan'),
        'logo_max_auroc':  float(np.max(lsc_fold_aucs)) if lsc_fold_aucs else float('nan'),
    })
    if lsc_fold_aucs:
        print(f"  [LOGO summary] mean AUROC = {np.mean(lsc_fold_aucs):.3f} ± "
              f"{np.std(lsc_fold_aucs):.3f}", flush=True)

    # Mean-expression baseline
    s, f = stratified_group_cv(
        'Mean-expression (whole panel)',
        None,
        X, y, sids, n_pcs=N_PCS, n_folds=n_folds_use, feature_mode='mean')
    all_summaries.append(s); all_fold_records.extend(f)

    # ---- Save ----
    pd.DataFrame(all_summaries).to_csv("logo_baseline_comparison.csv", index=False)

    # Convert fold_records list-of-dicts (with list values) to flat CSV
    flat = pd.DataFrame(all_fold_records)
    flat['train_donors'] = flat['train_donors'].apply(lambda L: '|'.join(L))
    flat['val_donors']   = flat['val_donors'].apply(lambda L: '|'.join(L))
    flat.to_csv("logo_baseline_per_fold.csv", index=False)

    # ---- Final summary ----
    print("\n" + "="*70)
    print("DONOR-LEVEL CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(f"{'Model':<40s} | {'LOGO mean':>11s} | {'cell-level':>11s} | {'drop':>7s}")
    print("-"*78)
    cell_level = {
        'Logistic Regression (L2, balanced)': 0.999,
        'Random Forest (300 trees, balanced)': 0.998,
        'LSC17 (Ng 2016)': 0.453,
        'Mean-expression (whole panel)': 0.615,
    }
    for s in all_summaries:
        cl = cell_level.get(s['model'], None)
        cl_str = f"{cl:.3f}" if cl is not None else "n/a"
        if cl is not None and not np.isnan(s['logo_mean_auroc']):
            drop = cl - s['logo_mean_auroc']
            drop_str = f"{drop:+.3f}"
        else:
            drop_str = "n/a"
        print(f"{s['model'][:40]:<40s} | {s['logo_mean_auroc']:.3f}±{s['logo_std_auroc']:.3f} | "
              f"{cl_str:>11s} | {drop_str:>7s}")
    print("-"*78)
    print(f"\nReference: Transformer cell-level CV mean AUROC = 0.936 ± 0.014")
    print(f"            Transformer LOGO not run here (requires ~1 day GPU compute);")
    print(f"            re-run production_transformer_13092025.py with StratifiedGroupKFold.")
    print(f"\nSaved:")
    print(f"  logo_baseline_comparison.csv")
    print(f"  logo_baseline_per_fold.csv")
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
