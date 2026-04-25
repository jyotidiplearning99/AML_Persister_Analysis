#!/usr/bin/env python3
"""
Drug-Persister Correlation using Local DepMap Data (FINAL - ALL BUGS FIXED)
CRITICAL FIXES:
- HL-60 overwrite bug (keep distinct during DepMap matching)
- Correct log2 inverse transform
- CPM preprocessing sanity checks
- Hard consistency checks for PCA/scaler
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import joblib
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("drug_corr")

# ============================================================================
# Configuration
# ============================================================================

DEPMAP_DIR = Path("/scratch/project_2010751/DepMap_Datasets")
DRUG_DATA_FILE = Path("/scratch/project_2010376/JDs_Project/drug_data/41375_2020_978_MOESM7_ESM_AML cell lines.xlsx")

MODEL_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled")
MODEL_PATH = MODEL_DIR / "final_model.h5"
GENES_FILE = MODEL_DIR / "selected_genes.txt"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
PCA_PATH = MODEL_DIR / "pca.pkl"
RESIDUALIZER_PATH = MODEL_DIR / "residualizer.pkl"

OUTPUT_DIR = Path("/scratch/project_2010376/JDs_Project/drug_persister_correlation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ALL 28 DSRT cell lines
ALL_28_CELL_LINES = [
    'AML-193', 'AP-1060', 'GDM-1', 'HL-60', 'HL-60_TB', 'KASUMI-1',
    'KASUMI-6', 'KG-1', 'ME-1', 'ML-2', 'MOLM-13', 'MOLM-16',
    'MONO-MAC-1', 'MONO-MAC-6', 'MUTZ-2', 'MV4-11', 'NB-4', 'NOMO-1',
    'OCI-AML2', 'OCI-AML3', 'OCI-AML5', 'PL-21', 'SH-2', 'SHI-1',
    'SIG-M5', 'SKM-1', 'THP-1', 'UT-7'
]

MIN_N_FOR_CORRELATION = 8

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

def clean_gene_names(names):
    return [str(g).strip().upper().rsplit(".", 1)[0] for g in names]

def scrna_cpm_log1p(X):
    X = np.asarray(X, dtype=np.float32)
    X = np.maximum(X, 0.0)
    lib = X.sum(axis=1, keepdims=True)
    np.maximum(lib, 1.0, out=lib)
    X = (X / lib) * 1e4
    return np.log1p(X).astype(np.float32)

# ============================================================================
# Load Model and Preprocessing
# ============================================================================

log.info("Loading model and preprocessing files...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(GENES_FILE) as f:
    TRAINING_GENES = [line.strip().upper() for line in f if line.strip()]

def load_optional(path):
    if Path(path).exists():
        try:
            obj = joblib.load(path)
            log.info(f"✓ Loaded: {path.name}")
            return obj
        except Exception as e:
            log.error(f"✗ Failed to load {path.name}: {e}")
            return None
    else:
        log.warning(f"✗ File not found: {path}")
        return None

scaler = load_optional(SCALER_PATH)
pca = load_optional(PCA_PATH)
residualizer = load_optional(RESIDUALIZER_PATH)

log.info(f"✓ Model loaded: {len(TRAINING_GENES)} genes")
log.info(f"✓ Model expects input shape: {model.input_shape}")

# HARD CONSISTENCY CHECKS
if scaler is not None and hasattr(scaler, "n_features_in_"):
    assert scaler.n_features_in_ == len(TRAINING_GENES), (
        f"Scaler expects {scaler.n_features_in_} features, but TRAINING_GENES={len(TRAINING_GENES)}"
    )
    log.info(f"✓ Scaler consistency check passed")

if pca is not None:
    if hasattr(pca, "n_features_in_"):
        assert pca.n_features_in_ == len(TRAINING_GENES), (
            f"PCA expects {pca.n_features_in_} features, but TRAINING_GENES={len(TRAINING_GENES)}"
        )
        log.info(f"✓ PCA input consistency check passed")
    
    expected_model_input = model.input_shape[1]
    assert hasattr(pca, "n_components_") and pca.n_components_ == expected_model_input, (
        f"PCA components={getattr(pca,'n_components_',None)} but model expects {expected_model_input}"
    )
    log.info(f"✓ PCA output consistency check passed: {pca.n_components_} components")
else:
    log.error("PCA is REQUIRED but not loaded!")
    sys.exit(1)

# ============================================================================
# Preprocessing Pipeline
# ============================================================================

def preprocess_data(X):
    """
    Full preprocessing: CPM+log1p → Residualizer → Scaler → PCA
    Note: Residualizer applies only intercept (B[0]) for global mean-centering
    """
    # Step 1: CPM + log1p
    X = scrna_cpm_log1p(X)
    
    # Sanity check after CPM+log1p
    log.debug(f"  After CPM+log1p: min={X.min():.4f}, median={np.median(X):.4f}, max={X.max():.4f}")
    
    # Step 2: Residualizer (intercept-only)
    if residualizer and getattr(residualizer, "B", None) is not None:
        try:
            b0 = np.asarray(residualizer.B)[0]
            if b0.shape[0] == X.shape[1]:
                X = X - b0
                log.debug("  Applied residualizer (intercept-only)")
        except Exception as e:
            log.warning(f"Residualizer failed: {e}")
    
    # Step 3: Scaler
    if scaler:
        X = scaler.transform(X)
        log.debug("  Applied scaler")
    
    # Step 4: PCA (CRITICAL)
    if pca is None:
        raise RuntimeError("PCA not available — cannot reduce features to model input size")
    
    X = pca.transform(X).astype(np.float32)
    log.debug(f"  After PCA: shape={X.shape}")
    
    return X

# ============================================================================
# DepMap Loading
# ============================================================================

def load_depmap_expression_efficient():
    log.info("Loading DepMap expression (only training genes)...")
    
    expr_file = DEPMAP_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    
    if not expr_file.exists():
        log.error(f"Expression file not found: {expr_file}")
        sys.exit(1)
    
    header_df = pd.read_csv(expr_file, nrows=0)
    
    gene_to_col = {}
    for col in header_df.columns:
        if col == header_df.columns[0]:
            continue
        
        if ' (' in col:
            symbol = col.split(' (')[0].upper()
        else:
            symbol = col.upper()
        
        if symbol in TRAINING_GENES:
            gene_to_col.setdefault(symbol, col)  # Keep first occurrence
    
    log.info(f"Found {len(gene_to_col)}/{len(TRAINING_GENES)} training genes in DepMap")
    
    cols_to_load = [header_df.columns[0]] + list(gene_to_col.values())
    expr_df = pd.read_csv(expr_file, usecols=cols_to_load, index_col=0)
    
    log.info(f"✓ Loaded: {expr_df.shape[0]} cell lines × {expr_df.shape[1]} genes")
    
    sample_vals = expr_df.iloc[0, :min(50, expr_df.shape[1])].values
    log.info(f"Expression check: min={np.min(sample_vals):.2f}, median={np.median(sample_vals):.2f}, max={np.max(sample_vals):.2f}")
    log.info("(DepMap log2(TPM+1): typical max ~10-15)")
    
    return expr_df, gene_to_col

def load_depmap_metadata():
    log.info("Loading DepMap metadata...")
    
    metadata_file = DEPMAP_DIR / "Model.csv"
    
    if not metadata_file.exists():
        log.error(f"Metadata file not found: {metadata_file}")
        sys.exit(1)
    
    metadata = pd.read_csv(metadata_file)
    
    if 'ModelID' in metadata.columns:
        metadata = metadata.set_index('ModelID')
    elif 'DepMap_ID' in metadata.columns:
        metadata = metadata.set_index('DepMap_ID')
    
    log.info(f"✓ Loaded: {len(metadata)} cell lines")
    
    return metadata

# ============================================================================
# Cell Line Matching (FIXED: Don't overwrite HL-60)
# ============================================================================

def normalize_cell_line_name(name):
    """Normalize for matching (HL-60_TB → HL60 for search)"""
    name = name.replace('_TB', '')
    return name.upper().replace('-', '').replace('_', '')

def find_depmap_cell_line(cell_line_name, metadata, expr_df):
    search = normalize_cell_line_name(cell_line_name)
    
    for col in ['stripped_cell_line_name', 'cell_line_name', 'CellLineName']:
        if col not in metadata.columns:
            continue
        
        matches = metadata[
            metadata[col].astype(str).apply(normalize_cell_line_name) == search
        ]
        
        if not matches.empty:
            model_id = matches.index[0]
            display_name = matches[col].iloc[0]
            
            if model_id in expr_df.index:
                return model_id, display_name
    
    return None, None

def match_all_cell_lines(cell_lines, metadata, expr_df):
    """
    FIXED: Keep HL-60 and HL-60_TB distinct during DepMap matching
    Both will map to same DepMap ID, but stored under different keys
    """
    log.info("Matching all 28 DSRT cell lines to DepMap...")
    
    matched = {}
    unmatched = []
    
    for cell_line in cell_lines:
        model_id, display_name = find_depmap_cell_line(cell_line, metadata, expr_df)
        
        if model_id:
            # FIXED: Keep distinct keys (don't merge HL-60_TB → HL-60 here)
            key = cell_line
            matched[key] = (model_id, display_name)
            log.info(f"  ✓ {cell_line} → {model_id} ({display_name})")
        else:
            unmatched.append(cell_line)
            log.warning(f"  ✗ {cell_line}: Not found")
    
    log.info(f"✓ Matched {len(matched)}/{len(cell_lines)} cell lines")
    
    if unmatched:
        log.warning(f"Unmatched ({len(unmatched)}): {unmatched}")
    
    return matched

# ============================================================================
# Gene Alignment
# ============================================================================

def align_genes(expr_vec, gene_to_col, training_genes):
    aligned = np.zeros(len(training_genes), dtype=np.float32)
    found = 0
    
    for i, gene in enumerate(training_genes):
        if gene in gene_to_col:
            col = gene_to_col[gene]
            if col in expr_vec.index:
                aligned[i] = expr_vec[col]
                found += 1
    
    return aligned, found

# ============================================================================
# Persister Score Computation (FIXED LOG2 INVERSE)
# ============================================================================

def compute_persister_score(expr_vec, gene_to_col, training_genes, model):
    """Compute persister score with CORRECT log2 inverse"""
    aligned, found = align_genes(expr_vec, gene_to_col, training_genes)
    
    if found == 0:
        return None, 0
    
    # FIXED: DepMap is log2(TPM+1) → TPM = 2^x - 1
    tpm_like = np.expm1(aligned * np.log(2.0))  # == (2.0 ** aligned) - 1.0
    tpm_like = np.maximum(tpm_like, 0.0)
    
    # Sanity check TPM values
    log.debug(f"    TPM-like: min={tpm_like.min():.4f}, median={np.median(tpm_like):.4f}, max={tpm_like.max():.4f}")
    
    # Create matrix
    X = tpm_like.reshape(1, -1)
    
    # Apply CPM + log1p
    X_cpm = scrna_cpm_log1p(X)
    log.debug(f"    After CPM+log1p: min={X_cpm.min():.4f}, median={np.median(X_cpm):.4f}, max={X_cpm.max():.4f}")
    
    # Apply full preprocessing
    X_processed = preprocess_data(X)
    
    # Verify shape
    expected_shape = model.input_shape[1]
    if X_processed.shape[1] != expected_shape:
        log.error(f"Shape mismatch! Expected {expected_shape}, got {X_processed.shape[1]}")
        return None, 0
    
    # Predict
    prob = model.predict(X_processed, verbose=0).ravel()[0]
    
    return float(prob * 100), found

def compute_all_scores(matched_lines, expr_df, gene_to_col, training_genes, model):
    log.info("Computing persister scores...")
    
    persister_scores = {}
    
    for cell_line, (model_id, display_name) in matched_lines.items():
        expr_vec = expr_df.loc[model_id]
        
        score, n_genes = compute_persister_score(expr_vec, gene_to_col, training_genes, model)
        
        if score is not None:
            persister_scores[cell_line] = score
            match_pct = 100.0 * n_genes / len(training_genes)
            log.info(f"  {cell_line}: {score:.1f}% ({n_genes}/{len(training_genes)} genes, {match_pct:.1f}%)")
        else:
            log.warning(f"  {cell_line}: Failed")
    
    log.info(f"✓ Computed scores for {len(persister_scores)} cell lines")
    
    return persister_scores

# ============================================================================
# Drug Correlation
# ============================================================================

def load_drug_data():
    log.info("Loading drug screening data...")
    
    try:
        dss_df = pd.read_excel(DRUG_DATA_FILE, sheet_name='DSS', skiprows=2)
        
        # FIXED: Fail fast if column name changed
        drug_col = "Drug name"
        if drug_col not in dss_df.columns:
            raise KeyError(f"Expected '{drug_col}' column. Found: {list(dss_df.columns[:10])}")
        
        log.info(f"✓ Loaded {len(dss_df)} drugs × {len(dss_df.columns)-1} cell lines")
        return dss_df
    except Exception as e:
        log.error(f"Failed to load drug data: {e}")
        sys.exit(1)

def match_drug_columns(persister_scores, dss_df):
    """
    FIXED: Match with HL-60_TB fallback
    Allows using HL-60 score for HL-60_TB DSS column
    """
    matched = {}
    
    # Get HL-60 score for HL-60_TB fallback
    hl60_score = persister_scores.get("HL-60")
    
    for cell_line, score in persister_scores.items():
        variants = [
            cell_line,
            cell_line.replace('-', '_'),
            cell_line.replace('-', ''),
            cell_line.replace('_', '-')
        ]
        
        for variant in variants:
            if variant in dss_df.columns:
                matched[variant] = score
                log.info(f"  ✓ Drug match: {cell_line} → {variant}")
                break
    
    # FIXED: Explicit HL-60_TB fallback (use HL-60 score for HL-60_TB column)
    if hl60_score is not None and "HL-60_TB" in dss_df.columns and "HL-60_TB" not in matched:
        matched["HL-60_TB"] = hl60_score
        log.info("  ✓ Drug match: HL-60 → HL-60_TB (fallback)")
    
    log.info(f"✓ Matched {len(matched)} cell lines with drug data")
    return matched

def compute_correlations_with_fdr(dss_df, persister_scores, min_n=8):
    """Compute with Benjamini-Hochberg FDR"""
    log.info(f"Computing correlations (min N={min_n})...")
    
    results = []
    
    for idx, row in dss_df.iterrows():
        drug_name = row['Drug name']
        
        matched_dss = []
        matched_pers = []
        matched_lines = []
        
        for cell_line, pers_score in persister_scores.items():
            if cell_line in dss_df.columns:
                dss_value = row[cell_line]
                
                if pd.notna(dss_value) and pd.notna(pers_score):
                    matched_dss.append(float(dss_value))
                    matched_pers.append(float(pers_score))
                    matched_lines.append(cell_line)
        
        if len(matched_dss) >= min_n:
            # FIXED: Use np.isclose for constant check
            if np.isclose(np.std(matched_pers), 0) or np.isclose(np.std(matched_dss), 0):
                continue
            
            try:
                spearman_r, spearman_p = spearmanr(matched_pers, matched_dss)
                
                if np.isnan(spearman_r) or np.isnan(spearman_p):
                    continue
                
                results.append({
                    'Drug': drug_name,
                    'Spearman_R': spearman_r,
                    'Spearman_P': spearman_p,
                    'N_lines': len(matched_dss),
                    'Cell_lines': ','.join(matched_lines),
                    'Mean_DSS': np.mean(matched_dss),
                    'Mean_Persister': np.mean(matched_pers)
                })
            except:
                pass
    
    if not results:
        log.error("No correlations computed!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    results_df['Significant_uncorrected'] = results_df['Spearman_P'] < 0.05
    
    # Bonferroni (NOT "FDR")
    results_df['Bonferroni_P'] = np.minimum(results_df['Spearman_P'] * len(results_df), 1.0)
    results_df['Bonferroni_sig'] = results_df['Bonferroni_P'] < 0.05
    
    # Benjamini-Hochberg FDR (proper FDR)
    reject, pvals_corrected, _, _ = multipletests(results_df['Spearman_P'], alpha=0.05, method='fdr_bh')
    results_df['BH_FDR'] = pvals_corrected
    results_df['BH_FDR_sig'] = reject
    
    # Sort ascending for easy resister selection
    results_df = results_df.sort_values('Spearman_R', ascending=True)
    
    log.info(f"✓ Computed correlations for {len(results_df)} drugs")
    log.info(f"  Uncorrected p<0.05: {results_df['Significant_uncorrected'].sum()}")
    log.info(f"  Bonferroni p<0.05: {results_df['Bonferroni_sig'].sum()}")
    log.info(f"  BH FDR<0.05: {results_df['BH_FDR_sig'].sum()}")
    
    return results_df

# ============================================================================
# Output
# ============================================================================

def save_outputs(results_df, persister_scores):
    results_df.to_csv(OUTPUT_DIR / 'drug_persister_correlations.csv', index=False)
    
    pd.DataFrame([
        {'Cell_Line': k, 'Persister_Pct': v}
        for k, v in sorted(persister_scores.items(), key=lambda x: x[1], reverse=True)
    ]).to_csv(OUTPUT_DIR / 'persister_scores.csv', index=False)
    
    # FIXED: Killers (positive R, sort descending)
    killers = results_df[(results_df['Spearman_R'] > 0.3) & (results_df['BH_FDR_sig'])].sort_values('Spearman_R', ascending=False).head(50)
    killers.to_csv(OUTPUT_DIR / 'persister_killing_drugs.csv', index=False)
    
    # FIXED: Resisters (negative R, already sorted ascending so head() works)
    resisters = results_df[(results_df['Spearman_R'] < -0.3) & (results_df['BH_FDR_sig'])].head(50)
    resisters.to_csv(OUTPUT_DIR / 'persister_resistant_drugs.csv', index=False)
    
    log.info(f"✓ Saved: persister_killing_drugs.csv ({len(killers)} drugs)")
    log.info(f"✓ Saved: persister_resistant_drugs.csv ({len(resisters)} drugs)")

def print_summary(results_df, persister_scores):
    killers = results_df[(results_df['Spearman_R'] > 0.3) & (results_df['BH_FDR_sig'])].sort_values('Spearman_R', ascending=False)
    resisters = results_df[(results_df['Spearman_R'] < -0.3) & (results_df['BH_FDR_sig'])].sort_values('Spearman_R', ascending=True)
    
    print("\n" + "="*80)
    print("DRUG-PERSISTER CORRELATION RESULTS")
    print("="*80)
    
    print(f"\nCell Lines Analyzed: {len(persister_scores)}")
    for cl, score in sorted(persister_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cl:20s}: {score:6.1f}%")
    
    print(f"\nDrugs Analyzed: {len(results_df)} (min N={MIN_N_FOR_CORRELATION})")
    print(f"Significant (uncorrected p<0.05): {results_df['Significant_uncorrected'].sum()}")
    print(f"Significant (Bonferroni p<0.05): {results_df['Bonferroni_sig'].sum()}")
    print(f"Significant (BH FDR<0.05): {results_df['BH_FDR_sig'].sum()}")
    
    print("\n" + "-"*80)
    print("PERSISTER-KILLING DRUGS (Positive Correlation, BH FDR<0.05)")
    print("-"*80)
    
    if not killers.empty:
        print(f"Found {len(killers)} drugs\n")
        print(f"{'Rank':<6} {'Drug':<35} {'R':>8} {'P':>10} {'FDR':>10} {'N':>4}")
        print("-"*80)
        for rank, (_, row) in enumerate(killers.head(20).iterrows(), 1):
            print(f"{rank:<6} {row['Drug']:<35} {row['Spearman_R']:>8.3f} {row['Spearman_P']:>10.4f} {row['BH_FDR']:>10.4f} {row['N_lines']:>4}")
    else:
        print("None found")
    
    print("\n" + "-"*80)
    print("PERSISTER-RESISTANT DRUGS (Negative Correlation, BH FDR<0.05)")
    print("-"*80)
    
    if not resisters.empty:
        print(f"Found {len(resisters)} drugs\n")
        print(f"{'Rank':<6} {'Drug':<35} {'R':>8} {'P':>10} {'FDR':>10} {'N':>4}")
        print("-"*80)
        for rank, (_, row) in enumerate(resisters.head(20).iterrows(), 1):
            print(f"{rank:<6} {row['Drug']:<35} {row['Spearman_R']:>8.3f} {row['Spearman_P']:>10.4f} {row['BH_FDR']:>10.4f} {row['N_lines']:>4}")
    else:
        print("None found")

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("DRUG-PERSISTER CORRELATION (ALL CRITICAL BUGS FIXED)")
    print("="*80)
    
    expr_df, gene_to_col = load_depmap_expression_efficient()
    metadata = load_depmap_metadata()
    
    matched_lines = match_all_cell_lines(ALL_28_CELL_LINES, metadata, expr_df)
    
    if len(matched_lines) < MIN_N_FOR_CORRELATION:
        log.error(f"Only {len(matched_lines)} matched. Need {MIN_N_FOR_CORRELATION}.")
        sys.exit(1)
    
    persister_scores = compute_all_scores(matched_lines, expr_df, gene_to_col, TRAINING_GENES, model)
    
    if len(persister_scores) < MIN_N_FOR_CORRELATION:
        log.error(f"Only {len(persister_scores)} scores. Need {MIN_N_FOR_CORRELATION}.")
        sys.exit(1)
    
    dss_df = load_drug_data()
    matched_drug_scores = match_drug_columns(persister_scores, dss_df)
    
    if len(matched_drug_scores) < MIN_N_FOR_CORRELATION:
        log.error(f"Only {len(matched_drug_scores)} matched with drug data.")
        sys.exit(1)
    
    results_df = compute_correlations_with_fdr(dss_df, matched_drug_scores, MIN_N_FOR_CORRELATION)
    
    if results_df.empty:
        log.error("No correlations!")
        sys.exit(1)
    
    save_outputs(results_df, persister_scores)
    print_summary(results_df, persister_scores)
    
    print(f"\n✓ Results: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
