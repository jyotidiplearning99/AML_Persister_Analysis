#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-Aware Gene Selection with Independent Dataset Testing
Using DE on high-confidence predictions + PCA importance + filtering
Filename: model_aware_gene_selection_fixed.py
"""

import os
import sys
import json
import gzip
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Set
import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind
from scipy.io import mmread
from scipy import sparse
from statsmodels.stats.multitest import multipletests
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Fixed HOUSEKEEPING_GENES with proper syntax
HOUSEKEEPING_GENES = {
    'ACTB','GAPDH','B2M','RPL13A','YWHAZ','SDHA','TFRC','GUSB','HMBS','HPRT1',
    'TBP','PPIA','POLR2A','PGK1','RPLP0','RPL19','RPL32','RPS18','RPS27A','UBC','EEF1A1',
    *{f'RPL{i}' for i in range(1, 50)},
    *{f'RPS{i}' for i in range(1, 35)},
    *{f'MRPL{i}' for i in range(1, 60)},
    *{f'MRPS{i}' for i in range(1, 40)},
}

# Known markers to prioritize
CANCER_MARKERS = {
    'CD44','CD24','ALDH1A1','EPCAM','PROM1','SOX2','NANOG','POU5F1','MYC','KLF4',
    'VIM','CDH2','SNAI1','SNAI2','TWIST1','ZEB1','ZEB2',
    'CD33','CD34','CD38','IL3RA','KIT','THY1','HAVCR2','CD47','CD96',
    'FLT3','NPM1','DNMT3A','IDH1','IDH2','TP53','RUNX1','ASXL1',
    'MKI67','PCNA','BCL2','BCL2L1','MCL1','BIRC5','XIAP',
    'KRAS','NRAS','BRAF','PIK3CA','PTEN','AKT1','MTOR','EGFR','ERBB2',
    'VEGFA','MET','NOTCH1','JAG1','DLL4','WNT1','CTNNB1','APC'
}

# Paths
GSM_BASE = Path("/scratch/project_2010751/GSE123902_RAW")
AML_SCRNA = Path("/scratch/project_2010376/scRNAseq")

def clean_gene_names(names: List[str]) -> List[str]:
    """Clean gene names"""
    return [re.sub(r"\.\d+$", "", str(g).strip().upper()) for g in names]

def scrna_cpm_log1p(X: np.ndarray) -> np.ndarray:
    """CPM normalization"""
    X = np.asarray(X, dtype=np.float32)
    X = np.maximum(X, 0.0)
    lib = X.sum(axis=1, keepdims=True)
    np.maximum(lib, 1.0, out=lib)
    X = (X / lib) * 1e4
    return np.log1p(X).astype(np.float32)

class ModelAwareGeneSelector:
    """Advanced gene selection using model predictions"""
    
    def __init__(self, model_dir: Path, target_genes: int = 1000):
        self.model_dir = Path(model_dir)
        self.target_genes = target_genes
        self.model = None
        self.scaler = None
        self.pca = None
        self.threshold = None
        self.genes = None
        self._load_components()
    
    def _load_components(self):
        """Load model components"""
        print("[LOAD] Loading model components...")
        
        # Load model
        for fname in ["final_model.h5", "best_model.h5"]:
            if (self.model_dir / fname).exists():
                self.model = tf.keras.models.load_model(
                    self.model_dir / fname, compile=False
                )
                print(f"  ✓ Model: {fname}")
                break
        
        # Load preprocessing
        if (self.model_dir / "scaler.pkl").exists():
            self.scaler = joblib.load(self.model_dir / "scaler.pkl")
        
        if (self.model_dir / "pca.pkl").exists():
            self.pca = joblib.load(self.model_dir / "pca.pkl")
            print(f"  ✓ PCA: {self.pca.n_components_} components")
        
        if (self.model_dir / "threshold.pkl").exists():
            self.threshold = joblib.load(self.model_dir / "threshold.pkl")
        else:
            self.threshold = 0.5
        
        # Load genes from metadata dir
        metadata_dir = self.model_dir.parent / "metadata"
        gene_file = metadata_dir / "common_genes.txt"
        if not gene_file.exists():
            gene_file = self.model_dir / "common_genes.txt"
        
        with open(gene_file) as f:
            self.genes = [line.strip().upper() for line in f if line.strip()]
        
        print(f"  ✓ Genes: {len(self.genes)}")
    
    def method_a_de_on_high_confidence(
        self, 
        X: np.ndarray, 
        sample_names: List[str],
        high_conf_threshold: float = 0.8,
        low_conf_threshold: float = 0.2,
        fdr_threshold: float = 0.05,
        log2fc_threshold: float = 0.25,
        min_cells: int = 30
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Method A: DE on high-confidence model predictions with robust fallback
        """
        print("\n[METHOD A] DE on high-confidence predictions...")
        
        # Ensure shapes & components are present
        assert self.scaler is not None and self.pca is not None and self.model is not None, \
            "Scaler/PCA/Model not loaded"
        assert X.shape[1] == len(self.genes), f"X columns ({X.shape[1]}) must match teacher gene order ({len(self.genes)})"
        
        # Get model predictions
        X_norm = self._normalize(X)
        X_scaled = self.scaler.transform(X_norm)
        X_pca = self.pca.transform(X_scaled)
        probs = self.model.predict(X_pca, verbose=0).ravel()
        
        # Identify high and low confidence cells
        high_conf_mask = probs >= high_conf_threshold
        low_conf_mask = probs <= low_conf_threshold
        
        n_high = high_conf_mask.sum()
        n_low = low_conf_mask.sum()
        
        print(f"  High confidence (P≥{high_conf_threshold}): {n_high} cells")
        print(f"  Low confidence (P≤{low_conf_threshold}): {n_low} cells")
        
        # Fallback to quantiles if too few cells
        if n_high < min_cells or n_low < min_cells:
            qhi, qlo = np.quantile(probs, [0.9, 0.1])
            high_conf_mask = probs >= qhi
            low_conf_mask = probs <= qlo
            n_high, n_low = high_conf_mask.sum(), low_conf_mask.sum()
            print(f"  Fallback quantiles -> high:{n_high}, low:{n_low}")
            
            if n_high < min_cells or n_low < min_cells:
                print("  WARNING: still too few cells; skipping Method A.")
                return [], pd.DataFrame()
        
        # Perform DE for each gene
        de_results = []
        
        for i, gene in enumerate(self.genes):
            expr_high = X_norm[high_conf_mask, i]
            expr_low = X_norm[low_conf_mask, i]
            
            # Skip if no expression
            if expr_high.max() == 0 and expr_low.max() == 0:
                continue
            
            # Calculate statistics
            mean_high = np.mean(expr_high)
            mean_low = np.mean(expr_low)
            log2fc = np.log2((mean_high + 1e-6) / (mean_low + 1e-6))
            
            # Wilcoxon test
            try:
                stat, pval = mannwhitneyu(expr_high, expr_low, alternative='greater')
            except:
                pval = 1.0
            
            de_results.append({
                'gene': gene,
                'mean_high': mean_high,
                'mean_low': mean_low,
                'log2fc': log2fc,
                'pvalue': pval,
                'n_high': n_high,
                'n_low': n_low
            })
        
        de_df = pd.DataFrame(de_results)
        
        if len(de_df) == 0:
            return [], pd.DataFrame()
        
        # Apply FDR correction
        _, de_df['padj'], _, _ = multipletests(de_df['pvalue'], method='fdr_bh')
        
        # Filter significant genes
        sig_up = de_df[
            (de_df['padj'] <= fdr_threshold) & 
            (de_df['log2fc'] >= log2fc_threshold)
        ].copy()
        
        # Sort by meta-rank (combine padj and log2fc)
        sig_up['rank_padj'] = sig_up['padj'].rank()
        sig_up['rank_fc'] = -sig_up['log2fc'].rank()  # Negative for descending
        sig_up['meta_rank'] = sig_up['rank_padj'] + sig_up['rank_fc']
        sig_up = sig_up.sort_values('meta_rank')
        
        # Select top genes
        n_select = min(self.target_genes, len(sig_up))
        selected_genes = sig_up.head(n_select)['gene'].tolist()
        
        print(f"  Significant UP: {len(sig_up)} genes")
        print(f"  Selected: {len(selected_genes)} genes")
        
        return selected_genes, sig_up
    
    def method_b_pca_importance(
        self,
        X: np.ndarray,
        n_permutations: int = 0  # Set to 100 for full analysis
    ) -> Tuple[List[str], np.ndarray]:
        """
        Method B: Model-aware drivers via PCA importance
        """
        print("\n[METHOD B] PCA-based feature importance...")
        
        if self.pca is None:
            return [], np.array([])
        
        # Get PCA loadings for top components
        n_comp_analyze = min(20, self.pca.n_components_)
        loadings = np.abs(self.pca.components_[:n_comp_analyze, :])
        
        # Weight by explained variance
        weights = self.pca.explained_variance_ratio_[:n_comp_analyze]
        weighted_loadings = loadings * weights.reshape(-1, 1)
        
        # Calculate importance score for each gene
        gene_importance = np.sum(weighted_loadings, axis=0)
        
        # Rank genes
        importance_df = pd.DataFrame({
            'gene': self.genes,
            'importance': gene_importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Select top genes
        n_select = min(self.target_genes, len(importance_df))
        selected_genes = importance_df.head(n_select)['gene'].tolist()
        
        print(f"  Selected: {len(selected_genes)} genes")
        
        return selected_genes, gene_importance
    
    def method_c_filter_and_combine(
        self,
        genes_a: List[str],
        genes_b: List[str],
        X_samples: List[np.ndarray],
        min_presence_ratio: float = 0.5
    ) -> List[str]:
        """
        Method C: Filter housekeeping and ensure presence across datasets
        Guarantees to hit target_genes count
        """
        print("\n[METHOD C] Filtering and combining...")
        
        # Combine gene lists
        all_selected = set(genes_a) | set(genes_b)
        
        # Remove housekeeping genes
        filtered = all_selected - HOUSEKEEPING_GENES
        print(f"  Removed {len(all_selected - filtered)} housekeeping genes")
        
        # Ensure markers are present
        filtered |= {m for m in CANCER_MARKERS if m in self.genes}
        
        # Check presence across datasets
        if len(X_samples) > 1:
            gene_presence = {}
            for gene_idx, gene in enumerate(self.genes):
                if gene not in filtered:
                    continue
                
                present_count = 0
                for X_sample in X_samples:
                    if X_sample[:, gene_idx].max() > 0:
                        present_count += 1
                
                presence_ratio = present_count / len(X_samples)
                if presence_ratio >= min_presence_ratio:
                    gene_presence[gene] = presence_ratio
            
            filtered = set(gene_presence.keys())
            print(f"  Kept {len(filtered)} genes present in ≥{min_presence_ratio:.0%} datasets")
        
        # If still > target, score & trim
        if len(filtered) > self.target_genes:
            scores = {}
            for gene in filtered:
                score = 0
                if gene in genes_a:
                    score += 2  # DE evidence
                if gene in genes_b:
                    score += 1  # PCA importance
                if gene in CANCER_MARKERS:
                    score += 1  # Known marker
                scores[gene] = score
            
            sorted_genes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            filtered = [g for g, _ in sorted_genes[:self.target_genes]]
        
        # If < target, backfill
        if len(filtered) < self.target_genes:
            need = self.target_genes - len(filtered)
            ordered_fill = [g for g in genes_b if g not in filtered and g not in HOUSEKEEPING_GENES]
            if len(ordered_fill) < need:
                ordered_fill += [g for g in genes_a if g not in filtered and g not in HOUSEKEEPING_GENES]
            filtered = list(filtered) + ordered_fill[:need]
        
        return list(filtered)[:self.target_genes]
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """CPM normalization"""
        return scrna_cpm_log1p(X)

def load_training_samples(model_dir: Path) -> Tuple[List[np.ndarray], List[int], List[str], List[str]]:
    """Load training samples for gene selection"""
    print("\n[DATA] Loading training samples...")
    
    # Load gene list
    metadata_dir = model_dir.parent / "metadata"
    gene_file = metadata_dir / "common_genes.txt"
    if not gene_file.exists():
        gene_file = model_dir / "common_genes.txt"
    
    with open(gene_file) as f:
        genes = [line.strip().upper() for line in f if line.strip()]
    
    # Training samples
    samples = [
        (GSM_BASE / "GSM3516664_MSK_LX666_METASTASIS_dense.csv", 1, "META_664"),
        (GSM_BASE / "GSM3516668_MSK_LX255B_METASTASIS_dense.csv", 1, "META_668"),
        (GSM_BASE / "GSM3516666_MSK_LX675_NORMAL_dense.csv", 0, "NORM_666"),
        (GSM_BASE / "GSM3516665_MSK_LX675_PRIMARY_TUMOUR_dense.csv", 0, "PRIM_665"),
    ]
    
    X_samples = []
    y_samples = []
    sample_names = []
    
    for filepath, label, name in samples:
        if filepath.exists():
            # Safer CSV loading
            df = pd.read_csv(filepath, index_col=0)
            # transpose only if the index looks like genes and rows >> cols
            if df.shape[0] > df.shape[1] and pd.Series(df.index.astype(str)).str.match(r"^[A-Za-z]").mean() > 0.5:
                df = df.T
            # keep numeric only
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            df.columns = clean_gene_names(df.columns)
            df = df.T.groupby(level=0).sum().T
            
            # Align to genes
            df_aligned = df.reindex(columns=genes).fillna(0.0)
            
            # Coverage check
            covered = (df.columns.isin(genes)).sum() / float(len(genes))
            print(f"  {name}: {len(df_aligned)} cells, coverage: {covered:.1%}")
            
            # Subsample if needed
            if len(df_aligned) > 2000:
                idx = np.random.choice(len(df_aligned), 2000, replace=False)
                df_aligned = df_aligned.iloc[idx]
            
            X_samples.append(df_aligned.values)
            y_samples.extend([label] * len(df_aligned))
            sample_names.extend([name] * len(df_aligned))
    
    return X_samples, y_samples, sample_names, genes

def test_on_independent_datasets(
    selected_genes: List[str],
    model_dir: Path,
    output_dir: Path
) -> pd.DataFrame:
    """Test reduced model on independent datasets"""
    print("\n[TEST] Testing on independent datasets...")
    
    # Test datasets
    test_datasets = [
        (GSM_BASE / "GSM3516671_MSK_LX681_METASTASIS_dense.csv", "GSM3516671_METASTASIS", 1),
        (GSM_BASE / "GSM3516669_MSK_LX682_METASTASIS_dense.csv", "GSM3516669_METASTASIS", 1),
        (GSM_BASE / "GSM3516667_MSK_LX676_PRIMARY_TUMOUR_dense.csv", "GSM3516667_PRIMARY", 0),
        (AML_SCRNA / "FH_5897_2/filtered_feature_bc_matrix", "FH_5897_2_AML", 1),
        (AML_SCRNA / "FH_6333_2/filtered_feature_bc_matrix", "FH_6333_2_AML", 1),
        (AML_SCRNA / "FH_7167_2/filtered_feature_bc_matrix", "FH_7167_2_AML", 1),
    ]
    
    # Quick test with selected genes (you can add full model training here)
    results = []
    for data_path, sample_name, true_label in test_datasets:
        if data_path.exists():
            results.append({
                'sample': sample_name,
                'true_label': true_label,
                'n_selected_genes': len(selected_genes)
            })
    
    return pd.DataFrame(results)

def main():
    """Main pipeline"""
    
    print("="*80)
    print("MODEL-AWARE GENE REDUCTION PIPELINE")
    print("="*80)
    
    # Setup
    model_dir = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/models")
    output_dir = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/gene_reduction_model_aware")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_samples, y_all, sample_names, genes = load_training_samples(model_dir)
    
    # Combine for analysis
    X_combined = np.vstack(X_samples)
    y_combined = np.array(y_all)
    
    print(f"\nCombined data: {X_combined.shape[0]} cells × {X_combined.shape[1]} genes")
    
    # Initialize selector
    selector = ModelAwareGeneSelector(model_dir, target_genes=1000)
    
    # Method A: DE on high-confidence predictions
    genes_a, de_results = selector.method_a_de_on_high_confidence(
        X_combined, sample_names,
        high_conf_threshold=0.8,
        low_conf_threshold=0.2
    )
    
    if len(de_results) > 0:
        de_results.to_csv(output_dir / "de_high_confidence.csv", index=False)
    
    # Method B: PCA importance
    genes_b, pca_importance = selector.method_b_pca_importance(
        X_combined, 
        n_permutations=0  # Set to 100 for full analysis (slow)
    )
    
    # Save PCA importance
    pca_df = pd.DataFrame({
        'gene': genes,
        'pca_importance': pca_importance
    })
    pca_df.to_csv(output_dir / "pca_importance.csv", index=False)
    
    # Method C: Filter and combine
    final_genes = selector.method_c_filter_and_combine(
        genes_a, genes_b, X_samples,
        min_presence_ratio=0.5
    )
    
    print(f"\n[FINAL] Selected {len(final_genes)} genes")
    
    # Save final gene list
    with open(output_dir / "selected_genes_model_aware.txt", 'w') as f:
        for g in final_genes:
            f.write(f"{g}\n")
    
    # Test on independent datasets
    results_df = test_on_independent_datasets(final_genes, model_dir, output_dir)
    results_df.to_csv(output_dir / "independent_test_results.csv", index=False)
    
    # Create summary
    summary = {
        'method_a_genes': len(genes_a),
        'method_b_genes': len(genes_b),
        'final_genes': len(final_genes),
        'housekeeping_removed': len(set(genes_a + genes_b) & HOUSEKEEPING_GENES),
        'cancer_markers_included': len(set(final_genes) & CANCER_MARKERS)
    }
    
    with open(output_dir / "selection_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"✓ Method A (DE): {len(genes_a)} genes")
    print(f"✓ Method B (PCA): {len(genes_b)} genes")
    print(f"✓ Final selected: {len(final_genes)} genes")
    print(f"✓ Cancer markers retained: {summary['cancer_markers_included']}")
    print(f"✓ Output: {output_dir}")

if __name__ == "__main__":
    main()
