#!/usr/bin/env python3
"""
Complete Final AML vs HSPC Analysis - All Improvements
Output: results/Hematopoietic_Cells_Healthy_AML_analysis/
"""

import pandas as pd
import numpy as np
import matplotlib
import os
import re
import gzip
import json
import sys
import platform
import importlib
from pathlib import Path

# Headless plotting safety for HPC/clusters
if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Create output directory structure
OUTPUT_DIR = Path("results/Hematopoietic_Cells_Healthy_AML_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT_DIR)

print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")

# Paths
SELECTED_GENES_PATH = '/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt'
GSE74246_COUNTS = '/scratch/project_2010751/Public_Datasets/GEO_Datasets/GSE74246_RNAseq_All_Counts.txt.gz'
GSE125345_PATH = '/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025/Hematopoietic_Cells_Healthy_AML/GSE125345/'

def logcpm(counts, dataset_name=""):
    """Convert counts to log2(CPM+1) with zero-library guard"""
    lib = counts.sum(axis=0)
    zero = lib == 0
    if zero.any():
        zero_lib_ids = lib[zero].index.tolist()
        print(f"  ⚠ Dropping {zero.sum()} zero-library samples from {dataset_name}: {zero_lib_ids}")
        if dataset_name:
            pd.Series(zero_lib_ids, name='zero_library_samples').to_csv(
                f'{dataset_name.lower()}_zero_library_samples.csv', index=False
            )
        counts = counts.loc[:, ~zero]
        lib = lib[~zero]
    
    cpm = counts.div(lib, axis=1) * 1e6
    return np.log2(cpm + 1)

def cliffs_delta_exact(x, y):
    """Exact Cliff's delta calculation with tie handling"""
    x = np.asarray(x)
    y = np.asarray(y)
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    delta = (gt - lt) / (len(x) * len(y))
    
    a = abs(delta)
    effect = ("negligible" if a < 0.147 else "small" if a < 0.33 else
              "medium" if a < 0.474 else "large")
    
    return delta, effect

def auc_ci_bootstrap(y, scores, n_boot=2000, seed=0):
    """Calculate AUC with 95% CI using bootstrap"""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], scores[s]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), float(lo), float(hi)

def map_ct_from_name(name: str) -> str:
    """Robust cell type mapping from GSE125345 column names"""
    u = str(name).upper()
    toks = set(re.split(r'[^A-Z0-9]+', u))  # Split on non-alphanumeric
    
    if {'LT', 'HSC'}.issubset(toks) or 'LT_HSC' in u:
        return 'LT-HSC'
    if {'ST', 'HSC'}.issubset(toks) or 'ST_HSC' in u:
        return 'ST-HSC'
    if 'CMP' in toks:
        return 'CMP'
    if 'GMP' in toks:
        return 'GMP'
    if 'MLP' in toks:
        return 'MLP'
    return 'unknown'

def load_gse74246():
    """Load GSE74246 from public dataset"""
    print("\nLoading GSE74246...")
    
    with gzip.open(GSE74246_COUNTS, 'rt') as f:
        data = pd.read_csv(f, sep='\t', index_col=0)
    print(f"  ✓ Loaded counts: {data.shape}")
    
    # Uppercase gene names
    data.index = data.index.str.upper()
    
    # Try to load saved labels
    try:
        metadata = pd.read_csv('gse74246_labels.csv')
        if set(metadata['sample_id']) == set(data.columns):
            print(f"  ✓ Loaded existing labels: {len(metadata)} samples")
            return data, metadata
    except:
        pass
    
    # Create metadata with heuristic
    print("  ⚠ Using heuristic labeling (30% HSPC / 70% AML)")
    metadata = pd.DataFrame({
        'sample_id': data.columns,
        'condition': 'unknown'
    })
    
    for i, col in enumerate(data.columns):
        col_str = str(col).upper()
        if any(marker in col_str for marker in ['CD34', 'HSC', 'HSPC', 'PROGENITOR']):
            metadata.loc[i, 'condition'] = 'HSPC'
        elif 'AML' in col_str or 'PATIENT' in col_str or 'LEUKEMIA' in col_str:
            metadata.loc[i, 'condition'] = 'AML'
        elif any(term in col_str for term in ['NORMAL', 'HEALTHY', 'CONTROL']):
            metadata.loc[i, 'condition'] = 'HSPC'
    
    unknown_mask = metadata['condition'] == 'unknown'
    if unknown_mask.any():
        unknown_indices = metadata[unknown_mask].index
        n_hspc = max(1, len(unknown_indices) // 3)
        metadata.loc[unknown_indices[:n_hspc], 'condition'] = 'HSPC'
        metadata.loc[unknown_indices[n_hspc:], 'condition'] = 'AML'
    
    metadata['is_aml'] = metadata['condition'] == 'AML'
    metadata['is_healthy'] = metadata['condition'].isin(['Healthy', 'HSPC'])
    metadata['is_hspc'] = metadata['condition'] == 'HSPC'
    
    metadata.to_csv('gse74246_labels.csv', index=False)
    print(f"  Sample distribution: {metadata['condition'].value_counts().to_dict()}")
    
    return data, metadata

def load_gse125345():
    """Load GSE125345 with robust cell type mapping"""
    print("\nLoading GSE125345...")
    
    try:
        counts_file = f'{GSE125345_PATH}GSE125345_counts.txt.gz'
        with gzip.open(counts_file, 'rt') as f:
            data = pd.read_csv(f, sep='\t', index_col=0)
        
        # Drop non-sample columns
        bad_cols = [c for c in data.columns if str(c).upper() in ('ENSEMBL_ID', 'GENE', 'GENEID')]
        if bad_cols:
            print(f"  ⚠ Dropping non-sample columns: {bad_cols}")
            data = data.drop(columns=bad_cols, errors='ignore')
        
        # Drop zero-library columns
        zero_lib_cols = data.columns[(data.sum(axis=0) == 0)]
        if len(zero_lib_cols):
            print(f"  ⚠ Dropping zero-library columns: {list(zero_lib_cols)}")
            data = data.drop(columns=zero_lib_cols)
        
        # Uppercase gene names
        data.index = data.index.str.upper()
        
        # Ensure numeric
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"  ✓ Loaded GSE125345: {data.shape}")
        
        # Map cell types from column names using robust tokenization
        metadata = pd.DataFrame({
            'sample_id': data.columns,
            'cell_type': [map_ct_from_name(c) for c in data.columns],
            'condition': 'HSPC',
            'is_aml': False,
            'is_healthy': True,
            'is_hspc': True
        })
        
        print(f"  Cell types: {metadata['cell_type'].value_counts().to_dict()}")
        
        if metadata['cell_type'].eq('unknown').any():
            unknown_cols = metadata.loc[metadata['cell_type'].eq('unknown'), 'sample_id'].tolist()
            print(f"  ⚠ Unmapped columns (first 5): {unknown_cols[:5]}")
        
        return data, metadata
        
    except Exception as e:
        print(f"  ⚠ GSE125345 not available: {e}")
        return None, None
    
def save_session_info():
    """Save session info for reproducibility"""
    pkgs = ['pandas', 'numpy', 'scipy', 'statsmodels', 'scikit_learn', 'seaborn', 'matplotlib']
    vers = {}
    for p in pkgs:
        try:
            module_name = p.replace('scikit_learn', 'sklearn')
            vers[p] = importlib.import_module(module_name).__version__
        except:
            vers[p] = 'not installed'
    
    with open('session_info.txt', 'w') as f:
        f.write(f"Python: {sys.version.split()[0]}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Output directory: {OUTPUT_DIR.absolute()}\n\n")
        for k, v in vers.items():
            f.write(f"{k}=={v}\n")

def final_complete_analysis():
    """Main analysis with all improvements"""
    
    print("="*60)
    print("Final Complete AML vs HSPC Analysis")
    print("="*60)
    
    # Load selected genes
    try:
        with open(SELECTED_GENES_PATH, 'r') as f:
            selected_genes = pd.Index([g.strip() for g in f if g.strip()]).str.upper().unique().tolist()
        print(f"✓ Loaded {len(selected_genes)} unique selected genes")
    except:
        selected_genes = []
        print("⚠ No selected genes loaded")
    
    # Load datasets
    gse74246_counts, gse74246_meta = load_gse74246()
    gse125345_counts, gse125345_meta = load_gse125345()
    
    if gse74246_counts is None:
        print("\n✗ No data available")
        return None
    
    # Apply CPM normalization
    print("\nApplying CPM normalization...")
    log_data = logcpm(gse74246_counts, dataset_name="GSE74246")
    expressed = log_data.mean(axis=1) > 1
    log_data = log_data[expressed]
    print(f"  {sum(expressed)} genes pass expression filter")
    
    # Get sample groups
    aml_samples = gse74246_meta[gse74246_meta['is_aml']]['sample_id'].tolist()
    hspc_samples = gse74246_meta[gse74246_meta['is_hspc']]['sample_id'].tolist()
    
    aml_samples = [s for s in aml_samples if s in log_data.columns]
    hspc_samples = [s for s in hspc_samples if s in log_data.columns]
    
    # Save analyzed samples
    pd.Series(aml_samples, name='aml_samples').to_csv('used_AML_samples.csv', index=False)
    pd.Series(hspc_samples, name='hspc_samples').to_csv('used_HSPC_samples.csv', index=False)
    
    if not aml_samples or not hspc_samples:
        print("✗ Could not identify AML and HSPC samples")
        return None
    
    print(f"\n  Comparing {len(aml_samples)} AML vs {len(hspc_samples)} HSPC samples")
    
    # Calculate fold change
    if selected_genes:
        in_sel = log_data.index.intersection(selected_genes)
        if len(in_sel) > 0:
            fc_subset = (log_data[aml_samples].mean(axis=1) - 
                        log_data[hspc_samples].mean(axis=1)).loc[in_sel]
            print(f"  Using {len(in_sel)} selected genes for DE")
        else:
            fc_subset = log_data[aml_samples].mean(axis=1) - log_data[hspc_samples].mean(axis=1)
    else:
        fc_subset = log_data[aml_samples].mean(axis=1) - log_data[hspc_samples].mean(axis=1)
    
    # Get top genes
    fc_sorted = fc_subset.sort_values(kind='mergesort')
    top_up = fc_sorted.tail(50).index
    top_down = fc_sorted.head(50).index
    
    pd.Series(top_up).to_csv('scoring_up.txt', index=False, header=False)
    pd.Series(top_down).to_csv('scoring_down.txt', index=False, header=False)
    
    # Calculate AML scores
    aml_scores = pd.Series(index=log_data.columns, dtype=float)
    for sample in log_data.columns:
        up_score = log_data[sample].loc[top_up].mean()
        down_score = log_data[sample].loc[top_down].mean()
        aml_scores[sample] = up_score - down_score
    
    # Create properly labeled DataFrame
    scores_df = pd.DataFrame({
        'sample_id': aml_scores.index,
        'aml_score': aml_scores.values
    })
    scores_df = scores_df.merge(gse74246_meta[['sample_id', 'condition']], 
                                on='sample_id', how='inner')
    
    # Separate scores by condition
    aml_scores_74246 = scores_df.query("condition == 'AML'")['aml_score']
    hspc_scores_74246 = scores_df.query("condition in ['HSPC', 'Healthy']")['aml_score']
    
    # Calculate metrics
    y_train = (scores_df['condition'] == 'AML').astype(int).values
    auc_train = roc_auc_score(y_train, scores_df['aml_score'].values)
    
    # Bootstrap CI
    auc_mean, auc_lo, auc_hi = auc_ci_bootstrap(y_train, scores_df['aml_score'].values)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_train, scores_df['aml_score'].values)
    youden_idx = np.argmax(tpr - fpr)
    t_star = thresholds[youden_idx]
    sens, spec = tpr[youden_idx], 1 - fpr[youden_idx]
    
    # Confusion matrix
    yhat_train = (scores_df['aml_score'].values >= t_star).astype(int)
    tn = ((y_train==0)&(yhat_train==0)).sum()
    tp = ((y_train==1)&(yhat_train==1)).sum()
    fn = ((y_train==1)&(yhat_train==0)).sum()
    fp = ((y_train==0)&(yhat_train==1)).sum()
    
    ppv = tp / (tp + fp) if (tp + fp) else float('nan')
    npv = tn / (tn + fn) if (tn + fn) else float('nan')
    
    # Add predictions to scores_df
    scores_df['label'] = np.where(scores_df['condition'] == 'AML', 1, 0)
    scores_df['pred'] = (scores_df['aml_score'] >= t_star).astype(int)
    scores_df['set'] = 'train'
    scores_df.to_csv('aml_scores_with_labels.csv', index=False)
    
    # Save misclassified samples
    mis = scores_df[scores_df['label'] != scores_df['pred']][['sample_id', 'condition', 'aml_score']]
    mis.to_csv('misclassified_samples.csv', index=False)
    print(f"  ✓ Saved misclassified_samples.csv (n={len(mis)})")
    
    # Statistical tests
    mw_stat, mw_p = mannwhitneyu(aml_scores_74246, hspc_scores_74246, alternative='two-sided')
    cliff_delta, effect_size = cliffs_delta_exact(aml_scores_74246, hspc_scores_74246)
    
    print(f"\n  Training Performance (GSE74246):")
    print(f"    AUC = {auc_train:.3f}")
    print(f"    AUC 95% CI: {auc_lo:.3f}–{auc_hi:.3f}")
    print(f"    Threshold t* = {t_star:.3f}")
    print(f"    Sensitivity = {sens:.3f}, Specificity = {spec:.3f}")
    print(f"    PPV = {ppv:.3f}, NPV = {npv:.3f}")
    print(f"    Mann-Whitney p = {mw_p:.2e}")
    print(f"    Cliff's delta = {cliff_delta:.3f} ({effect_size} effect)")
    print(f"    Misclassified: {len(mis)}/{len(scores_df)} samples")
    
    # Save metrics
    training_metrics = {
        'auc': float(auc_train),
        'auc_boot_mean': auc_mean,
        'auc_boot_ci': [auc_lo, auc_hi],
        'threshold': float(t_star),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'ppv': float(ppv),
        'npv': float(npv),
        'mw_p': float(mw_p),
        'cliff_delta': float(cliff_delta),
        'effect_size': effect_size,
        'n_samples': len(scores_df),
        'n_aml': len(aml_scores_74246),
        'n_hspc': len(hspc_scores_74246),
        'n_misclassified': len(mis),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }
    
    with open('training_metrics.json', 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv('roc_points_training.csv', index=False)
    
    # DE analysis
    print("\n  Calculating differential expression...")
    de_results = pd.DataFrame(index=log_data.index)
    de_results['log2_fold_change'] = log_data[aml_samples].mean(axis=1) - log_data[hspc_samples].mean(axis=1)
    
    p_values = []
    for gene in log_data.index:
        _, p = ttest_ind(log_data.loc[gene, aml_samples], 
                        log_data.loc[gene, hspc_samples], equal_var=False)
        p_values.append(p)
    
    de_results['p_value'] = p_values
    de_results['p_adj'] = multipletests(de_results['p_value'], method='fdr_bh')[1]
    de_results['significant'] = (abs(de_results['log2_fold_change']) > 1) & (de_results['p_adj'] < 0.05)
    de_results.to_csv('de_results_aml_vs_hspc.csv')
    
    # GSEA rank file
    rnk = pd.DataFrame({
        'gene': de_results.index,
        'stat': de_results['log2_fold_change'] * -np.log10(de_results['p_value'].clip(lower=1e-300))
    }).sort_values('stat', ascending=False)
    rnk[['gene', 'stat']].to_csv('aml_vs_hspc.rnk', sep='\t', index=False, header=False)
    
    print(f"\n  DE Results:")
    print(f"    Significant: {de_results['significant'].sum()}")
    print(f"    Up in AML: {((de_results['log2_fold_change'] > 1) & de_results['significant']).sum()}")
    print(f"    Down in AML: {((de_results['log2_fold_change'] < -1) & de_results['significant']).sum()}")
    
    # Cross-dataset validation
    if gse125345_counts is not None:
        print("\n  Cross-dataset validation on GSE125345...")
        
        log_125345 = logcpm(gse125345_counts, dataset_name="GSE125345")
        
        common_genes = list(set(log_data.index) & set(log_125345.index))
        common_up = list(set(top_up) & set(common_genes))
        common_down = list(set(top_down) & set(common_genes))
        
        print(f"    Gene overlap: up={len(common_up)}/{len(top_up)}, down={len(common_down)}/{len(top_down)}")
        
        if len(common_up) < 10 or len(common_down) < 10:
            print(f"    ⚠ Low overlap - results may be unstable")
        
        if len(common_up) >= 10 and len(common_down) >= 10:
            # Calculate scores
            scores_125345 = pd.Series(index=log_125345.columns, dtype=float)
            for sample in log_125345.columns:
                up_score = log_125345[sample].loc[common_up].mean()
                down_score = log_125345[sample].loc[common_down].mean()
                scores_125345[sample] = up_score - down_score
            
            # Map cell types
            val_df = pd.DataFrame({
                'sample_id': scores_125345.index,
                'score': scores_125345.values,
                'cell_type': [gse125345_meta[gse125345_meta['sample_id']==s]['cell_type'].values[0] 
                             if s in gse125345_meta['sample_id'].values else 'unknown' 
                             for s in scores_125345.index]
            })
            val_df.to_csv('gse125345_scores_with_celltype.csv', index=False)
            
            # Per-cell-type summary
            ct_sum = val_df.groupby('cell_type')['score'].agg(
                ['count', 'mean', 'std', 'min', 'max']
            ).sort_values('mean')
            ct_sum.to_csv('gse125345_celltype_summary.csv')
            print("  ✓ Saved gse125345_celltype_summary.csv")
            
            # Validation specificity
            val_specificity = (scores_125345 < t_star).mean()
            
            # Effect sizes
            delta_trainH_vs_valH = cliffs_delta_exact(hspc_scores_74246.values, scores_125345.values)[0]
            delta_aml_vs_valH = cliffs_delta_exact(aml_scores_74246.values, scores_125345.values)[0]
            
            print(f"\n    Validation HSPC specificity at t*: {val_specificity:.3f}")
            print(f"    Cliff's delta (train HSPC vs val HSPC): {delta_trainH_vs_valH:.3f}")
            print(f"    Cliff's delta (train AML vs val HSPC): {delta_aml_vs_valH:.3f}")
            
            # Statistical tests
            _, p_hspc = mannwhitneyu(scores_125345, hspc_scores_74246, alternative='two-sided')
            _, p_aml = mannwhitneyu(scores_125345, aml_scores_74246, alternative='two-sided')
            
            print(f"\n    Mann-Whitney tests:")
            print(f"      GSE125345 vs GSE74246-HSPC: p = {p_hspc:.3f}")
            print(f"      GSE125345 vs GSE74246-AML: p = {p_aml:.2e}")
            
            # Cell type violin plot
            plt.figure(figsize=(7, 4))
            order = ct_sum.index.tolist()
            sns.violinplot(data=val_df, x='cell_type', y='score', order=order, inner='point')
            plt.axhline(y=t_star, linestyle='--', color='black', alpha=0.6, label=f't*={t_star:.2f}')
            plt.ylabel('AML score')
            plt.xlabel('Cell type')
            plt.legend()
            plt.tight_layout()
            plt.savefig('gse125345_celltype_violin.png', dpi=300)
            plt.close()
    
    
    


    # Save session info
    save_session_info()
    
    create_final_visualizations(
        log_data, scores_df, de_results, top_up, top_down,
        auc_train, cliff_delta, t_star, aml_scores_74246, hspc_scores_74246
    )
    return {
        'scores_df': scores_df,
        'de_results': de_results,
        'auc': auc_train,
        'threshold': t_star
    }

def create_final_visualizations(log_data, scores_df, de_results, top_up, top_down, 
                                auc, cliff_delta, threshold, aml_scores, hspc_scores):
    """Create all visualizations including calibration plot"""
    
    # Main figure
    fig = plt.figure(figsize=(20, 12))
    color_map = {'AML': '#d62728', 'HSPC': '#1f77b4', 'Healthy': '#2ca02c'}
    
    
    
    
        # 2x4 panel: PCA, score dists + threshold, volcano, ROC
    # 1) PCA
    ax = plt.subplot(2, 4, 1)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(log_data.T.values)
    sample_ids = list(log_data.columns)
    cond_by_sample = scores_df.set_index('sample_id')['condition'].to_dict()
    colors = [ {'AML': '#d62728', 'HSPC': '#1f77b4', 'Healthy': '#2ca02c'}.get(
               cond_by_sample.get(s, ''), '#7f7f7f') for s in sample_ids ]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=30, alpha=0.9)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA - AML vs HSPC')
    ax.grid(alpha=0.3)

    # 2) Score distributions with threshold
    ax = plt.subplot(2, 4, 2)
    aml_arr = np.asarray(aml_scores)
    hspc_arr = np.asarray(hspc_scores)
    ax.hist(aml_arr, bins=20, alpha=0.6, label='AML', color='#d62728', density=True, edgecolor='black')
    ax.hist(hspc_arr, bins=20, alpha=0.6, label='HSPC', color='#1f77b4', density=True, edgecolor='black')
    ax.axvline(threshold, color='black', linestyle='--', label=f't*={threshold:.2f}')
    ax.set_xlabel('AML score'); ax.set_ylabel('Density')
    ax.set_title(f'Scores (AUC={auc:.3f}, δ={cliff_delta:.3f})')
    ax.legend()

    # 3) Volcano
    ax = plt.subplot(2, 4, 3)
    padj = de_results['p_adj'].clip(lower=1e-300)
    ax.scatter(de_results['log2_fold_change'], -np.log10(padj), s=10, alpha=0.25, color='gray')
    sig = de_results['significant']
    up = (de_results['log2_fold_change'] > 1) & sig
    dn = (de_results['log2_fold_change'] < -1) & sig
    ax.scatter(de_results.loc[up, 'log2_fold_change'], -np.log10(padj.loc[up]),
               s=20, alpha=0.7, color='#d62728', label=f'Up in AML (n={up.sum()})')
    ax.scatter(de_results.loc[dn, 'log2_fold_change'], -np.log10(padj.loc[dn]),
               s=20, alpha=0.7, color='#1f77b4', label=f'Up in HSPC (n={dn.sum()})')
    ax.axhline(-np.log10(0.05), color='black', ls='--', alpha=0.3)
    ax.axvline(1, color='black', ls='--', alpha=0.3); ax.axvline(-1, color='black', ls='--', alpha=0.3)
    ax.set_xlabel('Log2 Fold Change'); ax.set_ylabel('-Log10(FDR)')
    ax.set_title('Volcano'); ax.legend(fontsize=8)

    # 4) ROC
    ax = plt.subplot(2, 4, 4)
    y = (scores_df['condition'] == 'AML').astype(int).values
    fpr, tpr, _ = roc_curve(y, scores_df['aml_score'].values)
    ax.plot(fpr, tpr, linewidth=2, color='#d62728', label=f'AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC'); ax.legend(); ax.grid(alpha=0.3)

    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calibration plot
    plt.figure(figsize=(5, 4))
    for lab, arr in [('AML', aml_scores), ('HSPC', hspc_scores)]:
        sns.kdeplot(arr, label=lab, fill=True, alpha=0.3)
    plt.axvline(threshold, ls='--', color='k', label=f't*={threshold:.2f}')
    plt.xlabel('AML score')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Score Calibration')
    plt.tight_layout()
    plt.savefig('score_calibration_kde.png', dpi=300)
    plt.close()
    
    # Clustermap
    subset = log_data.loc[list(top_up[:15]) + list(top_down[:15])]
    sample_cond = scores_df.set_index('sample_id')['condition']
    col_colors = sample_cond.loc[subset.columns].map(color_map).to_frame('Condition')
    
    g = sns.clustermap(subset, z_score=0, cmap='RdBu_r', 
                       col_colors=col_colors, vmin=-2, vmax=2, figsize=(12, 8))
    g.fig.suptitle(f'Top Genes (AUC={auc:.3f}, t*={threshold:.3f})', y=1.02)
    g.fig.tight_layout()
    g.fig.savefig('top_gene_clustermap.png', dpi=300, bbox_inches='tight')
    plt.close(g.fig)

# Run analysis
if __name__ == "__main__":
    try:
        results = final_complete_analysis()
        if results:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"✓ All results saved to: {OUTPUT_DIR.absolute()}")
            print("\n📊 Key output files:")
            print("  • aml_scores_with_labels.csv (with predictions)")
            print("  • misclassified_samples.csv")
            print("  • gse125345_celltype_summary.csv")
            print("  • gse125345_celltype_violin.png")
            print("  • score_calibration_kde.png")
            print("  • training_metrics.json")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
