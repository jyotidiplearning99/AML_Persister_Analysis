#!/usr/bin/env python3
"""
Enhanced Persister Analysis - FINAL VERSION
Analyzes multi-modal distributions and identifies bin-specific signatures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from pathlib import Path
import os
import gzip
import gc

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis")
RESULTS_DIR = BASE_DIR / "results"
BEATAML_DIR = RESULTS_DIR / "bulk_BeatAML"
TCGA_DIR = RESULTS_DIR / "bulk_TCGA"

os.makedirs(RESULTS_DIR / "comprehensive_analysis", exist_ok=True)

# ============================================================================
# PART 1: LOAD DATA WITH FIXED BIN ANALYSIS
# ============================================================================

def load_all_data():
    """Load all datasets with predictions"""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    datasets = {}
    
    # Load saved files with bins
    for name in ['beataml', 'tcga', 'gse74246']:
        file_path = RESULTS_DIR / f"{name}_with_bins.csv"
        if file_path.exists():
            data = pd.read_csv(file_path)
            datasets[name] = data
            print(f"{name.upper()}: {len(data)} samples loaded")
    
    return datasets

# ============================================================================
# PART 2: ANALYZE BIN-SPECIFIC GENE SIGNATURES
# ============================================================================

def analyze_bin_signatures(datasets, expression_file=None):
    """
    Identify genes that differentiate between persister bins
    This addresses what makes each bin unique
    """
    print("\n" + "="*60)
    print("BIN-SPECIFIC SIGNATURE ANALYSIS")
    print("="*60)
    
    results = {}
    
    # If we have expression data, do differential analysis
    if expression_file and expression_file.exists():
        print(f"Loading expression data from {expression_file}")
        
        # Load expression
        if expression_file.suffix == '.gz':
            with gzip.open(expression_file, 'rt') as f:
                expr = pd.read_csv(f, sep='\t', index_col=0)
        else:
            expr = pd.read_csv(expression_file, index_col=0)
        
        expr.index = [str(g).upper() for g in expr.index]
        
        # For each dataset with bins
        for dataset_name, data in datasets.items():
            if 'persister_category' not in data.columns:
                continue
                
            print(f"\nAnalyzing {dataset_name.upper()}...")
            
            # Match samples between expression and predictions
            common_samples = list(set(data['sample_id']) & set(expr.columns))
            
            if len(common_samples) < 10:
                print(f"  Insufficient samples for analysis ({len(common_samples)})")
                continue
            
            # Get expression for common samples
            expr_subset = expr[common_samples]
            data_subset = data[data['sample_id'].isin(common_samples)]
            
            # Create sample-to-bin mapping
            sample_to_bin = dict(zip(data_subset['sample_id'], data_subset['persister_category']))
            
            # Differential expression between bins
            bins = sorted(data_subset['persister_category'].unique())
            
            if len(bins) >= 2:
                # Compare highest vs lowest bin
                high_bin = bins[-1]  # Highest probability bin
                low_bin = bins[0]    # Lowest probability bin
                
                high_samples = [s for s, b in sample_to_bin.items() if b == high_bin]
                low_samples = [s for s, b in sample_to_bin.items() if b == low_bin]
                
                if len(high_samples) >= 3 and len(low_samples) >= 3:
                    print(f"  Comparing {high_bin} (n={len(high_samples)}) vs {low_bin} (n={len(low_samples)})")
                    
                    # Calculate fold changes
                    high_expr = expr_subset[high_samples].mean(axis=1)
                    low_expr = expr_subset[low_samples].mean(axis=1)
                    
                    # Avoid division by zero
                    fold_change = (high_expr + 1) / (low_expr + 1)
                    
                    # Perform t-tests
                    p_values = []
                    for gene in expr_subset.index:
                        high_vals = expr_subset.loc[gene, high_samples]
                        low_vals = expr_subset.loc[gene, low_samples]
                        _, p = stats.ttest_ind(high_vals, low_vals)
                        p_values.append(p)
                    
                    # Create results dataframe
                    diff_expr = pd.DataFrame({
                        'gene': expr_subset.index,
                        'mean_high': high_expr.values,
                        'mean_low': low_expr.values,
                        'fold_change': fold_change.values,
                        'log2_fc': np.log2(fold_change.values),
                        'p_value': p_values
                    })
                    
                    # Sort by fold change
                    diff_expr = diff_expr.sort_values('log2_fc', ascending=False)
                    
                    # Get top differentially expressed genes
                    top_up = diff_expr.head(20)
                    top_down = diff_expr.tail(20)
                    
                    results[dataset_name] = {
                        'diff_expr': diff_expr,
                        'top_up': top_up,
                        'top_down': top_down
                    }
                    
                    print(f"\n  Top upregulated in {high_bin}:")
                    for _, row in top_up.head(10).iterrows():
                        print(f"    {row['gene']}: FC={row['fold_change']:.2f}, p={row['p_value']:.3e}")
                    
                    print(f"\n  Top downregulated in {high_bin}:")
                    for _, row in top_down.head(10).iterrows():
                        print(f"    {row['gene']}: FC={row['fold_change']:.2f}, p={row['p_value']:.3e}")
    
    return results

# ============================================================================
# PART 3: FIXED RISK STRATIFICATION
# ============================================================================

def analyze_risk_by_bins(data, dataset_name='Dataset'):
    """
    Properly analyze risk stratification across bins
    FIXED: Correct identification of high/low persister bins
    """
    print(f"\n{'='*60}")
    print(f"RISK STRATIFICATION - {dataset_name}")
    print(f"{'='*60}")
    
    if 'persister_category' not in data.columns:
        print("No binning data available")
        return
    
    # Get bins sorted by mean persister probability
    bin_stats = data.groupby('persister_category')['persister_probability'].agg(['mean', 'count'])
    bin_stats = bin_stats.sort_values('mean')
    
    print("\nBin hierarchy (by mean persister probability):")
    for bin_name, row in bin_stats.iterrows():
        print(f"  {bin_name}: mean={row['mean']:.3f}, n={row['count']:.0f}")
    
    # Identify true high/low bins based on probability
    bins_ordered = list(bin_stats.index)
    n_bins = len(bins_ordered)
    
    if n_bins >= 3:
        low_persister_bins = bins_ordered[:2]   # Bottom 2 bins (lowest probabilities)
        high_persister_bins = bins_ordered[-2:]  # Top 2 bins (highest probabilities)
    else:
        low_persister_bins = [bins_ordered[0]]  # Lowest bin
        high_persister_bins = [bins_ordered[-1]] # Highest bin
    
    print(f"\nTrue Low Persister bins (low probability): {low_persister_bins}")
    print(f"True High Persister bins (high probability): {high_persister_bins}")
    
    # Analyze cell types if available
    if 'cell_type' in data.columns:
        print("\nCell Type Distribution:")
        
        cell_type_analysis = []
        for cell_type in data['cell_type'].unique():
            ct_data = data[data['cell_type'] == cell_type]
            if len(ct_data) >= 2:
                in_high = ct_data['persister_category'].isin(high_persister_bins).mean() * 100
                in_low = ct_data['persister_category'].isin(low_persister_bins).mean() * 100
                
                cell_type_analysis.append({
                    'cell_type': cell_type,
                    'n': len(ct_data),
                    'pct_in_high_persister': in_high,
                    'pct_in_low_persister': in_low,
                    'enrichment': in_high / (in_low + 0.1)  # Avoid division by zero
                })
        
        ct_df = pd.DataFrame(cell_type_analysis).sort_values('pct_in_high_persister', ascending=False)
        
        print("\nTop cell types enriched in HIGH persister bins:")
        for _, row in ct_df.head(5).iterrows():
            print(f"  {row['cell_type']}: {row['pct_in_high_persister']:.1f}% (n={row['n']})")
        
        print("\nTop cell types enriched in LOW persister bins:")
        for _, row in ct_df.sort_values('pct_in_low_persister', ascending=False).head(5).iterrows():
            print(f"  {row['cell_type']}: {row['pct_in_low_persister']:.1f}% (n={row['n']})")
    
    # AML vs Normal analysis
    if 'is_aml' in data.columns:
        aml_data = data[data['is_aml'] == True]
        normal_data = data[data['is_aml'] == False]
        
        if len(aml_data) > 0 and len(normal_data) > 0:
            aml_high = aml_data['persister_category'].isin(high_persister_bins).mean() * 100
            normal_high = normal_data['persister_category'].isin(high_persister_bins).mean() * 100
            
            print(f"\nAML vs Normal in HIGH persister bins:")
            print(f"  AML samples: {aml_high:.1f}%")
            print(f"  Normal samples: {normal_high:.1f}%")
            
            if normal_high > 0:
                print(f"  AML enrichment: {aml_high/normal_high:.2f}x")

# ============================================================================
# PART 4: COMPREHENSIVE VISUALIZATION
# ============================================================================

def create_final_figures(datasets):
    """Create final comprehensive figures with all insights"""
    
    print("\n" + "="*60)
    print("CREATING FINAL VISUALIZATIONS")
    print("="*60)
    
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    # Row 1: Distribution and GMM fits for each dataset
    for i, (name, data) in enumerate(datasets.items()):
        if i < 3:
            ax = fig.add_subplot(gs[0, i])
            
            # Plot distribution
            ax.hist(data['persister_probability'], bins=30, alpha=0.6, 
                   density=True, edgecolor='black', color='skyblue')
            
            # Add vertical lines for bin boundaries if available
            if 'persister_category' in data.columns:
                bin_stats = data.groupby('persister_category')['persister_probability'].agg(['min', 'max', 'mean'])
                for _, row in bin_stats.iterrows():
                    ax.axvline(x=row['mean'], color='red', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Persister Probability')
            ax.set_ylabel('Density')
            ax.set_title(f'{name.upper()}\n({len(data)} samples)')
    
    # Row 2: Bin composition pie charts
    for i, (name, data) in enumerate(datasets.items()):
        if i < 3 and 'persister_category' in data.columns:
            ax = fig.add_subplot(gs[1, i])
            
            # Sort bins by mean probability for consistent coloring
            bin_order = data.groupby('persister_category')['persister_probability'].mean().sort_values().index
            bin_counts = data['persister_category'].value_counts()[bin_order]
            
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(bin_counts)))
            
            wedges, texts, autotexts = ax.pie(bin_counts.values, 
                                              labels=bin_counts.index,
                                              colors=colors,
                                              autopct='%1.1f%%',
                                              startangle=90)
            
            ax.set_title(f'{name.upper()}\n{len(bin_counts)} bins detected')
    
    # Row 3: Heatmap of bin distribution across datasets
    ax_heat = fig.add_subplot(gs[2, :2])
    
    # Create matrix of bin proportions
    bin_matrix = []
    dataset_names = []
    all_bins = set()
    
    for name, data in datasets.items():
        if 'persister_category' in data.columns:
            dataset_names.append(name.upper())
            bin_props = data['persister_category'].value_counts(normalize=True) * 100
            bin_matrix.append(bin_props)
            all_bins.update(bin_props.index)
    
    if bin_matrix:
        # Create aligned dataframe
        matrix_df = pd.DataFrame(bin_matrix, index=dataset_names)
        matrix_df = matrix_df.fillna(0)
        
        # Sort columns by typical probability range
        col_order = sorted(matrix_df.columns, 
                          key=lambda x: datasets[dataset_names[0].lower()][
                              datasets[dataset_names[0].lower()]['persister_category'] == x
                          ]['persister_probability'].mean() if x in datasets[dataset_names[0].lower()]['persister_category'].values else 0)
        
        matrix_df = matrix_df[col_order]
        
        sns.heatmap(matrix_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=ax_heat, cbar_kws={'label': '% of samples'})
        ax_heat.set_title('Bin Distribution Across Datasets')
        ax_heat.set_xlabel('Persister Bin')
        ax_heat.set_ylabel('Dataset')
    
    # Row 3-4: Statistical summary
    ax_stats = fig.add_subplot(gs[2:, 2:])
    ax_stats.axis('off')
    
    stats_text = "MULTI-MODAL PERSISTER ANALYSIS SUMMARY\n" + "="*60 + "\n\n"
    
    for name, data in datasets.items():
        if 'persister_category' in data.columns:
            n_bins = len(data['persister_category'].unique())
            stats_text += f"{name.upper()}:\n"
            stats_text += f"  • {n_bins} distinct persister states detected\n"
            
            # Get bin stats
            bin_stats = data.groupby('persister_category')['persister_probability'].agg(['mean', 'std', 'count'])
            bin_stats = bin_stats.sort_values('mean')
            
            for bin_name, row in bin_stats.iterrows():
                stats_text += f"    - {bin_name}: μ={row['mean']:.3f}, σ={row['std']:.3f}, n={row['count']:.0f}\n"
            
            stats_text += "\n"
    
    stats_text += "\nKEY INSIGHTS:\n"
    stats_text += "• BeatAML shows 4 distinct persister states\n"
    stats_text += "• TCGA shows 5 states (most complex distribution)\n"
    stats_text += "• GSE74246 shows binary classification (high model confidence)\n"
    stats_text += "• Multi-modal distributions suggest discrete persister phenotypes\n"
    stats_text += "• These may represent different stages in stemness hierarchy\n"
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Multi-Modal Persister State Discovery', fontsize=16, y=0.98)
    plt.savefig(RESULTS_DIR / "comprehensive_analysis" / "final_multimodal_analysis.png", 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    print("  Saved: comprehensive_analysis/final_multimodal_analysis.png")

# ============================================================================
# PART 5: PATHWAY ENRICHMENT FOR BINS
# ============================================================================

def analyze_bin_pathways(diff_results):
    """
    Analyze pathway enrichment for genes differentiating bins
    """
    print("\n" + "="*60)
    print("PATHWAY ANALYSIS FOR BIN-SPECIFIC GENES")
    print("="*60)
    
    # Define relevant pathways
    pathways = {
        "STEMNESS": ["CD34", "KIT", "FLT3", "DNMT3A", "TET2", "HOXA9", "MEIS1", "MYC"],
        "QUIESCENCE": ["CDKN1A", "CDKN1B", "CDKN1C", "TP53", "RB1", "FOXO1", "FOXO3"],
        "DRUG_RESISTANCE": ["ABCB1", "ABCC1", "ABCG2", "BCL2", "MCL1", "XIAP"],
        "METABOLISM": ["HK2", "PKM", "LDHA", "G6PD", "IDH1", "IDH2", "ENO1"],
        "OXIDATIVE_STRESS": ["SOD1", "SOD2", "CAT", "GPX1", "GPX4", "NQO1", "HMOX1"]
    }
    
    for dataset_name, results in diff_results.items():
        if 'top_up' not in results:
            continue
            
        print(f"\n{dataset_name.upper()} - Enriched in high persister bins:")
        
        top_genes = list(results['top_up']['gene'].values[:50])  # Top 50 upregulated
        
        for pathway_name, pathway_genes in pathways.items():
            overlap = [g for g in pathway_genes if g in top_genes]
            if overlap:
                print(f"  {pathway_name}: {', '.join(overlap)}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete multi-modal analysis"""
    
    print("\n" + "="*80)
    print("FINAL MULTI-MODAL PERSISTER ANALYSIS")
    print("="*80)
    
    # Load all datasets with bins
    datasets = load_all_data()
    
    if not datasets:
        print("No datasets with binning found!")
        return
    
    # Analyze each dataset
    for name, data in datasets.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {name.upper()}")
        print(f"{'='*60}")
        
        # Fixed risk stratification
        analyze_risk_by_bins(data, name)
    
    # Analyze bin-specific gene signatures if expression available
    expr_file = BEATAML_DIR / "processed/expression.csv"
    if expr_file.exists():
        diff_results = analyze_bin_signatures(datasets, expr_file)
        
        # Analyze pathways
        if diff_results:
            analyze_bin_pathways(diff_results)
            
            # Save differential expression results
            for dataset_name, results in diff_results.items():
                output_file = RESULTS_DIR / f"{dataset_name}_bin_signatures.csv"
                results['diff_expr'].to_csv(output_file, index=False)
                print(f"\nSaved bin signatures to: {output_file}")
    
    # Create final visualizations
    create_final_figures(datasets)
    
    # Generate final report
    generate_final_report(datasets)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

def generate_final_report(datasets):
    """Generate comprehensive final report"""
    
    report = []
    report.append("\n" + "="*80)
    report.append("FINAL MULTI-MODAL PERSISTER ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"Generated: {pd.Timestamp.now()}")
    
    report.append("\n1. DISCOVERY OF DISCRETE PERSISTER STATES")
    report.append("-"*40)
    report.append("KEY FINDING: Multiple discrete persister states identified across cohorts")
    
    for name, data in datasets.items():
        if 'persister_category' in data.columns:
            n_bins = len(data['persister_category'].unique())
            report.append(f"\n{name.upper()}: {n_bins} states")
            
            # Get bin stats sorted by probability
            bin_stats = data.groupby('persister_category')['persister_probability'].agg(['mean', 'count'])
            bin_stats = bin_stats.sort_values('mean')
            
            for i, (bin_name, row) in enumerate(bin_stats.iterrows()):
                report.append(f"  State {i+1} ({bin_name}): mean={row['mean']:.3f}, n={row['count']:.0f}")
    
    report.append("\n2. BIOLOGICAL INTERPRETATION")
    report.append("-"*40)
    report.append("The multi-modal distributions suggest:")
    report.append("  • Discrete persister phenotypes rather than continuous spectrum")
    report.append("  • Potential hierarchical organization (4-5 distinct states)")
    report.append("  • Different therapeutic vulnerabilities for each state")
    report.append("  • GSE74246 shows high confidence binary classification")
    
    report.append("\n3. CLINICAL IMPLICATIONS")
    report.append("-"*40)
    report.append("  • Patient stratification: Can classify patients into 4-5 risk groups")
    report.append("  • Targeted therapy: Each state may require different treatment")
    report.append("  • Monitoring: Track transitions between states during treatment")
    
    # Save report
    report_text = "\n".join(report)
    report_path = RESULTS_DIR / "comprehensive_analysis" / "final_multimodal_report.txt"
    
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n  Saved final report to: {report_path}")

if __name__ == "__main__":
    main()
