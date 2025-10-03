#!/usr/bin/env python3
"""
Comprehensive Bulk RNA-seq Persister Analysis Pipeline
Includes all clinical associations, module scoring, and sensitivity analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
import json
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base paths
BASE_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis")
RESULTS_DIR = BASE_DIR / "results"
BEATAML_DIR = RESULTS_DIR / "bulk_BeatAML"
TCGA_DIR = RESULTS_DIR / "bulk_TCGA"
MODULES_DIR = BASE_DIR / "custom_modules"

# Create output directories
os.makedirs(BEATAML_DIR / "analysis", exist_ok=True)
os.makedirs(TCGA_DIR / "analysis", exist_ok=True)
os.makedirs(RESULTS_DIR / "comprehensive_analysis", exist_ok=True)

# ============================================================================
# PART 1: LOAD ALL DATA
# ============================================================================

def load_all_data():
    """Load all prediction and metadata files"""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # BeatAML
    beataml_pred = pd.read_csv(BEATAML_DIR / "predictions_fixed.csv")
    beataml_meta = pd.read_csv(BEATAML_DIR / "processed/metadata.csv")
    beataml = beataml_pred.merge(beataml_meta, on="sample_id", how="left")
    print(f"BeatAML: {len(beataml)} samples loaded")
    
    # TCGA
    tcga_pred = pd.read_csv(TCGA_DIR / "predictions_fixed.csv")
    tcga_meta = pd.read_csv(TCGA_DIR / "processed/metadata.csv")
    tcga = tcga_pred.merge(tcga_meta, on="sample_id", how="left")
    print(f"TCGA: {len(tcga)} samples loaded")
    
    return beataml, tcga

# ============================================================================
# PART 2: COMPREHENSIVE MODULE DEFINITIONS
# ============================================================================

def create_comprehensive_modules():
    """Create comprehensive gene modules for scoring"""
    
    modules = {
        "RTK_SIGNALING_FULL": [
            # Core RTK receptors
            "FLT3", "KIT", "PDGFRA", "PDGFRB", "CSF1R", "FMS",
            "EGFR", "ERBB2", "ERBB3", "ERBB4",
            "FGFR1", "FGFR2", "FGFR3", "FGFR4",
            "VEGFR1", "VEGFR2", "VEGFR3", "FLT1", "KDR", "FLT4",
            "MET", "RON", "MST1R", "AXL", "TYRO3", "MERTK",
            "NTRK1", "NTRK2", "NTRK3", "RET", "ALK", "ROS1",
            # Downstream signaling
            "GRB2", "SOS1", "SOS2", "SHC1", "SHC2", "GAB1", "GAB2",
            "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "PIK3R1", "PIK3R2",
            "AKT1", "AKT2", "AKT3", "MTOR", "RICTOR", "RAPTOR",
            "HRAS", "KRAS", "NRAS", "RAF1", "BRAF", "ARAF",
            "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "MAPK14",
            "STAT3", "STAT5A", "STAT5B", "JAK1", "JAK2", "JAK3"
        ],
        
        "MORPHOGENESIS_EMT": [
            # EMT transcription factors
            "SNAI1", "SNAI2", "SNAI3", "ZEB1", "ZEB2", 
            "TWIST1", "TWIST2", "TCF3", "TCF4", "FOXC2",
            # Adhesion and mesenchymal markers
            "CDH1", "CDH2", "CDH11", "EPCAM", "CLDN1", "CLDN3", "CLDN7",
            "VIM", "FN1", "COL1A1", "COL1A2", "COL3A1", "COL5A1", 
            "SPARC", "ACTA2", "MMP2", "MMP9", "MMP14",
            # ECM remodeling
            "ITGA5", "ITGAV", "ITGA6", "ITGB1", "ITGB3", "ITGB5",
            "LAMA1", "LAMA2", "LAMA3", "LAMA4", "LAMA5",
            "LAMB1", "LAMB2", "LAMB3", "LAMC1", "LAMC2",
            # Signaling
            "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2",
            "SMAD2", "SMAD3", "SMAD4", "BMP1", "BMP2", "BMP4"
        ],
        
        "BCL2_FAMILY": [
            "BCL2", "BCL2L1", "MCL1", "BCL2L2", "BCL2A1", "BCL2L10",
            "BAX", "BAK1", "BOK", "BID", "BIM", "BAD", "PUMA", "NOXA"
        ],
        
        "CELL_CYCLE_ARREST": [
            "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B", "RB1", "TP53",
            "GADD45A", "GADD45B", "GADD45G"
        ],
        
        "AML_MARKERS": [
            "FLT3", "NPM1", "DNMT3A", "IDH1", "IDH2", "TET2", "RUNX1",
            "CEBPA", "KIT", "WT1", "ASXL1", "BCOR", "EZH2"
        ]
    }
    
    # Save modules
    os.makedirs(MODULES_DIR, exist_ok=True)
    with open(MODULES_DIR / "comprehensive_modules.json", "w") as f:
        json.dump(modules, f, indent=2)
    
    print(f"\nCreated {len(modules)} comprehensive modules")
    return modules

# ============================================================================
# PART 3: CLINICAL ASSOCIATION ANALYSIS
# ============================================================================

def analyze_clinical_associations(beataml, tcga):
    """Analyze clinical associations with persister probability"""
    
    print("\n" + "="*60)
    print("CLINICAL ASSOCIATION ANALYSIS")
    print("="*60)
    
    results = {}
    
    # BeatAML: Check for blast percentage correlation
    if "blast_percent" in beataml.columns:
        valid = beataml[["persister_probability", "blast_percent"]].dropna()
        if len(valid) > 10:
            rho, p = spearmanr(valid["persister_probability"], valid["blast_percent"])
            results['beataml_blast_correlation'] = {
                'n': len(valid),
                'rho': rho,
                'p_value': p,
                'significant': p < 0.05
            }
            print(f"\nBeatAML Blast % Correlation:")
            print(f"  n = {len(valid)}")
            print(f"  Spearman ρ = {rho:.3f}")
            print(f"  p-value = {p:.3e}")
            print(f"  {'SIGNIFICANT' if p < 0.05 else 'Not significant'}")
    
    # Check for response associations if available
    resp_cols = ['response', 'best_response', 'clinical_response']
    for col in resp_cols:
        if col in beataml.columns:
            resp_data = beataml[["persister_probability", col]].dropna()
            if len(resp_data) > 0:
                # Try to identify CR vs NR
                cr_mask = resp_data[col].astype(str).str.contains("CR|Complete", case=False, na=False)
                nr_mask = resp_data[col].astype(str).str.contains("NR|No|Refract|Resist", case=False, na=False)
                
                if cr_mask.sum() >= 5 and nr_mask.sum() >= 5:
                    cr_vals = resp_data.loc[cr_mask, "persister_probability"]
                    nr_vals = resp_data.loc[nr_mask, "persister_probability"]
                    stat, p = mannwhitneyu(cr_vals, nr_vals)
                    
                    results['beataml_response'] = {
                        'n_cr': len(cr_vals),
                        'n_nr': len(nr_vals),
                        'mean_cr': cr_vals.mean(),
                        'mean_nr': nr_vals.mean(),
                        'p_value': p
                    }
                    print(f"\nBeatAML Response Analysis:")
                    print(f"  Complete Response (n={len(cr_vals)}): mean={cr_vals.mean():.3f}")
                    print(f"  No Response (n={len(nr_vals)}): mean={nr_vals.mean():.3f}")
                    print(f"  Mann-Whitney U p={p:.3e}")
                    break
    
    # TCGA: Survival analysis placeholder
    if "os_time" in tcga.columns and "os_event" in tcga.columns:
        surv_data = tcga[["os_time", "os_event", "persister_probability"]].dropna()
        if len(surv_data) > 20:
            # Dichotomize by median
            tcga['high_risk'] = tcga['persister_probability'] >= tcga['persister_probability'].median()
            results['tcga_survival_available'] = True
            print(f"\nTCGA Survival Data:")
            print(f"  n = {len(surv_data)} with survival data")
            print(f"  Median follow-up: {surv_data['os_time'].median():.1f} months")
    
    return results

# ============================================================================
# PART 4: MODULE SCORING
# ============================================================================

def calculate_module_scores(expression_file, modules):
    """Calculate module scores using PCA method"""
    
    print("\n" + "="*60)
    print("MODULE SCORING")
    print("="*60)
    
    # Load expression data
    expr = pd.read_csv(expression_file, index_col=0)
    expr.index = [str(g).upper() for g in expr.index]
    
    module_scores = pd.DataFrame(index=expr.columns)
    
    for module_name, gene_list in modules.items():
        # Find overlapping genes
        genes_upper = [g.upper() for g in gene_list]
        overlap = [g for g in genes_upper if g in expr.index]
        
        if len(overlap) >= 3:  # Need at least 3 genes for PCA
            # Extract module genes
            module_expr = expr.loc[overlap]
            
            # PCA scoring
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize
            scaler = StandardScaler()
            module_expr_std = scaler.fit_transform(module_expr.T)
            
            # PCA
            pca = PCA(n_components=1)
            scores = pca.fit_transform(module_expr_std).flatten()
            
            module_scores[module_name] = scores
            print(f"  {module_name}: {len(overlap)}/{len(gene_list)} genes, "
                  f"PC1 var={pca.explained_variance_ratio_[0]:.1%}")
    
    return module_scores

# ============================================================================
# PART 5: SENSITIVITY ANALYSES
# ============================================================================

def threshold_sensitivity_analysis(beataml_file, tcga_file):
    """Analyze sensitivity to threshold selection"""
    
    print("\n" + "="*60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Load raw predictions
    beataml = pd.read_csv(beataml_file)
    tcga = pd.read_csv(tcga_file)
    
    thresholds = np.arange(0.25, 0.41, 0.02)
    results = []
    
    for thr in thresholds:
        beat_pers = (beataml['persister_probability'] >= thr).mean()
        tcga_pers = (tcga['persister_probability'] >= thr).mean()
        
        results.append({
            'threshold': thr,
            'beataml_pct': beat_pers * 100,
            'tcga_pct': tcga_pers * 100,
            'difference': abs(beat_pers - tcga_pers) * 100,
            'mean_pct': (beat_pers + tcga_pers) * 50
        })
    
    sensitivity_df = pd.DataFrame(results)
    
    # Find optimal threshold
    optimal_idx = sensitivity_df['difference'].idxmin()
    optimal_thr = sensitivity_df.loc[optimal_idx, 'threshold']
    
    print(f"\nThreshold Impact on Persister Classification:")
    print(sensitivity_df.to_string())
    print(f"\nOptimal threshold for consistency: {optimal_thr:.2f}")
    print(f"  BeatAML: {sensitivity_df.loc[optimal_idx, 'beataml_pct']:.1f}%")
    print(f"  TCGA: {sensitivity_df.loc[optimal_idx, 'tcga_pct']:.1f}%")
    print(f"  Difference: {sensitivity_df.loc[optimal_idx, 'difference']:.1f}%")
    
    return sensitivity_df

# ============================================================================
# PART 6: COMPREHENSIVE VISUALIZATIONS
# ============================================================================

def create_comprehensive_figures(beataml, tcga, sensitivity_df, module_scores_beat=None):
    """Create all visualization figures"""
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. BeatAML distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(beataml['persister_probability'], bins=25, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=0.31, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_xlabel('Persister Probability')
    ax1.set_ylabel('Count')
    ax1.set_title(f'BeatAML (n={len(beataml)})')
    ax1.legend()
    
    # 2. TCGA distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(tcga['persister_probability'], bins=25, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(x=0.31, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Persister Probability')
    ax2.set_title(f'TCGA (n={len(tcga)})')
    
    # 3. Combined violin plot
    ax3 = fig.add_subplot(gs[0, 2])
    combined_data = pd.DataFrame({
        'Dataset': ['BeatAML']*len(beataml) + ['TCGA']*len(tcga),
        'Probability': list(beataml['persister_probability']) + list(tcga['persister_probability'])
    })
    sns.violinplot(data=combined_data, x='Dataset', y='Probability', ax=ax3)
    ax3.axhline(y=0.31, color='red', linestyle='--', alpha=0.5)
    ax3.set_title('Distribution Comparison')
    
    # 4. Q-Q plots
    ax4 = fig.add_subplot(gs[0, 3])
    stats.probplot(beataml['persister_probability'], dist="norm", plot=ax4)
    ax4.set_title('BeatAML Q-Q Plot')
    
    # 5. ECDF comparison
    ax5 = fig.add_subplot(gs[1, 0:2])
    for dataset, label, color in [(beataml, 'BeatAML', 'blue'), (tcga, 'TCGA', 'red')]:
        sorted_data = np.sort(dataset['persister_probability'])
        ecdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        ax5.plot(sorted_data, ecdf, label=label, linewidth=2, color=color)
    ax5.axvline(x=0.31, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Persister Probability')
    ax5.set_ylabel('Empirical CDF')
    ax5.set_title('Cumulative Distribution Functions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Threshold sensitivity
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.plot(sensitivity_df['threshold'], sensitivity_df['beataml_pct'], 'o-', label='BeatAML', markersize=8)
    ax6.plot(sensitivity_df['threshold'], sensitivity_df['tcga_pct'], 's-', label='TCGA', markersize=8)
    ax6.axvline(x=0.31, color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Threshold')
    ax6.set_ylabel('% Classified as Persisters')
    ax6.set_title('Threshold Sensitivity Analysis')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Scatter plot comparison
    ax7 = fig.add_subplot(gs[2, 0])
    # Sample 100 points from each for clarity
    n_sample = min(100, len(beataml), len(tcga))
    beat_sample = beataml['persister_probability'].sample(n_sample, random_state=42).sort_values()
    tcga_sample = tcga['persister_probability'].sample(n_sample, random_state=42).sort_values()
    ax7.scatter(range(n_sample), beat_sample, alpha=0.5, label='BeatAML')
    ax7.scatter(range(n_sample), tcga_sample, alpha=0.5, label='TCGA')
    ax7.axhline(y=0.31, color='red', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Sample Rank')
    ax7.set_ylabel('Persister Probability')
    ax7.set_title('Sample-wise Comparison')
    ax7.legend()
    
    # 8. Statistical test results
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')
    
    # Perform statistical tests
    ks_stat, ks_p = stats.ks_2samp(beataml['persister_probability'], tcga['persister_probability'])
    mw_stat, mw_p = mannwhitneyu(beataml['persister_probability'], tcga['persister_probability'])
    
    stats_text = f"""Statistical Tests
{'='*30}

Kolmogorov-Smirnov Test:
  Statistic: {ks_stat:.3f}
  p-value: {ks_p:.4f}
  
Mann-Whitney U Test:
  Statistic: {mw_stat:.0f}
  p-value: {mw_p:.4f}
  
Interpretation:
  {'No significant difference' if mw_p > 0.05 else 'Significant difference'}
  between datasets (α=0.05)
"""
    ax8.text(0.1, 0.5, stats_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace')
    
    # 9. Blast correlation if available
    ax9 = fig.add_subplot(gs[2, 2:4])
    if "blast_percent" in beataml.columns:
        valid = beataml[['persister_probability', 'blast_percent']].dropna()
        if len(valid) > 0:
            ax9.scatter(valid['blast_percent'], valid['persister_probability'], alpha=0.6)
            # Add trend line
            z = np.polyfit(valid['blast_percent'], valid['persister_probability'], 1)
            p = np.poly1d(z)
            ax9.plot(valid['blast_percent'], p(valid['blast_percent']), "r--", alpha=0.8)
            ax9.set_xlabel('Blast %')
            ax9.set_ylabel('Persister Probability')
            ax9.set_title('BeatAML: Blast % Correlation')
            ax9.grid(True, alpha=0.3)
    else:
        ax9.axis('off')
        ax9.text(0.5, 0.5, 'Blast % data not available', 
                ha='center', va='center', transform=ax9.transAxes)
    
    # 10. Summary statistics
    ax10 = fig.add_subplot(gs[3, :2])
    ax10.axis('off')
    
    summary_text = f"""
SUMMARY STATISTICS
{'='*60}
BeatAML (n={len(beataml)}):
  Persisters: {(beataml['persister_probability'] >= 0.31).sum()} ({100*(beataml['persister_probability'] >= 0.31).mean():.1f}%)
  Mean probability: {beataml['persister_probability'].mean():.3f}
  Median: {beataml['persister_probability'].median():.3f}
  Std: {beataml['persister_probability'].std():.3f}

TCGA (n={len(tcga)}):
  Persisters: {(tcga['persister_probability'] >= 0.31).sum()} ({100*(tcga['persister_probability'] >= 0.31).mean():.1f}%)
  Mean probability: {tcga['persister_probability'].mean():.3f}
  Median: {tcga['persister_probability'].median():.3f}
  Std: {tcga['persister_probability'].std():.3f}

Cross-dataset consistency:
  Mean difference: {abs(beataml['persister_probability'].mean() - tcga['persister_probability'].mean()):.3f}
  Persister % difference: {abs((beataml['persister_probability'] >= 0.31).mean() - (tcga['persister_probability'] >= 0.31).mean())*100:.1f}%
  Optimal threshold for consistency: {sensitivity_df.loc[sensitivity_df['difference'].idxmin(), 'threshold']:.2f}
"""
    
    ax10.text(0.05, 0.9, summary_text, transform=ax10.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 11. Module score heatmap (if available)
    if module_scores_beat is not None:
        ax11 = fig.add_subplot(gs[3, 2:])
        # Sample some scores for visualization
        n_show = min(50, len(module_scores_beat))
        sample_idx = np.random.choice(len(module_scores_beat), n_show, replace=False)
        
        scores_subset = module_scores_beat.iloc[sample_idx]
        sns.heatmap(scores_subset.T, cmap='RdBu_r', center=0, ax=ax11,
                   xticklabels=False, yticklabels=True, cbar_kws={'label': 'Module Score'})
        ax11.set_title(f'Module Scores (n={n_show} samples)')
        ax11.set_xlabel('Samples')
    
    plt.suptitle('Comprehensive Bulk RNA-seq Persister Analysis', fontsize=16, y=1.02)
    plt.savefig(RESULTS_DIR / "comprehensive_analysis" / "all_analyses.png", 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    print("  Saved: comprehensive_analysis/all_analyses.png")

# ============================================================================
# PART 7: GENERATE REPORT
# ============================================================================

def generate_report(beataml, tcga, clinical_results, sensitivity_df):
    """Generate comprehensive text report"""
    
    report = []
    report.append("\n" + "="*80)
    report.append("COMPREHENSIVE BULK RNA-SEQ PERSISTER ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"Generated: {pd.Timestamp.now()}")
    
    # Dataset summary
    report.append("\n1. DATASET SUMMARY")
    report.append("-"*40)
    report.append(f"BeatAML: {len(beataml)} samples")
    report.append(f"  - Persisters: {(beataml['persister_probability'] >= 0.31).sum()} ({100*(beataml['persister_probability'] >= 0.31).mean():.1f}%)")
    report.append(f"  - Mean probability: {beataml['persister_probability'].mean():.3f}")
    report.append(f"TCGA-LAML: {len(tcga)} samples")
    report.append(f"  - Persisters: {(tcga['persister_probability'] >= 0.31).sum()} ({100*(tcga['persister_probability'] >= 0.31).mean():.1f}%)")
    report.append(f"  - Mean probability: {tcga['persister_probability'].mean():.3f}")
    
    # Statistical comparison
    report.append("\n2. CROSS-DATASET VALIDATION")
    report.append("-"*40)
    ks_stat, ks_p = stats.ks_2samp(beataml['persister_probability'], tcga['persister_probability'])
    mw_stat, mw_p = mannwhitneyu(beataml['persister_probability'], tcga['persister_probability'])
    report.append(f"Kolmogorov-Smirnov test: p={ks_p:.4f}")
    report.append(f"Mann-Whitney U test: p={mw_p:.4f}")
    report.append(f"Conclusion: {'No significant difference' if mw_p > 0.05 else 'Significant difference'} between datasets")
    
    # Clinical associations
    report.append("\n3. CLINICAL ASSOCIATIONS")
    report.append("-"*40)
    if 'beataml_blast_correlation' in clinical_results:
        res = clinical_results['beataml_blast_correlation']
        report.append(f"BeatAML Blast % correlation:")
        report.append(f"  - Spearman ρ = {res['rho']:.3f} (p={res['p_value']:.3e})")
        report.append(f"  - {'SIGNIFICANT' if res['significant'] else 'Not significant'}")
    
    # Threshold sensitivity
    report.append("\n4. THRESHOLD SENSITIVITY")
    report.append("-"*40)
    optimal_idx = sensitivity_df['difference'].idxmin()
    report.append(f"Optimal threshold for consistency: {sensitivity_df.loc[optimal_idx, 'threshold']:.2f}")
    report.append(f"  - Maximizes agreement between datasets")
    report.append(f"  - Difference at optimal: {sensitivity_df.loc[optimal_idx, 'difference']:.1f}%")
    
    # Save report
    report_text = "\n".join(report)
    with open(RESULTS_DIR / "comprehensive_analysis" / "analysis_report.txt", "w") as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print("\n  Saved: comprehensive_analysis/analysis_report.txt")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all analyses"""
    
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE BULK RNA-SEQ PERSISTER ANALYSIS")
    print("="*80)
    
    # Load all data
    beataml, tcga = load_all_data()
    
    # Create comprehensive modules
    modules = create_comprehensive_modules()
    
    # Calculate module scores for BeatAML
    print("\nCalculating module scores for BeatAML...")
    module_scores_beat = calculate_module_scores(
        BEATAML_DIR / "processed/expression.csv", 
        modules
    )
    module_scores_beat.to_csv(BEATAML_DIR / "analysis/module_scores.csv")
    
    # Analyze clinical associations
    clinical_results = analyze_clinical_associations(beataml, tcga)
    
    # Threshold sensitivity analysis
    sensitivity_df = threshold_sensitivity_analysis(
        BEATAML_DIR / "predictions_fixed.csv",
        TCGA_DIR / "predictions_fixed.csv"
    )
    
    # Create comprehensive visualizations
    create_comprehensive_figures(beataml, tcga, sensitivity_df, module_scores_beat)
    
    # Generate report
    generate_report(beataml, tcga, clinical_results, sensitivity_df)
    
    # Module-persister correlation
    if module_scores_beat is not None:
        print("\n" + "="*60)
        print("MODULE-PERSISTER CORRELATIONS")
        print("="*60)
        
        beataml_with_modules = beataml.set_index('sample_id').join(module_scores_beat)
        
        for module in module_scores_beat.columns:
            if module in beataml_with_modules.columns:
                valid = beataml_with_modules[['persister_probability', module]].dropna()
                if len(valid) > 10:
                    rho, p = spearmanr(valid['persister_probability'], valid[module])
                    print(f"  {module}: ρ={rho:.3f}, p={p:.3e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nAll results saved to:")
    print(f"  - {RESULTS_DIR}/comprehensive_analysis/")
    print(f"  - {BEATAML_DIR}/analysis/")
    print(f"  - {TCGA_DIR}/analysis/")

if __name__ == "__main__":
    main()
