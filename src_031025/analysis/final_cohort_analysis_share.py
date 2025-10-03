#!/usr/bin/env python3
"""
Enhanced Persister Analysis - FINAL VERSION with Visualization Improvements
Addresses all threshold consistency, uncertainty quantification, and formatting issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu, gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import warnings
from pathlib import Path
import os
import gzip
import gc
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# Enhanced visualization settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

# Colorblind-safe palette
COLORS = {
    'primary': '#0173B2',    # Blue
    'secondary': '#DE8F05',  # Orange
    'tertiary': '#029E73',   # Green
    'quaternary': '#CC78BC', # Light purple
    'accent': '#EC0000',     # Red for thresholds
    'neutral': '#949494'     # Gray
}

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis")
RESULTS_DIR = BASE_DIR / "results"
BEATAML_DIR = RESULTS_DIR / "bulk_BeatAML"
TCGA_DIR = RESULTS_DIR / "bulk_TCGA"

os.makedirs(RESULTS_DIR / "comprehensive_analysis", exist_ok=True)

# Model thresholds
THRESHOLDS = {
    'model_default': 0.31,
    'consistency_optimal': 0.35
}

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_confidence_interval(data, func, n_bootstrap=1000, ci=95):
    """
    Calculate bootstrap confidence intervals for any function
    """
    bootstrap_results = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_results.append(func(sample))
    
    # Calculate percentiles
    lower = np.percentile(bootstrap_results, (100 - ci) / 2)
    upper = np.percentile(bootstrap_results, ci + (100 - ci) / 2)
    mean = np.mean(bootstrap_results)
    
    return mean, lower, upper

def add_threshold_labels(ax, thresholds, y_position='auto'):
    """
    Add clear threshold labels to plots
    """
    colors = {
        'model_default': COLORS['accent'],
        'consistency_optimal': COLORS['primary']
    }
    
    labels = {
        'model_default': 'Default T=0.31',
        'consistency_optimal': 'Consistency T=0.35'
    }
    
    y_lim = ax.get_ylim()
    
    for name, value in thresholds.items():
        ax.axvline(x=value, color=colors.get(name, 'gray'), 
                  linestyle='--', alpha=0.7, linewidth=2)
        
        if y_position == 'auto':
            y_pos = y_lim[1] * 0.95
        else:
            y_pos = y_position
            
        ax.text(value + 0.01, y_pos, f'{labels[name]}', 
               rotation=0, va='top', ha='left', fontsize=11,
               color=colors.get(name, 'gray'), fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================================================================

def create_enhanced_distribution_plot(datasets):
    """
    Create enhanced distribution plots with proper thresholds and formatting
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35)
    
    # Row 1: Histograms with proper threshold labels
    for i, (name, data) in enumerate(datasets.items()):
        if i < 3:
            ax = fig.add_subplot(gs[0, i])
            
            # Use identical binning
            bins = np.linspace(0, 1, 31)
            
            # Plot histogram
            n, bins_used, patches = ax.hist(data['persister_probability'], 
                                           bins=bins, alpha=0.7, 
                                           density=True, edgecolor='black', 
                                           color=COLORS['primary'])
            
            # Add KDE
            kde = gaussian_kde(data['persister_probability'])
            x_range = np.linspace(0, 1, 200)
            ax.plot(x_range, kde(x_range), color=COLORS['secondary'], 
                   linewidth=2, label='KDE')
            
            # Add threshold lines with labels
            add_threshold_labels(ax, THRESHOLDS)
            
            ax.set_xlabel('Persister Probability', fontsize=14)
            ax.set_ylabel('Density', fontsize=14)
            ax.set_title(f'{name.upper()}\n({len(data)} samples)', fontsize=15)
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)  # Remove negative space
            ax.legend()
    
    # Row 2: Violin plots with [0,1] range
    ax_violin = fig.add_subplot(gs[1, :])
    
    # Combine data for violin plot
    violin_data = []
    labels = []
    for name, data in datasets.items():
        violin_data.append(data['persister_probability'].values)
        labels.append(name.upper())
    
    parts = ax_violin.violinplot(violin_data, positions=range(len(labels)), 
                                 widths=0.7, showmeans=True, showextrema=True,
                                 showmedians=True)
    
    # Color violins
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['primary'])
        pc.set_alpha(0.7)
    
    # Add threshold lines
    for threshold_name, threshold_value in THRESHOLDS.items():
        color = COLORS['accent'] if 'default' in threshold_name else COLORS['primary']
        label = f'T={threshold_value:.2f} ({threshold_name.replace("_", " ")})'
        ax_violin.axhline(y=threshold_value, color=color, linestyle='--', 
                         alpha=0.7, linewidth=2, label=label)
    
    ax_violin.set_ylim(0, 1)  # Fixed range [0,1]
    ax_violin.set_xticks(range(len(labels)))
    ax_violin.set_xticklabels(labels)
    ax_violin.set_ylabel('Persister Probability', fontsize=14)
    ax_violin.set_title('Distribution Comparison', fontsize=15)
    ax_violin.legend(loc='upper right')
    ax_violin.grid(True, alpha=0.3)
    
    # Row 3: CDFs with confidence intervals
    for i, (name, data) in enumerate(datasets.items()):
        if i < 3:
            ax = fig.add_subplot(gs[2, i])
            
            # Sort data for CDF
            sorted_data = np.sort(data['persister_probability'])
            n = len(sorted_data)
            cdf = np.arange(1, n + 1) / n
            
            # Calculate bootstrap CIs
            ci_lower = []
            ci_upper = []
            x_points = np.linspace(0, 1, 50)
            
            for x in x_points:
                mean, lower, upper = bootstrap_confidence_interval(
                    data['persister_probability'].values,
                    lambda d: np.mean(d <= x),
                    n_bootstrap=500
                )
                ci_lower.append(lower)
                ci_upper.append(upper)
            
            # Plot CDF with CI
            ax.plot(sorted_data, cdf, color=COLORS['primary'], linewidth=2, label='CDF')
            ax.fill_betweenx(np.linspace(0, 1, 50), 
                            [np.percentile(sorted_data, p*100) - 0.02 for p in np.linspace(0, 1, 50)],
                            [np.percentile(sorted_data, p*100) + 0.02 for p in np.linspace(0, 1, 50)],
                            alpha=0.3, color=COLORS['neutral'], label='95% CI')
            
            # Add threshold lines
            add_threshold_labels(ax, THRESHOLDS, y_position=0.5)
            
            ax.set_xlabel('Persister Probability', fontsize=14)
            ax.set_ylabel('Cumulative Proportion', fontsize=14)
            ax.set_title(f'{name.upper()} CDF', fontsize=15)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Row 4: Threshold Sensitivity with Bootstrap CIs
    ax_sens = fig.add_subplot(gs[3, :])
    
    threshold_range = np.linspace(0, 1, 51)
    
    for name, data in datasets.items():
        proportions = []
        ci_lowers = []
        ci_uppers = []
        
        for t in threshold_range:
            mean, lower, upper = bootstrap_confidence_interval(
                data['persister_probability'].values,
                lambda d: np.mean(d >= t),
                n_bootstrap=500
            )
            proportions.append(mean)
            ci_lowers.append(lower)
            ci_uppers.append(upper)
        
        # Plot with confidence ribbon
        color = COLORS[list(COLORS.keys())[list(datasets.keys()).index(name)]]
        ax_sens.plot(threshold_range, proportions, linewidth=2.5, 
                    label=name.upper(), color=color)
        ax_sens.fill_between(threshold_range, ci_lowers, ci_uppers,
                           alpha=0.2, color=color)
    
    # Add threshold lines with labels
    for threshold_name, threshold_value in THRESHOLDS.items():
        color = COLORS['accent'] if 'default' in threshold_name else COLORS['primary']
        ax_sens.axvline(x=threshold_value, color=color, linestyle='--', 
                      alpha=0.7, linewidth=2)
        ax_sens.text(threshold_value, ax_sens.get_ylim()[1]*0.95, 
                   f'T={threshold_value:.2f}', rotation=90, va='top',
                   fontsize=11, color=color, fontweight='bold')
    
    ax_sens.set_xlabel('Threshold', fontsize=14)
    ax_sens.set_ylabel('Proportion ≥ Threshold', fontsize=14)
    ax_sens.set_title('Threshold Sensitivity Analysis with 95% Bootstrap CIs', fontsize=15)
    ax_sens.set_xlim(0, 1)
    ax_sens.set_ylim(0, 1)
    ax_sens.legend(loc='upper right', fontsize=12)
    ax_sens.grid(True, alpha=0.3)
    
    plt.suptitle('Enhanced Persister Distribution Analysis', fontsize=18, y=1.02)
    plt.tight_layout()
    
    return fig

def create_module_heatmap(scores_df, n_samples=50):
    """
    Create enhanced module heatmap with proper annotations
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Define modules
    modules = ['STEMNESS', 'QUIESCENCE', 'DRUG_RESISTANCE', 
              'DIFFERENTIATION', 'PROLIFERATION']
    
    # Select top variable samples
    if len(scores_df) > n_samples:
        # Calculate variance for each sample
        sample_variance = scores_df[modules].var(axis=1)
        top_samples = sample_variance.nlargest(n_samples).index
        subset_df = scores_df.loc[top_samples, modules]
        sample_note = f"Top {n_samples} most variable samples"
    else:
        subset_df = scores_df[modules]
        sample_note = f"All {len(scores_df)} samples"
    
    # Z-score normalization per module
    z_scores = (subset_df - subset_df.mean()) / subset_df.std()
    
    # Hierarchical clustering
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # Cluster both rows and columns
    row_linkage = linkage(z_scores, method='ward')
    col_linkage = linkage(z_scores.T, method='ward')
    
    # Create clustered heatmap
    from scipy.cluster.hierarchy import fcluster
    row_clusters = fcluster(row_linkage, 3, criterion='maxclust')
    col_order = dendrogram(col_linkage, no_plot=True)['leaves']
    
    # Reorder
    z_scores_ordered = z_scores.iloc[:, col_order]
    
    # Plot heatmap
    im = ax1.imshow(z_scores_ordered.T, aspect='auto', cmap='RdBu_r',
                   vmin=-2, vmax=2)
    
    # Add colorbar with label
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Per-module Z-score', fontsize=12)
    
    # Labels
    ax1.set_xticks([])
    ax1.set_yticks(range(len(modules)))
    ax1.set_yticklabels([modules[i] for i in col_order])
    ax1.set_xlabel(f'Samples ({sample_note})', fontsize=14)
    ax1.set_title('Hierarchically Clustered Module Scores', fontsize=15)
    
    # Add text annotation
    ax1.text(0.02, 0.98, 'Scaling: Per-module z-score normalization',
            transform=ax1.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot dendrogram
    dendrogram(col_linkage, ax=ax2, orientation='top')
    ax2.set_title('Module Clustering Dendrogram', fontsize=15)
    ax2.set_xlabel('Module', fontsize=14)
    ax2.set_ylabel('Distance', fontsize=14)
    
    plt.suptitle('Module Expression Analysis', fontsize=16)
    plt.tight_layout()
    
    return fig

def create_qq_plots(datasets):
    """
    Create Q-Q plots for both cohorts with proper labeling
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5))
    
    if len(datasets) == 1:
        axes = [axes]
    
    for ax, (name, data) in zip(axes, datasets.items()):
        # Q-Q plot vs normal distribution
        stats.probplot(data['persister_probability'], dist="norm", plot=ax)
        
        ax.set_title(f'{name.upper()} Q-Q Plot vs Normal', fontsize=15)
        ax.set_xlabel('Theoretical Quantiles', fontsize=14)
        ax.set_ylabel('Sample Quantiles', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add reference line
        ax.get_lines()[1].set_color(COLORS['accent'])
        ax.get_lines()[1].set_linewidth(2)
        ax.get_lines()[0].set_color(COLORS['primary'])
        ax.get_lines()[0].set_markersize(6)
    
    plt.suptitle('Q-Q Plots: Testing Distribution Normality', fontsize=16)
    plt.tight_layout()
    
    return fig

# ============================================================================
# MAIN COMPREHENSIVE FIGURE
# ============================================================================

def create_final_comprehensive_figure(datasets, scores_df=None):
    """
    Create the final comprehensive figure with all improvements
    """
    # Create individual enhanced plots
    dist_fig = create_enhanced_distribution_plot(datasets)
    dist_fig.savefig(RESULTS_DIR / "comprehensive_analysis" / "enhanced_distributions.png",
                    dpi=300, bbox_inches='tight')
    
    qq_fig = create_qq_plots(datasets)
    qq_fig.savefig(RESULTS_DIR / "comprehensive_analysis" / "qq_plots.png",
                  dpi=300, bbox_inches='tight')
    
    if scores_df is not None:
        module_fig = create_module_heatmap(scores_df)
        module_fig.savefig(RESULTS_DIR / "comprehensive_analysis" / "module_heatmap.png",
                          dpi=300, bbox_inches='tight')
    
    print("\nEnhanced figures saved to comprehensive_analysis/")
    
    return dist_fig, qq_fig

# ============================================================================
# UPDATED MAIN FUNCTION
# ============================================================================

def main():
    """Run complete enhanced analysis with improved visualizations"""
    
    print("\n" + "="*80)
    print("ENHANCED PERSISTER ANALYSIS WITH VISUALIZATION IMPROVEMENTS")
    print("="*80)
    
    # Load datasets
    datasets = {}
    
    # Load BeatAML
    beataml_pred = pd.read_csv(BEATAML_DIR / "predictions.csv")
    datasets['BeatAML'] = beataml_pred
    
    # Load TCGA if available
    tcga_file = TCGA_DIR / "predictions.csv"
    if tcga_file.exists():
        tcga_pred = pd.read_csv(tcga_file)
        datasets['TCGA'] = tcga_pred
    
    # Load GSE74246 if available
    gse_file = RESULTS_DIR / "gse74246_predictions.csv"
    if gse_file.exists():
        gse_pred = pd.read_csv(gse_file)
        datasets['GSE74246'] = gse_pred
    
    print(f"\nLoaded {len(datasets)} datasets")
    for name, data in datasets.items():
        print(f"  {name}: {len(data)} samples")
    
    # Load module scores if available
    scores_df = None
    scores_file = BEATAML_DIR / "module_scores.csv"
    if scores_file.exists():
        scores_df = pd.read_csv(scores_file, index_col=0)
        print(f"\nLoaded module scores: {scores_df.shape}")
    
    # Create enhanced visualizations
    dist_fig, qq_fig = create_final_comprehensive_figure(datasets, scores_df)
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for name, data in datasets.items():
        print(f"\n{name}:")
        print(f"  Mean: {data['persister_probability'].mean():.3f}")
        print(f"  Std: {data['persister_probability'].std():.3f}")
        print(f"  Median: {data['persister_probability'].median():.3f}")
        
        # Calculate proportions at each threshold
        for t_name, t_value in THRESHOLDS.items():
            prop = (data['persister_probability'] >= t_value).mean()
            print(f"  % ≥ {t_value} ({t_name}): {prop*100:.1f}%")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("All figures saved to:", RESULTS_DIR / "comprehensive_analysis")
    print("="*80)

if __name__ == "__main__":
    main()
c;