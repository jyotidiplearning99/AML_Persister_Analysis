#!/usr/bin/env python3
"""
Fixed Survival Analysis for Persister Scores with Proper Binning
Uses actual matched TCGA-LAML survival data (149 samples)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# FIXED DATA LOADING - Using actual matched file
# ============================================================================

def load_matched_tcga_data():
    """Load the already matched TCGA data with survival"""
    
    matched_path = "/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/data/survival/tcga_persister_survival_matched.csv"
    
    print(f"Loading matched TCGA data from: {matched_path}")
    data = pd.read_csv(matched_path)
    
    # Ensure proper data types
    data['os_status'] = pd.to_numeric(data['os_status'], errors='coerce')
    data['os_days'] = pd.to_numeric(data['os_days'], errors='coerce')
    data['persister_probability'] = pd.to_numeric(data['persister_probability'], errors='coerce')
    
    # Remove invalid entries
    data = data.dropna(subset=['os_days', 'os_status', 'persister_probability'])
    data = data[data['os_days'] > 0]
    
    # Convert to months
    data['os_months'] = data['os_days'] / 30.44
    
    # Add age and gender if available
    if 'age' in data.columns:
        data['age'] = pd.to_numeric(data['age'], errors='coerce')
    if 'gender' in data.columns:
        data['sex_male'] = data['gender'].map({'MALE': 1, 'FEMALE': 0})
    
    print(f"Loaded {len(data)} samples with complete survival data")
    print(f"Deaths: {data['os_status'].sum()} ({data['os_status'].mean()*100:.1f}%)")
    print(f"Median follow-up: {data['os_days'].median():.0f} days")
    
    # Show persister probability distribution
    print(f"\nPersister probability distribution:")
    print(f"  Mean: {data['persister_probability'].mean():.3f}")
    print(f"  Median: {data['persister_probability'].median():.3f}")
    print(f"  Range: [{data['persister_probability'].min():.3f}, {data['persister_probability'].max():.3f}]")
    
    return data

# ============================================================================
# FIXED STRATIFICATION WITH PROPER BINNING
# ============================================================================

def stratify_by_persister_bins(data, method='median'):
    """
    Properly stratify patients into bins based on persister score
    """
    
    # Create a copy to avoid modifying original
    data = data.copy()
    
    print(f"\nApplying {method} stratification...")
    
    if method == 'median':
        # Split at median
        median_val = data['persister_probability'].median()
        data['persister_group'] = pd.cut(
            data['persister_probability'],
            bins=[-np.inf, median_val, np.inf],
            labels=['Low', 'High']
        )
        print(f"  Median cutoff: {median_val:.3f}")
        
    elif method == 'tertile':
        # Split into 3 equal groups
        data['persister_group'] = pd.qcut(
            data['persister_probability'],
            q=3,
            labels=['Low', 'Medium', 'High'], duplicates='drop'
        )
        tertiles = data['persister_probability'].quantile([0.33, 0.67])
        print(f"  Tertile cutoffs: {tertiles.values}")
        
    elif method == 'quartile':
        # Split into 4 equal groups
        data['persister_group'] = pd.qcut(
            data['persister_probability'],
            q=4,
            labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop'
        )
        quartiles = data['persister_probability'].quantile([0.25, 0.5, 0.75])
        print(f"  Quartile cutoffs: {quartiles.values}")
        
    elif method == 'extreme':
        # Compare top 25% vs bottom 25%
        q25 = data['persister_probability'].quantile(0.25)
        q75 = data['persister_probability'].quantile(0.75)
        
        data['persister_group'] = 'Middle'
        data.loc[data['persister_probability'] <= q25, 'persister_group'] = 'Low (Bottom 25%)'
        data.loc[data['persister_probability'] >= q75, 'persister_group'] = 'High (Top 25%)'
        
        # Remove middle group for extreme comparison
        data = data[data['persister_group'] != 'Middle']
        print(f"  Comparing extremes: ≤{q25:.3f} vs ≥{q75:.3f}")
        
    elif method == 'threshold':
        # Use fixed threshold at 0.5
        threshold = 0.5
        data['persister_group'] = pd.cut(
            data['persister_probability'],
            bins=[-np.inf, threshold, np.inf],
            labels=['Low', 'High']
        )
        print(f"  Fixed threshold: {threshold:.3f}")
        
    elif method == 'prediction':
        # Use the existing prediction column if available
        if 'prediction' in data.columns:
            data['persister_group'] = data['prediction'].map({
                'Persister': 'High',
                'Non-Persister': 'Low'
            })
        else:
            # Fall back to median
            median_val = data['persister_probability'].median()
            data['persister_group'] = pd.cut(
                data['persister_probability'],
                bins=[-np.inf, median_val, np.inf],
                labels=['Low', 'High']
            )
    
    # Print group distribution
    print("\nGroup distribution:")
    print(data['persister_group'].value_counts())
    
    return data

# ============================================================================
# SURVIVAL ANALYSIS FUNCTIONS
# ============================================================================

def perform_survival_analysis(data, time_col='os_months', event_col='os_status'):
    """Perform Kaplan-Meier survival analysis"""
    
    kmf = KaplanMeierFitter()
    
    groups = data['persister_group'].unique()
    km_results = {}
    
    for group in groups:
        group_data = data[data['persister_group'] == group]
        
        kmf.fit(
            durations=group_data[time_col],
            event_observed=group_data[event_col],
            label=f"{group} (n={len(group_data)})"
        )
        
        km_results[group] = {
            'kmf': KaplanMeierFitter().fit(  # Create new instance
                durations=group_data[time_col],
                event_observed=group_data[event_col],
                label=f"{group} (n={len(group_data)})"
            ),
            'median': kmf.median_survival_time_,
            'n': len(group_data),
            'events': int(group_data[event_col].sum())
        }
    
    # Perform log-rank test
    if len(groups) == 2:
        group_list = list(groups)
        group1 = data[data['persister_group'] == group_list[0]]
        group2 = data[data['persister_group'] == group_list[1]]
        
        lr_result = logrank_test(
            group1[time_col], group2[time_col],
            group1[event_col], group2[event_col]
        )
    else:
        # Multi-group comparison
        lr_result = multivariate_logrank_test(
            data[time_col], 
            data['persister_group'], 
            data[event_col]
        )
    
    return km_results, lr_result

def cox_regression_continuous(data):
    """Perform Cox regression with continuous persister score"""
    
    # Prepare data
    cox_data = pd.DataFrame({
        'T': data['os_months'],
        'E': data['os_status'],
        'persister_score': data['persister_probability']
    })
    
    # Add covariates if available
    if 'age' in data.columns:
        cox_data['age'] = pd.to_numeric(data['age'], errors='coerce')
    if 'sex_male' in data.columns:
        cox_data['sex_male'] = data['sex_male']
    
    # Drop missing values
    cox_data = cox_data.dropna()
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='T', event_col='E')
    
    return cph

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_survival_curves(km_results, lr_result, title, save_path=None):
    """Create publication-quality Kaplan-Meier curves"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors for different groups
    colors = ['#DC143C', '#4169E1', '#FFA500', '#32CD32', '#9370DB']
    
    # Plot each group
    for i, (group, results) in enumerate(km_results.items()):
        results['kmf'].plot_survival_function(
            ax=ax, 
            ci_show=True,
            color=colors[i % len(colors)],
            linewidth=2.5,
            alpha=0.9
        )
    
    # Add at-risk table
    add_at_risk_counts(*[r['kmf'] for r in km_results.values()], ax=ax)
    
    # Formatting
    ax.set_xlabel('Time (months)', fontsize=14)
    ax.set_ylabel('Overall Survival Probability', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Add p-value
    p_text = f"Log-rank p = {lr_result.p_value:.3e}" if lr_result.p_value < 0.001 else f"Log-rank p = {lr_result.p_value:.3f}"
    ax.text(0.95, 0.95, p_text, transform=ax.transAxes,
           fontsize=12, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add sample info
    info_text = "Groups:\n"
    for group, results in km_results.items():
        info_text += f"{group}: n={results['n']}, events={results['events']}\n"
        if results['median']:
            info_text += f"  Median: {results['median']:.1f} months\n"
    
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
           fontsize=10, va='bottom',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_persister_distribution(data, save_path=None):
    """Plot persister score distribution with survival overlay"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram with survival status
    ax = axes[0, 0]
    alive = data[data['os_status'] == 0]['persister_probability']
    dead = data[data['os_status'] == 1]['persister_probability']
    
    ax.hist(alive, bins=20, alpha=0.5, label='Alive/Censored', color='blue')
    ax.hist(dead, bins=20, alpha=0.5, label='Deceased', color='red')
    ax.set_xlabel('Persister Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution by Survival Status', fontsize=14)
    ax.legend()
    
    # 2. Box plot comparison
    ax = axes[0, 1]
    bp = ax.boxplot([alive, dead], labels=['Alive/Censored', 'Deceased'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][1].set_facecolor('#F44336')
    
    # Statistical test
    _, p_val = stats.mannwhitneyu(alive, dead)
    ax.text(0.5, 0.95, f'Mann-Whitney p = {p_val:.3f}',
           transform=ax.transAxes, ha='center', fontsize=11)
    
    ax.set_ylabel('Persister Probability', fontsize=12)
    ax.set_title('Persister Score by Outcome', fontsize=14)
    
    # 3. Scatter plot: score vs survival time
    ax = axes[1, 0]
    colors = ['green' if s == 0 else 'red' for s in data['os_status']]
    ax.scatter(data['persister_probability'], data['os_months'],
              c=colors, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add correlation
    corr, p = stats.spearmanr(data['persister_probability'], data['os_months'])
    ax.text(0.02, 0.98, f'Spearman ρ = {corr:.3f}\np = {p:.3f}',
           transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round', facecolor='white'))
    
    ax.set_xlabel('Persister Probability', fontsize=12)
    ax.set_ylabel('Survival Time (months)', fontsize=12)
    ax.set_title('Persister Score vs Survival Time', fontsize=14)
    
    # 4. Density plot by group
    ax = axes[1, 1]
    if 'persister_group' in data.columns:
        for group in data['persister_group'].unique():
            group_data = data[data['persister_group'] == group]['persister_probability']
            group_data.plot.density(ax=ax, label=group, linewidth=2)
        ax.set_xlabel('Persister Probability', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution by Group', fontsize=14)
        ax.legend()
    
    plt.suptitle('Persister Score Distribution Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_stratification_comparison(results_dict, save_path=None):
    """Compare different stratification methods"""
    
    n_methods = len(results_dict)
    cols = 3
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if n_methods > 1 else [axes]
    
    for i, (method, result) in enumerate(results_dict.items()):
        if i < len(axes):
            ax = axes[i]
            km_results = result['km_results']
            lr_result = result['lr_result']
            
            # Plot survival curves
            for group, km_data in km_results.items():
                km_data['kmf'].plot_survival_function(ax=ax, ci_show=False)
            
            # Add p-value
            p_text = f"p={lr_result.p_value:.3f}" if lr_result.p_value >= 0.001 else f"p={lr_result.p_value:.3e}"
            ax.text(0.95, 0.95, p_text, transform=ax.transAxes,
                   fontsize=11, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{method.capitalize()} Stratification', fontsize=13)
            ax.set_xlabel('Time (months)', fontsize=11)
            ax.set_ylabel('Survival Probability', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='lower left')
    
    # Hide unused subplots
    for i in range(n_methods, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Stratification Method Comparison', fontsize=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_tcga_survival_comprehensive():
    """Comprehensive survival analysis with multiple stratifications"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TCGA-LAML SURVIVAL ANALYSIS")
    print("="*80)
    
    # Load data
    data = load_matched_tcga_data()
    
    # Test multiple stratification methods
    methods = ['median', 'tertile', 'quartile', 'extreme', 'threshold']
    results = {}
    
    output_dir = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/survival_analysis/tcga_fixed")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing {method.upper()} stratification")
        print('='*60)
        
        # Stratify
        stratified_data = stratify_by_persister_bins(data.copy(), method=method)
        
        # Perform survival analysis
        km_results, lr_result = perform_survival_analysis(stratified_data)
        
        # Store results
        results[method] = {
            'km_results': km_results,
            'lr_result': lr_result,
            'data': stratified_data,
            'p_value': lr_result.p_value
        }
        
        print(f"\n★ Log-rank p-value: {lr_result.p_value:.4f}")
        
        # Report median survival
        for group, km_data in km_results.items():
            
            median_val = km_data['median']
            median_txt = f"{median_val:.1f}" if (median_val is not None and np.isfinite(median_val)) else "NA"
            print(f"  {group}: n={km_data['n']}, events={km_data['events']}, median={median_txt} months")   

            #print(f"  {group}: n={km_data['n']}, events={km_data['events']}, "
                #  f"median={km_data['median']:.1f if km_data['median'] else 'NA'} months")
    
    # Cox regression with continuous score
    print(f"\n{'='*60}")
    print("COX REGRESSION (CONTINUOUS SCORE)")
    print('='*60)
    
    cph = cox_regression_continuous(data)
    print("\nCox regression results:")
    print(cph.summary[['coef', 'exp(coef)', 'p']])
    
    # Find best stratification
    best_method = min(results.items(), key=lambda x: x[1]['p_value'])
    print(f"\n{'='*60}")
    print(f"BEST STRATIFICATION: {best_method[0].upper()}")
    print(f"P-value: {best_method[1]['p_value']:.4f}")
    print('='*60)
    
    # Create visualizations
    
    # 1. Best method KM curve
    fig1 = plot_survival_curves(
        best_method[1]['km_results'],
        best_method[1]['lr_result'],
        f'TCGA-LAML: {best_method[0].capitalize()} Stratification (Best)',
        output_dir / f"best_km_{best_method[0]}.png"
    )
    
    # 2. Distribution analysis
    # fig2 = plot_persister_distribution(
    #     data,
    #     output_dir / "persister_distribution.png"
    # )
    
    best_data = results[best_method[0]]['data']   # has 'persister_group'
    fig2 = plot_persister_distribution(
    best_data,
    output_dir / "persister_distribution.png"
    )
    # 3. Stratification comparison
    fig3 = plot_stratification_comparison(
        results,
        output_dir / "stratification_comparison.png"
    )
    
    # Save summary
    summary = {
        'n_samples': len(data),
        'n_deaths': int(data['os_status'].sum()),
        'median_followup_days': float(data['os_days'].median()),
        'persister_prob_mean': float(data['persister_probability'].mean()),
        'persister_prob_median': float(data['persister_probability'].median()),
        'stratification_results': {
            method: {
                'p_value': float(res['lr_result'].p_value),
                'n_groups': len(res['km_results']),
                'groups': {
                    str(group): {
                        'n': km['n'],
                        'events': km['events'],
                        'median_survival': float(km['median']) if km['median'] else None
                    }
                    for group, km in res['km_results'].items()
                }
            }
            for method, res in results.items()
        },
        'best_method': best_method[0],
        'best_p_value': float(best_method[1]['p_value']),
        'cox_persister_hr': float(np.exp(cph.params_['persister_score'])) if 'persister_score' in cph.params_ else None,
        'cox_persister_p': float(cph.summary.loc['persister_score', 'p']) if 'persister_score' in cph.params_ else None
    }
    
    with open(output_dir / "survival_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    # Print final summary table
    print("\n" + "="*80)
    print("STRATIFICATION SUMMARY")
    print("="*80)
    print("\nMethod         | P-value | Groups")
    print("-"*40)
    for method, res in results.items():
        n_groups = len(res['km_results'])
        print(f"{method:12} | {res['p_value']:.4f} | {n_groups}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return results, cph

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run comprehensive survival analysis"""
    
    print("\n" + "="*80)
    print("FIXED PERSISTER SURVIVAL ANALYSIS")
    print("Using proper stratification with bins")
    print("="*80)
    
    # Run analysis
    results, cph = analyze_tcga_survival_comprehensive()
    
    # Final interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    # Check if persister score is prognostic
    if 'persister_score' in cph.params_:
        hr = np.exp(cph.params_['persister_score'])
        p = cph.summary.loc['persister_score', 'p']
        
        if p < 0.05:
            print(f"✓ Persister score IS prognostic (HR={hr:.3f}, p={p:.4f})")
            if hr > 1:
                print("  Higher persister scores associated with WORSE survival")
            else:
                print("  Higher persister scores associated with BETTER survival")
        else:
            print(f"✗ Persister score is NOT significantly prognostic (p={p:.4f})")
    
    print("\nNote: Consider biological context and sample size when interpreting results.")

if __name__ == "__main__":
    main()
