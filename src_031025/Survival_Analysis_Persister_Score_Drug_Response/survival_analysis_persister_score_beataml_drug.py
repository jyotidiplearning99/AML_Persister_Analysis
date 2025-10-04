#!/usr/bin/env python3
"""
BeatAML Survival Analysis for Persister Scores with Drug Response Integration
Complete analysis parallel to TCGA with drug associations
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
# DATA LOADING - BeatAML with proper survival
# ============================================================================

def load_matched_beataml_data():
    """Load the matched BeatAML data with survival"""
    
    matched_path = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/data/survival/beataml_persister_survival_matched.csv")
    
    if not matched_path.exists():
        raise FileNotFoundError(f"Matched file not found: {matched_path}\nPlease run download_beataml_survival_fixed.py first")
    
    print(f"Loading matched BeatAML data from: {matched_path}")
    data = pd.read_csv(matched_path)
    
    # Check if we have survival data
    has_survival = False
    if 'os_status' in data.columns and 'os_days' in data.columns:
        # Ensure proper data types
        data['os_status'] = pd.to_numeric(data['os_status'], errors='coerce')
        data['os_days'] = pd.to_numeric(data['os_days'], errors='coerce')
        data['persister_probability'] = pd.to_numeric(data['persister_probability'], errors='coerce')
        
        # Remove invalid entries
        valid_survival = data.dropna(subset=['os_days', 'os_status', 'persister_probability'])
        valid_survival = valid_survival[valid_survival['os_days'] > 0]
        
        if len(valid_survival) > 10:
            has_survival = True
            data = valid_survival
            data['os_months'] = data['os_days'] / 30.44
            
            print(f"Loaded {len(data)} samples with complete survival data")
            print(f"Deaths: {data['os_status'].sum()} ({data['os_status'].mean()*100:.1f}%)")
            print(f"Median follow-up: {data['os_days'].median():.0f} days")
    
    if not has_survival:
        print(f"Loaded {len(data)} samples (no valid survival data)")
        # Still keep persister predictions for drug analysis
        data['persister_probability'] = pd.to_numeric(data['persister_probability'], errors='coerce')
        data = data.dropna(subset=['persister_probability'])
    
    # Add clinical variables if available
    if 'age' in data.columns:
        data['age'] = pd.to_numeric(data['age'], errors='coerce')
    if 'gender' in data.columns:
        data['sex_male'] = data['gender'].map({'MALE': 1, 'FEMALE': 0, 'Male': 1, 'Female': 0})
    
    # Show persister probability distribution
    print(f"\nPersister probability distribution:")
    print(f"  Mean: {data['persister_probability'].mean():.3f}")
    print(f"  Median: {data['persister_probability'].median():.3f}")
    print(f"  Range: [{data['persister_probability'].min():.3f}, {data['persister_probability'].max():.3f}]")
    
    return data, has_survival

# ============================================================================
# DRUG RESPONSE DATA LOADING
# ============================================================================

def load_drug_response_data(data):
    """Load and merge drug response data"""
    
    drug_dir = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/data/beataml_drugs")
    
    # Look for drug response files
    drug_files = [
        drug_dir / "processed" / "beataml_key_drugs_auc.csv",
        drug_dir / "beataml_drug_auc_sample.csv"
    ]
    
    drug_data = None
    for file_path in drug_files:
        if file_path.exists():
            drug_data = pd.read_csv(file_path)
            print(f"\nLoaded drug data from: {file_path.name}")
            print(f"  Shape: {drug_data.shape}")
            break
    
    if drug_data is None:
        print("\n⚠ No drug response data found")
        return data, False
    
    # List available drugs
    drug_cols = [col for col in drug_data.columns if 'AUC' in col or 'auc' in col]
    print(f"  Available drugs: {drug_cols[:5]}...")
    
    # Merge with survival data
    merged = pd.merge(data, drug_data, on='sample_id', how='inner')
    
    if len(merged) > 0:
        print(f"  Merged {len(merged)} samples with both survival and drug data")
        return merged, True
    else:
        print("  No matching samples found - trying alternative matching")
        # Try matching with extracted IDs
        data['sample_key'] = data['sample_id'].str.extract(r'(\d+-\d+)')
        drug_data['sample_key'] = drug_data['sample_id'].str.extract(r'(\d+-\d+)')
        
        if 'sample_key' in data.columns and 'sample_key' in drug_data.columns:
            merged = pd.merge(data, drug_data, on='sample_key', how='inner', suffixes=('', '_drug'))
            if len(merged) > 0:
                print(f"  Merged {len(merged)} samples using extracted IDs")
                return merged, True
        
        return data, False

# ============================================================================
# STRATIFICATION (Same as TCGA)
# ============================================================================

def stratify_by_persister_bins(data, method='median'):
    """Properly stratify patients into bins based on persister score"""
    
    data = data.copy()
    
    print(f"\nApplying {method} stratification...")
    
    if method == 'median':
        median_val = data['persister_probability'].median()
        data['persister_group'] = pd.cut(
            data['persister_probability'],
            bins=[-np.inf, median_val, np.inf],
            labels=['Low', 'High']
        )
        print(f"  Median cutoff: {median_val:.3f}")
        
    elif method == 'tertile':
        data['persister_group'] = pd.qcut(
            data['persister_probability'],
            q=3,
            labels=['Low', 'Medium', 'High'], 
            duplicates='drop'
        )
        tertiles = data['persister_probability'].quantile([0.33, 0.67])
        print(f"  Tertile cutoffs: {tertiles.values}")
        
    elif method == 'quartile':
        data['persister_group'] = pd.qcut(
            data['persister_probability'],
            q=4,
            labels=['Q1', 'Q2', 'Q3', 'Q4'], 
            duplicates='drop'
        )
        
    elif method == 'extreme':
        q25 = data['persister_probability'].quantile(0.25)
        q75 = data['persister_probability'].quantile(0.75)
        
        data['persister_group'] = 'Middle'
        data.loc[data['persister_probability'] <= q25, 'persister_group'] = 'Low (Bottom 25%)'
        data.loc[data['persister_probability'] >= q75, 'persister_group'] = 'High (Top 25%)'
        
        data = data[data['persister_group'] != 'Middle']
        print(f"  Comparing extremes: ≤{q25:.3f} vs ≥{q75:.3f}")
        
    elif method == 'threshold':
        threshold = 0.5
        data['persister_group'] = pd.cut(
            data['persister_probability'],
            bins=[-np.inf, threshold, np.inf],
            labels=['Low', 'High']
        )
    
    print("\nGroup distribution:")
    print(data['persister_group'].value_counts())
    
    return data

# ============================================================================
# SURVIVAL ANALYSIS FUNCTIONS (Same as TCGA)
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
            'kmf': KaplanMeierFitter().fit(
                durations=group_data[time_col],
                event_observed=group_data[event_col],
                label=f"{group} (n={len(group_data)})"
            ),
            'median': kmf.median_survival_time_,
            'n': len(group_data),
            'events': int(group_data[event_col].sum())
        }
    
    # Log-rank test
    if len(groups) == 2:
        group_list = list(groups)
        group1 = data[data['persister_group'] == group_list[0]]
        group2 = data[data['persister_group'] == group_list[1]]
        
        lr_result = logrank_test(
            group1[time_col], group2[time_col],
            group1[event_col], group2[event_col]
        )
    else:
        lr_result = multivariate_logrank_test(
            data[time_col], 
            data['persister_group'], 
            data[event_col]
        )
    
    return km_results, lr_result

def cox_regression_continuous(data):
    """Perform Cox regression with continuous persister score"""
    
    cox_data = pd.DataFrame({
        'T': data['os_months'],
        'E': data['os_status'],
        'persister_score': data['persister_probability']
    })
    
    if 'age' in data.columns:
        cox_data['age'] = pd.to_numeric(data['age'], errors='coerce')
    if 'sex_male' in data.columns:
        cox_data['sex_male'] = data['sex_male']
    
    cox_data = cox_data.dropna()
    
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='T', event_col='E')
    
    return cph

# ============================================================================
# DRUG RESPONSE ANALYSIS
# ============================================================================

def compute_overall_drug_resistance(data):
    """Calculate mean drug response across all drugs"""
    
    # Find drug columns
    drug_cols = [col for col in data.columns 
                 if ('AUC' in col or 'auc' in col) and col not in ['sample_id', 'sample_id_drug']]
    
    if not drug_cols:
        return None
    
    print(f"\nAnalyzing {len(drug_cols)} drugs for overall resistance")
    
    # Calculate mean drug response
    data['mean_drug_response'] = data[drug_cols].mean(axis=1, skipna=True)
    
    # Correlation with persister score
    valid_data = data.dropna(subset=['mean_drug_response', 'persister_probability'])
    
    corr, p_val = stats.spearmanr(
        valid_data['persister_probability'],
        valid_data['mean_drug_response']
    )
    
    print(f"Overall drug resistance correlation:")
    print(f"  Spearman r = {corr:.3f}, p = {p_val:.3e}")
    
    # Group comparison
    if 'persister_group' in data.columns:
        groups = data.groupby('persister_group')['mean_drug_response'].agg(['mean', 'median', 'std'])
        print(f"\nDrug resistance by group:")
        print(groups.round(3))
    
    return {
        'correlation': corr,
        'p_value': p_val,
        'n_samples': len(valid_data),
        'n_drugs': len(drug_cols)
    }

def analyze_key_drugs(data):
    """Focus on key AML drugs"""
    
    key_drugs = {
        'venetoclax': ['Venetoclax', 'venetoclax', 'ABT-199', 'ABT199'],
        'cytarabine': ['Cytarabine', 'cytarabine', 'ARA-C', 'Ara-C'],
        'daunorubicin': ['Daunorubicin', 'daunorubicin', 'DNR'],
        'idarubicin': ['Idarubicin', 'idarubicin', 'IDA'],
        'midostaurin': ['Midostaurin', 'midostaurin', 'PKC412'],
        'gilteritinib': ['Gilteritinib', 'gilteritinib', 'ASP2215'],
        'quizartinib': ['Quizartinib', 'quizartinib', 'AC220']
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("KEY DRUG ANALYSIS")
    print("="*60)
    
    for drug_name, synonyms in key_drugs.items():
        # Find matching columns
        matching_cols = []
        for col in data.columns:
            if any(syn.lower() in col.lower() for syn in synonyms):
                matching_cols.append(col)
        
        if matching_cols:
            for col in matching_cols:
                valid_data = data[['persister_probability', col]].dropna()
                
                if len(valid_data) > 5:
                    corr, p_val = stats.spearmanr(
                        valid_data['persister_probability'],
                        valid_data[col]
                    )
                    
                    results[drug_name] = {
                        'column': col,
                        'correlation': corr,
                        'p_value': p_val,
                        'n_samples': len(valid_data),
                        'significant': p_val < 0.05
                    }
                    
                    sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"{drug_name:15} | r={corr:6.3f} | p={p_val:.3e} {sig_marker} | n={len(valid_data)}")
    
    return results

def adjusted_drug_associations(data, drug_col, covariates=['age', 'sex_male']):
    """Test drug-persister association adjusting for covariates"""
    
    try:
        import statsmodels.api as sm
    except ImportError:
        print("Installing statsmodels...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
        import statsmodels.api as sm
    
    available_covariates = [cov for cov in covariates if cov in data.columns]
    
    if not available_covariates:
        print("No covariates available for adjustment")
        return None
    
    model_data = data[['persister_probability', drug_col] + available_covariates].dropna()
    
    if len(model_data) < 20:
        return None
    
    X = model_data[['persister_probability'] + available_covariates]
    X = sm.add_constant(X)
    y = model_data[drug_col]
    
    model = sm.OLS(y, X).fit()
    
    print(f"\nAdjusted analysis for {drug_col}:")
    print(f"  Persister coefficient: {model.params['persister_probability']:.3f}")
    print(f"  P-value: {model.pvalues['persister_probability']:.3e}")
    print(f"  R-squared: {model.rsquared:.3f}")
    
    return {
        'persister_coef': model.params['persister_probability'],
        'persister_pval': model.pvalues['persister_probability'],
        'rsquared': model.rsquared
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_survival_curves(km_results, lr_result, title, save_path=None):
    """Create publication-quality Kaplan-Meier curves"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#DC143C', '#4169E1', '#FFA500', '#32CD32', '#9370DB']
    
    for i, (group, results) in enumerate(km_results.items()):
        results['kmf'].plot_survival_function(
            ax=ax, 
            ci_show=True,
            color=colors[i % len(colors)],
            linewidth=2.5,
            alpha=0.9
        )
    
    add_at_risk_counts(*[r['kmf'] for r in km_results.values()], ax=ax)
    
    ax.set_xlabel('Time (months)', fontsize=14)
    ax.set_ylabel('Overall Survival Probability', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    p_text = f"Log-rank p = {lr_result.p_value:.3e}" if lr_result.p_value < 0.001 else f"Log-rank p = {lr_result.p_value:.3f}"
    ax.text(0.95, 0.95, p_text, transform=ax.transAxes,
           fontsize=12, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
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
    if 'os_status' in data.columns:
        alive = data[data['os_status'] == 0]['persister_probability']
        dead = data[data['os_status'] == 1]['persister_probability']
        
        ax.hist(alive, bins=20, alpha=0.5, label='Alive/Censored', color='blue')
        ax.hist(dead, bins=20, alpha=0.5, label='Deceased', color='red')
        ax.legend()
    else:
        ax.hist(data['persister_probability'], bins=20, alpha=0.7, color='steelblue')
    
    ax.set_xlabel('Persister Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution by Survival Status', fontsize=14)
    
    # 2. Box plot comparison
    ax = axes[0, 1]
    if 'os_status' in data.columns:
        alive = data[data['os_status'] == 0]['persister_probability']
        dead = data[data['os_status'] == 1]['persister_probability']
        
        bp = ax.boxplot([alive, dead], labels=['Alive/Censored', 'Deceased'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('#4CAF50')
        bp['boxes'][1].set_facecolor('#F44336')
        
        _, p_val = stats.mannwhitneyu(alive, dead)
        ax.text(0.5, 0.95, f'Mann-Whitney p = {p_val:.3f}',
               transform=ax.transAxes, ha='center', fontsize=11)
    else:
        ax.boxplot([data['persister_probability']], labels=['All Samples'])
    
    ax.set_ylabel('Persister Probability', fontsize=12)
    ax.set_title('Persister Score by Outcome', fontsize=14)
    
    # 3. Scatter plot: score vs survival time
    ax = axes[1, 0]
    if 'os_months' in data.columns and 'os_status' in data.columns:
        colors = ['green' if s == 0 else 'red' for s in data['os_status']]
        ax.scatter(data['persister_probability'], data['os_months'],
                  c=colors, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        corr, p = stats.spearmanr(data['persister_probability'], data['os_months'])
        ax.text(0.02, 0.98, f'Spearman ρ = {corr:.3f}\np = {p:.3f}',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white'))
        
        ax.set_ylabel('Survival Time (months)', fontsize=12)
    else:
        ax.hist(data['persister_probability'], bins=20)
        ax.set_ylabel('Frequency', fontsize=12)
    
    ax.set_xlabel('Persister Probability', fontsize=12)
    ax.set_title('Persister Score vs Survival Time', fontsize=14)
    
    # 4. Density plot by group
    ax = axes[1, 1]
    if 'persister_group' in data.columns:
        sns.histplot(
            data=data,
            x='persister_probability',
            hue='persister_group',
            bins=20,
            element='step',
            stat='density',
            common_norm=False,
            ax=ax
        )
        ax.set_xlim(0, 1)
        ax.set_title('Distribution by Group', fontsize=14)
        ax.set_xlabel('Persister Probability', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
    
    plt.suptitle('Persister Score Distribution Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_drug_associations(drug_results, save_path=None):
    """Plot drug-persister associations"""
    
    if not drug_results:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data
    drugs = list(drug_results.keys())
    correlations = [drug_results[d]['correlation'] for d in drugs]
    p_values = [drug_results[d]['p_value'] for d in drugs]
    
    # 1. Correlation barplot
    ax = axes[0, 0]
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    bars = ax.barh(drugs, correlations, color=colors)
    ax.set_xlabel('Spearman Correlation', fontsize=12)
    ax.set_title('Drug-Persister Correlations', fontsize=14)
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    # Add significance markers
    for i, (corr, p) in enumerate(zip(correlations, p_values)):
        if p < 0.001:
            marker = '***'
        elif p < 0.01:
            marker = '**'
        elif p < 0.05:
            marker = '*'
        else:
            marker = ''
        ax.text(corr + 0.02 if corr > 0 else corr - 0.02, i, marker,
               va='center', fontsize=12, fontweight='bold')
    
    # 2. Volcano plot
    ax = axes[0, 1]
    neg_log_p = [-np.log10(p) if p > 0 else 0 for p in p_values]
    
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    ax.scatter(correlations, neg_log_p, c=colors, s=100, alpha=0.6, edgecolors='black')
    
    for drug, corr, nlp, p in zip(drugs, correlations, neg_log_p, p_values):
        if p < 0.05:
            ax.annotate(drug, (corr, nlp), fontsize=9,
                       xytext=(5, 5), textcoords='offset points')
    
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Correlation', fontsize=12)
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    ax.set_title('Statistical Significance', fontsize=14)
    ax.legend()
    
    # 3. Sample sizes
    ax = axes[1, 0]
    sample_sizes = [drug_results[d]['n_samples'] for d in drugs]
    ax.bar(drugs, sample_sizes, color='steelblue', alpha=0.7)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Sample Sizes', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    # 4. Effect size heatmap
    ax = axes[1, 1]
    effect_data = pd.DataFrame({
        'Drug': drugs,
        'Correlation': correlations,
        'Significant': [p < 0.05 for p in p_values]
    })
    
    # Create heatmap data
    heatmap_data = pd.DataFrame([correlations], columns=drugs)
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-0.5, vmax=0.5, ax=ax,
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Drug Response Correlations', fontsize=14)
    
    plt.suptitle('BeatAML Drug-Persister Associations', fontsize=16)
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
# MAIN COMPREHENSIVE ANALYSIS
# ============================================================================

def analyze_beataml_survival_comprehensive():
    """Comprehensive BeatAML survival analysis with drug response"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BEATAML ANALYSIS")
    print("Survival + Drug Response Integration")
    print("="*80)
    
    # Load data
    data, has_survival = load_matched_beataml_data()
    
    # Load drug response
    data_with_drugs, has_drugs = load_drug_response_data(data)
    
    output_dir = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/survival_analysis/beataml")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize results
    results_summary = {
        'n_samples': len(data),
        'has_survival': has_survival,
        'has_drugs': has_drugs,
        'persister_prob_mean': float(data['persister_probability'].mean()),
        'persister_prob_median': float(data['persister_probability'].median())
    }
    
    # ========================================================================
    # PART 1: SURVIVAL ANALYSIS (if available)
    # ========================================================================
    
    if has_survival:
        print("\n" + "="*60)
        print("PART 1: SURVIVAL ANALYSIS")
        print("="*60)
        
        methods = ['median', 'tertile', 'quartile', 'extreme', 'threshold']
        survival_results = {}
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Testing {method.upper()} stratification")
            print('='*60)
            
            try:
                stratified_data = stratify_by_persister_bins(data.copy(), method=method)
                km_results, lr_result = perform_survival_analysis(stratified_data)
                
                survival_results[method] = {
                    'km_results': km_results,
                    'lr_result': lr_result,
                    'p_value': lr_result.p_value
                }
                
                print(f"\n★ Log-rank p-value: {lr_result.p_value:.4f}")
                
                for group, km_data in km_results.items():
                    median_val = km_data['median']
                    median_txt = f"{median_val:.1f}" if (median_val is not None and np.isfinite(median_val)) else "NA"
                    print(f"  {group}: n={km_data['n']}, events={km_data['events']}, median={median_txt} months")
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        # Cox regression
        print(f"\n{'='*60}")
        print("COX REGRESSION (CONTINUOUS SCORE)")
        print('='*60)
        
        try:
            cph = cox_regression_continuous(data)
            print("\nCox regression results:")
            print(cph.summary[['coef', 'exp(coef)', 'p']])
            
            if 'persister_score' in cph.params_:
                results_summary['cox_hr'] = float(np.exp(cph.params_['persister_score']))
                results_summary['cox_p'] = float(cph.summary.loc['persister_score', 'p'])
        except Exception as e:
            print(f"Cox regression failed: {e}")
        
        # Find best stratification
        if survival_results:
            best_method = min(survival_results.items(), key=lambda x: x[1]['p_value'])
            print(f"\n{'='*60}")
            print(f"BEST STRATIFICATION: {best_method[0].upper()}")
            print(f"P-value: {best_method[1]['p_value']:.4f}")
            print('='*60)
            
            # Plot best survival curve
            fig1 = plot_survival_curves(
                best_method[1]['km_results'],
                best_method[1]['lr_result'],
                f'BeatAML: {best_method[0].capitalize()} Stratification (Best)',
                output_dir / f"best_km_{best_method[0]}.png"
            )
            
            # Plot distribution
            best_data = survival_results[best_method[0]].get('data', stratify_by_persister_bins(data.copy(), best_method[0]))
            fig2 = plot_persister_distribution(
                best_data,
                output_dir / "persister_distribution.png"
            )
            
            # Plot comparison
            fig3 = plot_stratification_comparison(
                survival_results,
                output_dir / "stratification_comparison.png"
            )
            
            results_summary['best_survival_method'] = best_method[0]
            results_summary['best_survival_p'] = float(best_method[1]['p_value'])
    else:
        print("\n" + "="*60)
        print("PART 1: SURVIVAL ANALYSIS")
        print("="*60)
        print("No survival data available - skipping survival analysis")
    
    # ========================================================================
    # PART 2: DRUG RESPONSE ANALYSIS
    # ========================================================================
    
    if has_drugs:
        print("\n" + "="*60)
        print("PART 2: DRUG RESPONSE ANALYSIS")
        print("="*60)
        
        # Stratify for drug analysis
        data_for_drugs = stratify_by_persister_bins(data_with_drugs.copy(), method='median')
        
        # Overall drug resistance
        print("\nA. OVERALL DRUG RESISTANCE")
        print("-"*40)
        overall_results = compute_overall_drug_resistance(data_for_drugs)
        if overall_results:
            results_summary['overall_drug_resistance'] = overall_results
        
        # Key drugs
        key_drug_results = analyze_key_drugs(data_for_drugs)
        if key_drug_results:
            results_summary['key_drugs'] = key_drug_results
            
            # Plot drug associations
            fig = plot_drug_associations(
                key_drug_results,
                output_dir / "drug_associations.png"
            )
            
            # Focus on venetoclax with adjustment
            if 'venetoclax' in key_drug_results:
                ven_col = key_drug_results['venetoclax']['column']
                adj_results = adjusted_drug_associations(data_for_drugs, ven_col)
                if adj_results:
                    results_summary['venetoclax_adjusted'] = adj_results
    else:
        print("\n" + "="*60)
        print("PART 2: DRUG RESPONSE ANALYSIS")
        print("="*60)
        print("No drug response data available - skipping drug analysis")
    
    # Save comprehensive summary
    with open(output_dir / "beataml_comprehensive_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    # Print final summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nTotal samples: {results_summary['n_samples']}")
    
    if has_survival and 'best_survival_p' in results_summary:
        print(f"\nSurvival analysis:")
        print(f"  Best method: {results_summary['best_survival_method']}")
        print(f"  P-value: {results_summary['best_survival_p']:.4f}")
        
        if 'cox_p' in results_summary:
            print(f"  Cox HR: {results_summary['cox_hr']:.3f} (p={results_summary['cox_p']:.4f})")
    
    if has_drugs and 'overall_drug_resistance' in results_summary:
        res = results_summary['overall_drug_resistance']
        print(f"\nDrug response:")
        print(f"  Overall correlation: r={res['correlation']:.3f}, p={res['p_value']:.3e}")
        
        if 'key_drugs' in results_summary:
            sig_drugs = [d for d, r in results_summary['key_drugs'].items() if r['significant']]
            print(f"  Significant drugs: {', '.join(sig_drugs)}")
    
    return results_summary

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run comprehensive BeatAML analysis"""
    
    print("\n" + "="*80)
    print("BEATAML PERSISTER ANALYSIS PIPELINE")
    print("Complete Survival + Drug Response Analysis")
    print("="*80)
    
    # Run analysis
    results = analyze_beataml_survival_comprehensive()
    
    # Final interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if results and 'cox_p' in results:
        p = results['cox_p']
        hr = results['cox_hr']
        
        if p < 0.05:
            print(f"✓ Persister score IS prognostic (HR={hr:.3f}, p={p:.4f})")
            if hr > 1:
                print("  Higher persister scores → WORSE survival")
            else:
                print("  Higher persister scores → BETTER survival")
        else:
            print(f"✗ Persister score is NOT significantly prognostic (p={p:.4f})")
    
    if results and 'overall_drug_resistance' in results:
        corr = results['overall_drug_resistance']['correlation']
        p = results['overall_drug_resistance']['p_value']
        
        if p < 0.05:
            print(f"\n✓ Persister score IS associated with drug resistance (r={corr:.3f}, p={p:.3e})")
            if corr > 0:
                print("  Higher persister scores → MORE drug resistance")
            else:
                print("  Higher persister scores → LESS drug resistance")
        else:
            print("✗ Persister score is NOT significantly associated with drug resistance")
    
    print("\nAnalysis complete!")
    print("\nNote: Consider biological context and sample size when interpreting results.")

if __name__ == "__main__":
    main()
