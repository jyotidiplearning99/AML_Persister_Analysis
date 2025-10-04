#!/usr/bin/env python3
"""
TCGA-LAML Analysis with Survival and Drug Response Integration
Note: TCGA lacks direct drug response data, includes alternative approaches
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
# DATA LOADING - TCGA with clinical response
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
# ALTERNATIVE DRUG RESPONSE APPROACHES FOR TCGA
# ============================================================================

def load_clinical_treatment_response(data):
    """
    Load clinical treatment response data from TCGA if available
    Note: TCGA doesn't have ex vivo drug screening like BeatAML
    """
    
    print("\n" + "="*60)
    print("CLINICAL TREATMENT RESPONSE DATA")
    print("="*60)
    
    # Try to load additional TCGA clinical data with treatment information
    tcga_clinical_path = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/data/tcga_clinical_full.csv")
    
    if tcga_clinical_path.exists():
        clinical_full = pd.read_csv(tcga_clinical_path)
        
        # Look for treatment response columns
        response_cols = [col for col in clinical_full.columns 
                        if any(term in col.lower() for term in 
                        ['response', 'treatment', 'therapy', 'remission', 'relapse'])]
        
        if response_cols:
            print(f"Found clinical response columns: {response_cols}")
            # Merge with main data
            data = pd.merge(data, clinical_full[['patient_id'] + response_cols], 
                          on='patient_id', how='left')
    
    print("\n⚠ Note: TCGA lacks ex vivo drug screening data")
    print("  Unlike BeatAML, TCGA doesn't have AUC/IC50 values")
    print("  Analyzing clinical treatment outcomes instead")
    
    return data

def predict_drug_response_from_beataml_model(data):
    """
    Use BeatAML-trained associations to predict potential drug response
    Based on persister scores
    """
    
    print("\n" + "="*60)
    print("PREDICTED DRUG RESPONSE (from BeatAML model)")
    print("="*60)
    
    # Load BeatAML drug associations if available
    beataml_results_path = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/survival_analysis/beataml/beataml_comprehensive_summary.json")
    
    if beataml_results_path.exists():
        with open(beataml_results_path, 'r') as f:
            beataml_results = json.load(f)
        
        if 'key_drugs' in beataml_results:
            print("\nUsing BeatAML drug-persister associations to predict TCGA response:")
            
            predictions = {}
            for drug, stats in beataml_results['key_drugs'].items():
                if stats.get('significant', False):
                    corr = stats['correlation']
                    # Predict resistance based on persister score
                    if corr > 0:
                        # Higher persister = more resistant
                        data[f'{drug}_predicted_resistance'] = data['persister_probability']
                    else:
                        # Higher persister = more sensitive
                        data[f'{drug}_predicted_resistance'] = 1 - data['persister_probability']
                    
                    predictions[drug] = {
                        'beataml_correlation': corr,
                        'prediction_basis': 'persister_score'
                    }
                    
                    print(f"  {drug}: predicted based on r={corr:.3f} from BeatAML")
            
            return data, predictions
    
    print("  No BeatAML results available for transfer learning")
    return data, {}

def analyze_treatment_outcomes(data):
    """
    Analyze actual clinical treatment outcomes if available
    """
    
    print("\n" + "="*60)
    print("CLINICAL TREATMENT OUTCOME ANALYSIS")
    print("="*60)
    
    # Check for common treatment outcome variables
    outcome_vars = ['complete_remission', 'CR', 'relapse', 'refractory']
    
    results = {}
    for var in outcome_vars:
        if var in data.columns:
            # Compare persister scores by treatment outcome
            responders = data[data[var] == 1]['persister_probability']
            non_responders = data[data[var] == 0]['persister_probability']
            
            if len(responders) > 0 and len(non_responders) > 0:
                _, p_val = stats.mannwhitneyu(responders, non_responders)
                
                results[var] = {
                    'responders_mean': responders.mean(),
                    'non_responders_mean': non_responders.mean(),
                    'p_value': p_val
                }
                
                print(f"\n{var}:")
                print(f"  Responders: mean persister = {responders.mean():.3f}")
                print(f"  Non-responders: mean persister = {non_responders.mean():.3f}")
                print(f"  P-value: {p_val:.3f}")
    
    if not results:
        print("  No treatment outcome data available in TCGA")
    
    return results

# ============================================================================
# STRATIFICATION (Same as before)
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
# SURVIVAL ANALYSIS FUNCTIONS (Same as before)
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
            
            for group, km_data in km_results.items():
                km_data['kmf'].plot_survival_function(ax=ax, ci_show=False)
            
            p_text = f"p={lr_result.p_value:.3f}" if lr_result.p_value >= 0.001 else f"p={lr_result.p_value:.3e}"
            ax.text(0.95, 0.95, p_text, transform=ax.transAxes,
                   fontsize=11, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{method.capitalize()} Stratification', fontsize=13)
            ax.set_xlabel('Time (months)', fontsize=11)
            ax.set_ylabel('Survival Probability', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='lower left')
    
    for i in range(n_methods, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Stratification Method Comparison', fontsize=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_drug_response_note(save_path=None):
    """Create informative plot about drug response data availability"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Data availability comparison
    ax = axes[0]
    datasets = ['TCGA-LAML', 'BeatAML']
    survival_data = [1, 1]
    drug_data = [0, 1]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax.bar(x - width/2, survival_data, width, label='Survival Data', color='steelblue')
    ax.bar(x + width/2, drug_data, width, label='Drug Response Data', color='coral')
    
    ax.set_ylabel('Data Available', fontsize=12)
    ax.set_title('Data Availability Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right panel: Analysis strategy
    ax = axes[1]
    ax.axis('off')
    
    strategy_text = """
    TCGA-LAML Drug Response Analysis Strategy:
    
    1. Direct drug response data: NOT AVAILABLE
       • TCGA lacks ex vivo drug screening (AUC/IC50)
       • No systematic drug sensitivity profiling
    
    2. Alternative approaches used:
       • Clinical treatment outcomes (if available)
       • Predicted response from BeatAML associations
       • Transfer learning from BeatAML persister-drug correlations
    
    3. Recommendation:
       • Use BeatAML for drug-persister associations
       • Apply findings to TCGA for validation of survival
       • Consider external databases (GDSC, PRISM) for cell lines
    """
    
    ax.text(0.1, 0.9, strategy_text, transform=ax.transAxes,
           fontsize=11, va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('TCGA Drug Response Analysis Limitations', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# ============================================================================
# MAIN COMPREHENSIVE ANALYSIS
# ============================================================================

def analyze_tcga_comprehensive():
    """Comprehensive TCGA analysis with survival and drug response notes"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TCGA-LAML ANALYSIS")
    print("Survival Analysis + Drug Response Considerations")
    print("="*80)
    
    # Load data
    data = load_matched_tcga_data()
    
    output_dir = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/survival_analysis/tcga_comprehensive")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize results
    results = {}
    
    # ========================================================================
    # PART 1: SURVIVAL ANALYSIS
    # ========================================================================
    
    print("\n" + "="*60)
    print("PART 1: SURVIVAL ANALYSIS")
    print("="*60)
    
    methods = ['median', 'tertile', 'quartile', 'extreme', 'threshold']
    survival_results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing {method.upper()} stratification")
        print('='*60)
        
        stratified_data = stratify_by_persister_bins(data.copy(), method=method)
        km_results, lr_result = perform_survival_analysis(stratified_data)
        
        survival_results[method] = {
            'km_results': km_results,
            'lr_result': lr_result,
            'data': stratified_data,
            'p_value': lr_result.p_value
        }
        
        print(f"\n★ Log-rank p-value: {lr_result.p_value:.4f}")
        
        for group, km_data in km_results.items():
            median_val = km_data['median']
            median_txt = f"{median_val:.1f}" if (median_val is not None and np.isfinite(median_val)) else "NA"
            print(f"  {group}: n={km_data['n']}, events={km_data['events']}, median={median_txt} months")
    
    # Cox regression
    print(f"\n{'='*60}")
    print("COX REGRESSION (CONTINUOUS SCORE)")
    print('='*60)
    
    cph = cox_regression_continuous(data)
    print("\nCox regression results:")
    print(cph.summary[['coef', 'exp(coef)', 'p']])
    
    # Find best stratification
    best_method = min(survival_results.items(), key=lambda x: x[1]['p_value'])
    print(f"\n{'='*60}")
    print(f"BEST STRATIFICATION: {best_method[0].upper()}")
    print(f"P-value: {best_method[1]['p_value']:.4f}")
    print('='*60)
    
    # Create survival visualizations
    fig1 = plot_survival_curves(
        best_method[1]['km_results'],
        best_method[1]['lr_result'],
        f'TCGA-LAML: {best_method[0].capitalize()} Stratification (Best)',
        output_dir / f"best_km_{best_method[0]}.png"
    )
    
    best_data = survival_results[best_method[0]]['data']
    fig2 = plot_persister_distribution(
        best_data,
        output_dir / "persister_distribution.png"
    )
    
    fig3 = plot_stratification_comparison(
        survival_results,
        output_dir / "stratification_comparison.png"
    )
    
    # ========================================================================
    # PART 2: DRUG RESPONSE ANALYSIS (Limited for TCGA)
    # ========================================================================
    
    print("\n" + "="*60)
    print("PART 2: DRUG RESPONSE ANALYSIS")
    print("="*60)
    
    # Check for clinical treatment response
    data_with_treatment = load_clinical_treatment_response(data)
    treatment_outcomes = analyze_treatment_outcomes(data_with_treatment)
    
    # Predict drug response based on BeatAML model
    data_with_predictions, drug_predictions = predict_drug_response_from_beataml_model(data)
    
    # Create drug response note plot
    fig4 = plot_drug_response_note(output_dir / "drug_response_note.png")
    
    # ========================================================================
    # SAVE COMPREHENSIVE SUMMARY
    # ========================================================================
    
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
            for method, res in survival_results.items()
        },
        'best_method': best_method[0],
        'best_p_value': float(best_method[1]['p_value']),
        'cox_persister_hr': float(np.exp(cph.params_['persister_score'])) if 'persister_score' in cph.params_ else None,
        'cox_persister_p': float(cph.summary.loc['persister_score', 'p']) if 'persister_score' in cph.params_ else None,
        'drug_response_note': 'TCGA lacks ex vivo drug response data. See BeatAML for drug-persister associations.',
        'treatment_outcomes': treatment_outcomes,
        'predicted_drug_response': drug_predictions
    }
    
    with open(output_dir / "tcga_comprehensive_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    # Print final summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print("\nSURVIVAL ANALYSIS:")
    print("Method         | P-value | Groups")
    print("-"*40)
    for method, res in survival_results.items():
        n_groups = len(res['km_results'])
        print(f"{method:12} | {res['p_value']:.4f} | {n_groups}")
    
    print("\nDRUG RESPONSE:")
    print("  ⚠ TCGA lacks ex vivo drug response data (AUC/IC50)")
    print("  ✓ See BeatAML analysis for drug-persister associations")
    
    if treatment_outcomes:
        print("\nClinical Treatment Outcomes:")
        for outcome, stats in treatment_outcomes.items():
            print(f"  {outcome}: p={stats['p_value']:.3f}")
    
    if drug_predictions:
        print("\nPredicted Drug Response (from BeatAML model):")
        for drug in list(drug_predictions.keys())[:3]:
            print(f"  {drug}: transferred from BeatAML associations")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return summary, cph

# ============================================================================
# COMPARE WITH BEATAML RESULTS
# ============================================================================

def compare_with_beataml():
    """Compare TCGA survival results with BeatAML"""
    
    print("\n" + "="*80)
    print("CROSS-DATASET COMPARISON: TCGA vs BeatAML")
    print("="*80)
    
    # Load both summaries if available
    tcga_summary_path = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/survival_analysis/tcga_comprehensive/tcga_comprehensive_summary.json")
    beataml_summary_path = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/survival_analysis/beataml/beataml_comprehensive_summary.json")
    
    comparison = {}
    
    if tcga_summary_path.exists():
        with open(tcga_summary_path, 'r') as f:
            tcga_summary = json.load(f)
            comparison['tcga'] = {
                'n_samples': tcga_summary.get('n_samples'),
                'best_survival_p': tcga_summary.get('best_p_value'),
                'cox_hr': tcga_summary.get('cox_persister_hr'),
                'has_drug_data': False
            }
    
    if beataml_summary_path.exists():
        with open(beataml_summary_path, 'r') as f:
            beataml_summary = json.load(f)
            comparison['beataml'] = {
                'n_samples': beataml_summary.get('n_samples'),
                'best_survival_p': beataml_summary.get('best_survival_p'),
                'cox_hr': beataml_summary.get('cox_hr'),
                'has_drug_data': 'overall_drug_resistance' in beataml_summary
            }
            
            if 'overall_drug_resistance' in beataml_summary:
                comparison['beataml']['drug_correlation'] = beataml_summary['overall_drug_resistance'].get('correlation')
    
    print("\nDataset Comparison:")
    print("-"*60)
    print("Feature              | TCGA-LAML      | BeatAML")
    print("-"*60)
    
    if 'tcga' in comparison and 'beataml' in comparison:
        print(f"Samples              | {comparison['tcga']['n_samples']:<14} | {comparison['beataml']['n_samples']}")
        print(f"Survival p-value     | {comparison['tcga']['best_survival_p']:<14.4f} | {comparison['beataml'].get('best_survival_p', 'N/A')}")
        print(f"Cox HR               | {comparison['tcga'].get('cox_hr', 'N/A'):<14.3f} | {comparison['beataml'].get('cox_hr', 'N/A')}")
        print(f"Drug response data   | {'No':<14} | {'Yes' if comparison['beataml']['has_drug_data'] else 'No'}")
        
        if comparison['beataml'].get('drug_correlation'):
            print(f"Drug-persister corr  | {'N/A':<14} | {comparison['beataml']['drug_correlation']:.3f}")
    
    return comparison

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run comprehensive TCGA analysis"""
    
    print("\n" + "="*80)
    print("TCGA-LAML PERSISTER ANALYSIS PIPELINE")
    print("Complete Survival Analysis + Drug Response Considerations")
    print("="*80)
    
    # Run analysis
    results, cph = analyze_tcga_comprehensive()
    
    # Compare with BeatAML
    comparison = compare_with_beataml()
    
    # Final interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    # Check if persister score is prognostic
    if 'persister_score' in cph.params_:
        hr = np.exp(cph.params_['persister_score'])
        p = cph.summary.loc['persister_score', 'p']
        
        if p < 0.05:
            print(f"✓ Persister score IS prognostic in TCGA (HR={hr:.3f}, p={p:.4f})")
            if hr > 1:
                print("  Higher persister scores → WORSE survival")
            else:
                print("  Higher persister scores → BETTER survival")
        else:
            print(f"✗ Persister score is NOT significantly prognostic in TCGA (p={p:.4f})")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("\n1. SURVIVAL: Analyzed with multiple stratifications")
    print("2. DRUG RESPONSE: TCGA lacks ex vivo data - use BeatAML for drug associations")
    print("3. RECOMMENDATION: Combine TCGA survival validation with BeatAML drug insights")
    print("\nNote: For drug-persister associations, refer to BeatAML analysis")

if __name__ == "__main__":
    main()
