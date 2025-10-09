#!/usr/bin/env python3
"""
ELN/MRD Enrichment Analysis - Fixed for Real Data Only
Handles duplicate columns and missing clinical data gracefully
Author: Jyotidip Barman
Date: October 2025
"""

import os
import sys
import json
import warnings
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class EnrichmentAnalyzer:
    """Performs ELN and MRD enrichment analysis with real data only"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir = self.output_dir / 'tables'
        self.tables_dir.mkdir(exist_ok=True)
        
    def analyze_cohort(
        self,
        predictions_file: Path,
        clinical_file: Optional[Path],
        cohort_name: str,
        tau: float = 0.31
    ) -> Dict:
        """Analyze one cohort for ELN/MRD enrichment"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {cohort_name}")
        logger.info(f"{'='*60}")
        
        # Load predictions
        try:
            pred_df = pd.read_csv(predictions_file)
            logger.info(f"Loaded predictions: {len(pred_df)} samples from {predictions_file.name}")
        except Exception as e:
            logger.error(f"Failed to load predictions: {e}")
            return {}
            
        # Find probability column in predictions
        prob_col = None
        for col in pred_df.columns:
            if 'prob' in col.lower() or 'persister' in col.lower():
                if 'non' not in col.lower():  # Exclude non-persister columns
                    prob_col = col
                    break
                    
        if not prob_col:
            logger.error("Could not find persister probability column in predictions")
            return {}
            
        logger.info(f"Using probability column: '{prob_col}'")
        probs = pred_df[prob_col].values
        
        # Calculate prevalence
        prevalence = (probs >= tau).mean()
        logger.info(f"Persister prevalence at τ={tau}: {prevalence:.1%} ({(probs >= tau).sum()}/{len(probs)})")
        
        # Create tertiles
        tertiles = self._create_tertiles(probs)
        
        # Initialize results
        results = {
            'cohort': cohort_name,
            'n': len(pred_df),
            'prevalence': prevalence,
            'tau': tau,
            'predictions_file': str(predictions_file)
        }
        
        # Try to load and merge clinical data if provided
        merged_df = pred_df.copy()
        
        if clinical_file and clinical_file.exists():
            try:
                clin_df = pd.read_csv(clinical_file)
                logger.info(f"Loading clinical data: {len(clin_df)} samples from {clinical_file.name}")
                logger.info(f"Clinical columns: {list(clin_df.columns)}")
                
                # Find ID column for merging
                id_cols = ['sample_id', 'sample', 'Sample_ID', 'patient_id', 'submitter_id']
                merge_col = None
                
                for col in id_cols:
                    if col in pred_df.columns and col in clin_df.columns:
                        merge_col = col
                        break
                        
                if merge_col:
                    # Remove duplicate columns before merge (except merge key)
                    overlap_cols = set(pred_df.columns) & set(clin_df.columns)
                    overlap_cols.discard(merge_col)
                    
                    if overlap_cols:
                        logger.info(f"Dropping duplicate columns from clinical: {overlap_cols}")
                        clin_df = clin_df.drop(columns=list(overlap_cols))
                    
                    # Now merge
                    merged_df = pred_df.merge(clin_df, on=merge_col, how='left')
                    logger.info(f"Merged on '{merge_col}': {len(merged_df)} samples")
                    logger.info(f"Merged columns: {list(merged_df.columns)}")
                else:
                    logger.warning("No common ID column found for merging")
                    
            except Exception as e:
                logger.warning(f"Could not merge clinical data: {e}")
        
        # Check for clinical columns and analyze if present
        self._check_and_analyze_clinical(merged_df, probs, tertiles, results, cohort_name)
        
        # Create basic visualizations
        self._create_basic_visualizations(probs, tertiles, cohort_name, tau, results)
        
        # Save merged data
        merged_df.to_csv(
            self.output_dir / f'{cohort_name}_data.csv',
            index=False
        )
        
        return results
        
    def _create_tertiles(self, probs: np.ndarray) -> pd.Series:
        """Create tertile groups"""
        thresholds = np.percentile(probs, [33.33, 66.67])
        tertiles = pd.cut(
            probs,
            bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
            labels=['Low', 'Intermediate', 'High']
        )
        logger.info(f"Tertile thresholds: {thresholds[0]:.3f}, {thresholds[1]:.3f}")
        logger.info(f"Tertile distribution: {tertiles.value_counts().sort_index().to_dict()}")
        return tertiles
        
    def _check_and_analyze_clinical(
        self,
        df: pd.DataFrame,
        probs: np.ndarray,
        tertiles: pd.Series,
        results: Dict,
        cohort_name: str
    ):
        """Check for clinical columns and analyze if present"""
        
        # Check for ELN risk columns
        eln_columns = [
            'eln_risk_2017', 'eln_risk_2022', 'eln_risk', 'ELN_risk', 
            'cytogenetic_risk', 'risk_group', 'risk_category'
        ]
        
        eln_col = None
        for col in eln_columns:
            if col in df.columns:
                eln_col = col
                break
                
        if eln_col:
            logger.info(f"Found ELN risk column: '{eln_col}'")
            results['eln_column'] = eln_col
            # Analyze ELN enrichment
            self._analyze_eln_enrichment(tertiles, df[eln_col], results, cohort_name)
        else:
            logger.info("No ELN risk column found - skipping ELN enrichment analysis")
            results['eln_column'] = None
            
        # Check for MRD columns
        mrd_columns = [
            'mrd_status', 'MRD_status', 'mrd', 'MRD', 'MRD_binary',
            'minimal_residual_disease', 'measurable_residual_disease'
        ]
        
        mrd_col = None
        for col in mrd_columns:
            if col in df.columns:
                mrd_col = col
                break
                
        if mrd_col:
            logger.info(f"Found MRD column: '{mrd_col}'")
            results['mrd_column'] = mrd_col
            # Analyze MRD association
            self._analyze_mrd_association(probs, df[mrd_col], tertiles, results, cohort_name)
        else:
            logger.info("No MRD column found - skipping MRD association analysis")
            results['mrd_column'] = None
            
    def _analyze_eln_enrichment(
        self,
        tertiles: pd.Series,
        eln_risk: pd.Series,
        results: Dict,
        cohort_name: str
    ):
        """Analyze ELN risk enrichment if data available"""
        
        # Standardize ELN categories
        eln_std = self._standardize_eln(eln_risk)
        
        # Remove missing values
        valid_mask = eln_std.notna()
        tertiles_clean = tertiles[valid_mask]
        eln_clean = eln_std[valid_mask]
        
        n_valid = len(eln_clean)
        logger.info(f"ELN analysis: {n_valid} samples with valid data")
        
        if n_valid < 30:
            logger.warning(f"Insufficient data for robust ELN analysis (n={n_valid})")
            results['eln_enrichment'] = {'n': n_valid, 'insufficient_data': True}
            return
            
        # Create contingency table
        contingency = pd.crosstab(tertiles_clean, eln_clean)
        logger.info(f"Contingency table:\n{contingency}")
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Cramer's V
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        logger.info(f"Chi-square: χ²={chi2:.2f}, p={p_value:.3e}, Cramer's V={cramers_v:.3f}")
        
        results['eln_enrichment'] = {
            'n': n_valid,
            'contingency_table': contingency.to_dict(),
            'chi2': chi2,
            'chi2_p_value': p_value,
            'cramers_v': cramers_v
        }
        
    def _analyze_mrd_association(
        self,
        probs: np.ndarray,
        mrd_status: pd.Series,
        tertiles: pd.Series,
        results: Dict,
        cohort_name: str
    ):
        """Analyze MRD association if data available"""
        
        # Standardize MRD status
        mrd_std = self._standardize_mrd(mrd_status)
        
        # Clean data
        valid_mask = mrd_std.notna()
        mrd_clean = mrd_std[valid_mask]
        probs_clean = probs[valid_mask]
        
        n_valid = len(mrd_clean)
        logger.info(f"MRD analysis: {n_valid} samples with valid data")
        
        if n_valid < 20:
            logger.warning(f"Insufficient data for MRD analysis (n={n_valid})")
            results['mrd_association'] = {'n': n_valid, 'insufficient_data': True}
            return
            
        # Separate MRD+ and MRD-
        mrd_pos = probs_clean[mrd_clean == 'MRD+']
        mrd_neg = probs_clean[mrd_clean == 'MRD-']
        
        n_pos = len(mrd_pos)
        n_neg = len(mrd_neg)
        
        logger.info(f"MRD+: n={n_pos}, MRD-: n={n_neg}")
        
        if n_pos < 5 or n_neg < 5:
            logger.warning(f"Too few samples in one MRD group")
            results['mrd_association'] = {'n_pos': n_pos, 'n_neg': n_neg, 'insufficient_data': True}
            return
            
        # Mann-Whitney U test
        statistic, p_value = mannwhitneyu(mrd_pos, mrd_neg, alternative='two-sided')
        
        # Effect size
        effect_size = 1 - (2*statistic) / (n_pos * n_neg)
        
        logger.info(f"Mann-Whitney U: p={p_value:.3e}, effect size={effect_size:.3f}")
        logger.info(f"Medians: MRD+={np.median(mrd_pos):.3f}, MRD-={np.median(mrd_neg):.3f}")
        
        results['mrd_association'] = {
            'n_mrd_pos': n_pos,
            'n_mrd_neg': n_neg,
            'mann_whitney_u': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'median_mrd_pos': np.median(mrd_pos),
            'median_mrd_neg': np.median(mrd_neg)
        }
        
    def _standardize_eln(self, eln_series: pd.Series) -> pd.Series:
        """Standardize ELN risk categories"""
        eln_lower = eln_series.astype(str).str.lower().str.strip()
        
        standardized = pd.Series(index=eln_series.index, dtype='object')
        
        # Map to standard categories
        standardized[eln_lower.str.contains('favorable|favourable|good|low', na=False)] = 'Favorable'
        standardized[eln_lower.str.contains('intermediate|int|medium', na=False)] = 'Intermediate'
        standardized[eln_lower.str.contains('adverse|poor|high|unfavorable', na=False)] = 'Adverse'
        standardized[eln_series.isna()] = np.nan
        
        return standardized
        
    def _standardize_mrd(self, mrd_series: pd.Series) -> pd.Series:
        """Standardize MRD status"""
        mrd_lower = mrd_series.astype(str).str.lower().str.strip()
        
        standardized = pd.Series(index=mrd_series.index, dtype='object')
        
        # Positive MRD
        standardized[mrd_lower.str.contains('positive|pos|\\+|1|true|yes|detected|present', na=False, regex=True)] = 'MRD+'
        # Negative MRD
        standardized[mrd_lower.str.contains('negative|neg|\\-|0|false|no|undetected|absent', na=False, regex=True)] = 'MRD-'
        standardized[mrd_series.isna()] = np.nan
        
        return standardized
        
    def _create_basic_visualizations(
        self,
        probs: np.ndarray,
        tertiles: pd.Series,
        cohort_name: str,
        tau: float,
        results: Dict
    ):
        """Create basic visualizations"""
        
        n_plots = 2  # Basic plots always created
        
        # Add plots if clinical data available
        if results.get('eln_enrichment') and not results['eln_enrichment'].get('insufficient_data'):
            n_plots += 1
        if results.get('mrd_association') and not results['mrd_association'].get('insufficient_data'):
            n_plots += 1
            
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
            
        # 1. Distribution
        ax = axes[0]
        ax.hist(probs, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(tau, color='red', linestyle='--', label=f'τ={tau}')
        ax.set_xlabel('Persister Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{cohort_name}\nPersister Distribution')
        ax.legend()
        
        # 2. Tertile counts
        ax = axes[1]
        tertile_counts = tertiles.value_counts().sort_index()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = ax.bar(range(len(tertile_counts)), tertile_counts.values, color=colors)
        ax.set_xticks(range(len(tertile_counts)))
        ax.set_xticklabels(tertile_counts.index)
        ax.set_xlabel('Tertile')
        ax.set_ylabel('Count')
        ax.set_title('Tertile Distribution')
        for bar, count in zip(bars, tertile_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom')
        
        plot_idx = 2
        
        # 3. ELN plot if available
        if results.get('eln_enrichment') and not results['eln_enrichment'].get('insufficient_data'):
            ax = axes[plot_idx]
            cont_table = pd.DataFrame(results['eln_enrichment']['contingency_table'])
            cont_table_norm = cont_table.div(cont_table.sum(axis=1), axis=0)
            cont_table_norm.plot(kind='bar', stacked=True, ax=ax,
                                 color=['#2ecc71', '#f39c12', '#e74c3c'])
            ax.set_xlabel('Persister Tertile')
            ax.set_ylabel('Proportion')
            ax.set_title(f'ELN Risk by Tertile\n(p={results["eln_enrichment"]["chi2_p_value"]:.3e})')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.legend(title='ELN Risk')
            plot_idx += 1
            
        # 4. MRD plot if available
        if results.get('mrd_association') and not results['mrd_association'].get('insufficient_data'):
            ax = axes[plot_idx]
            mrd = results['mrd_association']
            # Create simple comparison
            ax.bar(['MRD-', 'MRD+'], 
                  [mrd['median_mrd_neg'], mrd['median_mrd_pos']],
                  color=['lightblue', 'lightcoral'])
            ax.set_ylabel('Median Persister Probability')
            ax.set_title(f'MRD Association\n(p={mrd["p_value"]:.3e})')
            
        plt.tight_layout()
        fig_path = self.figures_dir / f'{cohort_name}_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved figure to: {fig_path}")

def main():
    parser = argparse.ArgumentParser(
        description='ELN/MRD Enrichment Analysis Using Real Data Only'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Path to results directory'
    )
    parser.add_argument(
        '--tau',
        type=float,
        default=0.31,
        help='Persister probability threshold (default: 0.31)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/clinical_enrichment_real',
        help='Output directory for results'
    )
    parser.add_argument(
        '--add-clinical',
        type=str,
        help='Path to additional clinical data CSV with ELN/MRD columns'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("ELN/MRD ENRICHMENT ANALYSIS - REAL DATA ONLY")
    logger.info("="*70)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Threshold τ: {args.tau}")
    
    # Initialize analyzer
    analyzer = EnrichmentAnalyzer(output_dir)
    
    # Define cohorts

    cohorts = {
    'BeatAML': {
        'predictions': results_dir / 'bulk_BeatAML' / 'predictions_final.csv',
        'clinical':   (results_dir / 'bulk_BeatAML' / 'clinical_real.csv'
                       if (results_dir / 'bulk_BeatAML' / 'clinical_real.csv').exists()
                       else results_dir / 'bulk_BeatAML' / 'clinical_predictions.csv')
    },
    'TCGA-LAML': {
        'predictions': results_dir / 'bulk_TCGA' / 'predictions_final.csv',
        'clinical':   (results_dir / 'bulk_TCGA' / 'clinical_real.csv'
                       if (results_dir / 'bulk_TCGA' / 'clinical_real.csv').exists()
                       else results_dir / 'bulk_TCGA' / 'clinical_predictions.csv')
    }
}
    
    # Override with additional clinical data if provided
    if args.add_clinical:
        clinical_path = Path(args.add_clinical)
        if clinical_path.exists():
            logger.info(f"Using additional clinical data from: {clinical_path}")
            for cohort_info in cohorts.values():
                cohort_info['clinical'] = clinical_path
                
    all_results = {}
    
    # Process each cohort
    for cohort_name, paths in cohorts.items():
        
        if not paths['predictions'].exists():
            logger.warning(f"Predictions not found for {cohort_name}: {paths['predictions']}")
            continue
            
        # Clinical file is optional
        clinical_file = paths.get('clinical')
        if clinical_file and not clinical_file.exists():
            logger.warning(f"Clinical file not found for {cohort_name}: {clinical_file}")
            clinical_file = None
            
        # Analyze cohort
        try:
            results = analyzer.analyze_cohort(
                paths['predictions'],
                clinical_file,
                cohort_name,
                args.tau
            )
            
            if results:
                all_results[cohort_name] = results
                
        except Exception as e:
            logger.error(f"Failed to analyze {cohort_name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Generate summary report
    if all_results:
        # Create summary
        summary_rows = []
        for cohort, results in all_results.items():
            row = {
                'Cohort': cohort,
                'N': results.get('n', 0),
                'Prevalence': f"{results.get('prevalence', 0)*100:.1f}%",
                'Tau': results.get('tau', args.tau),
                'ELN_Available': 'Yes' if results.get('eln_column') else 'No',
                'MRD_Available': 'Yes' if results.get('mrd_column') else 'No'
            }
            
            # Add ELN results if available
            if results.get('eln_enrichment') and not results['eln_enrichment'].get('insufficient_data'):
                eln = results['eln_enrichment']
                row['ELN_N'] = eln.get('n', 0)
                row['ELN_p'] = f"{eln.get('chi2_p_value', np.nan):.3e}"
                row['Cramers_V'] = f"{eln.get('cramers_v', np.nan):.3f}"
            
            # Add MRD results if available
            if results.get('mrd_association') and not results['mrd_association'].get('insufficient_data'):
                mrd = results['mrd_association']
                row['MRD_N'] = mrd.get('n_mrd_pos', 0) + mrd.get('n_mrd_neg', 0)
                row['MRD_p'] = f"{mrd.get('p_value', np.nan):.3e}"
                row['MRD_Effect'] = f"{mrd.get('effect_size', np.nan):.3f}"
                    
            summary_rows.append(row)
            
        summary_df = pd.DataFrame(summary_rows)
        
        # Save summary
        tables_dir = analyzer.tables_dir
        summary_df.to_csv(tables_dir / 'enrichment_summary.csv', index=False)
        logger.info(f"Saved summary table to: {tables_dir / 'enrichment_summary.csv'}")
        
        # Save full results as JSON
        with open(output_dir / 'enrichment_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
            
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*70)
        
        print(f"\n{summary_df.to_string()}\n")
        
        # Instructions for adding clinical data
        if not any(r.get('eln_column') or r.get('mrd_column') for r in all_results.values()):
            logger.info("\n" + "="*70)
            logger.info("HOW TO ADD CLINICAL DATA")
            logger.info("="*70)
            logger.info("""
To perform ELN/MRD enrichment analysis, you need to add clinical data:

1. Create a CSV file with the following columns:
   - sample_id (matching your predictions)
   - eln_risk_2017 (values: Favorable, Intermediate, Adverse)
   - mrd_status (values: MRD+, MRD-)

2. Run the script again with --add-clinical flag:
   python script.py --results-dir /path/to/results \\
                    --add-clinical /path/to/clinical_data.csv

3. Or update the existing clinical_predictions.csv files in:
   - bulk_BeatAML/clinical_predictions.csv
   - bulk_TCGA/clinical_predictions.csv
   
Example clinical data format:
sample_id,eln_risk_2017,mrd_status
SAMPLE_001,Adverse,MRD+
SAMPLE_002,Favorable,MRD-
SAMPLE_003,Intermediate,MRD+
...
""")
            
        logger.info(f"\n{'='*70}")
        logger.info(f"All outputs saved to: {output_dir}")
        
    else:
        logger.warning("No results generated")

if __name__ == "__main__":
    main()
