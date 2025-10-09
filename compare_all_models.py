#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Load results
df_13k = pd.read_csv('results/inference_39_new_aml_samples/inference_results_summary.csv')
df_1k_031 = pd.read_csv('results/inference_39_reduced_1k_thr031/inference_results_summary.csv')
df_1k_v3 = pd.read_csv('results/inference_39_reduced_1k_v3/inference_results_summary.csv')

# Load metadata (disease stage, risk, etc.)
metadata = pd.read_excel('/scratch/project_2010751/AML_scRNA/Table1.xlsx')
metadata['sample'] = metadata['Sample.id'].str.split('_').str[:2].str.join('_')

# Merge all results
comparison = pd.merge(
    df_13k[['sample', 'persister_pct', 'mean_prob', 'threshold', 'cells']],
    df_1k_031[['sample', 'persister_pct', 'mean_prob', 'threshold']],
    on='sample',
    suffixes=('_13k', '_1k_031')
)
comparison = pd.merge(
    comparison,
    df_1k_v3[['sample', 'persister_pct', 'mean_prob', 'threshold']],
    on='sample',
    suffixes=('', '_1k_v3')
)

# Rename last columns
comparison.rename(columns={
    'persister_pct': 'persister_pct_1k_v3',
    'mean_prob': 'mean_prob_1k_v3',
    'threshold': 'threshold_1k_v3'
}, inplace=True)

# Add metadata
comparison = pd.merge(comparison, metadata, on='sample', how='left')

print("="*80)
print("PERSISTER CELL PREDICTION COMPARISON")
print("="*80)
print(f"\nTotal samples analyzed: {len(comparison)}")
print(f"Total cells analyzed: {comparison['cells'].sum():,}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"{'Model':<25} {'Mean±SD Persister %':<25} {'Threshold':<15} {'Range'}")
print("-"*80)
print(f"13000 genes:             {comparison['persister_pct_13k'].mean():.1f}±{comparison['persister_pct_13k'].std():.1f}%"
      f"{'':10} {comparison['threshold_13k'].iloc[0]:.3f}"
      f"{'':10} {comparison['persister_pct_13k'].min():.1f}-{comparison['persister_pct_13k'].max():.1f}%")
print(f"1000 genes (thr=0.31):   {comparison['persister_pct_1k_031'].mean():.1f}±{comparison['persister_pct_1k_031'].std():.1f}%"
      f"{'':10} {comparison['threshold_1k_031'].iloc[0]:.3f}"
      f"{'':10} {comparison['persister_pct_1k_031'].min():.1f}-{comparison['persister_pct_1k_031'].max():.1f}%")
print(f"1000 genes (v3):         {comparison['persister_pct_1k_v3'].mean():.1f}±{comparison['persister_pct_1k_v3'].std():.1f}%"
      f"{'':10} {comparison['threshold_1k_v3'].iloc[0]:.3f}"
      f"{'':10} {comparison['persister_pct_1k_v3'].min():.1f}-{comparison['persister_pct_1k_v3'].max():.1f}%")

# By disease stage
print("\n" + "="*80)
print("BY DISEASE STAGE")
print("="*80)
stages = comparison.groupby('Disease.stage').agg({
    'persister_pct_13k': 'mean',
    'persister_pct_1k_031': 'mean',
    'persister_pct_1k_v3': 'mean',
    'sample': 'count'
}).round(1)
stages.columns = ['13k genes', '1k (0.31)', '1k (v3)', 'N']
print(stages)

# By risk classification
print("\n" + "="*80)
print("BY ELN2022 RISK CLASSIFICATION")
print("="*80)
risk = comparison.groupby('ELN2022.classification').agg({
    'persister_pct_13k': 'mean',
    'persister_pct_1k_031': 'mean', 
    'persister_pct_1k_v3': 'mean',
    'sample': 'count'
}).round(1)
risk.columns = ['13k genes', '1k (0.31)', '1k (v3)', 'N']
print(risk)

# Sample-by-sample comparison
print("\n" + "="*80)
print("SAMPLE-BY-SAMPLE COMPARISON (sorted by 13k persister %)")
print("="*80)
print(f"{'Sample':<20} {'13k':<10} {'1k(0.31)':<10} {'1k(v3)':<10} {'Disease Stage':<20}")
print("-"*80)
sorted_comp = comparison.sort_values('persister_pct_13k', ascending=False)
for _, row in sorted_comp.head(20).iterrows():
    print(f"{row['sample']:<20} {row['persister_pct_13k']:<10.1f} "
          f"{row['persister_pct_1k_031']:<10.1f} {row['persister_pct_1k_v3']:<10.1f} "
          f"{row['Disease.stage'] if pd.notna(row['Disease.stage']) else 'Unknown':<20}")

# Save comparison
comparison.to_csv('model_comparison_all_39_samples.csv', index=False)
print(f"\nFull comparison saved to: model_comparison_all_39_samples.csv")
