# compare_models.py
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df_13k = pd.read_csv('./results/inference_39_aml_13k/inference_results_summary.csv')
df_1k_935 = pd.read_csv('./results/inference_39_reduced_1k/inference_results_summary.csv')
df_1k_031 = pd.read_csv('./results/inference_39_reduced_1k_thr031/inference_results_summary.csv')

# Merge for comparison
comparison = pd.merge(
    df_13k[['sample', 'persister_pct', 'mean_prob']],
    df_1k_935[['sample', 'persister_pct', 'mean_prob']],
    on='sample',
    suffixes=('_13k', '_1k_935')
)
comparison = pd.merge(
    comparison,
    df_1k_031[['sample', 'persister_pct', 'mean_prob']],
    on='sample'
)
comparison.rename(columns={'persister_pct': 'persister_pct_1k_031', 
                          'mean_prob': 'mean_prob_1k_031'}, inplace=True)

print("Persister % Comparison:")
print("="*80)
print(f"{'Sample':<20} {'13k(0.31)':<12} {'1k(0.935)':<12} {'1k(0.31)':<12}")
print("-"*80)
for _, row in comparison.iterrows():
    print(f"{row['sample']:<20} {row['persister_pct_13k']:<12.1f} "
          f"{row['persister_pct_1k_935']:<12.1f} {row['persister_pct_1k_031']:<12.1f}")

print("\nSummary Statistics:")
print(f"13k (thr=0.31):  Mean={comparison['persister_pct_13k'].mean():.1f}% ± {comparison['persister_pct_13k'].std():.1f}%")
print(f"1k  (thr=0.935): Mean={comparison['persister_pct_1k_935'].mean():.1f}% ± {comparison['persister_pct_1k_935'].std():.1f}%")
print(f"1k  (thr=0.31):  Mean={comparison['persister_pct_1k_031'].mean():.1f}% ± {comparison['persister_pct_1k_031'].std():.1f}%")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot: 13k vs 1k (0.935)
axes[0].scatter(comparison['persister_pct_13k'], comparison['persister_pct_1k_935'], alpha=0.6)
axes[0].plot([0, 100], [0, 100], 'r--', alpha=0.5)
axes[0].set_xlabel('13k genes (thr=0.31)')
axes[0].set_ylabel('1k genes (thr=0.935)')
axes[0].set_title('Original thresholds')

# Scatter plot: 13k vs 1k (0.31)
axes[1].scatter(comparison['persister_pct_13k'], comparison['persister_pct_1k_031'], alpha=0.6)
axes[1].plot([0, 100], [0, 100], 'r--', alpha=0.5)
axes[1].set_xlabel('13k genes (thr=0.31)')
axes[1].set_ylabel('1k genes (thr=0.31)')
axes[1].set_title('Same threshold (0.31)')

plt.tight_layout()
plt.savefig('./model_comparison.png', dpi=150)
print("\nPlot saved as model_comparison.png")
