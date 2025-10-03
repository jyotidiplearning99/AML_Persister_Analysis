# compare_distributions.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
b = pd.read_csv("results/bulk_BeatAML/predictions_fixed.csv")
t = pd.read_csv("results/bulk_TCGA/predictions_fixed.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Overlaid histograms
ax = axes[0, 0]
ax.hist(b["persister_probability"], bins=25, alpha=0.5, label='BeatAML', color='blue', density=True)
ax.hist(t["persister_probability"], bins=25, alpha=0.5, label='TCGA', color='red', density=True)
ax.axvline(x=0.31, color='black', linestyle='--', alpha=0.7, label='Threshold')
ax.set_xlabel("Persister Probability")
ax.set_ylabel("Density")
ax.set_title("Distribution Comparison")
ax.legend()

# Panel 2: ECDFs
ax = axes[0, 1]
b_sorted = np.sort(b["persister_probability"])
t_sorted = np.sort(t["persister_probability"])
b_ecdf = np.arange(1, len(b_sorted)+1) / len(b_sorted)
t_ecdf = np.arange(1, len(t_sorted)+1) / len(t_sorted)

ax.plot(b_sorted, b_ecdf, label='BeatAML', linewidth=2)
ax.plot(t_sorted, t_ecdf, label='TCGA', linewidth=2)
ax.axvline(x=0.31, color='gray', linestyle='--', alpha=0.7)
ax.set_xlabel("Persister Probability")
ax.set_ylabel("Empirical CDF")
ax.set_title("Cumulative Distribution Functions")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Box plots
ax = axes[1, 0]
data_combined = pd.DataFrame({
    'Dataset': ['BeatAML']*len(b) + ['TCGA']*len(t),
    'Persister_Probability': list(b['persister_probability']) + list(t['persister_probability'])
})
bp = ax.boxplot([b['persister_probability'], t['persister_probability']], 
                 labels=['BeatAML', 'TCGA'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)
ax.axhline(y=0.31, color='red', linestyle='--', alpha=0.7)
ax.set_ylabel("Persister Probability")
ax.set_title("Box Plot Comparison")
ax.grid(True, axis='y', alpha=0.3)

# Panel 4: Statistics
ax = axes[1, 1]
ax.axis('off')

# Perform statistical tests
ks_stat, ks_p = stats.ks_2samp(b['persister_probability'], t['persister_probability'])
mw_stat, mw_p = stats.mannwhitneyu(b['persister_probability'], t['persister_probability'])

stats_text = f"""
Statistical Comparison
{'='*30}

BeatAML (n={len(b)}):
  Mean: {b['persister_probability'].mean():.3f}
  Median: {b['persister_probability'].median():.3f}
  Std: {b['persister_probability'].std():.3f}
  Persisters: {(b['persister_probability'] >= 0.31).sum()} ({100*(b['persister_probability'] >= 0.31).mean():.1f}%)

TCGA (n={len(t)}):
  Mean: {t['persister_probability'].mean():.3f}
  Median: {t['persister_probability'].median():.3f}
  Std: {t['persister_probability'].std():.3f}
  Persisters: {(t['persister_probability'] >= 0.31).sum()} ({100*(t['persister_probability'] >= 0.31).mean():.1f}%)

Statistical Tests:
  KS test: p={ks_p:.4f}
  Mann-Whitney U: p={mw_p:.4f}
  
Interpretation:
  {'No significant difference' if ks_p > 0.05 else 'Significant difference'} between datasets
"""

ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.suptitle("BeatAML vs TCGA Persister Probability Comparison", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("results/comparison_beataml_tcga.png", dpi=150, bbox_inches='tight')
plt.show()
