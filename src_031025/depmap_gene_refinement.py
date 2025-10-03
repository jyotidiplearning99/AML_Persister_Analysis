#!/usr/bin/env python3
"""
Optimized DepMap Integration for Gene Refinement v2.0
Fixed: proper dependency scoring, efficient matching, JSON serialization
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Paths
DEPMAP_DIR = Path("/scratch/project_2010751/DepMap_Datasets")
GENES_FILE = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt")
OUTPUT_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/depmap_refined")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("OPTIMIZED DEPMAP GENE REFINEMENT PIPELINE v2.0")
print("="*80)

# 1. Load your 1000 genes
with open(GENES_FILE) as f:
    your_genes = [line.strip().upper() for line in f if line.strip()]
print(f"Loaded {len(your_genes)} persister genes")

# 2. Load metadata first and identify AML ModelIDs
print("\nLoading metadata...")
metadata = pd.read_csv(DEPMAP_DIR / "Model.csv")

# Use ModelID column properly
if 'ModelID' not in metadata.columns:
    print("ERROR: No ModelID column in Model.csv")
    exit(1)

# Set ModelID as index for easy lookup
metadata = metadata.set_index('ModelID')

# Filter for AML using multiple columns
aml_mask = pd.Series([False] * len(metadata), index=metadata.index)

# Check all relevant columns
for col in ['OncotreePrimaryDisease', 'OncotreeLineage', 'OncotreeSubtype']:
    if col in metadata.columns:
        aml_mask |= metadata[col].astype(str).str.contains(
            'Acute Myeloid|AML|Myeloid', na=False, case=False
        )

aml_metadata = metadata[aml_mask]
aml_model_ids = aml_metadata.index.tolist()
print(f"Found {len(aml_model_ids)} AML cell lines in metadata")

# 3. Load CRISPR data and filter to AML lines that exist
print("Loading CRISPR effect data...")
crispr_effect = pd.read_csv(DEPMAP_DIR / "CRISPRGeneEffect.csv", index_col=0)

# Keep only AML lines that exist in CRISPR data
aml_model_ids = [mid for mid in aml_model_ids if mid in crispr_effect.index]
print(f"Filtered to {len(aml_model_ids)} AML lines with CRISPR data")

if not aml_model_ids:
    print("ERROR: No AML lines found with CRISPR data")
    exit(1)

# Get AML subset - use float32 for memory efficiency
aml_deps = crispr_effect.loc[aml_model_ids].astype(np.float32)

# 4. Build efficient gene mapping ONCE
print("\nBuilding gene map...")
# DepMap columns are "SYMBOL (ENTREZ)"
symbol_to_columns = {}  # Maps gene symbol -> list of DepMap columns

for col in aml_deps.columns:
    if ' (' in col:
        symbol = col.split(' (')[0].upper()
    else:
        symbol = col.upper()
    
    if symbol not in symbol_to_columns:
        symbol_to_columns[symbol] = []
    symbol_to_columns[symbol].append(col)

# Match your genes
matched_genes = []
unmatched_genes = []

for gene in your_genes:
    if gene in symbol_to_columns:
        matched_genes.append(gene)
    else:
        unmatched_genes.append(gene)

print(f"Matched {len(matched_genes)}/{len(your_genes)} genes")
if unmatched_genes[:10]:
    print(f"First 10 unmatched: {unmatched_genes[:10]}")

# 5. Calculate dependency metrics (DO NOT NEGATE)
print("\nCalculating AML dependency metrics...")
results = []

for gene in matched_genes:
    cols = symbol_to_columns[gene]
    
    # Handle duplicate columns by averaging
    if len(cols) > 1:
        deps = aml_deps[cols].mean(axis=1)
        print(f"  {gene}: averaged {len(cols)} duplicate columns")
    else:
        deps = aml_deps[cols[0]]
    
    # Remove NaNs
    deps = deps.dropna()
    
    if len(deps) == 0:
        continue
    
    # CRISPR effect: more negative = more essential (KEEP NEGATIVE)
    median_effect = float(deps.median())
    mean_effect = float(deps.mean())
    
    # Dependency fractions at different cutoffs
    frac_dep_lt_minus05 = float((deps < -0.5).mean())  # Strong dependency
    frac_dep_lt_minus03 = float((deps < -0.3).mean())  # Moderate dependency
    
    # Additional metrics
    min_effect = float(deps.min())
    max_effect = float(deps.max())
    q25, q75 = deps.quantile([0.25, 0.75])
    
    results.append({
        'gene': gene,
        'median_effect': median_effect,  # Keep negative (more negative = more essential)
        'mean_effect': mean_effect,
        'min_effect': min_effect,
        'max_effect': max_effect,
        'q25_effect': float(q25),
        'q75_effect': float(q75),
        'frac_dep_lt_minus05': frac_dep_lt_minus05,
        'frac_dep_lt_minus03': frac_dep_lt_minus03,
        'n_cell_lines': len(deps),
        'is_strongly_essential': frac_dep_lt_minus05 > 0.5
    })

# Create results dataframe
results_df = pd.DataFrame(results)

# 6. Optional: Add dependency probability
try:
    print("\nAdding dependency probability...")
    dep_prob = pd.read_csv(DEPMAP_DIR / "CRISPRGeneDependency.csv", index_col=0)
    aml_dep_prob = dep_prob.loc[aml_model_ids].astype(np.float32)
    
    for idx, row in results_df.iterrows():
        gene = row['gene']
        cols = symbol_to_columns[gene]
        
        if len(cols) > 1:
            probs = aml_dep_prob[cols].mean(axis=1)
        else:
            probs = aml_dep_prob[cols[0]]
        
        probs = probs.dropna()
        
        if len(probs) > 0:
            results_df.loc[idx, 'dep_prob_median'] = float(probs.median())
            results_df.loc[idx, 'dep_prob_mean'] = float(probs.mean())
            results_df.loc[idx, 'frac_prob_gt_05'] = float((probs > 0.5).mean())
    
    print("✓ Added dependency probability metrics")
except Exception as e:
    print(f"Could not add dependency probability: {e}")

# 7. Optional: Add expression data
try:
    print("\nAdding expression data...")
    expression = pd.read_csv(
        DEPMAP_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        index_col=0
    )
    
    # Filter to AML lines that exist in expression data
    aml_in_expr = [mid for mid in aml_model_ids if mid in expression.index]
    
    if aml_in_expr:
        aml_expr = expression.loc[aml_in_expr].astype(np.float32)
        
        for idx, row in results_df.iterrows():
            gene = row['gene']
            cols = symbol_to_columns[gene]
            
            if cols[0] in aml_expr.columns:
                expr = aml_expr[cols[0]].dropna()
                if len(expr) > 0:
                    results_df.loc[idx, 'aml_mean_tpm_log1p'] = float(expr.mean())
                    results_df.loc[idx, 'aml_median_tpm_log1p'] = float(expr.median())
        
        print("✓ Added expression metrics")
except Exception as e:
    print(f"Could not add expression data: {e}")

# 8. Rank genes for wet lab
print("\nRanking genes for wet lab validation...")

# Create rank score (more negative median_effect = better)
# Also weight by fraction of lines dependent
results_df['rank_score'] = (
    -results_df['median_effect'] * 0.4 +  # More negative = higher score
    results_df['frac_dep_lt_minus05'] * 0.3 +
    results_df['frac_dep_lt_minus03'] * 0.2
)

# Add dependency probability if available
if 'dep_prob_median' in results_df.columns:
    results_df['rank_score'] += results_df['dep_prob_median'] * 0.1

# Mark known AML targets
aml_targets = {
    'FLT3', 'NPM1', 'DNMT3A', 'IDH1', 'IDH2', 'CD33', 'CD34',
    'BCL2', 'MCL1', 'MYC', 'BIRC5', 'KIT', 'RUNX1', 'TP53',
    'CEBPA', 'NRAS', 'KRAS', 'TET2', 'ASXL1'
}
results_df['is_known_aml_target'] = results_df['gene'].isin(aml_targets)

# Sort by rank score
results_df = results_df.sort_values('rank_score', ascending=False)

# 9. Create outputs for wet lab
print("\n" + "="*60)
print("OUTPUTS FOR WET LAB")
print("="*60)

# A. Top genes for validation
top_100 = results_df.head(100).copy()
top_30 = results_df.head(30).copy()

# B. Save full ranked list
results_df.to_csv(OUTPUT_DIR / 'aml_dependencies_ranked.csv', index=False)
print(f"✓ Full ranked list: {len(results_df)} genes")

# C. Save top 100 for wet lab with key columns
wet_lab_cols = [
    'gene', 'median_effect', 'mean_effect',
    'frac_dep_lt_minus05', 'frac_dep_lt_minus03',
    'n_cell_lines', 'is_known_aml_target'
]

if 'dep_prob_median' in results_df.columns:
    wet_lab_cols.append('dep_prob_median')

wet_lab_df = top_100[wet_lab_cols].copy()
wet_lab_df.to_csv(OUTPUT_DIR / 'wet_lab_top100_genes.csv', index=False)
print(f"✓ Wet lab list: Top 100 genes")

# D. Save reduced 500 gene list
top_500 = results_df.head(500)['gene'].tolist()
with open(OUTPUT_DIR / 'genes_500_depmap.txt', 'w') as f:
    for gene in top_500:
        f.write(f"{gene}\n")
print(f"✓ Reduced gene list: 500 genes")

# E. Save unmatched genes
if unmatched_genes:
    pd.DataFrame({'gene': unmatched_genes}).to_csv(
        OUTPUT_DIR / 'unmatched_genes.csv', index=False
    )
    print(f"✓ Unmatched genes: {len(unmatched_genes)}")

# 10. Summary statistics
print("\n" + "="*60)
print("SUMMARY FOR WET LAB")
print("="*60)
print(f"Total AML lines analyzed: {len(aml_model_ids)}")
print(f"Genes matched: {len(matched_genes)}/{len(your_genes)}")
print(f"\nTop 30 genes for immediate validation:")
print("-"*60)

for idx, row in top_30.iterrows():
    marker = "⭐" if row['is_known_aml_target'] else "  "
    print(f"{marker} {row['gene']:12s} | Effect: {row['median_effect']:6.2f} | "
          f"Dep@-0.5: {row['frac_dep_lt_minus05']*100:5.1f}% | "
          f"Dep@-0.3: {row['frac_dep_lt_minus03']*100:5.1f}%")

# 11. Create summary JSON (FIX: convert numpy types to Python types)
summary = {
    'n_aml_lines': len(aml_model_ids),
    'n_genes_analyzed': len(matched_genes),
    'n_genes_unmatched': len(unmatched_genes),
    'top_10_dependencies': top_30.head(10)['gene'].tolist(),
    'known_targets_in_top30': int(top_30['is_known_aml_target'].sum()),  # Convert to int
    'median_effect_range': [
        float(results_df['median_effect'].min()),
        float(results_df['median_effect'].max())
    ],
    'top_5_strongest': [
        {'gene': row['gene'], 'effect': float(row['median_effect'])}
        for _, row in results_df.head(5).iterrows()
    ]
}

with open(OUTPUT_DIR / 'depmap_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
print(f"All outputs saved to: {OUTPUT_DIR}")
print("\nKey files for wet lab:")
print("  1. wet_lab_top100_genes.csv - Prioritized genes for validation")
print("  2. aml_dependencies_ranked.csv - Full ranked list with metrics")
print("  3. genes_500_depmap.txt - Reduced gene panel for model")
print("\nInterpretation guide:")
print("  • median_effect: More negative → stronger essentiality in AML")
print("  • frac_dep_lt_minus05: Fraction of AML lines with effect < -0.5 (strong dependency)")
print("  • dep_prob_median: Higher → more confident dependency (if available)")
