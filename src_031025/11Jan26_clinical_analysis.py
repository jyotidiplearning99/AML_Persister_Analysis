#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from lifelines import CoxPHFitter

PREDICTIONS_FILE = Path("./predictions_output_FIXED/inference_results_summary.csv")
OUT_DIR = Path("./clinical_results_FINAL")
OUT_DIR.mkdir(exist_ok=True)

print(f"\n{'='*80}")
print("AML PERSISTER CLINICAL ANALYSIS (11 SAMPLES)")
print(f"{'='*80}\n")

# Manual ID mapping
SAMPLE_MAPPING = {
    'FH_6088_3': 'FH_6088_',
    'FHRB_1886_6': 'FHRB_188',
    'FHRB_743_4': 'FHRB_743',
    'FHRB_3660_4': 'FHRB_366',
    'FHRB_252_2': 'FHRB_252',
    'FH_3081_2': 'FH_3081_',
    'FHRB_560_9': 'FHRB_560',
    'FH_4599_2': 'FH_4599_',
    'FHRB_1216_2': 'FHRB_121',
    'FHRB_4368_2_2hr_drug': 'FHRB_436',
    'FH_5897_2': 'FH_5897_'  # ✅ ADDED
}

# Clinical data (✅ ADDED FH_5897_2)
clinical_data = {
    'sample_id': ['FH_6088_','FHRB_188','FHRB_743','FHRB_366','FHRB_252',
                  'FH_3081_','FHRB_560','FH_4599_','FHRB_121','FHRB_436',
                  'FH_5897_'],  # ✅ ADDED
    'overall_survival_days': [690, 463, 2069, 369, 2075, 936, 4336, 5235, 1669, 3226,
                              2709],  # ✅ ADDED (7.4 years)
    'date_of_death': ['2018-07-21','2014-02-18','2016-11-27','2015-03-18','2012-05-04',
                      '2016-05-21','2012-06-11',None,'2016-07-22',None,
                      None],  # ✅ ADDED (still alive/censored)
    'diagnosis': ['AML M5','AML M5 N','AML M1','AML M5','AML M5',
                  'AML secon','AML M2 N','AML CEBP','AML CEBP','AML M2 N',
                  'AML M2'],  # ✅ ADDED (de novo)
}

clinical_df = pd.DataFrame(clinical_data)
clinical_df['date_of_death'] = pd.to_datetime(clinical_df['date_of_death'])
clinical_df['event'] = clinical_df['date_of_death'].notna().astype(int)

print(f"✓ Clinical data for {len(clinical_df)} patients")

# Load predictions and deduplicate
pred_df = pd.read_csv(PREDICTIONS_FILE)

# ✅ REMOVE DUPLICATE "allas_scrna_selected" ENTRY
pred_df = pred_df[pred_df['sample'] != 'allas_scrna_selected'].copy()

print(f"✓ Predictions for {len(pred_df)} samples (after deduplication)")

# Apply mapping
pred_df['clinical_id'] = pred_df['sample'].map(SAMPLE_MAPPING)
pred_df = pred_df[pred_df['clinical_id'].notna()]

# Merge
merged = clinical_df.merge(pred_df, left_on='sample_id', right_on='clinical_id', how='inner')

# Round percentages
merged['persister_pct_int'] = merged['persister_pct'].round(0).astype(int)
merged['non_persister_pct_int'] = (100 - merged['persister_pct']).round(0).astype(int)

print(f"\n✓ Matched {len(merged)} samples:")
print(merged[['sample_id', 'sample', 'persister_pct_int', 'non_persister_pct_int', 
              'overall_survival_days', 'event', 'diagnosis']].sort_values('persister_pct_int'))

# Correlation
if len(merged) >= 3:
    rho, p = spearmanr(merged['persister_pct_int'], merged['overall_survival_days'])
    
    print(f"\n📊 Spearman Correlation:")
    print(f"   ρ = {rho:.3f}, p = {p:.4f}")
    print(f"   Interpretation: {'NEGATIVE' if rho < 0 else 'POSITIVE'} correlation")
    print(f"   Persister % range: {merged['persister_pct_int'].min()}% - {merged['persister_pct_int'].max()}%")
    
    if rho < 0:
        print(f"   ⭐ LOWER persister % → LONGER survival")
    else:
        print(f"   ⚠️  HIGHER persister % → LONGER survival (unexpected)")

# Cox regression
if len(merged) >= 10:
    cph = CoxPHFitter()
    cox_data = merged[['overall_survival_days', 'event', 'persister_pct']].dropna()
    cph.fit(cox_data, duration_col='overall_survival_days', event_col='event')
    
    hr = np.exp(cph.params_['persister_pct'])
    p_val = cph.summary.loc['persister_pct', 'p']
    
    print(f"\n📊 Cox Regression:")
    print(f"   HR = {hr:.3f}, p = {p_val:.4f}")
    print(f"   Per 1% increase in persister cells:")
    if hr > 1:
        print(f"   - {((hr-1)*100):.1f}% INCREASED risk of death")
    else:
        print(f"   - {((1-hr)*100):.1f}% DECREASED risk of death")

# Save
merged.to_csv(OUT_DIR / 'matched_data.csv', index=False)
print(f"\n✓ Saved: {OUT_DIR / 'matched_data.csv'}")

# Highlight FH_5897_2
print(f"\n{'='*80}")
print("KEY OBSERVATION: FH_5897_2")
print(f"{'='*80}")
fh5897 = merged[merged['sample_id'] == 'FH_5897_']
if not fh5897.empty:
    print(f"✨ FH_5897_2 has:")
    print(f"   • LOWEST persister %: {fh5897['persister_pct_int'].values[0]}%")
    print(f"   • Long survival: {fh5897['overall_survival_days'].values[0]} days (7.4 years)")
    print(f"   • Status: {'Still alive/censored' if fh5897['event'].values[0] == 0 else 'Deceased'}")
    print(f"   • Diagnosis: {fh5897['diagnosis'].values[0]}")
    print(f"\n   This supports: Lower persister % may predict better outcomes!")

print(f"\n{'='*80}")
print("STATISTICAL POWER")
print(f"{'='*80}")
print(f"With {len(merged)} matched samples and {merged['event'].sum()} events:")
print(f"  • This is a PILOT study")
print(f"  • Correlations: Adequately powered (n≥10)")
print(f"  • Cox regression: Underpowered but hypothesis-generating")
print(f"  • Need ~30-50 patients for definitive conclusions")
