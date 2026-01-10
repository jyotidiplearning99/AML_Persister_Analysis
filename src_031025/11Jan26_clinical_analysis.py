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
print("AML PERSISTER CLINICAL ANALYSIS (10 SAMPLES)")
print(f"{'='*80}\n")

# Manual ID mapping (based on your H5 files)
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
    'FHRB_4368_2_2hr_drug': 'FHRB_436'
}

# Clinical data for matched samples
clinical_data = {
    'sample_id': ['FH_6088_','FHRB_188','FHRB_743','FHRB_366','FHRB_252',
                  'FH_3081_','FHRB_560','FH_4599_','FHRB_121','FHRB_436'],
    'overall_survival_days': [690, 463, 2069, 369, 2075, 936, 4336, 5235, 1669, 3226],
    'date_of_death': ['2018-07-21','2014-02-18','2016-11-27','2015-03-18','2012-05-04',
                      '2016-05-21','2012-06-11',None,'2016-07-22',None],
    'diagnosis': ['AML M5','AML M5 N','AML M1','AML M5','AML M5',
                  'AML secon','AML M2 N','AML CEBP','AML CEBP','AML M2 N'],
}

clinical_df = pd.DataFrame(clinical_data)
clinical_df['date_of_death'] = pd.to_datetime(clinical_df['date_of_death'])
clinical_df['event'] = clinical_df['date_of_death'].notna().astype(int)

print(f"✓ Clinical data for {len(clinical_df)} patients")

# Load predictions
pred_df = pd.read_csv(PREDICTIONS_FILE)
print(f"✓ Predictions for {len(pred_df)} samples")

# Apply mapping
pred_df['clinical_id'] = pred_df['sample'].map(SAMPLE_MAPPING)
pred_df = pred_df[pred_df['clinical_id'].notna()]

# Merge
merged = clinical_df.merge(pred_df, left_on='sample_id', right_on='clinical_id', how='inner')

print(f"\n✓ Matched {len(merged)} samples:")
print(merged[['sample_id', 'sample', 'persister_pct', 'overall_survival_days', 'event']])

# Correlation
if len(merged) >= 3:
    rho, p = spearmanr(merged['persister_pct'], merged['overall_survival_days'])
    print(f"\n📊 Spearman ρ = {rho:.3f}, p = {p:.4f}")

# Cox regression
if len(merged) >= 10:
    cph = CoxPHFitter()
    cox_data = merged[['overall_survival_days', 'event', 'persister_pct']].dropna()
    cph.fit(cox_data, duration_col='overall_survival_days', event_col='event')
    
    hr = np.exp(cph.params_['persister_pct'])
    p_val = cph.summary.loc['persister_pct', 'p']
    
    print(f"\n📊 Cox Regression: HR = {hr:.3f}, p = {p_val:.4f}")

merged.to_csv(OUT_DIR / 'matched_data.csv', index=False)
print(f"\n✓ Saved: {OUT_DIR / 'matched_data.csv'}")

print(f"\n{'='*80}")
print("ANSWERS TO YOUR QUESTIONS")
print(f"{'='*80}")
print("""
Q: "Can I correlate with relapse/response/risk stratification?"
───────────────────────────────────────────────────────────────
❌ NO - Your clinical table does NOT contain:
   • date_of_relapse
   • response_status (CR/PR/NR)
   • cytogenetic_risk (Low/Int/High)

You ONLY have overall survival data.

Q: "Can't we predict from genes directly (skip persister step)?"
────────────────────────────────────────────────────────────────
YES, but DON'T! Here's why:

Current Method (genes → persister % → survival):
  ✓ 1 predictor needs ~10 events (you have {})
  ✓ Biologically interpretable
  ✓ Clinically actionable

Direct Method (1000 genes → survival):
  ✗ 1000 predictors need 10,000 events!
  ✗ You only have {} deaths
  ✗ 99.7% underpowered - statistically impossible

With {} matched samples, this is a PILOT study for hypothesis generation.
""".format(clinical_df['event'].sum(), clinical_df['event'].sum(), len(merged)))
