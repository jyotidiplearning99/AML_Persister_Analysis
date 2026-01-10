#!/usr/bin/env python3
"""
Complete AML Persister Analysis with Sample Mapping
"""

import os, sys, logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger()

# ============================================================================
# CONFIGURATION
# ============================================================================

AML_ROOT = Path("/scratch/project_2010751/AML_scRNA_decrypted")
OUT_DIR = Path("./aml_analysis_final")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: Discover Actual Samples
# ============================================================================

def discover_samples():
    """Find all scRNA-seq samples with proper patient IDs"""
    samples = []
    
    if not AML_ROOT.exists():
        log.error(f"AML root not found: {AML_ROOT}")
        return samples
    
    SKIP = {"filtered_feature_bc_matrix", "outs", "count", "AGG"}
    
    for matrix_dir in AML_ROOT.rglob("filtered_feature_bc_matrix"):
        if not matrix_dir.is_dir():
            continue
        
        # Walk up to find patient ID
        parent = matrix_dir.parent
        for _ in range(10):  # Safety limit
            if parent.name not in SKIP and parent != AML_ROOT:
                patient_id = parent.name
                samples.append((str(matrix_dir), patient_id))
                break
            if parent.parent == parent:
                break
            parent = parent.parent
    
    return samples

log.info("\n" + "="*80)
log.info("STEP 1: Discovering scRNA-seq Samples")
log.info("="*80)

found_samples = discover_samples()
log.info(f"\nFound {len(found_samples)} scRNA-seq samples:")
for path, sid in found_samples:
    log.info(f"  {sid}")

# ============================================================================
# STEP 2: Load Clinical Data
# ============================================================================

log.info("\n" + "="*80)
log.info("STEP 2: Loading Clinical Data")
log.info("="*80)

clinical_data = {
    'sample_id': ['FHRB_706','FHRB_188','FHRB_436','FHRB_560','BERG_470','FHRB_279',
                  'FHRB_268','FHRB_437','FHRB_252','FHRB_434','FHRB_382','FHRB_106',
                  'FHRB_121','FHRB_209','FHRB_366','FHRB_139','FHRB_600','FHRB_393',
                  'FHRB_743','FHRB_349','FH_4991_','FH_5713_','FH_3081_','FH_4599_',
                  'FH_5034_','FH_5184_','FHRB_468','FH_5776_','FH_6088_','FH_5897_',
                  'FH_6310_','FH_6323_','FH_6389_','FH_6512_','FH_6532_','FH_6545_',
                  'FH_6565_','FH_6576_','FH_6525_','FH_6940_','FH_7087_','FH_6810_',
                  'FH_7289_'],
    'overall_survival_days': [1001,463,3226,4336,None,567,67,3218,2075,241,288,1222,
                              1669,20,369,107,247,94,2069,3626,66,615,936,5235,23,48,
                              364,31,690,2709,151,718,2546,2490,2471,172,2452,2448,
                              2442,2281,2225,2137,2175],
    'date_of_death': ['2013-12-02','2014-02-18',None,'2012-06-11',None,'2015-03-06',
                      '2013-08-25',None,'2012-05-04','2015-11-02','2015-04-29',
                      '2015-01-03','2016-07-22','2013-04-01','2015-03-18','2012-07-12',
                      '2011-08-07','2015-02-13','2016-11-27',None,'2016-04-14',
                      '2018-02-27','2016-05-21',None,'2016-03-09','2016-05-08',
                      '2016-09-13','2016-09-03','2018-07-21',None,'2017-04-23',
                      '2018-11-26',None,None,None,'2017-09-29',None,None,None,None,
                      None,None,None],
    'diagnosis': ['AML NPM1','AML M5 N','AML NPM1','AML M2 N','CMML -> A','AML Thera',
                  'AML M2 N','AML M5 N','AML M5','AML NPM1','AML MLLT','AML Thera',
                  'AML CEBP','AML M1','AML M5','CMML -> A','AML M5','AML/MDS',
                  'AML M1','AML NPM1','AML M1','AML Thera','AML secon','AML CEBP',
                  'MDS => AML','AML MLLT','AML Thera','AML M1','AML M5','AML M2 N',
                  'AML Thera','AML M2 N','AML M2 N','AML M0','AML Thera','AML M1 or',
                  'AML Thera','AML M1','AML M2 s','AML M0','AML M5','MDS progre',
                  'AML NPM1']
}

clinical_df = pd.DataFrame(clinical_data).drop_duplicates(subset=['sample_id'])
clinical_df['date_of_death'] = pd.to_datetime(clinical_df['date_of_death'])
clinical_df['event'] = clinical_df['date_of_death'].notna().astype(int)
clinical_df['sample_clean'] = clinical_df['sample_id'].str.rstrip('_')

log.info(f"\n✓ Loaded {len(clinical_df)} patients")
log.info(f"  Deaths: {clinical_df['event'].sum()}")
log.info(f"  Censored: {(1-clinical_df['event']).sum()}")

# ============================================================================
# STEP 3: Create Sample Mapping
# ============================================================================

log.info("\n" + "="*80)
log.info("STEP 3: Matching scRNA-seq ↔ Clinical IDs")
log.info("="*80)

# Extract scRNA sample IDs
scrna_ids = [sid for _, sid in found_samples]
clinical_ids = clinical_df['sample_clean'].tolist()

log.info(f"\nscRNA-seq samples found: {scrna_ids}")
log.info(f"\nClinical table expects: {clinical_ids[:10]}...")

# Try fuzzy matching
mapping = []
for scrna_id in scrna_ids:
    # Remove trailing numbers (_2, _3)
    base_id = scrna_id.rstrip('0123456789_')
    
    # Look for partial matches in clinical IDs
    matches = [cid for cid in clinical_ids if base_id in cid or cid in scrna_id]
    
    if matches:
        mapping.append({'scrna_id': scrna_id, 'clinical_id': matches[0], 'match_type': 'auto'})
        log.info(f"  ✓ {scrna_id} → {matches[0]}")
    else:
        mapping.append({'scrna_id': scrna_id, 'clinical_id': 'UNKNOWN', 'match_type': 'manual_needed'})
        log.info(f"  ✗ {scrna_id} → NO MATCH")

mapping_df = pd.DataFrame(mapping)
mapping_df.to_csv(OUT_DIR / 'sample_mapping.csv', index=False)
log.info(f"\n✓ Saved mapping: {OUT_DIR / 'sample_mapping.csv'}")

# ============================================================================
# STEP 4: Load Predictions (if available)
# ============================================================================

log.info("\n" + "="*80)
log.info("STEP 4: Loading Predictions")
log.info("="*80)

pred_file = Path("./predictions_output/inference_results_summary.csv")

if pred_file.exists():
    pred_df = pd.read_csv(pred_file)
    log.info(f"\n✓ Loaded predictions for {len(pred_df)} samples")
    
    # Apply mapping
    pred_df = pred_df.merge(mapping_df, left_on='sample', right_on='scrna_id', how='left')
    pred_df['matched_clinical_id'] = pred_df['clinical_id'].fillna(pred_df['sample'])
    
    # Merge with clinical
    merged = clinical_df.merge(pred_df, left_on='sample_clean', right_on='matched_clinical_id', how='inner')
    
    log.info(f"\n✓ Matched {len(merged)} samples with BOTH clinical + predictions:")
    if len(merged) > 0:
        print("\n" + merged[['sample_id', 'sample', 'persister_pct', 'overall_survival_days', 'event']].to_string(index=False))
        
        # Save
        merged.to_csv(OUT_DIR / 'matched_data.csv', index=False)
        
        # Simple correlation if ≥3 samples
        if len(merged) >= 3:
            valid = merged.dropna(subset=['persister_pct', 'overall_survival_days'])
            if len(valid) >= 3:
                rho, p = spearmanr(valid['persister_pct'], valid['overall_survival_days'])
                log.info(f"\n📊 Correlation: Spearman ρ={rho:.3f}, p={p:.4f}")
                if p < 0.05:
                    direction = "NEGATIVE (higher persister → shorter survival)" if rho < 0 else "POSITIVE"
                    log.info(f"   → Significant {direction} correlation ✓")
                else:
                    log.info(f"   → Not significant (need more samples)")
    else:
        log.warning("\n❌ NO MATCHES! See mapping file to manually link IDs.")
else:
    log.warning(f"\n⚠ No predictions found at {pred_file}")
    log.info("\nRun inference first:")
    log.info("  python <inference_script>.py \\")
    log.info("    --model-dir /scratch/.../reduced_model_distilled \\")
    log.info("    --genes-file .../selected_genes.txt \\")
    log.info("    --out-dir ./predictions_output \\")
    log.info("    --aml-root /scratch/project_2010751/AML_scRNA_decrypted")

# ============================================================================
# ANSWER YOUR QUESTIONS
# ============================================================================

log.info("\n" + "="*80)
log.info("ANSWERS TO YOUR QUESTIONS")
log.info("="*80)

log.info("""
Q1: "Can I correlate with relapse/response/risk stratification?"
────────────────────────────────────────────────────────────────
❌ NO - Your clinical table does NOT contain:

Missing Data              | Column Name Needed
─────────────────────────┼────────────────────────────────────
Relapse timing            | date_of_relapse
Treatment response        | response_status (CR/PR/NR)
Risk stratification       | cytogenetic_risk or eln_risk (Low/Int/High)

✓ What you HAVE:
  • overall_survival_days
  • date_of_death
  • diagnosis (NPM1, M5, etc.)

✓ What you CAN analyze:
  • Overall survival correlation
  • Cox regression (if ≥10 matched samples)
  • Kaplan-Meier curves (if ≥6 matched samples)

To get missing data, contact your clinical collaborator!

────────────────────────────────────────────────────────────────
Q2: "Can't we predict from genes directly (skip persister step)?"
────────────────────────────────────────────────────────────────
YES, but your CURRENT approach is BETTER! Here's why:

Approach A (CURRENT):
  1000 genes → ML model → Persister % → Survival
  ✓ Biologically interpretable (persister cells = known mechanism)
  ✓ Dimensionality reduction (1000 → 1 score)
  ✓ Clinically actionable (can target persisters)

Approach B (DIRECT):
  1000 genes → Cox regression → Survival
  ✗ Overfitting risk (1000 predictors for ~3 patients!)
  ✗ Not interpretable (which genes matter?)
  ✗ Requires huge sample size (need 100+ patients)

RECOMMENDATION: Keep genes → persister % → survival workflow!

────────────────────────────────────────────────────────────────
SUMMARY OF YOUR SITUATION
────────────────────────────────────────────────────────────────
• Clinical table: 44 patients
• scRNA-seq data: ~6 samples (different IDs!)
• Matched samples: Likely 2-3 after mapping
• Statistical power: TOO LOW for robust analysis

WHAT TO DO:
1. Check the generated sample_mapping.csv
2. Manually correct any UNKNOWN mappings
3. Run inference with corrected mapping
4. Accept that analysis will be DESCRIPTIVE (not statistical)
5. Collect more samples OR get relapse/response data
""")

log.info("\n" + "="*80)
log.info("COMPLETE")
log.info("="*80)
log.info(f"\nFiles saved to: {OUT_DIR}")
log.info("  • sample_mapping.csv (check & correct manually)")
log.info("  • matched_data.csv (if predictions available)")
