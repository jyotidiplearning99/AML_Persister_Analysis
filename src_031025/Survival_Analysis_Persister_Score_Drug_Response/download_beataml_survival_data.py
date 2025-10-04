#!/usr/bin/env python3
"""
Fixed Download and process BeatAML survival data from cBioPortal
Handles the actual data files correctly
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path

def download_beataml_survival_fixed():
    """Download BeatAML clinical data with proper handling"""
    
    print("="*80)
    print("DOWNLOADING BEATAML SURVIVAL DATA")
    print("="*80)
    
    # Create directories
    data_dir = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/data")
    survival_dir = data_dir / "survival"
    cbio_dir = data_dir / "cbioportal_beataml"
    
    survival_dir.mkdir(exist_ok=True, parents=True)
    cbio_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n1. Downloading BeatAML clinical data from cBioPortal...")
    
    # Direct download URLs for the actual data files
    files_to_download = {
        "data_clinical_patient.txt": "https://github.com/cBioPortal/datahub/raw/master/public/aml_ohsu_2018/data_clinical_patient.txt",
        "data_clinical_sample.txt": "https://github.com/cBioPortal/datahub/raw/master/public/aml_ohsu_2018/data_clinical_sample.txt"
    }
    
    for filename, url in files_to_download.items():
        file_path = cbio_dir / filename
        
        print(f"   Downloading {filename}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"   ✓ Saved: {file_path}")
            
            # Check file size to ensure we got actual data
            file_size = file_path.stat().st_size
            print(f"     File size: {file_size:,} bytes")
            
            if file_size < 1000:  # If file is too small, it might be a pointer
                print("     Warning: File seems too small, might be a pointer")
        else:
            print(f"   Error downloading {filename}: Status {response.status_code}")
    
    return cbio_dir

def process_beataml_survival_fixed(cbio_dir):
    """Process BeatAML survival data with fixed parsing"""
    
    print("\n" + "="*80)
    print("PROCESSING BEATAML SURVIVAL DATA")
    print("="*80)
    
    # Paths
    SURV_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/data/survival")
    PRED = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/bulk_BeatAML/predictions_60pct.csv")
    
    # Load predictions
    print("\n1. Loading persister predictions...")
    pred = pd.read_csv(PRED)
    print(f"   Loaded {len(pred)} predictions")
    
    # Check what the sample IDs look like
    print(f"   Sample ID examples: {pred['sample_id'].head(3).tolist()}")
    
    # Load clinical files with proper parsing
    patient_file = cbio_dir / "data_clinical_patient.txt"
    sample_file = cbio_dir / "data_clinical_sample.txt"
    
    print("\n2. Loading clinical data...")
    
    # Read patient file - skip comment lines starting with #
    pat = pd.read_table(patient_file, comment='#')
    print(f"   Loaded {len(pat)} patients")
    print(f"   Patient columns: {pat.columns.tolist()}")
    
    # Read sample file - skip comment lines
    samp = pd.read_table(sample_file, comment='#')
    print(f"   Loaded {len(samp)} samples")
    print(f"   Sample columns: {samp.columns.tolist()}")
    
    # Keep only necessary columns from sample file
    if 'SAMPLE_ID' in samp.columns and 'PATIENT_ID' in samp.columns:
        samp = samp[['SAMPLE_ID', 'PATIENT_ID']]
        print(f"   Sample ID examples: {samp['SAMPLE_ID'].head(3).tolist()}")
    
    print("\n3. Processing survival fields...")
    
    # Harmonize survival fields
    if 'OS_STATUS' in pat.columns:
        # Map text values to binary
        pat['OS_STATUS_BIN'] = pat['OS_STATUS'].map({
            'DECEASED': 1, '1:DECEASED': 1, 'DEAD': 1, 1: 1,
            'LIVING': 0, '0:LIVING': 0, 'ALIVE': 0, 0: 0
        })
        print(f"   OS_STATUS values: {pat['OS_STATUS'].value_counts().to_dict()}")
    
    if 'OS_MONTHS' in pat.columns:
        pat = pat.rename(columns={'OS_MONTHS': 'os_months'})
        pat['os_days'] = pat['os_months'] * 30.44
        print(f"   Survival time available (median: {pat['os_months'].median():.1f} months)")
    
    # Additional clinical variables
    if 'AGE' in pat.columns:
        pat['age'] = pd.to_numeric(pat['AGE'], errors='coerce')
    if 'SEX' in pat.columns:
        pat['gender'] = pat['SEX']
    
    print("\n4. Matching predictions with survival data...")
    
    # The BeatAML prediction sample IDs look like: "aml_ohsu_2018_12-00023"
    # The clinical SAMPLE_IDs might be in format: "12-00023"
    
    # Try direct match first
    m = pred.merge(samp, left_on='sample_id', right_on='SAMPLE_ID', how='left')
    matched_count = m['PATIENT_ID'].notna().sum()
    
    if matched_count == 0:
        print("   Direct match failed. Trying to extract ID parts...")
        
        # Extract the numeric part after the last underscore from predictions
        pred['sample_key'] = pred['sample_id'].str.extract(r'(\d+-\d+)')
        
        # Try matching on this extracted key
        if 'sample_key' in pred.columns and pred['sample_key'].notna().any():
            m = pred.merge(samp, left_on='sample_key', right_on='SAMPLE_ID', how='left')
            matched_count = m['PATIENT_ID'].notna().sum()
            print(f"   Matched {matched_count} samples using extracted IDs")
    else:
        print(f"   Matched {matched_count} samples directly")
    
    # Merge with patient data
    if 'PATIENT_ID' in m.columns and matched_count > 0:
        # Select columns to merge
        pat_cols = ['PATIENT_ID', 'OS_STATUS_BIN', 'os_months', 'os_days']
        pat_cols += [col for col in ['age', 'gender'] if col in pat.columns]
        
        m = m.merge(pat[pat_cols], on='PATIENT_ID', how='left')
    
    # Rename final columns
    m = m.rename(columns={'OS_STATUS_BIN': 'os_status'})
    
    # Keep essential columns
    essential_cols = ['sample_id', 'persister_probability', 'prediction', 
                     'os_status', 'os_days', 'os_months']
    optional_cols = ['PATIENT_ID', 'SAMPLE_ID', 'age', 'gender']
    
    final_cols = []
    for col in essential_cols + optional_cols:
        if col in m.columns:
            final_cols.append(col)
    
    m = m[final_cols]
    
    # Save matched data
    out = SURV_DIR / "beataml_persister_survival_matched.csv"
    m.to_csv(out, index=False)
    
    print(f"\n✓ Wrote: {out}")
    print(f"  Total samples: {len(m)}")
    print(f"  With survival: {m['os_status'].notna().sum()}")
    
    if m['os_status'].notna().sum() > 0:
        valid = m.dropna(subset=['os_status', 'os_days'])
        print(f"  Deaths: {valid['os_status'].sum()} ({valid['os_status'].mean()*100:.1f}%)")
        print(f"  Median OS: {valid['os_days'].median():.0f} days")
    
    return m

def main():
    """Main function"""
    
    print("\n" + "="*80)
    print("BEATAML SURVIVAL DATA PIPELINE - FIXED")
    print("="*80)
    
    # Download data
    cbio_dir = download_beataml_survival_fixed()
    
    # Process and match
    matched_data = process_beataml_survival_fixed(cbio_dir)
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    
    if matched_data['os_status'].notna().sum() > 0:
        print("\n✓ Survival data successfully matched!")
        print("Next steps:")
        print("1. Run survival analysis: python survival_analysis_persister_score_beataml.py")
        print("2. Run drug response analysis")
    else:
        print("\n⚠ No survival data matched. Check sample ID formats.")

if __name__ == "__main__":
    main()
