#!/usr/bin/env python3
"""
Fixed version - Process downloaded survival data
Handles missing openpyxl and processes already downloaded files
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Install missing dependencies if needed
import subprocess
import sys

def install_package(package):
    """Install missing package"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try to import openpyxl, install if missing
try:
    import openpyxl
except ImportError:
    print("Installing openpyxl...")
    install_package('openpyxl')
    import openpyxl

DATA_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/data/survival")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# PROCESS ALREADY DOWNLOADED FILES
# ============================================================================

def process_downloaded_files():
    """
    Process the Excel files that were already downloaded
    """
    print("\n" + "="*80)
    print("PROCESSING DOWNLOADED SURVIVAL DATA")
    print("="*80)
    
    # 1. Process TCGA data
    print("\n1. Processing TCGA-LAML data...")
    
    # Download the TCGA survival data properly
    tcga_survival_url = "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
    
    print("   Downloading TCGA curated survival data...")
    response = requests.get(tcga_survival_url)
    tcga_path = DATA_DIR / "TCGA-CDR-SupplementalTableS1.xlsx"
    
    with open(tcga_path, "wb") as f:
        f.write(response.content)
    
    # Now process it
    try:
        # Read Excel file
        tcga_clinical = pd.read_excel(tcga_path, engine='openpyxl')
        
        # Filter for LAML
        tcga_laml = tcga_clinical[tcga_clinical['type'] == 'LAML'].copy()
        
        # Standardize columns
        tcga_laml_clean = pd.DataFrame({
            'patient_id': tcga_laml['bcr_patient_barcode'],
            'os_status': tcga_laml['OS'],  # 0=alive, 1=dead
            'os_days': tcga_laml['OS.time'],
            'age': tcga_laml['age_at_initial_pathologic_diagnosis'],
            'gender': tcga_laml['gender']
        })
        
        # Remove missing values
        tcga_laml_clean = tcga_laml_clean.dropna(subset=['os_days', 'os_status'])
        
        # Save
        output_path = DATA_DIR / "tcga_laml_survival_clean.csv"
        tcga_laml_clean.to_csv(output_path, index=False)
        
        print(f"   ✓ Processed TCGA-LAML survival data")
        print(f"   Samples: {len(tcga_laml_clean)}")
        print(f"   Deaths: {tcga_laml_clean['os_status'].sum()} ({tcga_laml_clean['os_status'].mean()*100:.1f}%)")
        print(f"   Median survival: {tcga_laml_clean['os_days'].median():.0f} days")
        print(f"   Saved: {output_path}")
        
    except Exception as e:
        print(f"   Error processing TCGA data: {e}")
    
    # 2. Process BeatAML data
    print("\n2. Processing BeatAML data...")
    
    clinical_path = DATA_DIR / "beataml_clinical_raw.xlsx"
    survival_path = DATA_DIR / "beataml_survival_raw.xlsx"
    
    if clinical_path.exists():
        try:
            # Read clinical data
            clinical_sheets = pd.read_excel(clinical_path, sheet_name=None, engine='openpyxl')
            
            # Get the main sheet (usually first one)
            sheet_name = list(clinical_sheets.keys())[0]
            clinical_df = clinical_sheets[sheet_name]
            
            print(f"   Found columns: {clinical_df.columns.tolist()[:10]}...")
            
            # Try to identify survival columns
            beataml_clean = pd.DataFrame()
            
            # Common column patterns
            id_patterns = ['Patient', 'patient', 'Sample', 'sample', 'ID', 'id']
            survival_patterns = ['survival', 'Survival', 'OS', 'os', 'death', 'Death']
            status_patterns = ['status', 'Status', 'vital', 'Vital', 'alive', 'Alive']
            
            # Find matching columns
            for col in clinical_df.columns:
                # Patient ID
                if any(p in col for p in id_patterns):
                    beataml_clean['patient_id'] = clinical_df[col]
                    print(f"   Found patient ID column: {col}")
                    break
            
            for col in clinical_df.columns:
                # Survival time
                if any(p in col for p in survival_patterns) and 'day' in col.lower():
                    beataml_clean['os_days'] = pd.to_numeric(clinical_df[col], errors='coerce')
                    print(f"   Found survival time column: {col}")
                    break
            
            for col in clinical_df.columns:
                # Survival status
                if any(p in col for p in status_patterns):
                    beataml_clean['os_status_raw'] = clinical_df[col]
                    print(f"   Found survival status column: {col}")
                    break
            
            # Convert status to binary
            if 'os_status_raw' in beataml_clean.columns:
                beataml_clean['os_status'] = beataml_clean['os_status_raw'].map({
                    'Dead': 1, 'Alive': 0, 'DEAD': 1, 'ALIVE': 0,
                    'death': 1, 'alive': 0, 'Death': 1, 'Alive': 0,
                    1: 1, 0: 0, '1': 1, '0': 0
                })
                beataml_clean = beataml_clean.drop('os_status_raw', axis=1)
            
            # Clean and save
            beataml_clean = beataml_clean.dropna(subset=['patient_id'])
            
            if len(beataml_clean) > 0:
                output_path = DATA_DIR / "beataml_survival_clean.csv"
                beataml_clean.to_csv(output_path, index=False)
                print(f"   ✓ Processed BeatAML survival data")
                print(f"   Samples: {len(beataml_clean)}")
                if 'os_status' in beataml_clean.columns:
                    print(f"   Deaths: {beataml_clean['os_status'].sum()}")
                print(f"   Saved: {output_path}")
            
        except Exception as e:
            print(f"   Error processing BeatAML data: {e}")
    
    # 3. Fix TARGET-AML data
    print("\n3. Fixing TARGET-AML data...")
    
    # Re-download with proper parameters
    endpoint = "https://api.gdc.cancer.gov/cases"
    
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "project.project_id",
                    "value": ["TARGET-AML"]
                }
            }
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "fields": "submitter_id,demographic.vital_status,demographic.days_to_death,diagnoses.days_to_last_follow_up",
        "expand": "demographic,diagnoses",
        "format": "json",
        "size": "500"
    }
    
    response = requests.get(endpoint, params=params)
    target_data = response.json()
    
    # Process TARGET data properly
    target_list = []
    for case in target_data['data']['hits']:
        patient_data = {
            'patient_id': case.get('submitter_id', '')
        }
        
        # Get demographic data
        if case.get('demographic'):
            demo = case['demographic'][0] if isinstance(case['demographic'], list) else case['demographic']
            patient_data['vital_status'] = demo.get('vital_status', '')
            patient_data['days_to_death'] = demo.get('days_to_death')
        
        # Get follow-up from diagnoses
        if case.get('diagnoses'):
            diag = case['diagnoses'][0] if isinstance(case['diagnoses'], list) else case['diagnoses']
            patient_data['days_to_last_follow_up'] = diag.get('days_to_last_follow_up')
        
        target_list.append(patient_data)
    
    target_df = pd.DataFrame(target_list)
    
    # Clean survival data
    target_df['os_status'] = target_df['vital_status'].map({'Dead': 1, 'Alive': 0, 'dead': 1, 'alive': 0})
    target_df['os_days'] = target_df['days_to_death'].fillna(target_df['days_to_last_follow_up'])
    
    # Remove invalid entries
    target_clean = target_df[['patient_id', 'os_status', 'os_days']].dropna()
    
    # Save
    output_path = DATA_DIR / "target_aml_survival_clean.csv"
    target_clean.to_csv(output_path, index=False)
    
    print(f"   ✓ Processed TARGET-AML survival data")
    print(f"   Samples with survival: {len(target_clean)}")
    if len(target_clean) > 0:
        print(f"   Deaths: {target_clean['os_status'].sum()} ({target_clean['os_status'].mean()*100:.1f}%)")
        print(f"   Median follow-up: {target_clean['os_days'].median():.0f} days")
    print(f"   Saved: {output_path}")

# ============================================================================
# MATCH WITH YOUR PERSISTER PREDICTIONS
# ============================================================================

def match_survival_with_predictions():
    """
    Match the cleaned survival data with your persister predictions
    """
    print("\n" + "="*80)
    print("MATCHING WITH PERSISTER PREDICTIONS")
    print("="*80)
    
    pred_dir = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results")
    
    # 1. Match TCGA
    tcga_surv = DATA_DIR / "tcga_laml_survival_clean.csv"
    tcga_pred = pred_dir / "bulk_TCGA/predictions_60pct.csv"
    
    if tcga_surv.exists() and tcga_pred.exists():
        print("\n1. Matching TCGA data...")
        
        survival = pd.read_csv(tcga_surv)
        predictions = pd.read_csv(tcga_pred)
        
        # TCGA IDs are usually in format TCGA-XX-XXXX
        # Extract the patient barcode (first 12 characters)
        predictions['patient_barcode'] = predictions['sample_id'].str[:12]
        
        # Merge
        merged = pd.merge(
            predictions,
            survival,
            left_on='patient_barcode',
            right_on='patient_id',
            how='inner'
        )
        
        if len(merged) > 0:
            output_path = DATA_DIR / "tcga_persister_survival_matched.csv"
            merged.to_csv(output_path, index=False)
            print(f"   ✓ Matched {len(merged)} TCGA samples")
            print(f"   High persisters: {(merged['prediction'] == 'Persister').sum()}")
            print(f"   Deaths in high persisters: {merged[merged['prediction'] == 'Persister']['os_status'].sum()}")
            print(f"   Saved: {output_path}")
        else:
            print("   No matches found - checking ID formats...")
            print(f"   Sample prediction IDs: {predictions['sample_id'].head()}")
            print(f"   Sample survival IDs: {survival['patient_id'].head()}")
    
    # 2. Match BeatAML
    beataml_surv = DATA_DIR / "beataml_survival_clean.csv"
    beataml_pred = pred_dir / "bulk_BeatAML/predictions_60pct.csv"
    
    if beataml_surv.exists() and beataml_pred.exists():
        print("\n2. Matching BeatAML data...")
        
        survival = pd.read_csv(beataml_surv)
        predictions = pd.read_csv(beataml_pred)
        
        # Try direct matching
        merged = pd.merge(
            predictions,
            survival,
            left_on='sample_id',
            right_on='patient_id',
            how='inner'
        )
        
        if len(merged) > 0:
            output_path = DATA_DIR / "beataml_persister_survival_matched.csv"
            merged.to_csv(output_path, index=False)
            print(f"   ✓ Matched {len(merged)} BeatAML samples")
            print(f"   High persisters: {(merged['prediction'] == 'Persister').sum()}")
            if 'os_status' in merged.columns:
                print(f"   Deaths in high persisters: {merged[merged['prediction'] == 'Persister']['os_status'].sum()}")
            print(f"   Saved: {output_path}")
        else:
            print("   No direct matches - may need manual ID mapping")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

def create_summary_report():
    """
    Create summary of all available survival data
    """
    print("\n" + "="*80)
    print("SURVIVAL DATA SUMMARY")
    print("="*80)
    
    for file in DATA_DIR.glob("*_clean.csv"):
        print(f"\n{file.stem}:")
        df = pd.read_csv(file)
        print(f"  Samples: {len(df)}")
        
        if 'os_days' in df.columns and 'os_status' in df.columns:
            valid = df.dropna(subset=['os_days', 'os_status'])
            print(f"  Valid survival data: {len(valid)}")
            print(f"  Deaths: {valid['os_status'].sum()} ({valid['os_status'].mean()*100:.1f}%)")
            print(f"  Median follow-up: {valid['os_days'].median():.0f} days ({valid['os_days'].median()/365:.1f} years)")
    
    print("\nMatched datasets (ready for survival analysis):")
    for file in DATA_DIR.glob("*_matched.csv"):
        df = pd.read_csv(file)
        print(f"  {file.stem}: {len(df)} samples")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("PROCESSING AML SURVIVAL DATA")
    print("="*80)
    
    # Process downloaded files
    process_downloaded_files()
    
    # Match with predictions
    match_survival_with_predictions()
    
    # Create summary
    create_summary_report()
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Check matched files in:", DATA_DIR)
    print("2. Run survival analysis: python survival_analysis.py")

if __name__ == "__main__":
    main()
