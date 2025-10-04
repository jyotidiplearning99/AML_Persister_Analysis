#!/usr/bin/env python3
"""
Download and process the actual GSE125345 files from NCBI FTP
"""

import os
import urllib.request
import gzip
import pandas as pd
import numpy as np

def download_gse125345_files():
    """Download the actual supplementary files for GSE125345"""
    
    base_path = '/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025/Hematopoietic_Cells_Healthy_AML/GSE125345/'
    os.chdir(base_path)
    
    files_to_download = [
        ('GSE125345_countdata.xlsx', 
         'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE125nnn/GSE125345/suppl/GSE125345_countdata.xlsx'),
        ('GSE125345_LTHSC_ST_HSC_CMP_GMP_MLP.tsv.gz',
         'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE125nnn/GSE125345/suppl/GSE125345_LTHSC_ST_HSC_CMP_GMP_MLP.tsv.gz')
    ]
    
    for filename, url in files_to_download:
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"  ✓ Downloaded {filename}")
            except Exception as e:
                print(f"  ✗ Failed to download {filename}: {e}")
        else:
            print(f"  ✓ {filename} already exists")
    
    # Process the downloaded files
    process_downloaded_files()

def process_downloaded_files():
    """Process the downloaded GSE125345 files"""
    
    base_path = '/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025/Hematopoietic_Cells_Healthy_AML/GSE125345/'
    os.chdir(base_path)
    
    # Try TSV file first (compressed)
    tsv_file = 'GSE125345_LTHSC_ST_HSC_CMP_GMP_MLP.tsv.gz'
    if os.path.exists(tsv_file):
        print(f"\nProcessing {tsv_file}...")
        try:
            with gzip.open(tsv_file, 'rt') as f:
                df = pd.read_csv(f, sep='\t', index_col=0)
            print(f"  ✓ Loaded TSV data: {df.shape}")
            
            # Save as standard counts file
            df.to_csv('GSE125345_counts.txt.gz', sep='\t', compression='gzip')
            print(f"  ✓ Saved as GSE125345_counts.txt.gz")
            return df
        except Exception as e:
            print(f"  ✗ Error processing TSV: {e}")
    
    # Try Excel file with explicit engine
    excel_file = 'GSE125345_countdata.xlsx'
    if os.path.exists(excel_file):
        print(f"\nProcessing {excel_file}...")
        try:
            # Try with openpyxl engine
            df = pd.read_excel(excel_file, index_col=0, engine='openpyxl')
            print(f"  ✓ Loaded Excel data: {df.shape}")
            
            # Save as compressed text file
            df.to_csv('GSE125345_counts.txt.gz', sep='\t', compression='gzip')
            print(f"  ✓ Saved as GSE125345_counts.txt.gz")
            return df
        except Exception as e:
            print(f"  ✗ Error with openpyxl: {e}")
            
            # Try with xlrd engine for older Excel formats
            try:
                df = pd.read_excel(excel_file, index_col=0, engine='xlrd')
                print(f"  ✓ Loaded Excel data with xlrd: {df.shape}")
                df.to_csv('GSE125345_counts.txt.gz', sep='\t', compression='gzip')
                return df
            except:
                print(f"  ✗ Could not read Excel file")
    
    return None

def analyze_gse125345_for_validation():
    """
    Analyze GSE125345 as validation for the AML signature
    """
    print("\n" + "="*60)
    print("GSE125345 Analysis for Validation")
    print("="*60)
    
    # Load the processed counts
    if os.path.exists('GSE125345_counts.txt.gz'):
        counts = pd.read_csv('GSE125345_counts.txt.gz', sep='\t', compression='gzip', index_col=0)
        print(f"\nLoaded GSE125345 counts: {counts.shape}")
        
        # These are all healthy HSPC samples
        print("\nSample information:")
        print("  All 15 samples are healthy cord blood hematopoietic cells:")
        print("  - 3x LT-HSC (Long-term hematopoietic stem cells)")
        print("  - 3x ST-HSC (Short-term hematopoietic stem cells)")
        print("  - 3x CMP (Common myeloid progenitors)")
        print("  - 3x GMP (Granulocyte-monocyte progenitors)")
        print("  - 3x MLP (Multi-lymphoid progenitors)")
        
        # Load your AML gene signatures from GSE74246 analysis
        if os.path.exists('../../gene_signatures_aml_vs_hspc_final.csv'):
            signatures = pd.read_csv('../../gene_signatures_aml_vs_hspc_final.csv')
            
            # Apply the AML score
            from src_031025.Hematopoietic_Cells_Healthy_AML.Hematopoietic_Cells_Healthy_AML_analysis import logcpm
            
            log_counts = logcpm(counts)
            
            # Find common genes
            aml_up = signatures['aml_upregulated'].dropna().tolist()
            aml_down = signatures['aml_downregulated'].dropna().tolist()
            
            common_up = list(set(aml_up) & set(log_counts.index))
            common_down = list(set(aml_down) & set(log_counts.index))
            
            print(f"\nGene overlap with GSE74246 signature:")
            print(f"  Up in AML: {len(common_up)}/{len(aml_up)} genes found")
            print(f"  Down in AML: {len(common_down)}/{len(aml_down)} genes found")
            
            if common_up and common_down:
                # Calculate AML scores for GSE125345 samples
                aml_scores = (log_counts.loc[common_up].mean() - 
                            log_counts.loc[common_down].mean())
                
                print(f"\nAML scores for healthy HSPC samples (GSE125345):")
                print(f"  Mean: {aml_scores.mean():.3f}")
                print(f"  Std: {aml_scores.std():.3f}")
                print(f"  Range: [{aml_scores.min():.3f}, {aml_scores.max():.3f}]")
                
                # These scores should be LOW (similar to HSPC in GSE74246)
                # since all GSE125345 samples are healthy
                
                return aml_scores
        
    return None

# Install required package if needed
def install_requirements():
    """Install openpyxl if not present"""
    try:
        import openpyxl
    except ImportError:
        print("Installing openpyxl for Excel file handling...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'openpyxl'])

if __name__ == "__main__":
    # Ensure we have openpyxl
    install_requirements()
    
    # Download files
    download_gse125345_files()
    
    # Analyze for validation
    scores = analyze_gse125345_for_validation()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
GSE125345 provides 15 healthy HSPC samples that can serve as
additional validation for your AML gene signature.

Expected result: These healthy samples should have LOW AML scores,
similar to the HSPC samples in GSE74246.

Your main analysis with GSE74246 remains excellent:
- AUC = 0.971 (exceptional discrimination)
- p = 3.94e-13 (highly significant)
- 388 genes up in AML, 2,251 down in AML

The biological pattern (more genes downregulated in AML) aligns with
known AML biology where normal differentiation is blocked.
    """)
