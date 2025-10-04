#!/usr/bin/env python3
"""
Debug script to examine GEO file structures
"""

import gzip
import pandas as pd

# Paths
GSE74246_PATH = '/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025/Hematopoietic_Cells_Healthy_AML/GSE74246/'
GSE125345_PATH = '/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025/Hematopoietic_Cells_Healthy_AML/GSE125345/'

def examine_series_matrix(filepath):
    """Examine series matrix file structure"""
    print(f"\n{'='*60}")
    print(f"Examining: {filepath}")
    print('='*60)
    
    try:
        with gzip.open(filepath, 'rt') as f:
            line_count = 0
            for line in f:
                if line_count < 50 or line.startswith('!'):  # Show first 50 lines or metadata lines
                    if line.startswith('!'):
                        print(f"Line {line_count}: {line[:100]}...")
                line_count += 1
                if line_count == 100:  # Stop after 100 lines for inspection
                    break
            print(f"\nTotal lines examined: {line_count}")
    except Exception as e:
        print(f"Error reading file: {e}")

def examine_counts_file(filepath):
    """Examine counts file structure"""
    print(f"\n{'='*60}")
    print(f"Examining: {filepath}")
    print('='*60)
    
    try:
        # Try reading first few lines
        with gzip.open(filepath, 'rt') as f:
            for i in range(5):
                line = f.readline()
                print(f"Line {i}: {line[:200]}...")
        
        # Try reading with pandas
        print("\nTrying pandas read_csv with different parameters...")
        
        # Try tab-separated
        try:
            df = pd.read_csv(filepath, sep='\t', compression='gzip', nrows=5)
            print(f"Tab-separated worked! Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()[:5]}...")
        except Exception as e:
            print(f"Tab-separated failed: {e}")
        
        # Try comma-separated
        try:
            df = pd.read_csv(filepath, sep=',', compression='gzip', nrows=5)
            print(f"Comma-separated worked! Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()[:5]}...")
        except Exception as e:
            print(f"Comma-separated failed: {e}")
            
    except Exception as e:
        print(f"Error examining file: {e}")

# Run examinations
print("EXAMINING GSE74246 FILES")
examine_series_matrix(f'{GSE74246_PATH}GSE74246_series_matrix.txt.gz')
examine_counts_file(f'{GSE74246_PATH}GSE74246_RNAseq_All_Counts.txt')

print("\n" + "="*80)
print("EXAMINING GSE125345 FILES")
examine_series_matrix(f'{GSE125345_PATH}GSE125345_series_matrix.txt.gz')
examine_counts_file(f'{GSE125345_PATH}GSE125345_counts.txt.gz')
