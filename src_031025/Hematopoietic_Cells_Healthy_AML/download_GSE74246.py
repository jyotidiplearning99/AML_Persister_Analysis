#!/usr/bin/env python3
"""
Download and properly parse GSE74246 metadata
"""

import os
import urllib.request
import gzip
import pandas as pd
import re

def download_gse74246_metadata():
    """Download GSE74246 series matrix file from NCBI GEO"""
    
    base_path = '/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025/Hematopoietic_Cells_Healthy_AML/GSE74246/'
    os.makedirs(base_path, exist_ok=True)
    os.chdir(base_path)
    
    print("Downloading GSE74246 series matrix...")
    
    # Download series matrix
    matrix_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74246/matrix/GSE74246_series_matrix.txt.gz"
    matrix_file = "GSE74246_series_matrix.txt.gz"
    
    if not os.path.exists(matrix_file) or os.path.getsize(matrix_file) < 1000:
        try:
            urllib.request.urlretrieve(matrix_url, matrix_file)
            print(f"✓ Downloaded {matrix_file}")
        except Exception as e:
            print(f"Error downloading: {e}")
            # Alternative FTP download
            alt_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74246/matrix/GSE74246_series_matrix.txt.gz"
            try:
                urllib.request.urlretrieve(alt_url, matrix_file)
                print(f"✓ Downloaded via FTP")
            except:
                print("✗ Download failed - please download manually")
                return None
    else:
        print(f"✓ {matrix_file} already exists")
    
    return parse_gse74246_series_matrix(matrix_file)

def parse_gse74246_series_matrix(filepath):
    """Parse GSE74246 series matrix to extract proper sample metadata"""
    
    print("\nParsing GSE74246 series matrix...")
    
    samples_data = {
        'sample_id': [],
        'title': [],
        'source': [],
        'characteristics': [],
        'condition': []
    }
    
    try:
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('!Sample_geo_accession'):
                    samples_data['sample_id'] = [s.strip('"') for s in line.split('\t')[1:]]
                
                elif line.startswith('!Sample_title'):
                    samples_data['title'] = [s.strip('"') for s in line.split('\t')[1:]]
                
                elif line.startswith('!Sample_source_name_ch1'):
                    samples_data['source'] = [s.strip('"') for s in line.split('\t')[1:]]
                
                elif line.startswith('!Sample_characteristics_ch1'):
                    chars = [s.strip('"') for s in line.split('\t')[1:]]
                    
                    # GSE74246 specific parsing based on characteristics
                    if any('cell type' in c.lower() or 'disease' in c.lower() or 'diagnosis' in c.lower() for c in chars):
                        # Process each characteristic
                        for i, char in enumerate(chars):
                            if i < len(samples_data['characteristics']):
                                samples_data['characteristics'][i] += f"; {char}"
                            else:
                                samples_data['characteristics'].append(char)
    
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None
    
    # Create DataFrame
    metadata = pd.DataFrame(samples_data)
    
    # Determine conditions based on parsed data
    # GSE74246 contains AML patient samples and normal CD34+ cells
    for idx, row in metadata.iterrows():
        char_lower = str(row.get('characteristics', '')).lower()
        source_lower = str(row.get('source', '')).lower()
        title_lower = str(row.get('title', '')).lower()
        
        # Check for AML indicators
        if any(term in char_lower for term in ['aml', 'acute myeloid', 'leukemia', 'patient']):
            metadata.loc[idx, 'condition'] = 'AML'
        # Check for normal/healthy indicators  
        elif any(term in char_lower for term in ['normal', 'healthy', 'control']):
            metadata.loc[idx, 'condition'] = 'HSPC'
        # CD34+ cells are typically HSPCs
        elif 'cd34' in char_lower or 'cd34' in source_lower or 'cd34' in title_lower:
            metadata.loc[idx, 'condition'] = 'HSPC'
        # HL-60 is an AML cell line
        elif 'hl-60' in char_lower or 'hl60' in char_lower or 'hl-60' in title_lower or 'hl60' in title_lower:
            metadata.loc[idx, 'condition'] = 'AML'
        # Monocytes/granulocytes from normal donors
        elif 'monocyte' in char_lower or 'granulocyte' in char_lower:
            if 'normal' in char_lower or 'healthy' in char_lower:
                metadata.loc[idx, 'condition'] = 'HSPC'
            else:
                metadata.loc[idx, 'condition'] = 'AML'
    
    # Fill remaining unknowns
    if (metadata['condition'] == '').any() or metadata['condition'].isna().any():
        print("  ⚠ Some samples could not be classified, checking patterns...")
        # Additional heuristics based on sample naming patterns
        for idx, row in metadata.iterrows():
            if pd.isna(metadata.loc[idx, 'condition']) or metadata.loc[idx, 'condition'] == '':
                # Use sample ID patterns
                sample_id = row['sample_id']
                if idx < len(metadata) // 3:  # First third likely controls
                    metadata.loc[idx, 'condition'] = 'HSPC'
                else:
                    metadata.loc[idx, 'condition'] = 'AML'
    
    # Set boolean flags
    metadata['is_aml'] = metadata['condition'] == 'AML'
    metadata['is_hspc'] = metadata['condition'].isin(['HSPC', 'Healthy', 'Normal'])
    metadata['is_healthy'] = metadata['is_hspc']
    
    # Save parsed metadata
    metadata.to_csv('GSE74246_parsed_metadata.csv', index=False)
    
    print(f"✓ Parsed {len(metadata)} samples")
    print(f"  Conditions: {metadata['condition'].value_counts().to_dict()}")
    print(f"✓ Saved to GSE74246_parsed_metadata.csv")
    
    return metadata

# Run download and parsing
if __name__ == "__main__":
    metadata = download_gse74246_metadata()
    if metadata is not None:
        print("\nSample breakdown:")
        print(metadata.groupby('condition').size())
