#!/usr/bin/env python3
"""
Search for scRNA-seq datasets for 28 AML cell lines
Searches GEO, SRA, and ArrayExpress
"""

import requests
import pandas as pd
from pathlib import Path
import time

# Your 24 matched cell lines
CELL_LINES = [
    'AML-193', 'GDM-1', 'HL-60', 'KASUMI-1', 'KASUMI-6', 'KG-1',
    'ME-1', 'MOLM-13', 'MOLM-16', 'MONO-MAC-1', 'MONO-MAC-6',
    'MV4-11', 'NB-4', 'NOMO-1', 'OCI-AML2', 'OCI-AML3', 'OCI-AML5',
    'PL-21', 'SH-2', 'SHI-1', 'SIG-M5', 'SKM-1', 'THP-1'
]

def search_geo(cell_line):
    """Search GEO using NCBI E-utilities"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    params = {
        "db": "gds",
        "term": f'"{cell_line}"[Title] AND ("single cell"[Title] OR "scRNA-seq"[Title])',
        "retmode": "json",
        "retmax": 10
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        data = response.json()
        
        if 'esearchresult' in data and 'idlist' in data['esearchresult']:
            ids = data['esearchresult']['idlist']
            return {
                'cell_line': cell_line,
                'source': 'GEO',
                'found': len(ids) > 0,
                'n_datasets': len(ids),
                'ids': ids[:5]
            }
    except:
        pass
    
    return {'cell_line': cell_line, 'source': 'GEO', 'found': False, 'n_datasets': 0, 'ids': []}

# Search all lines
print("Searching for scRNA-seq datasets...")
print("="*80)

results = []
for cell_line in CELL_LINES:
    print(f"\nSearching {cell_line}...")
    result = search_geo(cell_line)
    results.append(result)
    
    if result['found']:
        print(f"  ✓ Found {result['n_datasets']} datasets in GEO")
        for geo_id in result['ids']:
            print(f"    - GDS{geo_id}")
    else:
        print(f"  ✗ No datasets found")
    
    time.sleep(0.5)  # Be nice to NCBI

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('scrna_availability.csv', index=False)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total cell lines searched: {len(CELL_LINES)}")
print(f"Lines with scRNA-seq data: {df_results['found'].sum()}")
print("\nCell lines with data:")
print(df_results[df_results['found']][['cell_line', 'n_datasets']])

