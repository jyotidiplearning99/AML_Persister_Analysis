import os
import pandas as pd
import numpy as np
import requests
import gzip
import shutil
from urllib.request import urlretrieve

# Create directory for data storage
os.makedirs('data/AML_datasets', exist_ok=True)

# ============================================
# 1. Direct download of GSE74246 series-level data
# ============================================
print("Downloading GSE74246 series-level supplementary files...")

# Direct URLs for GSE74246 supplementary files
gse74246_urls = {
    'counts': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74246/suppl/GSE74246_RNAseq_All_counts.txt.gz',
    'rpkm': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74246/suppl/GSE74246_RNAseq_All_RPKM.txt.gz'
}

for data_type, url in gse74246_urls.items():
    output_file = f'data/AML_datasets/GSE74246_{data_type}.txt.gz'
    try:
        print(f"Downloading {data_type} data...")
        urlretrieve(url, output_file)
        
        # Decompress the file
        with gzip.open(output_file, 'rb') as f_in:
            with open(output_file.replace('.gz', ''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ {data_type} data downloaded and decompressed")
        
        # Load and check the data
        if data_type == 'counts':
            expression_74246 = pd.read_csv(output_file.replace('.gz', ''), sep='\t', index_col=0)
            print(f"  GSE74246 expression shape: {expression_74246.shape}")
            expression_74246.to_csv('data/AML_datasets/GSE74246_expression.csv')
            
    except Exception as e:
        print(f"✗ Error downloading {data_type}: {e}")

# ============================================
# 2. Download GSE125345 series-level data
# ============================================
print("\nDownloading GSE125345 series-level supplementary files...")

# Direct URL for GSE125345
gse125345_url = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE125nnn/GSE125345/suppl/GSE125345_human_cord_blood_counts_normalized.csv.gz'

try:
    output_file = 'data/AML_datasets/GSE125345_counts.csv.gz'
    print("Downloading normalized count data...")
    urlretrieve(gse125345_url, output_file)
    
    # Decompress
    with gzip.open(output_file, 'rb') as f_in:
        with open(output_file.replace('.gz', ''), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Load and check
    count_matrix_125345 = pd.read_csv(output_file.replace('.gz', ''), index_col=0)
    print(f"✓ GSE125345 count matrix shape: {count_matrix_125345.shape}")
    
except Exception as e:
    print(f"✗ Error with main file, trying alternative: {e}")
    
    # Try alternative URL for raw data
    alt_url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE125345&format=file&file=GSE125345%5FRAW%2Etar'
    try:
        import tarfile
        tar_file = 'data/AML_datasets/GSE125345_RAW.tar'
        urlretrieve(alt_url, tar_file)
        
        # Extract tar file
        with tarfile.open(tar_file, 'r') as tar:
            tar.extractall('data/AML_datasets/GSE125345/')
        print("✓ Downloaded and extracted RAW data")
        
        # Process individual files
        import glob
        gz_files = glob.glob('data/AML_datasets/GSE125345/*.gz')
        for gz_file in gz_files[:3]:  # Process first 3 files as example
            with gzip.open(gz_file, 'rt') as f:
                sample_data = pd.read_csv(f, sep='\t', index_col=0, nrows=5)
                print(f"  Sample file shape: {sample_data.shape}")
                
    except Exception as e2:
        print(f"✗ Alternative download also failed: {e2}")

# ============================================
# 3. Load metadata using GEOparse (this worked fine)
# ============================================
print("\nLoading metadata...")
import GEOparse

# For metadata, we can still use GEOparse
gse74246 = GEOparse.get_GEO(geo="GSE74246", destdir="./data/AML_datasets/", silent=True)
metadata_74246 = gse74246.phenotype_data
metadata_74246.to_csv('data/AML_datasets/GSE74246_metadata.csv')
print(f"GSE74246 metadata saved: {metadata_74246.shape}")

gse125345 = GEOparse.get_GEO(geo="GSE125345", destdir="./data/AML_datasets/", silent=True)
metadata_125345 = gse125345.phenotype_data
metadata_125345.to_csv('data/AML_datasets/GSE125345_metadata.csv')
print(f"GSE125345 metadata saved: {metadata_125345.shape}")

# ============================================
# 4. ArrayExpress data - Alternative download
# ============================================
print("\nDownloading ArrayExpress data via HTTPS...")

arrayexpress_urls = {
    'E-MTAB-5456': {
        'processed': 'https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-5456/E-MTAB-5456.processed.1.zip',
        'sdrf': 'https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-5456/E-MTAB-5456.sdrf.txt'
    }
}

for accession, urls in arrayexpress_urls.items():
    os.makedirs(f'data/AML_datasets/{accession}', exist_ok=True)
    
    for file_type, url in urls.items():
        try:
            output_file = f'data/AML_datasets/{accession}/{os.path.basename(url)}'
            print(f"Downloading {accession} {file_type}...")
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                # If it's a zip file, extract it
                if output_file.endswith('.zip'):
                    import zipfile
                    with zipfile.ZipFile(output_file, 'r') as zip_ref:
                        zip_ref.extractall(f'data/AML_datasets/{accession}/')
                    print(f"✓ {file_type} downloaded and extracted")
                else:
                    print(f"✓ {file_type} downloaded")
            else:
                print(f"✗ HTTP {response.status_code} for {file_type}")
                
        except Exception as e:
            print(f"✗ Error downloading {file_type}: {e}")
