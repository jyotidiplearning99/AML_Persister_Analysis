#!/bin/bash
# Download scRNA-seq data for 3 priority cell lines (fastest path to results)

OUTPUT_DIR="/scratch/project_2010376/JDs_Project/scrna_data"
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

echo "Downloading scRNA-seq data for 3 priority cell lines..."
echo "================================================================"

# 1. MOLM-13 (GSE300177)
echo -e "\n1. MOLM-13 (GSE300177)..."
wget -q --show-progress "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE300177&format=file" -O MOLM13_GSE300177.tar || echo "Failed - try manual"

# 2. MV4-11 (GSE228326) 
echo -e "\n2. MV4-11 (GSE228326)..."
wget -q --show-progress "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE228326&format=file" -O MV411_GSE228326.tar || echo "Failed - try manual"

# 3. THP-1 (GSE302068)
echo -e "\n3. THP-1 (GSE302068)..."
wget -q --show-progress "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE302068&format=file" -O THP1_GSE302068.tar || echo "Failed - try manual"

echo -e "\nDone! Check downloaded files and extract H5/MTX files."
