#!/bin/bash
set -euo pipefail

WORKDIR=/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025
BASE=${WORKDIR}/data/GSE146590
mkdir -p ${BASE}/AML1566_Ctrl ${BASE}/AML1566_AraC

# Processed 10x files from GEO sample pages GSM4396313 and GSM4396314.
wget -c -O ${BASE}/AML1566_Ctrl/barcodes.tsv.gz 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4396313&format=file&file=GSM4396313_Ctrl_barcodes.tsv.gz'
wget -c -O ${BASE}/AML1566_Ctrl/features.tsv.gz 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4396313&format=file&file=GSM4396313_Ctrl_features.tsv.gz'
wget -c -O ${BASE}/AML1566_Ctrl/matrix.mtx.gz   'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4396313&format=file&file=GSM4396313_Ctrl_matrix.mtx.gz'

wget -c -O ${BASE}/AML1566_AraC/barcodes.tsv.gz 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4396314&format=file&file=GSM4396314_AraC_barcodes.tsv.gz'
wget -c -O ${BASE}/AML1566_AraC/features.tsv.gz 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4396314&format=file&file=GSM4396314_AraC_features.tsv.gz'
wget -c -O ${BASE}/AML1566_AraC/matrix.mtx.gz   'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4396314&format=file&file=GSM4396314_AraC_matrix.mtx.gz'

ls -lh ${BASE}/AML1566_Ctrl ${BASE}/AML1566_AraC
