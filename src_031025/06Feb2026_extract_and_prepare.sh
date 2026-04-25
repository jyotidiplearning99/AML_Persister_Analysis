#!/bin/bash
# Simple extraction - just extract everything

cd /scratch/project_2010376/JDs_Project/scrna_data

echo "Extracting TAR files..."
echo "================================================================"

# Extract MV4-11 (remove -z flag since it's .tar not .tar.gz)
echo "Extracting MV411_GSE228326.tar..."
tar -xf MV411_GSE228326.tar
echo "✓ Extracted"

# List what was extracted
echo ""
echo "Extracted files:"
ls -lh GSM7118799* | head -5

echo ""
echo "================================================================"
echo "Done! Files are now in current directory"
echo "================================================================"
