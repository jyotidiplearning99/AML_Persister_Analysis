#!/bin/bash

echo "Searching for AML scRNA-seq samples..."
echo "======================================"

AML_ROOT="/scratch/project_2010751/AML_scRNA_decrypted"

# Find all filtered_feature_bc_matrix folders
find "$AML_ROOT" -type d -name "filtered_feature_bc_matrix" | while read matrix_dir; do
    # Extract patient ID by going up directories
    parent=$(dirname "$matrix_dir")
    
    # Skip generic folder names
    while [[ "$(basename $parent)" =~ ^(outs|count|filtered_feature_bc_matrix)$ ]]; do
        parent=$(dirname "$parent")
    done
    
    patient_id=$(basename "$parent")
    echo "$patient_id → $matrix_dir"
done
