#!/usr/bin/env python3
"""
File availability checker for CNN-RNN Pipeline
"""
import os
from pathlib import Path
from datetime import datetime

print("=" * 80)
print(f"FILE AVAILABILITY CHECK - {datetime.now()}")
print("=" * 80)

# Define all required files
required_files = {
    "Core Training Data": [
        ("/scratch/project_2010376/GSE150949_metaData_with_lineage.txt", "CRITICAL"),
        ("/scratch/project_2010376/normalized_GSE150949_pc9_count.csv", "CRITICAL"),
        ("/scratch/project_2010376/GSE150949_pc9_count_matrix.csv", "REQUIRED"),
    ],
    "Additional Training Data": [
        ("/scratch/project_2010751/GSM8118468_ob_treated.csv", "MISSING - Commented out"),
        ("/scratch/project_2010751/GSE134836_GSM3972651_PC9D0_untreated_filtered.csv", "REQUIRED"),
        ("/scratch/project_2010751/GSE138693_GSM4116265_PC9_1_invitro_normalized_untreated.csv", "REQUIRED"),
        ("/scratch/project_2010751/GSE134839_GSM3972657_PC90D0_untreated.dge.csv", "REQUIRED"),
    ],
    "Independent Test Data": [
        ("/scratch/project_2010376/GSM4869650_xCtrl.dge.csv", "OPTIONAL"),
        ("/scratch/project_2010376/new_GSM4869650_xCtrl.dge.csv", "OPTIONAL"),
        ("/scratch/project_2010751/GSE149383_GSM3972669_D0_untreated.dge.csv", "OPTIONAL"),
        ("/scratch/project_2010751/GSE160244_GSM4869650_day3_untreated.dge.csv", "OPTIONAL"),
        ("/scratch/project_2010751/GSE160244_GSM4869652_xOsi_day3_dge.csv", "OPTIONAL"),
        ("/scratch/project_2010751/normalized_GSE150949_pc9_count.csv", "OPTIONAL"),
        ("/scratch/project_2010751/GSM4869650_xCtrl.dge.csv", "OPTIONAL"),
        ("/scratch/project_2010751/GSM4869653_xOsiCriz.dge.csv", "OPTIONAL"),
    ],
    "GSE123902 Data": [
        ("/scratch/project_2010751/GSE123902_RAW/GSM3516666_MSK_LX675_NORMAL_dense.csv", "OPTIONAL"),
        ("/scratch/project_2010751/GSE123902_RAW/GSM3516677_MSK_LX699_METASTASIS_dense.csv", "OPTIONAL"),
    ],
    "10x Genomics Data": [
        ("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/FH_5897_2/filtered_feature_bc_matrix", "10X_DIR"),
        ("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/FH_6333_2/filtered_feature_bc_matrix", "10X_DIR"),
        ("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/FH_6753_2/filtered_feature_bc_matrix", "10X_DIR"),
    ]
}

# Check files
critical_missing = []
required_missing = []
found_count = 0
missing_count = 0

for category, files in required_files.items():
    print(f"\n{category}:")
    print("-" * 60)
    
    for file_path, priority in files:
        path = Path(file_path)
        
        if priority == "10X_DIR":
            if path.exists() and path.is_dir():
                # Check for required 10x files
                mtx = path / "matrix.mtx.gz"
                features = path / "features.tsv.gz"
                barcodes = path / "barcodes.tsv.gz"
                
                if mtx.exists() and features.exists() and barcodes.exists():
                    print(f"  ✅ {path.name} - Complete 10x dataset")
                    found_count += 1
                else:
                    missing_parts = []
                    if not mtx.exists(): missing_parts.append("matrix.mtx.gz")
                    if not features.exists(): missing_parts.append("features.tsv.gz")
                    if not barcodes.exists(): missing_parts.append("barcodes.tsv.gz")
                    print(f"  ⚠️  {path.name} - Missing: {', '.join(missing_parts)}")
                    missing_count += 1
            else:
                print(f"  ❌ {path.name} - Directory not found")
                missing_count += 1
                if priority == "CRITICAL":
                    critical_missing.append(file_path)
        else:
            if path.exists():
                size_mb = path.stat().st_size / (1024**2)
                print(f"  ✅ {path.name} ({size_mb:.1f} MB) [{priority}]")
                found_count += 1
            else:
                print(f"  ❌ {path.name} [{priority}]")
                missing_count += 1
                
                if priority == "CRITICAL":
                    critical_missing.append(file_path)
                elif priority == "REQUIRED":
                    required_missing.append(file_path)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Files found: {found_count}")
print(f"Files missing: {missing_count}")

if critical_missing:
    print(f"\n🔴 CRITICAL FILES MISSING ({len(critical_missing)}):")
    for f in critical_missing:
        print(f"  - {f}")
    print("\n⚠️  Cannot run training without critical files!")
elif required_missing:
    print(f"\n🟡 REQUIRED FILES MISSING ({len(required_missing)}):")
    for f in required_missing:
        print(f"  - {f}")
    print("\n⚠️  Training will work but with reduced data")
else:
    print("\n✅ All critical and required files are present!")
    print("Ready to run the pipeline!")

# Check for existing models/artifacts
artifacts_dir = Path("/scratch/project_2010376/scRNAseq/artifacts")
if artifacts_dir.exists():
    print(f"\n📁 Artifacts directory exists: {artifacts_dir}")
    for f in artifacts_dir.glob("*"):
        print(f"  - {f.name}")
else:
    print(f"\n📁 Artifacts directory not found - will be created during training")

