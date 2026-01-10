#!/usr/bin/env python3
from pathlib import Path

AML_ROOT = Path("/scratch/project_2010751/AML_scRNA_decrypted")

print("Samples in scRNA-seq directory:")
print("="*80)

for p in AML_ROOT.rglob("filtered_feature_bc_matrix"):
    if p.is_dir():
        # Walk up to find patient ID
        parent = p.parent
        skip_names = {"outs", "count", "filtered_feature_bc_matrix"}
        
        while parent.name in skip_names and parent != AML_ROOT:
            parent = parent.parent
        
        sample_id = parent.name
        print(f"{sample_id:30s} → {p}")

print("\n" + "="*80)
print("Clinical table expects these IDs:")
print("="*80)

clinical_ids = [
    'FHRB_706','FHRB_188','FHRB_436','FHRB_560','FHRB_279',
    'FHRB_268','FHRB_437','FHRB_252','FHRB_434','FHRB_382',
    'FHRB_106','FHRB_121','FHRB_209','FHRB_366','FHRB_139',
    'FHRB_600','FHRB_393','FHRB_743','FHRB_349','FH_4991_',
    'FH_5713_','FH_3081_','FH_4599_','FH_5034_','FH_5184_',
    'FHRB_468','FH_5776_','FH_6088_','FH_5897_','FH_6310_',
    'FH_6323_','FH_6389_','FH_6512_','FH_6532_','FH_6545_',
    'FH_6565_','FH_6576_','FH_6525_','FH_6940_','FH_7087_',
    'FH_6810_','FH_7289_'
]

for cid in sorted(clinical_ids):
    print(f"  {cid}")
