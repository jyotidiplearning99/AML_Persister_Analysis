#!/bin/bash
# download_loom_samples.sh
cd /scratch/project_2010751/AML_scRNA_decrypted
mkdir -p ./loom_encrypted
mkdir -p ./loom_decrypted
mkdir -p ./converted_10x

echo "Downloading 11 loom files for 6 samples..."

# Download all available loom files
swift download Acute_Leukemia "AML_Leukemogenesis/Batch1_160221/Output_210301_A00464_0292_AH3YTYDRXY/FH_6753_3/FH_6753_3.cells.loom.c4gh" --output-dir ./loom_encrypted/
swift download Acute_Leukemia "AML_Leukemogenesis/Batch2-8_040321-180521/Output_210601_A00464_0340_BHVN5KDMXX/FH_6753_2/results/FH_6753_2.cells.loom.c4gh" --output-dir ./loom_encrypted/
swift download Acute_Leukemia "AML_Leukemogenesis/Batch2-8_040321-180521/Output_210601_A00464_0340_BHVN5KDMXX/FHRB_1886_10/results/FHRB_1886_10.cells.loom.c4gh" --output-dir ./loom_encrypted/
swift download Acute_Leukemia "AML_Leukemogenesis/Batch2-8_040321-180521/Output_210601_A00464_0340_BHVN5KDMXX/FHRB_1886_6/results/FHRB_1886_6.cells.loom.c4gh" --output-dir ./loom_encrypted/
swift download Acute_Leukemia "AML_Leukemogenesis/Batch2-8_040321-180521/Output_210601_A00464_0340_BHVN5KDMXX/FHRB_3443_2/results/FHRB_3443_2.cells.loom.c4gh" --output-dir ./loom_encrypted/
swift download Acute_Leukemia "AML_Leukemogenesis/Batch2-8_040321-180521/Output_210601_A00464_0340_BHVN5KDMXX/FHRB_4697_2/results/FHRB_4697_2.cells.loom.c4gh" --output-dir ./loom_encrypted/
swift download Acute_Leukemia "AML_Leukemogenesis/Batch2-8_040321-180521/Output_210601_A00464_0340_BHVN5KDMXX/FHRB_4697_3/results/FHRB_4697_3.cells.loom.c4gh" --output-dir ./loom_encrypted/
swift download Acute_Leukemia "AML_Leukemogenesis/Batch2-8_040321-180521/Output_210601_A00464_0340_BHVN5KDMXX/VX_11_2/results/VX_11_2.cells.loom.c4gh" --output-dir ./loom_encrypted/
swift download Acute_Leukemia "AML_Leukemogenesis/Batch2-8_040321-180521/Output_210601_A00464_0340_BHVN5KDMXX/VX_11_6/results/VX_11_6.cells.loom.c4gh" --output-dir ./loom_encrypted/
swift download Acute_Leukemia "AML_Leukemogenesis/Batch2-8_040321-180521/Output_210601_A00464_0340_BHVN5KDMXX/VX_3_2/results/VX_3_2.cells.loom.c4gh" --output-dir ./loom_encrypted/
swift download Acute_Leukemia "AML_Leukemogenesis/Batch2-8_040321-180521/Output_210601_A00464_0340_BHVN5KDMXX/VX_3_6/results/VX_3_6.cells.loom.c4gh" --output-dir ./loom_encrypted/

echo "Download complete!"
ls -lh ./loom_encrypted/