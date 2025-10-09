#!/bin/bash
# check_missing_samples_v2.sh

# All 72 samples from your Excel
ALL_SAMPLES=(
    "FH_2030" "FH_209" "FH_3081" "FH_4485" "FH_4599" "FH_4634"
    "FH_4739" "FH_5143" "FH_5238" "FH_5249" "FH_5750" "FH_5897"
    "FH_6088" "FH_6187" "FH_6333" "FH_6386" "FH_6412" "FH_6753"
    "FH_8111" "FH_8445" "FH_8668" "FHRB_1216" "FHRB_1280" "FHRB_1574"
    "FHRB_1886" "FHRB_252" "FHRB_3443" "FHRB_3579" "FHRB_3660"
    "FHRB_3667" "FHRB_3708" "FHRB_40438" "FHRB_40767" "FHRB_4697"
    "FHRB_5199" "FHRB_560" "FHRB_743" "FHRB_MV_54" "FHRB_MV_57"
    "FHRB_MV_58" "FHRB_MV_60" "FHRB_MV_67" "FHRB_MV_69" "FHRB_MV_71"
    "FHRB_MV_76" "VX_11" "VX_3" "VX_40" "VX_41" "VX_77" "VX_9"
)

# Samples you have locally (based on your discovery)
AVAILABLE=(
    "FH_5143"
    "FH_6088"  # you have both _2 and _3
    "FH_5238"
    "FH_6333"
)

echo "=== Checking for missing samples ==="
echo "Total samples needed: ${#ALL_SAMPLES[@]}"
echo ""

MISSING=()
for sample in "${ALL_SAMPLES[@]}"; do
    found=false
    for available in "${AVAILABLE[@]}"; do
        if [[ "$sample" == "$available" ]]; then
            found=true
            break
        fi
    done
    
    if [ "$found" = false ]; then
        MISSING+=("$sample")
    fi
done

echo "Found locally: ${#AVAILABLE[@]} samples"
echo "Missing: ${#MISSING[@]} samples"
echo ""
echo "=== Missing samples list ==="
printf '%s\n' "${MISSING[@]}" | tee missing_samples.txt

# Count by type
echo ""
echo "=== Missing samples by prefix ==="
echo "FH samples missing: $(printf '%s\n' "${MISSING[@]}" | grep -c '^FH_')"
echo "FHRB samples missing: $(printf '%s\n' "${MISSING[@]}" | grep -c '^FHRB_')"  
echo "VX samples missing: $(printf '%s\n' "${MISSING[@]}" | grep -c '^VX_')"
