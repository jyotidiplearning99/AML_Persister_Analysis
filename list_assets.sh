#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-/scratch/project_2010376/JDs_Project/AML_Persister_Analysis}"
OUT="${2:-$DIR/AML_assets_$(date +%Y-%m-%d_%H%M).csv}"
HASH="${HASH:-0}"  # set HASH=1 to include sha256 (slower)

echo "path,ext,size_bytes,mtime_iso$([ "$HASH" -eq 1 ] && echo ",sha256")" > "$OUT"

find "$DIR" -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.csv' -o -iname '*.txt' \) -print0 |
while IFS= read -r -d '' f; do
  ext="${f##*.}"; ext="${ext,,}"
  size=$(stat -c %s "$f")
  mtime=$(date -d "@$(stat -c %Y "$f")" --iso-8601=seconds)
  if [ "$HASH" -eq 1 ]; then
    sha=$(sha256sum "$f" | awk '{print $1}')
    printf '"%s",%s,%s,%s,%s\n' "${f//\"/\"\"}" "$ext" "$size" "$mtime" "$sha" >> "$OUT"
  else
    printf '"%s",%s,%s,%s\n' "${f//\"/\"\"}" "$ext" "$size" "$mtime" >> "$OUT"
  fi
done

echo "Wrote $OUT"
