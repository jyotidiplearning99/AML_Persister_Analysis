"""
hpa_annotate.py
Annotate candidate target tables (S7, S8) with HPA protein-level information.

Outputs:
  Supplementary_Table_S7_top30_targets_HPA.csv
  Supplementary_Table_S8_candidate_ranking_HPA.csv

Usage:
  python hpa_annotate.py \
      --s7 Supplementary_Table_S7_top30_targets.csv \
      --s8 Supplementary_Table_S8_candidate_ranking.csv \
      --outdir ./output
"""

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

HPA_TSV_URL = "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"
HPA_LOCAL = "proteinatlas.tsv"

# HPA columns we want to keep (names match the current HPA download header)
HPA_COLS_OF_INTEREST = [
    "Gene",
    "Ensembl",
    "Gene synonym",
    "RNA tissue specificity",
    "RNA tissue distribution",
    "Tissue expression cluster",
    "Antibody",
    "Reliability (IH)",                # immunohistochemistry reliability
    "Reliability (IF)",                # immunofluorescence reliability
    "Subcellular location",
    "Secretome location",
    "Secretome function",
    "Protein class",
    "Blood concentration - Conc. blood IM [pg/L]",
    "RNA tissue cell type enrichment",
]


def download_hpa(local_path: str = HPA_LOCAL) -> str:
    """Download HPA tsv (zipped) if not already present, return local path."""
    if os.path.exists(local_path):
        print(f"[ok] HPA file already present at {local_path}")
        return local_path

    print(f"[..] Downloading HPA file from {HPA_TSV_URL} (this is ~150 MB)...")
    with urlopen(HPA_TSV_URL) as resp:
        data = resp.read()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # extract the only file inside
        inner_name = zf.namelist()[0]
        with zf.open(inner_name) as src, open(local_path, "wb") as dst:
            dst.write(src.read())

    print(f"[ok] HPA file saved to {local_path}")
    return local_path


def load_hpa(path: str) -> pd.DataFrame:
    """Load HPA tsv and keep only columns relevant to surface-target annotation."""
    print(f"[..] Loading HPA table...")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    print(f"[ok] HPA rows: {len(df):,} | columns: {len(df.columns)}")

    # Keep what exists (HPA occasionally renames a column)
    available = [c for c in HPA_COLS_OF_INTEREST if c in df.columns]
    missing = [c for c in HPA_COLS_OF_INTEREST if c not in df.columns]
    if missing:
        print(f"[warn] HPA columns not found (will skip): {missing}")

    df = df[available].copy()

    # Surface / cell-surface flag derived from Protein class + Subcellular location
    def is_surface(row):
        pclass = str(row.get("Protein class", "") or "")
        subloc = str(row.get("Subcellular location", "") or "")
        if any(x in pclass for x in [
            "Predicted membrane proteins",
            "CD markers",
            "G-protein coupled receptors",
            "Ion channels",
            "Transporters",
            "Receptors",
        ]):
            return "Yes"
        if "Plasma membrane" in subloc or "Cell membrane" in subloc:
            return "Yes"
        return "No"

    df["HPA_surface_protein"] = df.apply(is_surface, axis=1)

    # Best reliability label (IH if available, else IF) — used as a single field for sorting
    def best_reliability(row):
        for col in ("Reliability (IH)", "Reliability (IF)"):
            v = str(row.get(col, "") or "")
            if v and v.lower() != "nan":
                return v
        return "Not available"

    df["HPA_antibody_reliability"] = df.apply(best_reliability, axis=1)

    return df


def annotate(candidates: pd.DataFrame, hpa: pd.DataFrame,
             symbol_col: str = "gene") -> pd.DataFrame:
    """Join candidate table to HPA on HGNC symbol (Gene column in HPA)."""
    if symbol_col not in candidates.columns:
        # try common alternates
        for alt in ("Gene", "gene_symbol", "symbol", "hgnc_symbol"):
            if alt in candidates.columns:
                symbol_col = alt
                break
        else:
            raise ValueError(
                f"Could not find gene-symbol column. "
                f"Available columns: {list(candidates.columns)[:10]}"
            )
    print(f"[..] Joining on candidate column '{symbol_col}' ↔ HPA 'Gene'")

    merged = candidates.merge(
        hpa,
        how="left",
        left_on=symbol_col,
        right_on="Gene",
        suffixes=("", "_hpa"),
    )

    matched = merged["Gene"].notna().sum()
    print(f"[ok] HPA match: {matched}/{len(candidates)} candidates "
          f"({100*matched/len(candidates):.1f}%)")

    # Tidy up: drop the duplicate HPA "Gene" column if both sides had it
    if "Gene" in merged.columns and symbol_col != "Gene":
        merged = merged.drop(columns=["Gene"])

    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s7", required=True, help="Path to S7 top-30 candidates CSV")
    ap.add_argument("--s8", required=True, help="Path to S8 250-candidate ranking CSV")
    ap.add_argument("--outdir", default="./hpa_output", help="Output directory")
    ap.add_argument("--symbol-col", default="gene",
                    help="Column name in S7/S8 that contains HGNC symbols")
    ap.add_argument("--hpa-file", default=HPA_LOCAL,
                    help="Local path for HPA tsv (downloads if not present)")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    hpa_path = download_hpa(args.hpa_file)
    hpa = load_hpa(hpa_path)

    for label, path in [("S7", args.s7), ("S8", args.s8)]:
        print(f"\n=== Annotating {label}: {path} ===")
        cand = pd.read_csv(path)
        print(f"[ok] Loaded {label} | rows: {len(cand)} | cols: {len(cand.columns)}")

        out = annotate(cand, hpa, symbol_col=args.symbol_col)

        out_path = os.path.join(
            args.outdir,
            os.path.basename(path).replace(".csv", "_HPA.csv"),
        )
        out.to_csv(out_path, index=False)
        print(f"[ok] Wrote {out_path}")

        # Print a quick summary
        if "HPA_surface_protein" in out.columns:
            surf = (out["HPA_surface_protein"] == "Yes").sum()
            print(f"     Surface-flagged (HPA): {surf}/{len(out)}")
        if "HPA_antibody_reliability" in out.columns:
            rel = out["HPA_antibody_reliability"].value_counts().to_dict()
            print(f"     Antibody reliability distribution: {rel}")

    print(f"\n[done] All outputs in {args.outdir}")


if __name__ == "__main__":
    main()