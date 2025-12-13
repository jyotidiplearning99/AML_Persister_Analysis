#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OKABE_ITO = {
    "green":  "#009E73",
    "orange": "#D55E00",
    "sky":    "#56B4E9",
    "blue":   "#0072B2",
    "pink":   "#CC79A7",
    "yellow": "#F0E442",
    "black":  "#000000",
    "grey":   "#999999",
}
STATUS_PALETTE = {
    "healthy":   OKABE_ITO["green"],
    "remission": OKABE_ITO["blue"],
    "disease":   OKABE_ITO["orange"],
    "other":     OKABE_ITO["grey"],
}

def infer_status(name: str) -> str:
    n = (str(name) or "").lower()
    if any(k in n for k in ["norm", "healthy", "control", "pbmc"]):
        return "healthy"
    if any(k in n for k in ["rem", "remission", "cr", "mr"]):
        return "remission"
    if any(k in n for k in ["aml", "prim", "relapse", "meta", "leuk"]):
        return "disease"
    return "other"

def choose_sample_col(df: pd.DataFrame) -> str:
    cand = [c for c in df.columns if re.search(r"sample", c, re.I)]
    return cand[0] if cand else df.columns[0]

def choose_fraction_col(df: pd.DataFrame) -> str | None:
    cand = [c for c in df.columns if re.search(r"fraction", c, re.I) and re.search(r"pos|positive", c, re.I)]
    if cand: return cand[0]
    cand = [c for c in df.columns if re.search(r"fraction", c, re.I)]
    if cand: return cand[0]
    cand = [c for c in df.columns if re.search(r"(persister.*(frac|fraction)|frac.*persister)", c, re.I)]
    if cand: return cand[0]
    return None

def main():
    ap = argparse.ArgumentParser(description="Validate Excel vs CSV and plot per-sample persister fractions (no h5ad needed).")
    ap.add_argument("--excel", required=True, help="Excel with per-sample fractions (e.g., persister_agreement_validated.xlsx)")
    ap.add_argument("--csv",   required=True, help="Merged CSV (per-sample fraction, or per-cell with call/score)")
    ap.add_argument("--tau",   type=float, default=0.31, help="Operating threshold τ for score→call (default 0.31)")
    ap.add_argument("--outdir", default="per_sample_outputs", help="Output folder")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load Excel
    excel_df = pd.read_excel(args.excel)
    excel_df.columns = [c.strip() for c in excel_df.columns]
    excel_sample_col = choose_sample_col(excel_df)
    excel_frac_col   = choose_fraction_col(excel_df)
    excel_df[excel_frac_col] = pd.to_numeric(excel_df[excel_frac_col], errors="coerce")
    excel_clean = excel_df[[excel_sample_col, excel_frac_col]].rename(
        columns={excel_sample_col:"sample_id", excel_frac_col:"fraction_positive_excel"}
    )

    # Load CSV
    merged = pd.read_csv(args.csv)
    merged.columns = [c.strip() for c in merged.columns]
    m_sample_col = choose_sample_col(merged)
    m_frac_col = choose_fraction_col(merged)

    fractions_from_csv = None
    n_cells_from_csv = None

    if m_frac_col is not None:
        tmp = merged[[m_sample_col, m_frac_col]].dropna()
        fractions_from_csv = tmp.groupby(m_sample_col)[m_frac_col].mean()
    else:
        call_cols  = [c for c in merged.columns if re.fullmatch(r"persister[_ ]?call", c, re.I)]
        score_cols = [c for c in merged.columns if re.fullmatch(r"persister[_ ]?score", c, re.I)]
        if call_cols:
            ccol = call_cols[0]
            grp = merged.groupby(m_sample_col)[ccol]
            fractions_from_csv = grp.mean()
            n_cells_from_csv = grp.count()
        elif score_cols:
            scol = score_cols[0]
            calls = (pd.to_numeric(merged[scol], errors="coerce") >= args.tau).astype(int)
            grp = calls.groupby(merged[m_sample_col])
            fractions_from_csv = grp.mean()
            n_cells_from_csv = grp.count()

    if fractions_from_csv is None:
        raise SystemExit("Could not infer per-sample fractions from CSV. "
                         "Expected a fraction column, or per-cell 'persister_call' or 'persister_score'.")

    csv_summary = pd.DataFrame({"fraction_positive": fractions_from_csv})
    if n_cells_from_csv is not None:
        csv_summary["n_cells"] = n_cells_from_csv
    csv_summary.index.name = "sample_id"
    csv_summary = csv_summary.reset_index()

    # Compare Excel vs CSV
    csv_clean = csv_summary.rename(columns={"fraction_positive": "fraction_positive_csv"})
    comp = pd.merge(csv_clean, excel_clean, on="sample_id", how="outer")
    comp["abs_diff"] = (comp["fraction_positive_csv"] - comp["fraction_positive_excel"]).abs()
    comp["status"] = [infer_status(s) for s in comp["sample_id"].astype(str)]

    # Plot bar chart
    plot_df = comp.sort_values("fraction_positive_csv", ascending=False).reset_index(drop=True)
    base_rate_unweighted = float(plot_df["fraction_positive_csv"].dropna().mean())
    if "n_cells" in plot_df.columns and plot_df["n_cells"].notna().any():
        w = plot_df["n_cells"].fillna(0).astype(float)
        x = plot_df["fraction_positive_csv"].fillna(0).astype(float)
        base_rate_weighted = float((w*x).sum() / w.sum()) if w.sum() else np.nan
    else:
        base_rate_weighted = np.nan

    status_counts = plot_df["status"].value_counts().to_dict()
    caption = f"τ = {args.tau:.2f}. Base rate (mean across samples): {base_rate_unweighted:.3f}"
    if not np.isnan(base_rate_weighted):
        caption += f" | Weighted by n_cells: {base_rate_weighted:.3f}"
    caption += f". Status counts: {status_counts}."

    bar_png = outdir/"per_sample_fraction_colored_status.png"
    plt.figure(figsize=(max(6, 0.25 * len(plot_df)), 5))
    colors = [STATUS_PALETTE.get(s, STATUS_PALETTE["other"]) for s in plot_df["status"]]
    plt.bar(plot_df["sample_id"].astype(str), plot_df["fraction_positive_csv"].astype(float), color=colors)
    plt.xticks(rotation=90)
    plt.ylabel("Persister-positive fraction (≥ τ)")
    plt.title("Persister-positive fraction by sample (colored by inferred status)")
    plt.gcf().text(0.01, -0.08, caption, ha="left", va="top", fontsize=9)
    plt.tight_layout()
    plt.savefig(bar_png, dpi=300, bbox_inches="tight")
    plt.close()

    # Histogram for bimodality
    hist_png = outdir/"per_sample_fraction_hist.png"
    vals = plot_df["fraction_positive_csv"].dropna().astype(float).values
    plt.figure(figsize=(6.5, 4.8))
    plt.hist(vals, bins=30)
    plt.xlabel("Persister-positive fraction (per sample)")
    plt.ylabel("Count")
    plt.title("Distribution of per-sample persister fractions")
    plt.gcf().text(0.01, -0.08, f"τ = {args.tau:.2f}. n = {len(vals)} samples.", ha="left", va="top", fontsize=9)
    plt.tight_layout()
    plt.savefig(hist_png, dpi=300, bbox_inches="tight")
    plt.close()

    # Export comparison workbook
    comparison_xlsx = outdir/"validation_comparison.xlsx"
    with pd.ExcelWriter(comparison_xlsx) as xl:
        comp.to_excel(xl, sheet_name="excel_vs_csv_comparison", index=False)
        csv_summary.to_excel(xl, sheet_name="csv_per_sample_summary", index=False)
        excel_df.to_excel(xl, sheet_name="original_excel_raw", index=False)

    print(f"[OK] Wrote:\n  - {bar_png}\n  - {hist_png}\n  - {comparison_xlsx}")

if __name__ == "__main__":
    main()
