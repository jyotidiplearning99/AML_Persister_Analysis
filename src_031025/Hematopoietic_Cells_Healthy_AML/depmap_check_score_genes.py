#!/usr/bin/env python3
"""
DepMap check for AML-score genes (one script):
  A) Summarize dependency of score genes in AML lines (uses aml_dependencies_ranked.csv if available,
     or computes light-weight metrics directly from DepMap for the score genes only).
  B) Test AML-selectivity (AML vs non-AML) using CRISPRGeneEffect + Model metadata.

Inputs (defaults are HPC-friendly; override via CLI):
  --score-dir    : folder with scoring_up.txt and scoring_down.txt
  --depmap-dir   : folder with DepMap CSVs: Model.csv, CRISPRGeneEffect.csv, (optional) CRISPRGeneDependency.csv
  --depmap-ranked: optional precomputed aml_dependencies_ranked.csv to reuse
  --out-dir      : where to write results (defaults to --score-dir)

Outputs:
  depmap_score_genes_summary.csv
  depmap_score_genes_top20_by_effect.csv
  depmap_score_genes_top20_by_rank.csv
  depmap_score_genes_AML_selectivity.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def load_score_genes(score_dir: Path) -> pd.Index:
    up   = pd.read_csv(score_dir/"scoring_up.txt",   header=None)[0].astype(str).str.upper()
    down = pd.read_csv(score_dir/"scoring_down.txt", header=None)[0].astype(str).str.upper()
    return pd.Index(up.tolist() + down.tolist()).unique()

def build_aml_mask(model_csv: Path) -> pd.Series:
    meta = pd.read_csv(model_csv)
    if "ModelID" not in meta.columns:
        raise ValueError("Model.csv missing ModelID column")
    meta = meta.set_index("ModelID")
    aml_mask = pd.Series(False, index=meta.index)
    for col in ["OncotreePrimaryDisease","OncotreeLineage","OncotreeSubtype"]:
        if col in meta.columns:
            aml_mask |= meta[col].astype(str).str.contains("Acute Myeloid|AML|Myeloid", case=False, na=False)
    return aml_mask

def map_symbol_to_cols(columns: pd.Index) -> dict:
    # DepMap columns often look like "TP53 (7157)"; map SYMBOL -> list of columns
    m = {}
    for c in columns:
        cc = str(c)
        sym = cc.split(" (")[0].upper()
        m.setdefault(sym, []).append(cc)
    return m

def col_for_symbol(columns: pd.Index, sym: str) -> str | None:
    # pick the first column matching SYMBOL (or exact)
    sym = sym.upper()
    exact = [c for c in columns if c == sym]
    if exact:
        return exact[0]
    pref = [c for c in columns if str(c).startswith(sym + " (")]
    return pref[0] if pref else None

def summarize_from_ranked(score_genes: pd.Index, ranked_csv: Path) -> pd.DataFrame:
    dep = pd.read_csv(ranked_csv)
    dep['gene'] = dep['gene'].astype(str).str.upper()
    sub = dep[dep['gene'].isin(score_genes)].copy()
    if 'frac_dep_lt_minus05' in sub.columns:
        sub['is_strong_dep@-0.5'] = sub['frac_dep_lt_minus05'] > 0.5
    return sub

def compute_light_metrics_for_score_genes(score_genes: pd.Index, depmap_dir: Path) -> pd.DataFrame:
    # Load Model + GeneEffect
    ge_path   = depmap_dir/"CRISPRGeneEffect.csv"
    model_csv = depmap_dir/"Model.csv"
    ge = pd.read_csv(ge_path, index_col=0)
    aml_mask = build_aml_mask(model_csv)
    aml_ids  = [i for i in ge.index if i in aml_mask.index and aml_mask.loc[i]]
    if not aml_ids:
        raise ValueError("No AML lines present in CRISPRGeneEffect.csv after matching Model.csv")
    sym_map = map_symbol_to_cols(ge.columns)

    rows = []
    for g in score_genes:
        cols = sym_map.get(g, [])
        if not cols:
            continue
        # If duplicates exist, average per line
        vals = ge.loc[aml_ids, cols].mean(axis=1, skipna=True)
        vals = vals.dropna()
        if len(vals) == 0:
            continue
        median_effect = float(vals.median())
        mean_effect   = float(vals.mean())
        frac_dep_lt_minus05 = float((vals < -0.5).mean())
        frac_dep_lt_minus03 = float((vals < -0.3).mean())
        q25, q75 = vals.quantile([0.25, 0.75])
        rows.append({
            "gene": g,
            "median_effect": median_effect,
            "mean_effect": mean_effect,
            "min_effect": float(vals.min()),
            "max_effect": float(vals.max()),
            "q25_effect": float(q25),
            "q75_effect": float(q75),
            "frac_dep_lt_minus05": frac_dep_lt_minus05,
            "frac_dep_lt_minus03": frac_dep_lt_minus03,
            "n_cell_lines": int(len(vals)),
            "is_strongly_essential": frac_dep_lt_minus05 > 0.5
        })
    out = pd.DataFrame(rows)

    # Try to add dependency probability if present
    dep_prob_path = depmap_dir/"CRISPRGeneDependency.csv"
    if out.shape[0] and dep_prob_path.exists():
        dp = pd.read_csv(dep_prob_path, index_col=0)
        # intersect AML IDs
        aml_ids_dp = [i for i in dp.index if i in aml_ids]
        for i, r in out.iterrows():
            g = r["gene"]
            col = col_for_symbol(dp.columns, g)
            if not col: 
                continue
            probs = dp.loc[aml_ids_dp, col].dropna()
            if len(probs):
                out.loc[i, "dep_prob_median"] = float(probs.median())
                out.loc[i, "dep_prob_mean"] = float(probs.mean())
                out.loc[i, "frac_prob_gt_05"] = float((probs > 0.5).mean())

    # rank score similar to your previous formula
    out["rank_score"] = (
        (-out["median_effect"])*0.4 +
         out["frac_dep_lt_minus05"]*0.3 +
         out["frac_dep_lt_minus03"]*0.2 +
         out.get("dep_prob_median", pd.Series(0, index=out.index))*0.1
    )
    return out

def aml_selectivity(score_genes: pd.Index, depmap_dir: Path) -> pd.DataFrame:
    ge = pd.read_csv(depmap_dir/"CRISPRGeneEffect.csv", index_col=0)
    aml_mask = build_aml_mask(depmap_dir/"Model.csv")
    aml_ids  = [i for i in ge.index if i in aml_mask.index and aml_mask.loc[i]]
    non_ids  = [i for i in ge.index if i in aml_mask.index and not aml_mask.loc[i]]
    sym_map  = map_symbol_to_cols(ge.columns)

    rows = []
    for g in score_genes:
        cols = sym_map.get(g, [])
        if not cols:
            continue
        a = ge.loc[aml_ids, cols].mean(axis=1, skipna=True).dropna()
        b = ge.loc[non_ids, cols].mean(axis=1, skipna=True).dropna()
        if len(a) >= 4 and len(b) >= 10:
            # "less" means AML median is more negative (stronger dependency)
            p = mannwhitneyu(a, b, alternative="less").pvalue
        else:
            p = np.nan
        rows.append({
            "gene": g,
            "median_aml": float(a.median()) if len(a) else np.nan,
            "median_nonaml": float(b.median()) if len(b) else np.nan,
            "mw_p_aml_less_non": float(p) if not np.isnan(p) else np.nan,
            "n_aml": int(len(a)),
            "n_nonaml": int(len(b))
        })
    sel = pd.DataFrame(rows).sort_values(["mw_p_aml_less_non","median_aml"], ascending=[True, True])
    return sel

def main():
    ap = argparse.ArgumentParser(description="DepMap dependency & AML-selectivity for AML score genes")
    ap.add_argument("--score-dir",    type=Path, default=Path("results/Hematopoietic_Cells_Healthy_AML_analysis"))
    ap.add_argument("--depmap-dir",   type=Path, default=Path("/scratch/project_2010751/DepMap_Datasets"))
    ap.add_argument("--depmap-ranked",type=Path, default=Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/depmap_refined/aml_dependencies_ranked.csv"))
    ap.add_argument("--out-dir",      type=Path, default=None)
    args = ap.parse_args()

    score_dir = args.score_dir.resolve()
    depmap_dir = args.depmap_dir.resolve()
    out_dir = (args.out_dir or args.score_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DepMap check for AML score genes")
    print("="*80)
    print(f"SCORE DIR   : {score_dir}")
    print(f"DEPMAP DIR  : {depmap_dir}")
    print(f"OUT DIR     : {out_dir}")
    if args.depmap_ranked:
        print(f"RANKED CSV  : {args.depmap_ranked}")

    # 1) Load score genes
    score_genes = load_score_genes(score_dir)
    print(f"Loaded {len(score_genes)} score genes (up+down)")

    # 2) Part A — dependency summary in AML lines
    use_ranked = args.depmap_ranked and args.depmap_ranked.exists()
    if use_ranked:
        print("Using precomputed aml_dependencies_ranked.csv for summary …")
        sub = summarize_from_ranked(score_genes, args.depmap_ranked)
        if "rank_score" not in sub.columns:
            # If rank_score isn't there (unlikely), synthesize a simple one
            sub["rank_score"] = (
                (-sub["median_effect"])*0.4 +
                 sub["frac_dep_lt_minus05"]*0.3 +
                 sub["frac_dep_lt_minus03"]*0.3
            )
    else:
        print("No precomputed ranked CSV found; computing light-weight metrics directly from DepMap …")
        sub = compute_light_metrics_for_score_genes(score_genes, depmap_dir)

    if sub.empty:
        print("WARNING: No score genes matched in DepMap. Exiting after AML-selectivity step…")

    # Save Part A outputs
    if not sub.empty:
        sub_sorted_effect = sub.sort_values("median_effect")  # most negative first
        sub_sorted_rank   = sub.sort_values("rank_score", ascending=False)

        sub.to_csv(out_dir/"depmap_score_genes_summary.csv", index=False)
        sub_sorted_effect.head(20).to_csv(out_dir/"depmap_score_genes_top20_by_effect.csv", index=False)
        sub_sorted_rank.head(20).to_csv(out_dir/"depmap_score_genes_top20_by_rank.csv", index=False)

        print("\n[Part A] Summary:")
        print(f"  Score genes in DepMap: {sub.shape[0]}/{len(score_genes)}")
        if "frac_dep_lt_minus05" in sub.columns:
            strong = (sub["frac_dep_lt_minus05"] > 0.5).mean()
            print(f"  Strong AML dependency (effect < -0.5 in >50% AML lines): {strong*100:.1f}%")

        print(f"  Saved: depmap_score_genes_summary.csv")
        print(f"         depmap_score_genes_top20_by_effect.csv")
        print(f"         depmap_score_genes_top20_by_rank.csv")

    # 3) Part B — AML selectivity test
    print("\nComputing AML selectivity (AML vs non-AML) …")
    sel = aml_selectivity(score_genes, depmap_dir)
    sel.to_csv(out_dir/"depmap_score_genes_AML_selectivity.csv", index=False)
    n_sig = sel["mw_p_aml_less_non"].lt(0.05).sum(skipna=True)
    print(f"[Part B] AML-selective (MWU p<0.05, AML more negative): {int(n_sig)}/{sel.shape[0]}")
    print("  Saved: depmap_score_genes_AML_selectivity.csv")

    # 4) Tiny console digest
    print("\nDone.")
    print(f"Outputs written to: {out_dir}")

if __name__ == "__main__":
    main()
