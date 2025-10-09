#!/usr/bin/env python3
"""
Auto clinical enrichment for persister probability (BeatAML / TCGA)
- Auto-discovers predictions in results/bulk_*
- Robust ID matching (chooses best-overlap clinical ID column)
- Fuzzy detection of ELN/MRD/Blast% columns
- ELN chi2 + Cramer's V + Fisher 2x2; MRD MWU + OR per 0.1 + AUC; Blast% Spearman
- Saves figures + CSV summary + JSON dump
Author: Jyotidip Barman (robust join + fuzzy columns version)
"""

import os, re, json, warnings, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------- basic utilities ----------

PRED_FILE_CANDIDATES = [
    "predictions_final.csv",
    "predictions_calibrated.csv",
    "predictions_enhanced.csv",
    "predictions_optimal.csv",
    "predictions_fixed_v2.csv",
    "predictions_fixed.csv",
    "predictions_reduced_1k.csv",
    "predictions_neutral.csv",
    "predictions_nostrat.csv",
    "predictions.csv",
]

PROB_COLS = [
    "persister_probability","persister_prob","prob","probability",
    "p_persister","pred_persister","persister","p"
]

NONP_COLS = ["non_persister_probability","non_persister_prob","non_persister","p_nonpersister"]

ID_COLS_PRED = [
    "sample","sample_id","Sample","SampleID","id","ID",
    "barcode","submitter_id","patient_id","PATIENT","SAMPLE_ID"
]

# loose substrings for fuzzy find
ELN_SUBSTR = ["eln"]
MRD_SUBSTR = ["mrd", "residual"]  # permissive; we still check the data type
BLAST_SUBSTR = ["blast", "blasts", "bm_blast", "bone_marrow_blast"]

def read_any_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists(): return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return None

def load_predictions(cohort_dir: Path) -> Tuple[pd.DataFrame, str]:
    for name in PRED_FILE_CANDIDATES:
        f = cohort_dir / name
        if f.exists():
            df = read_any_csv(f)
            if df is not None and len(df) > 0:
                return df, name
    raise FileNotFoundError(f"No predictions_*.csv found in {cohort_dir}")

def extract_prob(df: pd.DataFrame) -> pd.Series:
    # direct columns
    for c in df.columns:
        if c.lower() in [x.lower() for x in PROB_COLS]:
            return df[c].astype(float)
    # 1 - non-persister
    for c in df.columns:
        if c.lower() in [x.lower() for x in NONP_COLS]:
            return 1.0 - df[c].astype(float)
    # sometimes a softmax column "Persister"
    for c in df.columns:
        if c.lower() == "persister":
            return df[c].astype(float)
    raise ValueError("Could not locate persister probability column in predictions.")

def normalize_id_series(s: pd.Series) -> pd.Series:
    # Upper, strip spaces, remove trailing/leading whitespace
    x = s.astype(str).str.strip().str.upper()
    # common cleanup for TCGA barcodes (keep first 12 chars) if it helps overlap
    return x

def pick_best_id_column(clin: pd.DataFrame, pred_ids: pd.Series) -> Optional[str]:
    """Choose clinical ID column with maximal overlap to predictions IDs."""
    pred_norm = normalize_id_series(pred_ids)
    # candidate columns = all object-like cols + common synonyms
    candidates = []
    for c in clin.columns:
        if pd.api.types.is_string_dtype(clin[c]) or clin[c].dtype == object:
            candidates.append(c)
    # put likely names first
    likely_first = ["sample","sample_id","submitter_id","patient_id","barcode","SAMPLE_ID","PATIENT"]
    candidates = list(dict.fromkeys(likely_first + candidates))  # dedupe, keep order
    best, best_overlap = None, 0
    for c in candidates:
        c_norm = normalize_id_series(clin[c])
        # try raw
        overlap = pred_norm.isin(set(c_norm)).sum()
        # try TCGA 12-char core if improves
        if overlap < (pred_norm.size // 10):  # heuristic
            c_core = c_norm.str.replace(r"[^A-Z0-9\-]+", "", regex=True).str.slice(0,12)
            overlap_core = pred_norm.str.slice(0,12).isin(set(c_core)).sum()
            if overlap_core > overlap:
                overlap = overlap_core
        if overlap > best_overlap:
            best, best_overlap = c, overlap
    return best

def fuzzy_find_column(df: pd.DataFrame, substr_list: List[str], want_numeric: bool=False) -> Optional[str]:
    cols = []
    for c in df.columns:
        cl = c.lower()
        if any(s in cl for s in substr_list):
            cols.append(c)
    if not cols:
        return None
    if want_numeric:
        # prefer numeric-looking
        num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num:
            return num[0]
    return cols[0]

def load_clinical_table(cohort_dir: Path) -> pd.DataFrame:
    # 1) clinical_predictions.csv
    df = read_any_csv(cohort_dir / "clinical_predictions.csv")
    if df is not None:
        return df
    # 2) metadata JSON/CSV
    j = cohort_dir / "predictions_enhanced_metadata.json"
    if j.exists():
        try:
            payload = json.load(open(j))
            if isinstance(payload, dict):
                return pd.json_normalize(payload)
            elif isinstance(payload, list):
                return pd.json_normalize(payload)
        except Exception:
            pass
    c = cohort_dir / "predictions_enhanced_metadata.csv"
    if c.exists():
        df = read_any_csv(c)
        if df is not None:
            return df
    return pd.DataFrame()

def stratify_tertiles(p: np.ndarray) -> pd.Series:
    thr = np.percentile(p, [33.33, 66.67])
    return pd.cut(p, bins=[-np.inf, thr[0], thr[1], np.inf], labels=["Low","Intermediate","High"])

def cramer_v(chi2: float, table: pd.DataFrame) -> float:
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r,k)-1)))) if min(r,k) > 1 and n>0 else np.nan

def fisher_or_ci(a,b,c,d):
    if min(a,b,c,d) == 0:
        return np.inf, (np.nan, np.nan)
    OR = (a*d)/(b*c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = 1.959964
    return float(OR), (float(np.exp(np.log(OR)-z*se)), float(np.exp(np.log(OR)+z*se)))

def analyze_cohort(cohort_dir: Path, cohort_name: str, outdir: Path, tau: float) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)

    # predictions
    pred_df, used_file = load_predictions(cohort_dir)
    # ID
    pred_id_col = None
    for c in pred_df.columns:
        if c.lower() in [x.lower() for x in ID_COLS_PRED]:
            pred_id_col = c; break
    if pred_id_col is None:
        pred_ids = pred_df.index.astype(str)
        pred_id_col = "<index>"
    else:
        pred_ids = pred_df[pred_id_col].astype(str)

    # prob
    prob = extract_prob(pred_df).to_numpy()
    n = prob.size
    frac = float((prob >= tau).mean())

    # clinical
    clin = load_clinical_table(cohort_dir)
    chosen_id = None
    if len(clin):
        chosen_id = pick_best_id_column(clin, pred_ids)
    if chosen_id is None:
        # no clinical or no match — still return distributions & prevalence
        chosen_id = "<none>"
        merged = pd.DataFrame({"sample": pred_ids, "prob": prob})
    else:
        left = pd.DataFrame({"sample": normalize_id_series(pred_ids), "prob": prob})
        right = clin.copy()
        right["_join_id_"] = normalize_id_series(right[chosen_id])
        merged = left.merge(right, left_on="sample", right_on="_join_id_", how="left")
    matched = int(merged["_join_id_"].notna().sum()) if "_join_id_" in merged else 0

    # fuzzy clinical column detection
    eln_col = None; mrd_bin_col = None; mrd_cont_col = None; blast_col = None
    if len(clin):
        # exact-ish names first
        for c in clin.columns:
            if c.lower() in ["eln_risk_2022","eln_risk","eln","elnrisk","eln_2017","eln2017"]: eln_col = c
            if c.lower() in ["mrd_status","mrd","mrd_binary","mrd_bin"]: mrd_bin_col = mrd_bin_col or c
            if c.lower() in ["mrd_level","mrd_fraction","mrd_cont","mrd_percent","mrd_pct"]: mrd_cont_col = mrd_cont_col or c
        # fuzzy fallback
        eln_col = eln_col or fuzzy_find_column(clin, ELN_SUBSTR, want_numeric=False)
        mrd_bin_col = mrd_bin_col or fuzzy_find_column(clin, MRD_SUBSTR, want_numeric=False)
        # For continuous MRD prefer numeric:
        mrd_cont_col = mrd_cont_col or fuzzy_find_column(clin, MRD_SUBSTR, want_numeric=True)
        blast_col = fuzzy_find_column(clin, BLAST_SUBSTR, want_numeric=True)

    results = {
        "cohort": cohort_name,
        "n": n,
        "tau": tau,
        "frac_ge_tau": frac,
        "pred_file": used_file,
        "join": {
            "pred_id_col": pred_id_col,
            "clin_id_col": chosen_id,
            "matched_rows": matched
        },
        "columns": {
            "eln": eln_col,
            "mrd_binary": mrd_bin_col,
            "mrd_cont": mrd_cont_col,
            "blast_pct": blast_col
        }
    }

    figs_dir = outdir / "figures"; figs_dir.mkdir(parents=True, exist_ok=True)

    # 1) distribution
    plt.figure(figsize=(5,4))
    plt.hist(prob, bins=30, edgecolor="k", alpha=0.8)
    plt.axvline(tau, color="r", ls="--", label=f"τ={tau}")
    plt.title(f"{cohort_name} persister probability")
    plt.xlabel("Persister probability"); plt.ylabel("Count")
    plt.legend(); plt.tight_layout()
    plt.savefig(figs_dir / f"{cohort_name}_distribution.png", dpi=300)
    plt.close()

    # 2) ELN by tertile
    if eln_col and matched > 0:
        groups = stratify_tertiles(prob)
        eln_vals = merged[eln_col].astype(str).str.strip()
        tab = pd.crosstab(groups, eln_vals)
        if tab.values.sum() > 0 and tab.shape[0] > 1 and tab.shape[1] > 1:
            chi2, pval, dof, exp = chi2_contingency(tab)
            results["eln_enrichment"] = {
                "chi2_p": float(pval),
                "cramers_v": float(cramer_v(chi2, tab))
            }
            # fisher High vs Low, Adverse vs Favorable if present
            if "High" in tab.index and "Low" in tab.index and "Adverse" in tab.columns and "Favorable" in tab.columns:
                a = int(tab.loc["High","Adverse"]); b = int(tab.loc["High","Favorable"])
                c = int(tab.loc["Low","Adverse"]);  d = int(tab.loc["Low","Favorable"])
                OR, CI = fisher_or_ci(a,b,c,d)
                results["eln_enrichment"]["fisher_high_low_adverse_OR"] = OR
                results["eln_enrichment"]["fisher_CI95"] = CI
            # plot
            ax = (tab.div(tab.sum(axis=1), axis=0)).plot(kind="bar", stacked=True, figsize=(6,4),
                                                         color=["#2ecc71","#f39c12","#e74c3c"])
            ax.set_xlabel("Persister tertile"); ax.set_ylabel("Proportion"); ax.set_title(f"{cohort_name} | ELN by tertile")
            ax.legend(title="ELN", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(figs_dir / f"{cohort_name}_eln_by_tertile.png", dpi=300)
            plt.close()

    # 3) MRD binary
    if mrd_bin_col and matched > 0:
        mrd = merged[mrd_bin_col].astype(str)
        pos_mask = mrd.str.upper().isin(["MRD+","POS","POSITIVE","1","TRUE","YES"])
        neg_mask = mrd.str.upper().isin(["MRD-","NEG","NEGATIVE","0","FALSE","NO"])
        ppos, pneg = prob[pos_mask.values], prob[neg_mask.values]
        out = {"n_pos": int(ppos.size), "n_neg": int(pneg.size)}
        if ppos.size>0 and pneg.size>0:
            u, pw = mannwhitneyu(ppos, pneg, alternative="two-sided")
            out["mannwhitney_p"] = float(pw)
            try:
                X = prob.reshape(-1,1); y = pos_mask.astype(int).values
                lr = LogisticRegression(solver="liblinear").fit(X,y)
                out["OR_per_0.1"] = float(np.exp(lr.coef_[0,0]*0.1))
                out["AUC"] = float(roc_auc_score(y, lr.predict_proba(X)[:,1]))
            except Exception:
                out["OR_per_0.1"] = np.nan; out["AUC"] = np.nan
        results["mrd_binary"] = out

        # plot
        data = pd.DataFrame({"prob": prob, "MRD": np.where(pos_mask, "MRD+", np.where(neg_mask,"MRD-","NA"))})
        order = [x for x in ["MRD-","MRD+","NA"] if x in set(data["MRD"])]
        plt.figure(figsize=(5,4))
        parts = plt.violinplot([data.loc[data["MRD"]==lab,"prob"] for lab in order], positions=list(range(1,len(order)+1)),
                               widths=0.8, showmeans=True)
        plt.xticks(list(range(1,len(order)+1)), order)
        plt.ylabel("Persister probability"); plt.title(f"{cohort_name} | MRD")
        if "mannwhitney_p" in out and np.isfinite(out["mannwhitney_p"]):
            ylim = plt.gca().get_ylim()
            plt.text(1.5, ylim[1]*0.95, f"MWU p={out['mannwhitney_p']:.3g}", ha="center")
        plt.tight_layout(); plt.savefig(figs_dir / f"{cohort_name}_mrd_binary.png", dpi=300); plt.close()

    # 4) MRD continuous
    if mrd_cont_col and matched > 0:
        y = pd.to_numeric(merged[mrd_cont_col], errors="coerce").to_numpy()
        if np.isfinite(y).sum() > 2:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(prob, y, alpha=0.5, edgecolors="k", linewidths=0.3)
            z = np.polyfit(prob, y, 1)
            xx = np.linspace(prob.min(), prob.max(), 100)
            ax.plot(xx, np.poly1d(z)(xx), "r--", lw=1)
            r, p = spearmanr(prob, y, nan_policy="omit")
            ax.text(0.02, 0.98, f"Spearman r={r:.3f}\np={p:.3g}", transform=ax.transAxes,
                    va="top", ha="left", bbox=dict(facecolor="w", edgecolor="0.7"))
            ax.set_xlabel("Persister probability"); ax.set_ylabel("MRD (continuous)"); ax.set_title(f"{cohort_name} | MRD (cont.)")
            plt.tight_layout(); plt.savefig(figs_dir / f"{cohort_name}_mrd_continuous.png", dpi=300); plt.close()
            results["mrd_continuous"] = {"spearman_r": float(r), "p": float(p), "n": int(np.isfinite(y).sum())}

    # 5) Blasts
    if blast_col and matched > 0:
        y = pd.to_numeric(merged[blast_col], errors="coerce").to_numpy()
        if np.isfinite(y).sum() > 2:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(prob, y, alpha=0.5, edgecolors="k", linewidths=0.3)
            z = np.polyfit(prob, y, 1)
            xx = np.linspace(prob.min(), prob.max(), 100)
            ax.plot(xx, np.poly1d(z)(xx), "r--", lw=1)
            r, p = spearmanr(prob, y, nan_policy="omit")
            ax.text(0.02, 0.98, f"Spearman r={r:.3f}\np={p:.3g}", transform=ax.transAxes,
                    va="top", ha="left", bbox=dict(facecolor="w", edgecolor="0.7"))
            ax.set_xlabel("Persister probability"); ax.set_ylabel("Blast %"); ax.set_title(f"{cohort_name} | Blasts")
            plt.tight_layout(); plt.savefig(figs_dir / f"{cohort_name}_blasts.png", dpi=300); plt.close()
            results["blasts"] = {"spearman_r": float(r), "p": float(p), "n": int(np.isfinite(y).sum())}

    # save merged for inspection
    merged.to_csv(outdir / f"{cohort_name}_merged_table.csv", index=False)

    # prevalence table
    prev = pd.DataFrame({"sample": pred_ids, "prob": prob, "call_ge_tau": (prob >= tau).astype(int)})
    prev.to_csv(outdir / f"{cohort_name}_prevalence.csv", index=False)

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, help="Path to AML_Persister_Analysis/results")
    ap.add_argument("--tau", type=float, default=0.31)
    ap.add_argument("--include", nargs="*", default=None, help="Subset of cohorts, e.g., BeatAML TCGA-LAML")
    args = ap.parse_args()

    root = Path(args.results_root)
    out_root = root / "clinical_enrichment"
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = {"BeatAML": root / "bulk_BeatAML", "TCGA-LAML": root / "bulk_TCGA"}
    cohorts = {k:v for k,v in candidates.items() if (not args.include or k in args.include) and v.exists()}

    all_results = {}
    for name, cdir in cohorts.items():
        print(f"[INFO] Analyzing {name} in {cdir}")
        outdir = out_root / name
        try:
            cres = analyze_cohort(cdir, name, outdir, tau=args.tau)
            all_results[name] = cres
            # small per-cohort summary row
        except Exception as e:
            warnings.warn(f"{name} failed: {e}")

    # summary CSV
    rows = []
    for name, res in all_results.items():
        row = {
            "cohort": name,
            "n": res.get("n", np.nan),
            "frac_ge_tau": res.get("frac_ge_tau", np.nan),
            "eln_chi2_p": res.get("eln_enrichment", {}).get("chi2_p", np.nan),
            "eln_cramers_v": res.get("eln_enrichment", {}).get("cramers_v", np.nan),
            "mrd_mwu_p": res.get("mrd_binary", {}).get("mannwhitney_p", np.nan),
            "mrd_OR_per_0.1": res.get("mrd_binary", {}).get("OR_per_0.1", np.nan),
            "mrd_auc": res.get("mrd_binary", {}).get("AUC", np.nan),
            "mrd_cont_r": res.get("mrd_continuous", {}).get("spearman_r", np.nan),
            "mrd_cont_p": res.get("mrd_continuous", {}).get("p", np.nan),
            "blasts_r": res.get("blasts", {}).get("spearman_r", np.nan),
            "blasts_p": res.get("blasts", {}).get("p", np.nan),
            "matched_rows": res.get("join",{}).get("matched_rows", np.nan),
            "pred_id_col": res.get("join",{}).get("pred_id_col", ""),
            "clin_id_col": res.get("join",{}).get("clin_id_col", "")
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "enrichment_summary.csv", index=False)

    with open(out_root / "enrichment_results.json","w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[OK] Saved outputs → {out_root}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
