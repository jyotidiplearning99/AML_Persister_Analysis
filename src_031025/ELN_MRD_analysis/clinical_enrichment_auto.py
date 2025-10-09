#!/usr/bin/env python3
"""
Auto clinical enrichment for persister probability (BeatAML / TCGA)
- Auto-discovers predictions in results/bulk_*
- Tolerates many column name variants
- Pulls ELN/MRD/Blast% from clinical_predictions.csv OR predictions_enhanced_metadata.json
- Produces ELN enrichment (chi2 + Cramer's V + Fisher 2x2),
  MRD association (MWU + OR via logistic regression + ROC AUC), and
  optional correlations with blast%
- Saves figures + CSV summary + JSON dump for manuscript
Author: Jyotidip Barman (adapted for no-config run)
"""

import os, json, warnings, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr, kruskal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------- utility ----------

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

ID_COLS = ["sample","sample_id","Sample","SampleID","id","ID","barcode","submitter_id","patient_id","PATIENT","SAMPLE_ID"]

# clinical column synonyms
ELN_COLS = ["eln_risk_2022","eln_risk","ELN_risk","ELN","eln"]
MRD_BIN_COLS = ["mrd_status","MRD_status","mrd","MRD"]
MRD_CONT_COLS = ["mrd_level","MRD_level","mrd_cont","MRD_cont","mrd_fraction","MRD_fraction"]
BLAST_COLS = ["blast_pct","blast_percent","blast_percentage","blasts","BM_blast_pct","BM_blasts"]
OS_TIME_COLS = ["os_time","OS_time","overall_survival_months","OS_months","time"]
OS_EVENT_COLS = ["os_event","OS_event","event","status","death_event"]

def first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

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
    """Find a predictions_*.csv and return df + chosen filename."""
    for name in PRED_FILE_CANDIDATES:
        f = cohort_dir / name
        if f.exists():
            df = read_any_csv(f)
            if df is not None and len(df) > 0:
                return df, name
    raise FileNotFoundError(f"No predictions_*.csv found in {cohort_dir}")

def extract_prob(df: pd.DataFrame) -> pd.Series:
    pcol = first_col(df, PROB_COLS)
    if pcol:
        return df[pcol].astype(float)
    # synthesize from 2-class probs if available
    p_non = first_col(df, NONP_COLS)
    if p_non:
        return 1.0 - df[p_non].astype(float)
    # sometimes softmax columns like "Persister" / "NonPersister"
    two_cols = [c for c in df.columns if c.lower() in ("persister","non_persister","non-persister")]
    if "persister" in [c.lower() for c in two_cols]:
        # exact persister probability present
        for c in df.columns:
            if c.lower()=="persister":
                return df[c].astype(float)
    raise ValueError("Could not locate persister probability column.")

def extract_id(df: pd.DataFrame) -> pd.Series:
    icol = first_col(df, ID_COLS)
    if icol: 
        return df[icol].astype(str)
    # otherwise keep index as id
    return df.index.astype(str).to_series(index=df.index, name="sample")

def load_clinical_table(cohort_dir: Path) -> pd.DataFrame:
    """Try CSV first, then JSON."""
    # 1) clinical_predictions.csv (your screenshot shows this)
    df = read_any_csv(cohort_dir / "clinical_predictions.csv")
    if df is not None: 
        return df

    # 2) predictions_enhanced_metadata.json
    j = cohort_dir / "predictions_enhanced_metadata.json"
    if j.exists():
        try:
            with open(j) as f:
                payload = json.load(f)
            # try to coerce to DataFrame (accept dict-of-lists, list-of-dicts)
            if isinstance(payload, dict):
                # dict of lists
                lens = [len(v) for v in payload.values() if hasattr(v, "__len__")]
                if len(set(lens))==1:
                    return pd.DataFrame(payload)
                # otherwise try nested
                return pd.json_normalize(payload)
            elif isinstance(payload, list):
                return pd.json_normalize(payload)
        except Exception:
            pass
    # 3) nothing
    return pd.DataFrame()

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return first_col(df, candidates)

def cramer_v(chi2: float, table: pd.DataFrame) -> float:
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r,k)-1)))) if min(r,k) > 1 and n>0 else np.nan

# ---------- analyses ----------

def stratify_tertiles(p: np.ndarray) -> pd.Series:
    thr = np.percentile(p, [33.33, 66.67])
    return pd.cut(p, bins=[-np.inf, thr[0], thr[1], np.inf], labels=["Low","Intermediate","High"])

def fisher_or_ci(a,b,c,d, alpha=0.05):
    # Wald CI on log(OR); fine for reporting ballpark
    if min(a,b,c,d) == 0: 
        return np.inf, (np.nan, np.nan)
    OR = (a*d)/(b*c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = 1.959964
    return OR, (np.exp(np.log(OR)-z*se), np.exp(np.log(OR)+z*se))

def mrd_binary_stats(p: np.ndarray, mrd: pd.Series) -> Dict:
    mrd = mrd.astype(str)
    pos_mask = mrd.str.upper().isin(["MRD+","POS","POSITIVE","1","TRUE","YES"])
    neg_mask = mrd.str.upper().isin(["MRD-","NEG","NEGATIVE","0","FALSE","NO"])
    ppos, pneg = p[pos_mask], p[neg_mask]
    out = {"n_pos": int(ppos.size), "n_neg": int(pneg.size)}
    if ppos.size>0 and pneg.size>0:
        u, pw = mannwhitneyu(ppos, pneg, alternative="two-sided")
        out["mannwhitney_p"] = float(pw)
        # simple logistic odds per 0.1
        try:
            X = p.reshape(-1,1); y = pos_mask.astype(int).values
            lr = LogisticRegression(solver="liblinear").fit(X,y)
            out["OR_per_0.1"] = float(np.exp(lr.coef_[0,0]*0.1))
            out["AUC"] = float(roc_auc_score(y, lr.predict_proba(X)[:,1]))
        except Exception as e:
            out["OR_per_0.1"] = np.nan; out["AUC"] = np.nan
    return out

def eln_enrichment(p_groups: pd.Series, eln: pd.Series) -> Dict:
    eln = eln.astype(str).str.strip()
    # compress common variants
    eln = eln.replace({
        "adverse risk":"Adverse","intermediate risk":"Intermediate","favorable risk":"Favorable",
        "adverse":"Adverse","intermediate":"Intermediate","favorable":"Favorable",
        "High":"Adverse","Low":"Favorable"
    })
    tab = pd.crosstab(p_groups, eln)
    out = {"table": tab.to_dict()}
    if tab.values.size and tab.values.sum()>0 and tab.shape[0]>1 and tab.shape[1]>1:
        chi2, pval, dof, exp = chi2_contingency(tab)
        out["chi2_p"] = float(pval)
        out["cramers_v"] = float(cramer_v(chi2, tab))
        # High vs Low, Adverse vs Favorable if present
        for r in ["High","Low"]:
            if r not in tab.index: 
                return out
        if "Adverse" in tab.columns and "Favorable" in tab.columns:
            a = int(tab.loc["High","Adverse"]) if "Adverse" in tab.columns else 0
            b = int(tab.loc["High","Favorable"]) if "Favorable" in tab.columns else 0
            c = int(tab.loc["Low","Adverse"]) if "Adverse" in tab.columns else 0
            d = int(tab.loc["Low","Favorable"]) if "Favorable" in tab.columns else 0
            OR, CI = fisher_or_ci(a,b,c,d)
            out["fisher_high_vs_low_adverse_OR"] = float(OR)
            out["fisher_high_vs_low_adverse_CI95"] = [float(CI[0]) if np.isfinite(CI[0]) else np.nan,
                                                      float(CI[1]) if np.isfinite(CI[1]) else np.nan]
    return out

def scatter_with_fit(x, y, ax, xlabel, ylabel, title):
    ax.scatter(x, y, alpha=0.5, edgecolors="k", linewidths=0.3)
    if np.isfinite(x).sum()>2 and np.isfinite(y).sum()>2:
        z = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, np.poly1d(z)(xx), "r--", lw=1)
        r, p = spearmanr(x, y, nan_policy="omit")
        ax.text(0.02, 0.98, f"Spearman r={r:.3f}\np={p:.3g}", transform=ax.transAxes,
                va="top", ha="left",
                bbox=dict(facecolor="w", edgecolor="0.7"))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)

# ---------- main pipeline ----------

def analyze_cohort(cohort_dir: Path, cohort_name: str, outdir: Path, tau: float) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)

    # predictions
    pred_df, used_file = load_predictions(cohort_dir)
    pid = extract_id(pred_df)
    prob = extract_prob(pred_df).to_numpy()

    # quick tally at preset threshold
    frac = float((prob >= tau).mean())
    n = prob.size

    # clinical
    clin = load_clinical_table(cohort_dir)
    # harmonize join key
    cid_col = first_col(clin, ID_COLS) or "sample"
    if cid_col not in clin.columns:
        clin[cid_col] = np.nan
    left = pd.DataFrame({"sample": pid.values, "prob": prob})
    right = clin.rename(columns={cid_col:"sample"})
    merged = left.merge(right, on="sample", how="left")

    # pick columns
    eln_col = pick_col(merged, ELN_COLS)
    mrd_bin_col = pick_col(merged, MRD_BIN_COLS)
    mrd_cont_col = pick_col(merged, MRD_CONT_COLS)
    blast_col = pick_col(merged, BLAST_COLS)

    results = {
        "cohort": cohort_name,
        "n": n,
        "tau": tau,
        "frac_ge_tau": frac,
        "pred_file": used_file,
        "columns": {
            "eln": eln_col, "mrd_binary": mrd_bin_col, "mrd_cont": mrd_cont_col, "blast_pct": blast_col
        }
    }

    # plots folder
    figs_dir = outdir / "figures"; figs_dir.mkdir(exist_ok=True, parents=True)

    # 1) distribution plot
    plt.figure(figsize=(5,4))
    plt.hist(prob, bins=30, edgecolor="k", alpha=0.8)
    plt.axvline(tau, color="r", ls="--", label=f"τ={tau}")
    plt.title(f"{cohort_name} persister probability")
    plt.xlabel("Persister probability"); plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / f"{cohort_name}_distribution.png", dpi=300)
    plt.close()

    # 2) ELN enrichment
    if eln_col:
        groups = stratify_tertiles(prob)
        eln_res = eln_enrichment(groups, merged[eln_col])
        results["eln_enrichment"] = eln_res

        # stacked bar
        tab = pd.crosstab(groups, merged[eln_col], normalize="index")
        ax = tab.plot(kind="bar", stacked=True, figsize=(6,4), color=["#2ecc71","#f39c12","#e74c3c"])
        ax.set_xlabel("Persister tertile"); ax.set_ylabel("Proportion"); ax.set_title(f"{cohort_name} | ELN by tertile")
        ax.legend(title="ELN", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{cohort_name}_eln_by_tertile.png", dpi=300)
        plt.close()

    # 3) MRD binary
    if mrd_bin_col:
        mrd_res = mrd_binary_stats(prob, merged[mrd_bin_col])
        results["mrd_binary"] = mrd_res

        data = pd.DataFrame({"prob": prob, "MRD": merged[mrd_bin_col].astype(str)})
        order = ["MRD-","MRD+"] if set(data["MRD"]) & {"MRD-","MRD+"} else sorted(data["MRD"].unique())
        plt.figure(figsize=(5,4))
        parts = plt.violinplot([data.loc[data["MRD"]==lab,"prob"] for lab in order], positions=[1,2], widths=0.8, showmeans=True)
        plt.xticks([1,2], order); plt.ylabel("Persister probability"); plt.title(f"{cohort_name} | MRD")
        if "mannwhitney_p" in mrd_res:
            ylim = plt.gca().get_ylim()
            plt.text(1.5, ylim[1]*0.95, f"MWU p={mrd_res['mannwhitney_p']:.3g}", ha="center")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{cohort_name}_mrd_binary.png", dpi=300)
        plt.close()

    # 4) MRD continuous / blasts
    if mrd_cont_col:
        fig, ax = plt.subplots(figsize=(5,4))
        x = prob; y = pd.to_numeric(merged[mrd_cont_col], errors="coerce").to_numpy()
        scatter_with_fit(x, y, ax, "Persister probability", "MRD (continuous)", f"{cohort_name} | MRD (cont.)")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{cohort_name}_mrd_continuous.png", dpi=300)
        plt.close()

        r, p = spearmanr(x, y, nan_policy="omit")
        results["mrd_continuous"] = {"spearman_r": float(r), "p": float(p), "n": int(np.isfinite(y).sum())}

    if blast_col:
        fig, ax = plt.subplots(figsize=(5,4))
        y = pd.to_numeric(merged[blast_col], errors="coerce").to_numpy()
        scatter_with_fit(prob, y, ax, "Persister probability", "Blast %", f"{cohort_name} | Blasts")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{cohort_name}_blasts.png", dpi=300)
        plt.close()

        r, p = spearmanr(prob, y, nan_policy="omit")
        results["blasts"] = {"spearman_r": float(r), "p": float(p), "n": int(np.isfinite(y).sum())}

    # save merged table for debugging
    merged.to_csv(outdir / f"{cohort_name}_merged_table.csv", index=False)

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True,
                    help="Path to your AML_Persister_Analysis/results")
    ap.add_argument("--tau", type=float, default=0.31, help="Pre-specified threshold to summarize prevalence")
    ap.add_argument("--include", nargs="*", default=None,
                    help="Optional subset of cohorts to include. Use names: BeatAML, TCGA-LAML")
    args = ap.parse_args()

    root = Path(args.results_root)
    out_root = root / "clinical_enrichment"
    out_root.mkdir(parents=True, exist_ok=True)

    # discover cohorts
    candidates = {
        "BeatAML": root / "bulk_BeatAML",
        "TCGA-LAML": root / "bulk_TCGA",
    }
    if args.include:
        cohorts = {k:v for k,v in candidates.items() if k in args.include}
    else:
        cohorts = {k:v for k,v in candidates.items() if v.exists()}

    all_results = {}
    for name, cdir in cohorts.items():
        print(f"[INFO] Analyzing {name} in {cdir}")
        try:
            cres = analyze_cohort(cdir, name, out_root, tau=args.tau)
            all_results[name] = cres
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
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "enrichment_summary.csv", index=False)

    with open(out_root / "enrichment_results.json","w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[OK] Saved outputs → {out_root}")

if __name__ == "__main__":
    main()
