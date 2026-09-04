import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

BEAT_SURV = "data/survival/beataml_persister_survival_matched.csv"
BEAT_DRUG = "data/beataml_drugs/processed/beataml_key_drugs_auc.csv"
TCGA_SURV = "data/survival/tcga_persister_survival_matched.csv"

OUTDIR = "results/markus_validation"

import os
os.makedirs(OUTDIR, exist_ok=True)


def clean_sample_id(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x.lower() in {
        "entrez_gene_id",
        "gene_id",
        "sample_id",
        "nan",
        ""
    }:
        return np.nan
    return x


def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0)


def run_cox(df, cohort, adjust_age=False):
    cols = ["os_days", "os_status", "score_z"]

    if adjust_age and "age" in df.columns:
        cols.append("age")

    d = df[cols].copy()

    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna()
    d = d[d["os_days"] > 0]

    cph = CoxPHFitter()

    cph.fit(
        d,
        duration_col="os_days",
        event_col="os_status"
    )

    s = cph.summary.loc["score_z"]

    out = {
        "cohort": cohort,
        "model": "age_adjusted" if adjust_age else "univariate",
        "n": len(d),
        "events": int(d["os_status"].sum()),
        "HR_per_1SD": np.exp(s["coef"]),
        "CI95_low": np.exp(s["coef lower 95%"]),
        "CI95_high": np.exp(s["coef upper 95%"]),
        "p": s["p"],
    }

    print("\n" + "="*70)
    print(cohort, "-", out["model"])
    print("="*70)
    print(
        f"N={out['n']}, events={out['events']}\n"
        f"HR per 1-SD score = {out['HR_per_1SD']:.3f}\n"
        f"95% CI = {out['CI95_low']:.3f}–{out['CI95_high']:.3f}\n"
        f"p = {out['p']:.4g}"
    )

    return out


# ============================================================
# BEATAML
# ============================================================

beat = pd.read_csv(BEAT_SURV)

beat["sample_id"] = beat["sample_id"].map(clean_sample_id)

beat["persister_probability"] = pd.to_numeric(
    beat["persister_probability"],
    errors="coerce"
)

beat["os_days"] = pd.to_numeric(
    beat["os_days"],
    errors="coerce"
)

beat["os_status"] = pd.to_numeric(
    beat["os_status"],
    errors="coerce"
)

beat = beat.dropna(
    subset=[
        "sample_id",
        "persister_probability",
        "os_days",
        "os_status"
    ]
)

beat = beat[beat["os_days"] > 0].copy()

# Protect against duplicate samples
beat = beat.drop_duplicates(
    subset="sample_id",
    keep="first"
)

beat["score_z"] = zscore(
    beat["persister_probability"]
)

print("\n" + "#"*70)
print("BEATAML SURVIVAL COHORT")
print("#"*70)

print("N:", len(beat))
print("Deaths:", int(beat["os_status"].sum()))
print(
    "Score range:",
    beat["persister_probability"].min(),
    "to",
    beat["persister_probability"].max()
)


# ============================================================
# BEATAML SURVIVAL: continuous z-score
# ============================================================

cox_results = []

cox_results.append(
    run_cox(
        beat,
        cohort="BeatAML",
        adjust_age=False
    )
)


# ============================================================
# BeatAML KM tertiles - visualization/statistical supplement
# ============================================================

beat["score_tertile"] = pd.qcut(
    beat["persister_probability"],
    q=3,
    labels=["Low", "Middle", "High"]
)

lr = multivariate_logrank_test(
    beat["os_days"],
    beat["score_tertile"],
    beat["os_status"]
)

print("\nBeatAML score-tertile log-rank")
print("p =", lr.p_value)

tertile_summary = (
    beat
    .groupby("score_tertile", observed=True)
    .agg(
        n=("sample_id", "size"),
        deaths=("os_status", "sum"),
        median_score=("persister_probability", "median"),
        median_os_days=("os_days", "median")
    )
    .reset_index()
)

print(tertile_summary.to_string(index=False))

tertile_summary.to_csv(
    f"{OUTDIR}/BeatAML_survival_tertiles.csv",
    index=False
)


# ============================================================
# SAME BeatAML patients: score vs drug response
# ============================================================

drug = pd.read_csv(BEAT_DRUG)

drug["sample_id"] = drug["sample_id"].map(clean_sample_id)

drug = drug.dropna(subset=["sample_id"])
drug = drug.drop_duplicates("sample_id")

same = beat.merge(
    drug,
    on="sample_id",
    how="inner"
)

print("\n" + "#"*70)
print("BEATAML SAME-PATIENT SURVIVAL + DRUG COHORT")
print("#"*70)
print("N =", len(same))

drug_cols = [
    c for c in same.columns
    if c.endswith("_AUC")
]

drug_results = []

for col in drug_cols:

    d = same[
        ["persister_probability", col]
    ].dropna()

    if len(d) < 10:
        continue

    rho, p_rho = spearmanr(
        d["persister_probability"],
        d[col]
    )

    r, p_r = pearsonr(
        d["persister_probability"],
        d[col]
    )

    drug_name = col.replace("_AUC", "")

    drug_results.append({
        "drug": drug_name,
        "n": len(d),
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "pearson_r": r,
        "pearson_p": p_r
    })

drug_results = pd.DataFrame(drug_results)

print("\nPersister score vs AUC")
print(drug_results.round(4).to_string(index=False))

drug_results.to_csv(
    f"{OUTDIR}/BeatAML_same_patient_drug_correlations.csv",
    index=False
)


# ============================================================
# Re-run survival specifically in patients with BOTH endpoints
# ============================================================

same["score_z"] = zscore(
    same["persister_probability"]
)

cox_results.append(
    run_cox(
        same,
        cohort="BeatAML_same_patients",
        adjust_age=False
    )
)


# ============================================================
# TCGA
# ============================================================

tcga = pd.read_csv(TCGA_SURV)

tcga["sample_id"] = tcga["sample_id"].map(clean_sample_id)

tcga["persister_probability"] = pd.to_numeric(
    tcga["persister_probability"],
    errors="coerce"
)

tcga["os_days"] = pd.to_numeric(
    tcga["os_days"],
    errors="coerce"
)

tcga["os_status"] = pd.to_numeric(
    tcga["os_status"],
    errors="coerce"
)

tcga["age"] = pd.to_numeric(
    tcga["age"],
    errors="coerce"
)

tcga = tcga.dropna(
    subset=[
        "sample_id",
        "persister_probability",
        "os_days",
        "os_status"
    ]
)

tcga = tcga[tcga["os_days"] > 0].copy()

# One patient per survival observation
if "patient_id" in tcga.columns:
    tcga = tcga.drop_duplicates(
        subset="patient_id",
        keep="first"
    )
else:
    tcga = tcga.drop_duplicates(
        subset="sample_id",
        keep="first"
    )

tcga["score_z"] = zscore(
    tcga["persister_probability"]
)

print("\n" + "#"*70)
print("TCGA-LAML SURVIVAL COHORT")
print("#"*70)

print("N:", len(tcga))
print("Deaths:", int(tcga["os_status"].sum()))

cox_results.append(
    run_cox(
        tcga,
        cohort="TCGA-LAML",
        adjust_age=False
    )
)

cox_results.append(
    run_cox(
        tcga,
        cohort="TCGA-LAML",
        adjust_age=True
    )
)


# ============================================================
# FINAL TABLE
# ============================================================

cox_df = pd.DataFrame(cox_results)

cox_df.to_csv(
    f"{OUTDIR}/survival_cox_summary.csv",
    index=False
)

same.to_csv(
    f"{OUTDIR}/BeatAML_same_patient_dataset.csv",
    index=False
)

print("\n" + "#"*70)
print("FINAL SURVIVAL SUMMARY")
print("#"*70)

print(
    cox_df.round(4).to_string(index=False)
)

print("\nResults saved to:")
print(OUTDIR)
