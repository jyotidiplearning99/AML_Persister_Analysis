from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal, mannwhitneyu

warnings.filterwarnings("ignore")

ROOT = Path(".")
PRED = Path("results/bulk_BeatAML/predictions_60pct.csv")
BRIDGE = Path("data/survival/beataml_persister_survival_matched.csv")
OUT = Path("results/markus_validation/disease_severity")
OUT.mkdir(parents=True, exist_ok=True)

PRIORITY_FILES = [
    Path("results/clinical_real_final/BeatAML_clinical_real.csv"),
    Path("results/bulk_BeatAML/clinical_real.csv"),
    Path("data/beataml_drugs/beataml_waves_clinical.csv"),
    Path("results/clinical_enrichment_real_data/BeatAML_data.csv"),
    Path("data/beataml_drugs/beataml_clinical.xlsx"),
]

GENES = [
    "TP53", "RUNX1", "ASXL1", "FLT3", "NPM1",
    "DNMT3A", "IDH1", "IDH2", "TET2",
    "NRAS", "KRAS", "CEBPA"
]


def norm_col(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def norm_id(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x in ("", "nan", "none"):
        return np.nan
    return x


def short_sample(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"(\d+-\d+)", str(x))
    return m.group(1).lower() if m else np.nan


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)

    valid = np.isfinite(p)
    pv = p[valid]

    if len(pv) == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1)

    original = np.empty(n)
    original[order] = adjusted
    q[valid] = original

    return q


def read_table(path):
    if path.suffix.lower() in [".xlsx", ".xls"]:
        xl = pd.ExcelFile(path)
        best = None
        best_sheet = None

        for s in xl.sheet_names:
            try:
                d = pd.read_excel(path, sheet_name=s)
                if best is None or (len(d) * max(len(d.columns), 1)) > (
                    len(best) * max(len(best.columns), 1)
                ):
                    best = d
                    best_sheet = s
            except Exception:
                pass

        if best is None:
            raise ValueError("No readable Excel sheet")

        print("  Excel sheet selected:", best_sheet)
        return best

    return pd.read_csv(path, low_memory=False)


# ============================================================
# VERIFIED PERSISTER SCORE
# ============================================================

score = pd.read_csv(PRED)

score = score[
    ["sample_id", "persister_probability"]
].copy()

score["sample_id"] = score["sample_id"].astype(str)

score = score[
    score["sample_id"].str.lower() != "entrez_gene_id"
].copy()

score["persister_probability"] = pd.to_numeric(
    score["persister_probability"],
    errors="coerce"
)

score = score.dropna(
    subset=["sample_id", "persister_probability"]
)

score = score.drop_duplicates("sample_id")

print("=" * 80)
print("VERIFIED BEATAML PERSISTER SCORE")
print("=" * 80)
print("N =", len(score))
print(
    "Score range:",
    score.persister_probability.min(),
    "to",
    score.persister_probability.max()
)


# ============================================================
# SAMPLE <-> PATIENT BRIDGE
# ============================================================

bridge = pd.read_csv(BRIDGE)

keep = [
    c for c in
    ["sample_id", "PATIENT_ID", "SAMPLE_ID"]
    if c in bridge.columns
]

bridge = bridge[keep].drop_duplicates("sample_id")

base = score.merge(
    bridge,
    on="sample_id",
    how="left"
)

base["sample_full_key"] = base["sample_id"].map(norm_id)
base["sample_short_key"] = base["sample_id"].map(short_sample)

if "SAMPLE_ID" in base.columns:
    base["bridge_sample_key"] = base["SAMPLE_ID"].map(norm_id)
else:
    base["bridge_sample_key"] = np.nan

if "PATIENT_ID" in base.columns:
    base["patient_key"] = base["PATIENT_ID"].map(norm_id)
else:
    base["patient_key"] = np.nan


# ============================================================
# DISCOVER CLINICAL FILES
# ============================================================

candidate_files = []

for f in PRIORITY_FILES:
    if f.exists():
        candidate_files.append(f)

patterns = [
    "data/**/*beataml*clinical*.csv",
    "data/**/*BeatAML*clinical*.csv",
    "results/**/*beataml*clinical*.csv",
    "results/**/*BeatAML*clinical*.csv",
    "data/**/*beataml*waves*.csv",
]

for pat in patterns:
    for f in ROOT.glob(pat):
        if f not in candidate_files:
            candidate_files.append(f)

print("\n" + "=" * 80)
print("CLINICAL FILE DISCOVERY")
print("=" * 80)

for f in candidate_files:
    print(f)


# ============================================================
# JOIN CLINICAL TABLE TO VERIFIED SCORE
# ============================================================

def relevant_columns(df):
    out = []

    for c in df.columns:
        n = norm_col(c)

        if (
            "eln" in n
            or "risk" in n
            or "cytogen" in n
            or "karyotype" in n
            or "blast" in n
            or n == "age"
            or n.startswith("age_")
            or "age_at" in n
            or "mutation" in n
            or "mutated" in n
            or any(g.lower() in n for g in GENES)
        ):
            out.append(c)

    return out


def best_join(clin):
    cols = list(clin.columns)

    patient_cols = [
        c for c in cols
        if "patient" in norm_col(c)
        and "id" in norm_col(c)
    ]

    sample_cols = [
        c for c in cols
        if (
            ("sample" in norm_col(c) and "id" in norm_col(c))
            or ("specimen" in norm_col(c) and "id" in norm_col(c))
        )
    ]

    trials = []

    for c in patient_cols:
        temp = clin.copy()
        temp["__key"] = temp[c].map(norm_id)
        temp = temp.dropna(subset=["__key"]).drop_duplicates("__key")

        x = base.merge(
            temp,
            left_on="patient_key",
            right_on="__key",
            how="inner",
            suffixes=("", "_clinical")
        )

        trials.append(("patient", c, x))

    for c in sample_cols:
        # Full ID
        temp = clin.copy()
        temp["__key"] = temp[c].map(norm_id)
        temp = temp.dropna(subset=["__key"]).drop_duplicates("__key")

        x = base.merge(
            temp,
            left_on="sample_full_key",
            right_on="__key",
            how="inner",
            suffixes=("", "_clinical")
        )

        trials.append(("sample_full", c, x))

        # BeatAML compact sample ID
        x = base.merge(
            temp,
            left_on="bridge_sample_key",
            right_on="__key",
            how="inner",
            suffixes=("", "_clinical")
        )

        trials.append(("sample_bridge", c, x))

        # Extract XX-XXXXX form
        temp2 = clin.copy()
        temp2["__key"] = temp2[c].map(short_sample)
        temp2 = temp2.dropna(subset=["__key"]).drop_duplicates("__key")

        x = base.merge(
            temp2,
            left_on="sample_short_key",
            right_on="__key",
            how="inner",
            suffixes=("", "_clinical")
        )

        trials.append(("sample_short", c, x))

    if not trials:
        return None

    trials.sort(
        key=lambda z: z[2]["sample_id"].nunique()
        if "sample_id" in z[2].columns else len(z[2]),
        reverse=True
    )

    return trials[0]


usable = []

for f in candidate_files:

    print("\n" + "-" * 80)
    print("FILE:", f)

    try:
        d = read_table(f)
    except Exception as e:
        print("READ ERROR:", e)
        continue

    print("Shape:", d.shape)

    rel = relevant_columns(d)
    print("Relevant columns:", rel)

    j = best_join(d)

    if j is None:
        print("No usable patient/sample ID column found")
        continue

    mode, idcol, merged = j

    sample_col = "sample_id" if "sample_id" in merged.columns else (
        "sample_id_x" if "sample_id_x" in merged.columns else None
    )
    if sample_col is None:
        print("Merged columns:", merged.columns.tolist())
        continue
    nmatch = merged[sample_col].nunique()
    if sample_col != "sample_id":
        merged = merged.rename(columns={sample_col: "sample_id"})

    print(
        f"Best join: {mode} using {idcol}; "
        f"matched score samples = {nmatch}"
    )

    if nmatch >= 50 and len(rel) > 0:
        usable.append({
            "file": f,
            "clinical": d,
            "merged": merged,
            "mode": mode,
            "idcol": idcol,
            "nmatch": nmatch,
            "relevant": rel
        })


if not usable:
    raise SystemExit(
        "\nNo clinical table with both usable identifiers "
        "and severity/risk variables was found."
    )


# Prefer lots of matched patients AND useful variables
usable.sort(
    key=lambda z: (
        len(z["relevant"]),
        z["nmatch"]
    ),
    reverse=True
)

chosen = usable[0]

df = chosen["merged"].copy()

print("\n" + "=" * 80)
print("SELECTED CLINICAL TABLE")
print("=" * 80)
print("File:", chosen["file"])
print("Join:", chosen["mode"], "/", chosen["idcol"])
print("Matched score samples:", chosen["nmatch"])
print("Relevant columns:")
for c in chosen["relevant"]:
    print("  ", c)


# ============================================================
# AVOID REPEATED PATIENTS WHERE POSSIBLE
# ============================================================

if "PATIENT_ID" in df.columns:
    before = len(df)

    df = df.sort_values("sample_id").drop_duplicates(
        subset=["PATIENT_ID"],
        keep="first"
    )

    print(
        f"\nPatient independence check: {before} rows -> "
        f"{len(df)} unique patients"
    )


score_col = "persister_probability"

results = []


# ============================================================
# NUMERIC ASSOCIATIONS: AGE / BLASTS
# ============================================================

numeric_candidates = []

for c in chosen["relevant"]:
    n = norm_col(c)

    if (
        n == "age"
        or n.startswith("age_")
        or "age_at" in n
        or "blast" in n
    ):
        numeric_candidates.append(c)


for c in numeric_candidates:

    x = pd.DataFrame({
        "score": pd.to_numeric(df[score_col], errors="coerce"),
        "value": pd.to_numeric(df[c], errors="coerce")
    }).dropna()

    if len(x) < 20 or x["value"].nunique() < 5:
        continue

    rho, p = spearmanr(
        x["score"],
        x["value"]
    )

    results.append({
        "domain": "numeric",
        "variable": c,
        "test": "Spearman",
        "n": len(x),
        "effect": rho,
        "p": p,
        "details": (
            f"median clinical value={x['value'].median():.3g}"
        )
    })


# ============================================================
# ELN / CYTOGENETIC / RISK CATEGORIES
# ============================================================

risk_candidates = [
    c for c in chosen["relevant"]
    if any(
        token in norm_col(c)
        for token in ["eln", "risk", "cytogen", "karyotype"]
    )
]


for c in risk_candidates:

    temp = pd.DataFrame({
        "score": pd.to_numeric(df[score_col], errors="coerce"),
        "group": df[c]
    }).dropna()

    temp["group"] = (
        temp["group"]
        .astype(str)
        .str.strip()
    )

    counts = temp["group"].value_counts()

    valid_groups = counts[counts >= 5].index
    temp = temp[temp["group"].isin(valid_groups)]

    if temp["group"].nunique() < 2:
        continue

    if temp["group"].nunique() > 12:
        continue

    groups = [
        g["score"].values
        for _, g in temp.groupby("group")
    ]

    stat, p = kruskal(*groups)

    med = (
        temp.groupby("group")["score"]
        .agg(["count", "median"])
        .round(4)
        .to_dict("index")
    )

    results.append({
        "domain": "risk_category",
        "variable": c,
        "test": "Kruskal-Wallis",
        "n": len(temp),
        "effect": stat,
        "p": p,
        "details": str(med)
    })

    # Ordered biological risk trend where labels permit it
    def map_risk(v):
        s = str(v).lower()

        if any(k in s for k in ["favorable", "favourable", "good", "low"]):
            return 0
        if any(k in s for k in ["intermediate", "intermed", "standard"]):
            return 1
        if any(k in s for k in ["adverse", "poor", "high"]):
            return 2

        return np.nan

    temp["risk_order"] = temp["group"].map(map_risk)

    ordered = temp.dropna(subset=["risk_order"])

    if (
        len(ordered) >= 20
        and ordered["risk_order"].nunique() >= 2
    ):
        rho, p2 = spearmanr(
            ordered["score"],
            ordered["risk_order"]
        )

        results.append({
            "domain": "risk_trend",
            "variable": c,
            "test": "Spearman ordered risk",
            "n": len(ordered),
            "effect": rho,
            "p": p2,
            "details": "0=favorable/low, 1=intermediate, 2=adverse/high"
        })


# ============================================================
# MUTATION ASSOCIATIONS
# ============================================================

def binary_mutation(series):

    s = series.copy()

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")

        vals = set(x.dropna().unique())

        if vals.issubset({0, 1}):
            return x

    def conv(v):
        if pd.isna(v):
            return np.nan

        z = str(v).strip().lower()

        if z in {
            "1", "yes", "y", "true", "positive",
            "pos", "mut", "mutated", "mutation",
            "present"
        }:
            return 1

        if z in {
            "0", "no", "n", "false", "negative",
            "neg", "wt", "wildtype", "wild-type",
            "absent"
        }:
            return 0

        return np.nan

    return s.map(conv)


mutation_results = []

# A) Dedicated gene columns
for gene in GENES:

    gene_cols = [
        c for c in df.columns
        if gene.lower() in norm_col(c)
    ]

    for c in gene_cols:

        mut = binary_mutation(df[c])

        temp = pd.DataFrame({
            "score": pd.to_numeric(df[score_col], errors="coerce"),
            "mut": mut
        }).dropna()

        if temp["mut"].nunique() != 2:
            continue

        n_mut = int((temp["mut"] == 1).sum())
        n_wt = int((temp["mut"] == 0).sum())

        if n_mut < 5 or n_wt < 5:
            continue

        a = temp.loc[temp["mut"] == 1, "score"]
        b = temp.loc[temp["mut"] == 0, "score"]

        U, p = mannwhitneyu(
            a,
            b,
            alternative="two-sided"
        )

        rbc = (2 * U / (len(a) * len(b))) - 1

        mutation_results.append({
            "gene": gene,
            "source_column": c,
            "n_mut": n_mut,
            "n_wt": n_wt,
            "median_mut": a.median(),
            "median_wt": b.median(),
            "rank_biserial": rbc,
            "p": p
        })


# B) Mutation-list columns
mutation_list_cols = [
    c for c in df.columns
    if (
        "mutation" in norm_col(c)
        or "mutated_genes" in norm_col(c)
        or "gene_mut" in norm_col(c)
    )
]

for c in mutation_list_cols:

    text = df[c].fillna("").astype(str)

    # Only use columns that look like lists/annotations rather than binary flags
    if text.nunique() < 5:
        continue

    for gene in GENES:

        mut = text.str.contains(
            rf"\b{re.escape(gene)}\b",
            case=False,
            regex=True
        ).astype(int)

        temp = pd.DataFrame({
            "score": pd.to_numeric(df[score_col], errors="coerce"),
            "mut": mut
        }).dropna()

        n_mut = int((temp["mut"] == 1).sum())
        n_wt = int((temp["mut"] == 0).sum())

        if n_mut < 5 or n_wt < 5:
            continue

        a = temp.loc[temp["mut"] == 1, "score"]
        b = temp.loc[temp["mut"] == 0, "score"]

        U, p = mannwhitneyu(
            a,
            b,
            alternative="two-sided"
        )

        rbc = (2 * U / (len(a) * len(b))) - 1

        mutation_results.append({
            "gene": gene,
            "source_column": c,
            "n_mut": n_mut,
            "n_wt": n_wt,
            "median_mut": a.median(),
            "median_wt": b.median(),
            "rank_biserial": rbc,
            "p": p
        })


# ============================================================
# SAVE RESULTS + FDR
# ============================================================

res = pd.DataFrame(results)

if len(res):
    res["q_FDR"] = bh_fdr(res["p"].values)
    res = res.sort_values(["q_FDR", "p"])

    res.to_csv(
        OUT / "severity_associations.csv",
        index=False
    )

mutres = pd.DataFrame(mutation_results)

if len(mutres):

    # Remove duplicate gene/source analyses if identical
    mutres = mutres.drop_duplicates(
        subset=[
            "gene",
            "source_column",
            "n_mut",
            "n_wt",
            "median_mut",
            "median_wt"
        ]
    )

    mutres["q_FDR"] = bh_fdr(mutres["p"].values)

    mutres = mutres.sort_values(
        ["q_FDR", "p"]
    )

    mutres.to_csv(
        OUT / "mutation_associations.csv",
        index=False
    )


# Save merged source for audit
df.to_csv(
    OUT / "BeatAML_score_clinical_merged.csv",
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "#" * 80)
print("DISEASE-SEVERITY / RISK RESULTS")
print("#" * 80)

if len(res):
    show = res[
        [
            "domain",
            "variable",
            "test",
            "n",
            "effect",
            "p",
            "q_FDR",
            "details"
        ]
    ]

    print(show.to_string(index=False))
else:
    print(
        "No analyzable age/blast/risk variables found "
        "in selected clinical table."
    )


print("\n" + "#" * 80)
print("MUTATION RESULTS")
print("#" * 80)

if len(mutres):
    print(
        mutres[
            [
                "gene",
                "source_column",
                "n_mut",
                "n_wt",
                "median_mut",
                "median_wt",
                "rank_biserial",
                "p",
                "q_FDR"
            ]
        ].to_string(index=False)
    )
else:
    print(
        "No analyzable mutation variables found in "
        "the selected clinical table."
    )


print("\n" + "#" * 80)
print("INTERPRETATION GUIDE")
print("#" * 80)

print(
    """
1. Blast/age:
   Spearman rho > 0 means higher persister score accompanies higher value.

2. ELN/cytogenetic risk:
   A significant Kruskal-Wallis test means score distributions differ
   among risk groups.

3. Ordered risk:
   Positive rho means score increases from favorable -> intermediate -> adverse.

4. Mutations:
   Positive rank-biserial means mutation-positive cases tend to have
   higher persister scores.

5. Use q_FDR < 0.05 as the main multiple-testing criterion.

Do NOT call the persister score a disease-severity score solely because one
mutation is associated. The strongest evidence would be a reproducible
association with ELN/cytogenetic risk and/or blast burden.
"""
)

print("Selected clinical source:", chosen["file"])
print("Outputs:", OUT)
