#!/usr/bin/env python3
"""
Hypothesis-Generating Drug Analysis (N~23 is too small for FDR)
1) Rank by effect size + p-value threshold
2) Leave-one-out (LOO) stability (robust to missing/constant subsets)
3) Filter to high-variance drugs
4) Confound checks: global DSS + (optional) doubling time partial correlation
5) Save plots: AT9283 scatter, AT9283 LOO influence, Top-10 heatmap
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, t

import matplotlib
matplotlib.use("Agg")  # HPC-safe
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("/scratch/project_2010376/JDs_Project/drug_analysis_hypothesis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HYPOTHESIS-GENERATING DRUG ANALYSIS")
print("=" * 80)

# =============================================================================
# Helpers
# =============================================================================

def pct_to_logit(pct: float) -> float:
    prob = pct / 100.0
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(prob) - np.log(1 - prob)

def safe_spearman(x, y):
    """Spearman that returns (nan, nan) if constant/invalid."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.nan, np.nan
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan, np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan
    r, p = spearmanr(x, y)
    if np.isnan(r) or np.isnan(p):
        return np.nan, np.nan
    return float(r), float(p)

def loo_stability(pers_vals, dss_vals):
    """
    LOO correlation stability:
    - skips LOO splits that become constant/invalid (prevents ConstantInputWarning)
    - stability computed over VALID LOO runs only
    """
    r_full, p_full = safe_spearman(pers_vals, dss_vals)
    if not np.isfinite(r_full):
        return None

    pers_vals = np.asarray(pers_vals, dtype=float)
    dss_vals = np.asarray(dss_vals, dtype=float)

    loo_rs = []
    for i in range(len(pers_vals)):
        mask = np.arange(len(pers_vals)) != i
        r_loo, _ = safe_spearman(pers_vals[mask], dss_vals[mask])
        if np.isfinite(r_loo):
            loo_rs.append(r_loo)

    if len(loo_rs) == 0:
        return None

    loo_rs = np.array(loo_rs, dtype=float)
    sign_stable = np.mean(np.sign(loo_rs) == np.sign(r_full))

    return {
        "r_full": r_full,
        "p_full": p_full,
        "loo_median": float(np.nanmedian(loo_rs)),
        "loo_min": float(np.nanmin(loo_rs)),
        "loo_max": float(np.nanmax(loo_rs)),
        "loo_std": float(np.nanstd(loo_rs)),
        "sign_stable": float(sign_stable),
        "loo_valid_n": int(len(loo_rs)),
    }

def canonicalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    for ch in [" ", "-", "_", ".", "/", "\\"]:
        s = s.replace(ch, "")
    return s

def find_first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

def partial_spearman_rank_residual(x, y, c):
    """
    Partial Spearman ~ partial Pearson on ranks.
    Returns (r_partial, p_approx, n).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(c, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[ok], y[ok], c[ok]
    n = len(x)
    if n < 8:
        return np.nan, np.nan, n
    if np.std(c) < 1e-12:
        return np.nan, np.nan, n

    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    rc = pd.Series(c).rank(method="average").to_numpy()

    # residualize ranks vs rc
    A = np.column_stack([rc, np.ones_like(rc)])
    bx, _, _, _ = np.linalg.lstsq(A, rx, rcond=None)
    by, _, _, _ = np.linalg.lstsq(A, ry, rcond=None)
    rx_res = rx - (A @ bx)
    ry_res = ry - (A @ by)

    if np.std(rx_res) < 1e-12 or np.std(ry_res) < 1e-12:
        return np.nan, np.nan, n

    r = float(np.corrcoef(rx_res, ry_res)[0, 1])

    # approximate p-value via t-test with dof = n - k - 2, k=1 covariate => dof=n-3
    dof = n - 3
    if dof <= 0 or not np.isfinite(r):
        return r, np.nan, n
    t_stat = r * np.sqrt(dof / (1 - r**2 + 1e-12))
    p = float(2 * (1 - t.cdf(np.abs(t_stat), dof)))
    return r, p, n

# =============================================================================
# Load persister scores (logit-based)
# =============================================================================

bulk_scores = pd.read_csv("/scratch/project_2010376/JDs_Project/drug_persister_correlation/persister_scores.csv")
scrna_score = pd.read_csv("/scratch/project_2010376/JDs_Project/scrna_persister_scores/mv411_persister_score.csv")
mv411_pct = float(scrna_score["persister_pct"].iloc[0])

persister_logits = {}
for _, row in bulk_scores.iterrows():
    cl = row["Cell_Line"]
    pct = mv411_pct if cl == "MV4-11" else float(row["Persister_Pct"])
    persister_logits[cl] = pct_to_logit(pct)

# Remove duplicate
persister_logits.pop("HL-60_TB", None)

vals = np.array(list(persister_logits.values()), dtype=float)
print(f"\nPersister logit: N={len(vals)}, std={vals.std():.3f}, range=[{vals.min():.2f}, {vals.max():.2f}]")

# Order cell lines by persister for plotting/heatmaps
cell_order = sorted(persister_logits.keys(), key=lambda k: persister_logits[k])

# =============================================================================
# Load drug DSS data
# =============================================================================

dss_df = pd.read_excel(
    "/scratch/project_2010376/JDs_Project/drug_data/41375_2020_978_MOESM7_ESM_AML cell lines.xlsx",
    sheet_name="DSS",
    skiprows=2,
)

# =============================================================================
# Build a column map for each cell line (best matching DSS column)
# =============================================================================

dss_cols = set(map(str, dss_df.columns))
cl_to_col = {}
for cl in persister_logits.keys():
    variants = [cl, cl.replace("-", "_"), cl.replace("-", ""), cl.replace("_", "-")]
    chosen = None
    for v in variants:
        if v in dss_cols:
            chosen = v
            break
    if chosen:
        cl_to_col[cl] = chosen

missing_cols = [cl for cl in persister_logits.keys() if cl not in cl_to_col]
if missing_cols:
    print(f"\nWARNING: DSS columns missing for {len(missing_cols)} lines: {missing_cols}")

# =============================================================================
# Confound check A: Global DSS vs persister logit
# =============================================================================

global_dss = {}
for cl, colname in cl_to_col.items():
    series = pd.to_numeric(dss_df[colname], errors="coerce")
    m = float(np.nanmean(series.to_numpy()))
    global_dss[cl] = m

aligned_cls = [cl for cl in cell_order if cl in global_dss and np.isfinite(global_dss[cl])]
x_pers = np.array([persister_logits[cl] for cl in aligned_cls], dtype=float)
y_gdss = np.array([global_dss[cl] for cl in aligned_cls], dtype=float)
r_g, p_g = safe_spearman(x_pers, y_gdss)

print("\n" + "=" * 80)
print("CONFOUND CHECKS")
print("=" * 80)
print(f"Global mean DSS vs persister logit: R={r_g:+.3f} p={p_g:.4g} (N={len(aligned_cls)})")

# Plot global DSS confound scatter
plt.figure(figsize=(8, 6))
plt.scatter(x_pers, y_gdss)
for cl, x, y in zip(aligned_cls, x_pers, y_gdss):
    plt.text(x, y, cl, fontsize=8)
plt.xlabel("Persister logit")
plt.ylabel("Global mean DSS (across all drugs)")
plt.title("Confound check: global DSS vs persister score")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confound_global_dss_scatter.png", dpi=200)
plt.close()

print(f"✓ Saved: {OUTPUT_DIR / 'confound_global_dss_scatter.png'}")

# =============================================================================
# Confound check B (optional): doubling time from DepMap Model metadata
# =============================================================================

depmap_model_path = find_first_existing([
    "/scratch/project_2010376/JDs_Project/depmap/Model.csv",
    "/scratch/project_2010376/JDs_Project/DepMap/Model.csv",
    "/scratch/project_2010376/JDs_Project/depmap/model.csv",
    "/scratch/project_2010376/JDs_Project/depmap/DepMap_Model.csv",
])

depmap_dt = {}  # cl -> doubling time
dt_col_used = None

if depmap_model_path:
    try:
        model_df = pd.read_csv(depmap_model_path)
        # identify a name column
        name_cols = [c for c in model_df.columns if c.lower() in ["cell_line_name", "cellline", "line", "stripped_cell_line_name", "ccle_name", "cell_line"]]
        if not name_cols:
            # fallback: use CCLE_Name if present
            name_cols = [c for c in model_df.columns if "ccle" in c.lower() and "name" in c.lower()]
        name_col = name_cols[0] if name_cols else None

        # identify a doubling time column
        dt_candidates = [c for c in model_df.columns if ("doubling" in c.lower() and "time" in c.lower())]
        dt_col = dt_candidates[0] if dt_candidates else None
        dt_col_used = dt_col

        if name_col and dt_col:
            model_df["_key"] = model_df[name_col].astype(str).apply(canonicalize_name)

            # Some CCLE_Name values include tissue suffixes; keep first chunk before "_" too
            def key_variants(s):
                s = str(s)
                base = s.split("_")[0]
                return {canonicalize_name(s), canonicalize_name(base)}

            key_to_dt = {}
            for _, r in model_df.iterrows():
                dtv = pd.to_numeric(r.get(dt_col), errors="coerce")
                if not np.isfinite(dtv):
                    continue
                for kv in key_variants(r.get(name_col)):
                    if kv and kv not in key_to_dt:
                        key_to_dt[kv] = float(dtv)

            for cl in persister_logits.keys():
                k = canonicalize_name(cl)
                if k in key_to_dt:
                    depmap_dt[cl] = key_to_dt[k]

            n_dt = len(depmap_dt)
            print(f"\nDepMap Model found: {depmap_model_path}")
            print(f"Doubling time column: {dt_col_used}")
            print(f"Matched doubling time for {n_dt}/{len(persister_logits)} cell lines")

            # print correlation persister vs doubling time
            cls_dt = [cl for cl in cell_order if cl in depmap_dt]
            x_dt = np.array([persister_logits[cl] for cl in cls_dt], dtype=float)
            y_dt = np.array([depmap_dt[cl] for cl in cls_dt], dtype=float)
            r_dt, p_dt = safe_spearman(x_dt, y_dt)
            print(f"Persister logit vs doubling time: R={r_dt:+.3f} p={p_dt:.4g} (N={len(cls_dt)})")
        else:
            print(f"\nDepMap Model found ({depmap_model_path}) but couldn't identify name/doubling-time columns.")
    except Exception as e:
        print(f"\nWARNING: Failed reading DepMap model metadata: {e}")
else:
    print("\nDepMap Model.csv not found in expected locations; skipping doubling-time confound checks.")

# =============================================================================
# Compute correlations with LOO stability
# =============================================================================

print("\nComputing correlations with LOO stability...")

results = []
drug_to_points = {}  # store raw points for plots/heatmaps (per drug)

for _, row in dss_df.iterrows():
    drug = row.get("Drug name", None)
    if pd.isna(drug):
        continue
    drug = str(drug)

    dss_vals = []
    pers_vals = []
    cls_used = []

    for cl, logit in persister_logits.items():
        colname = cl_to_col.get(cl)
        if not colname:
            continue
        dss = row.get(colname, np.nan)
        if pd.notna(dss):
            dss_vals.append(float(dss))
            pers_vals.append(float(logit))
            cls_used.append(cl)

    if len(dss_vals) >= 8:
        dss_std = float(np.std(dss_vals))
        if np.std(pers_vals) > 0 and dss_std > 0:
            stability = loo_stability(pers_vals, dss_vals)
            if stability is None:
                continue

            results.append({
                "Drug": drug,
                "Spearman_R": stability["r_full"],
                "P_value": stability["p_full"],
                "N": len(dss_vals),
                "DSS_std": dss_std,
                "LOO_median_R": stability["loo_median"],
                "LOO_min_R": stability["loo_min"],
                "LOO_max_R": stability["loo_max"],
                "LOO_std": stability["loo_std"],
                "Sign_stable_pct": stability["sign_stable"] * 100.0,
                "LOO_valid_n": stability["loo_valid_n"],
            })

            drug_to_points[drug] = {
                "cell_lines": cls_used,
                "pers": np.array(pers_vals, dtype=float),
                "dss": np.array(dss_vals, dtype=float),
            }

results_df = pd.DataFrame(results)
print(f"✓ Analyzed {len(results_df)} drugs")

# =============================================================================
# Filter and rank
# =============================================================================

high_var = results_df[results_df["DSS_std"] >= 5.0].copy()
print(f"✓ {len(high_var)} drugs with DSS std >= 5.0")

high_var["Score"] = (
    np.abs(high_var["Spearman_R"]) * 2.0 +
    (-np.log10(high_var["P_value"] + 1e-10)) +
    (high_var["Sign_stable_pct"] / 100.0)
)

# Optional: partial correlation controlling doubling time
if depmap_dt:
    partial_r_list = []
    partial_p_list = []
    partial_n_list = []

    for _, r in high_var.iterrows():
        drug = r["Drug"]
        pts = drug_to_points.get(drug)
        if not pts:
            partial_r_list.append(np.nan)
            partial_p_list.append(np.nan)
            partial_n_list.append(0)
            continue

        # align to those cell lines that have doubling time
        cls_used = pts["cell_lines"]
        x = pts["pers"]
        y = pts["dss"]
        c = np.array([depmap_dt.get(cl, np.nan) for cl in cls_used], dtype=float)

        rp, pp, nn = partial_spearman_rank_residual(x, y, c)
        partial_r_list.append(rp)
        partial_p_list.append(pp)
        partial_n_list.append(nn)

    high_var["PartialR_DoublingTime"] = partial_r_list
    high_var["PartialP_DoublingTime"] = partial_p_list
    high_var["PartialN_DoublingTime"] = partial_n_list

high_var = high_var.sort_values("Score", ascending=False)
high_var.to_csv(OUTPUT_DIR / "drug_candidates_ranked.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'drug_candidates_ranked.csv'}")

# =============================================================================
# Reporting
# =============================================================================

resisters = high_var[
    (high_var["Spearman_R"] < -0.4) &
    (high_var["P_value"] < 0.01) &
    (high_var["Sign_stable_pct"] > 70)
].head(15)

killers = high_var[
    (high_var["Spearman_R"] > 0.4) &
    (high_var["P_value"] < 0.01) &
    (high_var["Sign_stable_pct"] > 70)
].head(15)

print("\n" + "=" * 80)
print("TOP CANDIDATES (Hypothesis Generation)")
print("=" * 80)

print("\n" + "-" * 80)
print("PERSISTER-RESISTANT DRUGS (R<-0.4, p<0.01, LOO stable)")
print("-" * 80)

if not resisters.empty:
    print(f"Found {len(resisters)} candidates\n")
    cols = ["Drug", "Spearman_R", "P_value", "LOO_median_R", "Sign_stable_pct", "N"]
    if "PartialR_DoublingTime" in resisters.columns:
        cols += ["PartialR_DoublingTime", "PartialP_DoublingTime", "PartialN_DoublingTime"]
    print(resisters[cols].to_string(index=False))
else:
    print("None found with strict criteria")

print("\n" + "-" * 80)
print("PERSISTER-KILLING DRUGS (R>+0.4, p<0.01, LOO stable)")
print("-" * 80)

if not killers.empty:
    print(f"Found {len(killers)} candidates\n")
    cols = ["Drug", "Spearman_R", "P_value", "LOO_median_R", "Sign_stable_pct", "N"]
    if "PartialR_DoublingTime" in killers.columns:
        cols += ["PartialR_DoublingTime", "PartialP_DoublingTime", "PartialN_DoublingTime"]
    print(killers[cols].to_string(index=False))
else:
    print("None found with strict criteria")

# =============================================================================
# Exploratory candidates (clean split by sign)
# =============================================================================

print("\n" + "=" * 80)
print("EXPLORATORY CANDIDATES (Relaxed: |R|>0.3, p<0.05)")
print("=" * 80)

exploratory = high_var[
    (np.abs(high_var["Spearman_R"]) > 0.3) &
    (high_var["P_value"] < 0.05)
].copy()

print(f"\nTotal exploratory: {len(exploratory)}")

neg = exploratory[exploratory["Spearman_R"] < 0].sort_values("Spearman_R").head(10)
pos = exploratory[exploratory["Spearman_R"] > 0].sort_values("Spearman_R", ascending=False).head(10)

print("\nTop negative correlations:")
if len(neg) == 0:
    print("  None")
else:
    for i, (_, r) in enumerate(neg.iterrows(), 1):
        print(f"{i:2d}. {r['Drug']:<30} R={r['Spearman_R']:+.3f} p={r['P_value']:.4f} stable={r['Sign_stable_pct']:.0f}%")

print("\nTop positive correlations:")
if len(pos) == 0:
    print("  None")
else:
    for i, (_, r) in enumerate(pos.iterrows(), 1):
        print(f"{i:2d}. {r['Drug']:<30} R={r['Spearman_R']:+.3f} p={r['P_value']:.4f} stable={r['Sign_stable_pct']:.0f}%")

# =============================================================================
# Plots: AT9283 scatter + LOO influence + Top10 heatmap
# =============================================================================

def plot_at9283_scatter():
    drug = "AT9283"
    if drug not in drug_to_points:
        print("\nWARNING: AT9283 points not found for plotting.")
        return

    pts = drug_to_points[drug]
    cls_used = pts["cell_lines"]
    x = pts["pers"]
    y = pts["dss"]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    for cl, xi, yi in zip(cls_used, x, y):
        plt.text(xi, yi, cl, fontsize=8)
    plt.xlabel("Persister logit")
    plt.ylabel("DSS")
    plt.title("AT9283: DSS vs persister score (labeled)")
    plt.tight_layout()
    out = OUTPUT_DIR / "scatter_AT9283.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")

def plot_at9283_loo_influence():
    drug = "AT9283"
    if drug not in drug_to_points:
        return

    pts = drug_to_points[drug]
    cls_used = pts["cell_lines"]
    x = pts["pers"]
    y = pts["dss"]

    # compute LOO r per removed cell line (skip invalid)
    loo_rows = []
    for i, cl in enumerate(cls_used):
        mask = np.arange(len(x)) != i
        r_loo, _ = safe_spearman(x[mask], y[mask])
        loo_rows.append((cl, r_loo))
    loo_df = pd.DataFrame(loo_rows, columns=["Removed_CellLine", "R_loo"]).sort_values("R_loo")

    plt.figure(figsize=(10, 6))
    plt.barh(loo_df["Removed_CellLine"], loo_df["R_loo"])
    plt.xlabel("Spearman R (LOO)")
    plt.title("AT9283: Leave-one-out influence (remove each cell line)")
    plt.tight_layout()
    out = OUTPUT_DIR / "loo_influence_AT9283.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")

def plot_top10_heatmap():
    top = high_var.head(10)
    if top.empty:
        return

    drugs = top["Drug"].tolist()
    cls = cell_order

    M = np.full((len(drugs), len(cls)), np.nan, dtype=float)

    # build a quick lookup: drug -> DSS per cell line
    for i, drug in enumerate(drugs):
        # find row in DSS table
        rows = dss_df[dss_df["Drug name"].astype(str) == str(drug)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        for j, cl in enumerate(cls):
            colname = cl_to_col.get(cl)
            if not colname:
                continue
            v = row.get(colname, np.nan)
            if pd.notna(v):
                M[i, j] = float(v)

    plt.figure(figsize=(max(10, 0.45 * len(cls)), 6))
    im = plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="DSS")
    plt.yticks(np.arange(len(drugs)), drugs)
    plt.xticks(np.arange(len(cls)), cls, rotation=90)
    plt.title("Top 10 candidate drugs (by Score) × cell lines (ordered by persister)")
    plt.tight_layout()
    out = OUTPUT_DIR / "heatmap_top10_candidates.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")

plot_at9283_scatter()
plot_at9283_loo_influence()
plot_top10_heatmap()

# =============================================================================
# Final interpretation
# =============================================================================

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print(
    "N~23 is insufficient for FDR correction across ~285 drugs.\n"
    "Use this output for hypothesis generation:\n"
    "  1) Effect size ranking (Spearman R)\n"
    "  2) LOO stability (sign consistency + LOO spread)\n"
    "  3) High-variance filter (DSS std >= 5)\n"
    "\n"
    f"✓ Ranked CSV: {OUTPUT_DIR / 'drug_candidates_ranked.csv'}\n"
    f"✓ Plots: {OUTPUT_DIR / 'scatter_AT9283.png'}, {OUTPUT_DIR / 'loo_influence_AT9283.png'}, {OUTPUT_DIR / 'heatmap_top10_candidates.png'}\n"
    f"✓ Confound plot: {OUTPUT_DIR / 'confound_global_dss_scatter.png'}\n"
)
