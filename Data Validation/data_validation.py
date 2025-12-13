import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# === inputs ===
csv_path = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/Data Validation/merged_persister_13k_1k_no_duplicates_v3.csv")   # adjust path if needed
out_png  = csv_path.with_name("persister_agreement_colored.png")
out_xlsx = csv_path.with_name("persister_agreement_validated.xlsx")

# --- load ---
df = pd.read_csv(csv_path)

# --- robust column detection for 13k (x) and 1k (y) persister % ---
def pick(cols, *needles):
    return next((c for c in cols if all(n in c.lower() for n in needles)), None)

cols = [c.strip() for c in df.columns]
x_col = (pick(cols, "13k", "pers") or pick(cols, "13k", "%") or
         pick(cols, "13 k", "pers") or pick(cols, "p13") or pick(cols, "13k"))
y_col = (pick(cols, "1k", "pers") or pick(cols, "1k", "%") or
         pick(cols, "1 k", "pers") or pick(cols, "p1")  or pick(cols, "1k"))

# Fallback: pick the two most complete 0–100 numeric columns
if not x_col or not y_col:
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cands = []
    for c in numeric:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) >= 10 and s.min() >= -1 and s.max() <= 101:
            cands.append((len(s), c))
    cands.sort(reverse=True)
    x_col, y_col = cands[0][1], cands[1][1]

z = df[[x_col, y_col]].copy()
z = z.rename(columns={x_col: "x_13k", y_col: "y_1k"}).apply(pd.to_numeric, errors="coerce").dropna()

# --- infer labels (Healthy / Remission / Other) from text columns ---
id_cols = [c for c in df.columns if any(k in c.lower() for k in
            ["id","sample","name","dataset","cohort","patient","source","group","status","note"])]
def infer_label(row):
    text = " ".join(str(row.get(c, "")).lower() for c in id_cols)
    if any(k in text for k in ["norm","healthy","control","ctrl"]):   return "Healthy"
    if any(k in text for k in ["remission"," cr","cr ","cri","mrd","post"]): return "Remission"
    return "Other"
labels = df.apply(infer_label, axis=1)
z = z.join(labels.rename("Label"))

# --- stats (for validation / caption) ---
x, y = z["x_13k"].to_numpy(), z["y_1k"].to_numpy()
n = len(z)
r = float(np.corrcoef(x, y)[0, 1])
X = np.vstack([np.ones_like(x), x]).T
intercept, slope = np.linalg.lstsq(X, y, rcond=None)[0]
yhat = intercept + slope * x
rmse = float(np.sqrt(np.mean((y - yhat)**2)))
mae  = float(np.mean(np.abs(y - yhat)))

print("Detected columns:", x_col, "(13k) |", y_col, "(1k)")
print(f"n={n}, r={r:.3f}, slope={slope:.3f}, intercept={intercept:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

# --- save a validation table so plot == spreadsheet is checkable one-to-one ---
z_out = z.copy()
# carry over any identifying columns to help you spot the samples
for c in id_cols: z_out[c] = df[c]
z_out["pred_fit"] = yhat
z_out["residual"] = y - yhat
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as w:
    z_out.to_excel(w, index=False, sheet_name="points")
print("Wrote:", out_xlsx)

# --- plot (Okabe–Ito palette; no seaborn) ---
PALETTE = {"Healthy": "#009E73", "Remission": "#CC79A7", "Other": "#0072B2"}
MARKERS = {"Healthy": "^", "Remission": "s", "Other": "o"}

fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)
for grp, gdf in z.groupby("Label"):
    ax.scatter(gdf["x_13k"], gdf["y_1k"], s=42, marker=MARKERS.get(grp,"o"),
               c=PALETTE.get(grp,"#0072B2"), edgecolor="none", label=f"{grp} (n={len(gdf)})")

xmin, xmax = 0, max(100, float(np.nanmax(x))*1.02)
ax.plot([xmin, xmax], [xmin, xmax], linewidth=2, label="y = x")
ax.plot([xmin, xmax], [intercept + slope*xmin, intercept + slope*xmax], linewidth=2, label="OLS fit")

ax.set_title("Agreement of Persister Fraction: 13k vs 1k", fontsize=16)
ax.set_xlabel("Persister % (13k)", fontsize=13)
ax.set_ylabel("Persister % (1k)", fontsize=13)

stats_txt = (f"n = {n}\n"
             f"r = {r:.3f}\n"
             f"Slope = {slope:.3f}\n"
             f"Intercept = {intercept:.2f}\n"
             f"RMSE = {rmse:.2f}\n"
             f"MAE = {mae:.2f}")
ax.text(0.05, 0.95, stats_txt, transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="none"))

ax.legend(frameon=False, loc="lower right")
ax.grid(True, linewidth=0.6, alpha=0.3)
ax.set_xlim(xmin, xmax); ax.set_ylim(xmin, xmax)
fig.tight_layout()
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print("Saved figure:", out_png)
