#!/usr/bin/env python3
"""
Calibration / Summary analysis

Two modes (auto-detected):
1) Labeled calibration:
   - Input CSV must contain: y_true (0/1), y_prob (0..1)
   - Outputs: reliability diagram, ECE/MCE, threshold sensitivity sweep

2) Label-free summary (your current inference_results_summary.csv schema):
   - Expected columns (min): sample_id, mean_prob[, persister_pct, cells, threshold]
   - Outputs: descriptive stats JSON, probability distribution plot,
              mean_prob vs persister_pct scatter (if available)

Usage examples:
  Labeled:
    python calibrate_or_summarize.py --predictions path/to/labeled.csv --out-dir results/eval

  Label-free (current data):
    python calibrate_or_summarize.py --predictions \
      /scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/inference/inference_results_summary.csv \
      --out-dir results/manuscript_assets/evaluation
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# ----------------------
# Helpers (labeled mode)
# ----------------------

def ece_mce(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        diff = abs(conf - acc)
        ece += (mask.mean()) * diff
        mce = max(mce, diff)
    return float(ece), float(mce)

def plot_reliability(y_true, y_prob, png_path: Path, n_bins=10, title="Reliability diagram"):
    # Guard: calibration_curve requires both classes
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true has a single class; cannot plot reliability diagram.")
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    ece, mce = ece_mce(y_true, y_prob, n_bins=n_bins)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    return {"ece": ece, "mce": mce}

def threshold_sweep(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    rows = []
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        tp = int(((y_true == 1) & (y_hat == 1)).sum())
        tn = int(((y_true == 0) & (y_hat == 0)).sum())
        fp = int(((y_true == 0) & (y_hat == 1)).sum())
        fn = int(((y_true == 1) & (y_hat == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        rows.append({
            "threshold": float(t),
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv
        })
    return pd.DataFrame(rows)

# --------------------------
# Label-free summary helpers
# --------------------------

def summarize_label_free(df, out_dir: Path):
    """
    Works with your current summary CSV:
      required: sample_id, mean_prob
      optional: persister_pct, cells, threshold
    Produces:
      - summary.json with descriptive statistics
      - probability_distribution.png
      - meanprob_vs_persisterpct.png (if persister_pct present)
    """
    # Coerce numeric columns safely
    for col in ["mean_prob", "persister_pct", "cells", "threshold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without mean_prob
    df = df.dropna(subset=["mean_prob"]).copy()

    # Descriptives
    desc = {
        "n_samples": int(len(df)),
        "mean_prob": {
            "mean": float(df["mean_prob"].mean()),
            "std": float(df["mean_prob"].std(ddof=1)) if len(df) > 1 else 0.0,
            "median": float(df["mean_prob"].median()),
            "p25": float(df["mean_prob"].quantile(0.25)),
            "p75": float(df["mean_prob"].quantile(0.75)),
        }
    }

    # Cell-weighted mean_prob if cells present
    if "cells" in df.columns and df["cells"].notna().any():
        valid = df.dropna(subset=["cells"])
        if len(valid) > 0 and valid["cells"].sum() > 0:
            wmean = float((valid["mean_prob"] * valid["cells"]).sum() / valid["cells"].sum())
            desc["mean_prob_weighted_by_cells"] = wmean

    # persister_pct stats (if available)
    if "persister_pct" in df.columns and df["persister_pct"].notna().any():
        desc["persister_pct"] = {
            "mean": float(df["persister_pct"].mean()),
            "std": float(df["persister_pct"].std(ddof=1)) if len(df) > 1 else 0.0,
            "median": float(df["persister_pct"].median()),
            "min": float(df["persister_pct"].min()),
            "max": float(df["persister_pct"].max()),
        }
        # Correlation between mean_prob and persister_pct (both at sample level)
        corr = float(np.corrcoef(df["mean_prob"], df["persister_pct"] / 100.0)[0, 1])
        desc["corr_mean_prob__persister_pct"] = corr

    # threshold summary (if present)
    if "threshold" in df.columns and df["threshold"].notna().any():
        desc["threshold"] = {
            "mean": float(df["threshold"].mean()),
            "std": float(df["threshold"].std(ddof=1)) if len(df) > 1 else 0.0,
            "median": float(df["threshold"].median()),
        }

    # Save summary JSON
    (out_dir / "summary.json").write_text(json.dumps(desc, indent=2))

    # Plots
    # 1) Distribution of mean_prob
    plt.figure(figsize=(6, 4))
    plt.hist(df["mean_prob"].values, bins=20, edgecolor="black")
    plt.xlabel("mean_prob (per sample)")
    plt.ylabel("Count")
    plt.title("Distribution of mean_prob across samples")
    plt.tight_layout()
    plt.savefig(out_dir / "probability_distribution.png", dpi=300)
    plt.close()

    # 2) mean_prob vs persister_pct (if available)
    if "persister_pct" in df.columns and df["persister_pct"].notna().any():
        plt.figure(figsize=(6, 5))
        plt.scatter(df["mean_prob"].values, df["persister_pct"].values, s=28)
        plt.xlabel("mean_prob (per sample)")
        plt.ylabel("persister_pct (%) at τ")
        plt.title("mean_prob vs persister_pct")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "meanprob_vs_persisterpct.png", dpi=300)
        plt.close()

    return desc

# -----------
# Entry point
# -----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True,
                    help="CSV. Labeled mode needs y_true,y_prob. Label-free mode needs mean_prob (and optionally persister_pct,cells,threshold).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--mode", choices=["auto", "labeled", "summary"], default="auto",
                    help="Force mode or let the script auto-detect from columns.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions)

    cols = set(df.columns.str.lower())
    has_labels = {"y_true", "y_prob"}.issubset(cols)
    has_summary = "mean_prob" in cols  # minimal requirement for your current file

    # Optional: normalize column names to expected ones for summary path
    # (Your file already matches these names.)
    # If someone provides 'sample' instead of 'sample_id', we won't break.

    # Decide mode
    mode = args.mode
    if mode == "auto":
        if has_labels:
            mode = "labeled"
        elif has_summary:
            mode = "summary"
        else:
            raise SystemExit("[ERROR] Could not detect input schema. Provide either y_true,y_prob (labeled) "
                             "OR mean_prob (summary).")

    if mode == "labeled":
        if not has_labels:
            raise SystemExit("[ERROR] Labeled mode selected but CSV lacks y_true,y_prob.")
        # Ensure both classes exist for reliability/ece
        if df["y_true"].nunique() < 2:
            raise SystemExit("[ERROR] y_true has a single class; cannot compute calibration metrics.")
        # Reliability + ECE/MCE
        png = args.out_dir / "reliability_diagram.png"
        cal = plot_reliability(df["y_true"].values, df["y_prob"].values, png, n_bins=args.n_bins)
        (args.out_dir / "calibration_metrics.json").write_text(json.dumps(cal, indent=2))
        # Threshold sweep
        sweep = threshold_sweep(df["y_true"].values, df["y_prob"].values)
        sweep.to_csv(args.out_dir / "threshold_sensitivity.csv", index=False)
        print(f"[OK] Labeled mode: wrote {png.name}, calibration_metrics.json, and threshold_sensitivity.csv to {args.out_dir}")

    else:
        # Label-free summary path (your current data)
        desc = summarize_label_free(df, args.out_dir)
        (args.out_dir / "summary_mode_note.txt").write_text(
            "Labels were not provided; produced descriptive statistics and plots only.\n"
            "To compute calibration (ECE/MCE) and reliability diagrams, supply a CSV with y_true,y_prob.\n"
        )
        print(f"[OK] Summary mode: wrote summary.json and plots to {args.out_dir}")

if __name__ == "__main__":
    main()
