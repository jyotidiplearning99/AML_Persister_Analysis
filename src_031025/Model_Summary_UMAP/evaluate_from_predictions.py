#!/usr/bin/env python3
"""
Evaluate predictions with graceful schema detection.

Supports TWO schemas:

A) Labeled row-wise predictions (preferred for AUROC/AUPRC)
   Required columns:
     sample_id, y_true (0/1), y_prob (0..1), donor_id, dataset
   Optional:
     oof_fold (int) -> if present, summary reported as donor-level OOF CV.

B) Per-sample summary (like inference_results_summary.csv)
   Required columns (any subset of these is fine):
     sample/sample_id, mean_prob, std_prob, persister_pct, cells, threshold
   Optional:
     --labels CSV providing: sample_id, y_true [, donor_id, dataset, oof_fold]
   If labels are provided, AUROC/AUPRC are computed using mean_prob as the
   sample-level probability; otherwise we produce descriptive statistics only.

Outputs:
  - evaluation_results.json
  - evaluation_results.md
  - descriptive_summary.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def bootstrap_metric_ci(y_true, y_prob, n_boot=1000, seed=42):
    """AUROC/AUPRC + 95% bootstrap CIs (percentile)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return {
            "note": "Only one class present; AUROC/AUPRC undefined.",
            "n": int(len(y_true)),
        }

    base = {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "n": int(len(y_true)),
    }

    rng = np.random.default_rng(seed)
    aurocs, auprcs = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # Skip degenerate bootstrap samples
        if np.unique(y_true[idx]).size < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        auprcs.append(average_precision_score(y_true[idx], y_prob[idx]))

    if aurocs:
        base["auroc_ci"] = [float(np.quantile(aurocs, 0.025)), float(np.quantile(aurocs, 0.975))]
    else:
        base["auroc_ci"] = [None, None]

    if auprcs:
        base["auprc_ci"] = [float(np.quantile(auprcs, 0.025)), float(np.quantile(auprcs, 0.975))]
    else:
        base["auprc_ci"] = [None, None]

    return base


def detect_schema(df: pd.DataFrame):
    """Return 'labeled' or 'summary' or None."""
    labeled_needed = {"sample_id", "y_true", "y_prob", "donor_id", "dataset"}
    if labeled_needed.issubset(df.columns):
        return "labeled"

    summary_candidates = {"sample", "sample_id", "mean_prob", "persister_pct", "std_prob", "cells", "threshold"}
    if len(summary_candidates.intersection(df.columns)) >= 2 and (
        "mean_prob" in df.columns or "persister_pct" in df.columns
    ):
        return "summary"

    return None


def coerce_sample_id(df: pd.DataFrame, inplace=True):
    """Ensure a 'sample_id' column exists (from 'sample' if needed)."""
    if "sample_id" in df.columns:
        return df if inplace else df.copy()
    if "sample" in df.columns:
        if inplace:
            df.rename(columns={"sample": "sample_id"}, inplace=True)
            return df
        out = df.copy()
        out.rename(columns={"sample": "sample_id"}, inplace=True)
        return out
    return df if inplace else df.copy()


def load_labels(labels_csv: Path) -> pd.DataFrame:
    """Load labels file with at least: sample_id, y_true."""
    lab = pd.read_csv(labels_csv)
    if "sample_id" not in lab.columns:
        if "sample" in lab.columns:
            lab = lab.rename(columns={"sample": "sample_id"})
        else:
            raise ValueError("Labels file must contain 'sample_id' (or 'sample').")
    if "y_true" not in lab.columns:
        raise ValueError("Labels file must contain 'y_true' (0/1).")
    return lab


def descriptive_from_summary(df: pd.DataFrame) -> dict:
    """Compute descriptive stats from summary schema."""
    out = {"n_samples": int(len(df))}
    if "mean_prob" in df.columns:
        out["mean_prob"] = {
            "mean": float(df["mean_prob"].mean()),
            "std": float(df["mean_prob"].std(ddof=1)),
            "median": float(df["mean_prob"].median()),
            "p25": float(df["mean_prob"].quantile(0.25)),
            "p75": float(df["mean_prob"].quantile(0.75)),
        }
        if "cells" in df.columns and df["cells"].sum() > 0:
            w = df["cells"].astype(float)
            mp = df["mean_prob"].astype(float)
            out["mean_prob_weighted_by_cells"] = float((w * mp).sum() / w.sum())

    if "persister_pct" in df.columns:
        out["persister_pct"] = {
            "mean": float(df["persister_pct"].mean()),
            "std": float(df["persister_pct"].std(ddof=1)),
            "median": float(df["persister_pct"].median()),
            "min": float(df["persister_pct"].min()),
            "max": float(df["persister_pct"].max()),
        }

    if "threshold" in df.columns:
        out["threshold"] = {
            "mean": float(df["threshold"].mean()),
            "std": float(df["threshold"].std(ddof=1)),
            "median": float(df["threshold"].median()),
        }

    # Correlation (when both present)
    if "mean_prob" in df.columns and "persister_pct" in df.columns:
        try:
            corr = float(df[["mean_prob", "persister_pct"]].corr().iloc[0, 1])
        except Exception:
            corr = None
        out["corr_mean_prob__persister_pct"] = corr

    return out


def evaluate(pred_csv: Path, out_dir: Path, labels_csv: Path | None, n_boot=1000, seed=42):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_csv)
    df = coerce_sample_id(df, inplace=False)

    schema = detect_schema(df)
    results = {"schema": schema, "source_csv": str(pred_csv)}

    # Optionally load labels & merge for summary schema
    lab = None
    if labels_csv is not None:
        lab = load_labels(labels_csv)

    if schema == "labeled":
        # Standard full evaluation
        # Donor-level OOF (if oof_fold present, we merely note it and evaluate pooled OOF predictions)
        if "oof_fold" in df.columns:
            results["donor_level_cv_oof"] = bootstrap_metric_ci(df["y_true"], df["y_prob"], n_boot=n_boot, seed=seed)
        else:
            results["donor_level_cv_oof"] = {
                "note": "No 'oof_fold' column present; reporting pooled metrics only."
            }

        # Per-dataset
        per_dataset = {}
        for dset, g in df.groupby("dataset"):
            per_dataset[dset] = bootstrap_metric_ci(g["y_true"], g["y_prob"], n_boot=n_boot, seed=seed)
        results["per_dataset"] = per_dataset

        # Overall pooled
        results["overall"] = bootstrap_metric_ci(df["y_true"], df["y_prob"], n_boot=n_boot, seed=seed)

    elif schema == "summary":
        # Descriptive always
        desc = descriptive_from_summary(df)
        results["descriptives"] = desc

        # If labels provided, compute AUROC/AUPRC using mean_prob as sample-level probability
        if lab is not None:
            merged = df.merge(lab, on="sample_id", how="inner", suffixes=("", "_lab"))
            results["merge_info"] = {
                "n_predictions": int(len(df)),
                "n_labels": int(len(lab)),
                "n_merged": int(len(merged)),
            }
            if "mean_prob" in merged.columns:
                results["sample_level_metrics"] = bootstrap_metric_ci(
                    merged["y_true"], merged["mean_prob"], n_boot=n_boot, seed=seed
                )
            else:
                results["sample_level_metrics"] = {
                    "note": "mean_prob not present; cannot compute sample-level AUROC/AUPRC."
                }

            # If donor_id/dataset provided in labels, report grouped metrics too
            if {"dataset"}.issubset(merged.columns):
                grouped = {}
                for dset, g in merged.groupby("dataset"):
                    if g["y_true"].nunique() < 2:
                        grouped[dset] = {"note": "Only one class present; metrics undefined."}
                    else:
                        grouped[dset] = bootstrap_metric_ci(
                            g["y_true"], g["mean_prob"], n_boot=n_boot, seed=seed
                        )
                results["per_dataset"] = grouped

            if {"donor_id"}.issubset(merged.columns) and "oof_fold" in merged.columns:
                # This merely reports pooled OOF metrics if you passed true OOF rows
                results["donor_level_cv_oof"] = bootstrap_metric_ci(
                    merged["y_true"], merged["mean_prob"], n_boot=n_boot, seed=seed
                )

        else:
            results["note"] = (
                "No labels provided. Produced descriptive statistics only. "
                "Provide --labels CSV (sample_id,y_true[,donor_id,dataset,oof_fold]) to compute AUROC/AUPRC."
            )

        # Save a compact descriptive CSV for quick manuscript tables
        keep_cols = [c for c in ["sample_id", "mean_prob", "std_prob", "persister_pct", "cells", "threshold"] if c in df.columns]
        if keep_cols:
            df[keep_cols].to_csv(out_dir / "descriptive_summary.csv", index=False)

    else:
        print(
            f"[ERROR] Could not recognize CSV schema. "
            f"Expected either labeled columns {{sample_id,y_true,y_prob,donor_id,dataset}} "
            f"or summary columns including mean_prob/persister_pct.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Write outputs
    (out_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    # Markdown view
    md = ["# Evaluation (from predictions)", "", f"- Input: `{pred_csv}`", f"- Schema: `{schema}`", ""]
    md.append("```json")
    md.append(json.dumps(results, indent=2))
    md.append("```")
    (out_dir / "evaluation_results.md").write_text("\n".join(md))

    print(f"[OK] Wrote results to: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True, help="Path to predictions CSV")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to write outputs")
    ap.add_argument(
        "--labels",
        type=Path,
        required=False,
        help="Optional labels CSV with columns: sample_id,y_true[,donor_id,dataset,oof_fold]",
    )
    ap.add_argument("--n-boot", type=int, default=1000, help="Bootstrap iterations (default 1000)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    evaluate(args.predictions, args.out_dir, args.labels, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
