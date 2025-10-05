#!/usr/bin/env python3
import os, re, json, gzip, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

SKIP_DIRS = {"filtered_feature_bc_matrix", "outs", "count"}

def load_list(p: Path):
    with open(p) as f:
        return [x.strip().upper() for x in f if x.strip()]

def clean_symbols(x):
    return [str(g).upper().rsplit(".", 1)[0] for g in x]

def find_10x_dirs(root: Path):
    out = []
    for p in root.rglob("filtered_feature_bc_matrix"):
        if not p.is_dir(): 
            continue
        parent = p.parent
        while parent.name in SKIP_DIRS and parent.parent != parent:
            parent = parent.parent
        out.append((p, parent.name))
    return out

def find_loose_triplets(root: Path):
    out = []
    for mtx in sorted(root.glob("*_matrix_*.mtx.gz")):
        m = re.match(r"^(.*)_matrix_.*\.mtx(\.gz)?$", mtx.name)
        if not m: 
            continue
        prefix = m.group(1)
        out.append((mtx, f"{prefix}_HEALTHY"))
    return out

def read_tsv(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [line.rstrip("\n").split("\t") for line in f]

def load_10x_matrix(matrix_dir: Path, max_cells=None):
    # matrix
    mat = None
    for cand in ["matrix.mtx.gz", "matrix.mtx"]:
        p = matrix_dir / cand
        if p.exists():
            mat = mmread(str(p))
            break
    if mat is None:
        raise FileNotFoundError(f"No matrix.mtx(.gz) in {matrix_dir}")

    # features / genes
    feat = None
    for cand in ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"]:
        p = matrix_dir / cand
        if p.exists():
            feat = read_tsv(p)
            break
    if feat is None:
        raise FileNotFoundError(f"No features/genes file in {matrix_dir}")

    # barcodes (optional)
    barcodes = None
    for cand in ["barcodes.tsv.gz", "barcodes.tsv"]:
        p = matrix_dir / cand
        if p.exists():
            op = gzip.open if str(p).endswith(".gz") else open
            with op(p, "rt") as fh:
                barcodes = [ln.strip() for ln in fh]
            break

    genes = [ (r[1] if len(r) >= 2 else r[0]) for r in feat ]
    genes = clean_symbols(genes)

    if sparse.isspmatrix(mat):
        X = mat.tocsc()  # genes x cells
    else:
        X = sparse.csc_matrix(mat)

    # optional subsample BEFORE heavy ops
    if max_cells and X.shape[1] > max_cells:
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(X.shape[1], max_cells, replace=False))
        X = X[:, keep]
        if barcodes:
            barcodes = [barcodes[i] for i in keep]
    elif barcodes is None:
        barcodes = [f"CELL_{i}" for i in range(X.shape[1])]

    return X, genes, barcodes

def load_loose_triplet(matrix_file: Path, max_cells=None):
    m = re.match(r"^(.*)_matrix_(.*)\.mtx(\.gz)?$", matrix_file.name)
    if not m:
        raise ValueError(f"Unrecognized matrix filename: {matrix_file}")
    prefix, suff = m.group(1), m.group(2)
    genes_file = matrix_file.with_name(f"{prefix}_genes_{suff}.tsv.gz")
    if not genes_file.exists():
        genes_file = matrix_file.with_name(f"{prefix}_genes_{suff}.tsv")
    barcodes_file = matrix_file.with_name(f"{prefix}_barcodes_{suff}.tsv.gz")
    if not barcodes_file.exists():
        barcodes_file = matrix_file.with_name(f"{prefix}_barcodes_{suff}.tsv")
    if not genes_file.exists() or not barcodes_file.exists():
        raise FileNotFoundError(f"Missing genes/barcodes for {matrix_file}")

    mat = mmread(str(matrix_file))
    if sparse.isspmatrix(mat):
        X = mat.tocsc()  # genes x cells
    else:
        X = sparse.csc_matrix(mat)

    if max_cells and X.shape[1] > max_cells:
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(X.shape[1], max_cells, replace=False))
        X = X[:, keep]
        opb = gzip.open if str(barcodes_file).endswith(".gz") else open
        with opb(barcodes_file, "rt") as fh:
            bc = [ln.strip() for ln in fh]
        barcodes = [bc[i] for i in keep]
    else:
        opb = gzip.open if str(barcodes_file).endswith(".gz") else open
        with opb(barcodes_file, "rt") as fh:
            barcodes = [ln.strip() for ln in fh]

    rows = read_tsv(genes_file)
    genes = [ (r[1] if len(r) >= 2 else r[0]) for r in rows ]
    genes = clean_symbols(genes)
    return X, genes, barcodes

def log2_cpm1_sparse(X_csc: sparse.csc_matrix):
    # X: genes x cells (counts)
    X = X_csc.copy().astype(np.float32)
    lib = np.asarray(X.sum(axis=0)).ravel()
    lib[lib == 0] = 1.0
    # scale columns by (1e6 / lib)
    scale = (1e6 / lib).astype(np.float32)
    for j in range(X.shape[1]):
        start, end = X.indptr[j], X.indptr[j+1]
        if end > start:
            X.data[start:end] *= scale[j]
    # log2(1 + x): zeros stay zero implicitly
    X.data = np.log2(X.data + 1.0, dtype=np.float32)
    return X  # genes x cells, log2(CPM+1)

def score_cells_sparse(X_log_cpm: sparse.csc_matrix, genes, up, down):
    # genes: list of symbols (len = X_log_cpm.shape[0])
    g2i = {g: i for i, g in enumerate(genes)}
    up_idx = [g2i[g] for g in up if g in g2i]
    dn_idx = [g2i[g] for g in down if g in g2i]
    if len(up_idx) < 5 or len(dn_idx) < 5:
        print(f"⚠ Low overlap: up={len(up_idx)}, down={len(dn_idx)} — results may be unstable.")
    # mean across rows (genes) without densifying
    up_mean = (X_log_cpm[up_idx, :].sum(axis=0) / max(1, len(up_idx))).A1
    dn_mean = (X_log_cpm[dn_idx, :].sum(axis=0) / max(1, len(dn_idx))).A1
    return up_mean - dn_mean, len(up_idx), len(dn_idx)

def load_threshold(tm_path: Path):
    if not tm_path.exists():
        return None
    try:
        tm = json.loads(tm_path.read_text())
        return float(tm.get("threshold", None))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Apply AML–HSPC score to scRNA datasets under AML/HEALTHY roots")
    ap.add_argument("--aml-root",      type=Path, required=True)
    ap.add_argument("--healthy-root",  type=Path, required=True)
    ap.add_argument("--up",            type=Path, default=Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/Hematopoietic_Cells_Healthy_AML_analysis/scoring_up.txt"))
    ap.add_argument("--down",          type=Path, default=Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/Hematopoietic_Cells_Healthy_AML_analysis/scoring_down.txt"))
    ap.add_argument("--training-metrics", type=Path, default=Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/Hematopoietic_Cells_Healthy_AML_analysis/training_metrics.json"))
    ap.add_argument("--outdir",        type=Path, required=True)
    ap.add_argument("--max-cells",     type=int, default=50000, help="Subsample per dataset (0=all)")
    ap.add_argument("--write-cells",   action="store_true", help="Write per-cell scores CSV (can be large)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    up = load_list(args.up)
    dn = load_list(args.down)
    print(f"Loaded scoring genes: up={len(up)}, down={len(dn)}")

    tstar = load_threshold(args.training_metrics)
    if tstar is not None:
        print(f"Using training threshold t* = {tstar:.3f}")
    else:
        print("No t* found; will skip threshold-based fractions.")

    # Discover datasets
    aml_sets = find_10x_dirs(args.aml_root)
    healthy_sets = find_loose_triplets(args.healthy_root)
    print(f"Found AML 10x datasets: {len(aml_sets)}")
    print(f"Found Healthy loose triplets: {len(healthy_sets)}")

    all_summ = []

    def process_entry(kind, path_obj, name):
        print("\n" + "="*70)
        print(f"{kind}: {name}")
        print("="*70)
        try:
            if kind == "AML":
                X, genes, barcodes = load_10x_matrix(path_obj, max_cells=args.max_cells or None)
            else:
                X, genes, barcodes = load_loose_triplet(path_obj, max_cells=args.max_cells or None)
        except Exception as e:
            print(f"[SKIP] load failed: {e}")
            return

        # Drop duplicate gene symbols by keeping first occurrence (warn only)
        uniq_seen = {}
        keep_rows = []
        dup = 0
        for i, g in enumerate(genes):
            if g not in uniq_seen:
                uniq_seen[g] = i
                keep_rows.append(i)
            else:
                dup += 1
        if dup > 0:
            X = X[keep_rows, :]
            genes = [genes[i] for i in keep_rows]
            print(f"Note: dropped {dup} duplicate gene rows (kept first occurrence).")

        Xlog = log2_cpm1_sparse(X)
        scores, up_n, dn_n = score_cells_sparse(Xlog, genes, up, dn)

        # Write per-cell (optional)
        if args.write_cells:
            df_cells = pd.DataFrame({"cell_id": barcodes, "aml_score": scores})
            df_cells.to_csv(args.outdir / f"{name}_{kind}_cell_scores.csv", index=False)

        # Sample-level summary
        q = np.quantile(scores, [0.1, 0.5, 0.9, 0.95, 0.99])
        above = float(np.mean(scores >= tstar)) if tstar is not None else np.nan
        all_summ.append({
            "sample": name,
            "group": kind,
            "cells": int(len(scores)),
            "up_overlap": int(up_n),
            "down_overlap": int(dn_n),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "q10": float(q[0]), "q50": float(q[1]), "q90": float(q[2]), "q95": float(q[3]), "q99": float(q[4]),
            "frac_above_tstar": above
        })

        # Small per-sample plot
        plt.figure(figsize=(5,4))
        plt.hist(scores, bins=40, edgecolor="black", alpha=0.7)
        if tstar is not None:
            plt.axvline(tstar, ls="--", c="k", label=f"t*={tstar:.2f}")
            plt.legend()
        plt.title(f"{name} ({kind}) — cell scores")
        plt.xlabel("AML score"); plt.ylabel("Cells")
        plt.tight_layout()
        plt.savefig(args.outdir / f"{name}_{kind}_hist.png", dpi=180)
        plt.close()

    for p, n in aml_sets:
        process_entry("AML", p, n)
    for p, n in healthy_sets:
        process_entry("HEALTHY", p, n)

    # Write summary
    df = pd.DataFrame(all_summ)
    df.to_csv(args.outdir / "scRNA_score_summary.csv", index=False)
    print(f"\nWrote summary: {args.outdir / 'scRNA_score_summary.csv'}")

    # Cohort-level metrics/plots (sample-level using medians)
    if not df.empty and df["group"].nunique() >= 2:
        med = df.pivot_table(index="sample", values="q50", columns="group")
        if set(["AML","HEALTHY"]).issubset(med.columns):
            y = np.r_[np.ones(len(med["AML"].dropna())), np.zeros(len(med["HEALTHY"].dropna()))]
            s = np.r_[med["AML"].dropna().values, med["HEALTHY"].dropna().values]
            if len(np.unique(y)) == 2 and len(y) >= 3:
                auc = roc_auc_score(y, s)
                fpr, tpr, _ = roc_curve(y, s)
                print(f"Sample-level AUC (medians): {auc:.3f}")
                plt.figure(figsize=(4,4))
                plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
                plt.plot([0,1],[0,1],"k--",alpha=0.3)
                plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (sample medians)")
                plt.legend(); plt.tight_layout()
                plt.savefig(args.outdir / "cohort_sample_median_ROC.png", dpi=300)
                plt.close()

    # Violin of sample medians by group
    if not df.empty:
        plt.figure(figsize=(5,4))
        data = [df.loc[df.group=="AML","q50"].dropna().values,
                df.loc[df.group=="HEALTHY","q50"].dropna().values]
        plt.violinplot(data, showmeans=True)
        plt.xticks([1,2], ["AML", "Healthy"])
        if tstar is not None:
            plt.axhline(tstar, ls="--", c="k")
        plt.ylabel("Sample median AML score")
        plt.title("Sample medians by group")
        plt.tight_layout()
        plt.savefig(args.outdir / "cohort_sample_median_violin.png", dpi=300)
        plt.close()

    print("Done.")

if __name__ == "__main__":
    main()
