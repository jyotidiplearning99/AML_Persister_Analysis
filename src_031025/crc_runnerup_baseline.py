#!/usr/bin/env python3
"""
crc_runnerup_from_csv.py
Reads GSE200997 raw GEO files directly (no h5ad needed) and runs the
mean-expression runner-up baseline, tumour vs normal.
"""
import argparse, sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

MATRIX = "GSE200997_GEO_processed_CRC_10X_raw_UMI_count_matrix.csv.gz"
ANNOT  = "GSE200997_GEO_processed_CRC_10X_cell_annotation.csv.gz"

def clean(names):
    return [str(g).strip().upper().rsplit(".",1)[0] for g in names]

def cpm_log1p(X):
    X = np.maximum(np.asarray(X, np.float32), 0.0)
    lib = X.sum(1, keepdims=True); np.maximum(lib,1.0,out=lib)
    return np.log1p((X/lib)*1e4).astype(np.float32)

def mean_expr_score(Xpanel):
    Xl = cpm_log1p(Xpanel)
    mu, sd = Xl.mean(0,keepdims=True), Xl.std(0,keepdims=True)
    sd[sd==0]=1.0
    return ((Xl-mu)/sd).mean(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes-file", default="selected_genes.txt")
    ap.add_argument("--condition-col", required=True,
                    help="annotation column with tumour/normal (find it with the inspect snippet)")
    ap.add_argument("--matrix", default=MATRIX)
    ap.add_argument("--annot", default=ANNOT)
    ap.add_argument("--max-per-group", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    panel = [l.strip().upper() for l in open(args.genes_file) if l.strip()]
    print(f"Panel: {len(panel)} genes")

    ann = pd.read_csv(args.annot, index_col=0)
    if args.condition_col not in ann.columns:
        sys.exit(f"'{args.condition_col}' not in annotation. Columns: {list(ann.columns)}")

    # GEO matrices are usually genes x cells. Load and orient to cells x genes.
    print("Loading matrix (this is ~71MB gzipped, takes a minute)...")
    mat = pd.read_csv(args.matrix, index_col=0)
    # decide orientation: cells should match annotation index
    if mat.columns.isin(ann.index).mean() > 0.5:
        # columns are cells -> transpose to cells x genes
        mat = mat.T
    # now rows = cells, columns = genes
    common_cells = mat.index.intersection(ann.index)
    print(f"Cells in matrix: {mat.shape[0]}, matching annotation: {len(common_cells)}")
    mat = mat.loc[common_cells]
    cond = ann.loc[common_cells, args.condition_col].astype(str)

    print(f"\nValues in '{args.condition_col}': {cond.value_counts().to_dict()}")

    def grp(v):
        v=v.lower()
        if any(t in v for t in ["tumor","tumour","crc","cancer","carcinoma"]): return "tumour"
        if any(t in v for t in ["normal","healthy","adjacent","control"]): return "normal"
        return None
    g = cond.map(grp)
    keep = g.isin(["tumour","normal"])
    mat, g = mat[keep], g[keep]
    print(f"tumour={int((g=='tumour').sum())}, normal={int((g=='normal').sum())}")

    genes = clean(mat.columns)
    g2i = {gn:i for i,gn in enumerate(genes)}
    cols = [g2i.get(p) for p in panel]
    found = sum(c is not None for c in cols)
    print(f"Panel gene match: {found}/{len(panel)} ({100*found/len(panel):.1f}%)")

    rng = np.random.default_rng(args.seed)
    res = {}
    arrs = {}
    for grpname in ["tumour","normal"]:
        idx = np.where(g.values==grpname)[0]
        if len(idx)>args.max_per_group:
            idx = rng.choice(idx,args.max_per_group,replace=False)
        sub = mat.iloc[idx].values.astype(np.float32)
        Xp = np.zeros((sub.shape[0], len(panel)), np.float32)
        for j,c in enumerate(cols):
            if c is not None: Xp[:,j]=sub[:,c]
        s = mean_expr_score(Xp)
        arrs[grpname]=s
        res[grpname]=dict(n=len(idx), mean=round(float(s.mean()),4),
                          median=round(float(np.median(s)),4))
        print(f"  {grpname}: n={len(idx)} mean={res[grpname]['mean']}")

    U,p = mannwhitneyu(arrs["tumour"],arrs["normal"],alternative="two-sided")
    direction = "tumour > normal" if res["tumour"]["mean"]>res["normal"]["mean"] else "normal >= tumour"
    print(f"\n=== Runner-up (mean-expression) on CRC ===")
    print(f"  {direction},  Mann-Whitney p={p:.3g}")
    if direction=="normal >= tumour":
        print("  -> Runner-up ALSO fails (normal >= tumour). Failure is signal-level")
        print("     (stemness/WNT), not Transformer-specific. Matches Markus's prediction.")
    else:
        print("  -> Runner-up separates tumour>normal unlike the Transformer; investigate.")

if __name__=="__main__":
    main()