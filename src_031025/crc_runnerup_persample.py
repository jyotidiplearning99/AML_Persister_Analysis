#!/usr/bin/env python3
"""crc_runnerup_persample.py — per-sample version of the CRC runner-up test."""
import argparse, sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

MATRIX = "GSE200997_GEO_processed_CRC_10X_raw_UMI_count_matrix.csv.gz"
ANNOT  = "GSE200997_GEO_processed_CRC_10X_cell_annotation.csv.gz"

def clean(n): return [str(g).strip().upper().rsplit(".",1)[0] for g in n]
def cpm_log1p(X):
    X=np.maximum(np.asarray(X,np.float32),0.0)
    lib=X.sum(1,keepdims=True); np.maximum(lib,1.0,out=lib)
    return np.log1p((X/lib)*1e4).astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--genes-file", default="selected_genes.txt")
    ap.add_argument("--condition-col", default="Condition")
    ap.add_argument("--sample-col", default="samples")
    ap.add_argument("--matrix", default=MATRIX)
    ap.add_argument("--annot", default=ANNOT)
    args=ap.parse_args()

    panel=[l.strip().upper() for l in open(args.genes_file) if l.strip()]
    ann=pd.read_csv(args.annot, index_col=0)
    print("Loading matrix...")
    mat=pd.read_csv(args.matrix, index_col=0)
    if mat.columns.isin(ann.index).mean()>0.5: mat=mat.T
    cells=mat.index.intersection(ann.index)
    mat=mat.loc[cells]; ann=ann.loc[cells]

    genes=clean(mat.columns); g2i={g:i for i,g in enumerate(genes)}
    cols=[g2i.get(p) for p in panel]
    found=sum(c is not None for c in cols)
    print(f"Gene match: {found}/{len(panel)}")

    # z-score genes ACROSS ALL cells once, so per-sample means are comparable
    Xall=np.zeros((mat.shape[0],len(panel)),np.float32)
    raw=mat.values.astype(np.float32)
    for j,c in enumerate(cols):
        if c is not None: Xall[:,j]=raw[:,c]
    Xl=cpm_log1p(Xall)
    mu,sd=Xl.mean(0,keepdims=True),Xl.std(0,keepdims=True); sd[sd==0]=1.0
    Z=(Xl-mu)/sd
    cell_score=Z.mean(1)

    df=pd.DataFrame({"sample":ann[args.sample_col].values,
                     "cond":ann[args.condition_col].astype(str).values,
                     "score":cell_score})
    per=df.groupby("sample").agg(cond=("cond","first"),
                                 mean_score=("score","mean"),
                                 n_cells=("score","size")).reset_index()
    per["grp"]=per["cond"].str.lower().map(
        lambda v:"tumour" if any(t in v for t in["tumor","tumour","cancer"]) else
                 ("normal" if "normal" in v else None))
    per=per.dropna(subset=["grp"])
    print("\nPer-sample mean scores:")
    print(per.sort_values(["grp","mean_score"]).to_string(index=False))

    t=per[per.grp=="tumour"]["mean_score"].values
    n=per[per.grp=="normal"]["mean_score"].values
    U,p=mannwhitneyu(t,n,alternative="two-sided")
    print(f"\n=== Per-sample comparison ===")
    print(f"  tumour samples: n={len(t)}, mean={t.mean():.4f}, median={np.median(t):.4f}")
    print(f"  normal samples: n={len(n)}, mean={n.mean():.4f}, median={np.median(n):.4f}")
    print(f"  direction: {'tumour > normal' if t.mean()>n.mean() else 'normal >= tumour'}")
    print(f"  Mann-Whitney U p = {p:.4g}  (n={len(t)} vs {len(n)} samples)")
    per.to_csv("crc_runnerup_persample.csv", index=False)

if __name__=="__main__": main()