#!/usr/bin/env python3
from __future__ import annotations
import argparse, itertools
from pathlib import Path
from typing import Dict, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

DEFAULT_MARKERS = ["FLT3","CD33","NECTIN2","CD44","KDR","EPHB2"]
SEED = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="results/surface_surrogate/cache_joined_candidate_expression")
    p.add_argument("--out-dir", default="results/biaxial_sorting")
    p.add_argument("--markers", nargs="+", default=DEFAULT_MARKERS)
    p.add_argument("--gate-quantile", type=float, default=0.50)
    p.add_argument("--score-extreme-q", type=float, default=0.25)
    p.add_argument("--min-cells-per-quadrant", type=int, default=30)
    p.add_argument("--cv-folds", type=int, default=5)
    return p.parse_args()

def pct_rank(s):
    return s.rank(method="average", pct=True)

def prep(df, donor, markers, q):
    miss=[g for g in markers if g not in df.columns]
    if miss: raise ValueError(f"{donor}: missing {miss}")
    if "score" not in df.columns: raise ValueError(f"{donor}: missing score")
    z=df[["score"]+list(markers)].copy()
    z["donor"]=donor
    z["score_pct"]=pct_rank(z["score"])
    z["score_high"]=z["score_pct"] >= 1-q
    z["score_low"]=z["score_pct"] <= q
    for g in markers: z[f"{g}_pct"]=pct_rank(z[g])
    return z

def load_donors(cache_dir, markers, q):
    fs=sorted(Path(cache_dir).glob("*.pkl"))
    if not fs: raise FileNotFoundError(f"No donor .pkl files in {cache_dir}")
    out={}
    for f in fs:
        try:
            out[f.stem]=prep(pd.read_pickle(f), f.stem, markers, q)
            print(f"[ok] {f.stem}: {len(out[f.stem]):,}")
        except Exception as e:
            print(f"[skip] {f.stem}: {e}")
    if len(out)<3: raise RuntimeError("Too few usable donors")
    return out

def qlabels(df,g1,g2,t):
    a=df[f"{g1}_pct"]>=t; b=df[f"{g2}_pct"]>=t
    return pd.Series(np.select(
        [a&b,a&~b,~a&b],
        [f"{g1}hi/{g2}hi",f"{g1}hi/{g2}lo",f"{g1}lo/{g2}hi"],
        default=f"{g1}lo/{g2}lo"), index=df.index)

def pair_summary(donors, ids, g1,g2,t,min_cells):
    rows=[]
    for donor in ids:
        d=donors[donor].copy()
        d["quadrant"]=qlabels(d,g1,g2,t)
        for q,z in d.groupby("quadrant"):
            if len(z)<min_cells: continue
            rows.append(dict(
                donor=donor,marker1=g1,marker2=g2,quadrant=q,n_cells=len(z),
                cell_fraction=len(z)/len(d),
                median_score=float(z["score"].median()),
                median_score_pct=float(z["score_pct"].median()),
                high_score_fraction=float(z["score_high"].mean()),
                low_score_fraction=float(z["score_low"].mean())))
    long=pd.DataFrame(rows)
    if long.empty:
        return long, dict(marker1=g1,marker2=g2,n_donors=0,separation=np.nan,
                          max_high_enrichment=np.nan,max_low_enrichment=np.nan,
                          min_quadrant_fraction=np.nan,pair_composite=np.nan)
    qagg=long.groupby("quadrant",as_index=False).agg(
        n_donors=("donor","nunique"),
        median_cell_fraction=("cell_fraction","median"),
        median_score_pct=("median_score_pct","median"),
        median_high_fraction=("high_score_fraction","median"),
        median_low_fraction=("low_score_fraction","median"))
    sep=float(qagg.median_score_pct.max()-qagg.median_score_pct.min())
    hi=float(qagg.median_high_fraction.max()/0.25)
    lo=float(qagg.median_low_fraction.max()/0.25)
    mq=float(qagg.median_cell_fraction.min())
    sf=min(1.0,mq/0.05) if np.isfinite(mq) else 0
    comp=0.60*sep+0.20*max(0,hi-1)+0.15*max(0,lo-1)+0.05*sf
    return long, dict(marker1=g1,marker2=g2,n_donors=int(long.donor.nunique()),
                      separation=sep,max_high_enrichment=hi,max_low_enrichment=lo,
                      min_quadrant_fraction=mq,pair_composite=comp)

def rank_pairs(donors, ids, markers, t, min_cells):
    S=[]; D=[]
    for g1,g2 in itertools.combinations(markers,2):
        long,s=pair_summary(donors,ids,g1,g2,t,min_cells)
        S.append(s)
        if not long.empty: D.append(long)
    rank=pd.DataFrame(S).sort_values(["pair_composite","separation"],ascending=False).reset_index(drop=True)
    det=pd.concat(D,ignore_index=True) if D else pd.DataFrame()
    return rank,det

def agg_quads(details,g1,g2):
    z=details[(details.marker1==g1)&(details.marker2==g2)].copy()
    a=z.groupby("quadrant",as_index=False).agg(
        n_donors=("donor","nunique"),
        median_cell_fraction=("cell_fraction","median"),
        median_score_pct=("median_score_pct","median"),
        median_high_fraction=("high_score_fraction","median"),
        median_low_fraction=("low_score_fraction","median"))
    a["high_enrichment_vs_baseline"]=a.median_high_fraction/0.25
    a["low_enrichment_vs_baseline"]=a.median_low_fraction/0.25
    return a.sort_values("median_score_pct",ascending=False).reset_index(drop=True)

def branch_ranking(donors, first_pair, first_quads, remaining, t, min_cells):
    g1,g2=first_pair; rows=[]
    for branch in first_quads:
        bd={}
        for donor,d in donors.items():
            q=qlabels(d,g1,g2,t)
            z=d.loc[q==branch].copy()
            if len(z)>=4*min_cells: bd[donor]=z
        if len(bd)<3 or len(remaining)<2: continue
        r,_=rank_pairs(bd,sorted(bd),remaining,t,min_cells)
        for i,row in r.iterrows():
            rows.append({"first_gate_branch":branch,"second_pair_rank":i+1,**row.to_dict()})
    return pd.DataFrame(rows)

def cv_first_gate(donors, markers, t, min_cells, nfolds):
    ids=np.array(sorted(donors)); k=min(nfolds,len(ids))
    rows=[]
    for fold,(tr,te) in enumerate(KFold(k,shuffle=True,random_state=SEED).split(ids),1):
        train=ids[tr].tolist(); test=ids[te].tolist()
        r,_=rank_pairs(donors,train,markers,t,min_cells)
        b=r.iloc[0]; g1,g2=str(b.marker1),str(b.marker2)
        _,s=pair_summary(donors,test,g1,g2,t,min_cells)
        rows.append(dict(fold=fold,n_train_donors=len(train),n_test_donors=len(test),
                         selected_marker1=g1,selected_marker2=g2,
                         train_separation=float(b.separation),test_separation=s["separation"],
                         test_max_high_enrichment=s["max_high_enrichment"],
                         test_max_low_enrichment=s["max_low_enrichment"]))
    return pd.DataFrame(rows)

def main():
    a=parse_args()
    markers=[m.upper() for m in a.markers]
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    donors=load_donors(a.cache_dir,markers,a.score_extreme_q)
    ids=sorted(donors); n=sum(len(x) for x in donors.values())
    print(f"Loaded {len(ids)} donors / {n:,} cells")
    rank,details=rank_pairs(donors,ids,markers,a.gate_quantile,a.min_cells_per_quadrant)
    rank.to_csv(out/"pairwise_gate_ranking.csv",index=False)
    details.to_csv(out/"pairwise_quadrant_details.csv",index=False)
    first=(str(rank.iloc[0].marker1),str(rank.iloc[0].marker2))
    fq=agg_quads(details,*first); fq.to_csv(out/"first_gate_quadrants.csv",index=False)
    rem=[g for g in markers if g not in first]
    br=branch_ranking(donors,first,fq.quadrant.tolist(),rem,a.gate_quantile,a.min_cells_per_quadrant)
    br.to_csv(out/"branch_second_gate_recommendations.csv",index=False)
    cv=cv_first_gate(donors,markers,a.gate_quantile,a.min_cells_per_quadrant,a.cv_folds)
    cv.to_csv(out/"donor_heldout_first_gate_cv.csv",index=False)
    txt=[]
    txt.append(f"Donors: {len(ids)}; cells: {n:,}")
    txt.append(f"Markers: {', '.join(markers)}")
    txt.append(f"Recommended first pair: {first[0]} vs {first[1]}")
    txt.append("")
    txt.append("First-gate quadrants (high to low persister score):")
    for _,r in fq.iterrows():
        txt.append(f"{r.quadrant}: median score pct={r.median_score_pct:.3f}; high enrich={r.high_enrichment_vs_baseline:.2f}x; low enrich={r.low_enrichment_vs_baseline:.2f}x; cells={100*r.median_cell_fraction:.1f}%")
    if not br.empty:
        txt.append("")
        txt.append("Best second pair within each first-gate branch:")
        for branch,z in br.groupby("first_gate_branch"):
            rr=z.sort_values("second_pair_rank").iloc[0]
            txt.append(f"{branch}: {rr.marker1} vs {rr.marker2}; separation={rr.separation:.3f}")
    txt.append("")
    txt.append("NOTE: transcript-level simulation only. Actual protein gates require flow/FMO controls.")
    (out/"proposed_sorting_scheme.txt").write_text("\n".join(txt)+"\n")
    print("\nTOP PAIRS")
    print(rank.head(10).to_string(index=False))
    print(f"\nRecommended first gate: {first[0]} vs {first[1]}")
    print("\nFIRST-GATE QUADRANTS")
    print(fq.to_string(index=False))
    if not br.empty:
        print("\nBEST SECOND PAIR PER BRANCH")
        print(br[br.second_pair_rank==1][["first_gate_branch","marker1","marker2","separation","pair_composite"]].to_string(index=False))
    print("\nDONOR-HELD-OUT CV")
    print(cv.to_string(index=False))
    print(f"\nOutputs: {out}")

if __name__=="__main__":
    main()
