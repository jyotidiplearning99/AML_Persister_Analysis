#!/usr/bin/env python3
"""
build_p1_p6_sorting_tree.py

Build an explicit six-population sorting proposal from the 39-donor AML
single-cell cache.

Uses:
  First gate: FLT3 x CD33
  Extreme branches:
    FLT3hi/CD33hi -> NECTIN2 x KDR
    FLT3lo/CD33lo -> NECTIN2 x EPHB2

Keeps the two mixed FLT3/CD33 branches as separate populations.

Final six populations:
  P1/P2 = highest and contrasting subpopulation within FLT3hi/CD33hi
  P3    = FLT3hi/CD33lo
  P4    = FLT3lo/CD33hi
  P5/P6 = contrasting and lowest subpopulation within FLT3lo/CD33lo

This is transcript-level prioritization only; flow gates must later be set
using protein fluorescence distributions/FMO controls.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FIRST = ("FLT3", "CD33")
HIGH_SECOND = ("NECTIN2", "KDR")
LOW_SECOND = ("NECTIN2", "EPHB2")
MARKERS = ["FLT3", "CD33", "NECTIN2", "CD44", "KDR", "EPHB2"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir",
                   default="results/surface_surrogate/cache_joined_candidate_expression")
    p.add_argument("--out-dir",
                   default="results/biaxial_sorting")
    p.add_argument("--gate-quantile", type=float, default=0.50)
    p.add_argument("--score-extreme-q", type=float, default=0.25)
    p.add_argument("--min-cells", type=int, default=30)
    return p.parse_args()

def pct_rank(s):
    return s.rank(method="average", pct=True)

def load_donors(cache_dir, q):
    out = {}
    for f in sorted(Path(cache_dir).glob("*.pkl")):
        d = pd.read_pickle(f)
        miss = [g for g in MARKERS if g not in d.columns]
        if miss:
            print(f"[skip] {f.stem}: missing {miss}")
            continue
        z = d[["score"] + MARKERS].copy()
        z["donor"] = f.stem
        z["score_pct"] = pct_rank(z["score"])
        z["score_high"] = z["score_pct"] >= (1-q)
        z["score_low"]  = z["score_pct"] <= q
        for g in MARKERS:
            z[f"{g}_pct"] = pct_rank(z[g])
        out[f.stem] = z
        print(f"[ok] {f.stem}: {len(z):,}")
    return out

def quad(df, g1, g2, t):
    a = df[f"{g1}_pct"] >= t
    b = df[f"{g2}_pct"] >= t
    return pd.Series(np.select(
        [a & b, a & ~b, ~a & b],
        [f"{g1}hi/{g2}hi", f"{g1}hi/{g2}lo", f"{g1}lo/{g2}hi"],
        default=f"{g1}lo/{g2}lo"), index=df.index)

def summarize_population(donors, selector, name, desc, min_cells):
    rows = []
    for donor, d in donors.items():
        z = selector(d)
        if len(z) < min_cells:
            continue
        rows.append({
            "population": name,
            "description": desc,
            "donor": donor,
            "n_cells": len(z),
            "cell_fraction": len(z)/len(d),
            "median_score_pct": float(z["score_pct"].median()),
            "high_score_fraction": float(z["score_high"].mean()),
            "low_score_fraction": float(z["score_low"].mean()),
        })
    return pd.DataFrame(rows)

def branch_second_quadrants(donors, branch_name, second_pair, t, min_cells):
    g1, g2 = FIRST
    s1, s2 = second_pair
    rows = []
    for donor, d in donors.items():
        q1 = quad(d, g1, g2, t)
        b = d.loc[q1 == branch_name].copy()
        if len(b) < min_cells:
            continue
        q2 = quad(b, s1, s2, t)
        b["second_quadrant"] = q2
        for qname, z in b.groupby("second_quadrant"):
            if len(z) < min_cells:
                continue
            rows.append({
                "branch": branch_name,
                "marker1": s1,
                "marker2": s2,
                "second_quadrant": qname,
                "donor": donor,
                "n_cells": len(z),
                "branch_fraction": len(z)/len(b),
                "whole_sample_fraction": len(z)/len(d),
                "median_score_pct": float(z["score_pct"].median()),
                "high_score_fraction": float(z["score_high"].mean()),
                "low_score_fraction": float(z["score_low"].mean()),
            })
    long = pd.DataFrame(rows)
    agg = (long.groupby(["branch","marker1","marker2","second_quadrant"], as_index=False)
           .agg(n_donors=("donor","nunique"),
                median_branch_fraction=("branch_fraction","median"),
                median_whole_sample_fraction=("whole_sample_fraction","median"),
                median_score_pct=("median_score_pct","median"),
                median_high_fraction=("high_score_fraction","median"),
                median_low_fraction=("low_score_fraction","median")))
    agg["high_enrichment_vs_baseline"] = agg["median_high_fraction"]/0.25
    agg["low_enrichment_vs_baseline"] = agg["median_low_fraction"]/0.25
    return long, agg.sort_values("median_score_pct", ascending=False).reset_index(drop=True)

def main():
    a = parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    donors = load_donors(a.cache_dir, a.score_extreme_q)
    if len(donors) != 39:
        print(f"[warning] loaded {len(donors)} donors, expected 39")

    t = a.gate_quantile
    high_branch = "FLT3hi/CD33hi"
    mid1_branch = "FLT3hi/CD33lo"
    mid2_branch = "FLT3lo/CD33hi"
    low_branch  = "FLT3lo/CD33lo"

    hi_long, hi_agg = branch_second_quadrants(
        donors, high_branch, HIGH_SECOND, t, a.min_cells)
    lo_long, lo_agg = branch_second_quadrants(
        donors, low_branch, LOW_SECOND, t, a.min_cells)

    hi_agg.to_csv(out/"high_branch_second_quadrants.csv", index=False)
    lo_agg.to_csv(out/"low_branch_second_quadrants.csv", index=False)
    hi_long.to_csv(out/"high_branch_second_quadrants_per_donor.csv", index=False)
    lo_long.to_csv(out/"low_branch_second_quadrants_per_donor.csv", index=False)

    # Pick the most extreme high and a contrasting high-branch subpopulation.
    hi_best = hi_agg.iloc[0]
    hi_contrast = hi_agg.iloc[-1]

    # Pick the most extreme low and a contrasting low-branch subpopulation.
    lo_highest = lo_agg.iloc[0]
    lo_lowest  = lo_agg.iloc[-1]

    def first_branch_selector(branch):
        return lambda d: d.loc[quad(d, *FIRST, t) == branch].copy()

    def nested_selector(branch, second_pair, second_q):
        def sel(d):
            q1 = quad(d, *FIRST, t)
            b = d.loc[q1 == branch].copy()
            if b.empty:
                return b
            q2 = quad(b, *second_pair, t)
            return b.loc[q2 == second_q].copy()
        return sel

    specs = [
        ("P1",
         f"{high_branch} -> {hi_best.second_quadrant}",
         nested_selector(high_branch, HIGH_SECOND, hi_best.second_quadrant)),
        ("P2",
         f"{high_branch} -> {hi_contrast.second_quadrant}",
         nested_selector(high_branch, HIGH_SECOND, hi_contrast.second_quadrant)),
        ("P3",
         mid1_branch,
         first_branch_selector(mid1_branch)),
        ("P4",
         mid2_branch,
         first_branch_selector(mid2_branch)),
        ("P5",
         f"{low_branch} -> {lo_highest.second_quadrant}",
         nested_selector(low_branch, LOW_SECOND, lo_highest.second_quadrant)),
        ("P6",
         f"{low_branch} -> {lo_lowest.second_quadrant}",
         nested_selector(low_branch, LOW_SECOND, lo_lowest.second_quadrant)),
    ]

    all_rows = []
    for name, desc, selector in specs:
        z = summarize_population(donors, selector, name, desc, a.min_cells)
        if not z.empty:
            all_rows.append(z)
    per_donor = pd.concat(all_rows, ignore_index=True)
    per_donor.to_csv(out/"p1_p6_per_donor_metrics.csv", index=False)

    summary = (per_donor.groupby(["population","description"], as_index=False)
               .agg(n_donors=("donor","nunique"),
                    median_cell_fraction=("cell_fraction","median"),
                    median_score_pct=("median_score_pct","median"),
                    median_high_fraction=("high_score_fraction","median"),
                    median_low_fraction=("low_score_fraction","median")))
    summary["high_enrichment_vs_baseline"] = summary["median_high_fraction"]/0.25
    summary["low_enrichment_vs_baseline"] = summary["median_low_fraction"]/0.25
    summary["population_order"] = summary["population"].str.extract(r'(\d+)').astype(int)
    summary = summary.sort_values("population_order").drop(columns="population_order")
    summary.to_csv(out/"proposed_P1_P6_sorting_tree.csv", index=False)

    print("\n" + "="*90)
    print("HIGH BRANCH: FLT3hi/CD33hi -> NECTIN2 x KDR")
    print("="*90)
    print(hi_agg.to_string(index=False))

    print("\n" + "="*90)
    print("LOW BRANCH: FLT3lo/CD33lo -> NECTIN2 x EPHB2")
    print("="*90)
    print(lo_agg.to_string(index=False))

    print("\n" + "="*90)
    print("PROPOSED SIX SORTED POPULATIONS")
    print("="*90)
    print(summary.to_string(index=False))

    print("\nSORTING TREE")
    print("------------")
    print("FLT3 x CD33")
    print(f"  ├─ FLT3hi/CD33hi -> NECTIN2 x KDR")
    print(f"  │    ├─ P1: {hi_best.second_quadrant}  [highest]")
    print(f"  │    └─ P2: {hi_contrast.second_quadrant}  [contrast]")
    print(f"  ├─ P3: FLT3hi/CD33lo")
    print(f"  ├─ P4: FLT3lo/CD33hi")
    print(f"  └─ FLT3lo/CD33lo -> NECTIN2 x EPHB2")
    print(f"       ├─ P5: {lo_highest.second_quadrant}  [contrast]")
    print(f"       └─ P6: {lo_lowest.second_quadrant}  [lowest]")

    txt = [
        "PROPOSED P1-P6 SORTING TREE",
        "==========================",
        "",
        "First gate: FLT3 x CD33",
        f"  FLT3hi/CD33hi -> NECTIN2 x KDR",
        f"    P1: {hi_best.second_quadrant} [highest-score subpopulation]",
        f"    P2: {hi_contrast.second_quadrant} [contrasting subpopulation]",
        f"  P3: FLT3hi/CD33lo",
        f"  P4: FLT3lo/CD33hi",
        f"  FLT3lo/CD33lo -> NECTIN2 x EPHB2",
        f"    P5: {lo_highest.second_quadrant} [contrasting subpopulation]",
        f"    P6: {lo_lowest.second_quadrant} [lowest-score subpopulation]",
        "",
        "Interpretation:",
        "Transcript-level prioritization only. Actual FACS thresholds require protein-level staining and FMO/control-defined gates.",
    ]
    (out/"proposed_P1_P6_sorting_tree.txt").write_text("\n".join(txt)+"\n")

    print("\nWrote:")
    for f in [
        "high_branch_second_quadrants.csv",
        "low_branch_second_quadrants.csv",
        "p1_p6_per_donor_metrics.csv",
        "proposed_P1_P6_sorting_tree.csv",
        "proposed_P1_P6_sorting_tree.txt",
    ]:
        print(" ", out/f)

if __name__ == "__main__":
    main()
