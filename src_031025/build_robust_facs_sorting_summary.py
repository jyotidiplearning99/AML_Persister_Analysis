#!/usr/bin/env python3
"""
build_robust_facs_sorting_summary.py

Robust FACS-oriented summary for the AML persister-score sorting analysis.

Core recommendation:
    FLT3 x CD33 four-population gate is the cohort-robust workhorse.

Optional refinements:
    FLT3hi/CD33hi -> NECTIN2 x KDR
    FLT3lo/CD33lo -> NECTIN2 x EPHB2

This script:
  1. Loads the 39-donor cached single-cell expression tables.
  2. Reconstructs the FLT3 x CD33 first gate.
  3. Builds a mutually exclusive/exhaustive six-population exploratory tree:
       P1 = HH -> NECTIN2hi/KDRhi
       P2 = remainder of HH
       P3 = FLT3hi/CD33lo
       P4 = FLT3lo/CD33hi
       P5 = remainder of LL
       P6 = LL -> NECTIN2lo/EPHB2lo
  4. Adds:
       - n_donors
       - donor_coverage_fraction
       - cohort_robust flag
       - enrichment_gain_vs_gate1_parent
       - second_gate_worth_it flag
  5. Writes a recommendation table separating:
       CORE / OPTIONAL / EXPLORATORY
  6. Keeps a prominent warning that transcript percentile gates are not
     literal protein/FACS thresholds.

Important:
This is transcript-level gate-order prioritization only.
Actual FACS thresholds require protein-level fluorescence distributions,
FMO controls, and experimental optimization.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MARKERS = ["FLT3", "CD33", "NECTIN2", "CD44", "KDR", "EPHB2"]
FIRST = ("FLT3", "CD33")
HIGH_SECOND = ("NECTIN2", "KDR")
LOW_SECOND = ("NECTIN2", "EPHB2")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-dir",
        default="results/surface_surrogate/cache_joined_candidate_expression",
    )
    p.add_argument(
        "--out-dir",
        default="results/biaxial_sorting",
    )
    p.add_argument("--gate-quantile", type=float, default=0.50)
    p.add_argument("--score-extreme-q", type=float, default=0.25)
    p.add_argument("--min-cells-per-donor", type=int, default=30)

    # User-requested honesty thresholds:
    p.add_argument(
        "--robust-donor-threshold",
        type=int,
        default=25,
        help="Minimum donor count for cohort-robust label.",
    )
    p.add_argument(
        "--worthwhile-enrichment-gain",
        type=float,
        default=1.30,
        help="Minimum multiplicative enrichment gain over gate-1 parent.",
    )
    return p.parse_args()

def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)

def quad(df: pd.DataFrame, g1: str, g2: str, t: float) -> pd.Series:
    a = df[f"{g1}_pct"] >= t
    b = df[f"{g2}_pct"] >= t
    vals = np.select(
        [a & b, a & ~b, ~a & b],
        [f"{g1}hi/{g2}hi", f"{g1}hi/{g2}lo", f"{g1}lo/{g2}hi"],
        default=f"{g1}lo/{g2}lo",
    )
    return pd.Series(vals, index=df.index)

def load_donors(cache_dir: str, score_q: float) -> dict[str, pd.DataFrame]:
    donors = {}
    for f in sorted(Path(cache_dir).glob("*.pkl")):
        d = pd.read_pickle(f)
        missing = [g for g in MARKERS if g not in d.columns]
        if missing:
            print(f"[skip] {f.stem}: missing {missing}")
            continue

        z = d[["score"] + MARKERS].copy()
        z["donor"] = f.stem
        z["score_pct"] = pct_rank(z["score"])
        z["score_high"] = z["score_pct"] >= (1.0 - score_q)
        z["score_low"] = z["score_pct"] <= score_q

        for g in MARKERS:
            z[f"{g}_pct"] = pct_rank(z[g])

        donors[f.stem] = z
        print(f"[ok] {f.stem}: {len(z):,}")

    return donors

def first_gate_labels(d: pd.DataFrame, t: float) -> pd.Series:
    return quad(d, *FIRST, t)

def assign_six_pops(d: pd.DataFrame, t: float) -> tuple[pd.Series, pd.Series]:
    q1 = first_gate_labels(d, t)
    pop = pd.Series(index=d.index, dtype="object")
    desc = pd.Series(index=d.index, dtype="object")

    hh = q1 == "FLT3hi/CD33hi"
    hl = q1 == "FLT3hi/CD33lo"
    lh = q1 == "FLT3lo/CD33hi"
    ll = q1 == "FLT3lo/CD33lo"

    pop.loc[hl] = "P3"
    desc.loc[hl] = "FLT3hi/CD33lo"

    pop.loc[lh] = "P4"
    desc.loc[lh] = "FLT3lo/CD33hi"

    if hh.any():
        b = d.loc[hh].copy()
        q2 = quad(b, *HIGH_SECOND, t)
        p1 = q2 == "NECTIN2hi/KDRhi"
        pop.loc[b.index[p1]] = "P1"
        desc.loc[b.index[p1]] = "FLT3hi/CD33hi -> NECTIN2hi/KDRhi"
        pop.loc[b.index[~p1]] = "P2"
        desc.loc[b.index[~p1]] = "FLT3hi/CD33hi -> remainder"

    if ll.any():
        b = d.loc[ll].copy()
        q2 = quad(b, *LOW_SECOND, t)
        p6 = q2 == "NECTIN2lo/EPHB2lo"
        pop.loc[b.index[~p6]] = "P5"
        desc.loc[b.index[~p6]] = "FLT3lo/CD33lo -> remainder"
        pop.loc[b.index[p6]] = "P6"
        desc.loc[b.index[p6]] = "FLT3lo/CD33lo -> NECTIN2lo/EPHB2lo"

    return pop, desc

def summarize_group(
    donors: dict[str, pd.DataFrame],
    selector,
    label: str,
    description: str,
    min_cells: int,
) -> pd.DataFrame:
    rows = []
    for donor, d in donors.items():
        z = selector(d)
        if len(z) < min_cells:
            continue
        rows.append(
            {
                "population": label,
                "description": description,
                "donor": donor,
                "n_cells": len(z),
                "cell_fraction": len(z) / len(d),
                "median_score_pct": float(z["score_pct"].median()),
                "high_score_fraction": float(z["score_high"].mean()),
                "low_score_fraction": float(z["score_low"].mean()),
            }
        )
    return pd.DataFrame(rows)

def aggregate_population(per_donor: pd.DataFrame, total_donors: int) -> pd.DataFrame:
    s = (
        per_donor.groupby(["population", "description"], as_index=False)
        .agg(
            n_donors=("donor", "nunique"),
            median_cell_fraction=("cell_fraction", "median"),
            median_score_pct=("median_score_pct", "median"),
            median_high_fraction=("high_score_fraction", "median"),
            median_low_fraction=("low_score_fraction", "median"),
        )
    )
    s["donor_coverage_fraction"] = s["n_donors"] / total_donors
    s["high_enrichment_vs_baseline"] = s["median_high_fraction"] / 0.25
    s["low_enrichment_vs_baseline"] = s["median_low_fraction"] / 0.25
    return s

def main():
    a = parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    donors = load_donors(a.cache_dir, a.score_extreme_q)
    total_donors = len(donors)
    total_cells = sum(len(d) for d in donors.values())

    print(f"\nLoaded {total_donors} donors / {total_cells:,} cells")
    if total_donors != 39:
        print(f"[warning] expected 39 donors, got {total_donors}")

    t = a.gate_quantile
    min_cells = a.min_cells_per_donor

    # ------------------------------------------------------------
    # 1) CORE: first-gate four populations
    # ------------------------------------------------------------
    first_rows = []
    first_order = [
        "FLT3hi/CD33hi",
        "FLT3hi/CD33lo",
        "FLT3lo/CD33hi",
        "FLT3lo/CD33lo",
    ]

    for qname in first_order:
        def sel(d, qname=qname):
            q = first_gate_labels(d, t)
            return d.loc[q == qname].copy()

        z = summarize_group(
            donors, sel, qname, qname, min_cells
        )
        if not z.empty:
            first_rows.append(z)

    first_per_donor = pd.concat(first_rows, ignore_index=True)
    first_summary = aggregate_population(first_per_donor, total_donors)

    first_summary["cohort_robust"] = (
        first_summary["n_donors"] >= a.robust_donor_threshold
    )
    first_summary["recommendation_level"] = "CORE"

    # Parent enrichment lookup
    parent_high = dict(
        zip(
            first_summary["population"],
            first_summary["high_enrichment_vs_baseline"],
        )
    )
    parent_low = dict(
        zip(
            first_summary["population"],
            first_summary["low_enrichment_vs_baseline"],
        )
    )

    # ------------------------------------------------------------
    # 2) Exploratory six-population pooled tree
    # ------------------------------------------------------------
    pop_rows = []

    for pname in ["P1", "P2", "P3", "P4", "P5", "P6"]:
        def sel(d, pname=pname):
            p, desc = assign_six_pops(d, t)
            z = d.loc[p == pname].copy()
            if len(z):
                z = z.copy()
                z["_desc"] = desc.loc[z.index].iloc[0]
            return z

        # derive a stable description from first donor where present
        desc_value = None
        for d in donors.values():
            p, desc = assign_six_pops(d, t)
            idx = p[p == pname].index
            if len(idx):
                desc_value = desc.loc[idx[0]]
                break

        if desc_value is None:
            continue

        z = summarize_group(
            donors, sel, pname, desc_value, min_cells
        )
        if not z.empty:
            pop_rows.append(z)

    six_per_donor = pd.concat(pop_rows, ignore_index=True)
    six_summary = aggregate_population(six_per_donor, total_donors)

    parent_map = {
        "P1": "FLT3hi/CD33hi",
        "P2": "FLT3hi/CD33hi",
        "P3": "FLT3hi/CD33lo",
        "P4": "FLT3lo/CD33hi",
        "P5": "FLT3lo/CD33lo",
        "P6": "FLT3lo/CD33lo",
    }

    direction_map = {
        "P1": "high",
        "P2": "high",
        "P3": "high",
        "P4": "high",
        "P5": "low",
        "P6": "low",
    }

    six_summary["gate1_parent"] = six_summary["population"].map(parent_map)
    six_summary["cohort_robust"] = (
        six_summary["n_donors"] >= a.robust_donor_threshold
    )

    gains = []
    for _, r in six_summary.iterrows():
        p = r["population"]
        parent = r["gate1_parent"]
        direction = direction_map[p]

        if p in ("P3", "P4"):
            gain = 1.0
        elif direction == "high":
            parent_enr = parent_high[parent]
            gain = (
                r["high_enrichment_vs_baseline"] / parent_enr
                if parent_enr > 0 else np.nan
            )
        else:
            parent_enr = parent_low[parent]
            gain = (
                r["low_enrichment_vs_baseline"] / parent_enr
                if parent_enr > 0 else np.nan
            )
        gains.append(gain)

    six_summary["enrichment_gain_vs_gate1_parent"] = gains
    six_summary["second_gate_worth_it"] = (
        six_summary["enrichment_gain_vs_gate1_parent"]
        >= a.worthwhile_enrichment_gain
    )

    levels = []
    notes = []

    for _, r in six_summary.iterrows():
        p = r["population"]
        robust = bool(r["cohort_robust"])
        worthwhile = bool(r["second_gate_worth_it"])

        if p in ("P3", "P4"):
            level = "CORE"
            note = "Defined directly by FLT3 x CD33 first gate."
        elif robust and worthwhile:
            level = "OPTIONAL"
            note = (
                "Second-gate refinement is cohort-backed and provides "
                "meaningful enrichment gain."
            )
        elif robust and not worthwhile:
            level = "OPTIONAL_LOW_GAIN"
            note = (
                "Cohort-backed, but second-gate enrichment gain is below "
                f"{a.worthwhile_enrichment_gain:.2f}x."
            )
        else:
            level = "EXPLORATORY"
            note = (
                f"Supported by fewer than {a.robust_donor_threshold} donors "
                "under the per-donor cell-count criterion."
            )

        levels.append(level)
        notes.append(note)

    six_summary["recommendation_level"] = levels
    six_summary["interpretation"] = notes

    # ------------------------------------------------------------
    # 3) Self-documenting recommendation table
    # ------------------------------------------------------------
    rec_rows = []

    # Core first gate
    for _, r in first_summary.iterrows():
        rec_rows.append(
            {
                "scheme": "4-population core",
                "population": r["population"],
                "definition": r["description"],
                "n_donors": int(r["n_donors"]),
                "donor_coverage_fraction": r["donor_coverage_fraction"],
                "median_cell_fraction": r["median_cell_fraction"],
                "median_score_pct": r["median_score_pct"],
                "high_enrichment_vs_baseline": r["high_enrichment_vs_baseline"],
                "low_enrichment_vs_baseline": r["low_enrichment_vs_baseline"],
                "enrichment_gain_vs_gate1_parent": 1.0,
                "cohort_robust": bool(r["cohort_robust"]),
                "recommendation_level": "CORE",
                "interpretation": "Primary cohort-robust sorting population.",
            }
        )

    # Optional/exploratory six-pop tree
    for _, r in six_summary.iterrows():
        rec_rows.append(
            {
                "scheme": "6-population exploratory",
                "population": r["population"],
                "definition": r["description"],
                "n_donors": int(r["n_donors"]),
                "donor_coverage_fraction": r["donor_coverage_fraction"],
                "median_cell_fraction": r["median_cell_fraction"],
                "median_score_pct": r["median_score_pct"],
                "high_enrichment_vs_baseline": r["high_enrichment_vs_baseline"],
                "low_enrichment_vs_baseline": r["low_enrichment_vs_baseline"],
                "enrichment_gain_vs_gate1_parent":
                    r["enrichment_gain_vs_gate1_parent"],
                "cohort_robust": bool(r["cohort_robust"]),
                "recommendation_level": r["recommendation_level"],
                "interpretation": r["interpretation"],
            }
        )

    recommendation = pd.DataFrame(rec_rows)

    # ------------------------------------------------------------
    # 4) Write outputs
    # ------------------------------------------------------------
    first_per_donor.to_csv(
        out / "core_4pop_first_gate_per_donor.csv", index=False
    )
    first_summary.to_csv(
        out / "core_4pop_first_gate_summary.csv", index=False
    )
    six_per_donor.to_csv(
        out / "exploratory_6pop_per_donor.csv", index=False
    )
    six_summary.to_csv(
        out / "exploratory_6pop_summary_with_robustness.csv", index=False
    )
    recommendation.to_csv(
        out / "facs_sorting_recommendation_self_documenting.csv", index=False
    )

    # ------------------------------------------------------------
    # 5) Console report
    # ------------------------------------------------------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\n" + "=" * 110)
    print("CORE RECOMMENDATION: FLT3 x CD33 FOUR-POPULATION GATE")
    print("=" * 110)
    cols_core = [
        "population",
        "n_donors",
        "donor_coverage_fraction",
        "median_cell_fraction",
        "median_score_pct",
        "high_enrichment_vs_baseline",
        "low_enrichment_vs_baseline",
        "cohort_robust",
    ]
    print(first_summary[cols_core].to_string(index=False))

    print("\n" + "=" * 110)
    print("EXPLORATORY SIX-POPULATION TREE WITH HONEST ROBUSTNESS FLAGS")
    print("=" * 110)
    cols_six = [
        "population",
        "description",
        "n_donors",
        "donor_coverage_fraction",
        "median_cell_fraction",
        "median_score_pct",
        "high_enrichment_vs_baseline",
        "low_enrichment_vs_baseline",
        "enrichment_gain_vs_gate1_parent",
        "cohort_robust",
        "second_gate_worth_it",
        "recommendation_level",
    ]
    print(six_summary[cols_six].to_string(index=False))

    print("\n" + "=" * 110)
    print("DECISION")
    print("=" * 110)
    print(
        "CORE: Sort FLT3 x CD33 into four populations. "
        "This is the cohort-robust result."
    )
    print(
        "OPTIONAL: Use NECTIN2/KDR or NECTIN2/EPHB2 only where the "
        "additional enrichment gain and donor coverage justify the extra channels."
    )
    print(
        f"Robustness rule: n_donors >= {a.robust_donor_threshold}."
    )
    print(
        f"Second-gate value rule: enrichment gain vs gate-1 parent >= "
        f"{a.worthwhile_enrichment_gain:.2f}x."
    )
    print(
        "\nIMPORTANT: transcript percentile gates are not literal FACS gates. "
        "Actual protein thresholds must be set experimentally using fluorescence "
        "distributions and FMO/control-defined gates."
    )

    print("\nWrote:")
    for fn in [
        "core_4pop_first_gate_per_donor.csv",
        "core_4pop_first_gate_summary.csv",
        "exploratory_6pop_per_donor.csv",
        "exploratory_6pop_summary_with_robustness.csv",
        "facs_sorting_recommendation_self_documenting.csv",
    ]:
        print(" ", out / fn)

if __name__ == "__main__":
    main()
