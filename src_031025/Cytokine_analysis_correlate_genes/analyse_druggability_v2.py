#!/usr/bin/env python3
"""
Persister Model Gene Annotation Pipeline (CPDB + AML; strict/lenient modes)

- Uses CellPhoneDB v5 if available, falls back to AML-curated lists
- Flags receptors / ligands / secreted / membrane / cytokines / kinases
- Druggability scoring with AML-aware boosts
- Strict vs Lenient modes for defining "surface" and "neutralizable" targets
    * strict  : surface = receptor & membrane; neutralizable = secreted | cytokine
    * lenient : surface = receptor | membrane; neutralizable = secreted | cytokine | ligand
- Saves: annotations, scores, top lists, summary JSON, and a PNG

Run:
  python analyse_druggability_v3.py --mode strict
  python analyse_druggability_v3.py --mode lenient
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

PALETTE = {
    "blue":   "#1f77b4",
    "orange": "#ff7f0e",
    "green":  "#2ca02c",
    "red":    "#d62728",
    "purple": "#9467bd",
    "teal":   "#17becf",
    "gray":   "#7f7f7f"
}

# =============================================================================
# CellPhoneDB integration
# =============================================================================

def annotate_with_cellphonedb(gene_list, output_dir):
    """
    Annotate model genes using CellPhoneDB (if available) + AML curation.
    Returns: df_annotations, interaction_partners, meta dict
    """
    print("="*80)
    print("CELLPHONEDB ANNOTATION")
    print("="*80)

    cpdb_used = False
    genes_db = proteins_db = complexes_db = interactions_db = pd.DataFrame()

    try:
        cpdb_url = "https://raw.githubusercontent.com/ventolab/cellphonedb-data/master/data/"
        genes_url = cpdb_url + "gene_input.csv"
        protein_url = cpdb_url + "protein_input.csv"
        complex_url = cpdb_url + "complex_input.csv"
        interaction_url = cpdb_url + "interaction_input.csv"

        print("Loading CellPhoneDB databases...")
        genes_db = pd.read_csv(genes_url)
        proteins_db = pd.read_csv(protein_url)
        complexes_db = pd.read_csv(complex_url)
        interactions_db = pd.read_csv(interaction_url)
        cpdb_used = True
        print("✓ CellPhoneDB loaded.")
        print(f"CPDB genes available: {len(genes_db)}")
    except Exception as e:
        print(f"Warning: Could not load CellPhoneDB online ({e}). Using AML curation only.")

    # Normalize gene names
    model_genes_upper = [str(g).upper() for g in gene_list]

    # CPDB matching (optional)
    matched_proteins = pd.DataFrame()
    if cpdb_used and "gene_name" in genes_db.columns:
        matched_genes = genes_db[genes_db["gene_name"].str.upper().isin(model_genes_upper)]
        if "uniprot" in matched_genes.columns and "uniprot" in proteins_db.columns:
            matched_proteins = proteins_db[proteins_db["uniprot"].isin(matched_genes["uniprot"])].copy()

    # Build category sets from CPDB
    cpdb_sets = {k: set() for k in ["receptor","ligand","secreted","membrane","cytokine","growth_factor","kinase","transcription_factor"]}
    if not matched_proteins.empty:
        # CPDB columns vary; be resilient
        def as_bool(x):
            if isinstance(x, (int, float, np.integer, np.floating)):
                return bool(x)
            s = str(x).strip().lower()
            return s in {"true","t","1","yes","y"}
        for _, row in matched_proteins.iterrows():
            g = str(row.get("gene_name","")).upper()
            if not g: 
                continue
            tags = str(row.get("tags","")).lower()
            # Guessers from CPDB fields
            if as_bool(row.get("receptor", False)) or "receptor" in tags:
                cpdb_sets["receptor"].add(g)
            if as_bool(row.get("secreted", False)) or "secreted" in tags:
                cpdb_sets["secreted"].add(g)
            if as_bool(row.get("transmembrane", False)) or "membrane" in tags:
                cpdb_sets["membrane"].add(g)
            if "cytokine" in tags:
                cpdb_sets["cytokine"].add(g)
            if "growth_factor" in tags:
                cpdb_sets["growth_factor"].add(g)
            if "ligand" in tags:
                cpdb_sets["ligand"].add(g)
            # heuristic kinase
            if "kinase" in tags:
                cpdb_sets["kinase"].add(g)

    # AML-focused curated categories
    aml_sets = categorize_aml_genes()

    # Union CPDB + AML within each category, but keep dict for later logic
    union_sets = {k: (cpdb_sets[k] | aml_sets[k]) for k in aml_sets.keys()}

    # Build per-gene annotation frame
    rows = []
    for gene in gene_list:
        gu = str(gene).upper()
        rows.append({
            "gene": gene,
            "is_receptor": gu in union_sets["receptor"],
            "is_ligand": gu in union_sets["ligand"],
            "is_secreted": gu in union_sets["secreted"],
            "is_membrane": gu in union_sets["membrane"],
            "is_cytokine": gu in union_sets["cytokine"],
            "is_growth_factor": gu in union_sets["growth_factor"],
            "is_kinase": gu in union_sets["kinase"],
            "is_transcription_factor": gu in union_sets["transcription_factor"]
        })
    df_annotations = pd.DataFrame(rows)

    # Interactions (optional)
    print("\nFinding interaction partners from CPDB interactions...")
    interaction_partners = pd.DataFrame()
    if cpdb_used and not interactions_db.empty:
        interaction_partners = find_interaction_partners(gene_list, interactions_db)

    # Save quick CPDB/AML summary
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df_annotations.to_csv(outdir/"cellphonedb_annotations.csv", index=False)
    if not interaction_partners.empty:
        interaction_partners.to_csv(outdir/"interaction_partners.csv", index=False)

    print("\nCellPhoneDB Summary (union with AML lists):")
    for k, title in [("receptor","Receptor"),("ligand","Ligand"),("secreted","Secreted"),
                     ("membrane","Membrane"),("cytokine","Cytokine"),("kinase","Kinase")]:
        print(f"  {title:<9}: {int(df_annotations[f'is_{k}'].sum())}")
    print(f"  CPDB used: {cpdb_used}")

    meta = {"cpdb_used": cpdb_used}
    return df_annotations, interaction_partners, meta


def categorize_aml_genes():
    """AML-focused curated lists (expanded RTKs counted as kinases)."""
    def up(s): return {x.upper() for x in s}

    receptor = up([
        "CD33","CD123","IL3RA","CD47","CD70","CD38","CD34",
        "FLT3","KIT","CSF1R","CSF3R","EPOR","MPL",
        "EGFR","ERBB2","ERBB3","ERBB4","MET","PDGFRA","PDGFRB",
        "FGFR1","FGFR2","FGFR3","FGFR4","FLT1","KDR","FLT4",
        "HAVCR2","PDCD1","CD274","CTLA4","TNFRSF8","CD30"
    ])
    ligand = up([
        "VEGFA","VEGFB","VEGFC","FGF1","FGF2","PDGFA","PDGFB","HBEGF",
        "CSF1","CSF2","CSF3","IL3","THPO","EPO","KITLG","TPO","EGF","AREG","EREG","BTC","EPGN","TGFA"
    ])
    secreted = up([
        "IL1B","IL2","IL3","IL4","IL5","IL6","IL7","CXCL8","IL10","IL12A","IL12B","IL15","IL17A","IL18",
        "TNF","IFNG","IFNA1","IFNB1","TGFB1","TGFB2","BMP2","BMP4","CXCL1","CXCL2","CXCL12","CCL2","CCL5","CSF1","CSF2","CSF3"
    ])
    membrane = up([
        "CD33","CD34","CD38","PTPRC","CD47","CD70","CD123","IL3RA","KIT","FLT3",
        "SDC1","ICAM1","ITGAM","SELL","CXCR4","CCR7"
    ])
    cytokine = up([
        "IL1B","IL2","IL3","IL4","IL5","IL6","IL7","IL10","IL12A","IL12B","IL15","IL17A","IL18","TNF","CSF1","CSF2","CSF3","IFNG","IFNA1","IFNB1"
    ])
    growth_factor = up(["VEGFA","VEGFB","VEGFC","FGF1","FGF2","PDGFA","PDGFB","EGF","TGFA","IGF1","IGF2","HGF"])
    kinase = up([
        # classical kinases + RTKs
        "FLT3","KIT","ABL1","ABL2","JAK1","JAK2","JAK3","BTK","SYK","SRC","LYN","LCK","MAPK1","MAPK3","MAP2K1","MAP2K2","BRAF","RAF1",
        "PIK3CA","PIK3CB","PIK3CD","PIK3CG","AKT1","AKT2","AKT3","MTOR","CDK4","CDK6",
        "EGFR","ERBB2","ERBB3","ERBB4","MET","PDGFRA","PDGFRB","FGFR1","FGFR2","FGFR3","FGFR4","FLT1","KDR","FLT4","AXL","MERTK","TYRO3","RET","ROS1","ALK"
    ])
    tf = up(["RUNX1","CEBPA","SPI1","GATA1","GATA2","MYC","MYB","ETV6","MECOM","EVI1","HOXA9","MEIS1","NPM1"])

    return {
        "receptor": receptor, "ligand": ligand, "secreted": secreted, "membrane": membrane,
        "cytokine": cytokine, "growth_factor": growth_factor, "kinase": kinase,
        "transcription_factor": tf
    }


def find_interaction_partners(gene_list, interactions_db):
    """Find CPDB ligand-receptor partners involving model genes."""
    if interactions_db.empty:
        return pd.DataFrame()
    genes_set = {str(g).upper() for g in gene_list}
    rows = []
    for _, r in interactions_db.iterrows():
        pa = str(r.get("partner_a",""))
        pb = str(r.get("partner_b",""))
        a_up, b_up = pa.upper(), pb.upper()
        if any(g in a_up or g in b_up for g in genes_set):
            rows.append({
                "partner_a": pa,
                "partner_b": pb,
                "annotation_strategy": r.get("annotation_strategy",""),
                "is_ligand_receptor": "ligand-receptor" in str(r.get("annotation_strategy","")).lower()
            })
    return pd.DataFrame(rows)

# =============================================================================
# Druggability
# =============================================================================

def assess_druggability(df_annotations, mode="strict"):
    """
    Score and flag druggability. Mode controls target definitions:
      strict  : surface = receptor & membrane; neutralizable = secreted | cytokine
      lenient : surface = receptor | membrane; neutralizable = secreted | cytokine | ligand
    """
    print("\n" + "="*80)
    print("DRUGGABILITY ASSESSMENT")
    print("="*80)

    clinical_or_approved = {
        # include AML-relevant approved/clinical targets
        "FLT3","KIT","BCL2","IDH1","IDH2","CD33","CD123","EGFR","MET","PDGFRA","PDGFRB","KDR","VEGFA","IL6","TNF"
    }
    antibody_targets = {
        "CD33","CD123","CD47","CD70","CD38","EGFR","HER2","PDCD1","CD274","CTLA4","ICAM1","SDC1"
    }

    rows = []
    for _, r in df_annotations.iterrows():
        g = str(r["gene"])
        gu = g.upper()
        # base score from flags
        score = 0
        score += 3 if r["is_receptor"] else 0
        score += 2 if r["is_membrane"] else 0
        score += 2 if r["is_secreted"] else 0
        score += 1 if r["is_cytokine"] else 0
        score += 3 if r.get("is_kinase", False) else 0

        # boosts
        is_clinical = gu in clinical_or_approved
        is_ab = gu in antibody_targets
        if is_clinical: score += 5
        if is_ab:       score += 4
        if r["is_receptor"] and r["is_membrane"]:
            score += 2  # extra when truly a surface receptor

        # mode-dependent target flags
        if mode == "strict":
            is_surface_target = bool(r["is_receptor"] and r["is_membrane"])
            is_secreted_target = bool(r["is_secreted"] or r["is_cytokine"])
        else:  # lenient
            is_surface_target = bool(r["is_receptor"] or r["is_membrane"])
            is_secreted_target = bool(r["is_secreted"] or r["is_cytokine"] or r["is_ligand"])

        rows.append({
            "gene": g,
            "total_druggability_score": score,
            "category": categorize_druggability(score),
            "is_surface_target": is_surface_target,
            "is_secreted_target": is_secreted_target,
            "is_fda_or_clinical": is_clinical,
            "is_antibody_target": is_ab,
            # keep raw flags for downstream filtering if needed
            "is_receptor": bool(r["is_receptor"]),
            "is_membrane": bool(r["is_membrane"]),
            "is_secreted": bool(r["is_secreted"]),
            "is_cytokine": bool(r["is_cytokine"]),
            "is_ligand": bool(r["is_ligand"]),
            "is_kinase": bool(r.get("is_kinase", False)),
        })
    return pd.DataFrame(rows)


def categorize_druggability(score):
    if score >= 12: return "High_Priority"
    if score >= 7:  return "Medium_Priority"
    if score >= 3:  return "Low_Priority"
    return "Not_Druggable"


def identify_therapeutic_targets(df_druggability, interaction_partners):
    """
    Create top lists for surface / secreted and LR pairs (if available).
    """
    print("\n" + "="*80)
    print("THERAPEUTIC TARGET IDENTIFICATION")
    print("="*80)

    high_priority = df_druggability[df_druggability["category"].isin(["High_Priority","Medium_Priority"])]

    surface_targets = high_priority[high_priority["is_surface_target"]].sort_values(
        ["total_druggability_score","is_antibody_target","is_fda_or_clinical"], ascending=[False, False, False]
    )
    secreted_targets = high_priority[high_priority["is_secreted_target"]].sort_values(
        ["total_druggability_score","is_fda_or_clinical"], ascending=[False, False]
    )

    lr_pairs = pd.DataFrame()
    if not interaction_partners.empty and "is_ligand_receptor" in interaction_partners.columns:
        lr_pairs = interaction_partners[interaction_partners["is_ligand_receptor"]].copy()

    print(f"\nTherapeutic Target Summary:")
    print(f"  Total high-priority targets: {len(high_priority)}")
    print(f"  Surface receptors: {len(surface_targets)}")
    print(f"  Secreted factors: {len(secreted_targets)}")
    print(f"  Ligand-receptor pairs: {len(lr_pairs)}")

    return {
        "surface_receptors": surface_targets.head(25),
        "secreted_factors": secreted_targets.head(25),
        "ligand_receptor_pairs": lr_pairs.head(25)
    }

# =============================================================================
# Visualization
# =============================================================================

def visualize_druggability(df_druggability, df_annotations, output_dir, mode):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1) Score distribution
    ax = axes[0, 0]
    ax.hist(df_druggability["total_druggability_score"], bins=20,
            color=PALETTE["blue"], edgecolor="black")
    ax.set_title("Druggability Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of Genes")

    # 2) Priority categories (fixed order + colors)
    ax = axes[0, 1]
    order = ["Not_Druggable", "Low_Priority", "Medium_Priority", "High_Priority"]
    cat_counts = df_druggability["category"].value_counts().reindex(order, fill_value=0)
    cat_colors = [PALETTE["gray"], PALETTE["orange"], PALETTE["purple"], PALETTE["red"]]
    ax.bar(range(len(cat_counts)), cat_counts.values, color=cat_colors)
    ax.set_xticks(range(len(cat_counts)))
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Druggability Categories")

    # 3) Protein types in model (fixed mapping -> color)
    ax = axes[0, 2]
    labels = ["Receptor", "Ligand", "Secreted", "Membrane", "Cytokine", "Kinase"]
    cols   = ["is_receptor", "is_ligand", "is_secreted", "is_membrane", "is_cytokine", "is_kinase"]
    vals   = [int(df_annotations.get(c, pd.Series(dtype=bool)).sum()) for c in cols]
    type_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
                   PALETTE["purple"], PALETTE["red"], PALETTE["teal"]]
    ax.bar(range(len(vals)), vals, color=time_colors if (time_colors := type_colors) else type_colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Protein Types in Model")

    # 4) Top 20 druggable targets (all same color for consistency)
    ax = axes[1, 0]
    top20 = df_druggability.nlargest(20, "total_druggability_score")
    ax.barh(top20["gene"], top20["total_druggability_score"], color=PALETTE["blue"])
    ax.invert_yaxis()
    ax.set_xlabel("Druggability Score")
    ax.set_title("Top 20 Druggable Targets")

    # 5) Target accessibility (surface/secreted/both)
    ax = axes[1, 1]
    s_cnt   = int(df_druggability["is_surface_target"].sum())
    sec_cnt = int(df_druggability["is_secreted_target"].sum())
    both_cnt = int((df_druggability["is_surface_target"] & df_druggability["is_secreted_target"]).sum())
    labels_acc = ["Surface", "Secreted", "Both"]
    vals_acc   = [s_cnt, sec_cnt, both_cnt]
    acc_colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"]]
    ax.bar(labels_acc, vals_acc, color=acc_colors)
    ax.set_title(f"Target Accessibility (mode={mode})")

    # 6) Validated buckets (FDA/clinical, antibody)
    ax = axes[1, 2]
    fda = int(df_druggability["is_fda_or_clinical"].sum())
    ab  = int(df_druggability["is_antibody_target"].sum())
    ax.bar(["FDA/Clinical", "Antibody"], [fda, ab],
           color=[PALETTE["purple"], PALETTE["orange"]])
    ax.set_title("Validated Target Buckets")

    plt.suptitle("Persister Model Druggability Analysis", fontsize=16)
    plt.tight_layout()

    outpath = Path(output_dir) / "druggability_analysis.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {outpath}")
    return fig

# def visualize_druggability(df_druggability, df_annotations, output_dir, mode):
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))

#     # 1. Score distribution
#     ax = axes[0,0]
#     ax.hist(df_druggability["total_druggability_score"], bins=20, edgecolor="black")
#     ax.set_title("Druggability Score Distribution"); ax.set_xlabel("Score"); ax.set_ylabel("Genes")

#     # 2. Categories
#     ax = axes[0,1]
#     counts = df_druggability["category"].value_counts()
#     ax.bar(range(len(counts)), counts.values)
#     ax.set_xticks(range(len(counts))); ax.set_xticklabels(counts.index, rotation=30); ax.set_title("Priority Categories")

#     # 3. Protein types
#     ax = axes[0,2]
#     prot_cols = ["is_receptor","is_ligand","is_secreted","is_membrane","is_cytokine","is_kinase"]
#     vals = [int(df_annotations.get(c, pd.Series(dtype=bool)).sum()) for c in prot_cols]
#     ax.bar(range(len(prot_cols)), vals)
#     ax.set_xticks(range(len(prot_cols))); ax.set_xticklabels(["Receptor","Ligand","Secreted","Membrane","Cytokine","Kinase"], rotation=30)
#     ax.set_title("Protein Types in Model")

#     # 4. Top 20
#     ax = axes[1,0]
#     top20 = df_druggability.nlargest(20, "total_druggability_score")
#     ax.barh(top20["gene"], top20["total_druggability_score"]); ax.invert_yaxis()
#     ax.set_title("Top 20 Druggable Targets"); ax.set_xlabel("Score")

#     # 5. Target accessibility
#     ax = axes[1,1]
#     s_cnt = int(df_druggability["is_surface_target"].sum())
#     sec_cnt = int(df_druggability["is_secreted_target"].sum())
#     both_cnt = int((df_druggability["is_surface_target"] & df_druggability["is_secreted_target"]).sum())
#     ax.bar(["Surface","Secreted","Both"], [s_cnt, sec_cnt, both_cnt])
#     ax.set_title(f"Target Accessibility (mode={mode})")

#     # 6. Validated targets
#     ax = axes[1,2]
#     fda = int(df_druggability["is_fda_or_clinical"].sum())
#     ab  = int(df_druggability["is_antibody_target"].sum())
#     ax.bar(["FDA Target","Antibody Target"], [fda, ab])
#     ax.set_title("Validated Target Buckets")

#     plt.suptitle("Persister Model Druggability Analysis", fontsize=16)
#     plt.tight_layout()

#     outpath = Path(output_dir)/"druggability_analysis.png"
#     plt.savefig(outpath, dpi=300, bbox_inches="tight")
#     print(f"\nVisualization saved to: {outpath}")
#     return fig

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Druggability annotation (CPDB + AML)")
    parser.add_argument("--model-dir", default="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled", help="Folder with selected_genes.txt")
    parser.add_argument("--output-dir", default="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/druggability_analysis", help="Output folder")
    parser.add_argument("--mode", choices=["strict","lenient"], default="strict", help="Target definition strictness")
    args = parser.parse_args()

    MODEL_DIR = Path(args.model_dir)
    OUTPUT_DIR = Path(args.output_dir)
    mode = args.mode

    print("\n" + "="*80)
    print("PERSISTER MODEL DRUGGABILITY ANALYSIS (CPDB + AML)")
    print("="*80)

    # Load model genes
    gene_file = MODEL_DIR/"selected_genes.txt"
    with open(gene_file) as f:
        model_genes = [x.strip() for x in f if x.strip()]
    print(f"Loaded {len(model_genes)} model genes")

    # Annotations (CPDB + AML)
    df_annotations, interaction_partners, meta = annotate_with_cellphonedb(model_genes, OUTPUT_DIR)

    # Scoring
    df_druggability = assess_druggability(df_annotations, mode=mode)

    # Targets
    target_report = identify_therapeutic_targets(df_druggability, interaction_partners)

    # Viz
    _ = visualize_druggability(df_druggability, df_annotations, OUTPUT_DIR, mode=mode)

    # Save tables
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_druggability.to_csv(OUTPUT_DIR/"druggability_scores.csv", index=False)
    (target_report["surface_receptors"]).to_csv(OUTPUT_DIR/"top_surface_receptors.csv", index=False)
    (target_report["secreted_factors"]).to_csv(OUTPUT_DIR/"top_secreted_factors.csv", index=False)
    if not target_report["ligand_receptor_pairs"].empty:
        target_report["ligand_receptor_pairs"].to_csv(OUTPUT_DIR/"ligand_receptor_pairs.csv", index=False)

    # Summary
    summary = {
        "mode": mode,
        "cpdb_used": bool(meta.get("cpdb_used", False)),
        "total_genes": len(model_genes),
        "druggable_genes": int((df_druggability["category"] != "Not_Druggable").sum()),
        "high_priority_targets": int((df_druggability["category"] == "High_Priority").sum()),
        "surface_receptors": int(df_annotations["is_receptor"].sum()),
        "secreted_factors": int(df_annotations["is_secreted"].sum()),
        "cytokines": int(df_annotations["is_cytokine"].sum()),
        "kinases": int(df_annotations.get("is_kinase", pd.Series(dtype=bool)).sum()),
        "clinical_or_approved_targets": int(df_druggability["is_fda_or_clinical"].sum()),
        "antibody_accessible_targets": int(df_druggability["is_antibody_target"].sum())
    }
    with open(OUTPUT_DIR/"druggability_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDRUGGABILITY SUMMARY:")
    for k,v in summary.items():
        print(f"  {k.replace('_',' ').title():<28}: {v}")

    print(f"\n✓ Analysis complete! Results saved to: {OUTPUT_DIR}")

    # Console suggestions
    print("\n" + "="*80)
    print("EXPERIMENTAL VALIDATION RECOMMENDATIONS")
    print("="*80)
    surf = target_report["surface_receptors"].head(5)
    secr = target_report["secreted_factors"].head(5)
    if not surf.empty:
        print("\n1. TOP ANTIBODY TARGETS (Surface receptors):")
        for _, r in surf.iterrows():
            print(f"   - {r['gene']}: Score={r['total_druggability_score']:.1f}")
    if not secr.empty:
        print("\n2. NEUTRALIZATION TARGETS (Secreted/cytokine/ligand):")
        for _, r in secr.iterrows():
            print(f"   - {r['gene']}: Score={r['total_druggability_score']:.1f}")
    print("\n3. RECOMMENDED EXPERIMENTS:")
    print("   • Flow cytometry validation of surface markers")
    print("   • ELISA/Luminex for secreted/cytokines")
    print("   • Antibody blocking or ligand traps")
    print("   • CRISPR KO/pooled screens of top targets")
    print("   • Small-molecule inhibitor tests (where available)")

if __name__ == "__main__":
    main()
