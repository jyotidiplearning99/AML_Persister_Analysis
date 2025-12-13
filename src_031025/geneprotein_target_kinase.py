#!/usr/bin/env python3
"""
FIXED: Accurate Surface Kinase Identification
Excludes ABC transporters and ATPases from kinase classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path('results/protein_go_pathway_analysis')

def fix_kinase_classification():
    """
    Fix the kinase classification to exclude ABC transporters and ATPases
    """
    print("\n" + "="*60)
    print("FIXING KINASE CLASSIFICATION")
    print("="*60)
    
    # Load the protein data
    protein_df = pd.read_csv(OUTPUT_DIR / 'gene_protein_mapping_enhanced.csv')
    
    # Reset kinase classification
    protein_df['is_kinase'] = False
    protein_df['is_true_kinase'] = False
    
    # ABC transporter and ATPase patterns to EXCLUDE
    exclude_patterns = [
        'ABC',     # ABC transporters (ABCA, ABCB, ABCC, etc.)
        'ATP1',    # ATPases (ATP1A, ATP1B, etc.)
        'ATP2',    # More ATPases
        'ATP5',    # ATP synthase subunits
        'ATP6',    # More ATP synthase
        'ATP7',    # Copper-transporting ATPases
        'ATP8',    # More ATPases
        'ATP9',    # More ATPases
        'ATP10',   # Phospholipid-transporting ATPases
        'ATP11',   # More phospholipid ATPases
        'ATP12',   # More ATPases
        'ATP13',   # More ATPases
        'SLC27',   # Fatty acid transporters (not kinases)
        'SLC',     # Solute carriers in general
        'ATPIF',   # ATP synthase inhibitor
        'ATPAF'    # ATP synthase assembly factors
    ]
    
    # True kinase patterns - must have these terms
    true_kinase_terms = [
        'kinase',
        'phosphorylation'
    ]
    
    # Additional kinase family identifiers
    kinase_families = [
        'CDK',     # Cyclin-dependent kinases
        'MAPK',    # MAP kinases
        'JAK',     # Janus kinases
        'SRC',     # Src family kinases
        'ABL',     # Abelson kinases
        'ALK',     # Anaplastic lymphoma kinase
        'AKT',     # AKT/PKB kinases
        'PKA',     # Protein kinase A
        'PKC',     # Protein kinase C
        'TYK',     # Tyrosine kinases
        'FLT',     # FMS-like tyrosine kinase
        'KIT',     # KIT proto-oncogene
        'MET',     # MET proto-oncogene
        'RET',     # RET proto-oncogene
        'EGFR',    # EGF receptor
        'ERBB',    # ERBB family
        'FGFR',    # FGF receptors
        'PDGFR',   # PDGF receptors
        'VEGFR',   # VEGF receptors
        'KDR',     # Kinase insert domain receptor
        'IGF1R',   # IGF-1 receptor
        'INSR',    # Insulin receptor
        'EPH',     # Ephrin receptors
        'TRK',     # Tropomyosin receptor kinases
        'BTK',     # Bruton's tyrosine kinase
        'ITK',     # IL2-inducible T-cell kinase
        'LCK',     # Lymphocyte-specific kinase
        'LYN',     # LYN proto-oncogene
        'SYK',     # Spleen tyrosine kinase
        'ZAP70',   # Zeta-chain-associated kinase
        'CSF1R',   # Colony stimulating factor 1 receptor
        'ROS1',    # ROS proto-oncogene 1
        'AXL',     # AXL receptor tyrosine kinase
        'MERTK',   # MER proto-oncogene
        'DDR',     # Discoidin domain receptors
        'ROR',     # Receptor tyrosine kinase-like orphan receptors
        'PTK',     # Protein tyrosine kinases
        'STK',     # Serine/threonine kinases
        'CAMK',    # Calcium/calmodulin-dependent kinases
        'DAPK',    # Death-associated protein kinases
        'ROCK',    # Rho-associated kinases
        'MLK',     # Mixed lineage kinases
        'PAK',     # p21-activated kinases
        'RAF',     # RAF proto-oncogene
        'BRAF',    # B-Raf proto-oncogene
        'MEK',     # MAP kinase kinase (MAP2K)
        'ERK',     # Extracellular signal-regulated kinases
        'RSK',     # Ribosomal S6 kinases
        'MSK',     # Mitogen- and stress-activated kinases
        'AMPK',    # AMP-activated protein kinase
        'MARK',    # MAP/microtubule affinity-regulating kinases
        'CHK',     # Checkpoint kinases
        'PLK',     # Polo-like kinases
        'AURK',    # Aurora kinases
        'NEK',     # NIMA-related kinases
        'WEE',     # WEE kinases
        'TTBK',    # Tau tubulin kinases
        'NUAK',    # NUAK family kinases
        'MELK',    # Maternal embryonic leucine zipper kinase
        'PIM',     # PIM proto-oncogene kinases
        'DYRK',    # Dual-specificity tyrosine kinases
        'CLK',     # CDC-like kinases
        'HIPK',    # Homeodomain interacting protein kinases
        'IRAK',    # Interleukin-1 receptor-associated kinases
        'RIPK',    # Receptor-interacting protein kinases
        'IKK',     # IκB kinases
        'TBK',     # TANK-binding kinases
        'NIK',     # NF-κB-inducing kinase (MAP3K14)
        'ASK',     # Apoptosis signal-regulating kinases
        'TAK',     # TGF-β activated kinases
        'MEKK',    # MEK kinases
        'MKK',     # MAP kinase kinases
        'MAPKAPK', # MAP kinase-activated protein kinases
        'MNK',     # MAP kinase-interacting kinases
        'BMPR',    # Bone morphogenetic protein receptors
        'TGFBR',   # TGF-β receptors
        'ACVR',    # Activin receptors
        'LTK'      # Leukocyte tyrosine kinase
    ]
    
    corrected_kinases = []
    false_kinases = []
    
    for idx, row in protein_df.iterrows():
        gene = row['gene']
        is_kinase = False
        
        # First check if it's an excluded pattern (ABC transporter, ATPase, etc.)
        is_excluded = any(gene.startswith(pattern) for pattern in exclude_patterns)
        
        if is_excluded:
            # This is NOT a kinase, even if it has ATP-binding
            false_kinases.append(gene)
            continue
        
        # Check if gene name contains kinase family identifier
        is_kinase_family = any(family in gene.upper() for family in kinase_families)
        
        if is_kinase_family:
            is_kinase = True
            corrected_kinases.append(gene)
        else:
            # Check keywords for true kinase terms (not just ATP-binding)
            if 'keywords' in row and pd.notna(row['keywords']):
                keywords = str(row['keywords']).lower()
                
                # Must have kinase or phosphorylation, not just ATP-binding
                if any(term in keywords for term in true_kinase_terms):
                    is_kinase = True
                    corrected_kinases.append(gene)
            
            # Check protein name
            if not is_kinase and 'protein_name' in row and pd.notna(row['protein_name']):
                protein_name = str(row['protein_name']).lower()
                if 'kinase' in protein_name:
                    is_kinase = True
                    corrected_kinases.append(gene)
        
        protein_df.at[idx, 'is_kinase'] = is_kinase
        protein_df.at[idx, 'is_true_kinase'] = is_kinase
    
    print(f"\n✓ Corrected kinase classification:")
    print(f"  • True kinases identified: {len(corrected_kinases)}")
    print(f"  • False positives removed: {len(false_kinases)}")
    
    if false_kinases:
        print(f"\n  Removed non-kinases (ABC transporters/ATPases):")
        for gene in sorted(set(false_kinases))[:10]:  # Show first 10
            print(f"    - {gene}")
        if len(false_kinases) > 10:
            print(f"    ... and {len(false_kinases)-10} more")
    
    # Save corrected data
    protein_df.to_csv(OUTPUT_DIR / 'gene_protein_mapping_enhanced_fixed.csv', index=False)
    
    return protein_df, corrected_kinases, false_kinases

def identify_true_surface_kinases(protein_df=None):
    """
    Get ACTUAL Surface ∩ Kinase intersection with corrected classification
    """
    print("\n" + "="*60)
    print("TRUE SURFACE KINASE INTERSECTION (CORRECTED)")
    print("="*60)
    
    if protein_df is None:
        # Load the fixed protein data
        protein_df = pd.read_csv(OUTPUT_DIR / 'gene_protein_mapping_enhanced_fixed.csv')
    
    # Get Surface ∩ True Kinase intersection
    true_surface_kinases = protein_df[
        (protein_df['is_surface'] == True) & 
        (protein_df['is_kinase'] == True)
    ].copy()
    
    print(f"\n✓ Found {len(true_surface_kinases)} TRUE surface kinases")
    
    # Sort by druggability score
    true_surface_kinases = true_surface_kinases.sort_values('druggability_score', ascending=False)
    
    # Identify receptor tyrosine kinases (RTKs)
    rtk_patterns = ['EGFR', 'ERBB', 'FGFR', 'PDGFR', 'VEGFR', 'KDR', 'FLT', 
                    'KIT', 'MET', 'RET', 'ROS1', 'ALK', 'AXL', 'MERTK', 
                    'DDR', 'EPH', 'TRK', 'ROR', 'CSF1R', 'IGF1R', 'INSR',
                    'TIE', 'TEK', 'NTRK', 'LTK', 'MUSK', 'PTK', 'RYK']
    
    # Create detailed analysis
    surface_kinase_analysis = []
    
    for _, row in true_surface_kinases.iterrows():
        gene = row['gene']
        
        # Check if it's an RTK
        is_rtk = any(pattern in gene.upper() for pattern in rtk_patterns)
        
        # Check if it has receptor in keywords
        if not is_rtk and 'keywords' in row and pd.notna(row['keywords']):
            keywords = str(row['keywords']).lower()
            if 'receptor' in keywords:
                is_rtk = True
        
        surface_kinase_analysis.append({
            'Gene': gene,
            'Protein_Name': row.get('protein_name', ''),
            'Druggability_Score': row['druggability_score'],
            'Is_RTK': is_rtk,
            'Is_Priority': row.get('is_priority', False),
            'UniProt_ID': row.get('uniprot_id', '')
        })
    
    sk_df = pd.DataFrame(surface_kinase_analysis)
    
    # Display results
    print("\nTop 20 TRUE Surface Kinases:")
    print("-" * 90)
    print(f"{'Rank':<5} {'Gene':<12} {'Score':<7} {'RTK':<5} {'Priority':<9} {'Protein Name':<40}")
    print("-" * 90)
    
    for i, (_, row) in enumerate(sk_df.head(20).iterrows(), 1):
        rtk = '✓' if row['Is_RTK'] else ''
        priority = '★★★' if row['Is_Priority'] else ''
        protein_name = row['Protein_Name'][:40] if pd.notna(row['Protein_Name']) else ''
        print(f"{i:<5} {row['Gene']:<12} {row['Druggability_Score']:<7.0f} "
              f"{rtk:<5} {priority:<9} {protein_name:<40}")
    
    # Save results
    sk_df.to_csv(OUTPUT_DIR / 'true_surface_kinases.csv', index=False)
    
    # Summary statistics
    print(f"\n📊 Surface Kinase Summary:")
    print(f"  • Total TRUE surface kinases: {len(sk_df)}")
    print(f"  • Receptor Tyrosine Kinases (RTKs): {sk_df['Is_RTK'].sum()}")
    print(f"  • Non-receptor kinases: {(~sk_df['Is_RTK']).sum()}")
    print(f"  • Priority targets (FLT3, etc.): {sk_df['Is_Priority'].sum()}")
    
    # Visualize
    create_corrected_visualization(sk_df)
    
    return sk_df

def create_corrected_visualization(df):
    """Create visualization for corrected surface kinases"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Top targets bar chart
    ax1 = axes[0, 0]
    top_10 = df.head(10)
    colors = ['red' if p else ('orange' if r else 'steelblue') 
              for p, r in zip(top_10['Is_Priority'], top_10['Is_RTK'])]
    
    ax1.barh(range(len(top_10)), top_10['Druggability_Score'], color=colors)
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10['Gene'])
    ax1.set_xlabel('Druggability Score')
    ax1.set_title('Top 10 TRUE Surface Kinases')
    ax1.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Priority Target'),
        Patch(facecolor='orange', label='RTK'),
        Patch(facecolor='steelblue', label='Other Kinase')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # 2. RTK vs Non-RTK distribution
    ax2 = axes[0, 1]
    rtk_counts = df['Is_RTK'].value_counts()
    colors = ['#FF9999', '#9999FF']
    wedges, texts, autotexts = ax2.pie(
        rtk_counts.values, 
        labels=['Non-RTK Kinases', 'RTKs'],
        colors=colors,
        autopct='%1.0f%%',
        startangle=90
    )
    ax2.set_title('RTK Distribution in TRUE Surface Kinases')
    
    # 3. Score distribution
    ax3 = axes[1, 0]
    ax3.hist(df['Druggability_Score'], bins=15, color='teal', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Druggability Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Druggability Score Distribution')
    ax3.axvline(df['Druggability_Score'].median(), color='red', 
                linestyle='--', label=f'Median: {df["Druggability_Score"].median():.0f}')
    ax3.legend()
    
    # 4. RTK vs Non-RTK scores comparison
    ax4 = axes[1, 1]
    rtk_scores = df[df['Is_RTK']]['Druggability_Score']
    non_rtk_scores = df[~df['Is_RTK']]['Druggability_Score']
    
    bp = ax4.boxplot([non_rtk_scores, rtk_scores], 
                      labels=['Non-RTK', 'RTK'],
                      patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('Druggability Score')
    ax4.set_title('Druggability Comparison: RTK vs Non-RTK')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('TRUE Surface Kinase Analysis (ABC Transporters/ATPases Excluded)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'true_surface_kinases_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualization to true_surface_kinases_analysis.png")

def main():
    """Run the corrected analysis"""
    
    print("\n" + "="*70)
    print("CORRECTING KINASE CLASSIFICATION AND SURFACE KINASE ANALYSIS")
    print("="*70)
    
    # 1. Fix the kinase classification
    protein_df, true_kinases, false_kinases = fix_kinase_classification()
    
    # 2. Get true surface kinases
    true_surface_kinases = identify_true_surface_kinases(protein_df)
    
    # 3. Generate final recommendations
    print("\n" + "="*60)
    print("FINAL RECOMMENDATIONS FOR VALIDATION")
    print("="*60)
    
    print("\n🎯 HIGH-PRIORITY TRUE SURFACE KINASES FOR FLOW/BLOCKING:")
    
    # Get top RTKs
    top_rtks = true_surface_kinases[true_surface_kinases['Is_RTK']].head(5)
    if not top_rtks.empty:
        print("\n  Receptor Tyrosine Kinases:")
        for i, (_, row) in enumerate(top_rtks.iterrows(), 1):
            print(f"    {i}. {row['Gene']} (Score: {row['Druggability_Score']:.0f})")
    
    # Get top non-RTK kinases
    top_non_rtks = true_surface_kinases[~true_surface_kinases['Is_RTK']].head(5)
    if not top_non_rtks.empty:
        print("\n  Non-Receptor Surface Kinases:")
        for i, (_, row) in enumerate(top_non_rtks.iterrows(), 1):
            print(f"    {i}. {row['Gene']} (Score: {row['Druggability_Score']:.0f})")
    
    print("\n✓ Analysis complete with corrected kinase classification!")
    print(f"✓ Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
