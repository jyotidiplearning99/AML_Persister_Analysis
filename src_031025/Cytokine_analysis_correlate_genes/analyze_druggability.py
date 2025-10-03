#!/usr/bin/env python3
"""
Persister Model Gene Annotation Pipeline - FIXED VERSION
Identifies druggable surface receptors and cytokines from your 1000-gene model
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: INTEGRATE WITH CELLPHONEDB
# ============================================================================

def annotate_with_cellphonedb(gene_list, output_dir):
    """
    Annotate model genes using CellPhoneDB v5 database
    """
    print("="*80)
    print("CELLPHONEDB ANNOTATION")
    print("="*80)
    
    # Try to download CellPhoneDB database files
    try:
        cpdb_url = "https://raw.githubusercontent.com/ventolab/cellphonedb-data/master/data/"
        
        # Gene input file
        genes_url = cpdb_url + "gene_input.csv"
        protein_url = cpdb_url + "protein_input.csv"
        complex_url = cpdb_url + "complex_input.csv"
        interaction_url = cpdb_url + "interaction_input.csv"
        
        # Load databases
        print("Loading CellPhoneDB databases...")
        genes_db = pd.read_csv(genes_url)
        proteins_db = pd.read_csv(protein_url)
        complexes_db = pd.read_csv(complex_url)
        interactions_db = pd.read_csv(interaction_url)
        
    except Exception as e:
        print(f"Warning: Could not load CellPhoneDB online. Using local annotations.")
        # Use fallback annotations for common genes
        genes_db, proteins_db, complexes_db, interactions_db = load_local_annotations()
    
    # Convert your gene list to uppercase for matching
    model_genes_upper = [str(g).upper() for g in gene_list]
    
    # Find matches in CellPhoneDB
    if 'gene_name' in genes_db.columns:
        matched_genes = genes_db[genes_db['gene_name'].str.upper().isin(model_genes_upper)]
        if 'uniprot' in matched_genes.columns and 'uniprot' in proteins_db.columns:
            matched_proteins = proteins_db[proteins_db['uniprot'].isin(matched_genes['uniprot'])]
        else:
            matched_proteins = pd.DataFrame()
    else:
        matched_genes = pd.DataFrame()
        matched_proteins = pd.DataFrame()
    
    # Categorize by protein type using manual annotations for common AML genes
    categories = categorize_aml_genes(model_genes_upper)
    
    # Create annotation dataframe
    annotations = []
    for gene in gene_list:
        gene_upper = str(gene).upper()
        annotation = {
            'gene': gene,
            'is_receptor': gene_upper in categories['receptor'],
            'is_ligand': gene_upper in categories['ligand'],
            'is_secreted': gene_upper in categories['secreted'],
            'is_membrane': gene_upper in categories['membrane'],
            'is_cytokine': gene_upper in categories['cytokine'],
            'is_growth_factor': gene_upper in categories['growth_factor'],
            'is_kinase': gene_upper in categories['kinase'],
            'is_transcription_factor': gene_upper in categories['transcription_factor']
        }
        
        # Druggability score
        druggability = 0
        if annotation['is_receptor']: druggability += 3
        if annotation['is_membrane']: druggability += 2
        if annotation['is_secreted']: druggability += 2
        if annotation['is_cytokine']: druggability += 1
        if annotation['is_kinase']: druggability += 3
        annotation['druggability_score'] = druggability
        
        annotations.append(annotation)
    
    df_annotations = pd.DataFrame(annotations)
    
    # Find interaction partners
    print("\nFinding interaction partners...")
    if not interactions_db.empty:
        interaction_partners = find_interaction_partners(
            gene_list, interactions_db, proteins_db
        )
    else:
        interaction_partners = pd.DataFrame()
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    df_annotations.to_csv(output_dir / 'cellphonedb_annotations.csv', index=False)
    if not interaction_partners.empty:
        interaction_partners.to_csv(output_dir / 'interaction_partners.csv', index=False)
    
    print(f"\nCellPhoneDB Summary:")
    print(f"  Receptors: {int(df_annotations['is_receptor'].sum())}")
    print(f"  Ligands: {int(df_annotations['is_ligand'].sum())}")
    print(f"  Secreted: {int(df_annotations['is_secreted'].sum())}")
    print(f"  Membrane: {int(df_annotations['is_membrane'].sum())}")
    print(f"  Cytokines: {int(df_annotations['is_cytokine'].sum())}")
    print(f"  Kinases: {int(df_annotations['is_kinase'].sum())}")
    
    return df_annotations, interaction_partners

def categorize_aml_genes(model_genes_upper):
    """Manual categorization of common AML-relevant genes"""
    
    # Define categories based on known AML biology
    categories = {
        'receptor': [
            'CD33', 'CD123', 'IL3RA', 'CD47', 'CD70', 'CD38', 'CD34',
            'FLT3', 'KIT', 'CSF1R', 'CSF3R', 'EPOR', 'MPL', 'THPO',
            'EGFR', 'ERBB2', 'ERBB3', 'ERBB4', 'MET', 'PDGFRA', 'PDGFRB',
            'FGFR1', 'FGFR2', 'FGFR3', 'VEGFR1', 'VEGFR2', 'VEGFR3',
            'TIM3', 'HAVCR2', 'PD1', 'PDCD1', 'PDL1', 'CD274', 'CTLA4',
            'CD19', 'CD20', 'MS4A1', 'CD22', 'TNFRSF8', 'CD30'
        ],
        'ligand': [
            'VEGFA', 'VEGFB', 'VEGFC', 'FGF1', 'FGF2', 'PDGFA', 'PDGFB',
            'EGF', 'TGFA', 'AREG', 'BTC', 'EPGN', 'EREG', 'HBEGF',
            'CSF1', 'CSF2', 'CSF3', 'IL3', 'TPO', 'EPO', 'SCF', 'KITLG'
        ],
        'secreted': [
            'IL1B', 'IL2', 'IL3', 'IL4', 'IL5', 'IL6', 'IL7', 'IL8',
            'CXCL8', 'IL10', 'IL12A', 'IL12B', 'IL15', 'IL17A', 'IL18',
            'TNF', 'TNFA', 'IFNG', 'IFNA1', 'IFNB1', 'TGFB1', 'TGFB2',
            'BMP2', 'BMP4', 'CXCL1', 'CXCL2', 'CXCL12', 'CCL2', 'CCL5'
        ],
        'membrane': [
            'CD33', 'CD34', 'CD38', 'CD45', 'PTPRC', 'CD47', 'CD70',
            'CD123', 'IL3RA', 'CD117', 'KIT', 'CD135', 'FLT3',
            'ABCB1', 'ABCG2', 'SLC29A1', 'SLC29A2'
        ],
        'cytokine': [
            'IL1B', 'IL2', 'IL3', 'IL4', 'IL5', 'IL6', 'IL7', 'IL8',
            'IL10', 'IL12A', 'IL12B', 'IL15', 'IL17A', 'IL18', 'TNF',
            'CSF1', 'CSF2', 'CSF3', 'IFNG', 'IFNA1', 'IFNB1'
        ],
        'growth_factor': [
            'VEGFA', 'VEGFB', 'VEGFC', 'FGF1', 'FGF2', 'PDGFA', 'PDGFB',
            'EGF', 'TGFA', 'NGF', 'BDNF', 'IGF1', 'IGF2', 'HGF'
        ],
        'kinase': [
            'FLT3', 'KIT', 'ABL1', 'ABL2', 'JAK1', 'JAK2', 'JAK3',
            'BTK', 'SYK', 'LCK', 'LYN', 'SRC', 'YES1', 'FYN',
            'MAPK1', 'MAPK3', 'MAP2K1', 'MAP2K2', 'BRAF', 'RAF1',
            'PI3K', 'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG',
            'AKT1', 'AKT2', 'AKT3', 'MTOR', 'CDK4', 'CDK6'
        ],
        'transcription_factor': [
            'RUNX1', 'CEBPA', 'PU1', 'SPI1', 'GATA1', 'GATA2',
            'MYC', 'MYCN', 'MYB', 'ETV6', 'MECOM', 'EVI1',
            'HOX', 'HOXA9', 'HOXB4', 'MEIS1', 'NPM1', 'FLT3'
        ]
    }
    
    # Convert all to uppercase for matching
    for key in categories:
        categories[key] = [g.upper() for g in categories[key]]
    
    return categories

def load_local_annotations():
    """Fallback function to create minimal annotation databases"""
    genes_db = pd.DataFrame()
    proteins_db = pd.DataFrame()
    complexes_db = pd.DataFrame()
    interactions_db = pd.DataFrame()
    return genes_db, proteins_db, complexes_db, interactions_db

def find_interaction_partners(gene_list, interactions_db, proteins_db):
    """Find known interaction partners for model genes"""
    
    if interactions_db.empty:
        return pd.DataFrame()
    
    partners = []
    gene_list_upper = [str(g).upper() for g in gene_list]
    
    for _, interaction in interactions_db.iterrows():
        partner_a = str(interaction.get('partner_a', ''))
        partner_b = str(interaction.get('partner_b', ''))
        
        # Check if any model gene is involved
        for gene in gene_list_upper:
            if gene in partner_a.upper() or gene in partner_b.upper():
                partners.append({
                    'model_gene': gene,
                    'partner_a': partner_a,
                    'partner_b': partner_b,
                    'interaction_type': interaction.get('annotation_strategy', ''),
                    'is_ligand_receptor': 'ligand-receptor' in str(interaction.get('annotation_strategy', '')).lower()
                })
    
    return pd.DataFrame(partners)

# ============================================================================
# STEP 2: DRUGGABILITY ASSESSMENT
# ============================================================================

def assess_druggability(df_annotations, expr_data=None):
    """
    Comprehensive druggability assessment
    """
    print("\n" + "="*80)
    print("DRUGGABILITY ASSESSMENT")
    print("="*80)
    
    # Known drug targets in AML
    drugbank_receptors = [
        'EGFR', 'VEGFR', 'FGFR', 'PDGFR', 'KIT', 'FLT3', 'MET', 'ALK',
        'CD33', 'CD123', 'CD47', 'CD70', 'TIM3', 'PD1', 'PDL1', 'CTLA4',
        'IL3RA', 'CSF1R', 'CSF3R', 'BCL2', 'MCL1', 'MDM2'
    ]
    
    antibody_targets = [
        'CD33', 'CD123', 'CD47', 'CD70', 'CD19', 'CD20', 'CD22', 'CD38',
        'BCMA', 'SLAMF7', 'EGFR', 'HER2', 'VEGF', 'IL6', 'TNF', 'IL1B'
    ]
    
    # Score each gene
    druggability_scores = []
    
    for _, gene_row in df_annotations.iterrows():
        gene = gene_row['gene']
        gene_upper = str(gene).upper()
        score = gene_row['druggability_score']
        
        # Additional scoring
        is_fda_target = False
        is_antibody_target = False
        
        if gene_upper in [d.upper() for d in drugbank_receptors]:
            score += 5
            is_fda_target = True
        
        if gene_upper in [a.upper() for a in antibody_targets]:
            score += 4
            is_antibody_target = True
        
        # Surface receptor bonus
        if gene_row['is_receptor'] and gene_row['is_membrane']:
            score += 3
        
        # Secreted factor bonus (can be neutralized)
        if gene_row['is_secreted']:
            score += 2
        
        # Kinase bonus
        if gene_row.get('is_kinase', False):
            score += 3
        
        druggability_scores.append({
            'gene': gene,
            'total_druggability_score': score,
            'category': categorize_druggability(score),
            'is_surface_target': gene_row['is_receptor'] or gene_row['is_membrane'],
            'is_secreted_target': gene_row['is_secreted'] or gene_row['is_cytokine'],
            'is_fda_target': is_fda_target,
            'is_antibody_target': is_antibody_target
        })
    
    df_druggability = pd.DataFrame(druggability_scores)
    
    # If expression data provided, prioritize by expression
    if expr_data is not None:
        df_druggability = prioritize_by_expression(df_druggability, expr_data)
    
    return df_druggability

def categorize_druggability(score):
    """Categorize druggability level"""
    if score >= 10:
        return 'High_Priority'
    elif score >= 5:
        return 'Medium_Priority'
    elif score >= 2:
        return 'Low_Priority'
    else:
        return 'Not_Druggable'

def prioritize_by_expression(df_druggability, expr_data):
    """Prioritize targets by expression in persister cells"""
    
    # Calculate mean expression for each gene
    mean_expr = expr_data.mean(axis=1)
    
    df_druggability['mean_expression'] = df_druggability['gene'].map(
        lambda g: mean_expr.get(g, 0)
    )
    
    # Calculate expression percentile
    df_druggability['expression_percentile'] = df_druggability['mean_expression'].rank(pct=True)
    
    # Combined priority score
    df_druggability['combined_priority'] = (
        df_druggability['total_druggability_score'] * 0.6 +
        df_druggability['expression_percentile'] * 40 * 0.4
    )
    
    return df_druggability.sort_values('combined_priority', ascending=False)

# ============================================================================
# STEP 3: THERAPEUTIC TARGET IDENTIFICATION
# ============================================================================

def identify_therapeutic_targets(df_druggability, df_annotations, interaction_partners):
    """
    Identify top therapeutic targets for experimental validation
    """
    print("\n" + "="*80)
    print("THERAPEUTIC TARGET IDENTIFICATION")
    print("="*80)
    
    # Filter for high-priority targets
    high_priority = df_druggability[
        df_druggability['category'].isin(['High_Priority', 'Medium_Priority'])
    ]
    
    # Surface targets (antibody accessible)
    surface_targets = high_priority[high_priority['is_surface_target']]
    
    # Secreted targets (can be neutralized)
    secreted_targets = high_priority[high_priority['is_secreted_target']]
    
    # Ligand-receptor pairs (can disrupt communication)
    if not interaction_partners.empty and 'is_ligand_receptor' in interaction_partners.columns:
        lr_pairs = interaction_partners[interaction_partners['is_ligand_receptor']]
    else:
        lr_pairs = pd.DataFrame()
    
    print(f"\nTherapeutic Target Summary:")
    print(f"  Total high-priority targets: {len(high_priority)}")
    print(f"  Surface receptors: {len(surface_targets)}")
    print(f"  Secreted factors: {len(secreted_targets)}")
    print(f"  Ligand-receptor pairs: {len(lr_pairs)}")
    
    # Create target report
    target_report = {
        'surface_receptors': surface_targets.head(20),
        'secreted_factors': secreted_targets.head(20),
        'ligand_receptor_pairs': lr_pairs.head(20) if not lr_pairs.empty else pd.DataFrame()
    }
    
    return target_report

# ============================================================================
# STEP 4: VISUALIZATION
# ============================================================================

def visualize_druggability(df_druggability, df_annotations, output_dir):
    """Create comprehensive druggability visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Druggability score distribution
    ax = axes[0, 0]
    ax.hist(df_druggability['total_druggability_score'], bins=20, 
            color='steelblue', edgecolor='black')
    ax.set_xlabel('Druggability Score', fontsize=12)
    ax.set_ylabel('Number of Genes', fontsize=12)
    ax.set_title('Druggability Score Distribution', fontsize=14)
    
    # 2. Category breakdown
    ax = axes[0, 1]
    category_counts = df_druggability['category'].value_counts()
    ax.bar(range(len(category_counts)), category_counts.values)
    ax.set_xticks(range(len(category_counts)))
    ax.set_xticklabels(category_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Druggability Categories', fontsize=14)
    
    # 3. Protein type distribution
    ax = axes[0, 2]
    protein_types = ['is_receptor', 'is_ligand', 'is_secreted', 
                     'is_membrane', 'is_cytokine', 'is_kinase']
    type_counts = []
    for col in protein_types:
        if col in df_annotations.columns:
            type_counts.append(int(df_annotations[col].sum()))
        else:
            type_counts.append(0)
    
    ax.bar(range(len(protein_types)), type_counts, color='coral')
    ax.set_xticks(range(len(protein_types)))
    ax.set_xticklabels(['Receptor', 'Ligand', 'Secreted', 'Membrane', 'Cytokine', 'Kinase'],
                       rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Protein Types in Model', fontsize=14)
    
    # 4. Top 20 druggable targets
    ax = axes[1, 0]
    top_targets = df_druggability.nlargest(20, 'total_druggability_score')
    if not top_targets.empty:
        ax.barh(range(len(top_targets)), top_targets['total_druggability_score'].values)
        ax.set_yticks(range(len(top_targets)))
        ax.set_yticklabels(top_targets['gene'].values, fontsize=10)
        ax.set_xlabel('Druggability Score', fontsize=12)
        ax.set_title('Top 20 Druggable Targets', fontsize=14)
        ax.invert_yaxis()
    
    # 5. Surface vs Secreted targets
    ax = axes[1, 1]
    surface_count = int(df_druggability['is_surface_target'].sum())
    secreted_count = int(df_druggability['is_secreted_target'].sum())
    both_count = int((df_druggability['is_surface_target'] & 
                     df_druggability['is_secreted_target']).sum())
    
    venn_data = [surface_count - both_count, secreted_count - both_count, both_count]
    ax.bar(['Surface Only', 'Secreted Only', 'Both'], venn_data, 
           color=['skyblue', 'lightgreen', 'gold'])
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Target Accessibility', fontsize=14)
    
    # 6. Priority distribution
    ax = axes[1, 2]
    if 'combined_priority' in df_druggability.columns:
        ax.scatter(df_druggability['total_druggability_score'],
                  df_druggability['expression_percentile'],
                  c=df_druggability['combined_priority'],
                  cmap='RdYlGn', s=20, alpha=0.6)
        ax.set_xlabel('Druggability Score', fontsize=12)
        ax.set_ylabel('Expression Percentile', fontsize=12)
        ax.set_title('Combined Priority', fontsize=14)
        plt.colorbar(ax.collections[0], ax=ax, label='Priority')
    else:
        # Show FDA and antibody targets
        fda_count = int(df_druggability['is_fda_target'].sum()) if 'is_fda_target' in df_druggability.columns else 0
        antibody_count = int(df_druggability['is_antibody_target'].sum()) if 'is_antibody_target' in df_druggability.columns else 0
        
        ax.bar(['FDA Targets', 'Antibody Targets'], [fda_count, antibody_count],
               color=['purple', 'orange'])
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Validated Drug Targets', fontsize=14)
    
    plt.suptitle('Persister Model Druggability Analysis', fontsize=16)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'druggability_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    else:
        return obj

def main():
    """
    Complete pipeline for persister gene druggability analysis
    """
    
    # Configuration
    MODEL_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled")
    OUTPUT_DIR = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/druggability_analysis")
    
    print("\n" + "="*80)
    print("PERSISTER MODEL DRUGGABILITY ANALYSIS")
    print("="*80)
    
    # Load your model genes
    gene_file = MODEL_DIR / "selected_genes.txt"
    with open(gene_file) as f:
        model_genes = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(model_genes)} model genes")
    
    # Step 1: CellPhoneDB annotation
    df_annotations, interaction_partners = annotate_with_cellphonedb(
        model_genes, OUTPUT_DIR
    )
    
    # Step 2: Druggability assessment
    df_druggability = assess_druggability(df_annotations)
    
    # Step 3: Identify therapeutic targets
    target_report = identify_therapeutic_targets(
        df_druggability, df_annotations, interaction_partners
    )
    
    # Step 4: Visualization
    fig = visualize_druggability(df_druggability, df_annotations, OUTPUT_DIR)
    
    # Save comprehensive report
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save druggability scores
    df_druggability.to_csv(OUTPUT_DIR / 'druggability_scores.csv', index=False)
    
    # Save top targets
    for category, targets in target_report.items():
        if not targets.empty:
            targets.to_csv(OUTPUT_DIR / f'top_{category}.csv', index=False)
    
    # Create summary report - FIX: Convert numpy types to native Python types
    summary = {
        'total_genes': len(model_genes),
        'druggable_genes': int(len(df_druggability[df_druggability['category'] != 'Not_Druggable'])),
        'high_priority_targets': int(len(df_druggability[df_druggability['category'] == 'High_Priority'])),
        'surface_receptors': int(df_annotations['is_receptor'].sum()),
        'secreted_factors': int(df_annotations['is_secreted'].sum()),
        'cytokines': int(df_annotations['is_cytokine'].sum()),
        'kinases': int(df_annotations['is_kinase'].sum()) if 'is_kinase' in df_annotations.columns else 0,
        'FDA_approved_targets': int(df_druggability['is_fda_target'].sum()) if 'is_fda_target' in df_druggability.columns else 0,
        'antibody_accessible': int(df_druggability['is_antibody_target'].sum()) if 'is_antibody_target' in df_druggability.columns else 0
    }
    
    # Save JSON with proper type conversion
    with open(OUTPUT_DIR / 'druggability_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=convert_numpy_types)
    
    print(f"\nDRUGGABILITY SUMMARY:")
    print(f"  Total genes analyzed: {summary['total_genes']}")
    print(f"  Druggable genes: {summary['druggable_genes']}")
    print(f"  High-priority targets: {summary['high_priority_targets']}")
    print(f"  Surface receptors: {summary['surface_receptors']}")
    print(f"  Secreted factors: {summary['secreted_factors']}")
    print(f"  Kinases: {summary['kinases']}")
    print(f"  FDA-approved targets: {summary['FDA_approved_targets']}")
    print(f"  Antibody-accessible: {summary['antibody_accessible']}")
    
    print(f"\n✓ Analysis complete! Results saved to: {OUTPUT_DIR}")
    
    # Generate experimental validation recommendations
    print("\n" + "="*80)
    print("EXPERIMENTAL VALIDATION RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. TOP ANTIBODY TARGETS (Surface receptors):")
    if not target_report['surface_receptors'].empty:
        for i, row in target_report['surface_receptors'].head(5).iterrows():
            print(f"   - {row['gene']}: Score={row['total_druggability_score']:.1f}")
    
    print("\n2. NEUTRALIZATION TARGETS (Secreted factors):")
    if not target_report['secreted_factors'].empty:
        for i, row in target_report['secreted_factors'].head(5).iterrows():
            print(f"   - {row['gene']}: Score={row['total_druggability_score']:.1f}")
    
    print("\n3. RECOMMENDED EXPERIMENTS:")
    print("   • Flow cytometry validation of surface markers")
    print("   • ELISA/Luminex for secreted factors")
    print("   • Antibody blocking experiments")
    print("   • CRISPR knockout of top targets")
    print("   • Small molecule inhibitor screening")

if __name__ == "__main__":
    main()
