#!/usr/bin/env python3
"""
Cross-check 1000 persister genes with plasma membrane proteins
Identify targetable surface proteins for AML persister therapy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

class PersisterMembraneTargetAnalyzer:
    def __init__(self, persister_genes_file, membrane_excel_file, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load 1000-gene persister panel
        print("="*60)
        print("LOADING 1000-GENE PERSISTER PANEL")
        print("="*60)
        
        with open(persister_genes_file) as f:
            self.persister_genes = [line.strip().upper() for line in f if line.strip()]
        print(f"✓ Loaded {len(self.persister_genes)} persister genes")
        
        # Load membrane proteins from Excel
        print("\nLoading plasma membrane protein database...")
        self.membrane_df = pd.read_excel(membrane_excel_file)
        print(f"✓ Loaded {len(self.membrane_df)} membrane protein entries")
        
        # Process all membrane protein identifiers
        self.process_membrane_database()
        
        # Setup targetability categories
        self.setup_drug_target_categories()
        
    def process_membrane_database(self):
        """Process and unify membrane protein identifiers"""
        
        self.membrane_genes = set()
        self.membrane_mapping = {}  # Maps synonyms to primary names
        
        # Process each column
        for col in ['GSEA', 'ProteinAtlas', 'Synonyms']:
            if col in self.membrane_df.columns:
                for entry in self.membrane_df[col].dropna():
                    # Handle multiple genes separated by various delimiters
                    genes = re.split('[,;|/]', str(entry))
                    for gene in genes:
                        gene_clean = gene.strip().upper()
                        if gene_clean and gene_clean != 'NAN':
                            self.membrane_genes.add(gene_clean)
                            
                            # Create synonym mapping
                            if col == 'Synonyms':
                                row_idx = self.membrane_df[self.membrane_df[col] == entry].index
                                if len(row_idx) > 0:
                                    primary_gsea = self.membrane_df.loc[row_idx[0], 'GSEA']
                                    primary_atlas = self.membrane_df.loc[row_idx[0], 'ProteinAtlas']
                                    if pd.notna(primary_gsea):
                                        self.membrane_mapping[gene_clean] = primary_gsea.upper()
                                    elif pd.notna(primary_atlas):
                                        self.membrane_mapping[gene_clean] = primary_atlas.upper()
        
        print(f"✓ Processed {len(self.membrane_genes)} unique membrane protein identifiers")
    
    def setup_drug_target_categories(self):
        """Categorize targets by druggability"""
        
        # High-priority AML targets from literature
        self.known_aml_targets = {
            'CD33', 'CD123', 'IL3RA', 'CD47', 'CD38', 'CD25', 'IL2RA',
            'CD96', 'CD70', 'FLT3', 'KIT', 'IL1RAP', 'TIM3', 'HAVCR2',
            'CLEC12A', 'CLL1', 'LILRB2', 'LILRB4', 'CD44', 'CD93'
        }
        
        # JAK/STAT pathway (from meeting)
        self.jak_stat_genes = {
            'JAK1', 'JAK2', 'JAK3', 'TYK2', 
            'STAT1', 'STAT2', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B', 'STAT6'
        }
        
        # BCL2 family (important for venetoclax sensitivity)
        self.bcl2_family = {
            'BCL2', 'MCL1', 'BCL2L1', 'BCL2A1', 'BAX', 'BAK1', 'BID'
        }
        
        # MYC-related (from meeting)
        self.myc_genes = {
            'MYC', 'MYCN', 'MYCL', 'MAX', 'MXI1', 'MXD1', 'MGA'
        }
        
        # Categories for therapeutic approach
        self.target_categories = {
            'CD_markers': [],        # Antibody targets
            'Receptors': [],         # Small molecules/antibodies
            'Transporters': [],      # Small molecules
            'Ion_channels': [],      # Small molecules
            'Enzymes': [],          # Small molecules
            'Adhesion': [],         # Antibodies
            'Cytokine_receptors': [],  # Antibodies/small molecules
            'Growth_factor_receptors': [],  # TKIs
            'Immune_checkpoints': []  # Immunotherapy
        }
    
    def find_persister_membrane_targets(self):
        """Identify persister genes that are membrane proteins"""
        
        print("\n" + "="*60)
        print("IDENTIFYING MEMBRANE PERSISTER TARGETS")
        print("="*60)
        
        # Find matches
        direct_matches = []
        synonym_matches = []
        partial_matches = []
        
        for gene in self.persister_genes:
            # Direct match
            if gene in self.membrane_genes:
                direct_matches.append(gene)
            # Check synonyms
            elif gene in self.membrane_mapping:
                synonym_matches.append({
                    'persister': gene,
                    'membrane': self.membrane_mapping[gene]
                })
            # Check partial matches (for gene families)
            else:
                for mem_gene in self.membrane_genes:
                    # Check if persister gene is substring of membrane gene
                    if len(gene) > 3 and (gene in mem_gene or mem_gene in gene):
                        partial_matches.append({
                            'persister': gene,
                            'membrane': mem_gene
                        })
                        break
        
        # Combine all matches
        all_membrane_persisters = list(set(direct_matches))
        for match in synonym_matches:
            if match['persister'] not in all_membrane_persisters:
                all_membrane_persisters.append(match['persister'])
        
        print(f"\n✓ Found {len(all_membrane_persisters)} membrane persister genes")
        print(f"  - Direct matches: {len(direct_matches)}")
        print(f"  - Synonym matches: {len(synonym_matches)}")
        print(f"  - This represents {100*len(all_membrane_persisters)/len(self.persister_genes):.1f}% of persister genes")
        
        # Categorize by target type
        categorized_targets = self.categorize_targets(all_membrane_persisters)
        
        # Save results
        results_df = pd.DataFrame({
            'gene': all_membrane_persisters,
            'is_membrane': True,
            'source': 'GSEA/ProteinAtlas'
        })
        
        results_df.to_csv(self.output_dir / 'membrane_persister_genes_1000.csv', index=False)
        
        return all_membrane_persisters, categorized_targets
    
    def categorize_targets(self, membrane_persisters):
        """Categorize targets by therapeutic approach"""
        
        categorized = {
            'CD_markers': [],
            'Known_AML_targets': [],
            'JAK_STAT': [],
            'BCL2_family': [],
            'MYC_related': [],
            'Receptors': [],
            'Transporters': [],
            'Ion_channels': [],
            'Enzymes': [],
            'Cytokine_receptors': [],
            'Growth_factor_receptors': [],
            'Immune_checkpoints': []
        }
        
        for gene in membrane_persisters:
            # CD markers (excellent antibody targets)
            if gene.startswith('CD'):
                categorized['CD_markers'].append(gene)
            
            # Known AML targets
            if gene in self.known_aml_targets:
                categorized['Known_AML_targets'].append(gene)
            
            # JAK/STAT pathway
            if gene in self.jak_stat_genes:
                categorized['JAK_STAT'].append(gene)
            
            # BCL2 family
            if gene in self.bcl2_family:
                categorized['BCL2_family'].append(gene)
            
            # MYC-related
            if gene in self.myc_genes:
                categorized['MYC_related'].append(gene)
            
            # Receptors
            if any(x in gene for x in ['RECEPTOR', 'GPR', 'GPCR', 'CXCR', 'CCR', 'IL', 'TNF']):
                categorized['Receptors'].append(gene)
            
            # Transporters
            if any(x in gene for x in ['SLC', 'ABC', 'ATP']):
                categorized['Transporters'].append(gene)
            
            # Ion channels
            if any(x in gene for x in ['KCN', 'CACN', 'SCN', 'CLCN']):
                categorized['Ion_channels'].append(gene)
            
            # Enzymes
            if any(x in gene for x in ['ASE', 'KINASE', 'PHOSPHATASE']):
                categorized['Enzymes'].append(gene)
            
            # Cytokine receptors
            if any(x in gene for x in ['ILR', 'TNFR', 'IFNR', 'TGFR']):
                categorized['Cytokine_receptors'].append(gene)
            
            # Growth factor receptors
            if any(x in gene for x in ['FGFR', 'EGFR', 'VEGFR', 'PDGFR', 'IGF']):
                categorized['Growth_factor_receptors'].append(gene)
            
            # Immune checkpoints
            if gene in ['PD1', 'PDL1', 'CTLA4', 'TIM3', 'LAG3', 'TIGIT']:
                categorized['Immune_checkpoints'].append(gene)
        
        return categorized
    
    def prioritize_targets(self, membrane_persisters, categorized_targets):
        """Prioritize targets for therapeutic intervention"""
        
        print("\n" + "="*60)
        print("TARGET PRIORITIZATION")
        print("="*60)
        
        priority_scores = []
        
        for gene in membrane_persisters:
            score = 0
            categories = []
            therapeutic_approaches = []
            
            # Scoring system
            # CD markers get high priority (antibody targetable)
            if gene.startswith('CD'):
                score += 10
                categories.append('CD_marker')
                therapeutic_approaches.append('Antibody')
            
            # Known AML targets get highest priority
            if gene in self.known_aml_targets:
                score += 15
                categories.append('Known_AML_target')
                therapeutic_approaches.append('Clinical_precedent')
            
            # JAK/STAT pathway (from meeting emphasis)
            if gene in self.jak_stat_genes:
                score += 12
                categories.append('JAK_STAT')
                therapeutic_approaches.append('Small_molecule')
            
            # BCL2 family (venetoclax targets)
            if gene in self.bcl2_family:
                score += 12
                categories.append('BCL2_family')
                therapeutic_approaches.append('Venetoclax_related')
            
            # MYC-related (from meeting emphasis)
            if gene in self.myc_genes:
                score += 10
                categories.append('MYC_pathway')
                therapeutic_approaches.append('Experimental')
            
            # Receptor tyrosine kinases
            if any(x in gene for x in ['FLT', 'KIT', 'FGFR', 'EGFR', 'VEGFR']):
                score += 10
                categories.append('RTK')
                therapeutic_approaches.append('TKI')
            
            # G-protein coupled receptors
            if 'GPR' in gene or 'CXCR' in gene or 'CCR' in gene:
                score += 8
                categories.append('GPCR')
                therapeutic_approaches.append('Small_molecule')
            
            # Transporters
            if gene.startswith('SLC') or gene.startswith('ABC'):
                score += 5
                categories.append('Transporter')
                therapeutic_approaches.append('Small_molecule')
            
            # Ion channels
            if any(x in gene for x in ['KCN', 'CACN', 'SCN']):
                score += 6
                categories.append('Ion_channel')
                therapeutic_approaches.append('Small_molecule')
            
            # Immune checkpoints
            if gene in ['PD1', 'PDL1', 'CTLA4', 'TIM3', 'LAG3', 'TIGIT']:
                score += 14
                categories.append('Immune_checkpoint')
                therapeutic_approaches.append('Immunotherapy')
            
            priority_scores.append({
                'gene': gene,
                'priority_score': score,
                'categories': '|'.join(categories) if categories else 'Membrane',
                'therapeutic_approach': '|'.join(therapeutic_approaches) if therapeutic_approaches else 'TBD',
                'is_cd_marker': gene.startswith('CD'),
                'is_known_target': gene in self.known_aml_targets,
                'is_jak_stat': gene in self.jak_stat_genes,
                'is_bcl2_family': gene in self.bcl2_family,
                'is_myc_related': gene in self.myc_genes
            })
        
        df_priority = pd.DataFrame(priority_scores)
        df_priority = df_priority.sort_values('priority_score', ascending=False)
        
        # Save prioritized list
        df_priority.to_csv(self.output_dir / 'prioritized_membrane_targets_1000genes.csv', index=False)
        
        # Print category summary
        print("\nTargets by category:")
        for category, genes in categorized_targets.items():
            if genes:
                print(f"  {category}: {len(genes)} genes")
        
        return df_priority
    
    def generate_top30_wetlab_targets(self, df_priority):
        """Generate top 30 genes for wet-lab validation"""
        
        print("\n" + "="*60)
        print("TOP 30 GENES FOR WET-LAB VALIDATION")
        print("="*60)
        
        top30 = df_priority.head(30)
        
        # Separate by therapeutic approach
        antibody_targets = top30[top30['therapeutic_approach'].str.contains('Antibody', na=False)]
        small_molecule = top30[top30['therapeutic_approach'].str.contains('Small_molecule|TKI', na=False)]
        known_targets = top30[top30['is_known_target']]
        
        print("\n1. ANTIBODY TARGETS (Surface proteins):")
        print("-" * 40)
        for _, row in antibody_targets.head(10).iterrows():
            print(f"  {row['gene']:15s} Score: {row['priority_score']:3d} | {row['categories']}")
        
        print("\n2. SMALL MOLECULE TARGETS:")
        print("-" * 40)
        for _, row in small_molecule.head(10).iterrows():
            print(f"  {row['gene']:15s} Score: {row['priority_score']:3d} | {row['categories']}")
        
        print("\n3. KNOWN AML TARGETS (Benchmarking):")
        print("-" * 40)
        for _, row in known_targets.head(5).iterrows():
            print(f"  {row['gene']:15s} - Use as positive control")
        
        # JAK/STAT targets (meeting emphasis)
        jak_stat_targets = df_priority[df_priority['is_jak_stat']].head(5)
        if not jak_stat_targets.empty:
            print("\n4. JAK/STAT PATHWAY (Priority from meeting):")
            print("-" * 40)
            for _, row in jak_stat_targets.iterrows():
                print(f"  {row['gene']:15s} - Small molecule screening")
        
        # BCL2 family (venetoclax related)
        bcl2_targets = df_priority[df_priority['is_bcl2_family']].head(5)
        if not bcl2_targets.empty:
            print("\n5. BCL2 FAMILY (Venetoclax sensitivity):")
            print("-" * 40)
            for _, row in bcl2_targets.iterrows():
                print(f"  {row['gene']:15s} - Combination with venetoclax")
        
        # Save top 30
        top30.to_csv(self.output_dir / 'top30_wetlab_targets_1000genes.csv', index=False)
        
        # Create validation protocol
        self.create_validation_protocol(top30)
        
        return top30
    
    def create_validation_protocol(self, top30):
        """Create detailed validation protocol for top targets"""
        
        validation_protocol = []
        
        for _, row in top30.iterrows():
            protocol = {
                'gene': row['gene'],
                'priority': row['priority_score'],
                'validation_methods': [],
                'reagents_needed': [],
                'expected_timeline': ''
            }
            
            # Determine validation methods based on target type
            if row['is_cd_marker']:
                protocol['validation_methods'].extend([
                    'Flow_cytometry',
                    'Immunofluorescence',
                    'Antibody_blocking'
                ])
                protocol['reagents_needed'].append('Commercial_antibody')
                protocol['expected_timeline'] = '2-4 weeks'
            
            if row['is_known_target']:
                protocol['validation_methods'].append('Known_inhibitor_testing')
                protocol['reagents_needed'].append('Reference_compound')
                protocol['expected_timeline'] = '1-2 weeks'
            
            if row['is_jak_stat']:
                protocol['validation_methods'].extend([
                    'JAK_inhibitor_testing',
                    'STAT_phosphorylation_assay'
                ])
                protocol['reagents_needed'].extend(['Ruxolitinib', 'pSTAT_antibodies'])
            
            # Add standard validation methods
            protocol['validation_methods'].extend([
                'CRISPR_knockout',
                'shRNA_knockdown',
                'qPCR_expression',
                'Colony_formation_assay'
            ])
            protocol['reagents_needed'].extend([
                'sgRNA_design',
                'shRNA_constructs',
                'qPCR_primers'
            ])
            
            if not protocol['expected_timeline']:
                protocol['expected_timeline'] = '4-8 weeks'
            
            validation_protocol.append(protocol)
        
        pd.DataFrame(validation_protocol).to_csv(
            self.output_dir / 'validation_protocol_1000genes.csv', index=False
        )
    
    def generate_report(self, membrane_persisters, categorized_targets, df_priority, top30):
        """Generate comprehensive report with visualizations"""
        
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Category distribution
        ax1 = axes[0, 0]
        category_counts = {k: len(v) for k, v in categorized_targets.items() if v}
        if category_counts:
            bars = ax1.bar(range(len(category_counts)), list(category_counts.values()))
            ax1.set_xticks(range(len(category_counts)))
            ax1.set_xticklabels(list(category_counts.keys()), rotation=45, ha='right')
            ax1.set_xlabel('Target Category')
            ax1.set_ylabel('Number of Genes')
            ax1.set_title('Membrane Persister Genes by Category')
            
            # Color bars by importance
            colors = []
            for cat in category_counts.keys():
                if cat in ['Known_AML_targets', 'JAK_STAT', 'BCL2_family']:
                    colors.append('red')
                elif cat in ['CD_markers', 'Immune_checkpoints']:
                    colors.append('orange')
                else:
                    colors.append('skyblue')
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # 2. Priority score distribution
        ax2 = axes[0, 1]
        ax2.hist(df_priority['priority_score'], bins=20, edgecolor='black', color='skyblue')
        ax2.axvline(df_priority['priority_score'].quantile(0.9), color='red', 
                   linestyle='--', label='Top 10%', linewidth=2)
        ax2.set_xlabel('Priority Score')
        ax2.set_ylabel('Number of Genes')
        ax2.set_title('Distribution of Target Priority Scores')
        ax2.legend()
        
        # 3. Top 15 targets
        ax3 = axes[0, 2]
        top15 = df_priority.head(15)
        colors = ['red' if x else 'blue' for x in top15['is_known_target']]
        bars = ax3.barh(range(len(top15)), top15['priority_score'].values, color=colors)
        ax3.set_yticks(range(len(top15)))
        ax3.set_yticklabels(top15['gene'].values, fontsize=9)
        ax3.set_xlabel('Priority Score')
        ax3.set_title('Top 15 Priority Targets\n(Red = Known AML targets)')
        ax3.invert_yaxis()
        
        # 4. Therapeutic approach breakdown
        ax4 = axes[1, 0]
        approach_counts = {}
        for approaches in df_priority['therapeutic_approach']:
            for approach in approaches.split('|'):
                if approach and approach != 'TBD':
                    approach_counts[approach] = approach_counts.get(approach, 0) + 1
        
        if approach_counts:
            wedges, texts, autotexts = ax4.pie(approach_counts.values(), 
                                               labels=approach_counts.keys(),
                                               autopct='%1.1f%%',
                                               startangle=90)
            ax4.set_title('Therapeutic Approaches Available')
        
        # 5. Known vs Novel targets
        ax5 = axes[1, 1]
        known_counts = df_priority['is_known_target'].value_counts()
        colors = ['green', 'orange']
        bars = ax5.bar(['Novel Targets', 'Known AML Targets'], 
                      [known_counts.get(False, 0), known_counts.get(True, 0)],
                      color=colors)
        ax5.set_ylabel('Number of Genes')
        ax5.set_title('Known vs Novel Targets')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate percentages
        pct_membrane = 100 * len(membrane_persisters) / len(self.persister_genes)
        
        summary_text = f"""
        SUMMARY STATISTICS (1000-Gene Panel)
        =====================================
        Total persister genes: {len(self.persister_genes)}
        Membrane persisters: {len(membrane_persisters)}
        Percentage membrane: {pct_membrane:.1f}%
        
        High Priority Targets:
        • CD markers: {len([g for g in membrane_persisters if g.startswith('CD')])}
        • Known AML targets: {df_priority['is_known_target'].sum()}
        • JAK/STAT pathway: {df_priority['is_jak_stat'].sum()}
        • BCL2 family: {df_priority['is_bcl2_family'].sum()}
        • MYC-related: {df_priority['is_myc_related'].sum()}
        • Priority score ≥ 10: {(df_priority['priority_score'] >= 10).sum()}
        
        Therapeutic Potential:
        • Antibody targetable: {len([g for g in membrane_persisters if g.startswith('CD')])}
        • Small molecule: {df_priority['therapeutic_approach'].str.contains('Small_molecule').sum()}
        • TKI targetable: {df_priority['therapeutic_approach'].str.contains('TKI').sum()}
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
                fontfamily='monospace')
        
        plt.suptitle('1000-Gene Persister Panel: Membrane Target Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'membrane_target_report_1000genes.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'membrane_target_report_1000genes.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Report saved to {self.output_dir}")
        
        # Generate final recommendations
        self.generate_final_recommendations(df_priority, top30)
    
    def generate_final_recommendations(self, df_priority, top30):
        """Generate actionable recommendations"""
        
        with open(self.output_dir / 'RECOMMENDATIONS_1000genes.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("MEMBRANE PERSISTER TARGETS FROM 1000-GENE PANEL\n")
            f.write("VALIDATION RECOMMENDATIONS\n")
            f.write("="*70 + "\n\n")
            
            f.write("IMMEDIATE ACTION ITEMS:\n")
            f.write("-"*40 + "\n\n")
            
            # High priority CD markers
            cd_markers = top30[top30['is_cd_marker']].head(5)
            if not cd_markers.empty:
                f.write("1. CD MARKERS FOR ANTIBODY DEVELOPMENT:\n")
                for _, row in cd_markers.iterrows():
                    f.write(f"   • {row['gene']:12s} - Check commercial antibody availability\n")
                f.write("\n")
            
            # Known targets for benchmarking
            known = top30[top30['is_known_target']].head(5)
            if not known.empty:
                f.write("2. KNOWN TARGETS FOR BENCHMARKING:\n")
                for _, row in known.iterrows():
                    f.write(f"   • {row['gene']:12s} - Use as positive control\n")
                f.write("\n")
            
            # JAK/STAT targets (from meeting emphasis)
            jak_stat = df_priority[df_priority['is_jak_stat']].head(5)
            if not jak_stat.empty:
                f.write("3. JAK/STAT PATHWAY TARGETS (Meeting priority):\n")
                for _, row in jak_stat.iterrows():
                    f.write(f"   • {row['gene']:12s} - Test with ruxolitinib/JAK inhibitors\n")
                f.write("\n")
            
            # BCL2 family (venetoclax combination)
            bcl2 = df_priority[df_priority['is_bcl2_family']].head(5)
            if not bcl2.empty:
                f.write("4. BCL2 FAMILY (Venetoclax combination):\n")
                for _, row in bcl2.iterrows():
                    f.write(f"   • {row['gene']:12s} - Test combination with venetoclax\n")
                f.write("\n")
            
            # Novel high-score targets
            novel = top30[~top30['is_known_target']].head(10)
            f.write("5. NOVEL HIGH-PRIORITY TARGETS:\n")
            for i, (_, row) in enumerate(novel.iterrows(), 1):
                f.write(f"   {i:2d}. {row['gene']:12s} (Score: {row['priority_score']})\n")
            f.write("\n")
            
            f.write("\nVALIDATION WORKFLOW:\n")
            f.write("-"*40 + "\n")
            f.write("Phase 1 (Weeks 1-2): Expression validation\n")
            f.write("  • qPCR in AML patient samples vs healthy BM\n")
            f.write("  • Flow cytometry for CD markers\n")
            f.write("  • Western blot for key proteins\n\n")
            
            f.write("Phase 2 (Weeks 3-6): Functional validation\n")
            f.write("  • CRISPR/Cas9 knockout in AML cell lines\n")
            f.write("  • Colony formation assays\n")
            f.write("  • Apoptosis assays\n\n")
            
            f.write("Phase 3 (Weeks 6-8): Therapeutic validation\n")
            f.write("  • Test available inhibitors/antibodies\n")
            f.write("  • Combination with standard chemotherapy\n")
            f.write("  • Combination with venetoclax\n\n")
            
            f.write("INTEGRATION WITH CLINICAL DATA:\n")
            f.write("-"*40 + "\n")
            f.write("• Correlate expression with survival (TCGA/BeatAML)\n")
            f.write("• Check drug response profiles (BeatAML)\n")
            f.write("• Analyze relapsed vs diagnosis samples\n")
            f.write("• Contact Sadiq Shah for CellPhoneDB analysis\n")
        
        print("\n✓ Recommendations saved to RECOMMENDATIONS_1000genes.txt")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        
        # Find membrane persisters
        membrane_persisters, categorized_targets = self.find_persister_membrane_targets()
        
        # Prioritize targets
        df_priority = self.prioritize_targets(membrane_persisters, categorized_targets)
        
        # Generate top 30 for wet-lab
        top30 = self.generate_top30_wetlab_targets(df_priority)
        
        # Generate comprehensive report
        self.generate_report(membrane_persisters, categorized_targets, df_priority, top30)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"✓ Identified {len(membrane_persisters)} membrane targets from 1000-gene panel")
        print(f"✓ Prioritized {len(top30)} genes for wet-lab validation")
        print(f"✓ Results saved to: {self.output_dir}")
        
        return membrane_persisters, df_priority, top30

# Run the analysis
analyzer = PersisterMembraneTargetAnalyzer(
    persister_genes_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt',
    membrane_excel_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/PM proteins.xlsx',
    output_dir='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/membrane_targets_1000genes'
)

membrane_persisters, df_priority, top30 = analyzer.run_complete_analysis()

# Print final summary
print("\nTOP 10 IMMEDIATE TARGETS FOR VALIDATION:")
print("-" * 50)
for i, (_, row) in enumerate(top30.head(10).iterrows(), 1):
    approaches = row['therapeutic_approach'].replace('|', ', ')
    print(f"{i:2d}. {row['gene']:12s} - Score: {row['priority_score']:3d} - {approaches}")
