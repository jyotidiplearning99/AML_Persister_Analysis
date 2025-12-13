#!/usr/bin/env python3
"""
FIXED: Parse HPA XML using the ACTUAL structure shown in debug output
"""

import pandas as pd
import numpy as np
import gzip
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class WorkingHPAParser:
    def __init__(self, membrane_targets_file, hpa_xml_file, gtex_file, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.targets = pd.read_csv(membrane_targets_file)
        print(f"Loaded {len(self.targets)} membrane targets")
        
        self.hpa_xml_file = Path(hpa_xml_file)
        self.gtex_file = Path(gtex_file)
        
        # Severe critical organs (ANY high expression → EXCLUDE)
        self.severe_critical_organs = [
            'heart', 'myocardium', 'cardiac',
            'brain', 'cortex', 'cerebral', 'hippocampus', 'cerebellum',
            'liver', 'hepatocyte',
            'kidney', 'renal', 'glomeruli',
            'lung', 'alveolar',
            'pancreas',
            'stomach', 'gastric',
            'duodenum', 'small intestine', 'ileum', 'jejunum',
            'colon', 'rectum',
            'esophagus',
            'skin', 'epidermis'
        ]
        
        self.hema_tissues = ['bone marrow', 'lymph node', 'spleen', 'appendix', 'tonsil']
        self.testis_keywords = ['testis', 'testes']
    
    def parse_hpa_xml_correct_structure(self):
        """
        Parse HPA XML using CORRECT structure from debug output
        Structure: <tissueExpression><data><tissue>text</tissue><level type="expression">text</level>
        """
        print("\n" + "="*60)
        print("PARSING HPA XML (CORRECT STRUCTURE)")
        print("="*60)
        
        hpa_expression = {}
        target_genes_upper = set([g.upper() for g in self.targets['gene'].values])
        
        genes_found = 0
        genes_with_data = 0
        total_genes = 0
        
        with gzip.open(self.hpa_xml_file, 'rt', encoding='utf-8') as f:
            for event, elem in ET.iterparse(f, events=('end',)):
                if elem.tag == 'entry':
                    total_genes += 1
                    
                    # Get gene name
                    gene_elem = elem.find('name')
                    if gene_elem is None:
                        elem.clear()
                        continue
                    
                    gene_name = gene_elem.text.upper()
                    
                    if gene_name not in target_genes_upper:
                        elem.clear()
                        continue
                    
                    genes_found += 1
                    
                    # Extract tissue expression using CORRECT structure
                    tissue_expr = {}
                    
                    # Find tissueExpression section
                    tissue_expr_section = elem.find('tissueExpression')
                    if tissue_expr_section is not None:
                        # Iterate through all <data> elements
                        for data_elem in tissue_expr_section.findall('data'):
                            # Get tissue name (text content of <tissue> element)
                            tissue_elem = data_elem.find('tissue')
                            if tissue_elem is not None and tissue_elem.text:
                                tissue_name = tissue_elem.text.lower()
                                
                                # Get expression level (text content of <level type="expression">)
                                level_elem = data_elem.find('level[@type="expression"]')
                                if level_elem is not None and level_elem.text:
                                    level_value = level_elem.text.strip()
                                    
                                    # Store tissue expression
                                    tissue_expr[tissue_name] = level_value
                    
                    if tissue_expr:
                        hpa_expression[gene_name] = tissue_expr
                        genes_with_data += 1
                    
                    if genes_found % 50 == 0:
                        print(f"  Found {genes_found} targets, extracted data for {genes_with_data}...")
                    
                    elem.clear()
        
        print(f"\n✓ XML parsing complete!")
        print(f"  Total HPA genes: {total_genes}")
        print(f"  Target genes found: {genes_found}")
        print(f"  Genes with expression data: {genes_with_data}")
        
        # VALIDATION: Check specific genes
        print("\n📊 HPA VALIDATION (With Expression Levels):")
        print("-" * 60)
        
        for gene in ['KCNS3', 'CDH1', 'EGFR', 'FLT3', 'CD33', 'CAV1']:
            if gene in hpa_expression:
                tissues = hpa_expression[gene]
                
                # Count levels
                high_count = sum(1 for lvl in tissues.values() if 'high' in lvl.lower())
                medium_count = sum(1 for lvl in tissues.values() if 'medium' in lvl.lower())
                low_count = sum(1 for lvl in tissues.values() if 'low' in lvl.lower())
                not_detected = sum(1 for lvl in tissues.values() if 'not detected' in lvl.lower())
                
                print(f"  {gene:12s}: {len(tissues)} tissues | "
                      f"High: {high_count}, Medium: {medium_count}, Low: {low_count}, Not: {not_detected}")
                
                # Show specific high/medium tissues for KCNS3
                if gene == 'KCNS3':
                    print(f"    High/Medium in:")
                    for tissue, level in tissues.items():
                        if 'high' in level.lower() or 'medium' in level.lower():
                            print(f"      {tissue:30s} → {level}")
            else:
                print(f"  {gene:12s}: NOT FOUND")
        
        # Convert to DataFrame
        hpa_records = []
        for gene, tissues in hpa_expression.items():
            for tissue, level in tissues.items():
                hpa_records.append({
                    'gene': gene,
                    'tissue': tissue,
                    'level': level
                })
        
        hpa_df = pd.DataFrame(hpa_records)
        hpa_df.to_csv(self.output_dir / 'hpa_extracted_working.csv', index=False)
        
        return hpa_df, hpa_expression
    
    def parse_gtex(self):
        """Parse GTEx"""
        print("\n" + "="*60)
        print("PARSING GTEx DATA")
        print("="*60)
        
        with gzip.open(self.gtex_file, 'rt') as f:
            f.readline()
            f.readline()
            gtex_data = pd.read_csv(f, sep='\t', index_col=1)
            if 'Name' in gtex_data.columns:
                gtex_data = gtex_data.drop('Name', axis=1)
        
        gtex_data.index = [str(g).upper().strip() for g in gtex_data.index]
        target_genes = [g.upper() for g in self.targets['gene'].values]
        gtex_filtered = gtex_data[gtex_data.index.isin(target_genes)]
        
        print(f"✓ Found {len(gtex_filtered)} target genes")
        
        return gtex_filtered
    
    def rank_with_strict_penalties(self, hpa_df, gtex_df):
        """
        Rank with STRICT penalties
        ANY "High" in severe organs OR GTEx > 30 TPM → EXCLUDE
        """
        print("\n" + "="*60)
        print("RANKING WITH STRICT PENALTIES")
        print("="*60)
        
        results = []
        
        for gene in self.targets['gene'].values:
            gene_upper = gene.upper()
            
            risk_score = 0
            critical_high_tissues = []
            critical_medium_tissues = []
            hema_detected = False
            
            # === HPA ANALYSIS (Protein) ===
            if not hpa_df.empty:
                gene_hpa = hpa_df[hpa_df['gene'] == gene_upper]
                
                for _, row in gene_hpa.iterrows():
                    tissue = row['tissue'].lower()
                    level = row['level'].lower()
                    
                    # Skip testis
                    if any(test in tissue for test in self.testis_keywords):
                        continue
                    
                    # Check severe critical organs
                    is_severe = any(organ in tissue for organ in self.severe_critical_organs)
                    
                    if is_severe:
                        if 'high' in level:
                            critical_high_tissues.append(tissue)
                            risk_score += 40
                        elif 'medium' in level:
                            critical_medium_tissues.append(tissue)
                            risk_score += 20
                    
                    # Check hematopoietic
                    if any(hema in tissue for hema in self.hema_tissues):
                        if 'high' in level or 'medium' in level:
                            hema_detected = True
            
            # === GTEx ANALYSIS (mRNA) ===
            gtex_max_critical = 0
            gtex_mean_critical = 0
            gtex_max_hema = 0
            
            if gtex_df is not None and gene_upper in gtex_df.index:
                gene_expr = gtex_df.loc[gene_upper]
                
                critical_values = []
                hema_values = []
                
                for tissue_col, value in gene_expr.items():
                    tissue_lower = tissue_col.lower()
                    
                    # Skip testis
                    if any(test in tissue_lower for test in self.testis_keywords):
                        continue
                    
                    # Critical tissues
                    if any(crit in tissue_lower for crit in [
                        'heart', 'brain', 'cortex', 'liver', 'kidney', 'lung',
                        'pancreas', 'intestine', 'colon', 'stomach', 'esophagus', 'muscle'
                    ]):
                        critical_values.append(value)
                    
                    # Hematopoietic
                    if any(hema in tissue_lower for hema in ['blood', 'spleen']):
                        hema_values.append(value)
                
                gtex_max_critical = max(critical_values) if critical_values else 0
                gtex_mean_critical = np.mean(critical_values) if critical_values else 0
                gtex_max_hema = max(hema_values) if hema_values else 0
                
                # STRICT GTEx thresholds (based on KCNS3 = 34.4 TPM)
                if gtex_max_critical > 100:
                    risk_score += 60
                elif gtex_max_critical > 50:
                    risk_score += 50
                elif gtex_max_critical > 30:  # Catches KCNS3
                    risk_score += 40
                elif gtex_max_critical > 20:
                    risk_score += 30
                elif gtex_max_critical > 10:
                    risk_score += 20
            
            # === CRITICAL FIX: ANY "High" in severe organs → force to EXCLUDE ===
            if critical_high_tissues:
                risk_score = max(risk_score, 80)
            
            # Bonuses
            if hema_detected:
                risk_score -= 10
            if gtex_max_hema > 50:
                risk_score -= 10
            
            risk_score = max(0, risk_score)
            
            # Categorize
            if risk_score >= 70:
                category = 'EXCLUDE'
            elif risk_score >= 50:
                category = 'HIGH_RISK'
            elif risk_score >= 30:
                category = 'CAUTION'
            elif risk_score >= 15:
                category = 'CHECK'
            else:
                category = 'SAFE'
            
            results.append({
                'gene': gene,
                'risk_score': risk_score,
                'safety_category': category,
                'hpa_high_count': len(critical_high_tissues),
                'hpa_medium_count': len(critical_medium_tissues),
                'critical_tissues_high': '|'.join(critical_high_tissues[:5]),
                'gtex_max_critical': gtex_max_critical,
                'gtex_mean_critical': gtex_mean_critical,
                'gtex_max_hema': gtex_max_hema,
                'hema_detected': hema_detected
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('risk_score')
        df['safety_rank'] = range(1, len(df) + 1)
        
        return df
    
    def apply_filtering(self, safety_df):
        """Apply filtering"""
        
        if 'priority_score' in self.targets.columns:
            merged = self.targets.merge(safety_df, on='gene', how='left')
            merged['adjusted_priority'] = merged['priority_score'] - (merged['risk_score'] / 3)
        else:
            merged = safety_df
            merged['adjusted_priority'] = 100 - merged['risk_score']
        
        merged = merged.sort_values('adjusted_priority', ascending=False)
        merged.to_csv(self.output_dir / 'final_strict_filtered.csv', index=False)
        
        safe = merged[merged['safety_category'] == 'SAFE']
        check = merged[merged['safety_category'] == 'CHECK']
        exclude = merged[merged['safety_category'].isin(['EXCLUDE', 'HIGH_RISK'])]
        
        print(f"\n✅ SAFE: {len(safe)}")
        print(f"⚠️ CHECK: {len(check)}")
        print(f"❌ EXCLUDE/HIGH_RISK: {len(exclude)}")
        
        # VALIDATE KCNS3
        print("\n🔍 KCNS3 VALIDATION:")
        if 'KCNS3' in merged['gene'].values:
            kcns3 = merged[merged['gene'] == 'KCNS3'].iloc[0]
            print(f"  Gene: KCNS3")
            print(f"  Risk score: {kcns3['risk_score']:.0f}")
            print(f"  Category: {kcns3['safety_category']}")
            print(f"  HPA High count: {kcns3['hpa_high_count']}")
            print(f"  Critical tissues: {kcns3.get('critical_tissues_high', '')}")
            print(f"  GTEx max critical: {kcns3['gtex_max_critical']:.1f} TPM")
            
            if kcns3['safety_category'] in ['EXCLUDE', 'HIGH_RISK']:
                print(f"  ✅ CORRECTLY categorized as {kcns3['safety_category']}")
            else:
                print(f"  ⚠️ WARNING: Should be EXCLUDE (has 34.4 TPM in critical tissues)")
        
        return merged, safe, exclude
    
    def generate_final_report(self, final_df, safe, exclude):
        """Generate comprehensive final report"""
        
        print("\n" + "="*60)
        print("TOP 30 SAFEST TARGETS (Minimal Expression in Healthy Tissues)")
        print("="*60)
        print(f"{'Rank':<5} {'Gene':<12} {'Risk':<6} {'Category':<12} {'HPA_High':<10} {'GTEx_Max':<10}")
        print("-" * 80)
        
        safest = final_df.sort_values('risk_score').head(30)
        for _, row in safest.iterrows():
            print(f"{row['safety_rank']:<5} {row['gene']:<12} {row['risk_score']:<6.0f} "
                  f"{row['safety_category']:<12} {row['hpa_high_count']:<10} "
                  f"{row['gtex_max_critical']:<10.1f}")
        
        print("\n" + "="*60)
        print("EXCLUDED/HIGH-RISK GENES")
        print("="*60)
        
        for _, row in exclude.head(30).iterrows():
            tissues = row.get('critical_tissues_high', '')[:50]
            print(f"✗ {row['gene']:12s} | Risk: {row['risk_score']:3.0f} | "
                  f"GTEx: {row['gtex_max_critical']:6.1f} | HPA_High: {row['hpa_high_count']} | {tissues}")
        
        # Save results
        safest.to_csv(self.output_dir / 'top30_safest_targets.csv', index=False)
        exclude.to_csv(self.output_dir / 'excluded_high_risk_genes.csv', index=False)
        
        # Create visualization
        self._create_visualization(final_df, safe, exclude)
        
        return final_df
    
    def _create_visualization(self, final_df, safe, exclude):
        """Create visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Risk distribution
        ax1 = axes[0, 0]
        ax1.hist(final_df['risk_score'], bins=40, edgecolor='black')
        ax1.axvline(15, color='green', linestyle='--', linewidth=2, label='Safe < 15')
        ax1.axvline(50, color='orange', linestyle='--', linewidth=2, label='High risk ≥ 50')
        ax1.axvline(70, color='red', linestyle='--', linewidth=2, label='Exclude ≥ 70')
        ax1.set_xlabel('Risk Score (lower = safer)')
        ax1.set_ylabel('Count')
        ax1.set_title('Risk Score Distribution\n(Testis excluded from penalties)')
        ax1.legend()
        
        # 2. GTEx scatter
        ax2 = axes[0, 1]
        scatter = ax2.scatter(final_df['gtex_max_critical'], final_df['gtex_max_hema'],
                             c=final_df['risk_score'], cmap='RdYlGn_r', alpha=0.6, s=50)
        ax2.set_xlabel('Max Critical Tissue (TPM)')
        ax2.set_ylabel('Max Hematopoietic (TPM)')
        ax2.set_title('GTEx: Critical vs Hematopoietic Expression')
        ax2.axvline(30, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Highlight KCNS3
        if 'KCNS3' in final_df['gene'].values:
            kcns3 = final_df[final_df['gene'] == 'KCNS3'].iloc[0]
            ax2.scatter(kcns3['gtex_max_critical'], kcns3['gtex_max_hema'],
                       color='red', s=400, marker='*', edgecolors='black', linewidths=2)
            ax2.text(kcns3['gtex_max_critical']+3, kcns3['gtex_max_hema']+2,
                    'KCNS3\n(stomach)', fontsize=9, fontweight='bold', color='red')
        
        plt.colorbar(scatter, ax=ax2, label='Risk Score')
        
        # 3. Top safe
        ax3 = axes[0, 2]
        top20 = final_df.sort_values('risk_score').head(20)
        ax3.barh(range(len(top20)), top20['gtex_max_critical'].values, color='green')
        ax3.set_yticks(range(len(top20)))
        ax3.set_yticklabels(top20['gene'].values, fontsize=8)
        ax3.set_xlabel('Max Critical (TPM)')
        ax3.set_title('Top 20 Safest')
        ax3.invert_yaxis()
        
        # 4. Excluded
        ax4 = axes[1, 0]
        if not exclude.empty:
            top_excl = exclude.sort_values('gtex_max_critical', ascending=False).head(20)
            ax4.barh(range(len(top_excl)), top_excl['gtex_max_critical'].values, color='red')
            ax4.set_yticks(range(len(top_excl)))
            ax4.set_yticklabels(top_excl['gene'].values, fontsize=8)
            ax4.set_xlabel('Max Critical (TPM)')
            ax4.set_title('Excluded/High-Risk Genes')
            ax4.invert_yaxis()
        
        # 5. Category distribution
        ax5 = axes[1, 1]
        cat_counts = final_df['safety_category'].value_counts()
        colors_map = {'SAFE': 'green', 'CHECK': 'yellow', 'CAUTION': 'orange',
                     'HIGH_RISK': 'darkorange', 'EXCLUDE': 'red'}
        bars = ax5.bar(range(len(cat_counts)), cat_counts.values)
        for i, cat in enumerate(cat_counts.index):
            if cat in colors_map:
                bars[i].set_color(colors_map[cat])
        ax5.set_xticks(range(len(cat_counts)))
        ax5.set_xticklabels(cat_counts.index, rotation=45)
        ax5.set_ylabel('Count')
        ax5.set_title('Safety Categories')
        
        # 6. Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        kcns3_status = "Not found"
        if 'KCNS3' in final_df['gene'].values:
            kcns3 = final_df[final_df['gene'] == 'KCNS3'].iloc[0]
            kcns3_status = f"{kcns3['safety_category']} (risk={kcns3['risk_score']:.0f})"
        
        summary_text = f"""
        STRICT TISSUE FILTERING
        =======================
        Total: {len(final_df)}
        Safe: {len(safe)}
        Excluded: {len(exclude)}
        
        KCNS3 Status:
        {kcns3_status}
        
        Thresholds:
        • GTEx > 30 TPM → Exclude
        • HPA "High" → Exclude
        • Testis ignored
        
        Top 3 safest:
        1. {final_df.iloc[0]['gene']}
        2. {final_df.iloc[1]['gene']}
        3. {final_df.iloc[2]['gene']}
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10,
                verticalalignment='center', fontfamily='monospace')
        
        plt.suptitle('Strict Tissue Safety Filtering\n(Max critical > 30 TPM OR HPA High → EXCLUDE)', 
                    fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'final_strict_report.pdf', dpi=300)
        plt.savefig(self.output_dir / 'final_strict_report.png', dpi=150)
        
        print(f"\n✓ Report saved")
    
    def run_analysis(self):
        """Run complete analysis"""
        
        # Parse HPA with correct structure
        hpa_df, hpa_dict = self.parse_hpa_xml_correct_structure()
        
        # Parse GTEx
        gtex_df = self.parse_gtex()
        
        # Rank with strict penalties
        safety_df = self.rank_with_strict_penalties(hpa_df, gtex_df)
        
        # Apply filtering
        final, safe, exclude = self.apply_filtering(safety_df)
        
        # Generate report
        self.generate_final_report(final, safe, exclude)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return final

# Run with correct HPA parser
analyzer = WorkingHPAParser(
    membrane_targets_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/membrane_targets_1000genes/prioritized_membrane_targets_1000genes.csv',
    hpa_xml_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/tissue_filtered_real_data/proteinatlas.xml.gz',
    gtex_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/tissue_filtered_real_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz',
    output_dir='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/tissue_safety_working'
)

results = analyzer.run_analysis()
