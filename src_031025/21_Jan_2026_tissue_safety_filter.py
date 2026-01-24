#!/usr/bin/env python3
"""
FINAL PRODUCTION TISSUE SAFETY FILTER - ALL TWEAKS
===================================================
TWEAK 1: Deterministic alias (prefer gene_upper → primary → sorted)
TWEAK 2: Namespace-safe XML parsing
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import gzip
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

CD_PATTERN = re.compile(r"^CD\d{1,3}[A-Z0-9]*$", re.IGNORECASE)

def is_cd_surface_marker(gene: str) -> bool:
    return bool(CD_PATTERN.match(gene))

class FinalProduction_TissueFilter:
    
    def __init__(self, gene_list_file, membrane_excel_file, hpa_xml_file, gtex_file, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("PRODUCTION TISSUE SAFETY FILTER - ALL TWEAKS")
        print("="*70)
        
        with open(gene_list_file) as f:
            self.gene_list_1000 = [line.strip().upper() for line in f if line.strip()]
        print(f"✓ {len(self.gene_list_1000)} persister genes")
        
        self.membrane_db = pd.read_excel(membrane_excel_file)
        print(f"✓ {len(self.membrane_db)} membrane entries")
        
        self.create_gene_mappings()
        self.membrane_persisters = self.identify_membrane_genes()
        print(f"✓ {len(self.membrane_persisters)} membrane persisters")
        
        self.hpa_xml_file = Path(hpa_xml_file)
        self.gtex_file = Path(gtex_file)
        self._init_params()
    
    def create_gene_mappings(self):
        print("\nCreating gene mappings...")
        
        self.gene_to_aliases = {}
        self.alias_to_gene = {}
        self.ambiguous_aliases = set()
        
        for _, row in self.membrane_db.iterrows():
            primary_genes = []
            all_aliases = []
            
            for col in ['GSEA', 'ProteinAtlas', 'Gene', 'Gene name', 'HGNC symbol', 'Symbol']:
                if col in self.membrane_db.columns and pd.notna(row.get(col)):
                    genes = re.split(r'[,;|/\s]+', str(row[col]))
                    primary_genes.extend([g.strip().upper() for g in genes if g.strip()])
            
            for col in ['Synonyms', 'Alias', 'Aliases']:
                if col in self.membrane_db.columns and pd.notna(row.get(col)):
                    syns = re.split(r'[,;|/\s]+', str(row[col]))
                    all_aliases.extend([s.strip().upper() for s in syns if s.strip()])
            
            if primary_genes:
                primary = primary_genes[0]
                existing = self.gene_to_aliases.get(primary, set())
                existing |= set(primary_genes) | set(all_aliases) | {primary}
                self.gene_to_aliases[primary] = existing
                
                for alias in existing:
                    if alias in self.alias_to_gene and self.alias_to_gene[alias] != primary:
                        self.ambiguous_aliases.add(alias)
                    else:
                        self.alias_to_gene[alias] = primary
        
        print(f"✓ {len(self.gene_to_aliases)} genes, {len(self.ambiguous_aliases)} ambiguous")
    
    def identify_membrane_genes(self):
        membrane_genes = set()
        for col in ['GSEA', 'ProteinAtlas', 'Gene', 'HGNC symbol', 'Symbol']:
            if col in self.membrane_db.columns:
                for entry in self.membrane_db[col].dropna():
                    genes = re.split(r'[,;|/\s]+', str(entry))
                    membrane_genes.update([g.strip().upper() for g in genes if g.strip()])
        
        membrane_persisters = []
        seen = set()
        
        for gene in self.gene_list_1000:
            gene = gene.strip().upper()
            primary = self.alias_to_gene.get(gene, gene)
            
            if (gene in membrane_genes) or (primary in membrane_genes):
                if primary not in seen:
                    membrane_persisters.append(primary)
                    seen.add(primary)
        
        return membrane_persisters
    
    def _init_params(self):
        self.critical_organs = {
            'heart': ['heart', 'myocardium', 'cardiac'],
            'liver': ['liver', 'hepatocyte'],
            'kidney': ['kidney', 'glomeruli', 'tubules', 'renal'],
            'lung': ['lung', 'alveolar'],
            'pancreas': ['pancreas']
        }
        
        self.gi_organs = {
            'stomach': ['stomach', 'gastric'],
            'small_intestine': ['small intestine', 'duodenum', 'ileum'],
            'colon': ['colon', 'rectum']
        }
        
        self.brain_organs = {
            'brain': ['brain', 'cerebral', 'cerebell', 'hippocamp', 'caudate']
        }
        
        self.hema_tissues = {'bone marrow', 'lymph node', 'spleen', 'appendix', 'tonsil'}
        self.testis_keywords = ['testis', 'testes']
        self.known_good_aml = {'FLT3', 'CD33', 'CD44', 'CD123', 'IL3RA'}
        self.known_ubiquitous = {'CDH1', 'CTNNB1', 'CAV1', 'CAV2', 'EGFR', 'MET',
                                 'CDC42EP1', 'CDC42EP4', 'CDC42BPA', 'CDK7', 'TJP1', 'JUP'}
    
    def parse_hpa(self):
        """
        TWEAK 2: Namespace-safe XML parsing
        """
        print("\n" + "="*70)
        print("PARSING HPA (NAMESPACE-SAFE)")
        print("="*70)
        
        target_symbols = set()
        for gene in self.membrane_persisters:
            target_symbols.add(gene)
            if gene in self.gene_to_aliases:
                target_symbols.update(self.gene_to_aliases[gene])
            target_symbols.add(self.alias_to_gene.get(gene, gene))
        
        hpa_expression = {}
        genes_found = 0
        
        with gzip.open(self.hpa_xml_file, 'rt', encoding='utf-8') as f:
            for event, elem in ET.iterparse(f, events=('end',)):
                # TWEAK 2: Use .endswith() for namespace safety
                if elem.tag.endswith('entry'):
                    # TWEAK 2: Use .//{*} for namespace-agnostic search
                    gene_elem = elem.find('.//{*}name')
                    if gene_elem is None:
                        elem.clear()
                        continue
                    
                    gene_name = gene_elem.text.strip().upper()
                    if gene_name not in target_symbols:
                        elem.clear()
                        continue
                    
                    genes_found += 1
                    organs_high = {'critical': set(), 'gi': set(), 'brain': set(), 'hema': set()}
                    
                    # TWEAK 2: Namespace-safe search
                    tissue_expr_section = elem.find('.//{*}tissueExpression')
                    if tissue_expr_section is not None:
                        for data_elem in tissue_expr_section.findall('.//{*}data'):
                            tissue_elem = data_elem.find('.//{*}tissue')
                            level_elem = data_elem.find('.//{*}level[@type="expression"]')
                            
                            if tissue_elem is not None and tissue_elem.text and \
                               level_elem is not None and level_elem.text:
                                tissue = tissue_elem.text.lower()
                                level = level_elem.text.strip().lower()
                                
                                if any(t in tissue for t in self.testis_keywords):
                                    continue
                                
                                if 'high' in level:
                                    matched = False
                                    for organ_name, keywords in self.critical_organs.items():
                                        if any(kw in tissue for kw in keywords):
                                            organs_high['critical'].add(organ_name)
                                            matched = True
                                            break
                                    
                                    if not matched:
                                        for organ_name, keywords in self.gi_organs.items():
                                            if any(kw in tissue for kw in keywords):
                                                organs_high['gi'].add(organ_name)
                                                matched = True
                                                break
                                    
                                    if not matched:
                                        is_brain = any(kw in tissue for kw in self.brain_organs['brain'])
                                        if 'cortex' in tissue:
                                            if any(x in tissue for x in ['brain', 'cerebral']):
                                                is_brain = True
                                            elif any(x in tissue for x in ['kidney', 'adrenal']):
                                                is_brain = False
                                        if is_brain:
                                            organs_high['brain'].add('brain')
                                
                                if 'high' in level or 'medium' in level:
                                    if any(h in tissue for h in self.hema_tissues):
                                        organs_high['hema'].add('hematopoietic')
                    
                    hpa_expression[gene_name] = organs_high
                    
                    if genes_found % 50 == 0:
                        print(f"  {genes_found}...")
                    
                    elem.clear()
        
        print(f"✓ {len(hpa_expression)} genes")
        return hpa_expression
    
    def parse_gtex(self):
        print("\n" + "="*70)
        print("PARSING GTEx")
        print("="*70)
        
        with gzip.open(self.gtex_file, 'rt') as f:
            f.readline()
            f.readline()
            gtex_data = pd.read_csv(f, sep='\t', index_col=1)
            if 'Name' in gtex_data.columns:
                gtex_data = gtex_data.drop('Name', axis=1)
        
        gtex_data = gtex_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        gtex_data.index = [str(g).strip().upper() for g in gtex_data.index]
        gtex_data = gtex_data.groupby(gtex_data.index).max()
        
        target_symbols = set()
        for gene in self.membrane_persisters:
            target_symbols.add(gene)
            if gene in self.gene_to_aliases:
                target_symbols.update(self.gene_to_aliases[gene])
            target_symbols.add(self.alias_to_gene.get(gene, gene))
        
        gtex_filtered = gtex_data[gtex_data.index.isin(target_symbols)]
        print(f"✓ {len(gtex_filtered)} genes")
        
        return gtex_filtered
    
    def rank(self, hpa_dict, gtex_df):
        """
        TWEAK 1: Correct deterministic alias selection (prefer gene_upper → primary → sorted)
        """
        print("\n" + "="*70)
        print("RANKING (DETERMINISTIC: gene_upper → primary → sorted)")
        print("="*70)
        
        results = []
        hpa_genes = set(hpa_dict.keys())
        gtex_genes = set(gtex_df.index)
        
        for gene in self.membrane_persisters:
            gene_upper = gene.upper()
            
            # Build alias set
            aliases = set([gene_upper])
            primary = self.alias_to_gene.get(gene_upper, gene_upper)
            aliases.add(primary)
            aliases |= set(self.gene_to_aliases.get(primary, set()))
            aliases |= set(self.gene_to_aliases.get(gene_upper, set()))
            
            # TWEAK 1: Deterministic ordering (prefer gene_upper → primary → sorted rest)
            aliases = aliases - self.ambiguous_aliases
            
            ordered_aliases = [gene_upper]
            if primary != gene_upper:
                ordered_aliases.append(primary)
            ordered_aliases += sorted(aliases - {gene_upper, primary})
            
            # TWEAK 1: Use next() for first match in ordered list
            hpa_match = next((a for a in ordered_aliases if a in hpa_genes), None)
            gtex_match = next((a for a in ordered_aliases if a in gtex_genes), None)
            
            penalties = {
                'pen_critical_hpa': 0, 'pen_gi_hpa': 0, 'pen_brain_hpa': 0,
                'pen_critical_gtex': 0, 'pen_gi_gtex': 0, 'pen_brain_gtex': 0,
                'pen_missing': 0, 'pen_ubiquitous': 0
            }
            
            bonuses = {
                'bonus_hema': 0, 'bonus_specificity': 0, 'bonus_known_good': 0
            }
            
            is_cd_marker = is_cd_surface_marker(gene_upper)
            modality = 'antibody' if is_cd_marker else ('small_molecule' if gene_upper in {'FLT3', 'KIT', 'JAK1', 'EGFR'} else 'unknown')
            bbb_exempt = modality in {'antibody', 'adc', 'bispecific'}
            known_good_flag = gene_upper in self.known_good_aml
            
            has_hpa = hpa_match is not None
            has_gtex = gtex_match is not None
            
            if not has_hpa and not has_gtex:
                penalties['pen_missing'] = 20
                missing_data = True
            else:
                missing_data = False
            
            critical_organs_hit = set()
            gi_organs_hit = set()
            brain_organs_hit = set()
            hema_detected = False
            
            if has_hpa:
                organs = hpa_dict[hpa_match]
                critical_organs_hit = organs['critical']
                gi_organs_hit = organs['gi']
                brain_organs_hit = organs['brain']
                hema_detected = len(organs['hema']) > 0
                
                penalties['pen_critical_hpa'] = len(critical_organs_hit) * 40
                penalties['pen_gi_hpa'] = len(gi_organs_hit) * 20
                penalties['pen_brain_hpa'] = len(brain_organs_hit) * (5 if bbb_exempt else 25)
            
            gtex_max_critical = 0
            gtex_max_gi = 0
            gtex_max_brain = 0
            gtex_max_hema = 0
            gtex_mean_non_hema = 0
            
            if has_gtex:
                gene_expr = gtex_df.loc[gtex_match]
                if isinstance(gene_expr, pd.DataFrame):
                    gene_expr = gene_expr.max(axis=0)
                
                crit, gi, brain, hema, all_nh = [], [], [], [], []
                
                for tc, val in gene_expr.items():
                    tl = tc.lower()
                    if any(t in tl for t in self.testis_keywords):
                        continue
                    
                    if any(c in tl for c in ['heart', 'liver', 'kidney', 'lung', 'pancreas']):
                        crit.append(val)
                        all_nh.append(val)
                    elif any(g in tl for g in ['stomach', 'intestine', 'colon', 'esophagus']):
                        gi.append(val)
                        all_nh.append(val)
                    elif any(b in tl for b in ['brain', 'cerebellum', 'cerebral', 'hippocampus']):
                        brain.append(val)
                        all_nh.append(val)
                    elif 'cortex' in tl:
                        if any(x in tl for x in ['brain', 'cerebell']):
                            brain.append(val)
                        all_nh.append(val)
                    elif any(h in tl for h in ['blood', 'spleen']):
                        hema.append(val)
                    else:
                        all_nh.append(val)
                
                gtex_max_critical = max(crit) if crit else 0
                gtex_max_gi = max(gi) if gi else 0
                gtex_max_brain = max(brain) if brain else 0
                gtex_max_hema = max(hema) if hema else 0
                gtex_mean_non_hema = np.mean(all_nh) if all_nh else 0
                
                if gtex_max_critical > 200:
                    penalties['pen_critical_gtex'] = 50
                elif gtex_max_critical > 100:
                    penalties['pen_critical_gtex'] = 35
                elif gtex_max_critical > 50:
                    penalties['pen_critical_gtex'] = 20
                
                if gtex_max_gi > 100:
                    penalties['pen_gi_gtex'] = 25
                elif gtex_max_gi > 50:
                    penalties['pen_gi_gtex'] = 15
                
                if not bbb_exempt:
                    if gtex_max_brain > 50:
                        penalties['pen_brain_gtex'] = 30
                    elif gtex_max_brain > 20:
                        penalties['pen_brain_gtex'] = 15
            
            if hema_detected:
                bonuses['bonus_hema'] = -20
            if gtex_max_hema > 50:
                bonuses['bonus_hema'] += -20
            
            if gtex_mean_non_hema > 0:
                spec_ratio = gtex_max_hema / (gtex_mean_non_hema + 1)
                if spec_ratio > 10:
                    bonuses['bonus_specificity'] = -30
                elif spec_ratio > 5:
                    bonuses['bonus_specificity'] = -20
                elif spec_ratio > 2:
                    bonuses['bonus_specificity'] = -10
            else:
                spec_ratio = 0
            
            if known_good_flag:
                bonuses['bonus_known_good'] = -25
            
            if gene_upper in self.known_ubiquitous:
                penalties['pen_ubiquitous'] = 50
            
            penalty_total = sum(penalties.values())
            bonus_total = sum(bonuses.values())
            
            safety_risk = penalty_total
            priority_score = penalty_total + bonus_total
            
            results.append({
                'gene': gene,
                'hpa_symbol': hpa_match,
                'gtex_symbol': gtex_match,
                'aliases_used': '|'.join(ordered_aliases[:5]),  # TWEAK 1: Show ordered aliases
                'safety_risk': safety_risk,
                'priority_score': priority_score,
                'penalty_total': penalty_total,
                'bonus_total': bonus_total,
                'is_cd_marker': is_cd_marker,
                'therapeutic_modality': modality,
                'bbb_exempt': bbb_exempt,
                'known_good_aml': known_good_flag,
                'missing_data': missing_data,
                'hpa_critical_organs': len(critical_organs_hit),
                'hpa_gi_organs': len(gi_organs_hit),
                'critical_organs_list': '|'.join(sorted(critical_organs_hit)),
                'gi_organs_list': '|'.join(sorted(gi_organs_hit)),
                'gtex_max_critical': gtex_max_critical,
                'gtex_max_gi': gtex_max_gi,
                'gtex_max_hema': gtex_max_hema,
                'specificity_ratio': spec_ratio,
                **penalties,
                **bonuses
            })
        
        df = pd.DataFrame(results)
        print(f"✓ {len(df)} genes")
        
        return df
    
    def select_top_n(self, safety_df, target_count=40):
        print(f"\n🎯 TOP-N SELECTION")
        print("="*70)
        
        EXCLUDE_CUTOFF = 70
        HIGH_RISK_CUTOFF = 45
        
        def hard_cat(r):
            if r >= EXCLUDE_CUTOFF:
                return 'EXCLUDE'
            elif r >= HIGH_RISK_CUTOFF:
                return 'HIGH_RISK'
            else:
                return 'OK'
        
        safety_df['hard_category'] = safety_df['safety_risk'].apply(hard_cat)
        
        pool = safety_df[(safety_df['hard_category'] == 'OK') & (~safety_df['missing_data'])].copy()
        print(f"  Pool: {len(pool)}")
        
        actual_target = min(target_count, len(pool))
        
        pool_sorted = pool.sort_values(
            ['priority_score', 'safety_risk', 'bonus_total', 'specificity_ratio', 'gtex_max_hema', 'gene'],
            ascending=[True, True, True, False, False, True],
            kind='mergesort'
        )
        
        top_n = pool_sorted.head(actual_target).copy()
        top_n_sorted = top_n.sort_values(
            ['safety_risk', 'priority_score', 'gene'],
            ascending=[True, True, True],
            kind='mergesort'
        )
        
        cut = int(actual_target * 0.4)
        safe_indices = top_n_sorted.index[:cut]
        check_indices = top_n_sorted.index[cut:]
        
        safety_df['final_category'] = 'CAUTION'
        safety_df.loc[safe_indices, 'final_category'] = 'SAFE'
        safety_df.loc[check_indices, 'final_category'] = 'CHECK'
        safety_df.loc[safety_df['hard_category'] == 'HIGH_RISK', 'final_category'] = 'HIGH_RISK'
        safety_df.loc[safety_df['hard_category'] == 'EXCLUDE', 'final_category'] = 'EXCLUDE'
        
        print(f"✅ SAFE: {(safety_df['final_category']=='SAFE').sum()}, CHECK: {(safety_df['final_category']=='CHECK').sum()}")
        
        return safety_df
    
    def report(self, final_df):
        safe_check = final_df[final_df['final_category'].isin(['SAFE', 'CHECK'])].copy()
        
        cat_rank = safe_check['final_category'].map({'SAFE': 0, 'CHECK': 1}).fillna(2)
        safe_check_sorted = (
            safe_check.assign(_cat_rank=cat_rank)
                      .sort_values(['_cat_rank', 'priority_score', 'safety_risk', 'gene'], 
                                   ascending=[True, True, True, True],
                                   kind='mergesort')
                      .drop(columns=['_cat_rank'])
        )
        
        print("\n" + "="*70)
        print(f"1000 → {len(final_df)} membrane → {len(safe_check_sorted)} tissue-safe")
        print("="*70)
        
        print("\nTOP 50:")
        for idx, (_, row) in enumerate(safe_check_sorted.head(50).iterrows(), start=1):
            print(f"{idx:3d}. {row['gene']:12s} | {row['final_category']:6s} | Safety={row['safety_risk']:3.0f}, Priority={row['priority_score']:4.0f}")
        
        safe_check_sorted.to_csv(self.output_dir / 'FINAL_SAFE_CHECK_SORTED.csv', index=False)
        final_df.to_csv(self.output_dir / 'ALL_RANKED.csv', index=False)
        
        print(f"\n✓ Saved to {self.output_dir}")
        
        return final_df
    
    def run(self):
        hpa_dict = self.parse_hpa()
        gtex_df = self.parse_gtex()
        safety_df = self.rank(hpa_dict, gtex_df)
        safety_df = self.select_top_n(safety_df, target_count=40)
        final = self.report(safety_df)
        
        print("\n✅ COMPLETE - ALL TWEAKS APPLIED")
        return final

# RUN
analyzer = FinalProduction_TissueFilter(
    gene_list_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt',
    membrane_excel_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/PM proteins.xlsx',
    hpa_xml_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/tissue_filtered_real_data/proteinatlas.xml.gz',
    gtex_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/tissue_filtered_real_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz',
    output_dir='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/integrated_1000gene_analysis'
)

results = analyzer.run()
