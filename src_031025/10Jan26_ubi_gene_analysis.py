#!/usr/bin/env python3
"""
FINAL HARDENED PRODUCTION TISSUE SAFETY FILTER
===============================================

CRITICAL REGRESSION FIXES:
✓ Element truthiness: Use "is not None" not boolean check
✓ Ambiguous alias remap: Don't remap ambiguous aliases in rank()

ALL OTHER FIXES:
✓ No whitespace splitting (prevents garbage aliases)
✓ Coverage sanity checks
✓ Namespace-safe XML parsing
✓ Separated safety_risk from priority_score
✓ Deterministic stable sorting
✓ Properly sorted output
"""

# HPC-safe matplotlib
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
SYMBOL_LIKE = re.compile(r"^[A-Z0-9][A-Z0-9\-]{1,29}$")

def is_cd_surface_marker(gene: str) -> bool:
    return bool(CD_PATTERN.match(gene))

def _clean_tokens(s: str, split_pat=r"[,;|/]+"):
    """Clean tokens without whitespace splitting"""
    toks = [t.strip().upper() for t in re.split(split_pat, str(s)) if t and str(t).strip()]
    return [t for t in toks if SYMBOL_LIKE.match(t)]

def _child(elem, localname):
    """Namespace-safe child finder"""
    for c in elem:
        if str(c.tag).endswith(localname):
            return c
    return None

def _children(elem, localname):
    """Namespace-safe children finder"""
    return [c for c in elem if str(c.tag).endswith(localname)]


class FinalHardened_TissueFilter:
    
    def __init__(self, gene_list_file, membrane_excel_file, hpa_xml_file, gtex_file, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("FINAL HARDENED TISSUE SAFETY FILTER")
        print("="*70)
        
        with open(gene_list_file) as f:
            self.gene_list_1000 = [line.strip().upper() for line in f if line.strip()]
        print(f"✓ Loaded {len(self.gene_list_1000)} persister genes")
        
        self.membrane_db = pd.read_excel(membrane_excel_file)
        print(f"✓ Loaded {len(self.membrane_db)} membrane entries")
        
        self.create_gene_mappings()
        self.membrane_persisters = self.identify_membrane_genes()
        print(f"✓ Identified {len(self.membrane_persisters)} membrane persisters ({100*len(self.membrane_persisters)/len(self.gene_list_1000):.1f}%)")
        print(f"Ambiguous in membrane_persisters: {sum(g in self.ambiguous_aliases for g in self.membrane_persisters)}")
        self.membrane_persisters = [g for g in self.membrane_persisters if g not in self.ambiguous_aliases]
        self.hpa_xml_file = Path(hpa_xml_file)
        self.gtex_file = Path(gtex_file)
        self._init_params()
    
    def create_gene_mappings(self):
        """Create mappings WITHOUT whitespace splitting"""
        print("\nCreating gene mappings (NO whitespace split)...")
        
        self.gene_to_aliases = {}
        self.alias_to_gene = {}
        self.ambiguous_aliases = set()
        
        for _, row in self.membrane_db.iterrows():
            primary_genes = []
            all_aliases = []
            
            # NO whitespace splitting
            for col in ['GSEA', 'ProteinAtlas', 'Gene', 'Gene name', 'HGNC symbol', 'Symbol', 'gene_symbol']:
                if col in self.membrane_db.columns and pd.notna(row.get(col)):
                    genes = _clean_tokens(row[col], split_pat=r"[,;|/]+")
                    primary_genes.extend(genes)
            
            for col in ['Synonyms', 'Alias', 'Aliases', 'synonym']:
                if col in self.membrane_db.columns and pd.notna(row.get(col)):
                    syns = _clean_tokens(row[col], split_pat=r"[,;|/]+")
                    all_aliases.extend(syns)
            
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
        
        print(f"✓ Mapped {len(self.gene_to_aliases)} genes, {len(self.ambiguous_aliases)} ambiguous")
        print(f"  Total unique aliases: {len(self.alias_to_gene)}")
        

    def identify_membrane_genes(self):
        """Identify membrane genes"""
        
        membrane_genes = set()
        for col in ['GSEA', 'ProteinAtlas', 'Gene', 'Gene name', 'HGNC symbol', 'Symbol', 'gene_symbol']:
            if col in self.membrane_db.columns:
                for entry in self.membrane_db[col].dropna():
                    genes = _clean_tokens(entry, split_pat=r"[,;|/]+")
                    membrane_genes.update(genes)
        
        membrane_persisters = []
        seen = set()
        
        for gene in self.gene_list_1000:
            gene = gene.strip().upper()
            primary = gene if gene in self.ambiguous_aliases else self.alias_to_gene.get(gene, gene)
            
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
        self.known_ubiquitous = {
            'CDH1', 'CTNNB1', 'CAV1', 'CAV2', 'EGFR', 'MET',
            'CDC42EP1', 'CDC42EP4', 'CDC42BPA', 'CDK7', 'TJP1', 'JUP'
        }
        
        # Antibody targets beyond CD markers
        self.known_antibody_targets = {'IL3RA', 'CLEC12A', 'HAVCR2'}
    
    def parse_hpa(self):
        """Parse HPA with namespace-safe + element truthiness fix"""
        print("\n" + "="*70)
        print("PARSING HPA")
        print("="*70)

        target_symbols = set()
        for gene in self.membrane_persisters:
            if gene in self.gene_to_aliases:
                target_symbols.update(self.gene_to_aliases[gene])

        target_symbols -= self.ambiguous_aliases
        for gene in self.membrane_persisters:
            target_symbols.add(gene)
            if gene not in self.ambiguous_aliases:
                target_symbols.add(self.alias_to_gene.get(gene, gene))

        print(f"Searching {len(target_symbols)} symbols...")

        if len(target_symbols) > 10000:
            print(f"  ⚠️ WARNING: Unusually large target symbol set ({len(target_symbols)})")

        hpa_expression = {}
        genes_found = 0
        
        with gzip.open(self.hpa_xml_file, 'rt', encoding='utf-8') as f:
            for event, elem in ET.iterparse(f, events=('end',)):
                if str(elem.tag).endswith('entry'):
                    gene_elem = _child(elem, 'name')
                    if gene_elem is None or gene_elem.text is None:
                        elem.clear()
                        continue
                    
                    gene_name = gene_elem.text.strip().upper()
                    
                    if gene_name not in target_symbols:
                        elem.clear()
                        continue
                    
                    genes_found += 1
                    
                    organs_high = {'critical': set(), 'gi': set(), 'brain': set(), 'hema': set()}
                    
                    tissue_expr_section = _child(elem, 'tissueExpression')
                    if tissue_expr_section is not None:
                        for data_elem in _children(tissue_expr_section, 'data'):
                            tissue_elem = _child(data_elem, 'tissue')
                            
                            level_elem = None
                            for lv in _children(data_elem, 'level'):
                                if lv.attrib.get('type') == 'expression':
                                    level_elem = lv
                                    break
                            
                            # CRITICAL FIX 1: Element truthiness - use "is not None"
                            if (tissue_elem is not None and tissue_elem.text and
                                level_elem is not None and level_elem.text):
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
                        print(f"  Found {genes_found}...")
                    
                    elem.clear()
        
        print(f"✓ Extracted {len(hpa_expression)} genes")
        bad = set(hpa_expression.keys()) & self.ambiguous_aliases
        if bad:
            print("⚠️ Ambiguous symbols present in HPA matches:", list(sorted(bad))[:10])
        return hpa_expression
      
    def parse_gtex(self):
        """Parse GTEx"""
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
            if gene in self.gene_to_aliases:
                target_symbols.update(self.gene_to_aliases[gene])

        target_symbols -= self.ambiguous_aliases
        for gene in self.membrane_persisters:
            target_symbols.add(gene)
            if gene not in self.ambiguous_aliases:
                target_symbols.add(self.alias_to_gene.get(gene, gene))

        if len(target_symbols) > 10000:
            print(f"  ⚠️ WARNING: Unusually large target symbol set ({len(target_symbols)})")

        gtex_filtered = gtex_data[gtex_data.index.isin(target_symbols)]
        print(f"✓ Found {len(gtex_filtered)} genes")

        return gtex_filtered

    
    def rank(self, hpa_dict, gtex_df):
        """Calculate separated scores"""
        print("\n" + "="*70)
        print("CALCULATING SCORES")
        print("="*70)
        
        results = []
        hpa_genes = set(hpa_dict.keys())
        gtex_genes = set(gtex_df.index)
        
        for gene in self.membrane_persisters:
            gene_upper = gene.upper()
            
            # CRITICAL FIX 2: Don't remap if gene_upper is ambiguous
            primary = gene_upper if gene_upper in self.ambiguous_aliases else self.alias_to_gene.get(gene_upper, gene_upper)
            
            aliases = set([gene_upper, primary])
            aliases |= set(self.gene_to_aliases.get(primary, set()))
            aliases |= set(self.gene_to_aliases.get(gene_upper, set()))
            aliases = (aliases - self.ambiguous_aliases)

            # Make alias search deterministic (prefer gene_upper, then primary, then others sorted)
            ordered_aliases = list(dict.fromkeys([gene_upper, primary] + sorted(aliases - {gene_upper, primary})))

            # safety: never match ambiguous (should already be true, but belt + suspenders)
            ordered_aliases = [a for a in ordered_aliases if a not in self.ambiguous_aliases]

            hpa_match = next((a for a in ordered_aliases if a in hpa_genes), None)
            gtex_match = next((a for a in ordered_aliases if a in gtex_genes), None)
            
            penalties = {
                'pen_critical_hpa': 0, 'pen_gi_hpa': 0, 'pen_brain_hpa': 0,
                'pen_critical_gtex': 0, 'pen_gi_gtex': 0, 'pen_brain_gtex': 0,
                'pen_missing': 0, 'pen_ubiquitous': 0
            }
            
            bonuses = {
                'bonus_hema': 0,
                'bonus_specificity': 0,
                'bonus_known_good': 0
            }
            
            is_cd_marker = is_cd_surface_marker(gene_upper)
            is_known_antibody = gene_upper in self.known_antibody_targets
            
            if is_cd_marker or is_known_antibody:
                modality = 'antibody'
            elif gene_upper in {'FLT3', 'KIT', 'JAK1', 'JAK2', 'EGFR'}:
                modality = 'small_molecule'
            else:
                modality = 'unknown'
            
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
            priority_score = penalty_total + bonus_total  # Don't clip
            
            results.append({
                'gene': gene,
                'hpa_symbol': hpa_match,
                'gtex_symbol': gtex_match,
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
        print(f"✓ Ranked {len(df)} genes (safety_risk=0: {(df['safety_risk']==0).sum()})")
        
        return df
    
    def select_top_n(self, safety_df, target_count=40):
        """Deterministic top-N selection"""
        print(f"\n🎯 TOP-N SELECTION")
        print("="*70)
        
        EXCLUDE_CUTOFF = 70
        HIGH_RISK_CUTOFF = 45
        
        def hard_cat(r):
            return 'EXCLUDE' if r >= EXCLUDE_CUTOFF else ('HIGH_RISK' if r >= HIGH_RISK_CUTOFF else 'OK')
        
        safety_df['hard_category'] = safety_df['safety_risk'].apply(hard_cat)
        
        pool = safety_df[(safety_df['hard_category'] == 'OK') & (~safety_df['missing_data'])].copy()
        
        print(f"  Pool: {len(pool)} genes")
        
        actual_target = min(target_count, len(pool))
        if actual_target < target_count:
            print(f"  ⚠️ Only {actual_target} available")
        
        pool_sorted = pool.sort_values(
            ['priority_score', 'safety_risk', 'bonus_total', 'specificity_ratio', 'gtex_max_hema', 'gene'],
            ascending=[True, True, True, False, False, True],
            kind='mergesort'
        )
        
        top_n = pool_sorted.head(actual_target)
        
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
        
        n_safe = (safety_df['final_category'] == 'SAFE').sum()
        n_check = (safety_df['final_category'] == 'CHECK').sum()
        
        print(f"✅ SAFE: {n_safe}, CHECK: {n_check}")
        
        return safety_df
    
    def report(self, final_df):
        """Generate report"""
        
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
        print("FINAL RESULTS")
        print("="*70)
        print(f"1000 → {len(final_df)} membrane → {len(safe_check_sorted)} tissue-safe")
        
        print("\n" + "="*70)
        print("TOP 50 SAFE/CHECK (SORTED)")
        print("="*70)
        print(f"{'#':<4} {'Gene':<12} {'Cat':<8} {'Safety':<7} {'Priority':<9} {'CD':<4} {'BBB':<5}")
        print("-" * 70)
        
        for idx, (_, row) in enumerate(safe_check_sorted.head(50).iterrows(), start=1):
            cd = "Yes" if row['is_cd_marker'] else "No"
            bbb = "OK" if row['bbb_exempt'] else "No"
            print(f"{idx:<4} {row['gene']:<12} {row['final_category']:<8} "
                  f"{row['safety_risk']:<7.0f} {row['priority_score']:<9.0f} {cd:<4} {bbb:<5}")
        
        print("\n🔍 VALIDATIONS:")
        for gene in ['FLT3', 'CD33', 'KCNS3', 'CDH1', 'CDK7']:
            if gene in final_df['gene'].values:
                row = final_df[final_df['gene'] == gene].iloc[0]
                print(f"  {gene:12s}: {row['final_category']:10s} | Safety={row['safety_risk']:3.0f}, Priority={row['priority_score']:4.0f}")
        
        safe_check_sorted.to_csv(self.output_dir / 'FINAL_SAFE_CHECK_SORTED.csv', index=False)
        final_df.to_csv(self.output_dir / 'ALL_RANKED.csv', index=False)
        
        print(f"\n✓ Saved to {self.output_dir}")
        
        self._viz(final_df, safe_check_sorted)
        
        return final_df
    
    def _viz(self, final_df, safe_check):
        """Visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        for cat, col in [('SAFE', 'green'), ('CHECK', 'lightgreen'), ('CAUTION', 'yellow'),
                        ('HIGH_RISK', 'orange'), ('EXCLUDE', 'red')]:
            m = final_df['final_category'] == cat
            if m.sum() > 0:
                ax1.scatter(final_df[m]['safety_risk'], final_df[m]['priority_score'],
                           label=f'{cat} (n={m.sum()})', color=col, alpha=0.6)
        ax1.set_xlabel('Safety Risk')
        ax1.set_ylabel('Priority Score')
        ax1.set_title('Separated Scoring')
        ax1.legend(fontsize=8)
        
        ax2 = axes[0, 1]
        ax2.scatter(final_df['gtex_max_critical'], final_df['gtex_max_gi'],
                   c=final_df['safety_risk'], cmap='RdYlGn_r', alpha=0.6)
        ax2.set_xlabel('Max Critical')
        ax2.set_ylabel('Max GI')
        ax2.set_title('GTEx')
        
        ax3 = axes[1, 0]
        cat_counts = final_df['final_category'].value_counts()
        colors = {'SAFE': 'green', 'CHECK': 'lightgreen', 'CAUTION': 'yellow',
                 'HIGH_RISK': 'orange', 'EXCLUDE': 'red'}
        ax3.pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%',
               colors=[colors.get(x, 'gray') for x in cat_counts.index])
        ax3.set_title('Distribution')
        
        ax4 = axes[1, 1]
        top20 = safe_check.head(20)
        cols = ['green' if x == 'SAFE' else 'lightgreen' for x in top20['final_category']]
        ax4.barh(range(len(top20)), top20['safety_risk'].values, color=cols)
        ax4.set_yticks(range(len(top20)))
        ax4.set_yticklabels(top20['gene'].values, fontsize=8)
        ax4.set_xlabel('Safety Risk')
        ax4.set_title('Top 20')
        ax4.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'FINAL_REPORT.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("✓ Viz saved")
    
    def run(self):
        """Run complete analysis with coverage checks"""

        hpa_dict = self.parse_hpa()
        gtex_df = self.parse_gtex()

        # Rank ONCE (this defines primary-gene level matches)
        safety_df = self.rank(hpa_dict, gtex_df)

        # Coverage sanity checks (PRIMARY-GENE LEVEL)
        print("\n" + "="*70)
        print("COVERAGE SANITY CHECKS (PRIMARY-GENE LEVEL)")
        print("="*70)

        n = len(self.membrane_persisters)
        hpa_cov = safety_df['hpa_symbol'].notna().sum()
        gtex_cov = safety_df['gtex_symbol'].notna().sum()
        either_cov = (safety_df['hpa_symbol'].notna() | safety_df['gtex_symbol'].notna()).sum()

        print(f"HPA primary coverage:  {hpa_cov} / {n} ({100*hpa_cov/n:.1f}%)")
        print(f"GTEx primary coverage: {gtex_cov} / {n} ({100*gtex_cov/n:.1f}%)")
        print(f"Any source coverage:   {either_cov} / {n} ({100*either_cov/n:.1f}%)")

        # Symbol-level counts (OK to show, just label as symbol-level)
        print(f"\nHPA symbol matches extracted:  {len(hpa_dict)}")
        print(f"GTEx symbol matches extracted: {len(gtex_df)}")

        # Warnings must be based on PRIMARY coverage
        if hpa_cov < 0.5 * n:
            print("  ⚠️ WARNING: Low HPA primary coverage!")
        if gtex_cov < 0.5 * n:
            print("  ⚠️ WARNING: Low GTEx primary coverage!")

        # Continue pipeline (use the SAME safety_df from above)
        safety_df = self.select_top_n(safety_df, target_count=40)
        final = self.report(safety_df)

        print("\n✅ COMPLETE - All fixes applied!")
        return final


if __name__ == "__main__":
    
    analyzer = FinalHardened_TissueFilter(
        gene_list_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt',
        membrane_excel_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/PM proteins.xlsx',
        hpa_xml_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/tissue_filtered_real_data/proteinatlas.xml.gz',
        gtex_file='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/tissue_filtered_real_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz',
        output_dir='/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/integrated_1000gene_analysis'
    )
    
    results = analyzer.run()
