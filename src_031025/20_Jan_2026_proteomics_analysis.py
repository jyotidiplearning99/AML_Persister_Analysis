#!/usr/bin/env python3
"""
COMPLETE PROTEOMICS VALIDATION - TWEAK 3 APPLIED
=================================================
TWEAK 3: Handle UniProt 'FINISHED' status (critical for coverage)
"""

import pandas as pd
import numpy as np
import requests
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import warnings
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class ProteomicsValidator_Complete:
    
    def __init__(self, gene_list_file: Path, pm_proteins_file: Path,
                 hpa_xml_file: Path, tissue_safe_dir: Path, output_dir: Path):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("PROTEOMICS VALIDATION - TWEAK 3")
        print("="*80)
        
        with open(gene_list_file) as f:
            self.gene_list = [line.strip().upper() for line in f if line.strip()]
        print(f"✓ {len(self.gene_list)} genes")
        
        self.pm_proteins = pd.read_excel(pm_proteins_file)
        print(f"✓ {len(self.pm_proteins)} membrane proteins")
        
        self.tissue_safe_file = self._find_tissue_safe_file(tissue_safe_dir)
        self.hpa_xml_file = hpa_xml_file
        
        self._build_membrane_lookup()
        self._extract_uniprot_from_pm()
        self._load_fallback_mappings()
    
    def _find_tissue_safe_file(self, tissue_safe_dir: Path) -> Path:
        tissue_safe_dir = tissue_safe_dir.resolve()
        
        if tissue_safe_dir.exists():
            candidates = [
                'FINAL_SAFE_CHECK_SORTED.csv',
                'FINAL_SAFE_CHECK_TARGETS_SORTED.csv',
                'SAFE_CHECK_SORTED.csv'
            ]
            
            for name in candidates:
                candidate_path = tissue_safe_dir / name
                if candidate_path.exists():
                    print(f"✓ Found tissue-safe: {name}")
                    return candidate_path
        
        print("⚠️ No tissue-safe file found")
        return None
    
    def _build_membrane_lookup(self):
        self.membrane_genes = set()
        self.gene_to_info = {}
        
        for idx, row in self.pm_proteins.iterrows():
            genes = set()
            
            for col in ['GSEA', 'ProteinAtlas']:
                if col in self.pm_proteins.columns and pd.notna(row[col]):
                    g = str(row[col]).strip().upper()
                    if g:
                        genes.add(g)
            
            if 'Synonyms' in self.pm_proteins.columns and pd.notna(row['Synonyms']):
                syns = str(row['Synonyms']).split(',')
                for syn in syns:
                    s = syn.strip().upper()
                    if s and len(s) > 1:
                        genes.add(s)
            
            for gene in genes:
                self.membrane_genes.add(gene)
                self.gene_to_info[gene] = {
                    'gsea': row.get('GSEA', ''),
                    'hpa': row.get('ProteinAtlas', ''),
                    'synonyms': row.get('Synonyms', '')
                }
    
    def apply_to_tissue_safe(self, validated_df):
        """
        FIXED: Fill NaN AFTER merge, BEFORE astype(int)
        """
        print("\n" + "="*80)
        print("STEP 6: TISSUE-SAFE + PROTEIN")
        print("="*80)
        
        if self.tissue_safe_file is None or not self.tissue_safe_file.exists():
            print("⚠️ Tissue-safe file not found, using all validated")
            final = validated_df[validated_df['protein_validated']].copy()
        else:
            tissue_safe = pd.read_csv(self.tissue_safe_file)
            print(f"✓ Loaded {len(tissue_safe)} tissue-safe targets")
            
            # Merge
            final = tissue_safe.merge(validated_df, on='gene', how='left')
            
            # CRITICAL FIX: Fill NaN AFTER merge, BEFORE astype(int)
            # These columns exist after merge but may contain NaN
            for col in ['protein_validated', 'membrane_validated', 'proteomicsdb_detected', 
                        'proteomicsdb_hema', 'hpa_protein_detected', 'literature_validated']:
                if col in final.columns:
                    final[col] = final[col].fillna(False)
            
            # Fill numeric columns
            for col in ['proteomicsdb_max_expr']:
                if col in final.columns:
                    final[col] = final[col].fillna(0)
            
            # Fill string columns
            if 'confidence' in final.columns:
                final['confidence'] = final['confidence'].fillna('LOW')
        
        # Now safe to convert to int (no NaN values remain)
        final['protein_priority'] = (
            final['protein_validated'].astype(int) * 1000 +
            final.get('proteomicsdb_hema', pd.Series([False]*len(final))).fillna(False).astype(int) * 200 +
            final.get('proteomicsdb_detected', pd.Series([False]*len(final))).fillna(False).astype(int) * 100 +
            final.get('proteomicsdb_max_expr', pd.Series([0]*len(final))).fillna(0) * 0.1
        )
        
        final = final.sort_values('protein_priority', ascending=False)
        
        # Statistics
        protein_val = final['protein_validated'].sum()
        pdb_any = final['proteomicsdb_detected'].sum()
        pdb_hema = final['proteomicsdb_hema'].sum()
        high_conf = (final['confidence'] == 'HIGH').sum()
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Total tissue-safe: {len(final)}")
        print(f"   Protein-validated: {protein_val} ({100*protein_val/len(final):.1f}%)")
        print(f"   ProteomicsDB (any): {pdb_any}")
        print(f"   ProteomicsDB (hema): {pdb_hema}")
        print(f"   High confidence: {high_conf}")
        
        print(f"\n🎯 TOP 30 TISSUE-SAFE + PROTEIN-VALIDATED:")
        print("-" * 115)
        print(f"{'#':<4} {'Gene':<12} {'PDB_Any':<9} {'PDB_Hema':<10} {'HPA':<6} {'Lit':<6} {'Conf':<8} {'Priority':<10}")
        print("-" * 115)
        
        for idx, (_, row) in enumerate(final.head(30).iterrows(), start=1):
            pdb_any_mark = "✓" if row.get('proteomicsdb_detected', False) else "✗"
            pdb_hema_mark = "✓" if row.get('proteomicsdb_hema', False) else "✗"
            hpa_mark = "✓" if row.get('hpa_protein_detected', False) else "✗"
            lit_mark = "✓" if row.get('literature_validated', False) else "✗"
            
            print(f"{idx:<4} {row['gene']:<12} {pdb_any_mark:<9} {pdb_hema_mark:<10} {hpa_mark:<6} "
                f"{lit_mark:<6} {row.get('confidence', 'N/A'):<8} {row.get('protein_priority', 0):<10.1f}")
        
        final.to_csv(self.output_dir / '3_FINAL_TISSUE_SAFE_PROTEIN_VALIDATED.csv', index=False)
        
        print(f"\n✓ Saved: {self.output_dir / '3_FINAL_TISSUE_SAFE_PROTEIN_VALIDATED.csv'}")
        
        return final


    def _extract_uniprot_from_pm(self):
        self.pm_uniprot = {}
        
        uniprot_cols = ['UniProt', 'Uniprot', 'UNIPROT', 'UniProtKB', 'uniprot_id', 
                       'Entry', 'Accession', 'UniProt_ID', 'accession']
        
        uniprot_col = None
        for col in uniprot_cols:
            if col in self.pm_proteins.columns:
                uniprot_col = col
                break
        
        if uniprot_col:
            for idx, row in self.pm_proteins.iterrows():
                genes = set()
                
                for col in ['GSEA', 'ProteinAtlas']:
                    if col in self.pm_proteins.columns and pd.notna(row[col]):
                        genes.add(str(row[col]).strip().upper())
                
                uniprot_id = row.get(uniprot_col)
                if pd.notna(uniprot_id):
                    uniprot_id = str(uniprot_id).strip()
                    for gene in genes:
                        self.pm_uniprot[gene] = uniprot_id
    
    def _load_fallback_mappings(self):
        self.known_uniprot = {
            'CD33': 'P20138', 'CD34': 'P28906', 'CD38': 'P28907', 'CD44': 'P16070',
            'CD47': 'Q08722', 'CD96': 'P40200', 'CD99': 'P14209', 'CD123': 'P26951',
            'FLT3': 'P36888', 'KIT': 'P10721', 'EGFR': 'P00533', 'MET': 'P08581',
            'IL3RA': 'P26951', 'HAVCR2': 'Q8TDQ0', 'CLEC12A': 'Q5QGZ9', 'PTPRC': 'P08575',
            'ICAM1': 'P05362', 'PECAM1': 'P16284', 'ITGAM': 'P11215', 'ITGAX': 'P20702',
            'CDH1': 'P12830', 'CAV1': 'Q03135', 'EPCAM': 'P16422'
        }
    
    def validate_membrane(self):
        print("\n" + "="*80)
        print("STEP 1: MEMBRANE")
        print("="*80)
        
        results = []
        for gene in self.gene_list:
            is_membrane = gene in self.membrane_genes
            record = {'gene': gene, 'membrane_validated': is_membrane}
            
            if is_membrane:
                info = self.gene_to_info.get(gene, {})
                record.update({
                    'gsea_name': info.get('gsea', ''),
                    'hpa_name': info.get('hpa', ''),
                    'synonyms': info.get('synonyms', '')
                })
            
            results.append(record)
        
        df = pd.DataFrame(results)
        membrane_count = df['membrane_validated'].sum()
        print(f"✓ {membrane_count}/{len(df)} ({100*membrane_count/len(df):.1f}%)")
        
        df.to_csv(self.output_dir / '1_membrane.csv', index=False)
        return df
    
    def map_genes_to_uniprot(self, genes: List[str]) -> Dict[str, str]:
        """
        TWEAK 3: Handle FINISHED status (critical for coverage)
        """
        print("\n" + "="*80)
        print("STEP 2A: UNIPROT (WITH FINISHED HANDLING)")
        print("="*80)
        
        cache_file = self.output_dir / 'uniprot_cache.csv'
        
        if cache_file.exists():
            cache_df = pd.read_csv(cache_file)
            cached = dict(zip(cache_df['gene'], cache_df['uniprot_id']))
        else:
            cached = {}
        
        gene_to_uniprot = {}
        
        for gene in genes:
            if gene in self.pm_uniprot:
                gene_to_uniprot[gene] = self.pm_uniprot[gene]
        print(f"✓ Excel: {len(gene_to_uniprot)}/{len(genes)}")
        
        for gene in genes:
            if gene not in gene_to_uniprot and gene in cached:
                gene_to_uniprot[gene] = cached[gene]
        print(f"✓ Cache: {len(gene_to_uniprot)}/{len(genes)}")
        
        for gene in genes:
            if gene not in gene_to_uniprot and gene in self.known_uniprot:
                gene_to_uniprot[gene] = self.known_uniprot[gene]
        print(f"✓ Fallback: {len(gene_to_uniprot)}/{len(genes)}")
        
        remaining = [g for g in genes if g not in gene_to_uniprot]
        
        if len(remaining) > 0:
            print(f"\nAPI for {len(remaining)} genes...")
            
            batch_size = 20
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i+batch_size]
                
                try:
                    ids_string = "\n".join(batch)
                    url = "https://rest.uniprot.org/idmapping/run"
                    params = {'from': 'Gene_Name', 'to': 'UniProtKB', 'ids': ids_string, 'taxId': '9606'}
                    
                    response = requests.post(url, data=params, timeout=30)
                    
                    if response.status_code == 200:
                        job_data = response.json()
                        job_id = job_data.get('jobId')
                        
                        if job_id:
                            for attempt in range(10):
                                time.sleep(2)
                                status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
                                status_response = requests.get(status_url, timeout=10)
                                
                                if status_response.status_code == 200:
                                    status_data = status_response.json()
                                    
                                    # Handle embedded results
                                    if 'results' in status_data:
                                        for entry in status_data['results']:
                                            gene_name = entry.get('from', '').upper()
                                            to_entry = entry.get('to', {})
                                            uniprot_id = to_entry.get('primaryAccession', '')
                                            if gene_name and uniprot_id:
                                                gene_to_uniprot[gene_name] = uniprot_id
                                        break
                                    
                                    # TWEAK 3: Handle FINISHED status (critical!)
                                    elif status_data.get('jobStatus') == 'FINISHED':
                                        results_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
                                        results_response = requests.get(results_url, timeout=30)
                                        if results_response.status_code == 200:
                                            results = results_response.json()
                                            for entry in results.get('results', []):
                                                gene_name = entry.get('from', '').upper()
                                                to_entry = entry.get('to', {})
                                                uniprot_id = to_entry.get('primaryAccession', '')
                                                if gene_name and uniprot_id:
                                                    gene_to_uniprot[gene_name] = uniprot_id
                                        break
                except:
                    pass
        
        print(f"\n✓ TOTAL: {len(gene_to_uniprot)}/{len(genes)} ({100*len(gene_to_uniprot)/len(genes):.1f}%)")
        
        mapping_df = pd.DataFrame([
            {'gene': gene, 'uniprot_id': uniprot_id}
            for gene, uniprot_id in gene_to_uniprot.items()
        ])
        mapping_df.to_csv(cache_file, index=False)
        
        return gene_to_uniprot
    
    def query_proteomicsdb(self, membrane_df, gene_to_uniprot: Dict[str, str]):
        print("\n" + "="*80)
        print("STEP 2B: PROTEOMICSDB")
        print("="*80)
        
        genes_with_uniprot = [
            (row['gene'], gene_to_uniprot.get(row['gene']))
            for _, row in membrane_df[membrane_df['membrane_validated']].iterrows()
            if row['gene'] in gene_to_uniprot
        ]
        
        print(f"Querying {len(genes_with_uniprot)} genes...")
        
        results_all = []
        results_hema = []
        
        processed = 0
        success = 0
        fail = 0
        total_entries = 0
        hema_entries = 0
        
        hema_keywords = ['blood', 'bone marrow', 'leuk', 'lymph', 'plasma', 'serum', 
                        'pbmc', 'monocyte', 'macrophage', 'myeloid', 'erythro', 
                        'megakaryo', 'platelet', 'stem cell', 'cd34']
        
        base_url = "https://www.proteomicsdb.org/proteomicsdb/logic/api"
        
        for gene_name, uniprot_id in genes_with_uniprot:
            processed += 1
            
            try:
                url = (f"{base_url}/proteinexpression.xsodata/"
                       f"InputParams(PROTEINFILTER='{uniprot_id}',"
                       f"MS_LEVEL=1,"
                       f"TISSUE_ID_SELECTION='',"
                       f"TISSUE_CATEGORY_SELECTION='tissue;fluid',"
                       f"SCOPE_SELECTION=1,"
                       f"GROUP_BY_TISSUE=1,"
                       f"CALCULATION_METHOD=0,"
                       f"EXP_ID=-1)/Results?"
                       f"$select=UNIQUE_IDENTIFIER,TISSUE_NAME,NORMALIZED_INTENSITY"
                       f"&$format=json")
                
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    entries = data.get('d', {}).get('results', [])
                    
                    if len(entries) > 0:
                        success += 1
                        total_entries += len(entries)
                        
                        for entry in entries:
                            tissue = entry.get('TISSUE_NAME', '').lower()
                            expression = float(entry.get('NORMALIZED_INTENSITY', 0) or 0)
                            
                            row = {
                                'gene': gene_name,
                                'uniprot_id': uniprot_id,
                                'tissue': tissue,
                                'protein_expression': expression,
                                'detected': expression > 0,
                                'source': 'ProteomicsDB'
                            }
                            results_all.append(row)
                            
                            if any(kw in tissue for kw in hema_keywords):
                                results_hema.append(row)
                                hema_entries += 1
                else:
                    fail += 1
                
                time.sleep(0.3)
                
                if processed % 20 == 0:
                    print(f"  {processed}/{len(genes_with_uniprot)} (Success: {success})")
            
            except Exception as e:
                fail += 1
        
        df_all = pd.DataFrame(results_all)
        df_hema = pd.DataFrame(results_hema)
        
        print(f"\n✓ ProteomicsDB:")
        print(f"   Queried: {len(genes_with_uniprot)}")
        print(f"   Success: {success}")
        print(f"   Failed: {fail}")
        print(f"   Total entries (ALL): {total_entries}")
        print(f"   Hematological: {hema_entries}")
        
        if len(df_all) > 0:
            detected_all = df_all['gene'].nunique()
            print(f"   Genes with ANY tissue: {detected_all}")
            df_all.to_csv(self.output_dir / 'proteomicsdb_ALL.csv', index=False)
        
        if len(df_hema) > 0:
            detected_hema = df_hema['gene'].nunique()
            print(f"   Genes with HEMA: {detected_hema}")
            df_hema.to_csv(self.output_dir / 'proteomicsdb_HEMA.csv', index=False)
        
        return df_all, df_hema
    
    def extract_hpa_protein(self, membrane_df):
        print("\n" + "="*80)
        print("STEP 3: HPA")
        print("="*80)
        
        if not self.hpa_xml_file.exists():
            membrane_df['hpa_protein_detected'] = False
            return membrane_df
        
        target_genes = set(membrane_df[membrane_df['membrane_validated']]['gene'])
        
        hpa_confirmed = {}
        genes_found = 0
        
        try:
            with gzip.open(self.hpa_xml_file, 'rt', encoding='utf-8') as f:
                for event, elem in ET.iterparse(f, events=('end',)):
                    if elem.tag.endswith('entry'):
                        gene_elem = elem.find('.//{*}name')
                        if gene_elem is None:
                            elem.clear()
                            continue
                        
                        gene_name = gene_elem.text.strip().upper()
                        if gene_name not in target_genes:
                            elem.clear()
                            continue
                        
                        genes_found += 1
                        detected = False
                        
                        for antibody_elem in elem.findall('.//{*}antibody'):
                            for tissue_data in antibody_elem.findall('.//{*}tissueExpression/{*}data'):
                                level_elem = tissue_data.find('.//{*}level[@type="staining"]')
                                if level_elem is not None and level_elem.text:
                                    if 'not detected' not in level_elem.text.lower():
                                        detected = True
                                        break
                            if detected:
                                break
                        
                        hpa_confirmed[gene_name] = detected
                        
                        if genes_found % 50 == 0:
                            print(f"  {genes_found}")
                        
                        elem.clear()
        except Exception as e:
            print(f"⚠️ {e}")
        
        membrane_df['hpa_protein_detected'] = membrane_df['gene'].map(hpa_confirmed).fillna(False)
        print(f"✓ {membrane_df['hpa_protein_detected'].sum()} genes")
        
        return membrane_df
    
    def add_literature(self, membrane_df):
        print("\n" + "="*80)
        print("STEP 4: LITERATURE")
        print("="*80)
        
        known = {
            'CD33', 'CD123', 'CD44', 'CD47', 'CD96', 'FLT3', 'KIT',
            'IL3RA', 'HAVCR2', 'CLEC12A', 'PTPRC', 'CD34', 'CD38'
        }
        
        membrane_df['literature_validated'] = membrane_df['gene'].isin(known)
        print(f"✓ {membrane_df['literature_validated'].sum()} genes")
        
        return membrane_df
    
    def integrate_evidence(self, membrane_df, pdb_all, pdb_hema):
        print("\n" + "="*80)
        print("STEP 5: INTEGRATION")
        print("="*80)
        
        if len(pdb_all) > 0:
            pdb_all_detected = pdb_all[pdb_all['detected']]['gene'].unique()
            membrane_df['proteomicsdb_detected'] = membrane_df['gene'].isin(pdb_all_detected)
            max_expr_all = pdb_all.groupby('gene')['protein_expression'].max()
            membrane_df['proteomicsdb_max_expr'] = membrane_df['gene'].map(max_expr_all).fillna(0)
        else:
            membrane_df['proteomicsdb_detected'] = False
            membrane_df['proteomicsdb_max_expr'] = 0
        
        if len(pdb_hema) > 0:
            pdb_hema_detected = pdb_hema[pdb_hema['detected']]['gene'].unique()
            membrane_df['proteomicsdb_hema'] = membrane_df['gene'].isin(pdb_hema_detected)
        else:
            membrane_df['proteomicsdb_hema'] = False
        
        membrane_df['protein_validated'] = (
            membrane_df['membrane_validated'] &
            (membrane_df['proteomicsdb_detected'] |
             membrane_df['hpa_protein_detected'] |
             membrane_df['literature_validated'])
        )
        
        def calc_conf(row):
            score = 0
            if row['membrane_validated']: score += 1
            if row.get('proteomicsdb_detected', False): score += 2
            if row.get('proteomicsdb_hema', False): score += 1
            if row.get('hpa_protein_detected', False): score += 1
            if row.get('literature_validated', False): score += 1
            return 'HIGH' if score >= 5 else ('MEDIUM' if score >= 3 else 'LOW')
        
        membrane_df['confidence'] = membrane_df.apply(calc_conf, axis=1)
        
        validated = membrane_df['protein_validated'].sum()
        
        print(f"✓ Protein-validated: {validated}/{len(membrane_df)} ({100*validated/len(membrane_df):.1f}%)")
        
        membrane_df.to_csv(self.output_dir / '2_integrated.csv', index=False)
        
        return membrane_df
    
    
    def run(self):
        membrane_df = self.validate_membrane()
        
        membrane_genes = membrane_df[membrane_df['membrane_validated']]['gene'].tolist()
        gene_to_uniprot = self.map_genes_to_uniprot(membrane_genes)
        
        pdb_all, pdb_hema = self.query_proteomicsdb(membrane_df, gene_to_uniprot)
        
        membrane_df = self.extract_hpa_protein(membrane_df)
        membrane_df = self.add_literature(membrane_df)
        
        validated_df = self.integrate_evidence(membrane_df, pdb_all, pdb_hema)
        final_df = self.apply_to_tissue_safe(validated_df)
        
        print("\n✅ COMPLETE - ALL TWEAKS")
        return final_df


# RUN
validator = ProteomicsValidator_Complete(
    gene_list_file=Path('/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt'),
    pm_proteins_file=Path('/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/PM proteins.xlsx'),
    hpa_xml_file=Path('/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/tissue_filtered_real_data/proteinatlas.xml.gz'),
    tissue_safe_dir=Path('/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/integrated_1000gene_analysis'),
    output_dir=Path('./proteomics_validation_COMPLETE')
)

results = validator.run()
