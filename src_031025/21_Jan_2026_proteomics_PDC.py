#!/usr/bin/env python3
"""
PROTEOMICS VALIDATION WITH PDC INTEGRATION
===========================================
Integrates:
1. ProteomicsDB (general protein expression)
2. HPA (tissue-level IHC)
3. PDC/CPTAC (cancer-specific proteomics)
4. Literature (known AML targets)
"""

import pandas as pd
import numpy as np
import requests
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import warnings
import json
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

class ProteomicsValidator_PDC:
    
    def __init__(self, gene_list_file: Path, pm_proteins_file: Path,
                 hpa_xml_file: Path, tissue_safe_dir: Path, output_dir: Path):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("PROTEOMICS VALIDATION WITH PDC/CPTAC")
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
                    print(f"✓ Found: {name}")
                    return candidate_path
        
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
        print("\n" + "="*80)
        print("STEP 2A: UNIPROT")
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
                                    
                                    if 'results' in status_data:
                                        for entry in status_data['results']:
                                            gene_name = entry.get('from', '').upper()
                                            to_entry = entry.get('to', {})
                                            uniprot_id = to_entry.get('primaryAccession', '')
                                            if gene_name and uniprot_id:
                                                gene_to_uniprot[gene_name] = uniprot_id
                                        break
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
    
    def check_surface_accessibility(self, gene_to_uniprot: Dict[str, str]) -> Dict[str, dict]:
        """
        NEW: Check UniProt for extracellular accessibility
        Returns: {gene: {'accessible': bool, 'tm_helices': int, 'extracellular': bool, 'subcellular': str}}
        """
        print("\n" + "="*80)
        print("STEP 2A-EXTRA: UNIPROT TOPOLOGY (SURFACE ACCESSIBILITY)")
        print("="*80)
        
        # Check cache
        topo_cache = self.output_dir / 'uniprot_topology_cache.csv'
        if topo_cache.exists():
            print("✓ Loading topology cache...")
            cached_df = pd.read_csv(topo_cache)
            cached = {}
            for _, row in cached_df.iterrows():
                cached[row['gene']] = {
                    'accessible': row['surface_accessible'],
                    'tm_helices': row['tm_helices'],
                    'extracellular': row['has_extracellular'],
                    'subcellular': row['subcellular_location']
                }
            print(f"  Cached: {len(cached)} genes")
            return cached
        
        print(f"Checking topology for {len(gene_to_uniprot)} genes...")
        
        topology_data = {}
        
        for gene, uniprot_id in gene_to_uniprot.items():
            try:
                # Query UniProt features API
                url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract transmembrane helices
                    tm_helices = 0
                    has_extracellular = False
                    
                    features = data.get('features', [])
                    for feature in features:
                        feat_type = feature.get('type', '')
                        
                        if feat_type == 'Transmembrane':
                            tm_helices += 1
                        
                        if feat_type in ['Topological domain', 'Domain']:
                            description = feature.get('description', '').lower()
                            if 'extracellular' in description or 'lumenal' in description:
                                has_extracellular = True
                    
                    # Extract subcellular location
                    subcellular = []
                    comments = data.get('comments', [])
                    for comment in comments:
                        if comment.get('commentType') == 'SUBCELLULAR LOCATION':
                            locations = comment.get('subcellularLocations', [])
                            for loc in locations:
                                location = loc.get('location', {})
                                value = location.get('value', '')
                                if value:
                                    subcellular.append(value)
                    
                    subcellular_str = '; '.join(subcellular) if subcellular else 'Unknown'
                    
                    # Determine surface accessibility
                    # Criteria: Cell membrane + TM helices + extracellular domain
                    is_cell_membrane = any('membrane' in s.lower() for s in subcellular)
                    surface_accessible = is_cell_membrane and tm_helices > 0 and has_extracellular
                    
                    topology_data[gene] = {
                        'accessible': surface_accessible,
                        'tm_helices': tm_helices,
                        'extracellular': has_extracellular,
                        'subcellular': subcellular_str
                    }
                
                else:
                    # Default: assume not surface accessible
                    topology_data[gene] = {
                        'accessible': False,
                        'tm_helices': 0,
                        'extracellular': False,
                        'subcellular': 'Unknown'
                    }
                
                time.sleep(0.1)  # Rate limiting
                
                if len(topology_data) % 20 == 0:
                    print(f"  {len(topology_data)}/{len(gene_to_uniprot)}")
            
            except:
                topology_data[gene] = {
                    'accessible': False,
                    'tm_helices': 0,
                    'extracellular': False,
                    'subcellular': 'Unknown'
                }
        
        # Save cache
        cache_records = []
        for gene, data in topology_data.items():
            cache_records.append({
                'gene': gene,
                'uniprot_id': gene_to_uniprot[gene],
                'surface_accessible': data['accessible'],
                'tm_helices': data['tm_helices'],
                'has_extracellular': data['extracellular'],
                'subcellular_location': data['subcellular']
            })
        
        cache_df = pd.DataFrame(cache_records)
        cache_df.to_csv(topo_cache, index=False)
        
        accessible = sum(1 for d in topology_data.values() if d['accessible'])
        print(f"\n✓ TOPOLOGY:")
        print(f"   Total: {len(topology_data)}")
        print(f"   Surface-accessible: {accessible} ({100*accessible/len(topology_data):.1f}%)")
        print(f"   (Criteria: Cell membrane + TM helices + extracellular domain)")
        
        return topology_data



    def query_pdc_cptac(self, genes_with_uniprot: List[Tuple[str, str]], force_refresh: bool = False):
        """
        ENHANCED with auto-cache upgrade
        """
        print("\n" + "="*80)
        print("STEP 2C: PDC/BEAT AML (AUTO-UPGRADE)")
        print("="*80)
        
        # SUPERVISOR'S RECOMMENDATION: Auto-refresh when cache schema is old
        pdc_cache = self.output_dir / 'pdc_cache.csv'
        
        required_cols = {
            'gene', 'study_id', 'study_name', 'project_name', 'disease_type', 
            'primary_site', 'analytical_fraction', 'data_type', 'is_aml_specific', 
            'detected', 'source'
        }
        
        if pdc_cache.exists() and not force_refresh:
            cached_df = pd.read_csv(pdc_cache)
            missing = required_cols - set(cached_df.columns)
            
            if missing:
                print(f"  ℹ️ PDC cache missing columns: {sorted(missing)}")
                print("  → Auto-rebuilding cache with new schema...")
                force_refresh = True
            else:
                print("✓ Loading PDC cache...")
                print(f"  Cached: {cached_df['gene'].nunique()} genes from {cached_df['study_id'].nunique()} studies")
                # Show studies with AML-specific data
                if 'is_aml_specific' in cached_df.columns:
                    aml_specific = cached_df[cached_df['is_aml_specific']]['study_id'].unique()
                    if len(aml_specific) > 0:
                        print(f"  AML-specific studies: {', '.join(aml_specific)}")
                return cached_df
        
        if force_refresh:
            print("  force_refresh=True, re-querying PDC...")
        
        print(f"Querying PDC for {len(genes_with_uniprot)} genes...")
        
        results = []
        pdc_url = 'https://pdc.cancer.gov/graphql'
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        }
        
        your_gene_set = set(g[0].upper() for g in genes_with_uniprot)
        gene_to_uniprot = dict(genes_with_uniprot)
        uniprot_to_gene = {v: k for k, v in gene_to_uniprot.items()}
        
        # Get studies
        studies_query = """
        {
        getPaginatedUIStudy(limit: 500) {
            uiStudies {
            pdc_study_id
            submitter_id_name
            project_name
            disease_type
            primary_site
            analytical_fraction
            experiment_type
            }
        }
        }
        """
        
        try:
            response = requests.post(pdc_url, headers=headers, json={'query': studies_query}, timeout=30)
            
            if response.status_code != 200:
                return pd.DataFrame()
            
            data = response.json()
            studies = data.get('data', {}).get('getPaginatedUIStudy', {}).get('uiStudies', [])
            
            print(f"  ✓ Found {len(studies)} studies")
            
            # TWEAK D: Separate AML/heme from general blood keywords
            aml_heme_studies = []
            general_blood_studies = []
            
            for s in studies:
                all_text = f"{s.get('submitter_id_name', '')} {s.get('disease_type', '')} {s.get('primary_site', '')}".upper()
                
                # Specific AML/hematologic keywords
                if any(kw in all_text for kw in ['AML', 'ACUTE MYELOID', 'LEUKEMIA', 'BEAT AML']):
                    aml_heme_studies.append(s)
                # General blood keywords
                elif any(kw in all_text for kw in ['BLOOD', 'BONE MARROW', 'HEMATOLOGIC', 'MYELOID', 'LYMPH']):
                    general_blood_studies.append(s)
            
            print(f"  ✓ AML/hematologic-specific: {len(aml_heme_studies)} studies")
            print(f"  ✓ General blood-related: {len(general_blood_studies)} studies")
            
            # Deduplicate
            all_candidates = []
            seen_ids = set()
            for study in (aml_heme_studies + general_blood_studies):
                study_id = study.get('pdc_study_id', '')
                if study_id and study_id not in seen_ids:
                    all_candidates.append(study)
                    seen_ids.add(study_id)
            
            print(f"  ✓ Querying {len(all_candidates)} unique studies...")
            
            # Try multiple data types
            data_types_to_try = [
                'unshared_log2_ratio',
                'log2_ratio',
                'abundance',
                'itraq',
                'tmt',
                'precursor_area'
            ]
            
            for idx, study in enumerate(all_candidates[:20], 1):
                study_id = study.get('pdc_study_id', '')
                name = study.get('submitter_id_name', '')
                disease = study.get('disease_type', '')
                site = study.get('primary_site', '')
                fraction = study.get('analytical_fraction', '')
                project = study.get('project_name', '')
                
                # TWEAK D: Check if AML-specific
                is_aml_specific = any(kw in f"{name} {disease}".upper() for kw in ['AML', 'ACUTE MYELOID', 'LEUKEMIA', 'BEAT AML'])
                
                print(f"\n  [{idx}/{min(20, len(all_candidates))}] {study_id}: {name[:50]}")
                
                matrix_found = False
                
                for data_type in data_types_to_try:
                    if matrix_found:
                        break
                    
                    matrix_query = """
                    {
                    quantDataMatrix(
                        pdc_study_id: "%s"
                        data_type: "%s"
                        acceptDUA: true
                    )
                    }
                    """ % (study_id, data_type)
                    
                    try:
                        matrix_response = requests.post(pdc_url, headers=headers, json={'query': matrix_query}, timeout=60)
                        
                        if matrix_response.status_code == 200:
                            matrix_data = matrix_response.json()
                            
                            if 'errors' in matrix_data:
                                continue
                            
                            matrix = matrix_data.get('data', {}).get('quantDataMatrix', [])
                            
                            if matrix and len(matrix) >= 2:
                                print(f"      ✓ data_type: {data_type}")
                                matrix_found = True
                                
                                try:
                                    headers_row = matrix[0]
                                    data_rows = matrix[1:]
                                    
                                    df = pd.DataFrame(data_rows, columns=headers_row)
                                    print(f"      Matrix: {len(df)} rows x {len(df.columns)} cols")
                                    
                                    # TWEAK C: Try gene symbol first, then UniProt
                                    gene_col = None
                                    uniprot_col = None
                                    
                                    for col in df.columns:
                                        col_lower = str(col).lower()
                                        if ('gene' in col_lower or 'symbol' in col_lower) and 'id' not in col_lower:
                                            gene_col = col
                                        if 'uniprot' in col_lower or 'accession' in col_lower:
                                            uniprot_col = col
                                    
                                    matched_genes = set()
                                    
                                    # Try gene symbol match
                                    if gene_col:
                                        df[gene_col] = df[gene_col].astype(str).str.upper()
                                        matched_by_gene = df[df[gene_col].isin(your_gene_set)]
                                        matched_genes.update(matched_by_gene[gene_col].unique())
                                    
                                    # TWEAK C: Try UniProt match if gene column missing or low coverage
                                    if uniprot_col and len(matched_genes) < len(your_gene_set) * 0.1:
                                        df[uniprot_col] = df[uniprot_col].astype(str).str.upper()
                                        your_uniprot_set = set(gene_to_uniprot.values())
                                        matched_by_uniprot = df[df[uniprot_col].isin(your_uniprot_set)]
                                        
                                        for uniprot_id in matched_by_uniprot[uniprot_col].unique():
                                            if uniprot_id in uniprot_to_gene:
                                                matched_genes.add(uniprot_to_gene[uniprot_id])
                                    
                                    print(f"      Matched: {len(matched_genes)} genes")
                                    
                                    if len(matched_genes) > 0:
                                        # TWEAK B: Save study metadata
                                        for gene in matched_genes:
                                            results.append({
                                                'gene': gene,
                                                'study_id': study_id,
                                                'study_name': name,
                                                'project_name': project,
                                                'disease_type': disease,
                                                'primary_site': site,
                                                'analytical_fraction': fraction,
                                                'data_type': data_type,
                                                'is_aml_specific': is_aml_specific,  # TWEAK D
                                                'detected': True,
                                                'source': 'PDC'
                                            })
                                
                                except Exception as e:
                                    print(f"      Parse error: {str(e)[:80]}")
                        
                        time.sleep(0.2)
                    
                    except:
                        pass
                
                if not matrix_found:
                    print(f"      No quantDataMatrix")
        
        except Exception as e:
            print(f"  ℹ️ PDC error: {str(e)[:150]}")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            detected = df['gene'].nunique()
            studies_used = df['study_id'].nunique()
            aml_specific_count = df[df['is_aml_specific']]['gene'].nunique()
            
            print(f"\n✓ PDC SUCCESS:")
            print(f"   Total: {detected} genes across {studies_used} studies")
            print(f"   AML-specific: {aml_specific_count} genes")
            
            # TWEAK B: Save with full metadata
            df.to_csv(pdc_cache, index=False)
            
            # Show which studies contributed
            print(f"\n  Studies with matches:")
            for study_id in df['study_id'].unique():
                study_data = df[df['study_id'] == study_id].iloc[0]
                gene_count = len(df[df['study_id'] == study_id])
                print(f"    {study_id}: {study_data['study_name'][:40]} ({gene_count} genes)")
        else:
            print("\n  ℹ️ No PDC matches")
        
        return df


    
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
        print(f"   Total entries (ALL): {total_entries}")
        print(f"   Hematological: {hema_entries}")
        
        if len(df_all) > 0:
            print(f"   Genes (any tissue): {df_all['gene'].nunique()}")
            df_all.to_csv(self.output_dir / 'proteomicsdb_ALL.csv', index=False)
        
        if len(df_hema) > 0:
            print(f"   Genes (hema): {df_hema['gene'].nunique()}")
            df_hema.to_csv(self.output_dir / 'proteomicsdb_HEMA.csv', index=False)
        
        # Also query PDC
        #pdc_df = self.query_pdc_cptac(genes_with_uniprot)
        
        #return df_all, df_hema, pdc_df
        return df_all, df_hema

    
    def extract_hpa_protein(self, membrane_df):
        """
        QUALITY IMPROVEMENT A: Only count hematological tissue detection
        """
        print("\n" + "="*80)
        print("STEP 3: HPA (HEMATOLOGICAL TISSUES ONLY)")
        print("="*80)
        
        if not self.hpa_xml_file.exists():
            membrane_df['hpa_protein_detected'] = False
            return membrane_df
        
        target_genes = set(membrane_df[membrane_df['membrane_validated']]['gene'])
        
        # SUPERVISOR'S RECOMMENDATION: Hematological tissues only
        hema_tissues = {
            'bone marrow', 'blood', 'spleen', 'lymph node', 'tonsil', 
            'thymus', 'lymphoid', 'hematopoietic', 'leukocyte', 'lymphocyte'
        }
        
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
                                tissue_elem = tissue_data.find('.//{*}tissue')
                                level_elem = tissue_data.find('.//{*}level[@type="staining"]')
                                
                                if tissue_elem is not None and level_elem is not None:
                                    tissue_name = tissue_elem.text.lower() if tissue_elem.text else ''
                                    level = level_elem.text.lower() if level_elem.text else ''
                                    
                                    # Check if hematological tissue AND detected
                                    if any(ht in tissue_name for ht in hema_tissues):
                                        if 'not detected' not in level:
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
        hema_confirmed = membrane_df['hpa_protein_detected'].sum()
        print(f"✓ Hematological protein detection: {hema_confirmed} genes")
        print(f"  (Restricted to: {', '.join(list(hema_tissues)[:5])}...)")
        
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
    
    def integrate_evidence(self, membrane_df, pdb_all, pdb_hema, pdc_df, topology_data: Dict[str, dict]):
        """
        ENHANCED: Includes surface accessibility from UniProt topology
        """
        print("\n" + "="*80)
        print("STEP 5: INTEGRATION (WITH TOPOLOGY)")
        print("="*80)
        
        # ProteomicsDB
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
        
        # PDC with graceful fallback
        if len(pdc_df) > 0:
            pdc_all_detected = pdc_df['gene'].unique()
            membrane_df['pdc_detected'] = membrane_df['gene'].isin(pdc_all_detected)
            
            if 'is_aml_specific' in pdc_df.columns:
                pdc_aml_specific = pdc_df[pdc_df['is_aml_specific']]['gene'].unique()
                membrane_df['pdc_aml_specific'] = membrane_df['gene'].isin(pdc_aml_specific)
                print("  ✓ Using PDC AML-specific flag")
            else:
                membrane_df['pdc_aml_specific'] = membrane_df['pdc_detected']
                print("  ℹ️ Old cache, treating all PDC as AML-specific")
            
            # NEW: PDC study count (shows recurrence)
            pdc_study_count = pdc_df.groupby('gene')['study_id'].nunique()
            membrane_df['pdc_study_count'] = membrane_df['gene'].map(pdc_study_count).fillna(0).astype(int)
        else:
            membrane_df['pdc_detected'] = False
            membrane_df['pdc_aml_specific'] = False
            membrane_df['pdc_study_count'] = 0
        
        # NEW: Surface accessibility
        membrane_df['surface_accessible'] = membrane_df['gene'].map(
            lambda g: topology_data.get(g, {}).get('accessible', False)
        )
        membrane_df['tm_helices'] = membrane_df['gene'].map(
            lambda g: topology_data.get(g, {}).get('tm_helices', 0)
        )
        membrane_df['has_extracellular'] = membrane_df['gene'].map(
            lambda g: topology_data.get(g, {}).get('extracellular', False)
        )
        
        # Protein validation
        membrane_df['protein_validated'] = (
            membrane_df['membrane_validated'] &
            (membrane_df['proteomicsdb_detected'] |
            membrane_df['pdc_detected'] |
            membrane_df['hpa_protein_detected'] |
            membrane_df['literature_validated'])
        )
        
        # Enhanced confidence with surface accessibility
        def calc_conf(row):
            score = 0
            if row['membrane_validated']: score += 1
            if row.get('proteomicsdb_detected', False): score += 2
            if row.get('pdc_aml_specific', False): score += 3
            elif row.get('pdc_detected', False): score += 2
            if row.get('proteomicsdb_hema', False): score += 1
            if row.get('hpa_protein_detected', False): score += 1
            if row.get('literature_validated', False): score += 1
            if row.get('surface_accessible', False): score += 2  # NEW: Surface bonus
            return 'HIGH' if score >= 7 else ('MEDIUM' if score >= 4 else 'LOW')
        
        membrane_df['confidence'] = membrane_df.apply(calc_conf, axis=1)
        
        validated = membrane_df['protein_validated'].sum()
        surface_acc = membrane_df['surface_accessible'].sum()
        pdc_aml = membrane_df['pdc_aml_specific'].sum()
        
        print(f"✓ Protein-validated: {validated}/{len(membrane_df)} ({100*validated/len(membrane_df):.1f}%)")
        print(f"  ProteomicsDB: {membrane_df['proteomicsdb_detected'].sum()}")
        print(f"  PDC (AML-specific): {pdc_aml}")
        print(f"  Surface-accessible: {surface_acc} ({100*surface_acc/len(membrane_df):.1f}%)")
        
        membrane_df.to_csv(self.output_dir / '2_integrated.csv', index=False)
        
        return membrane_df

    
    def apply_to_tissue_safe(self, validated_df):
        """
        ENHANCED: Shows surface accessibility + PDC study recurrence
        """
        print("\n" + "="*80)
        print("STEP 6: TISSUE-SAFE + PROTEIN + TOPOLOGY")
        print("="*80)
        
        if self.tissue_safe_file is None or not self.tissue_safe_file.exists():
            final = validated_df[validated_df['protein_validated']].copy()
        else:
            tissue_safe = pd.read_csv(self.tissue_safe_file)
            print(f"✓ Loaded {len(tissue_safe)} tissue-safe targets")
            
            final = tissue_safe.merge(validated_df, on='gene', how='left')
            
            # Fill NaN
            boolean_cols = ['protein_validated', 'membrane_validated', 'proteomicsdb_detected', 
                        'proteomicsdb_hema', 'hpa_protein_detected', 'literature_validated',
                        'pdc_detected', 'pdc_aml_specific', 'surface_accessible', 
                        'has_extracellular']
            
            for col in boolean_cols:
                if col in final.columns:
                    final[col] = final[col].fillna(False)
            
            numeric_cols = ['proteomicsdb_max_expr', 'tm_helices', 'pdc_study_count']
            for col in numeric_cols:
                if col in final.columns:
                    final[col] = final[col].fillna(0)
            
            if 'confidence' in final.columns:
                final['confidence'] = final['confidence'].fillna('LOW')
        
        # Enhanced priority with surface accessibility
        final['protein_priority'] = (
            final['protein_validated'].astype(int) * 1000 +
            final.get('surface_accessible', pd.Series([False]*len(final))).fillna(False).astype(int) * 500 +  # HIGH WEIGHT!
            final.get('pdc_aml_specific', pd.Series([False]*len(final))).fillna(False).astype(int) * 300 +
            final.get('pdc_study_count', pd.Series([0]*len(final))).fillna(0) * 50 +  # Recurrence bonus
            final.get('proteomicsdb_hema', pd.Series([False]*len(final))).fillna(False).astype(int) * 200 +
            final.get('pdc_detected', pd.Series([False]*len(final))).fillna(False).astype(int) * 150 +
            final.get('proteomicsdb_detected', pd.Series([False]*len(final))).fillna(False).astype(int) * 100 +
            final.get('proteomicsdb_max_expr', pd.Series([0]*len(final))).fillna(0) * 0.1
        )
        
        final = final.sort_values('protein_priority', ascending=False)
        
        protein_val = final['protein_validated'].sum()
        surface_acc = final.get('surface_accessible', pd.Series([False]*len(final))).sum()
        pdc_aml = final.get('pdc_aml_specific', pd.Series([False]*len(final))).sum()
        
        print(f"\n📊 FINAL:")
        print(f"   Total: {len(final)}")
        print(f"   Protein-validated: {protein_val} ({100*protein_val/len(final):.1f}%)")
        print(f"   Surface-accessible: {surface_acc} ({100*surface_acc/len(final) if len(final) > 0 else 0:.1f}%)")
        print(f"   PDC (AML-specific): {pdc_aml}")
        
        print(f"\n🎯 TOP 30 (SURFACE-ACCESSIBLE PRIORITIZED):")
        print("-" * 145)
        print(f"{'#':<4} {'Gene':<12} {'Surface':<8} {'PDC_AML':<8} {'PDC_N':<7} {'PDB_Hema':<10} {'HPA':<6} {'TM':<4} {'Conf':<8} {'Priority':<10}")
        print("-" * 145)
        
        for idx, (_, row) in enumerate(final.head(30).iterrows(), 1):
            surface = "✓" if row.get('surface_accessible', False) else "✗"
            pdc_aml = "✓" if row.get('pdc_aml_specific', False) else "✗"
            pdc_count = int(row.get('pdc_study_count', 0))
            pdb_hema = "✓" if row.get('proteomicsdb_hema', False) else "✗"
            hpa = "✓" if row.get('hpa_protein_detected', False) else "✗"
            tm = int(row.get('tm_helices', 0))
            
            print(f"{idx:<4} {row['gene']:<12} {surface:<8} {pdc_aml:<8} {pdc_count:<7} {pdb_hema:<10} {hpa:<6} "
                f"{tm:<4} {row.get('confidence', 'N/A'):<8} {row.get('protein_priority', 0):<10.1f}")
        
        final.to_csv(self.output_dir / '3_FINAL_WITH_TOPOLOGY.csv', index=False)
        
        # NEW: Show PDC study details for AML-specific hits
        pdc_aml_genes = final[final.get('pdc_aml_specific', False)]['gene'].tolist()
        if len(pdc_aml_genes) > 0:
            print(f"\n📋 PDC AML STUDY RECURRENCE ({len(pdc_aml_genes)} genes):")
            print("-" * 80)
            for gene in pdc_aml_genes[:10]:
                studies = final[final['gene'] == gene]['pdc_study_count'].iloc[0] if gene in final['gene'].values else 0
                surface = "✓ Surface" if final[final['gene'] == gene]['surface_accessible'].iloc[0] else "✗ Junction"
                print(f"  {gene:<12} {surface:<12} in {studies} Beat AML studies")
        
        print(f"\n✓ Saved: 3_FINAL_WITH_TOPOLOGY.csv")
        
        return final

    
    def run(self, force_refresh_pdc: bool = False):
        """
        ENHANCED: Includes UniProt topology checking
        """
        membrane_df = self.validate_membrane()
        
        membrane_genes = membrane_df[membrane_df['membrane_validated']]['gene'].tolist()
        gene_to_uniprot = self.map_genes_to_uniprot(membrane_genes)
        
        # NEW: Check surface accessibility
        topology_data = self.check_surface_accessibility(gene_to_uniprot)
        
        genes_with_uniprot = [(g, gene_to_uniprot[g]) for g in membrane_genes if g in gene_to_uniprot]
        
        pdb_all, pdb_hema = self.query_proteomicsdb(membrane_df, gene_to_uniprot)
        pdc_df = self.query_pdc_cptac(genes_with_uniprot, force_refresh=force_refresh_pdc)
        
        membrane_df = self.extract_hpa_protein(membrane_df)
        membrane_df = self.add_literature(membrane_df)
        
        validated_df = self.integrate_evidence(membrane_df, pdb_all, pdb_hema, pdc_df, topology_data)
        final_df = self.apply_to_tissue_safe(validated_df)
        
        print("\n✅ COMPLETE WITH TOPOLOGY CHECK")
        print("\n📊 COMPREHENSIVE VALIDATION:")
        print(f"   • ProteomicsDB: {len(pdb_all['gene'].unique()) if len(pdb_all) > 0 else 0} genes")
        print(f"   • PDC/Beat AML: {len(pdc_df['gene'].unique()) if len(pdc_df) > 0 else 0} genes")
        print(f"   • Surface-accessible: {sum(1 for d in topology_data.values() if d['accessible'])} genes")
        print(f"   • HPA (hematological): confirmed")
        
        print("\n⚠️  THERAPEUTIC TARGETING:")
        print("   Surface-accessible targets (✓) are suitable for ADC/CAR-T")
        print("   Junction proteins (✗) may have limited accessibility")
        
        return final_df


# RUN with force_refresh option
validator = ProteomicsValidator_PDC(
    gene_list_file=Path('/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt'),
    pm_proteins_file=Path('/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/PM proteins.xlsx'),
    hpa_xml_file=Path('/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/tissue_filtered_real_data/proteinatlas.xml.gz'),
    tissue_safe_dir=Path('/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/integrated_1000gene_analysis'),
    output_dir=Path('./proteomics_validation_WITH_PDC')
)

# Set force_refresh_pdc=True to re-query PDC
results = validator.run(force_refresh_pdc=False)  # Use cache by default

