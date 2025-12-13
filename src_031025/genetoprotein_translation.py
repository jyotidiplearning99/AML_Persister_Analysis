#!/usr/bin/env python3
"""
Complete Production-Ready Persister Gene-to-Protein Translation and Functional Analysis
With improved druggability visualization (exclusive categories + UpSet plots)
All fixes applied: proper mapping counts, edge creation, DGIdb fallback
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import re
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple, Optional
from io import StringIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
warnings.filterwarnings('ignore')

# Try to import upsetplot for better overlap visualization
try:
    from upsetplot import UpSet, from_contents
    HAVE_UPSET = True
except ImportError:
    HAVE_UPSET = False
    print("  ⚠ upsetplot not installed. Install with: pip install upsetplot")

# Setup paths
GENES_FILE = '/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt'
OUTPUT_DIR = Path('results/protein_go_pathway_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===========================
# UTILITIES
# ===========================

def make_session():
    """Create requests session with retries and proper headers"""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "AML_Persister_Analysis/1.0 (Python)",
        "Accept": "application/json"
    })
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

# ===========================
# PROTEIN MAPPING
# ===========================

class ProteinMapper:
    """Enhanced protein mapping with chunked UniProt queries and MyGene.info rescue"""
    
    def __init__(self):
        self.uniprot_base = 'https://rest.uniprot.org'
        self.mygene_base = 'https://mygene.info/v3'
        self.session = make_session()
        
    def query_uniprot_batch(self, gene_symbols: List[str]) -> pd.DataFrame:
        """Query UniProt with chunking to avoid query length limits"""
        print(f"\n  Querying UniProt for {len(gene_symbols)} genes...")
        
        all_results = []
        chunk_size = 50
        
        for i in range(0, len(gene_symbols), chunk_size):
            chunk = gene_symbols[i:i+chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(gene_symbols) + chunk_size - 1) // chunk_size
            
            print(f"    Chunk {chunk_num}/{total_chunks} ({len(chunk)} genes)...", end='')
            
            query_parts = [f'gene_exact:{g}' for g in chunk]
            query = f"reviewed:true AND organism_id:9606 AND ({' OR '.join(query_parts)})"
            
            fields = [
                'accession',
                'gene_primary',
                'protein_name',
                'cc_subcellular_location',
                'keyword',
                'go_f',
                'go_p',
                'go_c',
                'xref_drugbank',
                'ft_transmem',
                'ft_signal'
            ]
            
            params = {
                'query': query,
                'format': 'tsv',
                'fields': ','.join(fields),
                'size': 500
            }
            
            try:
                time.sleep(0.5)
                response = self.session.get(f'{self.uniprot_base}/uniprotkb/search', 
                                          params=params, timeout=30)
                
                if response.status_code == 200 and response.text.strip():
                    df_chunk = pd.read_csv(StringIO(response.text), sep='\t')
                    if not df_chunk.empty:
                        all_results.append(df_chunk)
                        print(f" ✓ ({len(df_chunk)} entries)")
                    else:
                        print(" (no results)")
                else:
                    print(f" ⚠ Error {response.status_code}")
                    
            except Exception as e:
                print(f" ⚠ Error: {e}")
        
        if all_results:
            df = pd.concat(all_results, ignore_index=True)
            
            # Normalize column names
            column_mapping = {
                'Entry': 'uniprot_id',
                'Accession': 'uniprot_id',
                'Gene Names (primary)': 'gene',
                'Gene names  (primary )': 'gene',
                'Gene names (primary)': 'gene',
                'Protein names': 'protein_name',
                'Subcellular location [CC]': 'subcellular_location',
                'Keywords': 'keywords',
                'Gene Ontology (molecular function)': 'go_f',
                'Gene Ontology (biological process)': 'go_p',
                'Gene Ontology (cellular component)': 'go_c',
                'Cross-reference (DrugBank)': 'drugbank',
                'Transmembrane': 'transmembrane',
                'Signal peptide': 'signal_peptide'
            }
            
            df.rename(columns=column_mapping, inplace=True)
            
            if 'gene' in df.columns:
                df['gene'] = df['gene'].str.upper()
                df = df.drop_duplicates(subset=['gene'], keep='first')
                # Initialize rescued_via for direct UniProt hits
                df['rescued_via'] = ''
            
            print(f"\n    ✓ Total retrieved: {len(df)} unique UniProt entries")
            return df
        else:
            print(f"\n    ⚠ No UniProt results retrieved")
            return pd.DataFrame()
    
    def rescue_via_mygene(self, unmapped_genes: List[str]) -> Dict[str, str]:
        """Rescue unmapped genes via MyGene.info with proper chunking"""
        if not unmapped_genes:
            return {}
            
        print(f"\n  Rescuing {len(unmapped_genes)} unmapped genes via MyGene.info...")
        
        rescued = {}
        chunk_size = 100
        
        for i in range(0, len(unmapped_genes), chunk_size):
            chunk = unmapped_genes[i:i+chunk_size]
            
            params = {
                'q': chunk,
                'scopes': 'symbol,alias',
                'fields': 'symbol,uniprot.Swiss-Prot',
                'species': 'human'
            }
            
            try:
                time.sleep(0.2)
                response = self.session.post(f'{self.mygene_base}/query', 
                                           json=params, timeout=30)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    for hit in results:
                        if isinstance(hit, dict) and 'uniprot' in hit:
                            gene = hit.get('query', '').upper()
                            uniprot = hit['uniprot']
                            
                            uniprot_id = None
                            if isinstance(uniprot, dict):
                                uniprot_id = uniprot.get('Swiss-Prot', '')
                            elif isinstance(uniprot, list):
                                for u in uniprot:
                                    if isinstance(u, dict) and 'Swiss-Prot' in u:
                                        uniprot_id = u['Swiss-Prot']
                                        break
                            
                            if uniprot_id and gene:
                                rescued[gene] = uniprot_id
                                
            except Exception as e:
                print(f"    ⚠ MyGene error for chunk {i//chunk_size}: {e}")
        
        print(f"    ✓ Rescued {len(rescued)} genes via MyGene.info")
        return rescued
    
    def map_genes_to_proteins(self, gene_list: List[str]) -> pd.DataFrame:
        """Complete mapping pipeline with proper error handling"""
        genes_upper = [g.upper() for g in gene_list]
        
        # Step 1: Query UniProt with chunking
        uniprot_df = self.query_uniprot_batch(genes_upper)
        
        # Identify unmapped genes
        mapped_genes = set()
        if not uniprot_df.empty and 'gene' in uniprot_df.columns:
            mapped_genes = set(uniprot_df['gene'].str.upper().dropna())
        
        unmapped = [g for g in genes_upper if g not in mapped_genes]
        print(f"  {len(unmapped)} genes not found in UniProt")
        
        # Step 2: Rescue unmapped via MyGene.info
        rescued = {}
        if unmapped:
            rescued = self.rescue_via_mygene(unmapped)
            
            rescued_rows = []
            for gene, uniprot_id in rescued.items():
                rescued_rows.append({
                    'uniprot_id': uniprot_id,
                    'gene': gene,
                    'protein_name': f'{gene} protein',
                    'rescued_via': 'MyGene.info'
                })
            
            if rescued_rows:
                rescued_df = pd.DataFrame(rescued_rows)
                uniprot_df = pd.concat([uniprot_df, rescued_df], ignore_index=True)
        
        # Step 3: Add remaining unmapped as predicted
        still_unmapped = [g for g in genes_upper 
                         if g not in mapped_genes and g not in rescued]
        
        print(f"  {len(still_unmapped)} genes will be marked as predicted")
        
        for gene in still_unmapped:
            uniprot_df = pd.concat([uniprot_df, pd.DataFrame([{
                'uniprot_id': f'PREDICTED_{gene}',
                'gene': gene,
                'protein_name': f'{gene} (predicted)',
                'rescued_via': 'predicted'
            }])], ignore_index=True)
        
        # FIX: Normalize rescued_via column unconditionally
        if 'rescued_via' not in uniprot_df.columns:
            uniprot_df['rescued_via'] = ''
        else:
            uniprot_df['rescued_via'] = uniprot_df['rescued_via'].fillna('')
        
        # Keep only requested genes
        genes_upper_set = set(genes_upper)
        uniprot_df = uniprot_df[uniprot_df['gene'].isin(genes_upper_set)]
        
        print(f"\n  Final mapping summary for {len(gene_list)} genes:")
        print(f"    • Direct UniProt: {len(uniprot_df[uniprot_df['rescued_via'] == ''])}")
        print(f"    • MyGene rescue: {len(uniprot_df[uniprot_df['rescued_via'] == 'MyGene.info'])}")
        print(f"    • Predicted: {len(uniprot_df[uniprot_df['rescued_via'] == 'predicted'])}")
        
        return uniprot_df

# ===========================
# ENRICHR ANALYSIS WITH EXPORT FALLBACK
# ===========================

class EnrichrAnalyzer:
    """Enhanced Enrichr analysis with export fallback for gene lists"""
    
    def __init__(self):
        self.base_url = 'https://maayanlab.cloud/Enrichr'
        self.session = make_session()
        self.available_libraries = self._fetch_available_libraries()
    
    def _fetch_available_libraries(self) -> List[str]:
        """Auto-fetch available libraries from Enrichr"""
        try:
            response = self.session.get(f'{self.base_url}/json/libraries.json')
            if response.status_code == 200:
                data = response.json()
                
                libs = []
                if isinstance(data, dict):
                    if 'statistics' in data and isinstance(data['statistics'], list):
                        for stat in data['statistics']:
                            if isinstance(stat, dict) and 'libraryName' in stat:
                                libs.append(stat['libraryName'])
                    elif 'libraries' in data:
                        libs = data['libraries']
                    else:
                        libs = list(data.keys())
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if 'libraryName' in item:
                                libs.append(item['libraryName'])
                            elif 'name' in item:
                                libs.append(item['name'])
                        elif isinstance(item, str):
                            libs.append(item)
                
                print(f"  ✓ Found {len(libs)} Enrichr libraries")
                return libs
        except Exception as e:
            print(f"  ⚠ Could not fetch libraries: {e}")
        
        return [
            'GO_Biological_Process_2023',
            'GO_Molecular_Function_2023',
            'GO_Cellular_Component_2023',
            'KEGG_2021_Human',
            'Reactome_2022',
            'WikiPathways_2023_Human',
            'MSigDB_Hallmark_2020',
            'DrugMatrix',
            'DSigDB'
        ]
    
    def _select_best_libraries(self) -> List[str]:
        """Select most recent versions of key libraries"""
        selected = []
        
        priorities = {
            'GO_Biological_Process': r'GO_Biological_Process_(\d{4})',
            'GO_Molecular_Function': r'GO_Molecular_Function_(\d{4})',
            'GO_Cellular_Component': r'GO_Cellular_Component_(\d{4})',
            'KEGG': r'KEGG_(\d{4})_Human',
            'Reactome': r'Reactome_(\d{4})',
            'WikiPathways': r'WikiPathways_(\d{4})_Human',
            'MSigDB_Hallmark': r'MSigDB_Hallmark_(\d{4})',
        }
        
        for category, pattern in priorities.items():
            matches = []
            for lib in self.available_libraries:
                if isinstance(lib, str):
                    m = re.match(pattern, lib)
                    if m:
                        year = int(m.group(1))
                        matches.append((year, lib))
            
            if matches:
                matches.sort(reverse=True)
                selected.append(matches[0][1])
        
        always_include = ['DrugMatrix', 'DSigDB']
        for lib in always_include:
            if lib in self.available_libraries:
                selected.append(lib)
        
        return selected
    
    def _export_table_for_library(self, userlist_id: str, lib: str) -> pd.DataFrame:
        """Fetch the Enrichr export table which includes gene lists"""
        url = f"{self.base_url}/export"
        params = {
            "userListId": userlist_id, 
            "backgroundType": lib,
            "filename": "enrichr"
        }
        
        try:
            r = self.session.get(url, params=params, timeout=60)
            if r.status_code != 200 or not r.text.strip():
                return pd.DataFrame()
            
            # Auto-detect delimiter
            df = pd.read_csv(StringIO(r.text), sep=None, engine="python")
            
            # Normalize column names
            colmap = {
                "Term": "term",
                "P-value": "p_value",
                "Adjusted P-value": "adjusted_p",
                "Adjusted P-value ": "adjusted_p",  # Note trailing space
                "Z-score": "z_score",
                "Combined Score": "combined_score",
                "Genes": "genes",
                "Overlapping Genes": "genes"
            }
            
            for c in list(df.columns):
                if c in colmap:
                    df.rename(columns={c: colmap[c]}, inplace=True)
            
            # Build standardized frame
            out = pd.DataFrame({
                "library": lib,
                "term": df.get("term", pd.Series([], dtype=str)),
                "p_value": pd.to_numeric(df.get("p_value", 1.0), errors="coerce"),
                "adjusted_p": pd.to_numeric(df.get("adjusted_p", 1.0), errors="coerce"),
                "z_score": pd.to_numeric(df.get("z_score", 0), errors="coerce"),
                "combined_score": pd.to_numeric(df.get("combined_score", 0), errors="coerce"),
                "overlap_genes": df.get("genes", pd.Series([], dtype=str)).fillna("").map(
                    lambda s: [g.strip().upper() for g in re.split(r'[;,]', str(s)) if g.strip()]
                )
            })
            out["overlap_size"] = out["overlap_genes"].map(len)
            return out
            
        except Exception as e:
            print(f"    ⚠ Export error for {lib}: {e}")
            return pd.DataFrame()
    
    def analyze(self, gene_list: List[str], description: str = "Persister_Genes") -> pd.DataFrame:
        """Submit genes and retrieve enrichment with export fallback"""
        print(f"\nRunning Enrichr analysis for {len(gene_list)} genes...")
        
        genes_str = '\n'.join(gene_list)
        
        try:
            time.sleep(0.2)
            response = self.session.post(
                f'{self.base_url}/addList',
                files={'list': (None, genes_str), 'description': (None, description)}
            )
            
            if response.status_code != 200:
                print(f"  ⚠ Enrichr submission failed")
                return pd.DataFrame()
            
            result = response.json()
            userlist_id = result.get('userListId')
            
            if not userlist_id:
                return pd.DataFrame()
            
            print(f"  ✓ Submitted (ID: {userlist_id})")
            
            libraries = self._select_best_libraries()
            print(f"  Using {len(libraries)} libraries")
            
            all_results = []
            
            for lib in libraries:
                time.sleep(0.1)
                
                # Try JSON first for basic info
                resp = self.session.get(
                    f'{self.base_url}/enrich',
                    params={'userListId': userlist_id, 'backgroundType': lib}, 
                    timeout=60
                )
                
                lib_rows = []
                if resp.status_code == 200:
                    data = resp.json()
                    if lib in data:
                        terms = data[lib]
                        for term in terms[:30]:
                            lib_rows.append({
                                'library': lib,
                                'term': term[1] if len(term) > 1 else '',
                                'p_value': term[2] if len(term) > 2 else np.nan,
                                'z_score': term[3] if len(term) > 3 else np.nan,
                                'combined_score': term[4] if len(term) > 4 else np.nan,
                                'adjusted_p': term[6] if len(term) > 6 else (term[2] if len(term) > 2 else np.nan),
                                'overlap_genes': [],
                                'overlap_size': 0
                            })
                
                df_json = pd.DataFrame(lib_rows)
                
                # Always fetch export table to get gene lists
                df_export = self._export_table_for_library(userlist_id, lib)
                
                if not df_export.empty:
                    if df_json.empty:
                        lib_df = df_export
                    else:
                        # Merge to add overlap_genes
                        lib_df = df_json.merge(
                            df_export[['term', 'adjusted_p', 'overlap_genes', 'overlap_size']],
                            on='term', how='left', suffixes=('', '_export')
                        )
                        # Prefer export values
                        for col in ['adjusted_p', 'overlap_genes', 'overlap_size']:
                            if f'{col}_export' in lib_df.columns:
                                lib_df[col] = lib_df[f'{col}_export'].combine_first(lib_df[col])
                                lib_df.drop(columns=[f'{col}_export'], inplace=True, errors='ignore')
                else:
                    lib_df = df_json
                
                all_results.append(lib_df)
            
            df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
            
            if not df.empty:
                df = df[pd.to_numeric(df['adjusted_p'], errors='coerce').fillna(1.0) < 0.05] \
                       .sort_values('combined_score', ascending=False)
                print(f"  ✓ Found {len(df)} significant enrichments")
                print(f"  ↪ Terms with overlaps: {int((df['overlap_size'] > 0).sum())}")
                print(f"  ↪ Total overlapped genes: {int(df['overlap_size'].sum())}")
            
            return df
            
        except Exception as e:
            print(f"  ⚠ Enrichr error: {e}")
            return pd.DataFrame()

# ===========================
# ACTIONABILITY TAGGING - FIXED SECRETED LOGIC
# ===========================

class ActionabilityTagger:
    """Tag proteins by druggability based on subcellular location"""
    
    @staticmethod
    def tag_proteins(protein_df: pd.DataFrame) -> pd.DataFrame:
        """Tag proteins based on UniProt subcellular location and keywords"""
        print("\nTagging protein actionability...")
        
        # Initialize tags
        protein_df['is_surface'] = False
        protein_df['is_secreted'] = False
        protein_df['is_transmembrane'] = False
        protein_df['is_kinase'] = False
        protein_df['is_druggable'] = False
        protein_df['druggability_score'] = 0
        
        # Define patterns
        surface_terms = ['cell surface', 'plasma membrane', 'cell membrane']
        secreted_terms = ['secreted', 'exported']
        extracellular_terms = ['extracellular region', 'extracellular space']
        transmem_terms = ['transmembrane', 'multi-pass membrane', 'single-pass membrane']
        kinase_terms = ['kinase', 'phosphorylation', 'atp-binding']
        
        for idx, row in protein_df.iterrows():
            score = 0
            location = ''
            
            # Check subcellular location
            if 'subcellular_location' in row and pd.notna(row['subcellular_location']):
                location = str(row['subcellular_location']).lower()
                
                # Check if membrane-bound
                is_membrane = any(term in location for term in transmem_terms + surface_terms)
                
                # Surface proteins
                if any(term in location for term in surface_terms):
                    protein_df.at[idx, 'is_surface'] = True
                    score += 3
                
                # Stricter secreted criteria - NOT if membrane-bound
                if any(term in location for term in secreted_terms) and not is_membrane:
                    protein_df.at[idx, 'is_secreted'] = True
                    score += 3
                
                # Extracellular but membrane-bound = surface, not secreted
                elif any(term in location for term in extracellular_terms):
                    if is_membrane:
                        protein_df.at[idx, 'is_surface'] = True
                        score = max(score, 3)
                    else:
                        protein_df.at[idx, 'is_secreted'] = True
                        score += 3
                
                if any(term in location for term in transmem_terms):
                    protein_df.at[idx, 'is_transmembrane'] = True
                    score += 2
            
            # Check GO cellular component
            if 'go_c' in row and pd.notna(row['go_c']):
                go_cc = str(row['go_c']).lower()
                
                # Membrane location = surface, not secreted
                if ('plasma membrane' in go_cc or 'cell surface' in go_cc):
                    protein_df.at[idx, 'is_surface'] = True
                    score = max(score, 3)
                
                # Only mark as secreted if explicitly extracellular AND not membrane
                if 'extracellular' in go_cc and 'membrane' not in go_cc:
                    if not protein_df.at[idx, 'is_transmembrane']:
                        protein_df.at[idx, 'is_secreted'] = True
                        score = max(score, 3)
            
            # Check signal peptide - indicates secretion potential
            if 'signal_peptide' in row and pd.notna(row['signal_peptide']):
                # Only mark secreted if not already marked as transmembrane
                if not protein_df.at[idx, 'is_transmembrane']:
                    protein_df.at[idx, 'is_secreted'] = True
                    score += 2
            
            # Check keywords
            if 'keywords' in row and pd.notna(row['keywords']):
                keywords = str(row['keywords']).lower()
                
                if any(term in keywords for term in kinase_terms):
                    protein_df.at[idx, 'is_kinase'] = True
                    score += 2
                
                if 'receptor' in keywords:
                    # Receptors are usually surface, not secreted
                    protein_df.at[idx, 'is_surface'] = True
                    protein_df.at[idx, 'is_secreted'] = False
                    score += 2
                
                if 'enzyme' in keywords:
                    score += 1
            
            # Check transmembrane regions
            if 'transmembrane' in row and pd.notna(row['transmembrane']):
                protein_df.at[idx, 'is_transmembrane'] = True
                # Transmembrane = not secreted
                protein_df.at[idx, 'is_secreted'] = False
                score += 2
            
            # Set druggability
            protein_df.at[idx, 'druggability_score'] = score
            protein_df.at[idx, 'is_druggable'] = score >= 2
        
        # Add priority based on manuscript findings
        priority_genes = ['CD33', 'FLT3', 'TGFB2', 'KIT', 'JAK1', 'JAK2', 'BCL2']
        protein_df['is_priority'] = protein_df['gene'].str.upper().isin(priority_genes)
        
        # Summary
        print(f"  ✓ Tagged {len(protein_df)} proteins:")
        print(f"    • Surface: {protein_df['is_surface'].sum()}")
        print(f"    • Secreted: {protein_df['is_secreted'].sum()}")
        print(f"    • Transmembrane: {protein_df['is_transmembrane'].sum()}")
        print(f"    • Kinase: {protein_df['is_kinase'].sum()}")
        print(f"    • Druggable (score ≥2): {protein_df['is_druggable'].sum()}")
        print(f"    • Priority targets: {protein_df['is_priority'].sum()}")
        
        return protein_df

# ===========================
# DRUG TARGET MAPPING - WITH REST FALLBACK
# ===========================

class DrugTargetMapper:
    """Query DGIdb for drug interactions with robust fallback"""
    
    def __init__(self):
        self.session = make_session()
    
    def _rest_batch(self, genes: List[str]) -> pd.DataFrame:
        """Query DGIdb REST API in small batches"""
        rows = []
        
        # Process in small batches
        for i in range(0, len(genes), 20):
            chunk = sorted(set(genes[i:i+20]))
            
            try:
                genes_param = ','.join(chunk)
                url = f'https://www.dgidb.org/api/v2/interactions.json'
                r = self.session.get(url, params={"genes": genes_param}, timeout=60)
                
                # Check if we got JSON
                if 'application/json' not in r.headers.get('Content-Type', ''):
                    time.sleep(0.5)
                    continue
                
                if r.status_code == 200:
                    jd = r.json()
                    for match in jd.get('matchedTerms', []):
                        gene = match.get('geneName', '')
                        for inter in match.get('interactions', []):
                            rows.append({
                                'gene': gene,
                                'drug': inter.get('drugName', ''),
                                'interaction_types': ';'.join(inter.get('interactionTypes', []) or []),
                                'score': inter.get('score', 0) or 0
                            })
                
                time.sleep(0.25)  # Rate limiting
                
            except Exception as e:
                print(f"    ⚠ DGIdb batch error: {e}")
                continue
        
        return pd.DataFrame(rows)
    
    def query_dgidb(self, gene_list: List[str]) -> pd.DataFrame:
        """Query DGIdb with GraphQL and REST fallback"""
        print(f"\nQuerying DGIdb for {len(gene_list)} genes...")
        
        # Try GraphQL first (small subset)
        url = 'https://dgidb.org/api/graphql'
        query = """
        query($genes: [String!]) {
          genes(names: $genes) {
            name
            interactions {
              drugName
              interactionTypes
              interactionScore
            }
          }
        }
        """
        
        try:
            genes_subset = gene_list[:100]
            
            time.sleep(0.2)
            response = self.session.post(
                url,
                json={'query': query, 'variables': {'genes': genes_subset}},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                results = []
                if 'data' in data and 'genes' in data['data']:
                    genes_data = data['data']['genes']
                    
                    if isinstance(genes_data, dict) and 'nodes' in genes_data:
                        genes_data = genes_data['nodes']
                    elif not isinstance(genes_data, list):
                        genes_data = []
                    
                    for gene_data in genes_data:
                        if gene_data is None:
                            continue
                        
                        gene = gene_data.get('name', '')
                        interactions = gene_data.get('interactions', [])
                        
                        for interaction in interactions:
                            if interaction is None:
                                continue
                            
                            drug_name = (interaction.get('drugName') or 
                                       interaction.get('drug', {}).get('name') if isinstance(interaction.get('drug'), dict) else '')
                            
                            types = interaction.get('interactionTypes', [])
                            if types and isinstance(types, list):
                                types_str = ';'.join(types)
                            else:
                                types_str = ''
                            
                            score = interaction.get('interactionScore', 0)
                            
                            if drug_name:
                                results.append({
                                    'gene': gene,
                                    'drug': drug_name,
                                    'interaction_types': types_str,
                                    'score': score
                                })
                
                if results:
                    df = pd.DataFrame(results)
                    print(f"  ✓ Found {len(df)} drug-gene interactions via GraphQL")
                    return df
                    
        except Exception as e:
            print(f"  ⚠ DGIdb GraphQL error: {e}")
        
        # REST API Fallback
        print("  Trying DGIdb REST API fallback...")
        df_rest = self._rest_batch(gene_list[:200])  # Limit to 200 genes
        
        if not df_rest.empty:
            print(f"  ✓ DGIdb REST found {len(df_rest)} interactions")
        else:
            print("  ⚠ No drug interactions found")
        
        return df_rest

# ===========================
# CYTOSCAPE EXPORT - FIXED EDGE CREATION
# ===========================

class CytoscapeExporter:
    """Export network with nodes AND edges for Cytoscape"""
    
    @staticmethod
    def export_network(protein_df: pd.DataFrame, enrichment_df: pd.DataFrame, 
                       drug_df: pd.DataFrame, output_dir: Path):
        """Create Cytoscape-ready network files with proper edge mapping"""
        print("\nExporting Cytoscape network...")
        
        # Create nodes table
        nodes = []
        node_counter = 0
        
        # Gene/protein nodes
        gene_to_id = {}
        for _, row in protein_df.iterrows():
            node_counter += 1
            node_id = f"gene_{node_counter}"
            gene_name = row['gene']
            
            nodes.append({
                'id': node_id,
                'name': gene_name,
                'type': 'gene',
                'is_surface': row.get('is_surface', False),
                'is_secreted': row.get('is_secreted', False),
                'is_druggable': row.get('is_druggable', False),
                'druggability_score': row.get('druggability_score', 0)
            })
            gene_to_id[gene_name.upper()] = node_id
        
        # Pathway nodes - Only use pathways with overlaps
        pathway_to_id = {}
        edge_source_df = pd.DataFrame()  # Initialize empty
        
        if not enrichment_df.empty:
            # Filter to pathways with actual overlaps
            edge_source_df = enrichment_df[enrichment_df['overlap_size'] > 0] \
                              .sort_values('combined_score', ascending=False) \
                              .head(20)
            
            for _, row in edge_source_df.iterrows():
                node_counter += 1
                node_id = f"pathway_{node_counter}"
                
                nodes.append({
                    'id': node_id,
                    'name': row['term'][:50],
                    'type': 'pathway',
                    'library': row['library'],
                    'p_value': row['p_value'],
                    'combined_score': row['combined_score']
                })
                pathway_to_id[row['term']] = node_id
        
        # Drug nodes
        drug_to_id = {}
        if not drug_df.empty:
            for drug in drug_df['drug'].unique()[:30]:
                node_counter += 1
                node_id = f"drug_{node_counter}"
                
                nodes.append({
                    'id': node_id,
                    'name': drug,
                    'type': 'drug'
                })
                drug_to_id[drug] = node_id
        
        # Create edges table
        edges = []
        edge_counter = 0
        
        # Build edges from the SAME overlap-filtered pathways
        if not edge_source_df.empty:
            print(f"  ↪ Checking overlap genes in enrichment data...")
            has_overlaps = 0
            total_overlap_genes = 0
            
            for _, row in edge_source_df.iterrows():
                pathway_id = pathway_to_id.get(row['term'])
                if pathway_id and 'overlap_genes' in row:
                    # Robust parsing of overlap_genes
                    raw = row.get('overlap_genes', [])
                    
                    if isinstance(raw, str):
                        # Parse string format
                        genes_for_edge = [g.strip().upper() for g in re.split(r'[;,]', raw) if g.strip()]
                    elif isinstance(raw, (list, tuple, np.ndarray, pd.Series)):
                        genes_for_edge = [str(g).strip().upper() for g in raw if str(g).strip()]
                    else:
                        genes_for_edge = []
                    
                    if genes_for_edge:
                        has_overlaps += 1
                        total_overlap_genes += len(genes_for_edge)
                    
                    for gene in genes_for_edge:
                        if gene in gene_to_id:
                            edge_counter += 1
                            edges.append({
                                'id': f"edge_{edge_counter}",
                                'source': gene_to_id[gene],
                                'target': pathway_id,
                                'type': 'gene_pathway',
                                'weight': row['combined_score']
                            })
            
            print(f"    • Pathways with overlaps: {has_overlaps}")
            print(f"    • Total overlap genes: {total_overlap_genes}")
            print(f"    • Gene-pathway edges created: {edge_counter}")
        
        # Gene-drug edges
        if not drug_df.empty:
            drug_edges = 0
            for _, row in drug_df.iterrows():
                gene_upper = str(row['gene']).upper()
                if gene_upper in gene_to_id and row['drug'] in drug_to_id:
                    edge_counter += 1
                    drug_edges += 1
                    edges.append({
                        'id': f"edge_{edge_counter}",
                        'source': gene_to_id[gene_upper],
                        'target': drug_to_id[row['drug']],
                        'type': 'gene_drug',
                        'interaction': row.get('interaction_types', ''),
                        'weight': row.get('score', 1)
                    })
            print(f"    • Gene-drug edges created: {drug_edges}")
        
        # Save files
        nodes_df = pd.DataFrame(nodes)
        edges_df = pd.DataFrame(edges)
        
        # Always save, even if empty
        nodes_df.to_csv(output_dir / 'cytoscape_nodes.csv', index=False)
        edges_df.to_csv(output_dir / 'cytoscape_edges.csv', index=False)
        
        print(f"  ✓ Exported network:")
        print(f"    • Nodes: {len(nodes)} ({len(gene_to_id)} genes, "
              f"{len(pathway_to_id)} pathways, {len(drug_to_id)} drugs)")
        print(f"    • Edges: {len(edges)}")
        
        # Create import instructions
        with open(output_dir / 'cytoscape_import_instructions.txt', 'w') as f:
            f.write("Cytoscape Import Instructions\n")
            f.write("="*50 + "\n\n")
            f.write("1. Open Cytoscape\n")
            f.write("2. File > Import > Network from File\n")
            f.write("3. Select 'cytoscape_edges.csv'\n")
            f.write("4. Set 'source' as Source Node, 'target' as Target Node\n")
            f.write("5. Set 'id' as Edge Attribute\n")
            f.write("6. Import node attributes: File > Import > Table from File\n")
            f.write("7. Select 'cytoscape_nodes.csv', import as Node Table\n")
            f.write("8. Set 'id' as Key Column\n")
            f.write("9. Apply layout: Layout > yFiles Organic Layout\n")
            f.write("10. Style by node type and druggability score\n")

# ===========================
# IMPROVED DRUGGABILITY VISUALIZATION
# ===========================

def create_exclusive_druggability_plot(protein_df: pd.DataFrame, output_dir: Path):
    """Create stacked bar chart with truly exclusive categories"""
    
    # Create mutually exclusive categories
    categories = []
    for idx, row in protein_df.iterrows():
        # Build category string from flags
        cat_parts = []
        if row.get('is_surface', False):
            cat_parts.append('Surface')
        if row.get('is_secreted', False):
            cat_parts.append('Secreted')
        if row.get('is_kinase', False):
            cat_parts.append('Kinase')
        if row.get('is_transmembrane', False) and 'Surface' not in cat_parts:
            cat_parts.append('Transmembrane')
        
        # Assign exclusive category
        if not cat_parts:
            if row.get('is_druggable', False):
                category = 'Other druggable'
            else:
                category = 'Non-druggable'
        elif len(cat_parts) == 1:
            category = cat_parts[0] + ' only'
        else:
            # For combinations, use the most specific
            if 'Kinase' in cat_parts and 'Surface' in cat_parts:
                category = 'Surface Kinase'
            elif 'Kinase' in cat_parts:
                category = 'Kinase (combined)'
            elif 'Surface' in cat_parts and 'Secreted' in cat_parts:
                category = 'Surface+Secreted'
            elif 'Surface' in cat_parts:
                category = 'Surface (combined)'
            elif 'Secreted' in cat_parts:
                category = 'Secreted (combined)'
            else:
                category = '+'.join(sorted(cat_parts[:2]))
        
        categories.append(category)
    
    protein_df['exclusive_category'] = categories
    
    # Count categories
    cat_counts = protein_df['exclusive_category'].value_counts()
    
    # Define color scheme
    color_map = {
        'Surface only': '#ff9999',
        'Secreted only': '#66b3ff',
        'Kinase only': '#99ff99',
        'Transmembrane only': '#ffcc99',
        'Surface Kinase': '#ff6666',
        'Surface (combined)': '#ffb3b3',
        'Secreted (combined)': '#99ccff',
        'Kinase (combined)': '#b3ffb3',
        'Surface+Secreted': '#cc99ff',
        'Other druggable': '#dddddd',
        'Non-druggable': '#999999'
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Stacked bar chart
    category_order = [
        'Surface Kinase',
        'Surface only',
        'Surface (combined)',
        'Secreted only', 
        'Secreted (combined)',
        'Kinase only',
        'Kinase (combined)',
        'Transmembrane only',
        'Surface+Secreted',
        'Other druggable',
        'Non-druggable'
    ]
    
    # Filter to existing categories
    category_order = [c for c in category_order if c in cat_counts.index]
    
    # Create data for stacked bar
    values = [cat_counts.get(cat, 0) for cat in category_order]
    colors = [color_map.get(cat, '#cccccc') for cat in category_order]
    
    # Create horizontal stacked bar
    y_pos = 0
    left = 0
    for i, (cat, val, color) in enumerate(zip(category_order, values, colors)):
        if val > 0:
            ax1.barh(y_pos, val, left=left, color=color, label=f'{cat} ({val})')
            if val > 20:
                ax1.text(left + val/2, y_pos, str(val), 
                        ha='center', va='center', fontsize=9)
            left += val
    
    ax1.set_xlim(0, len(protein_df))
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.set_xlabel('Number of Proteins')
    ax1.set_title('Protein Actionability Categories (Exclusive)')
    ax1.axvline(x=len(protein_df[protein_df['is_druggable']]), 
               color='red', linestyle='--', alpha=0.3, label='Druggable threshold')
    ax1.legend(bbox_to_anchor=(0, -0.15), loc='upper left', ncol=3, fontsize=8)
    
    # 2. Druggability score distribution
    score_dist = protein_df['druggability_score'].value_counts().sort_index()
    ax2.bar(score_dist.index, score_dist.values, color='steelblue')
    ax2.set_xlabel('Druggability Score')
    ax2.set_ylabel('Number of Proteins')
    ax2.set_title('Distribution of Druggability Scores')
    ax2.axvline(x=2, color='red', linestyle='--', alpha=0.5, label='Druggable threshold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    plt.suptitle(f'Druggability Analysis (n={len(protein_df)} proteins)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'druggability_overview_exclusive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save category counts
    cat_summary = pd.DataFrame({
        'Category': cat_counts.index,
        'Count': cat_counts.values,
        'Percentage': (cat_counts.values / len(protein_df) * 100).round(1)
    })
    cat_summary.to_csv(output_dir / 'druggability_categories.csv', index=False)
    print(f"    • Saved exclusive category counts to druggability_categories.csv")

def create_upset_druggability_plot(protein_df: pd.DataFrame, output_dir: Path):
    """Create UpSet plot to show overlaps between druggability categories"""
    
    if not HAVE_UPSET:
        print("    • Skipping UpSet plot (install upsetplot package)")
        return
    
    # Create category sets for each protein
    category_sets = {
        'Surface': [],
        'Secreted': [],
        'Kinase': [],
        'Transmembrane': []
    }
    
    for idx, row in protein_df.iterrows():
        if row.get('is_surface', False):
            category_sets['Surface'].append(idx)
        if row.get('is_secreted', False):
            category_sets['Secreted'].append(idx)
        if row.get('is_kinase', False):
            category_sets['Kinase'].append(idx)
        if row.get('is_transmembrane', False):
            category_sets['Transmembrane'].append(idx)
    
    # Remove empty categories
    category_sets = {k: v for k, v in category_sets.items() if len(v) > 0}
    
    if len(category_sets) == 0:
        print("    • No categories with members, skipping UpSet plot")
        return
    
    # Create UpSet data using from_contents (more robust than from_memberships)
    upset_data = from_contents(category_sets)
    
    # Create the plot
    fig = plt.figure(figsize=(12, 6))
    upset = UpSet(upset_data, 
                  intersection_plot_elements=15,
                  show_counts=True,
                  sort_by='cardinality')
    
    upset.plot(fig=fig)
    
    plt.suptitle(f'Protein Category Overlaps (n={len(protein_df)} proteins)', 
                fontsize=12, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'druggability_upset.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    • Saved UpSet plot to druggability_upset.png")
    
    # Save overlap statistics
    overlap_stats = []
    
    # Count proteins in each intersection
    for idx, row in protein_df.iterrows():
        combo = []
        if row.get('is_surface', False):
            combo.append('Surface')
        if row.get('is_secreted', False):
            combo.append('Secreted')
        if row.get('is_kinase', False):
            combo.append('Kinase')
        if row.get('is_transmembrane', False):
            combo.append('Transmembrane')
        
        if not combo:
            combo = ['None']
        
        overlap_stats.append({
            'Categories': ' + '.join(combo),
            'Gene': row['gene']
        })
    
    overlap_df = pd.DataFrame(overlap_stats)
    overlap_summary = overlap_df.groupby('Categories').size().reset_index(name='Count')
    overlap_summary['Percentage'] = (overlap_summary['Count'] / len(protein_df) * 100).round(1)
    overlap_summary = overlap_summary.sort_values('Count', ascending=False)
    
    overlap_summary.to_csv(output_dir / 'druggability_overlaps.csv', index=False)
    print(f"    • Saved overlap statistics to druggability_overlaps.csv")

def create_visualizations(protein_df: pd.DataFrame, enrichment_df: pd.DataFrame, 
                         drug_df: pd.DataFrame, output_dir: Path):
    """Create key visualizations for the analysis"""
    
    # 1. Top enriched terms by category (existing code)
    if enrichment_df is not None and not enrichment_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        categories = {
            'GO_Biological_Process': 'Biological Process',
            'GO_Molecular_Function': 'Molecular Function', 
            'KEGG': 'KEGG Pathways',
            'MSigDB_Hallmark': 'Hallmark Gene Sets'
        }
        
        for ax, (lib_pattern, title) in zip(axes.flat, categories.items()):
            subset = enrichment_df[enrichment_df['library'].str.contains(lib_pattern, na=False)].head(15)
            
            if not subset.empty:
                subset['term_short'] = subset['term'].str[:45] + '...'
                subset = subset.sort_values('combined_score')
                
                ax.barh(range(len(subset)), subset['combined_score'], color='steelblue')
                ax.set_yticks(range(len(subset)))
                ax.set_yticklabels(subset['term_short'], fontsize=8)
                ax.set_xlabel('Combined Score')
                ax.set_title(title)
                ax.grid(alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No significant terms', ha='center', va='center')
                ax.set_title(title)
        
        plt.suptitle('Top Enriched Terms in Persister Gene Signature', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'enrichment_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Drug-relevant pathways
        drug_relevant_terms = [
            'apoptosis', 'cell cycle', 'DNA damage', 'p53', 'NF-kappa',
            'JAK-STAT', 'PI3K', 'mTOR', 'oxidative', 'metabolism',
            'stem cell', 'differentiation', 'resistance'
        ]
        
        relevant_df = enrichment_df[
            enrichment_df['term'].str.contains('|'.join(drug_relevant_terms), case=False, na=False)
        ].head(20)
        
        if not relevant_df.empty:
            plt.figure(figsize=(10, 8))
            
            x = pd.to_numeric(relevant_df['combined_score'], errors='coerce').fillna(0).values
            p = pd.to_numeric(relevant_df['adjusted_p'], errors='coerce').fillna(1.0).values
            p = np.maximum(p, 1e-300)
            y = -np.log10(p)
            cvals = pd.to_numeric(relevant_df['z_score'], errors='coerce').fillna(0).values
            terms = relevant_df['term'].astype(str).str[:30].values
            
            plt.scatter(x, y, s=100, alpha=0.6, c=cvals, cmap='RdYlBu_r')
            for i, term in enumerate(terms):
                plt.annotate(term, (x[i], y[i]), fontsize=7, alpha=0.7)
            
            plt.xlabel('Combined Score')
            plt.ylabel('-log10(Adjusted P-value)')
            plt.title('Drug-Relevant Pathways in Persister Signature')
            plt.colorbar(label='Z-score')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'drug_relevant_pathways.png', dpi=300)
            plt.close()
    
    # 3. IMPROVED Druggability overview with exclusive categories and UpSet
    if not protein_df.empty:
        # Create proper exclusive categories
        create_exclusive_druggability_plot(protein_df, output_dir)
        
        # Create UpSet plot if available
        if HAVE_UPSET:
            create_upset_druggability_plot(protein_df, output_dir)
    
    print("  ✓ Visualizations created successfully")

# ===========================
# MAIN FUNCTION
# ===========================

def main():
    print("="*60)
    print("Enhanced Persister Gene → Protein → GO/Pathway Analysis")
    print("="*60)
    
    # Load genes
    print("\nLoading persister genes...")
    try:
        with open(GENES_FILE, 'r') as f:
            genes = [line.strip().upper() for line in f if line.strip()]
        
        genes = list(dict.fromkeys(genes))
        print(f"✓ Loaded {len(genes)} unique persister genes")
        print(f"  First 10 genes: {genes[:10]}")
    except Exception as e:
        print(f"✗ Error loading genes: {e}")
        return
    
    # Map to proteins
    print("\nMapping genes to proteins...")
    mapper = ProteinMapper()
    protein_df = mapper.map_genes_to_proteins(genes)
    
    if protein_df.empty:
        print("\n⚠ ERROR: No protein mappings obtained.")
        return
    
    # Save raw mapping
    protein_df.to_csv(OUTPUT_DIR / 'gene_protein_mapping_raw.csv', index=False)
    print(f"  ✓ Saved raw mapping to gene_protein_mapping_raw.csv")
    
    # Tag actionability
    protein_df = ActionabilityTagger.tag_proteins(protein_df)
    protein_df.to_csv(OUTPUT_DIR / 'gene_protein_mapping_enhanced.csv', index=False)
    print(f"  ✓ Saved enhanced mapping with actionability tags")
    
    # Run Enrichr analysis
    enrichr = EnrichrAnalyzer()
    enrichment_df = enrichr.analyze(genes)
    
    if not enrichment_df.empty:
        enrichment_df.to_csv(OUTPUT_DIR / 'enrichment_results_enhanced.csv', index=False)
        print(f"  ✓ Saved enrichment results ({len(enrichment_df)} significant terms)")
    
    # Query drug interactions
    drug_mapper = DrugTargetMapper()
    drug_df = drug_mapper.query_dgidb(genes)
    
    # Always save drug interactions CSV (even if empty)
    drug_df.to_csv(OUTPUT_DIR / 'drug_interactions.csv', index=False)
    if not drug_df.empty:
        print(f"  ✓ Saved {len(drug_df)} drug interactions")
    else:
        print(f"  ✓ Saved empty drug interactions file")
    
    # Export validation targets
    validation_df = protein_df[
        protein_df['is_surface'] | 
        protein_df['is_secreted'] | 
        protein_df['is_kinase'] |
        protein_df['is_priority']
    ].sort_values('druggability_score', ascending=False)
    
    if not validation_df.empty:
        validation_df.to_csv(OUTPUT_DIR / 'validation_targets.csv', index=False)
        print(f"  ✓ Saved {len(validation_df)} validation targets")
    
    # Create visualizations with improved druggability plots
    print("\nCreating visualizations...")
    create_visualizations(protein_df, enrichment_df, drug_df, OUTPUT_DIR)
    
    # Export for Cytoscape
    CytoscapeExporter.export_network(protein_df, enrichment_df, drug_df, OUTPUT_DIR)
    
    # Summary report
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"✓ Results saved to: {OUTPUT_DIR.absolute()}")
    print("\n📊 Key outputs:")
    print("  • gene_protein_mapping_enhanced.csv - Complete protein annotations")
    print("  • validation_targets.csv - Prioritized for wet lab")
    print("  • drug_interactions.csv - DGIdb drug mappings") 
    print("  • cytoscape_nodes.csv + cytoscape_edges.csv - Network files")
    print("  • enrichment_overview.png - Top enriched terms visualization")
    print("  • drug_relevant_pathways.png - Drug-relevant pathway analysis")
    print("  • druggability_overview_exclusive.png - Exclusive category analysis")
    print("  • druggability_categories.csv - Category counts")
    if HAVE_UPSET:
        print("  • druggability_upset.png - Category overlap visualization")
        print("  • druggability_overlaps.csv - Overlap statistics")
    
    # Show top actionable targets
    print("\n🎯 Top actionable targets:")
    if not validation_df.empty:
        top = validation_df.head(10)[['gene', 'is_surface', 'is_secreted', 
                                      'is_kinase', 'druggability_score']]
        print(top.to_string(index=False))
    
    # Highlight manuscript targets
    manuscript_targets = ['CD33', 'FLT3', 'TGFB2']
    print("\n📌 Key targets from manuscript:")
    for target in manuscript_targets:
        if target in protein_df['gene'].values:
            row = protein_df[protein_df['gene'] == target].iloc[0]
            print(f"\n  • {target}:")
            if row.get('is_surface'):
                print(f"    - Surface antigen (antibody-targetable)")
            if row.get('is_secreted'):
                print(f"    - Secreted factor (neutralizable)")
            if row.get('is_kinase'):
                print(f"    - Kinase (small molecule inhibitors)")
            
            # Check for drugs
            if not drug_df.empty and target in drug_df['gene'].values:
                drugs = drug_df[drug_df['gene'] == target]['drug'].unique()[:3]
                print(f"    - Known drugs: {', '.join(drugs)}")
    
    # Export gene list for external tools
    with open(OUTPUT_DIR / 'gene_list_for_shinygo.txt', 'w') as f:
        f.write('\n'.join(genes))
    print("\n💡 Tip: Upload 'gene_list_for_shinygo.txt' to http://bioinformatics.sdstate.edu/go/")
    print("    for interactive GO analysis with visualization")

if __name__ == "__main__":
    main()
