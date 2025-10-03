#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Pathway Analysis Pipeline for Persister Genes
Integrates KEGG pathways, PathwayCommons, DepMap dependencies, TF enrichment, 
network visualization, and drug screening
UPDATED VERSION: With all suggested fixes implemented
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Union
import re
import argparse
from datetime import datetime
from math import isfinite
import time
import urllib.parse
import platform
from io import StringIO

import numpy as np
import pandas as pd
import scipy
import requests
from scipy.stats import hypergeom, fisher_exact, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import networkx as nx

# Set matplotlib backend before importing pyplot (for HPC headless operation)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log package versions for reproducibility
logging.info(f"Python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__} | scipy {scipy.__version__}")

# ========================================================================================
# HELPER FUNCTIONS
# ========================================================================================

def _safe_get(sess, url, tries=3, timeout=10, backoff=1.5):
    """Rate-limited GET with retry logic to handle transient errors"""
    for i in range(tries):
        try:
            r = sess.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:  # Rate limited
                time.sleep(backoff ** (i + 1))
        except Exception:
            pass
        time.sleep(backoff ** i)
    return None

# ========================================================================================
# SECTION 1: KEGG PATHWAY ANALYSIS - WITH FIXED BACKGROUND SIZE
# ========================================================================================

class KEGGPathwayAnalyzer:
    """Extract and analyze KEGG pathways for gene list"""
    
    def __init__(self, gene_list: List[str], organism: str = 'hsa', background_size: int = None):
        self.genes = [g.upper() for g in gene_list]
        self.organism = organism
        self.kegg_base = "http://rest.kegg.jp"
        self.pathways = {}
        self.gene_to_pathways = {}
        self.background_size = background_size
        self.session = requests.Session()
        
        # Estimate background size from KEGG if not provided
        if self.background_size is None:
            self.background_size = self._estimate_background_size()
        
    def _estimate_background_size(self) -> int:
        """Estimate background size from KEGG human gene list - FIXED ENDPOINT"""
        logging.info("Estimating background size from KEGG...")
        try:
            # FIXED: Use correct endpoint for listing all genes in organism
            r = _safe_get(self.session, f"{self.kegg_base}/list/{self.organism}")
            if r and r.status_code == 200:
                n_genes = len(r.text.strip().split('\n'))
                logging.info(f"Found {n_genes} genes in KEGG {self.organism}")
                return n_genes
        except Exception as e:
            logging.warning(f"Could not estimate background size: {e}")
        
        # Default fallback
        logging.info("Using default background size: 19000")
        return 19000
        
    def get_kegg_pathways(self) -> Dict[str, List[str]]:
        """Extract KEGG pathways containing your genes - WITH DE-DUPLICATION AND RATE LIMITING"""
        logging.info("Fetching KEGG pathways...")
        pathways: Dict[str, Set[str]] = {}
        self.gene_to_pathways: Dict[str, Set[str]] = {}

        for gene in self.genes:
            try:
                r = _safe_get(self.session, f"{self.kegg_base}/find/{self.organism}/{gene}")
                if r is None or r.status_code != 200:
                    continue
                for line in r.text.strip().split('\n'):
                    if not line: 
                        continue
                    kegg_id = line.split('\t')[0]
                    pr = _safe_get(self.session, f"{self.kegg_base}/link/pathway/{kegg_id}")
                    if pr is None or pr.status_code != 200:
                        continue
                    for path_line in pr.text.strip().split('\n'):
                        if not path_line:
                            continue
                        pid = path_line.split('\t')[1]
                        pathways.setdefault(pid, set()).add(gene)
                        self.gene_to_pathways.setdefault(gene, set()).add(pid)
            except Exception as e:
                logging.warning(f"Error fetching KEGG data for {gene}: {e}")
                continue

        # Convert sets to sorted lists for downstream use
        self.pathways = {pid: sorted(list(gs)) for pid, gs in pathways.items()}
        self.gene_to_pathways = {g: sorted(list(p)) for g, p in self.gene_to_pathways.items()}
        
        # Filter out huge/generic pathways that aren't diagnostic
        EXCLUDE = {
            "path:hsa01100",  # Metabolic pathways
            "path:hsa05200",  # Pathways in cancer (too broad)
        }
        self.pathways = {k: v for k, v in self.pathways.items() if k not in EXCLUDE}
        
        logging.info(f"Found {len(self.pathways)} pathways after filtering")
        
        return self.pathways
    
    def _get_pathway_size(self, pathway_id: str) -> int:
        """Get actual pathway size from KEGG for proper enrichment calculation - IMPROVED"""
        try:
            r = _safe_get(self.session, f"{self.kegg_base}/link/{self.organism}/{pathway_id}")
            if r is None or r.status_code != 200:
                return 0
            # Count unique genes in the pathway
            genes = set()
            for line in r.text.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        gene_id = parts[1].strip()  # Strip whitespace for safety
                        genes.add(gene_id)
            return len(genes)
        except Exception as e:
            logging.warning(f"Size fetch failed for {pathway_id}: {e}")
        return 0
    
    def _pathway_enrichment(self, overlap: int, pathway_size: int, total_genes: int) -> Tuple[float, float]:
        """Calculate real enrichment using Fisher's exact test"""
        N = self.background_size
        K = max(1, min(pathway_size, N))
        n = total_genes
        k = overlap
        
        # Build contingency table
        a = k
        b = n - k
        c = K - k
        d = max(0, N - K - b)
        
        try:
            odds, p = fisher_exact([[a, b], [c, d]], alternative='greater')
            if not isfinite(odds): 
                odds = float('inf')
            return odds, p
        except Exception:
            # Hypergeometric survival as fallback
            p = hypergeom.sf(k-1, N, K, n)
            return 1.0, p
    
    def identify_signaling_modules(self) -> Dict[str, Dict]:
        """Identify key signaling modules from pathways - WITH FDR CORRECTION"""
        signaling_keywords = [
            'signaling', 'pathway', 'cascade', 'receptor', 'kinase',
            'phosphorylation', 'mapk', 'pi3k', 'jak-stat', 'nf-kappa',
            'wnt', 'notch', 'mtor', 'tgf', 'vegf', 'egf', 'fgf',
            'hedgehog', 'hippo', 'calcium', 'camp', 'cgmp',
            # AML-specific additions
            'flt3', 'npm1', 'aml', 'myeloid', 'leukemia'
        ]
        
        signaling_modules = {}
        p_values_list = []
        pathway_names = []
        
        for pathway_id, genes in self.pathways.items():
            try:
                info_response = _safe_get(self.session, f"{self.kegg_base}/get/{pathway_id}")
                if info_response is None or info_response.status_code != 200:
                    continue
                
                name = None
                for line in info_response.text.splitlines():
                    if line.startswith("NAME"):
                        name = re.sub(r'^\s*NAME\s+', '', line).strip()
                        break
                if not name:
                    for line in info_response.text.splitlines():
                        if line.startswith("ENTRY"):
                            name = re.sub(r'^\s*ENTRY\s+', '', line).strip()
                            break
                if not name:
                    name = pathway_id
                
                lname = name.lower()
                # Calculate enrichment for all pathways
                overlap = len(set(genes))  # After dedup
                psize = self._get_pathway_size(pathway_id)
                
                if psize == 0:  # Skip if we couldn't get pathway size
                    continue
                    
                odds, p = self._pathway_enrichment(overlap, psize, len(self.genes))
                
                # Store all pathways but mark signaling ones
                is_signaling = any(kw in lname for kw in signaling_keywords)
                
                signaling_modules[name] = {
                    'pathway_id': pathway_id,
                    'genes': sorted(list(set(genes))),
                    'count': overlap,
                    'pathway_size': psize,
                    'odds_ratio': odds,
                    'p_value': p,
                    'is_signaling': is_signaling
                }
                
                p_values_list.append(p)
                pathway_names.append(name)
                
            except Exception as e:
                logging.warning(f"Error getting pathway info for {pathway_id}: {e}")
                continue
        
        # Apply FDR correction to all pathway p-values
        if p_values_list:
            _, p_adj, _, _ = multipletests(p_values_list, method='fdr_bh')
            
            # Add adjusted p-values to results
            for i, name in enumerate(pathway_names):
                if name in signaling_modules:
                    signaling_modules[name]['p_adj_bh'] = p_adj[i]
        
        logging.info(f"Calculated enrichment for {len(signaling_modules)} pathways with FDR correction")
        
        return signaling_modules
    
    def get_pathway_crosstalk(self) -> pd.DataFrame:
        """Identify genes involved in multiple pathways (crosstalk) - FIXED VERSION"""
        rows = []
        for gene, paths in self.gene_to_pathways.items():
            uniq = sorted(set(paths))
            if len(uniq) > 1:
                rows.append({
                    'gene': gene,
                    'n_pathways': len(uniq),
                    'pathways': '; '.join(uniq[:5])
                })
        return pd.DataFrame(rows).sort_values('n_pathways', ascending=False)

# ========================================================================================
# SECTION 2: PATHWAYCOMMONS INTEGRATION - WITH PAGINATION AND BROADER SEARCH
# ========================================================================================

class PathwayCommonsAnalyzer:
    """Integrate PathwayCommons data for expanded pathway coverage - FIXED VERSION"""
    
    def __init__(self, gene_list: List[str]):
        self.genes = [g.upper() for g in gene_list]
        self.pc_base_url = "https://www.pathwaycommons.org/pc2"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "persister-pipeline/1.0"})
        self.pathways = {}
        self._pc_cache = {}  # Cache for pathway gene lookups

    def get_pathways_from_pc(self, min_overlap: int = 1) -> Dict[str, Dict]:
        """Fetch pathways from PathwayCommons for gene list with pagination support"""
        logging.info("Fetching PathwayCommons data...")
        pathways = {}
        pathway_by_name = {}  # For deduplication by normalized name

        # Query per gene without datasource filter for broader coverage
        for gene in self.genes:
            try:
                page = 1
                while True:
                    # Search without datasource filter and with pagination
                    r = self.session.get(
                        f"{self.pc_base_url}/search.json",
                        params={"q": gene, "type": "Pathway", "page": page},
                        timeout=30
                    )
                    
                    if r.status_code != 200:
                        break
                    
                    data = r.json()
                    hits = data.get("searchHit", [])
                    
                    if not hits:
                        break
                    
                    for hit in hits:
                        uri = hit.get("uri")
                        name = hit.get("name", "")
                        datasource = hit.get("dataSource", ["unknown"])[0] if hit.get("dataSource") else "unknown"
                        
                        if not uri or not name:
                            continue
                        
                        # Normalize name for deduplication
                        normalized_name = re.sub(r'\s+', ' ', name.strip().lower())
                        dedupe_key = f"{datasource}:{normalized_name}"
                        
                        # Get pathway genes (with caching)
                        if uri in self._pc_cache:
                            members = self._pc_cache[uri]
                        else:
                            members = self._get_pathway_genes(uri)
                            self._pc_cache[uri] = members
                        
                        if not members:
                            continue
                        
                        overlap = sorted(set(self.genes) & set(members))
                        
                        if len(overlap) >= min_overlap:
                            # Check if we already have this pathway
                            if dedupe_key in pathway_by_name:
                                # Keep the one with larger overlap
                                existing = pathway_by_name[dedupe_key]
                                if len(overlap) > existing["count"]:
                                    pathway_by_name[dedupe_key] = {
                                        "name": name,
                                        "source": datasource,
                                        "genes": overlap,
                                        "count": len(overlap),
                                        "total_genes": len(members),
                                        "uri": uri
                                    }
                            else:
                                pathway_by_name[dedupe_key] = {
                                    "name": name,
                                    "source": datasource,
                                    "genes": overlap,
                                    "count": len(overlap),
                                    "total_genes": len(members),
                                    "uri": uri
                                }
                    
                    page += 1
                    time.sleep(0.2)  # Rate limiting
                    
                    # Limit pages to prevent infinite loops
                    if page > 5:  # Reasonable limit for per-gene searches
                        break
                        
            except Exception as e:
                logging.warning(f"PC search failed for {gene}: {e}")
                continue

        # Convert deduplicated results to final format
        for key, info in pathway_by_name.items():
            pid = info["uri"].split("/")[-1]
            pathways[f"{info['source']}:{pid}"] = info
        
        self.pathways = pathways
        logging.info(f"Found {len(pathways)} unique pathways from PathwayCommons after deduplication")
        return pathways

    def _get_pathway_genes(self, pathway_uri: str) -> List[str]:
        """Return unique gene symbols in a pathway via /traverse - FIXED VERSION"""
        try:
            # Primary: entityReference/displayName (usually HGNC symbol)
            r = self.session.get(
                f"{self.pc_base_url}/traverse",
                params={
                    "uri": pathway_uri,
                    "path": "Pathway/components*/entityReference/displayName",
                    "format": "json"
                },
                timeout=30
            )
            
            if r.status_code == 200:
                vals = {v.upper() for v in r.json().get("values", []) if isinstance(v, str)}
                if vals:
                    return sorted(vals)
            
            # Fallback: physicalEntity/participant/entityReference/displayName
            r2 = self.session.get(
                f"{self.pc_base_url}/traverse",
                params={
                    "uri": pathway_uri,
                    "path": "Pathway/components*/physicalEntity/entityReference/displayName",
                    "format": "json"
                },
                timeout=30
            )
            
            if r2.status_code == 200:
                vals = {v.upper() for v in r2.json().get("values", []) if isinstance(v, str)}
                return sorted(vals)
                
        except Exception:
            pass
        
        return []
    
    def calculate_enrichment(self, background_size: int = 25000) -> pd.DataFrame:
        """Calculate enrichment for PathwayCommons pathways"""
        results = []
        
        for pathway_id, info in self.pathways.items():
            n_overlap = info['count']
            n_pathway = info['total_genes']
            n_genes = len(self.genes)
            
            # Fisher's exact test
            try:
                odds_ratio, p_value = fisher_exact([
                    [n_overlap, n_genes - n_overlap],
                    [n_pathway - n_overlap, background_size - n_pathway - n_genes + n_overlap]
                ])
            except:
                odds_ratio, p_value = 1.0, 1.0
            
            results.append({
                'pathway_id': pathway_id,
                'pathway_name': info['name'],
                'source': info['source'],
                'n_genes': n_overlap,
                'pathway_size': n_pathway,
                'odds_ratio': odds_ratio,
                'p_value': p_value,
                'genes': ', '.join(info['genes'][:10])
            })
        
        df = pd.DataFrame(results)
        
        # Add FDR correction
        if len(df) > 0:
            _, p_adj, _, _ = multipletests(df['p_value'].values, method='fdr_bh')
            df['p_adj_bh'] = p_adj
            df = df.sort_values('p_adj_bh')
        
        return df

# ========================================================================================
# SECTION 3: DEPMAP DEPENDENCY ANALYSIS - FIXED WITH PROPER OPERATOR PRECEDENCE
# ========================================================================================

class DepMapAnalyzer:
    """Analyze gene dependencies using DepMap CRISPR data - FULLY FIXED VERSION"""
    
    def __init__(self, gene_list: List[str]):
        self.genes = [g.upper() for g in gene_list]
        self.session = requests.Session()
        self.dependency_data = None      # rows: cell lines; cols: genes
        self.sample_info = None          # metadata table with lineage/disease

    def load_depmap_data(self, 
                        local_file: Optional[Path] = None,
                        sample_info_file: Optional[Path] = None) -> pd.DataFrame:
        """Load DepMap CRISPR dependency data with proper cell line handling"""
        
        if local_file and local_file.exists():
            logging.info(f"Loading DepMap CRISPR data from {local_file}")
            self.dependency_data = pd.read_csv(local_file)
        else:
            logging.info("No local DepMap file provided; using mock data.")
            self.dependency_data = self._generate_mock_data()
            return self.dependency_data

        # Ensure a consistent index column
        if "DepMap_ID" in self.dependency_data.columns:
            self.dependency_data.set_index("DepMap_ID", inplace=True)

        # Load sample info
        if sample_info_file and sample_info_file.exists():
            self.sample_info = pd.read_csv(sample_info_file)
            logging.info(f"Loaded sample info: {self.sample_info.shape}")
        else:
            self.sample_info = pd.DataFrame()

        return self.dependency_data
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """Generate mock dependency data for demonstration - FIXED OPERATOR PRECEDENCE"""
        logging.info("Generating mock DepMap data for demonstration...")
        
        # Create realistic cell line names
        aml_lines = {
            'ACH000001': 'TF1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000002': 'HEL9217_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000003': 'HEL_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000004': 'F36P_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000005': 'OCIM2_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000006': 'EOL1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000007': 'P31FUJ_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000008': 'KASUMI1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000009': 'OCIAML2_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000010': 'U937_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000011': 'MOLM13_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000012': 'MV411_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000013': 'OCIAML3_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000014': 'CMK_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000015': 'CMK115_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
            'ACH000016': 'MOLM16_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
        }
        
        # Generate mock dependency scores (CRISPR gene effect)
        np.random.seed(42)
        
        data = {}
        for gene in self.genes:
            scores = []
            for depmap_id, ccle_name in aml_lines.items():
                # FIXED: Added parentheses for proper operator precedence
                if (gene in ['BCL2L1', 'GATA1', 'KLF1']) and ('TF1' in ccle_name or 'HEL' in ccle_name or 'F36P' in ccle_name):
                    score = np.random.normal(-1.5, 0.3)  # Essential in erythroid
                elif (gene in ['RUNX1', 'CEBPA']) and ('MOLM' in ccle_name or 'U937' in ccle_name):
                    score = np.random.normal(-1.2, 0.3)  # Essential in other AML
                else:
                    score = np.random.normal(-0.2, 0.4)  # Not essential
                
                scores.append(score)
            
            data[gene] = scores
        
        df = pd.DataFrame(data, index=list(aml_lines.keys()))
        
        # Create mock sample info
        sample_info = []
        for depmap_id, ccle_name in aml_lines.items():
            sublineage = 'Erythroid' if any(x in ccle_name for x in ['TF1', 'HEL', 'F36P']) else \
                        'Megakaryoblastic' if any(x in ccle_name for x in ['CMK', 'MOLM16']) else \
                        'Other'
            sample_info.append({
                'DepMap_ID': depmap_id,
                'CCLE_Name': ccle_name,
                'primary_disease': 'Acute Myeloid Leukemia',
                'lineage': 'Myeloid',
                'lineage_subtype': sublineage
            })
        
        self.sample_info = pd.DataFrame(sample_info)
        
        return df

    def get_aml_dependencies(self) -> pd.DataFrame:
        """Fetch and analyze dependencies for AML cell lines - FIXED VERSION"""
        if self.dependency_data is None:
            self.load_depmap_data()

        # Determine AML lines from sample_info if available
        if isinstance(self.sample_info, pd.DataFrame) and not self.sample_info.empty:
            # DepMap columns vary slightly by release; be permissive
            cols = {c.lower(): c for c in self.sample_info.columns}
            disease_col = cols.get("disease") or cols.get("primary_disease")
            lineage_col = cols.get("lineage")
            
            if disease_col or lineage_col:
                mask = False
                if disease_col:
                    mask = self.sample_info[disease_col].str.contains(
                        "acute myeloid leukemia|AML", case=False, na=False
                    )
                if lineage_col:
                    mask = mask | self.sample_info[lineage_col].str.contains(
                        "myeloid", case=False, na=False
                    )
                
                aml_ids = set(self.sample_info.loc[mask, "DepMap_ID"])
                have = aml_ids & set(self.dependency_data.index)
                
                if have:
                    df = self.dependency_data.loc[sorted(have)]
                else:
                    df = self.dependency_data  # fallback: all lines
            else:
                df = self.dependency_data
        else:
            df = self.dependency_data

        # Keep only genes present
        present = [g for g in self.genes if g in df.columns]
        if not present:
            logging.warning("None of the input genes are present in DepMap file.")
            return pd.DataFrame()
        
        result = df[present]
        logging.info(f"Extracted dependencies for {result.shape[0]} AML lines and {result.shape[1]} genes")
        
        return result
    
    def calculate_differential_dependencies(self, 
                                           erythroid_lines: Optional[List[str]] = None,
                                           other_aml_lines: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate differential dependencies between AML subtypes - FIXED COLUMN HANDLING"""
        
        if self.dependency_data is None:
            self.get_aml_dependencies()
        
        # Determine subtypes from sample_info if not provided
        if erythroid_lines is None or other_aml_lines is None:
            if isinstance(self.sample_info, pd.DataFrame) and not self.sample_info.empty:
                # FIXED: Properly handle column name variations
                cols = {c.lower(): c for c in self.sample_info.columns}
                sub_col = cols.get('lineage_subtype') or cols.get('lineage_subtype_detail')
                
                if sub_col and sub_col in self.sample_info.columns:
                    erythroid_mask = self.sample_info[sub_col].str.contains(
                        'erythroid|erythrocyte', case=False, na=False
                    )
                    megakaryocytic_mask = self.sample_info[sub_col].str.contains(
                        'megakaryoblastic|megakaryocyte', case=False, na=False
                    )
                    
                    if erythroid_lines is None:
                        erythroid_lines = self.sample_info.loc[erythroid_mask, 'DepMap_ID'].tolist()
                    if other_aml_lines is None:
                        other_mask = ~(erythroid_mask | megakaryocytic_mask)
                        other_aml_lines = self.sample_info.loc[other_mask, 'DepMap_ID'].tolist()
                else:
                    # Fallback: use hardcoded known lines
                    logging.warning("No lineage_subtype in sample_info; using default line lists")
                    if erythroid_lines is None:
                        erythroid_lines = ['ACH000001', 'ACH000002', 'ACH000003', 'ACH000004']
                    if other_aml_lines is None:
                        other_aml_lines = ['ACH000010', 'ACH000011', 'ACH000012', 'ACH000013']
        
        # Get available lines
        avail_ery = [l for l in erythroid_lines if l in self.dependency_data.index]
        avail_other = [l for l in other_aml_lines if l in self.dependency_data.index]
        
        if not avail_ery or not avail_other:
            logging.warning("Insufficient cell lines for differential analysis")
            return pd.DataFrame()
        
        results = []
        
        for gene in self.genes:
            if gene not in self.dependency_data.columns:
                continue
            
            # Get scores for each group
            ery_scores = self.dependency_data.loc[avail_ery, gene].dropna()
            other_scores = self.dependency_data.loc[avail_other, gene].dropna()
            
            if len(ery_scores) < 2 or len(other_scores) < 2:
                continue
            
            # Calculate statistics
            ery_mean = ery_scores.mean()
            other_mean = other_scores.mean()
            diff = ery_mean - other_mean
            
            # Statistical test
            try:
                _, p_value = mannwhitneyu(ery_scores, other_scores, alternative='two-sided')
            except:
                p_value = 1.0
            
            results.append({
                'gene': gene,
                'erythroid_mean': ery_mean,
                'other_aml_mean': other_mean,
                'difference': diff,
                'p_value': p_value,
                'essential_in_erythroid': ery_mean < -0.5,
                'essential_in_other': other_mean < -0.5
            })
        
        if results:
            df = pd.DataFrame(results)
            
            # Add FDR correction
            _, p_adj, _, _ = multipletests(df['p_value'].values, method='fdr_bh')
            df['p_adj_bh'] = p_adj
            
            df = df.sort_values('p_adj_bh')
            return df
        
        return pd.DataFrame()
    
    def identify_essential_genes(self, threshold: float = -0.5) -> List[str]:
        """Identify essential genes based on CRISPR scores"""
        
        if self.dependency_data is None:
            self.get_aml_dependencies()
        
        essential = []
        
        # Get AML lines
        aml_deps = self.get_aml_dependencies()
        
        for gene in self.genes:
            if gene not in aml_deps.columns:
                continue
            
            # Check if essential in majority of AML lines
            scores = aml_deps[gene].dropna()
            
            if len(scores) > 0:
                frac_essential = (scores < threshold).sum() / len(scores)
                
                if frac_essential > 0.5:  # Essential in >50% of lines
                    essential.append(gene)
        
        return essential
    
    def plot_dependency_heatmap(self, output_file: str = "depmap_heatmap.png") -> None:
        """Create heatmap of dependencies across AML subtypes"""
        
        aml_deps = self.get_aml_dependencies()
        
        if aml_deps.empty:
            logging.warning("No data to plot")
            return
        
        # Get subtype information if available
        subtype_colors = {}
        if isinstance(self.sample_info, pd.DataFrame) and not self.sample_info.empty:
            for depmap_id in aml_deps.index:
                if depmap_id in self.sample_info['DepMap_ID'].values:
                    row = self.sample_info[self.sample_info['DepMap_ID'] == depmap_id].iloc[0]
                    subtype = row.get('lineage_subtype', 'Unknown')
                    
                    if 'erythroid' in str(subtype).lower():
                        subtype_colors[depmap_id] = 'darkred'
                    elif 'megakaryoblastic' in str(subtype).lower():
                        subtype_colors[depmap_id] = 'orange'
                    else:
                        subtype_colors[depmap_id] = 'gray'
                else:
                    subtype_colors[depmap_id] = 'gray'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(aml_deps.values, cmap='RdBu_r', vmin=-2, vmax=1, aspect='auto')
        
        # Set ticks
        ax.set_xticks(range(len(aml_deps.columns)))
        ax.set_xticklabels(aml_deps.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(aml_deps.index)))
        
        # Set y-labels with colors if available
        if subtype_colors:
            for i, idx in enumerate(aml_deps.index):
                ax.text(-0.5, i, str(idx)[:15], ha='right', va='center', 
                       color=subtype_colors.get(idx, 'black'), fontsize=8)
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels([str(idx)[:15] for idx in aml_deps.index], fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('CRISPR Gene Effect')
        
        plt.title('DepMap CRISPR Dependencies in AML Cell Lines')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Dependency heatmap saved to {output_file}")
    
    def plot_depmap_volcano(self, diff_df: pd.DataFrame, output_file: str = "depmap_volcano.png") -> None:
        """Create volcano plot for differential dependencies - NEW"""
        
        if diff_df is None or diff_df.empty:
            logging.warning("No differential data to plot")
            return
        
        df = diff_df.copy()
        df["neglog10_fdr"] = -np.log10(df["p_adj_bh"].clip(lower=1e-300))
        
        plt.figure(figsize=(7, 6))
        
        # Main scatter plot
        plt.scatter(df["difference"], df["neglog10_fdr"], s=12, alpha=0.6, color='gray')
        
        # Highlight significant genes
        sig_mask = df["p_adj_bh"] < 0.15
        plt.scatter(df.loc[sig_mask, "difference"], 
                   df.loc[sig_mask, "neglog10_fdr"], 
                   s=20, alpha=0.8, color='red')
        
        # Add FDR threshold line
        plt.axhline(-np.log10(0.15), linestyle="--", linewidth=1, color='black', alpha=0.5)
        
        # Label key genes
        genes_to_label = ["BCL2L1", "MCL1", "GATA1", "RUNX1", "CEBPA", "MYC"]
        for g in genes_to_label:
            if g in set(df["gene"]):
                r = df[df["gene"] == g].iloc[0]
                plt.annotate(g, (r["difference"], r["neglog10_fdr"]), 
                           fontsize=8, ha='center')
        
        plt.xlabel("Differential CRISPR gene effect (erythroid − other AML)")
        plt.ylabel("−log10(FDR)")
        plt.title("Differential Gene Dependencies in AML Subtypes")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Volcano plot saved to {output_file}")
    
    def plot_gene_bars(self, genes: List[str], output_file: str = "depmap_gene_bars.png") -> None:
        """Create bar plot of gene dependencies by AML subtype - NEW"""
        
        if self.dependency_data is None or self.sample_info is None or not genes:
            logging.warning("Insufficient data for bar plot")
            return
        
        cols = {c.lower(): c for c in self.sample_info.columns}
        sub_col = cols.get('lineage_subtype') or cols.get('lineage_subtype_detail')
        
        if not sub_col or sub_col not in self.sample_info.columns:
            logging.warning("No subtype information available for bar plot")
            return
        
        # Define groups
        groups = {
            'Erythroid': self.sample_info[
                self.sample_info[sub_col].str.contains('erythroid', case=False, na=False)
            ]['DepMap_ID'],
            'Megakaryoblastic': self.sample_info[
                self.sample_info[sub_col].str.contains('megakary', case=False, na=False)
            ]['DepMap_ID'],
            'Other AML': self.sample_info[
                ~self.sample_info[sub_col].str.contains('erythroid|megakary', case=False, na=False)
            ]['DepMap_ID'],
        }
        
        # Calculate means for each gene and group
        means = []
        for g in genes:
            if g not in self.dependency_data.columns:
                continue
            row = {'Gene': g}
            for k, ids in groups.items():
                ids = [i for i in ids if i in self.dependency_data.index]
                if ids:
                    row[k] = self.dependency_data.loc[ids, g].mean()
            means.append(row)
        
        if not means:
            logging.warning("No data for specified genes")
            return
        
        df = pd.DataFrame(means).set_index('Gene')
        
        # Create bar plot
        ax = df.plot(kind='bar', figsize=(7, 5), color=['darkred', 'orange', 'gray'])
        ax.set_ylabel('CRISPR gene effect (lower = more essential)')
        ax.set_xlabel('Gene')
        ax.axhline(y=-0.5, linestyle='--', color='black', alpha=0.3, label='Essential threshold')
        ax.legend(loc='best')
        plt.title('Gene Dependencies by AML Subtype')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Gene dependency bar plot saved to {output_file}")

# ========================================================================================
# SECTION 4: TRANSCRIPTION FACTOR ENRICHMENT - WITH ROBUST FETCHING AND FIX
# ========================================================================================

class TranscriptionFactorAnalyzer:
    """Analyze transcription factor regulation of gene set"""
    
    def __init__(self, gene_list: List[str]):
        self.genes = [g.upper() for g in gene_list]
        self.tf_targets = {}
        self.tf_modules = {}
        self.session = requests.Session()
        
    def load_dorothea_data(self) -> Dict[str, Set[str]]:
        """Load TF-target relationships from DoRothEA - FIXED WITH PROPER PARSING"""
        logging.info("Loading DoRothEA TF-target data...")
        
        try:
            url = "https://omnipathdb.org/interactions?datasets=dorothea&organisms=9606&fields=sources,references&genesymbols=yes"
            
            # Use retry logic for robustness
            response = _safe_get(self.session, url, tries=3, timeout=30)
            
            if response:
                # Parse TSV data properly using StringIO
                df = pd.read_csv(StringIO(response.text), sep='\t')
                
                # Filter for high confidence (A, B, C) if column exists
                if 'dorothea_level' in df.columns:
                    df = df[df['dorothea_level'].isin(['A', 'B', 'C'])]
                
                # Build TF -> targets mapping
                tf_targets = {}
                for _, row in df.iterrows():
                    tf = str(row.get('source_genesymbol', '')).upper()
                    target = str(row.get('target_genesymbol', '')).upper()
                    
                    if tf and target and tf != 'NAN' and target != 'NAN':
                        if tf not in tf_targets:
                            tf_targets[tf] = set()
                        tf_targets[tf].add(target)
                
                self.tf_targets = tf_targets
                logging.info(f"Loaded {len(tf_targets)} TFs with targets")
                return tf_targets
            else:
                raise Exception("Failed to fetch DoRothEA data after retries")
                
        except Exception as e:
            logging.warning(f"Could not load DoRothEA data: {e}")
            return self._get_fallback_tf_targets()
    
    def _get_fallback_tf_targets(self) -> Dict[str, Set[str]]:
        """Fallback TF-target relationships for key TFs"""
        return {
            'MYC': {'CDK4', 'CCND1', 'LDHA', 'ENO1', 'HK2', 'PKM', 'SLC2A1'},
            'TP53': {'CDKN1A', 'BBC3', 'PUMA', 'NOXA', 'BAX', 'FAS', 'MDM2'},
            'STAT3': {'BCL2', 'BCL2L1', 'MCL1', 'BIRC5', 'MMP9', 'VEGFA', 'IL6'},
            'NF-KB': {'IL6', 'IL8', 'TNF', 'ICAM1', 'VCAM1', 'BCL2', 'BIRC5'},
            'JUN': {'MMP1', 'MMP3', 'MMP9', 'IL8', 'VEGFA', 'CCND1'},
            'FOS': {'MMP1', 'MMP3', 'IL6', 'VEGFA', 'CCND1'},
            'CEBPA': {'IL6', 'G-CSF', 'MPO', 'ELANE', 'PRTN3', 'CTSG'},
            'RUNX1': {'MPO', 'ELANE', 'CSF2RA', 'SPI1', 'CEBPA'},
            'GATA1': {'HBB', 'HBG1', 'HBG2', 'ALAS2', 'EPOR', 'GYPA'},
        }
    
    def identify_tf_modules(self, min_targets: int = 3) -> Dict[str, Dict]:
        """Identify TF regulatory modules in your gene set with FDR correction"""
        if not self.tf_targets:
            self.load_dorothea_data()
        
        tf_modules = {}
        p_values = []
        tf_names = []
        
        for tf, all_targets in self.tf_targets.items():
            # Find overlap with your genes
            overlap = set(self.genes) & all_targets
            
            if len(overlap) >= min_targets:
                # Calculate enrichment using Fisher's exact test
                n_overlap = len(overlap)
                n_targets = len(all_targets)
                n_genes = len(self.genes)
                n_total = 25000  # Approximate total genes
                
                # Fisher's exact test
                try:
                    odds_ratio, p_value = fisher_exact([
                        [n_overlap, n_genes - n_overlap],
                        [n_targets - n_overlap, n_total - n_targets - n_genes + n_overlap]
                    ])
                except:
                    odds_ratio, p_value = 1.0, 1.0
                
                tf_modules[tf] = {
                    'targets_in_list': list(overlap),
                    'n_targets': n_overlap,
                    'enrichment': odds_ratio,
                    'p_value': p_value,
                    'fraction_of_targets': n_overlap / n_targets if n_targets > 0 else 0
                }
                
                p_values.append(p_value)
                tf_names.append(tf)
        
        # Add FDR correction
        if p_values:
            _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
            
            for i, tf in enumerate(tf_names):
                if tf in tf_modules:
                    tf_modules[tf]['p_adj_bh'] = p_adj[i]
        
        self.tf_modules = tf_modules
        return tf_modules
    
    def get_tf_hierarchy(self) -> Dict[str, List[str]]:
        """Identify TF regulatory hierarchy (TFs regulating other TFs)"""
        tf_hierarchy = {}
        
        for tf1, module in self.tf_modules.items():
            regulated_tfs = []
            for tf2 in self.tf_modules.keys():
                if tf2 != tf1 and tf2 in module['targets_in_list']:
                    regulated_tfs.append(tf2)
            
            if regulated_tfs:
                tf_hierarchy[tf1] = regulated_tfs
        
        return tf_hierarchy

# ========================================================================================
# SECTIONS 5-6: Network Visualization and Drug Integration (unchanged)
# ========================================================================================

class PathwayNetworkVisualizer:
    """Create and visualize pathway-gene networks"""
    
    def __init__(self, pathway_data: Dict, tf_modules: Dict = None):
        self.pathway_data = pathway_data
        self.tf_modules = tf_modules or {}
        self.G = None
        
    def create_pathway_gene_network(self, min_genes: int = 2, p_cutoff: float = 0.05) -> nx.Graph:
        """Create bipartite network of pathways and genes - WITH FILTERING"""
        G = nx.Graph()
        
        logging.info(f"Building network with filters: min_genes={min_genes}, FDR cutoff={p_cutoff}")
        
        for pathway, info in self.pathway_data.items():
            # Filter by both gene count AND p-value to reduce noise
            if info['count'] >= min_genes and info.get('p_adj_bh', info.get('p_value', 1)) < p_cutoff:
                # Add pathway node with enrichment metrics
                G.add_node(pathway, 
                          node_type='pathway', 
                          size=info['count'],
                          odds_ratio=info.get('odds_ratio', 1.0),
                          p_value=info.get('p_value', 1.0),
                          p_adj=info.get('p_adj_bh', info.get('p_value', 1.0)),
                          is_signaling=info.get('is_signaling', False))
                
                # Add gene nodes and edges
                for gene in info['genes']:
                    # Check if gene is a TF
                    is_tf = gene in self.tf_modules
                    
                    G.add_node(gene, 
                              node_type='gene',
                              is_tf=is_tf)
                    G.add_edge(pathway, gene)
        
        self.G = G
        logging.info(f"Created network with {G.number_of_nodes()} nodes after filtering")
        return G
    
    def detect_functional_modules(self, n_modules: int = 10) -> Dict[int, List[str]]:
        """Detect densely connected modules using community detection"""
        if self.G is None:
            self.create_pathway_gene_network()
        
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.G)
            
            # Organize by module
            modules = {}
            for node, module_id in partition.items():
                if module_id not in modules:
                    modules[module_id] = []
                modules[module_id].append(node)
            
            return modules
        except ImportError:
            logging.warning("python-louvain not found, falling back to spectral clustering")
            return self._spectral_clustering(n_modules)
        except Exception as e:
            logging.warning(f"Community detection failed: {e}, using spectral clustering")
            return self._spectral_clustering(n_modules)
    
    def _spectral_clustering(self, n_modules: int) -> Dict[int, List[str]]:
        """Fallback module detection using spectral clustering"""
        if self.G is None or len(self.G) == 0:
            return {}
        
        # Create adjacency matrix
        nodes = list(self.G.nodes())
        adj_matrix = nx.adjacency_matrix(self.G).todense()
        
        # Spectral clustering
        n_clusters = min(n_modules, len(nodes))
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        try:
            labels = clustering.fit_predict(adj_matrix)
            
            # Assign modules
            modules = {}
            for i, node in enumerate(nodes):
                module_id = labels[i]
                if module_id not in modules:
                    modules[module_id] = []
                modules[module_id].append(node)
            
            return modules
        except:
            return {0: nodes}  # All in one module if clustering fails
    
    def visualize_network_interactive(self, 
                                     drug_targets: Dict[str, List[str]] = None,
                                     output_file: str = "pathway_network.html") -> go.Figure:
        """Create interactive network visualization with Plotly - FIXED HOVER AND COLOR"""
        if self.G is None:
            self.create_pathway_gene_network()
        
        # Calculate layout
        pos = nx.spring_layout(self.G, k=3, iterations=50, seed=42)
        
        # Prepare traces
        edge_traces = []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_traces.append(
                go.Scatter(x=[x0, x1, None], 
                          y=[y0, y1, None],
                          mode='lines',
                          line=dict(width=0.5, color='rgba(125,125,125,0.3)'),
                          hoverinfo='none',
                          showlegend=False)
            )
        
        # Separate nodes by type
        pathway_nodes = []
        gene_nodes = []
        tf_nodes = []
        
        for node in self.G.nodes():
            node_data = self.G.nodes[node]
            x, y = pos[node]
            
            if node_data.get('node_type') == 'pathway':
                pathway_nodes.append((node, x, y, node_data))
            elif node_data.get('is_tf', False):
                tf_nodes.append((node, x, y, node_data))
            else:
                gene_nodes.append((node, x, y, node_data))
        
        # Create node traces
        traces = []
        
        # Pathway nodes - FIXED WITH PROPER COLOR AND HOVER
        if pathway_nodes:
            # Collect p_adj and -log10(p_adj)
            p_adj_vals = [n[3].get('p_adj', n[3].get('p_value', 1.0)) for n in pathway_nodes]
            neglog10 = [-np.log10(max(p, 1e-300)) for p in p_adj_vals]  # avoid log(0)
            
            # Size adjustment: slightly larger if signaling pathway
            sizes = []
            for n in pathway_nodes:
                base = n[3].get('size', 5)
                bump = 3 if n[3].get('is_signaling', False) else 0
                size = 15 + (base * 0.5) + bump
                sizes.append(size)
            
            labels = [n[0][:30] + '...' if len(n[0]) > 30 else n[0] for n in pathway_nodes]
            
            pathway_trace = go.Scatter(
                x=[n[1] for n in pathway_nodes],
                y=[n[2] for n in pathway_nodes],
                mode='markers+text',
                name='Pathways',
                marker=dict(
                    size=sizes,
                    color=neglog10,
                    colorscale='Viridis',
                    symbol='square',
                    showscale=True,
                    colorbar=dict(title='-log10(FDR)')
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=9),
                # customdata: [genes_in_node, p_adj, -log10(p_adj)]
                customdata=np.column_stack([
                    [n[3].get('size', 0) for n in pathway_nodes],
                    p_adj_vals,
                    neglog10
                ]),
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Genes in pathway: %{customdata[0]}<br>'
                    'FDR (BH): %{customdata[1]:.2e}<br>'
                    '-log10(FDR): %{customdata[2]:.2f}'
                    '<extra></extra>'
                )
            )
            traces.append(pathway_trace)
        
        # TF nodes
        if tf_nodes:
            tf_trace = go.Scatter(
                x=[n[1] for n in tf_nodes],
                y=[n[2] for n in tf_nodes],
                mode='markers+text',
                name='Transcription Factors',
                marker=dict(size=12, color='red', symbol='diamond'),
                text=[n[0] for n in tf_nodes],
                textposition="top center",
                textfont=dict(size=8),
                hovertemplate='<b>TF: %{text}</b><extra></extra>'
            )
            traces.append(tf_trace)
        
        # Gene nodes
        if gene_nodes:
            gene_colors = []
            gene_texts = []
            
            for node_tuple in gene_nodes:
                gene = node_tuple[0]
                color = 'lightblue'
                text = gene
                
                if drug_targets:
                    for drug, targets in drug_targets.items():
                        if gene in targets:
                            color = 'orange'
                            text = f"{gene} ({drug})"
                            break
                
                gene_colors.append(color)
                gene_texts.append(text)
            
            gene_trace = go.Scatter(
                x=[n[1] for n in gene_nodes],
                y=[n[2] for n in gene_nodes],
                mode='markers',
                name='Genes',
                marker=dict(size=8, color=gene_colors),
                text=gene_texts,
                hovertemplate='<b>%{text}</b><extra></extra>'
            )
            traces.append(gene_trace)
        
        # Create figure
        fig = go.Figure(data=edge_traces + traces)
        
        fig.update_layout(
            title="Persister Gene Pathway Network",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        # Save to HTML
        fig.write_html(output_file)
        logging.info(f"Network visualization saved to {output_file}")
        
        return fig

class DrugPathwayIntegrator:
    """Integrate drug screening data with pathway analysis"""
    
    def __init__(self, pathway_modules: Dict, tf_modules: Dict = None):
        self.pathway_modules = pathway_modules
        self.tf_modules = tf_modules or {}
        self.drug_pathway_map = self._initialize_drug_map()
        
    def _initialize_drug_map(self) -> Dict[str, Dict]:
        """Initialize known drug-target-pathway relationships"""
        return {
            # BET inhibitors
            'JQ1': {
                'targets': ['BRD4', 'BRD2', 'BRD3'],
                'indirect': ['MYC', 'MYCN'],
                'pathways': ['Transcriptional regulation', 'Cell cycle', 'Apoptosis'],
                'mechanism': 'BET inhibition → MYC downregulation'
            },
            'OTX015': {
                'targets': ['BRD2', 'BRD3', 'BRD4'],
                'indirect': ['MYC'],
                'pathways': ['Transcriptional regulation', 'Cell cycle'],
                'mechanism': 'BET inhibition'
            },
            
            # BCL2 inhibitors
            'Venetoclax': {
                'targets': ['BCL2'],
                'indirect': ['MCL1', 'BCL2L1'],
                'pathways': ['Apoptosis', 'Mitochondrial', 'p53 signaling'],
                'mechanism': 'BCL2 inhibition → apoptosis induction'
            },
            'Navitoclax': {
                'targets': ['BCL2', 'BCL2L1', 'BCL2L2'],
                'pathways': ['Apoptosis', 'Mitochondrial'],
                'mechanism': 'Pan-BCL2 inhibition'
            },
            
            # FLT3 inhibitors
            'Midostaurin': {
                'targets': ['FLT3', 'KIT', 'PDGFRA', 'PDGFRB'],
                'pathways': ['PI3K-Akt', 'MAPK', 'JAK-STAT'],
                'mechanism': 'Multi-kinase inhibition'
            },
            'Gilteritinib': {
                'targets': ['FLT3', 'AXL'],
                'pathways': ['PI3K-Akt', 'MAPK'],
                'mechanism': 'FLT3/AXL dual inhibition'
            },
            'Quizartinib': {
                'targets': ['FLT3'],
                'pathways': ['PI3K-Akt', 'MAPK'],
                'mechanism': 'Selective FLT3 inhibition'
            },
            
            # IDH inhibitors
            'Enasidenib': {
                'targets': ['IDH2'],
                'pathways': ['Metabolic', 'Epigenetic regulation'],
                'mechanism': 'IDH2 inhibition → differentiation'
            },
            'Ivosidenib': {
                'targets': ['IDH1'],
                'pathways': ['Metabolic', 'Epigenetic regulation'],
                'mechanism': 'IDH1 inhibition → differentiation'
            },
            
            # Epigenetic modulators
            'Azacitidine': {
                'targets': ['DNMT1', 'DNMT3A', 'DNMT3B'],
                'pathways': ['DNA methylation', 'Epigenetic regulation'],
                'mechanism': 'DNA hypomethylation'
            },
            'Decitabine': {
                'targets': ['DNMT1'],
                'pathways': ['DNA methylation', 'Epigenetic regulation'],
                'mechanism': 'DNA hypomethylation'
            },
            'Vorinostat': {
                'targets': ['HDAC1', 'HDAC2', 'HDAC3', 'HDAC6'],
                'pathways': ['Histone modification', 'Transcriptional regulation'],
                'mechanism': 'HDAC inhibition'
            },
            
            # MEK inhibitors
            'Trametinib': {
                'targets': ['MEK1', 'MEK2'],
                'pathways': ['MAPK signaling'],
                'mechanism': 'MEK inhibition → MAPK blockade'
            },
            'Cobimetinib': {
                'targets': ['MEK1', 'MEK2'],
                'pathways': ['MAPK signaling'],
                'mechanism': 'MEK inhibition'
            },
            
            # PI3K/mTOR inhibitors
            'BEZ235': {
                'targets': ['PI3K', 'MTOR'],
                'pathways': ['PI3K-Akt-mTOR'],
                'mechanism': 'Dual PI3K/mTOR inhibition'
            },
            'Rapamycin': {
                'targets': ['MTOR'],
                'pathways': ['mTOR signaling', 'Autophagy'],
                'mechanism': 'mTOR inhibition'
            },
            
            # CDK inhibitors
            'Palbociclib': {
                'targets': ['CDK4', 'CDK6'],
                'pathways': ['Cell cycle', 'p53 signaling'],
                'mechanism': 'CDK4/6 inhibition → G1 arrest'
            },
            'Dinaciclib': {
                'targets': ['CDK1', 'CDK2', 'CDK5', 'CDK9'],
                'pathways': ['Cell cycle', 'Transcription'],
                'mechanism': 'Pan-CDK inhibition'
            }
        }
    
    def map_drugs_to_pathways(self, drug_sensitivity_data: pd.DataFrame = None) -> pd.DataFrame:
        """Map drugs to affected pathways based on their targets"""
        drug_pathway_mappings = []
        
        for drug, drug_info in self.drug_pathway_map.items():
            # Direct targets
            for pathway_name, pathway_info in self.pathway_modules.items():
                pathway_genes = pathway_info['genes']
                
                # Check overlap with drug targets
                direct_overlap = set(drug_info['targets']) & set(pathway_genes)
                indirect_overlap = set(drug_info.get('indirect', [])) & set(pathway_genes)
                
                if direct_overlap or indirect_overlap:
                    mapping = {
                        'Drug': drug,
                        'Pathway': pathway_name[:50],  # Truncate long names
                        'Direct_targets': ', '.join(direct_overlap),
                        'Indirect_targets': ', '.join(indirect_overlap),
                        'N_targets': len(direct_overlap) + len(indirect_overlap),
                        'Mechanism': drug_info.get('mechanism', 'Unknown'),
                        'Pathway_p_adj': pathway_info.get('p_adj_bh', pathway_info.get('p_value', 1.0))
                    }
                    
                    # Add sensitivity data if available
                    if drug_sensitivity_data is not None and drug in drug_sensitivity_data.columns:
                        mapping['Mean_sensitivity'] = drug_sensitivity_data[drug].mean()
                        mapping['Responders'] = (drug_sensitivity_data[drug] > 0.5).sum()
                    
                    drug_pathway_mappings.append(mapping)
        
        if drug_pathway_mappings:
            df = pd.DataFrame(drug_pathway_mappings)
            df = df.sort_values(['Pathway_p_adj', 'N_targets'], ascending=[True, False])
            return df
        return pd.DataFrame()
    
    def identify_combination_targets(self) -> List[Dict]:
        """Identify potential drug combinations based on pathway coverage"""
        combinations = []
        drug_list = list(self.drug_pathway_map.keys())
        
        for i, drug1 in enumerate(drug_list):
            for drug2 in drug_list[i+1:]:
                # Get pathways for each drug
                pathways1 = set(self.drug_pathway_map[drug1].get('pathways', []))
                pathways2 = set(self.drug_pathway_map[drug2].get('pathways', []))
                
                # Check for synergistic potential
                if pathways1 and pathways2:
                    # Non-overlapping pathways (complementary)
                    unique = (pathways1 | pathways2) - (pathways1 & pathways2)
                    
                    if len(unique) >= 2:
                        combinations.append({
                            'Drug1': drug1,
                            'Drug2': drug2,
                            'Pathways_drug1': ', '.join(pathways1),
                            'Pathways_drug2': ', '.join(pathways2),
                            'Unique_pathways': len(unique),
                            'Synergy_type': 'Complementary' if len(pathways1 & pathways2) == 0 else 'Overlapping'
                        })
        
        return combinations

# ========================================================================================
# SECTION 7: INTEGRATED ANALYSIS PIPELINE - WITH ALL FIXES
# ========================================================================================

class IntegratedPathwayPipeline:
    """Main pipeline integrating all analyses including PathwayCommons and DepMap"""
    
    def __init__(self, gene_list_file: Path, output_dir: Path, background_size: int = None):
        self.gene_list_file = gene_list_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.background_size = background_size
        
        # Load genes
        with open(gene_list_file) as f:
            self.genes = [line.strip().upper() for line in f if line.strip()]
        
        logging.info(f"Loaded {len(self.genes)} genes for analysis")
        
        # Initialize components
        self.kegg_analyzer = None
        self.pc_analyzer = None  # PathwayCommons
        self.depmap_analyzer = None  # DepMap
        self.tf_analyzer = None
        self.network_viz = None
        self.drug_integrator = None
        
        # Results storage
        self.results = {}
    
    def run_kegg_analysis(self) -> Dict:
        """Run KEGG pathway analysis with FDR correction"""
        logging.info("\n" + "="*60)
        logging.info("RUNNING KEGG PATHWAY ANALYSIS")
        logging.info("="*60)
        
        self.kegg_analyzer = KEGGPathwayAnalyzer(self.genes, background_size=self.background_size)
        
        # Get pathways
        pathways = self.kegg_analyzer.get_kegg_pathways()
        logging.info(f"Found {len(pathways)} pathways")
        
        # Get signaling modules with enrichment and FDR
        signaling_modules = self.kegg_analyzer.identify_signaling_modules()
        logging.info(f"Identified {len(signaling_modules)} pathways with enrichment scores")
        
        # Get crosstalk
        crosstalk_df = self.kegg_analyzer.get_pathway_crosstalk()
        logging.info(f"Found {len(crosstalk_df)} genes in multiple pathways")
        
        # Save results
        signaling_df = pd.DataFrame([
            {
                'pathway': name,
                'pathway_id': info['pathway_id'],
                'n_genes': info['count'],
                'pathway_size': info.get('pathway_size', 0),
                'odds_ratio': round(info.get('odds_ratio', 1.0), 3),
                'p_value': info.get('p_value', 1.0),
                'p_adj_bh': info.get('p_adj_bh', 1.0),
                'is_signaling': info.get('is_signaling', False),
                'genes': ', '.join(info['genes'][:10])
            }
            for name, info in signaling_modules.items()
        ])
        signaling_df = signaling_df.sort_values(['p_adj_bh', 'p_value'])
        signaling_df.to_csv(self.output_dir / 'kegg_signaling_modules_with_fdr.csv', index=False)
        
        # Also save top significant pathways separately
        significant_df = signaling_df[signaling_df['p_adj_bh'] < 0.05]
        if len(significant_df) > 0:
            significant_df.to_csv(self.output_dir / 'kegg_significant_pathways_fdr05.csv', index=False)
            logging.info(f"Found {len(significant_df)} significant KEGG pathways (FDR < 0.05)")
        
        crosstalk_df.to_csv(self.output_dir / 'kegg_pathway_crosstalk.csv', index=False)
        
        self.results['kegg'] = {
            'pathways': pathways,
            'signaling_modules': signaling_modules,
            'crosstalk': crosstalk_df
        }
        
        return signaling_modules
    
    def run_pathwaycommons_analysis(self, min_overlap: int = 1) -> Dict:
        """Run PathwayCommons analysis for expanded pathway coverage - ENHANCED VERSION"""
        logging.info("\n" + "="*60)
        logging.info("RUNNING PATHWAYCOMMONS ANALYSIS")
        logging.info("="*60)
        
        self.pc_analyzer = PathwayCommonsAnalyzer(self.genes)
        
        # Get pathways from PathwayCommons with minimum overlap filter
        pc_pathways = self.pc_analyzer.get_pathways_from_pc(min_overlap=min_overlap)
        logging.info(f"Found {len(pc_pathways)} pathways from PathwayCommons")
        
        # Calculate enrichment
        background = self.background_size if self.background_size else 25000
        pc_enrichment_df = self.pc_analyzer.calculate_enrichment(background)
        
        if len(pc_enrichment_df) > 0:
            pc_enrichment_df.to_csv(self.output_dir / 'pathwaycommons_enrichment.csv', index=False)
            
            # Save significant pathways
            significant_pc = pc_enrichment_df[pc_enrichment_df['p_adj_bh'] < 0.05]
            if len(significant_pc) > 0:
                significant_pc.to_csv(self.output_dir / 'pathwaycommons_significant_fdr05.csv', index=False)
                logging.info(f"Found {len(significant_pc)} significant PathwayCommons pathways (FDR < 0.05)")
        
        self.results['pathwaycommons'] = {
            'pathways': pc_pathways,
            'enrichment': pc_enrichment_df
        }
        
        return pc_pathways
    
    def run_depmap_analysis(self, 
                           depmap_file: Optional[Path] = None,
                           sample_info_file: Optional[Path] = None,
                           key_genes: List[str] = None) -> Dict:
        """Run DepMap dependency analysis for AML cell lines - COMPLETE VERSION"""
        logging.info("\n" + "="*60)
        logging.info("RUNNING DEPMAP DEPENDENCY ANALYSIS")
        logging.info("="*60)
        
        self.depmap_analyzer = DepMapAnalyzer(self.genes)
        
        # Load DepMap data with sample info
        self.depmap_analyzer.load_depmap_data(depmap_file, sample_info_file)
        
        # Get AML dependencies
        aml_deps = self.depmap_analyzer.get_aml_dependencies()
        
        if len(aml_deps) > 0:
            aml_deps.to_csv(self.output_dir / 'depmap_aml_dependencies.csv')
            logging.info(f"Saved dependency data for {aml_deps.shape[0]} AML lines and {aml_deps.shape[1]} genes")
        
        # Calculate differential dependencies
        diff_deps = self.depmap_analyzer.calculate_differential_dependencies()
        
        if len(diff_deps) > 0:
            diff_deps.to_csv(self.output_dir / 'depmap_differential_dependencies.csv', index=False)
            
            # Create volcano plot
            self.depmap_analyzer.plot_depmap_volcano(
                diff_deps, 
                str(self.output_dir / 'depmap_volcano.png')
            )
            
            # Create bar plots for key genes
            if key_genes is None:
                key_genes = ['BCL2L1', 'MCL1', 'GATA1', 'RUNX1']
            
            # Filter to genes that exist in data
            available_key_genes = [g for g in key_genes if g in diff_deps['gene'].values]
            if available_key_genes:
                self.depmap_analyzer.plot_gene_bars(
                    available_key_genes,
                    str(self.output_dir / 'depmap_gene_bars.png')
                )
            
            # Highlight significantly different dependencies
            sig_diff = diff_deps[diff_deps['p_adj_bh'] < 0.05]
            if len(sig_diff) > 0:
                sig_diff.to_csv(self.output_dir / 'depmap_significant_differential.csv', index=False)
                logging.info(f"Found {len(sig_diff)} genes with differential dependency (FDR < 0.05)")
            
            # Write out MCL1/BCL2L1 distributions for comparison with paper
            for gene in ['MCL1', 'BCL2L1']:
                if gene in diff_deps['gene'].values:
                    gene_row = diff_deps[diff_deps['gene'] == gene]
                    with open(self.output_dir / f'{gene}_dependency_stats.txt', 'w') as f:
                        f.write(f"{gene} Dependency Analysis\n")
                        f.write("="*40 + "\n")
                        f.write(f"Erythroid mean: {gene_row['erythroid_mean'].values[0]:.3f}\n")
                        f.write(f"Other AML mean: {gene_row['other_aml_mean'].values[0]:.3f}\n")
                        f.write(f"Difference: {gene_row['difference'].values[0]:.3f}\n")
                        f.write(f"P-value: {gene_row['p_value'].values[0]:.3e}\n")
                        f.write(f"FDR: {gene_row['p_adj_bh'].values[0]:.3e}\n")
                        f.write(f"Essential in erythroid: {gene_row['essential_in_erythroid'].values[0]}\n")
                        f.write(f"Essential in other: {gene_row['essential_in_other'].values[0]}\n")
        
        # Identify essential genes
        essential = self.depmap_analyzer.identify_essential_genes(threshold=-0.5)
        
        if essential:
            essential_df = pd.DataFrame({
                'gene': essential,
                'essential_in_aml': True
            })
            essential_df.to_csv(self.output_dir / 'depmap_essential_genes.csv', index=False)
            logging.info(f"Identified {len(essential)} essential genes in AML")
        
        # Create dependency heatmap
        self.depmap_analyzer.plot_dependency_heatmap(
            str(self.output_dir / 'depmap_dependency_heatmap.png')
        )
        
        self.results['depmap'] = {
            'dependencies': aml_deps,
            'differential': diff_deps,
            'essential_genes': essential
        }
        
        return self.results['depmap']
    
    def run_integrated_pathway_analysis(self) -> Dict:
        """Integrate KEGG and PathwayCommons results - FIXED CROSS-SOURCE OVERLAP"""
        logging.info("\n" + "="*60)
        logging.info("INTEGRATING PATHWAY SOURCES")
        logging.info("="*60)
        
        # Make sure both analyses have been run
        if 'kegg' not in self.results:
            self.run_kegg_analysis()
        if 'pathwaycommons' not in self.results:
            self.run_pathwaycommons_analysis()
        
        # Combine pathway results
        combined_pathways = {}
        
        # Add KEGG pathways
        for name, info in self.results['kegg']['signaling_modules'].items():
            combined_pathways[f"KEGG:{name}"] = {
                'source': 'KEGG',
                'original_name': name,
                'genes': info['genes'],
                'count': info['count'],
                'pathway_size': info.get('pathway_size', 0),
                'odds_ratio': info.get('odds_ratio', 1.0),
                'p_value': info.get('p_value', 1.0),
                'p_adj_bh': info.get('p_adj_bh', 1.0),
                'is_signaling': info.get('is_signaling', False)
            }
        
        # Add PathwayCommons pathways
        for pathway_id, info in self.results['pathwaycommons']['pathways'].items():
            combined_pathways[f"PC:{pathway_id}"] = {
                'source': info['source'],
                'original_name': info['name'],
                'genes': info['genes'],
                'count': info['count'],
                'pathway_size': info.get('total_genes', 0),
                'odds_ratio': 1.0,  # Will be calculated from enrichment df
                'p_value': 1.0,
                'p_adj_bh': 1.0,
                'is_signaling': False
            }
        
        # Update with enrichment data from PathwayCommons
        if 'enrichment' in self.results['pathwaycommons'] and not self.results['pathwaycommons']['enrichment'].empty:
            pc_enrich = self.results['pathwaycommons']['enrichment']
            for _, row in pc_enrich.iterrows():
                pc_key = f"PC:{row['pathway_id']}"
                if pc_key in combined_pathways:
                    combined_pathways[pc_key]['odds_ratio'] = row.get('odds_ratio', 1.0)
                    combined_pathways[pc_key]['p_value'] = row.get('p_value', 1.0)
                    combined_pathways[pc_key]['p_adj_bh'] = row.get('p_adj_bh', 1.0)
        
        # Identify overlapping pathways between sources (CROSS-SOURCE ONLY)
        pathway_overlap = []
        for p1, info1 in combined_pathways.items():
            for p2, info2 in combined_pathways.items():
                if p1 < p2 and info1['source'] != info2['source']:  # Added source check
                    # Calculate Jaccard similarity
                    genes1 = set(info1['genes'])
                    genes2 = set(info2['genes'])
                    
                    if genes1 and genes2:
                        jaccard = len(genes1 & genes2) / len(genes1 | genes2)
                        
                        if jaccard > 0.5:  # >50% overlap
                            pathway_overlap.append({
                                'pathway1': info1['original_name'][:50],
                                'source1': info1['source'],
                                'pathway2': info2['original_name'][:50],
                                'source2': info2['source'],
                                'jaccard_similarity': round(jaccard, 3),
                                'shared_genes': ', '.join(list(genes1 & genes2)[:10])
                            })
        
        if pathway_overlap:
            overlap_df = pd.DataFrame(pathway_overlap)
            overlap_df.to_csv(self.output_dir / 'pathway_source_overlap.csv', index=False)
            logging.info(f"Found {len(overlap_df)} overlapping pathways between different sources")
        else:
            logging.info("No cross-source pathway overlaps found (or only one source analyzed)")
        
        self.results['integrated_pathways'] = combined_pathways
        
        return combined_pathways
    
    def run_tf_analysis(self) -> Dict:
        """Run transcription factor enrichment analysis with FDR correction - FIXED VERSION"""
        logging.info("\n" + "="*60)
        logging.info("RUNNING TF ENRICHMENT ANALYSIS")
        logging.info("="*60)
        
        self.tf_analyzer = TranscriptionFactorAnalyzer(self.genes)
        
        # Load TF data
        self.tf_analyzer.load_dorothea_data()
        
        # Identify TF modules
        tf_modules = self.tf_analyzer.identify_tf_modules(min_targets=2)  # Lower threshold for small gene sets
        logging.info(f"Identified {len(tf_modules)} TF modules")
        
        # Get TF hierarchy
        tf_hierarchy = self.tf_analyzer.get_tf_hierarchy()
        logging.info(f"Found {len(tf_hierarchy)} TFs regulating other TFs")
        
        # Save results with FDR correction - FIXED: Check if tf_modules is not empty
        if tf_modules:
            tf_df = pd.DataFrame([
                {
                    'TF': tf,
                    'n_targets': info['n_targets'],
                    'enrichment': round(info['enrichment'], 3),
                    'p_value': info['p_value'],
                    'p_adj_bh': info.get('p_adj_bh', info['p_value']),
                    'fraction_of_targets': round(info['fraction_of_targets'], 3),
                    'top_targets': ', '.join(info['targets_in_list'][:10])
                }
                for tf, info in tf_modules.items()
            ])
            tf_df = tf_df.sort_values(['p_adj_bh', 'p_value'])
            tf_df.to_csv(self.output_dir / 'tf_modules_with_fdr.csv', index=False)
            
            # Save significant TFs
            significant_tfs = tf_df[tf_df['p_adj_bh'] < 0.05]
            if len(significant_tfs) > 0:
                significant_tfs.to_csv(self.output_dir / 'significant_tfs_fdr05.csv', index=False)
                logging.info(f"Found {len(significant_tfs)} significant TFs (FDR < 0.05)")
        else:
            logging.warning("No TF modules found - creating empty results file")
            # Create empty DataFrame with correct columns
            empty_df = pd.DataFrame(columns=['TF', 'n_targets', 'enrichment', 'p_value', 'p_adj_bh', 'fraction_of_targets', 'top_targets'])
            empty_df.to_csv(self.output_dir / 'tf_modules_with_fdr.csv', index=False)
        
        # Save hierarchy
        if tf_hierarchy:
            hierarchy_df = pd.DataFrame([
                {'TF': tf, 'Regulates': ', '.join(targets)}
                for tf, targets in tf_hierarchy.items()
            ])
            hierarchy_df.to_csv(self.output_dir / 'tf_hierarchy.csv', index=False)
        
        self.results['tf'] = {
            'modules': tf_modules,
            'hierarchy': tf_hierarchy
        }
        
        return tf_modules
    
    def run_network_analysis(self, min_genes: int = 2, p_cutoff: float = 0.05) -> Dict:
        """Create and analyze pathway-gene network with improved filtering"""
        logging.info("\n" + "="*60)
        logging.info("RUNNING NETWORK ANALYSIS")
        logging.info("="*60)
        
        # Use integrated pathways if available, otherwise just KEGG
        if 'integrated_pathways' in self.results:
            pathway_data = self.results['integrated_pathways']
        elif 'kegg' in self.results:
            pathway_data = self.results['kegg']['signaling_modules']
        else:
            self.run_kegg_analysis()
            pathway_data = self.results['kegg']['signaling_modules']
        
        if 'tf' not in self.results:
            self.run_tf_analysis()
        
        self.network_viz = PathwayNetworkVisualizer(
            pathway_data,
            self.results['tf']['modules']
        )
        
        # Create network with stricter filtering to reduce noise
        G = self.network_viz.create_pathway_gene_network(min_genes=min_genes, p_cutoff=p_cutoff)
        logging.info(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        if G.number_of_nodes() == 0:
            logging.warning(f"No nodes pass filters. Relaxing criteria...")
            # Try with more relaxed criteria
            G = self.network_viz.create_pathway_gene_network(min_genes=1, p_cutoff=0.1)
            logging.info(f"Relaxed network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Detect modules
        modules = self.network_viz.detect_functional_modules(n_modules=10)
        logging.info(f"Detected {len(modules)} functional modules")
        
        # Save module assignments
        module_df = pd.DataFrame([
            {'node': node, 'module': module_id, 'node_type': G.nodes[node].get('node_type', 'unknown')}
            for module_id, nodes in modules.items()
            for node in nodes
        ])
        module_df.to_csv(self.output_dir / 'network_modules.csv', index=False)
        
        # Create visualization
        fig = self.network_viz.visualize_network_interactive(
            output_file=str(self.output_dir / 'pathway_network.html')
        )
        
        # Calculate network statistics
        if G.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G) if G.number_of_edges() > 0 else {}
            
            # Hub genes (high degree centrality)
            hub_genes = sorted([
                (node, centrality) 
                for node, centrality in degree_centrality.items()
                if G.nodes[node].get('node_type') == 'gene'
            ], key=lambda x: x[1], reverse=True)[:20]
            
            hub_df = pd.DataFrame(hub_genes, columns=['gene', 'degree_centrality'])
            hub_df['betweenness'] = [betweenness.get(g, 0) for g in hub_df['gene']]
            hub_df.to_csv(self.output_dir / 'hub_genes.csv', index=False)
        else:
            hub_genes = []
            logging.warning("Network is empty - no hub genes calculated")
        
        self.results['network'] = {
            'graph': G,
            'modules': modules,
            'hub_genes': hub_genes
        }
        
        return modules
    
    def run_drug_integration(self, drug_screening_file: Path = None) -> Dict:
        """Integrate drug screening data with pathway analysis"""
        logging.info("\n" + "="*60)
        logging.info("RUNNING DRUG-PATHWAY INTEGRATION")
        logging.info("="*60)
        
        # Use integrated pathways if available
        if 'integrated_pathways' in self.results:
            pathway_data = self.results['integrated_pathways']
        elif 'kegg' in self.results:
            pathway_data = self.results['kegg']['signaling_modules']
        else:
            self.run_kegg_analysis()
            pathway_data = self.results['kegg']['signaling_modules']
        
        if 'tf' not in self.results:
            self.run_tf_analysis()
        
        self.drug_integrator = DrugPathwayIntegrator(
            pathway_data,
            self.results['tf']['modules']
        )
        
        # Load drug screening data if available
        drug_sensitivity = None
        if drug_screening_file and drug_screening_file.exists():
            drug_sensitivity = pd.read_csv(drug_screening_file, index_col=0)
            logging.info(f"Loaded drug screening data: {drug_sensitivity.shape}")
        
        # Map drugs to pathways
        drug_pathway_df = self.drug_integrator.map_drugs_to_pathways(drug_sensitivity)
        if len(drug_pathway_df) > 0:
            drug_pathway_df.to_csv(self.output_dir / 'drug_pathway_mapping.csv', index=False)
            logging.info(f"Mapped {len(drug_pathway_df)} drug-pathway relationships")
        else:
            logging.warning("No drug-pathway relationships found")
        
        # Identify combinations
        combinations = self.drug_integrator.identify_combination_targets()
        if combinations:
            combo_df = pd.DataFrame(combinations)
            combo_df = combo_df.sort_values('Unique_pathways', ascending=False)
            combo_df.to_csv(self.output_dir / 'drug_combinations.csv', index=False)
            logging.info(f"Identified {len(combo_df)} potential drug combinations")
        
        # If we have DepMap data, integrate with drug targets
        if 'depmap' in self.results and self.results['depmap']['essential_genes']:
            essential_targets = []
            
            for drug, info in self.drug_integrator.drug_pathway_map.items():
                targets = set(info['targets']) | set(info.get('indirect', []))
                essential_overlap = targets & set(self.results['depmap']['essential_genes'])
                
                if essential_overlap:
                    essential_targets.append({
                        'Drug': drug,
                        'Essential_targets': ', '.join(essential_overlap),
                        'N_essential': len(essential_overlap),
                        'Mechanism': info.get('mechanism', 'Unknown')
                    })
            
            if essential_targets:
                essential_df = pd.DataFrame(essential_targets)
                essential_df = essential_df.sort_values('N_essential', ascending=False)
                essential_df.to_csv(self.output_dir / 'drugs_targeting_essential_genes.csv', index=False)
                logging.info(f"Found {len(essential_df)} drugs targeting essential genes")
        
        self.results['drugs'] = {
            'pathway_mapping': drug_pathway_df,
            'combinations': combinations
        }
        
        return self.results['drugs']
    
    def generate_summary_report(self) -> None:
        """Generate comprehensive summary report with all analyses"""
        logging.info("\n" + "="*60)
        logging.info("GENERATING SUMMARY REPORT")
        logging.info("="*60)
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'n_genes_analyzed': len(self.genes),
            'background_size': self.background_size if self.background_size else 'Auto-estimated',
            'analyses_performed': list(self.results.keys())
        }
        
        # KEGG results
        if 'kegg' in self.results:
            summary['kegg'] = {
                'n_pathways': len(self.results['kegg']['pathways']),
                'n_signaling_modules': len(self.results['kegg']['signaling_modules']),
                'n_significant_fdr05': sum(1 for m in self.results['kegg']['signaling_modules'].values() 
                                          if m.get('p_adj_bh', 1) < 0.05)
            }
        
        # PathwayCommons results
        if 'pathwaycommons' in self.results:
            summary['pathwaycommons'] = {
                'n_pathways': len(self.results['pathwaycommons']['pathways']),
                'n_sources': len(set(p['source'] for p in self.results['pathwaycommons']['pathways'].values()))
            }
            
            if 'enrichment' in self.results['pathwaycommons'] and not self.results['pathwaycommons']['enrichment'].empty:
                pc_df = self.results['pathwaycommons']['enrichment']
                summary['pathwaycommons']['n_significant_fdr05'] = (pc_df['p_adj_bh'] < 0.05).sum()
        
        # DepMap results
        if 'depmap' in self.results:
            summary['depmap'] = {
                'n_essential_genes': len(self.results['depmap']['essential_genes']),
                'essential_genes': self.results['depmap']['essential_genes'][:10]
            }
            
            if 'differential' in self.results['depmap'] and len(self.results['depmap']['differential']) > 0:
                diff_df = self.results['depmap']['differential']
                summary['depmap']['n_differential'] = (diff_df['p_adj_bh'] < 0.05).sum()
        
        # TF results
        if 'tf' in self.results:
            summary['tf'] = {
                'n_modules': len(self.results['tf']['modules']),
                'n_significant_fdr05': sum(1 for m in self.results['tf']['modules'].values() 
                                          if m.get('p_adj_bh', 1) < 0.05)
            }
        
        # Network results
        if 'network' in self.results:
            summary['network'] = {
                'n_modules': len(self.results['network']['modules']),
                'n_hub_genes': len(self.results['network']['hub_genes']),
                'top_hubs': [g[0] for g in self.results['network']['hub_genes'][:5]]
            }
        
        # Drug results
        if 'drugs' in self.results:
            summary['drugs'] = {
                'n_pathway_mappings': len(self.results['drugs']['pathway_mapping']) if isinstance(self.results['drugs']['pathway_mapping'], pd.DataFrame) else 0,
                'n_combinations': len(self.results['drugs']['combinations']) if self.results['drugs']['combinations'] else 0
            }
        
        # Save summary
        with open(self.output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("="*80)
        print(f"Genes analyzed: {summary['n_genes_analyzed']}")
        print(f"Background size: {summary['background_size']}")
        print(f"Analyses performed: {', '.join(summary['analyses_performed'])}")
        
        if 'kegg' in summary:
            print(f"\nKEGG Analysis:")
            print(f"  - Pathways: {summary['kegg']['n_pathways']}")
            print(f"  - Signaling modules: {summary['kegg']['n_signaling_modules']}")
            print(f"  - Significant (FDR < 0.05): {summary['kegg']['n_significant_fdr05']}")
        
        if 'pathwaycommons' in summary:
            print(f"\nPathwayCommons Analysis:")
            print(f"  - Pathways: {summary['pathwaycommons']['n_pathways']}")
            print(f"  - Data sources: {summary['pathwaycommons']['n_sources']}")
            if 'n_significant_fdr05' in summary['pathwaycommons']:
                print(f"  - Significant (FDR < 0.05): {summary['pathwaycommons']['n_significant_fdr05']}")
        
        if 'depmap' in summary:
            print(f"\nDepMap Analysis:")
            print(f"  - Essential genes: {summary['depmap']['n_essential_genes']}")
            if 'n_differential' in summary['depmap']:
                print(f"  - Differential dependencies: {summary['depmap']['n_differential']}")
            if summary['depmap']['essential_genes']:
                print(f"  - Top essential: {', '.join(summary['depmap']['essential_genes'][:5])}")
        
        if 'tf' in summary:
            print(f"\nTF Analysis:")
            print(f"  - TF modules: {summary['tf']['n_modules']}")
            print(f"  - Significant (FDR < 0.05): {summary['tf']['n_significant_fdr05']}")
        
        if 'network' in summary:
            print(f"\nNetwork Analysis:")
            print(f"  - Functional modules: {summary['network']['n_modules']}")
            print(f"  - Hub genes: {summary['network']['n_hub_genes']}")
            if summary['network']['top_hubs']:
                print(f"  - Top hubs: {', '.join(summary['network']['top_hubs'])}")
        
        if 'drugs' in summary:
            print(f"\nDrug Integration:")
            print(f"  - Drug-pathway mappings: {summary['drugs']['n_pathway_mappings']}")
            print(f"  - Potential combinations: {summary['drugs']['n_combinations']}")
        
        print(f"\nResults saved to: {self.output_dir}")

# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Comprehensive pathway analysis pipeline with PathwayCommons and DepMap integration'
    )
    parser.add_argument(
        '--genes', 
        type=Path, 
        required=True,
        help='Path to gene list file (one gene per line)'
    )
    parser.add_argument(
        '--output', 
        type=Path, 
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--background-size',
        type=int,
        default=None,
        help='Background gene set size for enrichment (default: auto-estimate from KEGG)'
    )
    parser.add_argument(
        '--min-genes',
        type=int,
        default=2,
        help='Minimum genes per pathway for network inclusion (default: 2)'
    )
    parser.add_argument(
        '--min-overlap',
        type=int,
        default=1,  # Changed default to 1
        help='Minimum overlap for PathwayCommons pathways (default: 1)'
    )
    parser.add_argument(
        '--p-cutoff',
        type=float,
        default=0.05,
        help='FDR cutoff for network visualization (default: 0.05)'
    )
    parser.add_argument(
        '--drug-screening',
        type=Path,
        help='Optional: Drug screening data CSV (samples x drugs)'
    )
    parser.add_argument(
        '--depmap-file',
        type=Path,
        help='Optional: Local DepMap CRISPR data file'
    )
    parser.add_argument(
        '--sample-info-file',
        type=Path,
        help='Optional: DepMap sample info file with lineage data'
    )
    parser.add_argument(
        '--key-genes',
        nargs='+',
        default=['BCL2L1', 'MCL1', 'GATA1', 'RUNX1'],
        help='Key genes for bar plots (default: BCL2L1 MCL1 GATA1 RUNX1)'
    )
    parser.add_argument(
        '--skip-kegg',
        action='store_true',
        help='Skip KEGG analysis (for faster testing)'
    )
    parser.add_argument(
        '--skip-pathwaycommons',
        action='store_true',
        help='Skip PathwayCommons analysis'
    )
    parser.add_argument(
        '--skip-depmap',
        action='store_true',
        help='Skip DepMap analysis'
    )
    parser.add_argument(
        '--skip-network',
        action='store_true',
        help='Skip network visualization'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IntegratedPathwayPipeline(args.genes, args.output, args.background_size)
    
    # Run analyses
    if not args.skip_kegg:
        pipeline.run_kegg_analysis()
    
    if not args.skip_pathwaycommons:
        pipeline.run_pathwaycommons_analysis(min_overlap=args.min_overlap)
    
    if not args.skip_depmap:
        pipeline.run_depmap_analysis(args.depmap_file, args.sample_info_file, args.key_genes)
    
    # Run integrated pathway analysis if we have multiple sources
    if 'kegg' in pipeline.results and 'pathwaycommons' in pipeline.results:
        pipeline.run_integrated_pathway_analysis()
    
    pipeline.run_tf_analysis()
    
    if not args.skip_network:
        pipeline.run_network_analysis(min_genes=args.min_genes, p_cutoff=args.p_cutoff)
    
    if args.drug_screening:
        pipeline.run_drug_integration(args.drug_screening)
    
    # Generate summary
    pipeline.generate_summary_report()

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) == 1:
        print("ENHANCED PATHWAY ANALYSIS PIPELINE - UPDATED VERSION")
        print("="*60)
        print("EXAMPLE USAGE:")
        print("-"*60)
        print("python pathway_analysis_updated.py \\")
        print("  --genes /path/to/selected_genes.txt \\")
        print("  --output /path/to/results/pathway_analysis \\")
        print("  --min-genes 2 \\")
        print("  --min-overlap 1 \\")
        print("  --p-cutoff 0.05 \\")
        print("  --drug-screening /path/to/drug_screening.csv \\")
        print("  --depmap-file /scratch/project_2010751/DepMap_Datasets/CRISPRGeneEffect.csv \\")
        print("  --sample-info-file /scratch/project_2010751/DepMap_Datasets/Model.csv \\")
        print("  --key-genes BCL2L1 MCL1 GATA1 RUNX1")
        print("-"*60)
        print("\nKEY UPDATES IN THIS VERSION:")
        print("✓ Fixed KEGG background size estimation endpoint")
        print("✓ PathwayCommons: Removed datasource filter for broader search")
        print("✓ PathwayCommons: Added pagination support (up to 5 pages)")
        print("✓ Fixed cross-source pathway overlap reporting")
        print("✓ Added matplotlib Agg backend for headless HPC operation")
        print("✓ Default min_overlap changed to 1 for small gene lists")
        print("-"*60)
        print("\nAvailable DepMap files in /scratch/project_2010751/DepMap_Datasets/:")
        print("  - CRISPRGeneEffect.csv (gene dependencies)")
        print("  - CRISPRGeneDependency.csv (probability scores)")
        print("  - Model.csv (cell line metadata)")
        print("  - Gene.csv (gene information)")
        print("  - OmicsExpressionProteinCodingGenesTPMLogp1.csv (expression data)")
        print("  - OmicsSomaticMutationsMatrixDamaging.csv (damaging mutations)")
        print("  - OmicsSomaticMutationsMatrixHotspot.csv (hotspot mutations)")
    else:
        main()
