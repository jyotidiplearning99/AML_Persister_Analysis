#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Pathway Analysis Pipeline for Persister Genes
Integrates KEGG pathways, TF enrichment, network visualization, and drug screening
FULLY FIXED VERSION: Includes all patches, FDR correction, proper visualization, and rate limiting
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import re
import argparse
from datetime import datetime
from math import isfinite
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import hypergeom, fisher_exact
from statsmodels.stats.multitest import multipletests
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
# SECTION 1: KEGG PATHWAY ANALYSIS - FULLY FIXED VERSION
# ========================================================================================

class KEGGPathwayAnalyzer:
    """Extract and analyze KEGG pathways for gene list"""
    
    def __init__(self, gene_list: List[str], organism: str = 'hsa', background_size: int = 19000):
        self.genes = [g.upper() for g in gene_list]
        self.organism = organism
        self.kegg_base = "http://rest.kegg.jp"
        self.pathways = {}
        self.gene_to_pathways = {}
        self.background_size = background_size
        self.session = requests.Session()  # Reusable session for connection pooling
        
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
# SECTION 2: TRANSCRIPTION FACTOR ENRICHMENT - WITH FDR CORRECTION
# ========================================================================================

class TranscriptionFactorAnalyzer:
    """Analyze transcription factor regulation of gene set"""
    
    def __init__(self, gene_list: List[str]):
        self.genes = [g.upper() for g in gene_list]
        self.tf_targets = {}
        self.tf_modules = {}
        
    def load_dorothea_data(self) -> Dict[str, Set[str]]:
        """Load TF-target relationships from DoRothEA"""
        logging.info("Loading DoRothEA TF-target data...")
        
        try:
            url = "https://omnipathdb.org/interactions?datasets=dorothea&organisms=9606&fields=sources,references&genesymbols=yes"
            df = pd.read_csv(url, sep='\t')
            
            # Filter for high confidence (A, B, C)
            if 'dorothea_level' in df.columns:
                df = df[df['dorothea_level'].isin(['A', 'B', 'C'])]
            
            # Build TF -> targets mapping
            tf_targets = {}
            for _, row in df.iterrows():
                tf = str(row.get('source_genesymbol', '')).upper()
                target = str(row.get('target_genesymbol', '')).upper()
                
                if tf and target:
                    if tf not in tf_targets:
                        tf_targets[tf] = set()
                    tf_targets[tf].add(target)
            
            self.tf_targets = tf_targets
            logging.info(f"Loaded {len(tf_targets)} TFs with targets")
            return tf_targets
            
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
# SECTION 3: NETWORK VISUALIZATION - WITH IMPROVED FILTERING AND HOVER
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

# ========================================================================================
# SECTION 4: DRUG-PATHWAY INTEGRATION
# ========================================================================================

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
# SECTION 5: INTEGRATED ANALYSIS PIPELINE
# ========================================================================================

class IntegratedPathwayPipeline:
    """Main pipeline integrating all analyses"""
    
    def __init__(self, gene_list_file: Path, output_dir: Path, background_size: int = 19000):
        self.gene_list_file = gene_list_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.background_size = background_size
        
        # Load genes
        with open(gene_list_file) as f:
            self.genes = [line.strip().upper() for line in f if line.strip()]
        
        logging.info(f"Loaded {len(self.genes)} genes for analysis")
        logging.info(f"Using background size: {background_size} genes")
        
        # Initialize components
        self.kegg_analyzer = None
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
        
        # Save results with proper enrichment metrics and FDR
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
        signaling_df.to_csv(self.output_dir / 'signaling_modules_with_fdr.csv', index=False)
        
        # Also save top significant pathways separately
        significant_df = signaling_df[signaling_df['p_adj_bh'] < 0.05]
        if len(significant_df) > 0:
            significant_df.to_csv(self.output_dir / 'significant_pathways_fdr05.csv', index=False)
            logging.info(f"Found {len(significant_df)} significant pathways (FDR < 0.05)")
        
        crosstalk_df.to_csv(self.output_dir / 'pathway_crosstalk.csv', index=False)
        
        self.results['kegg'] = {
            'pathways': pathways,
            'signaling_modules': signaling_modules,
            'crosstalk': crosstalk_df
        }
        
        return signaling_modules
    
    def run_tf_analysis(self) -> Dict:
        """Run transcription factor enrichment analysis with FDR correction"""
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
        
        # Save results with FDR correction
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
        
        if not self.results.get('kegg'):
            self.run_kegg_analysis()
        if not self.results.get('tf'):
            self.run_tf_analysis()
        
        self.network_viz = PathwayNetworkVisualizer(
            self.results['kegg']['signaling_modules'],
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
        
        if not self.results.get('kegg'):
            self.run_kegg_analysis()
        if not self.results.get('tf'):
            self.run_tf_analysis()
        
        self.drug_integrator = DrugPathwayIntegrator(
            self.results['kegg']['signaling_modules'],
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
        
        self.results['drugs'] = {
            'pathway_mapping': drug_pathway_df,
            'combinations': combinations
        }
        
        return self.results['drugs']
    
    def generate_summary_report(self) -> None:
        """Generate comprehensive summary report with FDR-corrected statistics"""
        logging.info("\n" + "="*60)
        logging.info("GENERATING SUMMARY REPORT")
        logging.info("="*60)
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'n_genes_analyzed': len(self.genes),
            'background_size': self.background_size,
            'kegg_pathways': len(self.results['kegg']['pathways']) if 'kegg' in self.results else 0,
            'signaling_modules': len(self.results['kegg']['signaling_modules']) if 'kegg' in self.results else 0,
            'tf_modules': len(self.results['tf']['modules']) if 'tf' in self.results else 0,
            'network_modules': len(self.results['network']['modules']) if 'network' in self.results else 0,
        }
        
        # Count significant findings
        if 'kegg' in self.results:
            sig_pathways = sum(1 for m in self.results['kegg']['signaling_modules'].values() 
                             if m.get('p_adj_bh', 1) < 0.05)
            summary['significant_pathways_fdr05'] = sig_pathways
            
            top_pathways = sorted(
                self.results['kegg']['signaling_modules'].items(),
                key=lambda x: x[1].get('p_adj_bh', x[1].get('p_value', 1)),
            )[:10]
            summary['top_pathways'] = [
                {
                    'name': name, 
                    'n_genes': info['count'],
                    'odds_ratio': round(info.get('odds_ratio', 1.0), 2),
                    'p_value': info.get('p_value', 1.0),
                    'p_adj_bh': info.get('p_adj_bh', 1.0)
                }
                for name, info in top_pathways
            ]
        
        if 'tf' in self.results:
            sig_tfs = sum(1 for m in self.results['tf']['modules'].values() 
                        if m.get('p_adj_bh', 1) < 0.05)
            summary['significant_tfs_fdr05'] = sig_tfs
            
            top_tfs = sorted(
                self.results['tf']['modules'].items(),
                key=lambda x: x[1].get('p_adj_bh', x[1]['p_value'])
            )[:10]
            summary['top_tfs'] = [
                {
                    'tf': tf, 
                    'n_targets': info['n_targets'],
                    'p_value': info['p_value'],
                    'p_adj_bh': info.get('p_adj_bh', info['p_value'])
                }
                for tf, info in top_tfs
            ]
        
        if 'network' in self.results:
            summary['hub_genes'] = [
                {'gene': gene, 'centrality': round(centrality, 3)}
                for gene, centrality in self.results['network']['hub_genes'][:10]
            ]
        
        # Save summary
        with open(self.output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("="*80)
        print(f"Genes analyzed: {summary['n_genes_analyzed']}")
        print(f"Background size: {summary['background_size']}")
        print(f"KEGG pathways: {summary['kegg_pathways']}")
        print(f"Signaling modules: {summary['signaling_modules']}")
        print(f"  - Significant (FDR < 0.05): {summary.get('significant_pathways_fdr05', 0)}")
        print(f"TF modules: {summary['tf_modules']}")
        print(f"  - Significant (FDR < 0.05): {summary.get('significant_tfs_fdr05', 0)}")
        print(f"Network modules: {summary['network_modules']}")
        
        if 'top_pathways' in summary:
            print("\nTop 5 Pathways (by FDR):")
            for p in summary['top_pathways'][:5]:
                print(f"  - {p['name'][:50]}: {p['n_genes']} genes")
                print(f"    OR={p['odds_ratio']}, p={p['p_value']:.3e}, FDR={p['p_adj_bh']:.3e}")
        
        if 'top_tfs' in summary:
            print("\nTop 5 Transcription Factors (by FDR):")
            for t in summary['top_tfs'][:5]:
                print(f"  - {t['tf']}: {t['n_targets']} targets")
                print(f"    p={t['p_value']:.3e}, FDR={t['p_adj_bh']:.3e}")
        
        if 'hub_genes' in summary:
            print("\nTop 5 Hub Genes:")
            for h in summary['hub_genes'][:5]:
                print(f"  - {h['gene']}: centrality={h['centrality']}")
        
        print(f"\nResults saved to: {self.output_dir}")

# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Comprehensive pathway analysis pipeline for persister genes with FDR correction'
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
        default=19000,
        help='Background gene set size for enrichment (default: 19000)'
    )
    parser.add_argument(
        '--min-genes',
        type=int,
        default=2,
        help='Minimum genes per pathway for network inclusion (default: 2)'
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
        '--skip-kegg',
        action='store_true',
        help='Skip KEGG analysis (for faster testing)'
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
        print("EXAMPLE USAGE:")
        print("-"*60)
        print("python pathway_analysis_complete_fixed.py \\")
        print("  --genes /path/to/selected_genes.txt \\")
        print("  --output /path/to/results/pathway_analysis \\")
        print("  --background-size 19000 \\")
        print("  --min-genes 2 \\")
        print("  --p-cutoff 0.05 \\")
        print("  --drug-screening /path/to/drug_screening.csv")
        print("-"*60)
        print("\nFor testing with default paths:")
        print("python pathway_analysis_complete_fixed.py \\")
        print("  --genes /scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled/selected_genes.txt \\")
        print("  --output /scratch/project_2010376/JDs_Project/AML_Persister_Analysis/pathway_analysis_results_fixed")
    else:
        main()
