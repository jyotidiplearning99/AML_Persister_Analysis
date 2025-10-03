#!/usr/bin/env python3
"""
DepMap dependency analysis for AML module genes
Final version with primary line filtering and apoptosis control
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class DepMapAnalyzer:
    """Analyze DepMap dependencies for AML modules"""
    
    def __init__(self, depmap_dir: Path, primary_only: bool = True):
        self.depmap_dir = depmap_dir
        self.primary_only = primary_only
        self.dependency = None
        self.model_info = None
        self._gene_to_col = {}
        self.is_aml_mask = None
        self.disease_col = None
        
    def pick_col(self, df: pd.DataFrame, options: List[str]) -> str:
        """Find column name from list of options (case-insensitive)"""
        cols = {c.lower(): c for c in df.columns}
        for opt in options:
            if opt.lower() in cols:
                return cols[opt.lower()]
        raise KeyError(f"None of {options} found in columns")
    
    def cohens_d(self, a: pd.Series, b: pd.Series) -> float:
        """Calculate Cohen's d effect size"""
        va, vb = a.var(ddof=1), b.var(ddof=1)
        n1, n2 = len(a), len(b)
        if n1 <= 1 or n2 <= 1:
            return 0.0
        pooled = np.sqrt(((n1-1)*va + (n2-1)*vb) / (n1+n2-2))
        if pooled == 0:
            pooled = 1.0
        return (a.mean() - b.mean()) / pooled
    
    def load_data(self):
        """Load DepMap CRISPR dependency and model info"""
        print("Loading DepMap data...")
        
        # Load CRISPR gene effect
        dep_file = self.depmap_dir / "CRISPRGeneEffect.csv"
        if not dep_file.exists():
            dep_file = self.depmap_dir / "CRISPR_gene_effect.csv"
        
        self.dependency = pd.read_csv(dep_file, low_memory=False)
        
        # Set index
        try:
            depmap_col = self.pick_col(self.dependency, ["DepMap_ID", "depmap_id", "ModelID"])
            self.dependency = self.dependency.set_index(depmap_col)
        except:
            self.dependency = self.dependency.set_index(self.dependency.columns[0])
        
        # Create clean gene mapping
        clean_genes = self.dependency.columns.str.replace(r"\s*\(\d+\)$", "", regex=True)
        self._gene_to_col = dict(zip(clean_genes.str.upper(), self.dependency.columns))
        
        # Load model info
        model_file = self.depmap_dir / "Model.csv"
        if not model_file.exists():
            model_file = self.depmap_dir / "sample_info.csv"
        
        self.model_info = pd.read_csv(model_file, low_memory=False)
        
        # Set index
        try:
            model_id_col = self.pick_col(self.model_info, ["ModelID", "DepMap_ID", "depmap_id"])
            self.model_info = self.model_info.set_index(model_id_col)
        except:
            self.model_info = self.model_info.set_index(self.model_info.columns[0])
        
        # Filter to primary lines if requested
        if self.primary_only:
            try:
                site_col = self.pick_col(self.model_info, 
                    ["sample_collection_site", "SampleCollectionSite", "collection_site"])
                primary_mask = self.model_info[site_col].str.contains("primary", case=False, na=False)
                keep_idx = self.model_info.index[primary_mask]
                self.model_info = self.model_info.loc[keep_idx]
                self.dependency = self.dependency.loc[self.dependency.index.intersection(keep_idx)]
                print(f"Filtered to {len(keep_idx)} primary cell lines")
            except:
                print("Could not filter to primary lines - using all lines")
        
        # Identify disease column
        self.disease_col = self.pick_col(self.model_info, 
            ["OncotreePrimaryDisease", "primary_disease", "disease", "Oncotree Primary Disease"])
        
        # Create stable AML mask
        self.model_info[self.disease_col] = self.model_info[self.disease_col].astype(str)
        self.is_aml_mask = self.model_info[self.disease_col].str.contains(
            r'Acute Myeloid Leukemia|AML|Myeloid', case=False, na=False
        )
        
        # Align dependency to model info
        common_idx = self.dependency.index.intersection(self.model_info.index)
        self.dependency = self.dependency.loc[common_idx]
        self.model_info = self.model_info.loc[common_idx]
        self.is_aml_mask = self.is_aml_mask.loc[common_idx]
        
        print(f"Final dataset: {self.dependency.shape[0]} cell lines × {self.dependency.shape[1]} genes")
        print(f"AML/Myeloid cell lines: {self.is_aml_mask.sum()}")
    
    def get_gene_dependency(self, gene: str) -> Optional[pd.Series]:
        """Get dependency scores for a gene with alias handling"""
        gene_upper = gene.upper()
        
        # Direct match
        if gene_upper in self._gene_to_col:
            return self.dependency[self._gene_to_col[gene_upper]]
        
        # Common aliases
        aliases = {
            'ERBB1': 'EGFR',
            'ERBB2': 'HER2', 
            'CD274': 'PDL1',
            'PDCD1': 'PD1',
            'BCL2L1': 'BCLXL',
            'HER2': 'ERBB2',
            'PDL1': 'CD274',
            'PD1':  'PDCD1',
            'BCLXL': 'BCL2L1'
        }
        
        if gene_upper in aliases and aliases[gene_upper] in self._gene_to_col:
            return self.dependency[self._gene_to_col[aliases[gene_upper]]]
        
        print(f"  Warning: {gene} not found in DepMap")
        return None
    
    def analyze_module_dependencies(self, genes: List[str], module_name: str) -> pd.DataFrame:
        """Analyze dependencies with FDR correction and Cohen's d"""
        results = []
        
        for gene in genes:
            dep_scores = self.get_gene_dependency(gene)
            if dep_scores is None:
                continue
            
            # Use pre-computed AML mask
            aml_deps = dep_scores[self.is_aml_mask]
            other_deps = dep_scores[~self.is_aml_mask]
            
            if len(aml_deps) > 0 and len(other_deps) > 0:
                # Welch's t-test
                t_stat, p_val = stats.ttest_ind(aml_deps, other_deps, equal_var=False)
                
                # Effect size
                d_val = self.cohens_d(aml_deps, other_deps)
                
                # Metrics
                selectivity = aml_deps.mean() - other_deps.mean()
                aml_dependent_pct = (aml_deps < -0.5).mean() * 100
                
                results.append({
                    'gene': gene,
                    'module': module_name,
                    'aml_mean': aml_deps.mean(),
                    'other_mean': other_deps.mean(),
                    'selectivity': selectivity,
                    'cohens_d': d_val,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'aml_dependent_pct': aml_dependent_pct,
                    'n_aml': len(aml_deps),
                    'n_other': len(other_deps)
                })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Add FDR correction
        if len(df) > 1:
            df['q_value'] = multipletests(df['p_value'], method='fdr_bh')[1]
        else:
            df['q_value'] = df['p_value']
        
        # Mark selective
        df['is_selective'] = (df['selectivity'] < -0.3) & (df['q_value'] < 0.1)
        
        return df.sort_values('selectivity')
    
    def create_module_heatmap(self, genes: List[str], module_name: str, save_path: Path = None):
        """Create heatmap with consistent AML columns and dependency markers"""
        
        # Get consistent AML columns
        aml_lines = self.model_info.index[self.is_aml_mask].tolist()
        
        if not aml_lines:
            print(f"No AML lines found for {module_name}")
            return None
        
        # Build matrix
        mat = pd.DataFrame(index=[], columns=aml_lines, dtype=float)
        
        for gene in genes:
            dep_scores = self.get_gene_dependency(gene)
            if dep_scores is None:
                continue
            gene_scores = dep_scores.reindex(aml_lines)
            mat.loc[gene] = gene_scores.values
        
        if mat.empty:
            print(f"No dependency data for {module_name}")
            return None
        
        # Drop all-NaN columns
        mat = mat.dropna(axis=1, how='all')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(min(14, mat.shape[1]*0.4), max(6, len(mat.index)*0.35)))
        
        # Base heatmap
        im = ax.imshow(mat.values, cmap='RdBu_r', aspect='auto', 
                      vmin=-2, vmax=1, interpolation='nearest')
        
        # Mark dependent cells
        y_idx, x_idx = np.where(mat.values < -0.5)
        if len(y_idx) > 0:
            ax.scatter(x_idx, y_idx, s=20, color='black', alpha=0.6, 
                      marker='o', label='Dependent')
        
        # Labels
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=90, ha='center', fontsize=8)
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(mat.index, fontsize=10)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('CERES Score', rotation=270, labelpad=15)
        
        # Grid
        ax.set_xticks(np.arange(mat.shape[1]+1)-0.5, minor=True)
        ax.set_yticks(np.arange(mat.shape[0]+1)-0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
        
        plt.title(f'{module_name} Dependencies in AML Cell Lines\n• = Dependent (CERES<-0.5)', 
                 fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return mat

def main():
    """Run DepMap analysis with biological modules including apoptosis control"""
    
    # Modules including apoptosis
    modules = {
        'RTK_signaling': ['EGFR', 'MET', 'KDR', 'AREG', 'CSF1', 'MDK'],
        'Hub_genes': ['EGFR', 'MET', 'MYC', 'CD44', 'CTNNB1', 'WNT4', 'WNT7B',
                      'EPHA2', 'EPHA4', 'KLF4', 'SOX9', 'YAP1'],
        'Cell_adhesion': ['CD44', 'EPHA2', 'EPHA4', 'EPHB2', 'EPHB3', 'EPHB4',
                         'FERMT1', 'FERMT2', 'DDR1', 'CELSR1', 'SDC4', 'TNC'],
        'WNT_core': ['CTNNB1', 'WNT4', 'WNT7B', 'FZD6', 'LRP5', 'LGR4', 'PRICKLE1'],
        'Apoptosis_BCL2': ['MCL1', 'BCL2', 'BCL2L1', 'BCL2A1', 'BAX', 'BID']
    }
    
    # Initialize with primary line filtering
    analyzer = DepMapAnalyzer(
        depmap_dir=Path("/scratch/project_2010751/DepMap_Datasets"),
        primary_only=False  # Filter to primary lines for better patient relevance
    )
    
    # Load data
    analyzer.load_data()
    
    # Create output directory
    output_dir = Path("./depmap_aml_primary_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Analyze all modules
    all_results = []
    for module_name, genes in modules.items():
        print(f"\nAnalyzing {module_name}...")
        df = analyzer.analyze_module_dependencies(genes, module_name)
        if not df.empty:
            all_results.append(df)
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        
        # Save full results
        combined.to_csv(output_dir / 'all_dependencies_primary_lines.csv', index=False)
        
        # Create heatmaps for key modules
        for module_name in ['RTK_signaling', 'Apoptosis_BCL2', 'Hub_genes']:
            if module_name in modules:
                print(f"\nCreating heatmap for {module_name}...")
                analyzer.create_module_heatmap(
                    modules[module_name],
                    module_name,
                    output_dir / f'{module_name}_heatmap.png'
                )
        
        # Print summary focusing on apoptosis genes
        print("\n" + "="*60)
        print("APOPTOSIS MODULE ANALYSIS (MCL1 focus)")
        print("="*60)
        
        apoptosis_results = combined[combined['module'] == 'Apoptosis_BCL2']
        if not apoptosis_results.empty:
            for _, row in apoptosis_results.iterrows():
                print(f"\n{row['gene']}")
                print(f"  AML dependency:  {row['aml_mean']:.3f}")
                print(f"  Other cancers:   {row['other_mean']:.3f}")
                print(f"  Selectivity:     {row['selectivity']:.3f}")
                print(f"  % AML dependent: {row['aml_dependent_pct']:.1f}%")
                print(f"  q-value:         {row['q_value']:.3e}")
        
        # Top selective dependencies
        print("\n" + "="*60)
        print("TOP AML-SELECTIVE DEPENDENCIES (Primary Lines)")
        print("="*60)
        
        selective = combined[combined['is_selective']].head(10)
        for _, row in selective.iterrows():
            print(f"\n{row['gene']} ({row['module']})")
            print(f"  Selectivity: {row['selectivity']:.3f} (q={row['q_value']:.3e})")
            print(f"  % AML dependent: {row['aml_dependent_pct']:.1f}%")
    
    print(f"\n✓ Analysis complete. Results in {output_dir}/")
    
    return combined if all_results else None

if __name__ == "__main__":
    results = main()
