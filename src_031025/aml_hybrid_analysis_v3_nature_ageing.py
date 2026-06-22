"""
AML Hybrid State Analysis — v3 cycling/DNA-repair + Nature ageing modules
========================================================
Tests whether therapy-exposed AML cells show co-elevation of
senescence/ageing-like and stemness programs at single-cell resolution.

Datasets:
  - GSE146590 : AML1566 Control vs AraC-treated (Duy et al.)
  - GSE116256 : van Galen AML atlas (diagnosis vs post-treatment)

Usage:
  python aml_hybrid_analysis.py \
      --gse146590_ctrl  /path/to/AML1566_Ctrl/  \
      --gse146590_arac  /path/to/AML1566_AraC/  \
      --gse116256_h5ad  /path/to/van_galen.h5ad   \
      --output_dir      ./results/

Each path should be a 10x MTX folder containing:
  barcodes.tsv.gz  features.tsv.gz  matrix.mtx.gz
OR pass a single .h5ad file with --gse116256_h5ad.
"""

import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.mixture import GaussianMixture
import scipy.sparse as sp

sc.settings.verbosity = 1


def sanitize_adata(adata):
    """Make gene symbols uppercase and unique for robust marker matching."""
    adata.var_names = pd.Index([str(g).upper() for g in adata.var_names])
    adata.var_names_make_unique()
    return adata


def row_mean(mat, cols, n_rows):
    """Memory-safe mean expression for dense or sparse matrices."""
    if not cols:
        return np.zeros(n_rows)
    vals = mat[:, cols].mean(axis=1)
    return np.asarray(vals).ravel()

# ─────────────────────────────────────────────
# GENE MODULE DEFINITIONS
# ─────────────────────────────────────────────

MODULES = {

    # Senescence-associated cell cycle arrest
    'senescence_arrest': [
        'CDKN1A',  # p21 — canonical senescence arrest
        'CDKN2A',  # p16 — CDK4/6 inhibitor
        'TP53',    # tumour suppressor
        'RB1',     # retinoblastoma — G1 arrest
        'GADD45A', # DNA damage response / growth arrest
        'GADD45B', # stress-induced growth arrest
    ],

    # SASP — secretory component (keep separate from arrest)
    'sasp': [
        'CXCL8',  # IL-8 — canonical SASP cytokine
        'IL1B',   # IL-1β — SASP inflammasome
        'IL6',    # IL-6 — SASP cytokine
        'TNF',    # TNF-α
        'CXCL1',  # GRO-α
        'CXCL2',  # GRO-β
        'MMP1',   # matrix metalloprotease
        'MMP9',   # matrix metalloprotease
        'CCL2',   # MCP-1
        'CCL5',   # RANTES
    ],

    # LSC / stemness
    'stemness': [
        'CD34',   # HSC/progenitor marker
        'HOXA9',  # LSC transcription factor
        'MEIS1',  # HOXA9 co-factor
        'DNMT3B', # de novo methyltransferase — stem cell
        'FLT3',   # AML receptor tyrosine kinase
        'RUNX1',  # haematopoietic TF
        'GATA2',  # early haematopoietic TF
        'ERG',    # ETS TF — LSC maintenance
    ],

    # True cycling module. Keep this strict so it does not mix cycling
    # with DNA replication/repair stress.
    'cycling': [
        'MKI67',  # Ki-67 — active cycling
        'TOP2A',  # topoisomerase II alpha — S/G2/M cycling
    ],

    # DNA repair / replication-checkpoint module. These genes can rise after
    # chemotherapy because of DNA-repair/replication-stress response, not
    # necessarily because cells are simply proliferating faster.
    'dna_repair': [
        'PCNA',   # DNA repair/replication clamp
        'MCM2',   # replication licensing / replication stress
        'CDK1',   # G2/M checkpoint and repair-linked cell-cycle control
        'CHEK1',  # checkpoint kinase 1
    ],

    # DNA damage checkpoint / stress response, kept separate from repair genes.
    'dna_damage': [
        'H2AFX',  # γH2AX — DSB marker
        'GADD45A',
        'ATM',    # ATM kinase
        'CHEK2',  # checkpoint kinase 2
    ],

    # Mitochondrial stress / apoptosis resistance
    'mito_stress': [
        'MCL1',   # anti-apoptotic BCL2 family — therapy resistance
        'BCL2',   # anti-apoptotic
        'BAX',    # pro-apoptotic
        'VDAC1',  # mitochondrial membrane
        'SOD2',   # mitochondrial ROS scavenger
    ],

    # Quiescence / dormancy
    'quiescence': [
        'EGFR',   # quiescence signalling
        'ANGPT1', # HSC quiescence
        'THBS1',  # thrombospondin — dormancy
        'NR4A1',  # nuclear receptor — quiescence
        'CDKN1B', # p27 — quiescence arrest
    ],


    # Conserved mammalian ageing signatures from Tyshkovskiy et al., Nature 2026,
    # Supplementary Table 2, sheets O-R. Extraction rule: significant in mouse,
    # rat, macaque and human ageing multi-tissue signatures (FDR < 0.05) with
    # the same slope direction in all four species; top 50 ranked by mean
    # within-species significance rank. Symbols are uppercased for AML scRNA-seq.
    'ageing_up': [
        'ICAM1',
        'BHLHE40',
        'PARP3',
        'SERPING1',
        'CD44',
        'STAT3',
        'ZNFX1',
        'VSIG4',
        'GPNMB',
        'C1QC',
        'TNFRSF1B',
        'MGST1',
        'TNFRSF1A',
        'IL10RA',
        'NPC2',
        'CDKN1A',
        'AKNA',
        'LEPROT',
        'LAMP1',
        'CD74',
        'MVP',
        'SQSTM1',
        'PSAP',
        'GAS6',
        'C1QB',
        'FGR',
        'MYOF',
        'NINJ1',
        'LITAF',
        'TNIP1',
        'LCP1',
        'CTSS',
        'NLRP3',
        'TLR4',
        'CSF1',
        'PHF1',
        'TMBIM1',
        'SLC15A3',
        'CRIM1',
        'ABHD14B',
        'ANXA11',
        'MSN',
        'STOM',
        'NFE2L1',
        'PLXNB2',
        'C1QA',
        'RRAS',
        'IFNGR1',
        'CBX7',
        'CTSD',
    ],

    'ageing_down': [
        'NREP',
        'COL5A1',
        'SMC4',
        'HACD1',
        'STRBP',
        'NETO2',
        'SMC6',
        'EZH2',
        'HDAC2',
        'TIPRL',
        'TIMM21',
        'BCS1L',
        'GNPAT',
        'LYRM7',
        'GPCPD1',
        'MPC2',
        'HIKESHI',
        'COL5A2',
        'MKKS',
        'GTF3C3',
        'C1QTNF6',
        'POGLUT2',
        'FAHD1',
        'RAD17',
        'CIBAR1',
        'CDC16',
        'GPX7',
        'C1QTNF3',
        'FASTKD2',
        'METAP1',
        'COL27A1',
        'MRPS14',
        'GPAA1',
        'LEO1',
        'SLC25A40',
        'THAP1',
        'ZBTB26',
        'RAD51C',
        'SARS2',
        'NR2C1',
        'AMACR',
        'WDR18',
        'MUS81',
        'MRPL15',
        'ZDHHC16',
        'SPATS2L',
        'DNAJB11',
        'AGPAT5',
        'SPARC',
        'KIF22',
    ],

    # Normal myeloid (used for malignant cell gating)
    'normal_myeloid': [
        'S100A8', 'S100A9', 'CD14',   # monocyte
        'HBB', 'HBA1',                 # erythroid
        'CD79A', 'VPREB1',             # B cell
        'NKG7', 'GNLY',                # NK/T cell
        'MPO', 'ELANE',                # granulocyte
    ],
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_10x_mtx(folder_path, sample_name):
    """Load 10x MTX folder into AnnData."""
    print(f"  Loading {sample_name} from {folder_path}")
    adata = sc.read_10x_mtx(folder_path, var_names='gene_symbols', cache=False)
    adata = sanitize_adata(adata)
    adata.obs['sample'] = sample_name
    print(f"    {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def load_h5ad(path, sample_name):
    """Load pre-processed h5ad file."""
    print(f"  Loading {sample_name} from {path}")
    adata = sc.read_h5ad(path)
    adata = sanitize_adata(adata)
    adata.obs['sample'] = sample_name
    print(f"    {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(adata, min_genes=200, min_cells=3,
               max_pct_mito=30, n_top_genes=2000):
    """Standard scRNA-seq preprocessing pipeline."""
    print(f"  Preprocessing {adata.obs['sample'].iloc[0]}...")

    # Mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'],
                                percent_top=None, log1p=False, inplace=True)

    # QC filters
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs['pct_counts_mt'] < max_pct_mito].copy()

    print(f"    After QC: {adata.n_obs} cells x {adata.n_vars} genes")

    # Normalise and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store normalised counts for module scoring
    adata.layers['norm_log'] = adata.X.copy()

    # HVG, scale, PCA, neighbours, UMAP
    # IMPORTANT: run PCA/UMAP on HVGs only to avoid densifying the full matrix.
    n_top_genes = min(n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    if 'highly_variable' in adata.var.columns and adata.var['highly_variable'].sum() >= 50:
        adata_hvg = adata[:, adata.var['highly_variable']].copy()
    else:
        adata_hvg = adata.copy()

    sc.pp.scale(adata_hvg, max_value=10)
    n_comps = min(30, max(2, adata_hvg.n_obs - 2), max(2, adata_hvg.n_vars - 1))
    sc.tl.pca(adata_hvg, svd_solver='arpack', n_comps=n_comps)
    sc.pp.neighbors(adata_hvg, n_pcs=n_comps)
    sc.tl.umap(adata_hvg)
    try:
        sc.tl.leiden(adata_hvg, resolution=0.5, key_added='leiden')
        adata.obs['leiden'] = adata_hvg.obs['leiden'].astype(str).values
    except Exception as e:
        print(f"    WARNING: Leiden clustering skipped: {e}")
        adata.obs['leiden'] = 'not_run'

    adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
    adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
    return adata


# ─────────────────────────────────────────────
# MALIGNANT CELL GATING
# ─────────────────────────────────────────────

def gate_malignant_cells(adata):
    """
    Identify putative malignant cells using blast vs normal marker scoring.
    Uses Gaussian mixture model for dynamic threshold — no fixed percentile.
    Returns adata with 'is_malignant' column in obs.
    """
    # Get normalised matrix
    if 'norm_log' in adata.layers:
        mat = adata.layers['norm_log']
    else:
        mat = adata.X

    gene_index = {g: i for i, g in enumerate(adata.var_names)}

    blast_genes  = [g for g in MODULES['stemness'][:5]
                    if g in gene_index]  # CD34, HOXA9, MEIS1, DNMT3B, FLT3
    normal_genes = [g for g in MODULES['normal_myeloid']
                    if g in gene_index]

    if not blast_genes:
        print("    WARNING: No blast marker genes found — skipping malignant gating")
        adata.obs['is_malignant'] = True
        adata.obs['malignancy_score'] = 0.0
        return adata

    blast_score  = row_mean(mat, [gene_index[g] for g in blast_genes], adata.n_obs)
    normal_score = row_mean(mat, [gene_index[g] for g in normal_genes], adata.n_obs)

    malignancy_score = blast_score - normal_score
    adata.obs['malignancy_score'] = malignancy_score

    # GMM-based dynamic threshold
    try:
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
        gmm.fit(malignancy_score.reshape(-1, 1))
        labels = gmm.predict(malignancy_score.reshape(-1, 1))
        malignant_label = np.argmax(gmm.means_.flatten())
        is_malignant = labels == malignant_label
        method = 'GMM'
    except Exception:
        # Fallback: largest gap in top 40%
        sorted_scores = np.sort(malignancy_score)
        start = int(len(sorted_scores) * 0.60)
        gaps = np.diff(sorted_scores[start:])
        if gaps.max() > 0.1 * (sorted_scores.max() - sorted_scores.min()):
            threshold = sorted_scores[start + np.argmax(gaps)]
            method = 'largest-gap'
        else:
            threshold = np.percentile(malignancy_score, 75)
            method = '75th-percentile-fallback'
        is_malignant = malignancy_score > threshold

    adata.obs['is_malignant'] = is_malignant
    n_mal = is_malignant.sum()
    print(f"    Malignant gating ({method}): {n_mal}/{adata.n_obs} cells "
          f"({n_mal/adata.n_obs*100:.1f}%) retained as putative blasts")
    print(f"    Blast markers used: {blast_genes}")

    return adata


# ─────────────────────────────────────────────
# MODULE SCORING
# ─────────────────────────────────────────────

def score_modules(adata):
    """
    Score all cells on each gene module.
    Uses scanpy's score_genes (Seurat-style control subtraction).
    Falls back to mean expression if too few genes found.
    """
    if 'norm_log' in adata.layers:
        mat = adata.layers['norm_log']
    else:
        mat = adata.X

    gene_index = {g: i for i, g in enumerate(adata.var_names)}

    module_scores = {}
    print("  Scoring gene modules:")
    for module_name, gene_list in MODULES.items():
        if module_name == 'normal_myeloid':
            continue
        found = [g for g in gene_list if g in gene_index]
        missing = [g for g in gene_list if g not in gene_index]
        if not found:
            print(f"    {module_name:25s}: NO GENES FOUND — skipping")
            module_scores[module_name] = np.zeros(adata.n_obs)
            continue
        cols = [gene_index[g] for g in found]
        score = row_mean(mat, cols, adata.n_obs)
        module_scores[module_name] = score
        print(f"    {module_name:25s}: {len(found)}/{len(gene_list)} genes "
              f"| mean={score.mean():.3f} range=[{score.min():.3f},{score.max():.3f}]"
              + (f" | MISSING: {missing}" if missing else ""))

    for name, scores in module_scores.items():
        adata.obs[f'score_{name}'] = scores

    # Composite hybrid score
    adata.obs['score_hybrid'] = (
        adata.obs['score_senescence_arrest'] +
        adata.obs['score_sasp'] +
        adata.obs['score_stemness']
    ) / 3.0

    # Quiescence proxy: inverse of strict cycling, not DNA-repair stress.
    adata.obs['score_quiescence_proxy'] = (
        adata.obs['score_quiescence'] -
        adata.obs['score_cycling']
    )

    # Composite biological ageing score: conserved age-up minus age-down modules.
    # Higher values indicate stronger overlap with conserved mammalian ageing signatures.
    if 'score_ageing_up' in adata.obs.columns and 'score_ageing_down' in adata.obs.columns:
        adata.obs['score_bio_age'] = (
            adata.obs['score_ageing_up'] -
            adata.obs['score_ageing_down']
        )

    return adata


# ─────────────────────────────────────────────
# HYBRID STATE CLASSIFICATION
# ─────────────────────────────────────────────

def classify_hybrid_states(adata, subset_malignant=True):
    """
    Classify cells into hybrid state categories.
    Uses 75th percentile thresholds within the malignant compartment.
    """
    if subset_malignant and 'is_malignant' in adata.obs.columns:
        cells = adata.obs['is_malignant']
    else:
        cells = pd.Series([True] * adata.n_obs, index=adata.obs.index)

    df = adata.obs.copy()

    # Compute thresholds within malignant compartment
    mal_df = df[cells]
    p75_arrest = np.percentile(mal_df['score_senescence_arrest'], 75)
    p75_sasp   = np.percentile(mal_df['score_sasp'], 75)
    p75_stem   = np.percentile(mal_df['score_stemness'], 75)
    p25_cycling = np.percentile(mal_df['score_cycling'], 25)

    def classify(row):
        h_arr  = row['score_senescence_arrest'] > p75_arrest
        h_sasp = row['score_sasp'] > p75_sasp
        h_stem = row['score_stemness'] > p75_stem
        low_p  = row['score_cycling'] <= p25_cycling

        if h_arr and h_sasp and h_stem:
            return 'Hybrid (arrest+SASP+stem)'
        elif h_arr and h_sasp:
            return 'Senescent-like (arrest+SASP)'
        elif h_stem and low_p:
            return 'Quiescent stem-like'
        elif h_stem:
            return 'Stem-like (cycling)'
        elif h_arr:
            return 'Arrest-only'
        elif h_sasp:
            return 'SASP-only'
        elif low_p:
            return 'Quiescent-low signal'
        else:
            return 'Undifferentiated'

    df['cell_state'] = df.apply(classify, axis=1)
    adata.obs['cell_state'] = df['cell_state']

    print("\n  Cell state distribution (malignant compartment):")
    mal_states = df[cells]['cell_state'].value_counts()
    for state, count in mal_states.items():
        print(f"    {state:35s}: {count:5d} ({count/cells.sum()*100:.1f}%)")

    return adata


# ─────────────────────────────────────────────
# STATISTICAL TESTS
# ─────────────────────────────────────────────

def compare_conditions(adata_ctrl, adata_treat, label_ctrl, label_treat,
                        subset_malignant=True):
    """
    Compare module scores between two conditions (e.g. Ctrl vs AraC,
    or Diagnosis vs Post-treatment).
    Returns DataFrame of results with effect sizes and p-values.
    """
    results = []

    for adata, label in [(adata_ctrl, label_ctrl), (adata_treat, label_treat)]:
        if subset_malignant and 'is_malignant' in adata.obs.columns:
            mask = adata.obs['is_malignant']
        else:
            mask = pd.Series([True] * adata.n_obs, index=adata.obs.index)

        score_cols = [c for c in adata.obs.columns if c.startswith('score_')]
        for col in score_cols:
            results.append({
                'condition': label,
                'module': col.replace('score_', ''),
                'mean': adata.obs.loc[mask, col].mean(),
                'median': adata.obs.loc[mask, col].median(),
                'std': adata.obs.loc[mask, col].std(),
                'n_cells': mask.sum(),
            })

    df = pd.DataFrame(results)

    # Mann-Whitney U tests
    mw_results = []
    score_cols = [c for c in adata_ctrl.obs.columns if c.startswith('score_')]

    for col in score_cols:
        module = col.replace('score_', '')

        if subset_malignant and 'is_malignant' in adata_ctrl.obs.columns:
            a = adata_ctrl.obs.loc[adata_ctrl.obs['is_malignant'], col]
            b = adata_treat.obs.loc[adata_treat.obs['is_malignant'], col]
        else:
            a = adata_ctrl.obs[col]
            b = adata_treat.obs[col]

        if len(a) < 3 or len(b) < 3:
            continue

        stat, pval = mannwhitneyu(a, b, alternative='two-sided')
        # Effect size: rank-biserial correlation
        n1, n2 = len(a), len(b)
        effect = 1 - (2 * stat) / (n1 * n2)

        mw_results.append({
            'module': module,
            f'mean_{label_ctrl}': a.mean(),
            f'mean_{label_treat}': b.mean(),
            'fold_change': b.mean() / (a.mean() + 1e-9),
            'effect_size_r': round(effect, 3),
            'p_value': pval,
            'p_adj': None,  # filled below
            'significant': pval < 0.05,
        })

    mw_df = pd.DataFrame(mw_results)

    # Bonferroni correction
    if len(mw_df):
        mw_df['p_adj'] = np.minimum(mw_df['p_value'] * len(mw_df), 1.0)
        mw_df['significant_adj'] = mw_df['p_adj'] < 0.05
        mw_df = mw_df.sort_values('p_value')

    return df, mw_df


# ─────────────────────────────────────────────
# CO-OCCURRENCE TEST
# ─────────────────────────────────────────────

def test_co_occurrence(adata, condition_col, condition_a, condition_b,
                        subset_malignant=True):
    """
    Key test: are senescence arrest + SASP + stemness co-elevated
    in the SAME cells, not just at population average level?
    Uses Spearman correlation between module pairs within each condition.
    """
    print("\n  === CO-OCCURRENCE ANALYSIS ===")
    print("  (Testing within-cell co-elevation, not population averages)\n")

    pairs = [
        ('score_senescence_arrest', 'score_sasp',      'Arrest vs SASP'),
        ('score_senescence_arrest', 'score_stemness',   'Arrest vs Stemness'),
        ('score_sasp',              'score_stemness',   'SASP vs Stemness'),
        ('score_hybrid',            'score_stemness',   'Hybrid vs Stemness'),
        ('score_hybrid',            'score_cycling', 'Hybrid vs True cycling'),
        ('score_hybrid',            'score_dna_repair', 'Hybrid vs DNA repair'),
        ('score_bio_age',           'score_hybrid', 'Biological age vs Hybrid'),
        ('score_bio_age',           'score_stemness', 'Biological age vs Stemness'),
        ('score_bio_age',           'score_sasp', 'Biological age vs SASP'),
    ]

    results = []
    for cond in [condition_a, condition_b]:
        mask_cond = adata.obs[condition_col] == cond
        if subset_malignant and 'is_malignant' in adata.obs.columns:
            mask = mask_cond & adata.obs['is_malignant']
        else:
            mask = mask_cond

        sub = adata.obs[mask]
        if len(sub) < 5:
            print(f"  {cond}: too few cells ({len(sub)}) — skipping")
            continue

        print(f"  Condition: {cond} (n={len(sub)} malignant cells)")
        for col_a, col_b, label in pairs:
            if col_a not in sub.columns or col_b not in sub.columns:
                continue
            r, p = spearmanr(sub[col_a], sub[col_b])
            results.append({
                'condition': cond,
                'pair': label,
                'spearman_r': round(r, 3),
                'p_value': p,
                'n_cells': len(sub),
            })
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            print(f"    {label:35s}: r={r:.3f}  p={p:.4f}  {sig}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# AUDIT TABLE
# ─────────────────────────────────────────────

def build_audit_table(adata_dict):
    """
    Build the cell-count audit table (Table S1 equivalent).
    adata_dict: {sample_name: adata}
    """
    rows = []
    for name, adata in adata_dict.items():
        n_total = adata.n_obs
        n_mal   = adata.obs['is_malignant'].sum() if 'is_malignant' in adata.obs else 'N/A'
        n_norm  = n_total - n_mal if isinstance(n_mal, (int, np.integer)) else 'N/A'
        pct_mal = f"{n_mal/n_total*100:.1f}%" if isinstance(n_mal, (int, np.integer)) else 'N/A'

        state_dist = {}
        if 'cell_state' in adata.obs.columns and 'is_malignant' in adata.obs.columns:
            mal_states = adata.obs[adata.obs['is_malignant']]['cell_state'].value_counts()
            for state, count in mal_states.items():
                state_dist[state] = f"{count} ({count/n_mal*100:.1f}%)"

        rows.append({
            'sample': name,
            'total_cells': n_total,
            'malignant_estimated': n_mal,
            'normal_excluded': n_norm,
            'pct_malignant': pct_mal,
            'hybrid_cells': state_dist.get('Hybrid (arrest+SASP+stem)', '0'),
            'senescent_like': state_dist.get('Senescent-like (arrest+SASP)', '0'),
            'quiescent_stem': state_dist.get('Quiescent stem-like', '0'),
        })

    return pd.DataFrame(rows)



# ─────────────────────────────────────────────
# HYBRID VS NON-HYBRID BIOLOGICAL AGE TEST
# ─────────────────────────────────────────────

def compare_hybrid_bio_age(adata_dict, subset_malignant=True):
    """
    Test whether hybrid cells have higher biological ageing score than
    non-hybrid malignant cells within each condition/sample.
    """
    rows = []
    target_state = 'Hybrid (arrest+SASP+stem)'

    for name, adata in adata_dict.items():
        if 'score_bio_age' not in adata.obs.columns or 'cell_state' not in adata.obs.columns:
            continue

        if subset_malignant and 'is_malignant' in adata.obs.columns:
            obs = adata.obs[adata.obs['is_malignant']].copy()
        else:
            obs = adata.obs.copy()

        hybrid = obs[obs['cell_state'] == target_state]['score_bio_age']
        nonhyb = obs[obs['cell_state'] != target_state]['score_bio_age']

        if len(hybrid) < 3 or len(nonhyb) < 3:
            rows.append({
                'sample': name,
                'n_hybrid': len(hybrid),
                'n_nonhybrid': len(nonhyb),
                'mean_bio_age_hybrid': hybrid.mean() if len(hybrid) else np.nan,
                'mean_bio_age_nonhybrid': nonhyb.mean() if len(nonhyb) else np.nan,
                'effect_size_r': np.nan,
                'p_value': np.nan,
                'note': 'too few cells'
            })
            continue

        stat, pval = mannwhitneyu(hybrid, nonhyb, alternative='two-sided')
        n1, n2 = len(hybrid), len(nonhyb)
        # Positive means hybrid tends to have higher bio_age than non-hybrid.
        effect = 1 - (2 * stat) / (n1 * n2)

        rows.append({
            'sample': name,
            'n_hybrid': len(hybrid),
            'n_nonhybrid': len(nonhyb),
            'mean_bio_age_hybrid': hybrid.mean(),
            'median_bio_age_hybrid': hybrid.median(),
            'mean_bio_age_nonhybrid': nonhyb.mean(),
            'median_bio_age_nonhybrid': nonhyb.median(),
            'delta_mean_hybrid_minus_nonhybrid': hybrid.mean() - nonhyb.mean(),
            'effect_size_r': round(effect, 3),
            'p_value': pval,
            'note': ''
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out['p_adj'] = np.minimum(out['p_value'] * out['p_value'].notna().sum(), 1.0)
        out['significant_adj'] = out['p_adj'] < 0.05
    return out


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_all(adata_dict, comparison_pairs, stats_df, cooccurrence_df, output_dir):
    """Generate all figures."""
    os.makedirs(output_dir, exist_ok=True)

    colors = {
        'Hybrid (arrest+SASP+stem)':    '#dc2626',
        'Senescent-like (arrest+SASP)': '#ea580c',
        'Quiescent stem-like':          '#7c3aed',
        'Stem-like (cycling)':          '#2563eb',
        'Arrest-only':                  '#d97706',
        'SASP-only':                    '#16a34a',
        'Quiescent-low signal':         '#94a3b8',
        'Undifferentiated':             '#e2e8f0',
    }

    # ── Figure 1: UMAP per sample coloured by cell state ──────────────────
    n_samples = len(adata_dict)
    fig, axes = plt.subplots(1, n_samples, figsize=(7 * n_samples, 6))
    if n_samples == 1:
        axes = [axes]

    for ax, (name, adata) in zip(axes, adata_dict.items()):
        if 'X_umap' not in adata.obsm:
            ax.set_title(f"{name}\n(UMAP not available)")
            continue
        umap = adata.obsm['X_umap']
        if 'cell_state' in adata.obs.columns:
            for state, color in colors.items():
                mask = adata.obs['cell_state'] == state
                if mask.sum() == 0:
                    continue
                ax.scatter(umap[mask, 0], umap[mask, 1],
                           c=color, s=8, alpha=0.7, label=state)
            ax.legend(fontsize=6, markerscale=2, loc='upper right')
        else:
            ax.scatter(umap[:, 0], umap[:, 1], s=8, alpha=0.5)
        ax.set_title(f"{name}\n({adata.n_obs} cells)", fontsize=11)
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_umap_cell_states.png", dpi=150)
    plt.close()
    print(f"  Saved: fig1_umap_cell_states.png")

    # ── Figure 2: Module score violin plots per condition ─────────────────
    score_cols = ['score_senescence_arrest', 'score_sasp', 'score_stemness',
                  'score_cycling', 'score_dna_repair', 'score_dna_damage',
                  'score_ageing_up', 'score_ageing_down', 'score_bio_age',
                  'score_hybrid']
    module_labels = ['Senescence\nArrest', 'SASP', 'Stemness/LSC',
                     'True Cycling', 'DNA Repair/\nReplication Stress',
                     'DNA Damage\nCheckpoint', 'Ageing\nUp', 'Ageing\nDown',
                     'Bio-age\nScore', 'Hybrid\nScore']

    all_data = []
    for name, adata in adata_dict.items():
        if 'is_malignant' in adata.obs.columns:
            sub = adata.obs[adata.obs['is_malignant']]
        else:
            sub = adata.obs
        for col, lbl in zip(score_cols, module_labels):
            if col in sub.columns:
                for val in sub[col].values:
                    all_data.append({'sample': name, 'module': lbl, 'score': val})

    if all_data:
        df_plot = pd.DataFrame(all_data)
        fig, axes = plt.subplots(1, len(score_cols),
                                  figsize=(3.5 * len(score_cols), 5))
        for ax, col, lbl in zip(axes, score_cols, module_labels):
            sub = df_plot[df_plot['module'] == lbl]
            if sub.empty:
                continue
            samples = list(adata_dict.keys())
            data_per = [sub[sub['sample'] == s]['score'].values for s in samples]
            ax.violinplot(data_per, positions=range(len(samples)),
                          showmedians=True, showextrema=False)
            ax.set_xticks(range(len(samples)))
            ax.set_xticklabels(samples, rotation=30, ha='right', fontsize=8)
            ax.set_title(lbl, fontsize=9)
            ax.set_ylabel('Score')
        plt.suptitle('Module Scores — Malignant Compartment', fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig2_module_scores_violin.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: fig2_module_scores_violin.png")

    # ── Figure 3: Co-occurrence scatter plots ─────────────────────────────
    if len(adata_dict) >= 2:
        names = list(adata_dict.keys())
        fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 9))
        if len(names) == 1:
            axes = axes.reshape(2, 1)

        pairs_plot = [
            ('score_senescence_arrest', 'score_stemness',
             'Arrest score', 'Stemness score'),
            ('score_sasp', 'score_stemness',
             'SASP score', 'Stemness score'),
        ]

        for col_idx, (name, adata) in enumerate(adata_dict.items()):
            if 'is_malignant' in adata.obs.columns:
                sub = adata.obs[adata.obs['is_malignant']]
            else:
                sub = adata.obs

            for row_idx, (xa, ya, xl, yl) in enumerate(pairs_plot):
                ax = axes[row_idx, col_idx]
                if xa not in sub.columns or ya not in sub.columns:
                    continue
                x, y = sub[xa].values, sub[ya].values
                r, p = spearmanr(x, y)

                # Colour by hybrid state if available
                if 'cell_state' in sub.columns:
                    for state, color in colors.items():
                        m = sub['cell_state'] == state
                        ax.scatter(x[m.values], y[m.values],
                                   c=color, s=10, alpha=0.6, label=state)
                else:
                    ax.scatter(x, y, s=10, alpha=0.5, c='steelblue')

                ax.set_xlabel(xl, fontsize=8)
                ax.set_ylabel(yl, fontsize=8)
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                ax.set_title(f"{name}\nr={r:.3f} {sig} (n={len(sub)})", fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig3_cooccurrence_scatter.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: fig3_cooccurrence_scatter.png")

    # ── Figure 4: Cell state bar chart comparison ─────────────────────────
    state_order = list(colors.keys())
    state_data = {}
    for name, adata in adata_dict.items():
        if 'cell_state' not in adata.obs.columns:
            continue
        if 'is_malignant' in adata.obs.columns:
            sub = adata.obs[adata.obs['is_malignant']]
        else:
            sub = adata.obs
        counts = sub['cell_state'].value_counts()
        pcts = counts / len(sub) * 100
        state_data[name] = pcts

    if state_data:
        df_state = pd.DataFrame(state_data).fillna(0)
        df_state = df_state.reindex([s for s in state_order if s in df_state.index])
        fig, ax = plt.subplots(figsize=(8, 5))
        bottom = np.zeros(len(df_state.columns))
        for state in df_state.index:
            vals = df_state.loc[state].values
            ax.bar(df_state.columns, vals, bottom=bottom,
                   color=colors.get(state, '#ccc'), label=state)
            bottom += vals
        ax.set_ylabel('% of malignant cells')
        ax.set_title('Cell State Distribution — Malignant Compartment')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig4_cell_state_bars.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: fig4_cell_state_bars.png")

    # ── Figure 5: Statistical comparison heatmap ──────────────────────────
    if stats_df is not None and not stats_df.empty:
        pivot = stats_df.pivot_table(
            index='module', columns='condition',
            values='mean', aggfunc='mean')
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2), 6))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r',
                        center=0, ax=ax, linewidths=0.5)
            ax.set_title('Mean Module Scores by Condition\n(malignant compartment)')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/fig5_score_heatmap.png",
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: fig5_score_heatmap.png")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    adata_dict = {}

    # ── Load GSE146590 (Duy AraC dataset) ────────────────────────────────
    if args.gse146590_ctrl and os.path.exists(args.gse146590_ctrl):
        print("\n[1] Loading GSE146590 — Control")
        adata_ctrl = load_10x_mtx(args.gse146590_ctrl, 'AML1566_Ctrl')
        adata_ctrl = preprocess(adata_ctrl)
        adata_ctrl = gate_malignant_cells(adata_ctrl)
        adata_ctrl = score_modules(adata_ctrl)
        adata_ctrl = classify_hybrid_states(adata_ctrl)
        adata_ctrl.obs['condition'] = 'Control'
        adata_ctrl.obs['dataset'] = 'GSE146590'
        adata_dict['AML1566_Ctrl'] = adata_ctrl

    if args.gse146590_arac and os.path.exists(args.gse146590_arac):
        print("\n[2] Loading GSE146590 — AraC treated")
        adata_arac = load_10x_mtx(args.gse146590_arac, 'AML1566_AraC')
        adata_arac = preprocess(adata_arac)
        adata_arac = gate_malignant_cells(adata_arac)
        adata_arac = score_modules(adata_arac)
        adata_arac = classify_hybrid_states(adata_arac)
        adata_arac.obs['condition'] = 'AraC'
        adata_arac.obs['dataset'] = 'GSE146590'
        adata_dict['AML1566_AraC'] = adata_arac

    # ── Load GSE116256 (van Galen) ────────────────────────────────────────
    if args.gse116256_h5ad and os.path.exists(args.gse116256_h5ad):
        print("\n[3] Loading GSE116256 — van Galen atlas")
        adata_vg = load_h5ad(args.gse116256_h5ad, 'vanGalen_AML')
        if 'norm_log' not in adata_vg.layers:
            # If the h5ad is raw counts, preprocess it; if already log-normalised this may be skipped by setting a norm_log layer beforehand.
            adata_vg = preprocess(adata_vg)
        # Expect 'timepoint' or 'condition' column in obs
        # Split into diagnosis vs post-treatment
        tp_col = None
        for c in ['timepoint', 'condition', 'Time point', 'Timepoint']:
            if c in adata_vg.obs.columns:
                tp_col = c
                break
        if tp_col:
            for tp in adata_vg.obs[tp_col].unique():
                sub = adata_vg[adata_vg.obs[tp_col] == tp].copy()
                name = f"vanGalen_{tp}"
                sub = gate_malignant_cells(sub)
                sub = score_modules(sub)
                sub = classify_hybrid_states(sub)
                sub.obs['condition'] = str(tp)
                sub.obs['dataset'] = 'GSE116256'
                adata_dict[name] = sub
        else:
            adata_vg = gate_malignant_cells(adata_vg)
            adata_vg = score_modules(adata_vg)
            adata_vg = classify_hybrid_states(adata_vg)
            adata_dict['vanGalen_AML'] = adata_vg

    # ── Fallback: use uploaded MTX files if no paths given ────────────────
    if not adata_dict:
        upload_path = '/mnt/user-data/uploads'
        if (os.path.exists(f'{upload_path}/matrix_mtx.gz') and
                os.path.exists(f'{upload_path}/barcodes_tsv.gz')):
            print("\n[Auto] Using uploaded MTX files")
            import scipy.io, gzip, scipy.sparse as sp

            with gzip.open(f'{upload_path}/barcodes_tsv.gz', 'rt') as f:
                barcodes = [l.strip() for l in f]
            with gzip.open(f'{upload_path}/features_tsv.gz', 'rt') as f:
                rows = [l.strip().split('\t') for l in f]
                gene_ids   = [r[0] for r in rows]
                gene_names = [r[1] if len(r) > 1 else r[0] for r in rows]
            with gzip.open(f'{upload_path}/matrix_mtx.gz', 'rb') as f:
                mat = scipy.io.mmread(f).T.tocsr()

            import anndata
            adata = anndata.AnnData(
                X=mat,
                obs=pd.DataFrame(index=barcodes),
                var=pd.DataFrame({'gene_ids': gene_ids}, index=gene_names),
            )
            adata.obs['sample'] = 'uploaded_sample'
            adata = preprocess(adata)
            adata = gate_malignant_cells(adata)
            adata = score_modules(adata)
            adata = classify_hybrid_states(adata)
            adata.obs['condition'] = 'uploaded'
            adata_dict['uploaded_sample'] = adata

    if not adata_dict:
        print("\nERROR: No data loaded. Provide at least one dataset path.")
        return

    # ── Audit table ────────────────────────────────────────────────────────
    print("\n[AUDIT TABLE]")
    audit = build_audit_table(adata_dict)
    print(audit.to_string(index=False))
    audit.to_csv(f"{output_dir}/table_S1_audit.csv", index=False)

    # ── Statistical comparisons ───────────────────────────────────────────
    all_stats = []
    all_mw    = []
    cooccurrence_df = pd.DataFrame()
    hybrid_bio_age_df = pd.DataFrame()

    # GSE146590: Ctrl vs AraC
    if 'AML1566_Ctrl' in adata_dict and 'AML1566_AraC' in adata_dict:
        print("\n[STATS] GSE146590: Control vs AraC")
        stats_df, mw_df = compare_conditions(
            adata_dict['AML1566_Ctrl'], adata_dict['AML1566_AraC'],
            'Control', 'AraC')
        all_stats.append(stats_df)
        mw_df['comparison'] = 'Ctrl_vs_AraC'
        all_mw.append(mw_df)
        print(mw_df[['module','mean_Control','mean_AraC',
                      'effect_size_r','p_adj','significant_adj']].to_string(index=False))

        # Merge for co-occurrence test
        adata_combined = adata_dict['AML1566_Ctrl'].concatenate(
            adata_dict['AML1566_AraC'], batch_key='condition',
            batch_categories=['Control', 'AraC'])
        cooccurrence_df = test_co_occurrence(
            adata_combined, 'condition', 'Control', 'AraC')

    print("\n[STATS] Hybrid vs non-hybrid biological ageing score")
    hybrid_bio_age_df = compare_hybrid_bio_age(adata_dict)
    if not hybrid_bio_age_df.empty:
        print(hybrid_bio_age_df.to_string(index=False))
        hybrid_bio_age_df.to_csv(f"{output_dir}/table_hybrid_vs_nonhybrid_bio_age.csv", index=False)

    # GSE116256: Diagnosis vs Post-treatment
    diag_keys = [k for k in adata_dict if 'Diag' in k or 'diag' in k or 'D0' in k]
    post_keys = [k for k in adata_dict if 'post' in k.lower() or 'D14' in k or 'relapse' in k.lower()]
    if diag_keys and post_keys:
        print(f"\n[STATS] GSE116256: {diag_keys[0]} vs {post_keys[0]}")
        stats_df2, mw_df2 = compare_conditions(
            adata_dict[diag_keys[0]], adata_dict[post_keys[0]],
            'Diagnosis', 'Post-treatment')
        all_stats.append(stats_df2)
        mw_df2['comparison'] = 'Diag_vs_PostTreat'
        all_mw.append(mw_df2)
        print(mw_df2[['module','mean_Diagnosis','mean_Post-treatment',
                       'effect_size_r','p_adj','significant_adj']].to_string(index=False))

    # ── Save results ───────────────────────────────────────────────────────
    if all_stats:
        pd.concat(all_stats).to_csv(f"{output_dir}/table_module_scores.csv",
                                     index=False)
    if all_mw:
        pd.concat(all_mw).to_csv(f"{output_dir}/table_stats_mw.csv", index=False)
    if not cooccurrence_df.empty:
        cooccurrence_df.to_csv(f"{output_dir}/table_cooccurrence.csv", index=False)

    # Save per-cell scores for each sample
    for name, adata in adata_dict.items():
        score_cols = ['is_malignant', 'malignancy_score', 'cell_state'] + \
                     [c for c in adata.obs.columns if c.startswith('score_')]
        save_cols = [c for c in score_cols if c in adata.obs.columns]
        adata.obs[save_cols].to_csv(
            f"{output_dir}/cell_scores_{name}.csv")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[PLOTS]")
    combined_stats = pd.concat(all_stats) if all_stats else None
    plot_all(adata_dict, [], combined_stats, cooccurrence_df, output_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Files saved:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size/1024:.1f} KB)")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AML Hybrid State Analysis')
    parser.add_argument('--gse146590_ctrl',  default=None,
                        help='Path to AML1566 Control 10x MTX folder')
    parser.add_argument('--gse146590_arac',  default=None,
                        help='Path to AML1566 AraC 10x MTX folder')
    parser.add_argument('--gse116256_h5ad',  default=None,
                        help='Path to van Galen AML atlas .h5ad file')
    parser.add_argument('--gse116256_dir',   default=None,
                        help='Path to GSE116256 directory (auto-discovers samples)')
    parser.add_argument('--output_dir',      default='./aml_results',
                        help='Output directory for results and figures')
    args = parser.parse_args()
    run_pipeline(args)