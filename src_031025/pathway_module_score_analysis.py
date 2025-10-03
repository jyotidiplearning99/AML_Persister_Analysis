#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module-Based Persister Score Analysis (Upgraded)
------------------------------------------------
Purpose
- Take modules (GO terms, KEGG pathways, TF target sets, or any custom gene set)
- Compute a per-sample score for each module (PCA-based by default)
- Quantify separation between AML vs Healthy (AUC, Cohen's d, t-test, optional permutation p)
- Rank modules by a composite rank score
- Export per-sample module scores matrix + ranked summary

Highlights
- Robust PC1 orientation so that higher scores correspond to the positive class (e.g., AML)
- Optional scoring methods: pca | mean | zmean | ssgsea (if gseapy installed)
- Multiple inputs for modules: JSON, CSV, or directly from your KEGG/TF pipeline outputs
- Parallel execution over modules (n_jobs)
- FDR correction across modules
- Optional bootstrap CIs for AUC and permutation-based p-values

Expected inputs
- Expression CSV: genes x samples (rows=gene symbols, columns=sample IDs)
- Metadata CSV: samples x columns, must contain a column with class labels (default 'condition')
  where positive_label=AML by default.
- Modules via:
  a) JSON mapping {module_name: [genes,...]}
  b) CSV with columns [module,gene] (long) or [module,genes] (list)
  c) KEGG/TF pipeline directory containing 'modules_for_scoring.json' (see patch instructions)

Outputs (saved to --outdir)
- module_scores_ranked.csv      : ranked modules with metrics and FDR
- module_scores_matrix.tsv      : per-sample scores (rows=modules, cols=samples)
- module_auc_ci.csv (optional)  : bootstrap AUC 95% CI if --bootstrap > 0
- module_permutation_p.csv      : permutation p-values if --permutations > 0
- top_modules_violin.png        : violin plot for the top modules

Usage examples
--------------
1) Using a JSON of modules (incl. GO_morphogenesis_epithelium & TF_ESR1_targets):
   python module_score_analysis.py \
       --expression /path/to/expression.csv \
       --metadata /path/to/metadata.csv \
       --modules-json /path/to/modules.json \
       --label-column condition --positive-label AML \
       --method pca --n-jobs 8 --bootstrap 1000 \
       --outdir /path/to/module_analysis

2) Directly consume outputs from your KEGG/TF pipeline (after adding the export method):
   python module_score_analysis.py \
       --expression /path/to/expression.csv \
       --metadata /path/to/metadata.csv \
       --kegg-tf-dir /path/to/pathway_analysis_results_fixed \
       --method pca --outdir /path/to/module_analysis

3) Quickly test two inline modules (overrides other module inputs):
   python module_score_analysis.py \
       --expression expr.csv --metadata meta.csv \
       --inline-module "GO_morphogenesis_epithelium:AGT,AREG,BMP2,..." \
       --inline-module "TF_ESR1_targets:GENE1,GENE2,..." \
       --outdir out
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# Optional: ssGSEA via gseapy
try:
    import gseapy as gp  # type: ignore
    _HAVE_GSEAPY = True
except Exception:
    _HAVE_GSEAPY = False

# Optional: parallelism
try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    """Z-score per gene (row)."""
    m = matrix.mean(axis=1, keepdims=True)
    s = matrix.std(axis=1, ddof=1, keepdims=True)
    s[s == 0] = 1.0
    return (matrix - m) / s


def _orient_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Orient scores so that the positive class (1) has higher mean."""
    if scores.ndim > 1:
        scores = scores.ravel()
    pos = labels == 1
    neg = labels == 0
    pos_mean = scores[pos].mean() if pos.any() else 0.0
    neg_mean = scores[neg].mean() if neg.any() else 0.0
    return scores if pos_mean >= neg_mean else -scores


def _auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled = np.sqrt(((n1 - 1) * vx + (n2 - 1) * vy) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled


def _permutation_p(scores: np.ndarray, labels: np.ndarray, n: int = 0, random_state: int = 42) -> Optional[float]:
    if n <= 0:
        return None
    rng = np.random.default_rng(random_state)
    true_auc = _auc_safe(labels, scores)
    count = 0
    for _ in range(n):
        perm = rng.permutation(labels)
        if _auc_safe(perm, scores) >= true_auc:
            count += 1
    return (count + 1) / (n + 1)


def _bootstrap_auc_ci(scores: np.ndarray, labels: np.ndarray, n: int = 0, random_state: int = 42, alpha: float = 0.05) -> Optional[Tuple[float, float]]:
    if n <= 0:
        return None
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(labels))
    aucs = []
    for _ in range(n):
        samp = rng.choice(idx, size=len(idx), replace=True)
        aucs.append(_auc_safe(labels[samp], scores[samp]))
    lo = np.percentile(aucs, 100 * (alpha / 2))
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# --------------------------------------------------------------------------------------
# Scoring methods
# --------------------------------------------------------------------------------------

def score_module_pca(expr_sub: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Z-score genes; take PC1; orient towards positive class."""
    if expr_sub.shape[0] < 2:
        return np.zeros(expr_sub.shape[1])
    X = _zscore_rows(expr_sub.values)
    pca = PCA(n_components=1, random_state=42)
    s = pca.fit_transform(X.T).ravel()
    return _orient_scores(s, labels)


def score_module_mean(expr_sub: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Mean expression across genes; z-score across samples; orient."""
    if expr_sub.shape[0] == 0:
        return np.zeros(expr_sub.shape[1])
    s = expr_sub.mean(axis=0).values
    s_std = s.std(ddof=1)
    s = (s - s.mean()) / (s_std if s_std != 0 else 1.0)
    return _orient_scores(s, labels)


def score_module_zmean(expr_sub: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Z-score per gene then average across genes; orient."""
    if expr_sub.shape[0] == 0:
        return np.zeros(expr_sub.shape[1])
    X = _zscore_rows(expr_sub.values)
    s = X.mean(axis=0)
    return _orient_scores(s, labels)


def score_module_ssgsea(expr_sub: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """ssGSEA via gseapy. Returns oriented enrichment scores."""
    if not _HAVE_GSEAPY or expr_sub.shape[0] == 0:
        return score_module_pca(expr_sub, labels)
    gs_dict = {"MODULE": list(expr_sub.index)}
    try:
        res = gp.ssgsea(data=expr_sub, gene_sets=gs_dict, sample_norm_method="rank", outdir=None, verbose=False)
        s = res.res2d.loc["MODULE"].values.astype(float)
        return _orient_scores(s, labels)
    except Exception:
        return score_module_pca(expr_sub, labels)


SCORERS = {
    'pca': score_module_pca,
    'mean': score_module_mean,
    'zmean': score_module_zmean,
    'ssgsea': score_module_ssgsea,
}


# --------------------------------------------------------------------------------------
# Module loading
# --------------------------------------------------------------------------------------

def load_modules_from_json(path: Path) -> Dict[str, List[str]]:
    with open(path) as f:
        raw = json.load(f)
    modules = {k: sorted(list(dict.fromkeys([g.strip().upper() for g in v if g])))
               for k, v in raw.items()}
    return {k: v for k, v in modules.items() if len(v) >= 2}


def load_modules_from_csv(path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(path)
    modules: Dict[str, List[str]] = {}
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if {'module', 'gene'}.issubset(cols):
        for mod, sub in df.groupby('module'):
            modules[mod] = sorted(list(dict.fromkeys(
                [str(g).strip().upper() for g in sub['gene'] if str(g).strip() and str(g).upper() != 'NAN']
            )))
    elif 'module' in cols and any(c in cols for c in ['genes', 'gene_list']):
        gl_col = 'genes' if 'genes' in cols else 'gene_list'
        for _, row in df.iterrows():
            mod = str(row['module'])
            genes = str(row[gl_col])
            sep = ';' if ';' in genes else ','
            lst = [g.strip().upper() for g in genes.split(sep) if g.strip()]
            modules[mod] = sorted(list(dict.fromkeys(lst)))
    else:
        raise ValueError("CSV must have columns [module,gene] or [module,genes].")
    return {k: v for k, v in modules.items() if len(v) >= 2}


def load_modules_from_kegg_tf_dir(path: Path) -> Dict[str, List[str]]:
    """Prefer modules_for_scoring.json if present. Fallback tries TSV or a truncated CSV."""
    jpath = path / 'modules_for_scoring.json'
    if jpath.exists():
        return load_modules_from_json(jpath)
    tsv = path / 'modules_for_scoring.tsv'
    if tsv.exists():
        return load_modules_from_csv(tsv)
    sm = path / 'signaling_modules_with_fdr.csv'
    if sm.exists():
        df = pd.read_csv(sm)
        if 'pathway' in df.columns and 'genes' in df.columns:
            modules = {}
            for _, row in df.iterrows():
                mod = str(row['pathway'])
                genes = str(row['genes'])
                sep = ';' if ';' in genes else ','
                lst = [g.strip().upper() for g in genes.split(sep) if g.strip()]
                if len(lst) >= 2:
                    modules[mod] = sorted(list(dict.fromkeys(lst)))
            return modules
    raise FileNotFoundError("Could not find modules_for_scoring.json/tsv in KEGG/TF directory.")


# --------------------------------------------------------------------------------------
# Main analyzer
# --------------------------------------------------------------------------------------

class ModuleScoreAnalyzer:
    def __init__(self, expr_path: Path, meta_path: Path, label_col: str = 'condition', positive_label: str = 'AML'):
        self.expr = pd.read_csv(expr_path, index_col=0)
        self.meta = pd.read_csv(meta_path, index_col=0)

        # Align samples
        common = self.expr.columns.intersection(self.meta.index)
        if len(common) == 0:
            raise ValueError("No overlapping samples between expression and metadata.")
        self.expr = self.expr.loc[:, common]
        self.meta = self.meta.loc[common]

        # Labels as 0/1
        if label_col not in self.meta.columns:
            raise ValueError(f"Metadata is missing label column '{label_col}'.")
        self.y = (self.meta[label_col].astype(str).values == str(positive_label)).astype(int)

        logging.info(f"Expression: {self.expr.shape[0]} genes x {self.expr.shape[1]} samples")
        logging.info(f"Labels: positive='{positive_label}'  #pos={self.y.sum()}, #neg={len(self.y)-self.y.sum()}")

    def evaluate_one(self, name: str, genes: List[str], method: str,
                     permutations: int = 0, bootstrap: int = 0) -> Tuple[Dict, np.ndarray]:
        genes_u = [g.upper() for g in genes]
        avail = [g for g in genes_u if g in self.expr.index]
        sub = self.expr.loc[avail]
        if sub.shape[0] < 2:
            scores = np.zeros(self.expr.shape[1])
            metrics = {
                'module': name,
                'n_genes_used': sub.shape[0],
                'n_genes_total': len(genes_u),
                'auc': 0.5,
                'cohens_d': 0.0,
                't_stat': 0.0,
                'p_value': 1.0,
                'perm_p': None,
                'auc_ci_lo': None,
                'auc_ci_hi': None,
            }
            return metrics, scores

        scorer = SCORERS[method]
        scores = scorer(sub, self.y)

        aml_scores = scores[self.y == 1]
        ctl_scores = scores[self.y == 0]

        # Statistics
        from scipy import stats
        try:
            t_stat, p_val = stats.ttest_ind(aml_scores, ctl_scores, equal_var=False)
        except Exception:
            t_stat, p_val = 0.0, 1.0

        d = _cohens_d(aml_scores, ctl_scores)
        auc = _auc_safe(self.y, scores)
        perm_p = _permutation_p(scores, self.y, permutations) if permutations > 0 else None
        ci = _bootstrap_auc_ci(scores, self.y, bootstrap) if bootstrap > 0 else None

        metrics = {
            'module': name,
            'n_genes_used': sub.shape[0],
            'n_genes_total': len(genes_u),
            'auc': auc,
            'cohens_d': d,
            't_stat': t_stat,
            'p_value': p_val,
            'perm_p': perm_p,
            'auc_ci_lo': (ci[0] if ci else None),
            'auc_ci_hi': (ci[1] if ci else None),
        }
        return metrics, scores

    def evaluate_all(self, modules: Dict[str, List[str]], method: str = 'pca',
                     permutations: int = 0, bootstrap: int = 0, n_jobs: int = 1):
        names = list(modules.keys())
        genesets = [modules[n] for n in names]

        if _HAVE_JOBLIB and n_jobs != 1 and len(names) > 1:
            outs = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(self.evaluate_one)(n, g, method, permutations, bootstrap)
                for n, g in zip(names, genesets)
            )
        else:
            outs = [self.evaluate_one(n, g, method, permutations, bootstrap)
                    for n, g in zip(names, genesets)]

        metrics_list = [m for m, _ in outs]
        score_mat = np.vstack([s for _, s in outs]) if outs else np.zeros((0, self.expr.shape[1]))

        metrics_df = pd.DataFrame(metrics_list)
        scores_df = pd.DataFrame(score_mat, index=names, columns=self.expr.columns)

        # Composite rank: effect size (40%), significance (t-test p, 30%), discrimination (AUC from 0.5, 30%)
        metrics_df['rank_score'] = (
            metrics_df['cohens_d'].abs() * 0.4 +
            (1 - metrics_df['p_value'].clip(0, 1)) * 0.3 +
            (metrics_df['auc'] - 0.5).abs() * 2 * 0.3
        )

        # FDR over the t-test p-values
        try:
            _, qvals, _, _ = multipletests(metrics_df['p_value'].fillna(1.0).values, method='fdr_bh')
            metrics_df['p_adj_bh'] = qvals
        except Exception:
            metrics_df['p_adj_bh'] = np.nan

        # Sort metrics and reorder scores accordingly so plotting uses top-ranked
        metrics_df = metrics_df.sort_values('rank_score', ascending=False)
        scores_df = scores_df.loc[metrics_df['module']]

        return metrics_df, scores_df


# --------------------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------------------

def parse_inline_modules(inlines: List[str]) -> Dict[str, List[str]]:
    mods: Dict[str, List[str]] = {}
    for s in inlines:
        if ':' not in s:
            continue
        name, genes_s = s.split(':', 1)
        lst = [g.strip().upper() for g in genes_s.replace(';', ',').split(',') if g.strip()]
        if len(lst) >= 2:
            mods[name.strip()] = sorted(list(dict.fromkeys(lst)))
    return mods


def find_module_overlaps(modules: Dict[str, List[str]], min_overlap: int = 5) -> pd.DataFrame:
    """Find pairwise module overlaps; return sorted dataframe."""
    overlaps = []
    names = list(modules.keys())
    for i, m1 in enumerate(names):
        s1 = set(modules[m1])
        for m2 in names[i+1:]:
            shared = s1 & set(modules[m2])
            if len(shared) >= min_overlap:
                overlaps.append({
                    'module1': m1,
                    'module2': m2,
                    'n_shared': len(shared),
                    'shared_genes': ', '.join(sorted(list(shared))[:10])
                })
    if overlaps:
        df = pd.DataFrame(overlaps).sort_values('n_shared', ascending=False)
    else:
        df = pd.DataFrame(columns=['module1', 'module2', 'n_shared', 'shared_genes'])
    return df


def save_violin(scores_df: pd.DataFrame, meta: pd.DataFrame, label_col: str, topk: int, outpath: Path):
    """Make a quick violin plot for the top-k modules (by current order of scores_df)."""
    sel = scores_df.index[:min(topk, scores_df.shape[0])]
    n = len(sel)
    if n == 0:
        return

    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    groups = meta[label_col].astype(str)
    group_names = sorted(groups.unique())

    for i, mod in enumerate(sel):
        ax = axes[i]
        s = scores_df.loc[mod]
        vals = [s[groups == g].values for g in group_names]
        ax.violinplot(vals, showmeans=True)
        ax.set_title(mod)
        ax.set_xticks(range(1, len(group_names) + 1))
        ax.set_xticklabels(group_names, rotation=0)

    # Turn off extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Module-Based Persister Score Analysis (upgraded)")
    ap.add_argument('--expression', type=Path, required=True, help='Expression CSV (genes x samples)')
    ap.add_argument('--metadata', type=Path, required=True, help='Metadata CSV (samples x cols)')
    ap.add_argument('--label-column', type=str, default='condition', help='Column with class labels (default: condition)')
    ap.add_argument('--positive-label', type=str, default='AML', help='Positive class label (default: AML)')

    ap.add_argument('--modules-json', type=Path, help='JSON mapping {module: [genes,...]}')
    ap.add_argument('--modules-csv', type=Path, help='CSV with [module,gene] or [module,genes]')
    ap.add_argument('--kegg-tf-dir', type=Path, help='Directory containing modules_for_scoring.json from KEGG/TF pipeline')
    ap.add_argument('--inline-module', action='append', default=[], help='Inline "NAME:gene1,gene2,..." (can repeat)')

    ap.add_argument('--method', choices=list(SCORERS.keys()), default='pca', help='Scoring method')
    ap.add_argument('--permutations', type=int, default=0, help='Permutation count for module-level p-value (AUC)')
    ap.add_argument('--bootstrap', type=int, default=0, help='Bootstrap count for AUC CI')
    ap.add_argument('--n-jobs', type=int, default=1, help='Parallel jobs over modules (requires joblib)')

    ap.add_argument('--outdir', type=Path, required=True, help='Output directory')

    # Extras
    ap.add_argument('--export-top-genes', action='store_true', help='Export union of genes from top-N modules')
    ap.add_argument('--top-n-modules', type=int, default=3, help='How many top modules to export genes from')
    ap.add_argument('--dry-run', action='store_true', help='List modules and exit without scoring')
    ap.add_argument('--overlap-min', type=int, default=5, help='Min shared genes to report in module overlaps')

    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    analyzer = ModuleScoreAnalyzer(args.expression, args.metadata,
                                   label_col=args.label_column,
                                   positive_label=args.positive_label)

    # Assemble modules from provided sources
    modules: Dict[str, List[str]] = {}
    if args.modules_json and args.modules_json.exists():
        logging.info(f"Loading modules from JSON: {args.modules_json}")
        modules.update(load_modules_from_json(args.modules_json))
    if args.modules_csv and args.modules_csv.exists():
        logging.info(f"Loading modules from CSV: {args.modules_csv}")
        modules.update(load_modules_from_csv(args.modules_csv))
    if args.kegg_tf_dir and args.kegg_tf_dir.exists():
        logging.info(f"Loading modules from KEGG/TF dir: {args.kegg_tf_dir}")
        try:
            modules.update(load_modules_from_kegg_tf_dir(args.kegg_tf_dir))
        except Exception as e:
            logging.warning(str(e))

    # Inline overrides (add/replace)
    if args.inline_module:
        mods_inline = parse_inline_modules(args.inline_module)
        if mods_inline:
            logging.info(f"Adding inline modules: {list(mods_inline.keys())}")
            modules.update(mods_inline)

    # If no modules provided, seed with a few defaults as example
    if not modules:
        logging.warning("No modules provided; seeding with example GO/RTK/WNT sets (edit these!).")
        modules = {
            'GO_morphogenesis_epithelium': [
                'AGT','AREG','BMP2','CAMSAP3','CD44','CELSR1','CSF1','CTNNB1','DDR1','DEAF1','EFNB2','EGFR',
                'EPHA2','EPHA4','FERMT1','FERMT2','FOLR1','FOXQ1','FZD6','HNF1B','HOXB7','IFT57','INTU','IRX2',
                'IRX3','KDF1','KDM5B','KDR','KLF4','LAMA5','LGR4','LRG1','LRP5','LZTS2','MDK','MESP1','MET','MYC',
                'MYO9A','NAGLU','NKX2-1','NPHP1','PRICKLE1','PTEN','RPGRIP1L','SDC4','SNAI2','SOX9','SYNE4',
                'TCTN1','TGFB1I1','TGFB2','TNC','WNT4','WNT7B','YAP1'
            ],
            'GO_cell_adhesion_subset': ['EPHA2','EPHA4','EPHB2','EPHB3','EPHB4','EFNB2','CD44','FERMT1','FERMT2','DDR1','CELSR1','SDC4','TNC'],
            'WNT_pathway_subset': ['WNT4','WNT7B','FZD6','LRP5','CTNNB1','LGR4','PRICKLE1'],
            'RTK_signaling_subset': ['EGFR','MET','KDR','AREG','CSF1','MDK'],
            'Stemness_markers_subset': ['SOX9','KLF4','YAP1','MYC','CD44','SNAI2'],
        }

    # Dry-run option
    if args.dry_run:
        print(f"Would analyze {len(modules)} modules:")
        for name, genes in modules.items():
            print(f"  {name}: {len(genes)} genes")
        sys.exit(0)

    # Evaluate all
    metrics_df, scores_df = analyzer.evaluate_all(
        modules, method=args.method,
        permutations=args.permutations,
        bootstrap=args.bootstrap,
        n_jobs=args.n_jobs
    )

    # Save results
    metrics_path = outdir / 'module_scores_ranked.csv'
    scores_path  = outdir / 'module_scores_matrix.tsv'
    metrics_df.to_csv(metrics_path, index=False)
    scores_df.to_csv(scores_path, sep='\t')
    logging.info(f"Saved: {metrics_path}")
    logging.info(f"Saved: {scores_path}")

    # Optional: export union of genes from top-N modules
    if args.export_top_genes:
        top_n = args.top_n_modules if args.top_n_modules else 3
        top_modules = metrics_df.head(top_n)['module'].tolist()
        focused_genes = set()
        for m in top_modules:
            if m in modules:
                focused_genes.update(modules[m])
        out_genes = outdir / f'top{top_n}_modules_genes.txt'
        with open(out_genes, 'w') as f:
            for g in sorted(focused_genes):
                f.write(f"{g}\n")
        logging.info(f"Exported {len(focused_genes)} genes from top {top_n} modules -> {out_genes}")

    # Module overlap analysis
    try:
        overlaps_df = find_module_overlaps(modules, min_overlap=args.overlap_min)
        if not overlaps_df.empty:
            overlaps_path = outdir / 'module_overlaps.csv'
            overlaps_df.to_csv(overlaps_path, index=False)
            logging.info(f"Saved module overlaps (>= {args.overlap_min} shared genes): {overlaps_path}")
        else:
            logging.info("No module pairs met the overlap threshold.")
    except Exception as e:
        logging.warning(f"Overlap analysis failed: {e}")

    # Save bootstrap CIs separately (if any)
    if 'auc_ci_lo' in metrics_df.columns and metrics_df['auc_ci_lo'].notna().any():
        ci_df = metrics_df[['module', 'auc', 'auc_ci_lo', 'auc_ci_hi']]
        ci_df.to_csv(outdir / 'module_auc_ci.csv', index=False)

    # Save permutation p-values separately (if any)
    if 'perm_p' in metrics_df.columns and metrics_df['perm_p'].notna().any():
        perm_df = metrics_df[['module', 'perm_p']]
        perm_df.to_csv(outdir / 'module_permutation_p.csv', index=False)

    # Quick violin for top modules (scores_df already ordered by rank)
    try:
        save_violin(scores_df, analyzer.meta, args.label_column, topk=6, outpath=outdir / 'top_modules_violin.png')
    except Exception as e:
        logging.warning(f"Could not save violin plot: {e}")

    # Console summary
    print("\nTOP 10 MODULES (by rank_score)\n------------------------------")
    for _, r in metrics_df.head(10).iterrows():
        q = r['p_adj_bh'] if pd.notna(r['p_adj_bh']) else np.nan
        print(f"{r['module'][:50]:50s}  genes:{int(r['n_genes_used']):3d}/{int(r['n_genes_total']):3d}  "
              f"AUC:{r['auc']:.3f}  d:{r['cohens_d']:.3f}  p:{r['p_value']:.2e}  q:{q:.2e}")


if __name__ == '__main__':
    main()
