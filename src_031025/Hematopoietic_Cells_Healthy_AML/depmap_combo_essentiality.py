#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepMap Combination / Set Essentiality — fast + NaN-safe

Changes vs previous:
  • Impute NaNs per-gene (symbol) with column medians across cohort lines, once.
  • Precompute cohort indexer and cache symbol vectors already sliced to the cohort.
  • No per-permutation np.where; permutations now reuse cached symbol vectors.
  • AUC/MWU computed after NaN-impute → no AUC=nan.
  • Print permutation progress; Ctrl-C stops perms gracefully and still writes outputs.

Inputs default to your Mahti paths.
"""

import os, re, json, argparse, signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Defaults (match your env) ----------
DEFAULT_DEPMAP_DIR = Path("/scratch/project_2010751/DepMap_Datasets")
DEFAULT_SCORE_DIR  = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/Hematopoietic_Cells_Healthy_AML_analysis")
DEFAULT_DE_FILE    = DEFAULT_SCORE_DIR / "de_results_aml_vs_hspc.csv"
DEFAULT_RANKED     = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/depmap_refined/aml_dependencies_ranked.csv")

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

def read_gene_list(p: Path) -> List[str]:
    with open(p) as f:
        return [ln.strip().upper() for ln in f if ln.strip()]

def load_scoring_files(score_dir: Path, up_only: bool) -> Tuple[List[str], List[str]]:
    up_p = score_dir / "scoring_up.txt"
    dn_p = score_dir / "scoring_down.txt"
    if not up_p.exists():
        raise FileNotFoundError(f"scoring_up.txt not found in {score_dir}")
    up = read_gene_list(up_p)
    if up_only or not dn_p.exists():
        if not dn_p.exists():
            print(f"[WARN] scoring_down.txt missing; using up-only.")
        return up, []
    return up, read_gene_list(dn_p)

def load_metadata(depmap_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(depmap_dir / "Model.csv")
    if "ModelID" not in meta.columns:
        raise RuntimeError("Model.csv missing ModelID")
    meta = meta.set_index("ModelID")
    aml = pd.Series(False, index=meta.index)
    for col in ["OncotreePrimaryDisease","OncotreeLineage","OncotreeSubtype"]:
        if col in meta.columns:
            aml |= meta[col].astype(str).str.contains("Acute Myeloid|AML|Myeloid", case=False, na=False)
    meta["is_AML"] = aml
    return meta

def load_crispr_effect(depmap_dir: Path) -> pd.DataFrame:
    # rows: ModelID, cols: "SYMBOL (ENTREZ)"
    return pd.read_csv(depmap_dir / "CRISPRGeneEffect.csv", index_col=0)

def build_symbol_map(columns: List[str]) -> Dict[str, List[str]]:
    out: Dict[str,List[str]] = {}
    for c in columns:
        sym = c.split(" (")[0].upper()
        out.setdefault(sym, []).append(c)
    return out

def subset_matrix_by_symbols(effect: pd.DataFrame,
                             symmap: Dict[str, List[str]],
                             symbols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    present, missing, pieces = [], [], []
    for g in symbols:
        if g in symmap:
            present.append(g)
            cols = symmap[g]
            if len(cols) == 1:
                pieces.append(effect[cols[0]].rename(g))
            else:
                pieces.append(effect[cols].mean(axis=1).rename(g))
        else:
            missing.append(g)
    if not pieces:
        return pd.DataFrame(index=effect.index), present, missing
    M = pd.concat(pieces, axis=1)
    return M, present, missing

def compute_auc_mwu(pos: np.ndarray, neg: np.ndarray) -> Tuple[float, float]:
    """More negative = stronger essentiality for positives (AML).
       Flip sign for AUC so 'higher=positive'."""
    if len(pos)==0 or len(neg)==0:
        return float("nan"), float("nan")
    y_true  = np.array([1]*len(pos) + [0]*len(neg))
    y_score = np.concatenate([-pos, -neg])  # flip
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")
    try:
        _, p = mannwhitneyu(pos, neg, alternative="less")  # H1: pos < neg
    except Exception:
        p = float("nan")
    return float(auc), float(p)

def decile_bins(x: pd.Series) -> pd.Series:
    vals = x.values
    good = np.isfinite(vals)
    if good.sum() <= 1:
        return pd.Series(0, index=x.index)
    ranks = pd.Series(vals).rank(method="average")
    return pd.qcut(ranks, q=10, labels=False, duplicates="drop")

def top_m_rowwise(arr: np.ndarray, m: int) -> np.ndarray:
    if arr.size == 0:
        return np.array([])
    m = max(1, min(m, arr.shape[1]))
    part = np.partition(arr, m-1, axis=1)[:, :m]
    return part.mean(axis=1)

def weighted_mean_rowwise(arr: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    if arr.size == 0:
        return np.array([])
    if w is None:
        return arr.mean(axis=1)
    return arr @ w

def load_weights(mode: str,
                 genes_present: List[str],
                 de_file: Optional[Path],
                 ranked_file: Optional[Path],
                 custom_file: Optional[Path]) -> Optional[np.ndarray]:
    mode = (mode or "none").lower()
    if mode == "none":
        return None
    if mode == "file":
        if not custom_file or not custom_file.exists():
            raise FileNotFoundError("--weights file selected but --weights-file missing")
        df = pd.read_csv(custom_file)
        cols = {c.lower(): c for c in df.columns}
        gcol, wcol = cols["gene"], cols["weight"]
        m = {str(r[gcol]).strip().upper(): float(r[wcol]) for _, r in df.iterrows()}
    elif mode == "de":
        if not de_file or not de_file.exists():
            raise FileNotFoundError("--weights de selected but DE file missing")
        de = pd.read_csv(de_file, index_col=0)
        if "log2_fold_change" not in de.columns:
            raise RuntimeError("DE file lacks 'log2_fold_change'")
        # robust index
        if de.index.name is None or de.index.name.lower() not in ("gene","symbol"):
            if "gene" in de.columns: de = de.set_index("gene")
            elif "symbol" in de.columns: de = de.set_index("symbol")
        de.index = de.index.astype(str).str.upper()
        s = de["log2_fold_change"].clip(lower=0.0)  # weight >= 0
        m = s.to_dict()
    elif mode == "depmap":
        if not ranked_file or not ranked_file.exists():
            raise FileNotFoundError("--weights depmap selected but ranked file missing")
        df = pd.read_csv(ranked_file)
        df.columns = df.columns.str.lower()
        if "gene" not in df.columns:
            raise RuntimeError("ranked file lacks 'gene'")
        wcol = "dep_prob_median" if "dep_prob_median" in df.columns else (
               "frac_dep_lt_minus05" if "frac_dep_lt_minus05" in df.columns else None)
        if wcol is None:
            raise RuntimeError("ranked file lacks dep_prob_median/frac_dep_lt_minus05")
        df["gene"] = df["gene"].astype(str).str.upper()
        m = df.set_index("gene")[wcol].clip(lower=0.0).to_dict()
    else:
        raise ValueError(f"Unknown weights mode: {mode}")

    w = np.array([float(m.get(g, 0.0)) for g in genes_present], dtype=float)
    s = w.sum()
    return (w / s) if s > 0 else None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------------- Core evaluation ----------------
def evaluate_cohort(
    cohort_name: str,
    effect: pd.DataFrame,
    meta: pd.DataFrame,
    genes_up: List[str],
    genes_down: List[str],
    out_dir: Path,
    weights_mode: str = "none",
    de_file: Optional[Path] = None,
    ranked_file: Optional[Path] = None,
    custom_weights_file: Optional[Path] = None,
    use_up_only: bool = False,
    topm: int = 3,
    thr1: float = -0.5,
    thr2: float = -0.3,
    n_perm: int = 1000,
    seed: int = 42,
    permute_topm: bool = True,
    permute_mean: bool = True,
    pos_ids: Optional[List[str]] = None,
    neg_ids: Optional[List[str]] = None,
    limit_lines_to: Optional[List[str]] = None,
):
    rng = np.random.default_rng(seed)
    ensure_dir(out_dir)

    # Build symbol map once (full matrix)
    symmap = build_symbol_map(effect.columns.tolist())

    # Gene set
    if use_up_only:
        genes_set = sorted(set([g.upper() for g in genes_up]))
        set_label = "UP_ONLY"
    else:
        genes_set = sorted(set([g.upper() for g in (genes_up + genes_down)]))
        set_label = "UP_PLUS_DOWN"

    # Average duplicates to symbol-level matrix (all lines)
    M_all, present, missing = subset_matrix_by_symbols(effect, symmap, genes_set)
    print(f"[SET:{cohort_name}] Present in CRISPR: {len(present)}/{len(genes_set)} | Missing first 10: {missing[:10]}")

    if len(present) == 0:
        raise RuntimeError("No genes from the set found in CRISPRGeneEffect columns.")

    # Choose cohort lines
    if limit_lines_to is not None:
        keep = [mid for mid in M_all.index if mid in set(limit_lines_to)]
        M_all = M_all.loc[keep]
        meta_used = meta.loc[keep]
    else:
        meta_used = meta.loc[M_all.index]

    # ---- NaN impute (symbol-wise medians across cohort lines) ----
    M = M_all.copy()
    V = M.values.astype(float)
    col_median = np.nanmedian(V, axis=0)  # ignores NaN
    # If a column is all-NaN, set median to 0 (so it won’t contribute bias)
    col_median[~np.isfinite(col_median)] = 0.0
    nan_r, nan_c = np.where(~np.isfinite(V))
    if len(nan_r):
        V[nan_r, nan_c] = col_median[nan_c]
        M.iloc[:, :] = V

    # Define positive/negative line sets
    if pos_ids is None or neg_ids is None:
        pos_mask = meta_used["is_AML"].fillna(False).values
        pos_ids = M.index[pos_mask].tolist()
        neg_ids = M.index[~pos_mask].tolist()

    # Weights on the *present* list only
    weights_vec = load_weights(weights_mode, present, de_file, ranked_file, custom_weights_file)

    # Precompute cohort order & cache symbol vectors for speed
    cohort_ids = M.index.tolist()
    cohort_indexer = pd.Index(effect.index).get_indexer(cohort_ids)  # not used below but here for completeness
    sym_to_vec: Dict[str, np.ndarray] = {}
    for j, g in enumerate(present):
        sym_to_vec[g] = V[:, j]  # already imputed & cohort-sliced

    # Per-line metrics
    arr = V  # lines x genes (present)
    mean_scores = weighted_mean_rowwise(arr, weights_vec)
    topm_scores = top_m_rowwise(arr, topm)
    k1 = (arr < thr1).sum(axis=1).astype(float)
    k2 = (arr < thr2).sum(axis=1).astype(float)

    scores_df = pd.DataFrame({
        "ModelID": cohort_ids,
        "ses_mean": mean_scores,
        "ses_topm": topm_scores,
        f"k_lt_{thr1}": k1,
        f"k_lt_{thr2}": k2,
        "is_AML": [mid in meta.index[meta["is_AML"].fillna(False)] for mid in cohort_ids],
    }).set_index("ModelID")

    # Extract groups
    pos_mean = scores_df.loc[pos_ids, "ses_mean"].values
    neg_mean = scores_df.loc[neg_ids, "ses_mean"].values
    pos_topm = scores_df.loc[pos_ids, "ses_topm"].values
    neg_topm = scores_df.loc[neg_ids, "ses_topm"].values

    auc_mean, p_mwu_mean = compute_auc_mwu(pos_mean, neg_mean)
    auc_topm, p_mwu_topm = compute_auc_mwu(pos_topm, neg_topm)
    print(f"[EVAL:{cohort_name}] ses_mean : AUC={auc_mean:.3f} | MWU p={p_mwu_mean:.2e} (AML more essential)")
    print(f"[EVAL:{cohort_name}] ses_top{topm}: AUC={auc_topm:.3f} | MWU p={p_mwu_topm:.2e} (AML more essential)")

    # --------- Permutations (decile-matched by AML median) ----------
    emp_p_mean, emp_p_topm = None, None
    if permute_mean or permute_topm:
        # Build AML median per column over full matrix (to define bins)
        aml_ids_all = meta.index[meta["is_AML"].fillna(False)]
        aml_ids_all = [i for i in aml_ids_all if i in effect.index]
        med_full = effect.loc[aml_ids_all].median(axis=0)

        # Convert full medians to symbol medians
        # (use symmap; average duplicate columns)
        sym_median_full: Dict[str, float] = {}
        for sym, cols in symmap.items():
            vals = med_full[cols].values
            vals = vals[np.isfinite(vals)]
            if len(vals):
                sym_median_full[sym] = float(np.mean(vals))
        sym_median_full = pd.Series(sym_median_full)

        # Define decile bins for present set & universe
        target_bins = decile_bins(sym_median_full.loc[sym_median_full.index.intersection(present)])
        all_bins    = decile_bins(sym_median_full)
        univ_by_bin: Dict[int, np.ndarray] = {}
        for b, idx in all_bins.groupby(all_bins.values):
            univ_by_bin[int(b)] = idx.index.values

        # Helper to sample matched symbol list
        def sample_matched(rng_local: np.random.Generator) -> List[str]:
            out: List[str] = []
            counts = target_bins.value_counts()
            for b, cnt in counts.items():
                pool = univ_by_bin.get(int(b), np.array([], dtype=str))
                if len(pool) == 0:
                    continue
                replace = cnt > len(pool)
                pick = rng_local.choice(pool, size=int(cnt), replace=replace)
                out.extend(pick.tolist())
            # exact length match (rare mismatch → fill random from all universe)
            if len(out) < len(present):
                all_univ = np.concatenate(list(univ_by_bin.values())) if len(univ_by_bin) else np.array([], dtype=str)
                need = len(present) - len(out)
                if len(all_univ) >= need:
                    out.extend(rng_local.choice(all_univ, size=need, replace=False).tolist())
            return out[:len(present)]

        # Progress printing
        def should_print(i: int, total: int) -> bool:
            if total < 20: return True
            step = max(1, total // 10)
            return (i % step) == 0

        perm_auc_mean, perm_auc_topm = [], []
        try:
            for r in range(1, n_perm + 1):
                syms_perm = sample_matched(rng)

                # Build permuted matrix quickly from cache
                cols = [sym_to_vec[s] for s in syms_perm if s in sym_to_vec]
                if len(cols) == 0:
                    continue
                P = np.column_stack(cols)  # lines x genes

                if permute_topm:
                    pt = top_m_rowwise(P, topm)
                    ppos = pt[[mid in pos_ids for mid in scores_df.index]]
                    pneg = pt[[mid in neg_ids for mid in scores_df.index]]
                    auc_p, _ = compute_auc_mwu(ppos, pneg)
                    perm_auc_topm.append(auc_p)

                if permute_mean:
                    if weights_vec is not None and len(cols) == len(present):
                        # shuffle original weights — preserve distribution
                        w_perm = rng.permutation(weights_vec)
                        pm = weighted_mean_rowwise(P, w_perm)
                    else:
                        pm = weighted_mean_rowwise(P, None)
                    ppos = pm[[mid in pos_ids for mid in scores_df.index]]
                    pneg = pm[[mid in neg_ids for mid in scores_df.index]]
                    auc_p, _ = compute_auc_mwu(ppos, pneg)
                    perm_auc_mean.append(auc_p)

                if should_print(r, n_perm):
                    print(f"[PERM:{cohort_name}] {r}/{n_perm}…", flush=True)
        except KeyboardInterrupt:
            print(f"\n[PERM:{cohort_name}] Interrupted at {len(perm_auc_mean) or len(perm_auc_topm)} perms — saving partial nulls.")

        # Empirical p (right tail: null AUC ≥ real AUC)
        if permute_mean and len(perm_auc_mean):
            emp_p_mean = (1.0 + np.sum(np.array(perm_auc_mean) >= auc_mean)) / (1.0 + len(perm_auc_mean))
            print(f"[PERM:{cohort_name}] ses_mean : real AUC={auc_mean:.3f} | empirical p={emp_p_mean:.4f}")
        if permute_topm and len(perm_auc_topm):
            emp_p_topm = (1.0 + np.sum(np.array(perm_auc_topm) >= auc_topm)) / (1.0 + len(perm_auc_topm))
            print(f"[PERM:{cohort_name}] ses_top{topm}: real AUC={auc_topm:.3f} | empirical p={emp_p_topm:.4f}")

    # ---- Save per-line scores ----
    scores_out = out_dir / f"set_essentiality.{sanitize(cohort_name)}.scores.csv"
    scores_df.to_csv(scores_out)
    print(f"[SAVE] Per-line scores → {scores_out}")

    # ---- Save summary JSON ----
    summary = {
        "cohort": cohort_name,
        "set_label": set_label,
        "n_genes_present": int(len(present)),
        "n_genes_missing": int(len(missing)),
        "topm": int(topm),
        "thr1": float(thr1),
        "thr2": float(thr2),
        "weights_mode": weights_mode,
        "auc": {"ses_mean": float(auc_mean), "ses_topm": float(auc_topm)},
        "mwu_p": {"ses_mean": float(p_mwu_mean), "ses_topm": float(p_mwu_topm)},
        "perm_p": {"ses_mean": None if emp_p_mean is None else float(emp_p_mean),
                   "ses_topm": None if emp_p_topm is None else float(emp_p_topm)},
        "counts": {"n_pos": int(len(pos_ids)), "n_neg": int(len(neg_ids)), "n_lines": int(scores_df.shape[0])},
    }
    js_out = out_dir / f"set_essentiality.{sanitize(cohort_name)}.summary.json"
    with open(js_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Summary JSON → {js_out}")

    # ---- Plots ----
    plt.figure(figsize=(12, 8))
    grid_h = 2 if (emp_p_mean is not None or emp_p_topm is not None) else 1

    ax = plt.subplot(grid_h, 2, 1)
    dfp = scores_df.reset_index()
    dfp["group"] = np.where(dfp["is_AML"], "AML", "Non-AML")
    dfp["-ses_mean"] = -dfp["ses_mean"]
    sns.violinplot(data=dfp, x="group", y="-ses_mean", inner="box", ax=ax)
    ax.set_title(f"{cohort_name} | -ses_mean (AUC={auc_mean:.3f}, MWU p={p_mwu_mean:.1e})")
    ax.set_xlabel(""); ax.set_ylabel("More negative → higher")

    ax = plt.subplot(grid_h, 2, 2)
    dfp["-ses_topm"] = -dfp["ses_topm"]
    sns.violinplot(data=dfp, x="group", y="-ses_topm", inner="box", ax=ax)
    ax.set_title(f"{cohort_name} | -ses_top{topm} (AUC={auc_topm:.3f}, MWU p={p_mwu_topm:.1e})")
    ax.set_xlabel(""); ax.set_ylabel("More negative → higher")

    row = 2
    if emp_p_mean is not None:
        ax = plt.subplot(grid_h, 2, (row-1)*2+1)
        ax.hist(perm_auc_mean, bins=30, alpha=0.8)
        ax.axvline(auc_mean, color="red", lw=2, label=f"real={auc_mean:.3f}")
        ax.set_title(f"Permutation null: AUC(ses_mean), p={emp_p_mean:.4f}")
        ax.set_xlabel("AUC"); ax.legend()

    if emp_p_topm is not None:
        ax = plt.subplot(grid_h, 2, (row-1)*2+2)
        ax.hist(perm_auc_topm, bins=30, alpha=0.8)
        ax.axvline(auc_topm, color="red", lw=2, label=f"real={auc_topm:.3f}")
        ax.set_title(f"Permutation null: AUC(ses_top{topm}), p={emp_p_topm:.4f}")
        ax.set_xlabel("AUC"); ax.legend()

    plt.tight_layout()
    fig_out = out_dir / f"set_essentiality.{sanitize(cohort_name)}_plots.png"
    plt.savefig(fig_out, dpi=200)
    plt.close()
    print(f"[SAVE] Plots → {fig_out}")

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="DepMap set essentiality (fast, NaN-safe)")
    ap.add_argument("--depmap-dir",    type=Path, default=DEFAULT_DEPMAP_DIR)
    ap.add_argument("--score-dir",     type=Path, default=DEFAULT_SCORE_DIR)
    ap.add_argument("--gene-files",    type=Path, nargs="*", default=None,
                    help="Optional explicit gene files; unioned. If set, sign is ignored (treated as 'up').")
    ap.add_argument("--use-up-only",   action="store_true", help="Use only scoring_up.txt (ignore down).")
    # weights
    ap.add_argument("--weights",       type=str, default="none", choices=["none","de","depmap","file"])
    ap.add_argument("--weights-file",  type=Path, default=None)
    ap.add_argument("--de-file",       type=Path, default=DEFAULT_DE_FILE)
    ap.add_argument("--depmap-ranked", type=Path, default=DEFAULT_RANKED)
    # metrics / thresholds
    ap.add_argument("--topm",          type=int, default=3)
    ap.add_argument("--thr1",          type=float, default=-0.5)
    ap.add_argument("--thr2",          type=float, default=-0.3)
    # permutations
    ap.add_argument("--n-perm",        type=int, default=1000)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--no-permute-topm", action="store_true")
    ap.add_argument("--no-permute-mean", action="store_true")
    # subtype analysis
    ap.add_argument("--subset-file",   type=Path, default=None,
                    help="CSV with columns: ModelID,subset (AML lines). Compares each subset vs other AML.")
    # out
    ap.add_argument("--out-dir",       type=Path, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    depmap_dir = args.depmap_dir.resolve()
    score_dir  = args.score_dir.resolve()
    out_dir    = (args.out_dir or (score_dir / "depmap_combo")).resolve()
    ensure_dir(out_dir)

    print("="*80)
    print("DepMap Combination / Set Essentiality")
    print("="*80)
    print(f"DepMap dir : {depmap_dir}")
    if args.gene_files:
        print(f"Gene files : {', '.join(map(str, args.gene_files))}")
    else:
        print(f"Score dir  : {score_dir}")
        print(f"Gene files : <auto from score-dir>{' (UP ONLY)' if args.use_up_only else ''}")
    print(f"Out dir    : {out_dir}")
    print(f"topM       : {args.topm}")
    print(f"k-of-n     : thr1={args.thr1}, thr2={args.thr2}")
    print(f"Permutations: n={args.n_perm}")
    print(f"Weights    : {args.weights}")
    if args.weights == "file":
        print(f"  weights-file: {args.weights_file}")
    elif args.weights == "de":
        print(f"  de-file     : {args.de_file}")
    elif args.weights == "depmap":
        print(f"  depmap-ranked: {args.depmap_ranked}")
    print("="*80)

    # Load DepMap matrices
    meta   = load_metadata(depmap_dir)
    effect = load_crispr_effect(depmap_dir)
    have   = [i for i in meta.index if i in effect.index]
    meta   = meta.loc[have]
    effect = effect.loc[have]
    print(f"[META] AML lines in metadata: {meta['is_AML'].sum()}")
    print(f"[CRISPR] Effect matrix: {effect.shape[0]} lines x {effect.shape[1]} gene-columns")
    print(f"[META] Lines with CRISPR data: {len(have)} / {effect.shape[0]}")

    # Gene sets
    if args.gene_files:
        genes_union: List[str] = []
        for gf in args.gene_files:
            if not gf.exists():
                print(f"[WARN] Gene file not found: {gf}")
                continue
            genes_union.extend(read_gene_list(gf))
        genes_union = list(dict.fromkeys([g.upper() for g in genes_union]))
        genes_up, genes_down = genes_union, []
        print(f"[SET] Loaded {len(genes_union)} unique symbols from custom files.")
    else:
        up, dn = load_scoring_files(score_dir, up_only=args.use_up_only)
        genes_up, genes_down = up, dn
        print(f"[AUTO] {score_dir}")
        print(f"[AUTO] up={len(up)} | down={len(dn)} (use_up_only={args.use_up_only})")

    # Global AML vs non-AML
    evaluate_cohort(
        cohort_name="GLOBAL_AML_vs_nonAML",
        effect=effect,
        meta=meta,
        genes_up=genes_up,
        genes_down=genes_down,
        out_dir=out_dir,
        weights_mode=args.weights,
        de_file=args.de_file,
        ranked_file=args.depmap_ranked,
        custom_weights_file=args.weights_file,
        use_up_only=args.use_up_only,
        topm=args.topm,
        thr1=args.thr1,
        thr2=args.thr2,
        n_perm=args.n_perm,
        seed=args.seed,
        permute_topm=(not args.no_permute_topm),
        permute_mean=(not args.no_permute_mean),
    )

    # Optional: AML subsets vs other AML
    if args.subset_file and args.subset_file.exists():
        df_sub = pd.read_csv(args.subset_file)
        cols = {c.lower(): c for c in df_sub.columns}
        if "modelid" not in cols or "subset" not in cols:
            raise RuntimeError("--subset-file must have columns: ModelID, subset")
        df_sub[cols["modelid"]] = df_sub[cols["modelid"]].astype(str)
        df_sub[cols["subset"]]  = df_sub[cols["subset"]].astype(str)

        aml_ids = meta.index[meta["is_AML"].fillna(False)].tolist()
        df_sub = df_sub[df_sub[cols["modelid"]].isin(aml_ids)]
        if df_sub.empty:
            print("[SUBSET] No AML ModelIDs from subset-file present with CRISPR.")
        else:
            for subset_name, g in df_sub.groupby(cols["subset"]):
                pos_ids = g[cols["modelid"]].tolist()
                neg_ids = [i for i in aml_ids if i not in set(pos_ids)]
                if len(pos_ids)==0 or len(neg_ids)==0:
                    continue
                print(f"\n[SUBSET] {subset_name}: {len(pos_ids)} vs other AML {len(neg_ids)}")
                evaluate_cohort(
                    cohort_name=f"SUBSET_{subset_name}",
                    effect=effect, meta=meta,
                    genes_up=genes_up, genes_down=genes_down,
                    out_dir=out_dir, weights_mode=args.weights,
                    de_file=args.de_file, ranked_file=args.depmap_ranked,
                    custom_weights_file=args.weights_file,
                    use_up_only=args.use_up_only,
                    topm=args.topm, thr1=args.thr1, thr2=args.thr2,
                    n_perm=args.n_perm, seed=args.seed,
                    permute_topm=(not args.no_permute_topm),
                    permute_mean=(not args.no_permute_mean),
                    pos_ids=pos_ids, neg_ids=neg_ids,
                    limit_lines_to=aml_ids,
                )

    print("\nDone.")

if __name__ == "__main__":
    main()
