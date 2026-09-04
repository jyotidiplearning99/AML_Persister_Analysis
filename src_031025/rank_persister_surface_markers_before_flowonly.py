#!/usr/bin/env python3
"""
rank_persister_surface_markers.py

Donor-aware discovery of a sparse surface-marker surrogate for the AML
Transformer-derived persister score.

Design
------
1. Reuse existing per-cell inference outputs:
      <DONOR>_predictions.csv
   with columns: cell_id, prob_persister, pred_label.
2. Join those barcodes back to the ORIGINAL 10x count matrix for each donor.
3. Restrict expression to the existing surface-candidate list (nominally 250 genes).
4. Define HIGH and LOW *within each donor* using top/bottom score quantiles
   (default: top/bottom quartile), rather than the global persister threshold.
5. For each marker compute donor-aware:
   - Spearman rho(expression, continuous persister score)
   - high-vs-low log-expression difference
   - detection fraction difference
   - single-marker ROC AUC
   - direction consistency across donors
   - donor-level Wilcoxon test + BH FDR
6. Build nonredundant 1/3/6/10-marker panels and evaluate them with
   OUTER donor-held-out cross-validation. Marker selection is repeated
   using TRAINING DONORS ONLY inside each outer fold.
7. Write ranked tables, final panel candidates, and grouped-CV performance.

This is a transcriptomic SURROGATE discovery analysis. It does NOT demonstrate
that the proteins themselves are differentially abundant on the cell surface.
Protein-level validation requires flow/CITE-seq/index sorting.

Tested conceptually for the AML_Persister_Analysis repository layout.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
RNG = np.random.default_rng(SEED)

GENE_COLS = [
    "gene", "gene_symbol", "symbol", "hgnc_symbol", "genesymbol",
    "target", "candidate", "Gene", "Gene_Symbol", "HGNC_symbol"
]
TIER_COLS = [
    "tier", "safety_tier", "Safety_Tier", "safety tier", "status"
]
PRIORITY_COLS = [
    "priority_score", "adjusted_priority", "priority", "score",
    "Priority_Score", "Adjusted_Priority"
]

KNOWN_VENDOR_ALIASES = {
    # public/known AML-panel examples; final vendor verification should be
    # performed after the data-driven top list is produced.
    "FLT3": ["CD135", "FLT3"],
    "CD33": ["CD33"],
    "ROR1": ["ROR1", "CD338"],
    "KDR": ["CD309", "VEGFR2", "KDR"],
    "EGFR": ["EGFR", "EGF RECEPTOR"],
    "MET": ["MET", "C-MET"],
    "EPHA2": ["EPHA2"],
    "EPHB2": ["EPHB2"],
    "ROS1": ["ROS1"],
}


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def clean_gene(x: str) -> str:
    x = str(x).strip().upper()
    x = x.rsplit(".", 1)[0]
    return x


def normalise_sample_key(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"_predictions$", "", x, flags=re.I)
    x = re.sub(r"\.csv$", "", x, flags=re.I)
    return x


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if not np.any(ok):
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty(n)
    tmp[order] = q
    out[np.where(ok)[0]] = tmp
    return out


def rank01(s: pd.Series, higher_better: bool = True) -> pd.Series:
    x = s.copy()
    if not higher_better:
        x = -x
    return x.rank(pct=True, method="average").fillna(0.0)


def find_first_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lut = {str(c).lower().replace(" ", "_"): c for c in df.columns}
    for c in candidates:
        k = c.lower().replace(" ", "_")
        if k in lut:
            return lut[k]
    return None


def safe_spearman(x, y) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 5 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    try:
        return float(spearmanr(x, y, nan_policy="omit").statistic)
    except Exception:
        return np.nan


def safe_auc(y, x) -> float:
    y = np.asarray(y)
    x = np.asarray(x)
    if len(np.unique(y)) < 2 or np.nanstd(x) == 0:
        return np.nan
    try:
        return float(roc_auc_score(y, x))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def candidate_table_score(path: Path) -> int:
    n = path.name.lower()
    score = 0
    for token, w in [
        ("surface", 8), ("candidate", 7), ("target", 6),
        ("ranking", 6), ("rank", 4), ("s7", 3), ("s8", 3),
        ("hpa", 3), ("priority", 3)
    ]:
        if token in n:
            score += w
    return score


def discover_candidate_tables(root: Path, limit=100) -> List[Tuple[int, Path]]:
    hits = []
    for pat in ("*.csv", "*.tsv", "*.xlsx", "*.xls"):
        for p in root.rglob(pat):
            if not p.is_file():
                continue
            s = candidate_table_score(p)
            if s > 0:
                hits.append((s, p))
    hits.sort(key=lambda z: (-z[0], len(str(z[1]))))
    return hits[:limit]


def read_table_head(path: Path, nrows=10) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=nrows)
    raise ValueError(path)


def load_candidate_table(path: Path) -> Tuple[pd.DataFrame, str]:
    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path)
    elif suf == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suf in (".xlsx", ".xls"):
        # Look across sheets for the one with gene + tier-like columns.
        xls = pd.ExcelFile(path)
        best = None
        for sheet in xls.sheet_names:
            d = pd.read_excel(path, sheet_name=sheet)
            gc = find_first_column(d, GENE_COLS)
            if gc is not None:
                quality = (
                    int(find_first_column(d, TIER_COLS) is not None) * 5
                    + int(find_first_column(d, PRIORITY_COLS) is not None) * 3
                    + min(len(d), 1000) / 1000
                )
                if best is None or quality > best[0]:
                    best = (quality, d, sheet)
        if best is None:
            raise ValueError(f"No gene-symbol column found in workbook {path}")
        _, df, sheet = best
        print(f"[candidate] using sheet: {sheet}")
    else:
        raise ValueError(f"Unsupported candidate file: {path}")

    gc = find_first_column(df, GENE_COLS)
    if gc is None:
        raise ValueError(
            f"Could not identify gene column in {path}; columns={list(df.columns)}"
        )
    df = df.copy()
    df["gene"] = df[gc].map(clean_gene)
    df = df[df["gene"].str.len() > 0].drop_duplicates("gene")
    return df, gc


def auto_choose_candidate_file(root: Path) -> Path:
    hits = discover_candidate_tables(root)
    viable = []
    for score, p in hits[:30]:
        try:
            h = read_table_head(p, 20)
            gc = find_first_column(h, GENE_COLS)
            if gc is None:
                continue
            tc = find_first_column(h, TIER_COLS)
            pc = find_first_column(h, PRIORITY_COLS)
            bonus = 10 * (tc is not None) + 5 * (pc is not None)
            viable.append((score + bonus, p, list(h.columns)))
        except Exception:
            pass
    if not viable:
        raise FileNotFoundError(
            "Could not auto-identify the 250-candidate surface ranking. "
            "Run with --discover and then pass --candidate-file PATH."
        )
    viable.sort(key=lambda z: -z[0])
    print("[candidate] auto candidates:")
    for x in viable[:5]:
        print(f"  score={x[0]:2d}  {x[1]}")
    return viable[0][1]


def discover_prediction_dirs(root: Path) -> List[Tuple[int, Path]]:
    by_parent = Counter()
    for p in root.rglob("*_predictions.csv"):
        if not p.is_file():
            continue
        try:
            h = pd.read_csv(p, nrows=3)
            cols = {c.lower() for c in h.columns}
            if {"cell_id", "prob_persister"}.issubset(cols):
                by_parent[p.parent] += 1
        except Exception:
            pass
    return sorted([(n, p) for p, n in by_parent.items()], reverse=True)


def auto_choose_pred_dir(root: Path) -> Path:
    dirs = discover_prediction_dirs(root)
    if not dirs:
        raise FileNotFoundError(
            "No <sample>_predictions.csv files with cell_id + prob_persister "
            f"found under {root}. Pass --pred-dir explicitly."
        )
    print("[predictions] candidate directories:")
    for n, p in dirs[:8]:
        print(f"  n={n:3d}  {p}")
    return dirs[0][1]


def extract_patient_id_from_path(path: Path) -> str:
    skip = {
        "outs", "filtered_feature_bc_matrix", "count", "matrix",
        "filtered_feature_bc_matrix.h5"
    }
    parts = [
        p for p in path.parts[::-1]
        if p not in skip and not p.endswith(".gz") and not p.endswith(".h5")
    ]
    # 1) Known cohort prefixes take priority (most specific, avoids false matches).
    for part in parts:
        if re.match(r"^(FH|FHRB|BERG)_\d+", part, flags=re.I):
            return part
    # 2) General alphanumeric sample-id pattern: letters+digits, e.g. FPM_AML_131_01,
    #    AML123, P07, D12. Deliberately conservative: must contain a digit and be
    #    a plausible id (not a generic dirname like 'data' or 'results').
    generic = {"data", "results", "outs", "aml", "aml_scrna", "scrna",
               "counts", "count", "raw", "processed", "samples", "cellranger"}
    for part in parts:
        base = part.rsplit(".", 1)[0]
        if base.lower() in generic:
            continue
        if re.search(r"\d", base) and re.match(r"^[A-Za-z0-9][A-Za-z0-9_\-\.]{1,}$", base):
            return base
    # 3) nearest meaningful parent
    if path.is_dir() and path.name == "filtered_feature_bc_matrix":
        return path.parent.name
    return path.parent.name


def discover_10x_samples(
    search_roots: Sequence[Path], report_collisions: bool = True
) -> Dict[str, Tuple[Path, str]]:
    found: Dict[str, Tuple[Path, str]] = {}
    collisions: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)
    for root in search_roots:
        if not root.exists():
            continue
        print(f"[10x] scanning: {root}")
        # H5 first
        for p in root.rglob("filtered_feature_bc_matrix.h5"):
            sid = extract_patient_id_from_path(p)
            if sid in found and found[sid][0] != p:
                collisions[sid].append((p, "H5"))
            else:
                found.setdefault(sid, (p, "H5"))
        # MTX second
        for p in root.rglob("filtered_feature_bc_matrix"):
            if p.is_dir() and (p / "matrix.mtx.gz").exists():
                sid = extract_patient_id_from_path(p)
                if sid in found and found[sid][0] != p:
                    collisions[sid].append((p, "MTX"))
                else:
                    found.setdefault(sid, (p, "MTX"))
    if report_collisions and collisions:
        print(
            "\n[10x][WARNING] Duplicate inferred donor IDs — multiple raw datasets "
            "mapped to the same ID. Only the FIRST is used; the others are listed "
            "so you can disambiguate (rename dirs or pass explicit --aml-root):"
        )
        for sid, extras in sorted(collisions.items()):
            print(f"  ID '{sid}' also matched:")
            print(f"    [USED]    {found[sid][1]:3s} {found[sid][0]}")
            for p, fmt in extras:
                print(f"    [IGNORED] {fmt:3s} {p}")
    return found


def match_sample(pred_name: str, sample_map: Dict[str, Tuple[Path, str]]):
    # exact
    if pred_name in sample_map:
        return sample_map[pred_name]
    # case-insensitive exact
    low = {k.lower(): v for k, v in sample_map.items()}
    if pred_name.lower() in low:
        return low[pred_name.lower()]
    # prefix/contains fallback
    matches = []
    for k, v in sample_map.items():
        if k.lower().startswith(pred_name.lower()) or pred_name.lower().startswith(k.lower()):
            matches.append((k, v))
    if len(matches) == 1:
        return matches[0][1]
    return None


# ---------------------------------------------------------------------------
# 10x loading, candidate-only
# ---------------------------------------------------------------------------

def read_features_tsv(path: Path) -> List[str]:
    compression = "gzip" if path.suffix == ".gz" else None
    f = pd.read_csv(path, sep="\t", header=None, compression=compression)
    col = 1 if f.shape[1] >= 2 else 0
    return [clean_gene(x) for x in f[col].astype(str).tolist()]


def load_mtx_candidates(
    mtx_dir: Path, candidate_genes: Sequence[str]
) -> Tuple[sparse.csr_matrix, np.ndarray, List[str], List[str]]:
    matrix_file = mtx_dir / "matrix.mtx.gz"
    features_file = mtx_dir / "features.tsv.gz"
    if not features_file.exists():
        features_file = mtx_dir / "genes.tsv.gz"
    barcodes_file = mtx_dir / "barcodes.tsv.gz"

    with gzip.open(matrix_file, "rt") as fh:
        X = mmread(fh).T.tocsr().astype(np.float32)
    genes = read_features_tsv(features_file)
    with gzip.open(barcodes_file, "rt") as fh:
        barcodes = [line.strip() for line in fh]

    # full-library size BEFORE candidate restriction
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    idx_by_gene = {}
    for j, g in enumerate(genes):
        idx_by_gene.setdefault(g, []).append(j)

    cols = []
    present = []
    for g in candidate_genes:
        js = idx_by_gene.get(g, [])
        if not js:
            continue
        if len(js) == 1:
            col = X[:, js[0]]
        else:
            col = X[:, js].sum(axis=1)
            col = sparse.csr_matrix(col)
        cols.append(col)
        present.append(g)

    if not cols:
        raise ValueError(f"No candidate genes found in {mtx_dir}")
    Xc = sparse.hstack(cols, format="csr").astype(np.float32)
    return Xc, lib, barcodes, present


def load_h5_candidates(
    h5_path: Path, candidate_genes: Sequence[str]
) -> Tuple[sparse.csr_matrix, np.ndarray, List[str], List[str]]:
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for 10x H5 input: pip install scanpy")

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    genes = [clean_gene(g) for g in adata.var_names]
    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    X = X.astype(np.float32)
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    barcodes = [str(x) for x in adata.obs_names]

    idx_by_gene = {}
    for j, g in enumerate(genes):
        idx_by_gene.setdefault(g, []).append(j)

    cols, present = [], []
    for g in candidate_genes:
        js = idx_by_gene.get(g, [])
        if not js:
            continue
        if len(js) == 1:
            col = X[:, js[0]]
        else:
            col = sparse.csr_matrix(X[:, js].sum(axis=1))
        cols.append(col)
        present.append(g)

    if not cols:
        raise ValueError(f"No candidate genes found in {h5_path}")
    Xc = sparse.hstack(cols, format="csr").astype(np.float32)
    return Xc, lib, barcodes, present


def load_sample_candidates(path: Path, fmt: str, candidate_genes: Sequence[str]):
    if fmt == "H5":
        return load_h5_candidates(path, candidate_genes)
    return load_mtx_candidates(path, candidate_genes)


def cpm_log1p_candidate_matrix(Xc: sparse.csr_matrix, lib: np.ndarray) -> np.ndarray:
    scale = 1e4 / np.maximum(lib, 1.0)
    Xn = Xc.multiply(scale[:, None])
    # only 250 columns; dense is fine here
    arr = Xn.toarray().astype(np.float32)
    np.log1p(arr, out=arr)
    return arr


# ---------------------------------------------------------------------------
# build donor data
# ---------------------------------------------------------------------------

def read_prediction_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lut = {c.lower(): c for c in df.columns}
    if "cell_id" not in lut or "prob_persister" not in lut:
        raise ValueError(f"Bad prediction file columns: {path} -> {list(df.columns)}")
    return df[[lut["cell_id"], lut["prob_persister"]]].rename(
        columns={lut["cell_id"]: "cell_id", lut["prob_persister"]: "score"}
    )


def assign_extreme_labels(score: pd.Series, q: float) -> pd.Series:
    # rank-based within donor: robust to cohort-level score calibration shifts
    r = score.rank(method="average", pct=True)
    label = pd.Series(np.nan, index=score.index, dtype=float)
    label[r <= q] = 0.0
    label[r >= (1.0 - q)] = 1.0
    return label


def build_donor_frames(
    pred_dir: Path,
    sample_map: Dict[str, Tuple[Path, str]],
    candidate_genes: Sequence[str],
    q: float,
    min_cells: int,
    cache_dir: Path,
    min_join_rate: float = 0.90,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    donor_frames = {}
    audit = []

    pred_files = sorted(pred_dir.glob("*_predictions.csv"))
    for pf in pred_files:
        donor = normalise_sample_key(pf.stem)
        matched = match_sample(donor, sample_map)
        row = {"donor": donor, "prediction_file": str(pf), "status": ""}
        if matched is None:
            row["status"] = "NO_10X_MATCH"
            audit.append(row)
            print(f"[skip] {donor}: no 10x match")
            continue

        raw_path, fmt = matched
        row["raw_path"] = str(raw_path)
        row["format"] = fmt

        cache = cache_dir / f"{donor}.pkl"
        if cache.exists():
            d = pd.read_pickle(cache)
            present = [g for g in candidate_genes if g in d.columns]
            row.update(
                status="CACHE",
                n_joined=len(d),
                n_candidates_present=len(present),
                cache_used=True,
            )
            # NOTE: a cached donor bypasses the live join-rate gate. Caches are only
            # written after passing the gate, so this is safe within a run; but if
            # you changed --min-join-rate since a prior run, delete
            # results/surface_surrogate/cache_joined_candidate_expression/ to force
            # a clean re-join. Cache hits are marked so they are auditable.
        else:
            try:
                pred = read_prediction_file(pf)
                Xc, lib, barcodes, present = load_sample_candidates(
                    raw_path, fmt, candidate_genes
                )
                expr = cpm_log1p_candidate_matrix(Xc, lib)
                e = pd.DataFrame(expr, columns=present)
                # log1p(CPM) > 0 iff the raw count was > 0, so detection
                # fractions can be recovered without storing a second 250-column matrix.
                e["cell_id"] = barcodes
                # Guard against barcode-format mismatch (the classic '-1' suffix /
                # donor-prefix problem) with a readable diagnostic rather than a
                # cryptic pandas validate="one_to_one" crash.
                dup_pred = int(pred["cell_id"].duplicated().sum())
                dup_expr = int(e["cell_id"].duplicated().sum())
                if dup_pred or dup_expr:
                    raise ValueError(
                        f"duplicate cell_id barcodes (pred={dup_pred}, expr={dup_expr}) "
                        "— barcode column is not unique; check barcode convention."
                    )
                d = pred.merge(e, on="cell_id", how="inner")
                join_rate = len(d) / max(1, len(pred))
                row["join_rate"] = round(join_rate, 4)
                ex_pred = pred["cell_id"].head(2).tolist()
                ex_expr = e["cell_id"].head(2).tolist()
                if join_rate < min_join_rate:
                    # A low join rate is almost always a barcode-format mismatch
                    # (e.g. '-1' suffix or donor prefix on one side only), NOT a
                    # real biological drop. Accepting such a donor would silently
                    # corrupt the panel, so we SKIP rather than warn.
                    row["status"] = f"LOW_JOIN_RATE_{join_rate:.2f}"
                    audit.append(row)
                    print(
                        f"[skip] {donor}: only {join_rate:.0%} of prediction barcodes "
                        f"joined ({len(d):,}/{len(pred):,}) < required {min_join_rate:.0%}. "
                        f"Likely barcode-format mismatch. pred e.g. {ex_pred}; "
                        f"expr e.g. {ex_expr}. Not cached, not analysed."
                    )
                    continue
                row.update(
                    n_predictions=len(pred),
                    n_raw_barcodes=len(barcodes),
                    n_joined=len(d),
                    n_candidates_present=len(present),
                )
                # cache full joined candidate-only table
                d.to_pickle(cache)
                row["status"] = "OK"
            except Exception as exc:
                row["status"] = f"ERROR: {exc}"
                audit.append(row)
                print(f"[error] {donor}: {exc}")
                continue

        if len(d) < min_cells:
            row["status"] = f"TOO_FEW_CELLS_{len(d)}"
            audit.append(row)
            continue

        d["donor"] = donor
        d["extreme_label"] = assign_extreme_labels(d["score"], q)
        n0 = int((d["extreme_label"] == 0).sum())
        n1 = int((d["extreme_label"] == 1).sum())
        row["n_low"] = n0
        row["n_high"] = n1
        if min(n0, n1) < max(20, int(min_cells * q / 2)):
            row["status"] = "INSUFFICIENT_EXTREMES"
            audit.append(row)
            continue

        donor_frames[donor] = d
        audit.append(row)
        print(
            f"[ok] {donor}: n={len(d):,}, low={n0:,}, high={n1:,}, "
            f"candidates={len([g for g in candidate_genes if g in d.columns])}"
        )

    return donor_frames, pd.DataFrame(audit)


# ---------------------------------------------------------------------------
# marker statistics
# ---------------------------------------------------------------------------

def donor_gene_stats(df: pd.DataFrame, genes: Sequence[str]) -> List[dict]:
    out = []
    y_ext = df["extreme_label"]
    extreme = y_ext.notna()
    y = y_ext[extreme].astype(int).to_numpy()
    score = df["score"].to_numpy()

    for g in genes:
        if g not in df.columns:
            continue
        x = df[g].to_numpy(dtype=float)
        detected = (x > 0).astype(float)

        rho = safe_spearman(x, score)

        xe = x[extreme.to_numpy()]
        de = detected[extreme.to_numpy()]
        hi = y == 1
        lo = y == 0

        auc = safe_auc(y, xe)
        if np.isfinite(auc):
            oriented_auc = max(auc, 1.0 - auc)
            auc_direction = "UP" if auc >= 0.5 else "DOWN"
            signed_auc = 2.0 * auc - 1.0
        else:
            oriented_auc = np.nan
            auc_direction = "NA"
            signed_auc = np.nan

        mean_hi = float(np.nanmean(xe[hi])) if np.any(hi) else np.nan
        mean_lo = float(np.nanmean(xe[lo])) if np.any(lo) else np.nan
        med_hi = float(np.nanmedian(xe[hi])) if np.any(hi) else np.nan
        med_lo = float(np.nanmedian(xe[lo])) if np.any(lo) else np.nan
        det_hi = float(np.nanmean(de[hi])) if np.any(hi) else np.nan
        det_lo = float(np.nanmean(de[lo])) if np.any(lo) else np.nan

        out.append({
            "gene": g,
            "rho": rho,
            "mean_high": mean_hi,
            "mean_low": mean_lo,
            "delta_mean_logexpr": mean_hi - mean_lo,
            "median_high": med_hi,
            "median_low": med_lo,
            "delta_median_logexpr": med_hi - med_lo,
            "det_high": det_hi,
            "det_low": det_lo,
            "det_delta": det_hi - det_lo,
            "auc_raw": auc,
            "auc_oriented": oriented_auc,
            "auc_signed_effect": signed_auc,
            "auc_direction": auc_direction,
            "n_extreme": len(y),
        })
    return out


def aggregate_marker_stats(
    donor_frames: Dict[str, pd.DataFrame],
    genes: Sequence[str],
    donors: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if donors is None:
        donors = sorted(donor_frames)
    long_rows = []
    for donor in donors:
        for r in donor_gene_stats(donor_frames[donor], genes):
            r["donor"] = donor
            long_rows.append(r)
    long = pd.DataFrame(long_rows)

    agg_rows = []
    for g, z in long.groupby("gene", sort=False):
        def med(c):
            v = z[c].to_numpy(float)
            return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan

        rho_vals = z["rho"].dropna().to_numpy(float)
        delta_vals = z["delta_mean_logexpr"].dropna().to_numpy(float)
        det_delta_vals = z["det_delta"].dropna().to_numpy(float)
        auc_vals = z["auc_oriented"].dropna().to_numpy(float)

        direction = np.sign(np.nanmedian(delta_vals)) if len(delta_vals) else 0
        if direction == 0 and len(rho_vals):
            direction = np.sign(np.nanmedian(rho_vals))
        if direction == 0:
            direction = 1

        n_dir = 0
        n_eval = 0
        for v in delta_vals:
            if np.isfinite(v) and v != 0:
                n_eval += 1
                n_dir += int(np.sign(v) == direction)
        consistency = n_dir / n_eval if n_eval else np.nan

        try:
            p_delta = float(wilcoxon(
                delta_vals, alternative="two-sided", zero_method="wilcox"
            ).pvalue) if len(delta_vals) >= 5 and np.any(delta_vals != 0) else np.nan
        except Exception:
            p_delta = np.nan

        try:
            p_rho = float(wilcoxon(
                rho_vals, alternative="two-sided", zero_method="wilcox"
            ).pvalue) if len(rho_vals) >= 5 and np.any(rho_vals != 0) else np.nan
        except Exception:
            p_rho = np.nan

        agg_rows.append({
            "gene": g,
            "n_donors": int(z["donor"].nunique()),
            "median_rho": med("rho"),
            "median_abs_rho": med("rho") if False else (
                float(np.nanmedian(np.abs(rho_vals))) if len(rho_vals) else np.nan
            ),
            "rho_positive_fraction": (
                float(np.mean(rho_vals > 0)) if len(rho_vals) else np.nan
            ),
            "median_delta_mean_logexpr": med("delta_mean_logexpr"),
            "median_delta_median_logexpr": med("delta_median_logexpr"),
            "median_det_high": med("det_high"),
            "median_det_low": med("det_low"),
            "median_det_delta": med("det_delta"),
            "median_auc_oriented": (
                float(np.nanmedian(auc_vals)) if len(auc_vals) else np.nan
            ),
            "direction": "UP_IN_HIGH" if direction > 0 else "DOWN_IN_HIGH",
            "direction_consistency": consistency,
            "p_donor_delta": p_delta,
            "p_donor_rho": p_rho,
        })

    agg = pd.DataFrame(agg_rows)
    if agg.empty:
        return agg, long
    agg["q_donor_delta"] = bh_fdr(agg["p_donor_delta"].to_numpy())
    agg["q_donor_rho"] = bh_fdr(agg["p_donor_rho"].to_numpy())

    # Flow-surrogate relevance composite.
    # Ranking, not a new biological score. Exposes all raw metrics.
    agg["rank_abs_rho"] = rank01(agg["median_abs_rho"])
    agg["rank_auc"] = rank01(agg["median_auc_oriented"])
    agg["rank_abs_delta"] = rank01(agg["median_delta_mean_logexpr"].abs())
    agg["rank_abs_det_delta"] = rank01(agg["median_det_delta"].abs())
    agg["rank_consistency"] = rank01(agg["direction_consistency"])

    # Detectability: marker must be measurable in at least one extreme.
    agg["max_median_detection"] = agg[
        ["median_det_high", "median_det_low"]
    ].max(axis=1)

    agg["surrogate_score"] = (
        0.25 * agg["rank_abs_rho"]
        + 0.25 * agg["rank_auc"]
        + 0.20 * agg["rank_abs_delta"]
        + 0.15 * agg["rank_abs_det_delta"]
        + 0.15 * agg["rank_consistency"]
    )
    agg.loc[agg["max_median_detection"] < 0.10, "surrogate_score"] *= 0.50
    agg.loc[agg["n_donors"] < max(5, int(0.5 * len(donors))), "surrogate_score"] *= 0.50

    agg = agg.sort_values(
        ["surrogate_score", "median_auc_oriented", "median_abs_rho"],
        ascending=False
    ).reset_index(drop=True)
    agg["surrogate_rank"] = np.arange(1, len(agg) + 1)
    return agg, long


# ---------------------------------------------------------------------------
# redundancy + nested donor-held-out panel evaluation
# ---------------------------------------------------------------------------

def sample_extreme_cells(
    donor_frames: Dict[str, pd.DataFrame],
    donors: Sequence[str],
    genes: Sequence[str],
    max_per_class_per_donor: int,
) -> pd.DataFrame:
    parts = []
    for d in donors:
        x = donor_frames[d]
        x = x[x["extreme_label"].notna()].copy()
        x = x[x["extreme_label"].isin([0.0, 1.0])]
        for label in (0.0, 1.0):
            z = x[x["extreme_label"] == label]
            if len(z) > max_per_class_per_donor:
                z = z.sample(
                    max_per_class_per_donor,
                    random_state=SEED + int(label) + (abs(hash(d)) % 10000)
                )
            z = z.copy()
            missing = [g for g in genes if g not in z.columns]
            for g in missing:
                z.loc[:, g] = 0.0
            keep = ["donor", "score", "extreme_label"] + list(genes)
            parts.append(z[keep])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def compute_redundancy_matrix(
    donor_frames: Dict[str, pd.DataFrame],
    donors: Sequence[str],
    genes: Sequence[str],
    max_cells_per_donor: int = 1500,
) -> pd.DataFrame:
    parts = []
    for d in donors:
        z = donor_frames[d]
        cols = [g for g in genes if g in z.columns]
        if not cols:
            continue
        if len(z) > max_cells_per_donor:
            z = z.sample(max_cells_per_donor, random_state=SEED + abs(hash(d)) % 10000)
        parts.append(z[cols])
    if not parts:
        return pd.DataFrame()
    allx = pd.concat(parts, ignore_index=True)
    # Spearman == Pearson on ranks
    return allx.rank().corr(method="pearson")


def select_nonredundant(
    stats: pd.DataFrame,
    donor_frames: Dict[str, pd.DataFrame],
    donors: Sequence[str],
    n: int,
    redundancy_threshold: float,
    candidate_pool: int = 30,
) -> List[str]:
    pool = stats.head(candidate_pool)["gene"].tolist()
    corr = compute_redundancy_matrix(donor_frames, donors, pool)
    selected = []
    for g in pool:
        if len(selected) >= n:
            break
        if g not in corr.index:
            selected.append(g)
            continue
        redundant = False
        for h in selected:
            if h in corr.columns:
                c = corr.loc[g, h]
                if np.isfinite(c) and abs(c) >= redundancy_threshold:
                    redundant = True
                    break
        if not redundant:
            selected.append(g)
    return selected


def donor_folds(donors: Sequence[str], n_splits: int) -> List[Tuple[List[str], List[str]]]:
    donors = np.array(sorted(donors), dtype=object)
    rng = np.random.default_rng(SEED)
    rng.shuffle(donors)
    n_splits = min(n_splits, len(donors))
    chunks = np.array_split(donors, n_splits)
    folds = []
    for i in range(n_splits):
        test = chunks[i].tolist()
        train = np.concatenate([c for j, c in enumerate(chunks) if j != i]).tolist()
        folds.append((train, test))
    return folds


def evaluate_panel_cv(
    donor_frames: Dict[str, pd.DataFrame],
    genes: Sequence[str],
    panel_sizes: Sequence[int],
    n_splits: int,
    redundancy_threshold: float,
    max_per_class_per_donor: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    donors = sorted(donor_frames)
    folds = donor_folds(donors, n_splits)
    perf_rows = []
    selection_rows = []

    for fold_id, (train_d, test_d) in enumerate(folds, start=1):
        print(
            f"[CV] fold {fold_id}/{len(folds)}: "
            f"train donors={len(train_d)}, test donors={len(test_d)}"
        )
        train_stats, _ = aggregate_marker_stats(donor_frames, genes, train_d)

        for size in panel_sizes:
            panel = select_nonredundant(
                train_stats, donor_frames, train_d, int(size),
                redundancy_threshold=redundancy_threshold,
                candidate_pool=max(30, int(size) * 4),
            )
            for g in panel:
                selection_rows.append({
                    "fold": fold_id,
                    "panel_size": size,
                    "gene": g,
                })

            if not panel:
                continue

            tr = sample_extreme_cells(
                donor_frames, train_d, panel, max_per_class_per_donor
            )
            te = sample_extreme_cells(
                donor_frames, test_d, panel, max_per_class_per_donor
            )
            if tr.empty or te.empty:
                continue

            # All selected panel genes should exist in all donors, but be defensive.
            panel2 = [g for g in panel if g in tr.columns and g in te.columns]
            if not panel2:
                continue

            model = Pipeline([
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=SEED,
                )),
            ])
            model.fit(tr[panel2], tr["extreme_label"].astype(int))
            prob = model.predict_proba(te[panel2])[:, 1]

            overall_auc = roc_auc_score(te["extreme_label"].astype(int), prob)
            perf_rows.append({
                "fold": fold_id,
                "panel_size": size,
                "n_markers": len(panel2),
                "markers": ";".join(panel2),
                "test_donors": ";".join(test_d),
                "n_train_cells": len(tr),
                "n_test_cells": len(te),
                "auc_overall_cells": overall_auc,
                "auc_median_test_donor": np.nan,  # filled below
            })

            donor_aucs = []
            start = len(perf_rows) - 1
            for d in test_d:
                m = te["donor"] == d
                if m.sum() == 0 or te.loc[m, "extreme_label"].nunique() < 2:
                    continue
                p_d = model.predict_proba(te.loc[m, panel2])[:, 1]
                auc_d = roc_auc_score(te.loc[m, "extreme_label"].astype(int), p_d)
                donor_aucs.append(auc_d)
            if donor_aucs:
                perf_rows[start]["auc_median_test_donor"] = float(np.median(donor_aucs))
                perf_rows[start]["auc_min_test_donor"] = float(np.min(donor_aucs))
                perf_rows[start]["auc_max_test_donor"] = float(np.max(donor_aucs))

    return pd.DataFrame(perf_rows), pd.DataFrame(selection_rows)


# ---------------------------------------------------------------------------
# final panel
# ---------------------------------------------------------------------------

def final_panel_from_full_data(
    full_stats: pd.DataFrame,
    donor_frames: Dict[str, pd.DataFrame],
    selection: pd.DataFrame,
    n_final: int,
    redundancy_threshold: float,
) -> pd.DataFrame:
    st = full_stats.copy()
    if not selection.empty:
        sel10 = selection[selection["panel_size"] == selection["panel_size"].max()]
        freq = sel10.groupby("gene")["fold"].nunique()
        denom = max(1, sel10["fold"].nunique())
        st["cv_selection_frequency"] = st["gene"].map(freq).fillna(0) / denom
    else:
        st["cv_selection_frequency"] = 0.0

    st["final_priority"] = (
        0.70 * st["surrogate_score"]
        + 0.30 * st["cv_selection_frequency"]
    )
    st = st.sort_values(
        ["final_priority", "cv_selection_frequency", "surrogate_score"],
        ascending=False
    )

    donors = sorted(donor_frames)
    pool = st.head(max(40, n_final * 5))["gene"].tolist()
    corr = compute_redundancy_matrix(donor_frames, donors, pool)
    selected = []
    for g in pool:
        if len(selected) >= n_final:
            break
        bad = False
        for h in selected:
            if g in corr.index and h in corr.columns:
                c = corr.loc[g, h]
                if np.isfinite(c) and abs(c) >= redundancy_threshold:
                    bad = True
                    break
        if not bad:
            selected.append(g)

    out = st[st["gene"].isin(selected)].copy()
    out["final_panel_order"] = out["gene"].map(
        {g: i + 1 for i, g in enumerate(selected)}
    )
    out["vendor_search_aliases"] = out["gene"].map(
        lambda g: ";".join(KNOWN_VENDOR_ALIASES.get(g, [g]))
    )
    return out.sort_values("final_panel_order")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def identify_surface_genes(cand: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Return (surface_gene_list, description) using HPA annotations in the candidate
    table. A gene is 'surface' if an HPA surface flag says yes, OR its subcellular
    location / protein class names a plasma-membrane / cell-surface / receptor /
    CD-marker / transporter / channel term. Falls back to protein_class if no
    explicit surface flag column exists.
    """
    cols = {str(c).lower().replace(" ", "_"): c for c in cand.columns}

    # 1) explicit yes/no surface flag
    flag_col = None
    for k, orig in cols.items():
        if "hpa_surface" in k or k in ("surface", "surface_protein", "is_surface",
                                       "cell_surface"):
            flag_col = orig
            break
    if flag_col is not None:
        vals = cand[flag_col].astype(str).str.strip().str.lower()
        mask = vals.isin(["yes", "true", "1", "surface", "y"])
        genes = cand.loc[mask, "gene"].dropna().map(clean_gene).drop_duplicates().tolist()
        if genes:
            return genes, f"HPA surface flag column '{flag_col}'"

    # 2) subcellular-location / protein-class keyword match
    loc_cols = [orig for k, orig in cols.items()
                if "subcellular" in k or "location" in k or "protein_class" in k
                or "protein_type" in k]
    surface_terms = [
        "plasma membrane", "cell membrane", "cell surface", "surface",
        "membrane", "receptor", "cd marker", "cd-marker", "transporter",
        "ion channel", "gpcr", "g-protein", "transmembrane"
    ]
    if loc_cols:
        text = cand[loc_cols].astype(str).agg(" ".join, axis=1).str.lower()
        mask = text.apply(lambda t: any(term in t for term in surface_terms))
        genes = cand.loc[mask, "gene"].dropna().map(clean_gene).drop_duplicates().tolist()
        if genes:
            return genes, f"subcellular/protein-class keyword match on {loc_cols}"

    return [], "no usable HPA surface annotation found"



def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument(
        "--project-root", type=Path,
        default=Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis")
    )
    ap.add_argument("--candidate-file", type=Path, default=None)
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument(
        "--aml-root", type=Path, action="append", default=[],
        help="Root containing donor 10x H5/MTX data. Can be repeated."
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Default: <project-root>/results/surface_surrogate"
    )
    ap.add_argument("--quantile", type=float, default=0.25)
    ap.add_argument("--min-cells", type=int, default=200)
    ap.add_argument(
        "--min-donors", type=int, default=24,
        help="Hard floor on usable donors. For the 39-donor AML application, "
             "do not lower below ~24 — a thinner panel is not trustworthy under "
             "donor-held-out CV."
    )
    ap.add_argument(
        "--expected-donors", type=int, default=39,
        help="Expected donor count for this cohort; used only to report the "
             "recovery fraction and warn if many donors are lost."
    )
    ap.add_argument(
        "--min-join-rate", type=float, default=0.90,
        help="Minimum fraction of a donor's prediction barcodes that must join "
             "to the 10x matrix. Below this the donor is SKIPPED (a low rate "
             "signals a barcode-format mismatch, not a real drop)."
    )
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument(
        "--max-per-class-per-donor", type=int, default=1500,
        help="Balanced cell cap used for multivariable CV only."
    )
    ap.add_argument(
        "--redundancy-threshold", type=float, default=0.75,
        help="Absolute Spearman expression correlation for pruning redundant markers."
    )
    ap.add_argument("--final-markers", type=int, default=10)
    ap.add_argument(
        "--surface-only", action="store_true",
        help="Restrict panel selection and CV to genes annotated as genuine cell-"
             "surface proteins (HPA). Per-marker stats are still computed for all "
             "candidates so the full ranking remains visible, but the designable "
             "flow panel is built only from antibody-targetable surface genes. "
             "Use this for the panel you actually take to BD/BioLegend."
    )
    ap.add_argument(
        "--discover", action="store_true",
        help="Only report likely input files/directories and exit."
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if not (0.05 <= args.quantile <= 0.45):
        raise ValueError("--quantile should be between 0.05 and 0.45")

    root = args.project_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root not found: {root}")

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else root / "results" / "surface_surrogate"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("AML PERSISTER SURFACE-SURROGATE DISCOVERY")
    print("=" * 88)
    print(f"Project root : {root}")
    print(f"Output       : {out_dir}")
    print(
        "Primary contrast: within-donor bottom "
        f"{args.quantile:.0%} vs top {args.quantile:.0%} persister score"
    )

    # candidate file
    candidate_file = args.candidate_file
    if candidate_file is None:
        candidate_file = auto_choose_candidate_file(root)
    print(f"[candidate] chosen: {candidate_file}")

    # prediction directory
    pred_dir = args.pred_dir
    if pred_dir is None:
        pred_dir = auto_choose_pred_dir(root)
    print(f"[predictions] chosen: {pred_dir}")

    # raw-data roots
    search_roots = list(args.aml_root)
    if not search_roots:
        # Conservative defaults: repo and known AML scratch location, if present.
        defaults = [
            root,
            Path("/scratch/project_2010751/AML_scRNA"),
            root.parent,
        ]
        search_roots = []
        seen = set()
        for d in defaults:
            if d.exists() and str(d.resolve()) not in seen:
                search_roots.append(d.resolve())
                seen.add(str(d.resolve()))
    print("[10x] search roots:")
    for x in search_roots:
        print(f"  {x}")

    if args.discover:
        print("\n--- Candidate table alternatives ---")
        for s, p in discover_candidate_tables(root)[:20]:
            print(f"{s:3d}  {p}")
        print("\n--- Prediction directories ---")
        for n, p in discover_prediction_dirs(root)[:20]:
            print(f"{n:3d}  {p}")
        print("\n--- 10x sample discovery ---")
        smap = discover_10x_samples(search_roots)
        print(f"Found {len(smap)} unique candidate sample IDs.")
        for k, v in list(sorted(smap.items()))[:100]:
            print(f"{k:25s} {v[1]:3s} {v[0]}")

        # Critical preflight: show which prediction files will actually match a 10x
        # sample, so missing/mismatched donors are caught BEFORE the multi-hour run.
        print("\n--- PREDICTION → 10x MATCH PREFLIGHT ---")
        # Use the SAME pred_dir the main run resolved (respects an explicit
        # --pred-dir); do NOT re-discover, or the preflight could inspect a
        # different directory than the real run.
        pred_files = sorted(pred_dir.glob("*_predictions.csv"))
        if pred_files:
            n_ok = 0
            n_miss = 0
            for pf in pred_files:
                donor = normalise_sample_key(pf.stem)
                matched = match_sample(donor, smap)
                if matched is None:
                    print(f"  [NO MATCH] {pf.name} -> {donor} -> (none)")
                    n_miss += 1
                else:
                    raw_path, fmt = matched
                    print(f"  [OK] {pf.name} -> {donor} -> {fmt} {raw_path}")
                    n_ok += 1
            print(
                f"\nSummary: {len(pred_files)} prediction files; "
                f"{n_ok} match a 10x sample, {n_miss} unmatched."
            )
            if n_miss:
                print(
                    "  → Unmatched donors will be DROPPED. If the count is high, "
                    "pass the correct --aml-root, or the sample naming differs "
                    "between predictions and 10x paths and needs reconciling."
                )
        else:
            print(f"  (no *_predictions.csv found under {pred_dir}; pass --pred-dir)")

        # ID collisions are a STOP condition for a run that could guide antibody
        # purchasing: discover_10x_samples() already printed the [USED]/[IGNORED]
        # detail above; here we make the verdict explicit.
        _seen: Dict[str, int] = defaultdict(int)
        for _root in search_roots:
            if not _root.exists():
                continue
            for _p in _root.rglob("filtered_feature_bc_matrix.h5"):
                _seen[extract_patient_id_from_path(_p)] += 1
            for _p in _root.rglob("filtered_feature_bc_matrix"):
                if _p.is_dir() and (_p / "matrix.mtx.gz").exists():
                    _seen[extract_patient_id_from_path(_p)] += 1
        _collide = {k: v for k, v in _seen.items() if v > 1}
        if _collide:
            print(
                "\n[PREFLIGHT][STOP] Duplicate inferred donor IDs detected "
                f"({len(_collide)}): {sorted(_collide)}.\n"
                "  Multiple raw datasets map to the same ID; the main run would "
                "silently keep only one. Resolve these (rename dirs or pass an "
                "explicit --aml-root) BEFORE launching — do not proceed to a "
                "panel that could guide antibody purchasing with an unexplained "
                "collision present."
            )
        else:
            print("\n[PREFLIGHT] No donor-ID collisions detected.")
        return

    cand, original_gene_col = load_candidate_table(candidate_file)
    tier_col = find_first_column(cand, TIER_COLS)
    priority_col = find_first_column(cand, PRIORITY_COLS)

    # Prefer the 250-row full ranking if available; retain all rows of supplied file.
    genes = cand["gene"].dropna().map(clean_gene).drop_duplicates().tolist()
    print(f"[candidate] loaded {len(genes)} unique genes")
    if len(genes) < 20:
        raise ValueError(
            f"Only {len(genes)} candidate genes found. "
            "This does not look like the intended surface ranking."
        )

    sample_map = discover_10x_samples(search_roots)
    print(f"[10x] discovered {len(sample_map)} unique sample IDs")

    donor_frames, audit = build_donor_frames(
        pred_dir=pred_dir,
        sample_map=sample_map,
        candidate_genes=genes,
        q=args.quantile,
        min_cells=args.min_cells,
        cache_dir=out_dir / "cache_joined_candidate_expression",
        min_join_rate=args.min_join_rate,
    )
    audit.to_csv(out_dir / "input_join_audit.csv", index=False)

    n_usable = len(donor_frames)
    recovery = n_usable / max(1, args.expected_donors)
    print(
        f"\n[cohort] usable donors = {n_usable} / expected {args.expected_donors} "
        f"({recovery:.0%} recovered)"
    )
    if recovery < 0.80:
        print(
            "[cohort][WARNING] More than 20% of expected donors were lost. "
            "Inspect input_join_audit.csv 'status' column below before trusting "
            "any panel — this usually means path/ID/barcode mismatches, not biology."
        )
        print(audit["status"].value_counts(dropna=False).to_string())

    if n_usable < args.min_donors:
        print("\nINPUT MATCHING FAILED / INCOMPLETE")
        print(audit["status"].value_counts(dropna=False).to_string())
        raise RuntimeError(
            f"Only {n_usable} donors could be joined, below --min-donors "
            f"{args.min_donors}. Inspect {out_dir/'input_join_audit.csv'} and rerun "
            "with the correct --aml-root and/or --pred-dir. Do NOT lower --min-donors "
            "to force the run — a panel from too few donors is not trustworthy under "
            "donor-held-out CV."
        )

    print(f"\n[analysis] usable donors = {len(donor_frames)}")
    print(f"[analysis] total joined cells = {sum(len(x) for x in donor_frames.values()):,}")

    full_stats, long_stats = aggregate_marker_stats(donor_frames, genes)
    if full_stats.empty:
        raise RuntimeError("No marker statistics could be computed.")

    # merge original target annotations
    annotation_cols = ["gene"]
    for c in cand.columns:
        if c == "gene":
            continue
        # Keep useful compact annotations; avoid huge tissue matrices in the main output.
        cl = str(c).lower()
        if any(t in cl for t in [
            "tier", "priority", "mechanism", "therapeutic",
            "hpa_surface", "antibody", "protein_class", "subcellular"
        ]):
            annotation_cols.append(c)
    annotation_cols = list(dict.fromkeys(annotation_cols))
    ann = cand[annotation_cols].drop_duplicates("gene")
    full_stats = full_stats.merge(ann, on="gene", how="left")

    full_stats.to_csv(out_dir / "surface_marker_ranked_all.csv", index=False)
    long_stats.to_csv(out_dir / "surface_marker_per_donor_metrics.csv", index=False)

    # Panel gene pool: optionally restrict to genuine surface proteins so the
    # designable flow panel contains only antibody-targetable targets. Per-marker
    # stats above still cover ALL candidates, so non-surface genes remain visible
    # in surface_marker_ranked_all.csv (they just can't enter the flow panel).
    panel_genes = genes
    if args.surface_only:
        surf, how = identify_surface_genes(cand)
        surf = [g for g in surf if g in set(genes)]
        print(f"\n[surface-only] {len(surf)}/{len(genes)} candidates are surface-"
              f"annotated ({how}).")
        if len(surf) < 6:
            raise RuntimeError(
                f"--surface-only left only {len(surf)} surface genes; too few for a "
                "6–10 marker panel. Check the HPA annotation columns in the candidate "
                "table, or run without --surface-only and filter afterwards."
            )
        panel_genes = surf
        # tag the full stats so the ranking table shows what is panel-eligible
        full_stats["surface_panel_eligible"] = full_stats["gene"].isin(set(surf))
        full_stats.to_csv(out_dir / "surface_marker_ranked_all.csv", index=False)
        print(f"[surface-only] panel selection & CV restricted to these "
              f"{len(surf)} genes.")

    # nested donor-held-out evaluation
    perf, selection = evaluate_panel_cv(
        donor_frames=donor_frames,
        genes=panel_genes,
        panel_sizes=[1, 3, 6, 10],
        n_splits=args.cv_folds,
        redundancy_threshold=args.redundancy_threshold,
        max_per_class_per_donor=args.max_per_class_per_donor,
    )
    perf.to_csv(out_dir / "panel_grouped_cv_performance.csv", index=False)
    selection.to_csv(out_dir / "panel_grouped_cv_selections.csv", index=False)

    # Final panel is drawn from the (optionally surface-restricted) pool.
    final_stats_pool = (
        full_stats[full_stats["gene"].isin(set(panel_genes))].copy()
        if args.surface_only else full_stats
    )
    final = final_panel_from_full_data(
        full_stats=final_stats_pool,
        donor_frames=donor_frames,
        selection=selection,
        n_final=args.final_markers,
        redundancy_threshold=args.redundancy_threshold,
    )
    final.to_csv(out_dir / "final_surface_panel_candidates.csv", index=False)

    # concise summaries
    cv_summary = (
        perf.groupby("panel_size")
        .agg(
            folds=("fold", "nunique"),
            median_auc=("auc_median_test_donor", "median"),
            min_fold_median_auc=("auc_median_test_donor", "min"),
            max_fold_median_auc=("auc_median_test_donor", "max"),
        )
        .reset_index()
        if not perf.empty
        else pd.DataFrame()
    )
    cv_summary.to_csv(out_dir / "panel_grouped_cv_summary.csv", index=False)

    summary = {
        "analysis": "AML persister surface-surrogate discovery",
        "usable_donors": len(donor_frames),
        "total_joined_cells": int(sum(len(x) for x in donor_frames.values())),
        "n_candidate_genes_input": len(genes),
        "surface_only_mode": bool(args.surface_only),
        "n_panel_pool_genes": len(panel_genes),
        "quantile_each_extreme": args.quantile,
        "candidate_file": str(candidate_file),
        "prediction_dir": str(pred_dir),
        "raw_search_roots": [str(x) for x in search_roots],
        "final_panel": final["gene"].tolist(),
        "caveat": (
            "Targets are transcript-level surrogates for a transcriptomic model score. "
            "Flow/CITE-seq/index-sort protein validation is required before claiming "
            "surface-protein discrimination of persister cells."
        ),
    }
    with open(out_dir / "analysis_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 88)
    print("TOP 20 TRANSCRIPT-LEVEL SURFACE SURROGATES")
    print("=" * 88)
    show_cols = [
        "surrogate_rank", "gene", "direction", "median_rho",
        "median_auc_oriented", "median_delta_mean_logexpr",
        "median_det_high", "median_det_low",
        "direction_consistency", "q_donor_delta", "surrogate_score"
    ]
    print(full_stats[show_cols].head(20).to_string(index=False))

    print("\n" + "=" * 88)
    print("FINAL NONREDUNDANT PANEL CANDIDATES")
    print("=" * 88)
    show2 = [
        "final_panel_order", "gene", "direction", "median_rho",
        "median_auc_oriented", "direction_consistency",
        "cv_selection_frequency", "final_priority", "vendor_search_aliases"
    ]
    print(final[show2].to_string(index=False))

    if not cv_summary.empty:
        print("\n" + "=" * 88)
        print("DONOR-HELD-OUT PANEL PERFORMANCE")
        print("=" * 88)
        print(cv_summary.to_string(index=False))

    print("\nOutputs:")
    for p in [
        "input_join_audit.csv",
        "surface_marker_ranked_all.csv",
        "surface_marker_per_donor_metrics.csv",
        "panel_grouped_cv_performance.csv",
        "panel_grouped_cv_summary.csv",
        "panel_grouped_cv_selections.csv",
        "final_surface_panel_candidates.csv",
        "analysis_summary.json",
    ]:
        print(f"  {out_dir / p}")

    print(
        "\nNEXT STEP: cross-reference the final 10 (and top ~20 backups) against "
        "BD/BioLegend clones/fluorochromes, then design a practical 6–10 color "
        "panel with AML/backbone gating kept separate."
    )


if __name__ == "__main__":
    main()