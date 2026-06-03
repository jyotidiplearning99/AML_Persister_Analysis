import tarfile
import urllib.request
from pathlib import Path
import pandas as pd
import random
import re

OUTDIR = Path("GSE154873_breast_celllines_uploads")
OUTDIR.mkdir(exist_ok=True)

URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154873/suppl/GSE154873_RAW.tar"
TAR_PATH = OUTDIR / "GSE154873_RAW.tar"

N_CELLS = 10000
SEED = 42

BARCODE_RE = re.compile(r"^[ACGT]{12,20}(-\d+)?$")

def looks_like_cell_barcode(x):
    x = str(x).strip()
    return bool(BARCODE_RE.match(x))

if not TAR_PATH.exists():
    print("Downloading GSE154873 RAW tar...")
    urllib.request.urlretrieve(URL, TAR_PATH)

EXTRACT_DIR = OUTDIR / "extracted"
EXTRACT_DIR.mkdir(exist_ok=True)

print("Extracting...")
with tarfile.open(TAR_PATH, "r") as tar:
    tar.extractall(EXTRACT_DIR)

count_files = [
    f for f in EXTRACT_DIR.rglob("*")
    if f.is_file()
    and "counts" in f.name.lower()
    and f.name.endswith(".txt.gz")
]

print("\nDetected count files:")
for i, f in enumerate(count_files):
    print(i, f.name)

for f in count_files:
    print("\n===================================================")
    print("Checking:", f.name)

    preview = pd.read_csv(f, sep="\t", compression="gzip", nrows=5)
    print("Preview shape:", preview.shape)
    print("First 10 columns:", preview.columns[:10].tolist())

    first_col = str(preview.columns[0])

    if looks_like_cell_barcode(first_col):
        print("ERROR: First column is a cell barcode, not a gene column.")
        print("This file cannot be safely converted to PersisterAI CSV without a gene list.")
        print("Skipping:", f.name)
        continue

    # If first column looks like gene/symbol/unnamed row index, try conversion
    df = pd.read_csv(f, sep="\t", compression="gzip")
    first_col = df.columns[0]

    df = df.rename(columns={first_col: "gene"})

    # Basic safety checks
    if df["gene"].astype(str).str.match(r"^[0-9]+$").mean() > 0.5:
        print("ERROR: gene column looks numeric, not gene symbols.")
        print("Skipping:", f.name)
        continue

    cell_cols = [c for c in df.columns if c != "gene"]

    if len(cell_cols) < 100:
        print("ERROR: too few cell columns detected.")
        print("Skipping:", f.name)
        continue

    random.seed(SEED)
    n_take = min(N_CELLS, len(cell_cols))
    selected_cells = random.sample(cell_cols, n_take)

    out = df[["gene"] + selected_cells].copy()

    # Convert expression columns
    for c in selected_cells:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out = out.groupby("gene", as_index=False).sum()

    if out.shape[0] < 5000:
        print("ERROR: final gene count too small:", out.shape[0])
        print("Skipping because this is not a valid gene-by-cell matrix.")
        continue

    clean_name = f.name.replace(".txt.gz", "")
    out_csv = OUTDIR / f"{clean_name}_{n_take}cells_persisterai_upload.csv"

    out.to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    print("Final shape genes x cells:", out.shape[0], "x", out.shape[1] - 1)

print("\nDone.")