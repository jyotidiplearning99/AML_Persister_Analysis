import argparse
import random
import tarfile
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread, mmwrite
from scipy import sparse


URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176078/suppl/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz"


def safe_extract(tar, path):
    path = Path(path).resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(path)):
            raise RuntimeError(f"Unsafe tar path detected: {member.name}")
    tar.extractall(path)


def find_file(base, patterns):
    for pattern in patterns:
        hits = list(base.rglob(pattern))
        if hits:
            return hits[0]
    return None


def choose_gene_names(genes_file):
    genes_df = pd.read_csv(genes_file, sep="\t", header=None)

    if genes_df.shape[1] >= 2:
        col0 = genes_df.iloc[:, 0].astype(str)
        col1 = genes_df.iloc[:, 1].astype(str)

        if col0.str.startswith("ENSG").mean() > 0.5 and col1.str.startswith("ENSG").mean() < 0.5:
            genes = col1.tolist()
            print("Using genes.tsv column 2 as gene symbols.")
        else:
            genes = col0.tolist()
            print("Using genes.tsv column 1 as gene names.")
    else:
        genes = genes_df.iloc[:, 0].astype(str).tolist()
        print("Using single-column genes.tsv.")

    genes = [g.split(".")[0] for g in genes]
    return genes


def write_mtx_zip(matrix_genes_by_cells, genes, barcodes, out_zip):
    out_zip = Path(out_zip)
    tmpdir = out_zip.with_suffix("")
    tmpdir.mkdir(parents=True, exist_ok=True)

    matrix_path = tmpdir / "matrix.mtx"
    barcodes_path = tmpdir / "barcodes.tsv"
    genes_path = tmpdir / "genes.tsv"
    features_path = tmpdir / "features.tsv"

    mmwrite(str(matrix_path), matrix_genes_by_cells.tocoo())

    pd.Series(barcodes).to_csv(
        barcodes_path, sep="\t", index=False, header=False
    )

    pd.Series(genes).to_csv(
        genes_path, sep="\t", index=False, header=False
    )

    features = pd.DataFrame({
        "gene_id": genes,
        "gene_symbol": genes,
        "feature_type": "Gene Expression"
    })

    features.to_csv(
        features_path, sep="\t", index=False, header=False
    )

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in [matrix_path, barcodes_path, genes_path, features_path]:
            z.write(f, arcname=f.name)

    print("Saved:", out_zip)
    print("Size MB:", round(out_zip.stat().st_size / 1024 / 1024, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cells", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="output")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tar_path = outdir / "GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz"
    extract_dir = outdir / "extracted"
    extract_dir.mkdir(exist_ok=True)

    if not tar_path.exists():
        print("Downloading GSE176078 tar.gz...")
        urllib.request.urlretrieve(URL, tar_path)

    print("Tar file:", tar_path)
    print("Tar size MB:", round(tar_path.stat().st_size / 1024 / 1024, 2))

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        safe_extract(tar, extract_dir)

    print("\nExtracted files:")
    for f in extract_dir.rglob("*"):
        if f.is_file():
            print(f.name, round(f.stat().st_size / 1024 / 1024, 2), "MB")

    mtx_file = find_file(extract_dir, ["*sparse*.mtx", "*.mtx"])
    barcodes_file = find_file(extract_dir, ["*barcodes.tsv", "*barcodes.txt", "*barcode.tsv"])
    genes_file = find_file(extract_dir, ["*genes.tsv", "*features.tsv", "*genes.txt"])
    metadata_file = find_file(extract_dir, ["metadata.csv", "*metadata*.csv", "*meta*.csv"])

    print("\nDetected files:")
    print("MTX:", mtx_file)
    print("Barcodes:", barcodes_file)
    print("Genes:", genes_file)
    print("Metadata:", metadata_file)

    if mtx_file is None or barcodes_file is None or genes_file is None:
        raise FileNotFoundError("Could not find matrix/barcodes/genes files.")

    print("\nReading matrix...")
    M = mmread(str(mtx_file)).tocsr()

    barcodes = pd.read_csv(barcodes_file, sep="\t", header=None)[0].astype(str).tolist()
    genes = choose_gene_names(genes_file)

    print("Matrix shape:", M.shape)
    print("Genes:", len(genes))
    print("Barcodes:", len(barcodes))

    if M.shape[0] == len(genes) and M.shape[1] == len(barcodes):
        print("Matrix orientation: genes x cells")
    elif M.shape[0] == len(barcodes) and M.shape[1] == len(genes):
        print("Matrix orientation: cells x genes. Transposing.")
        M = M.T.tocsr()
    else:
        raise ValueError("Matrix dimensions do not match genes/barcodes.")

    random.seed(args.seed)

    all_indices = list(range(len(barcodes)))
    n_take = min(args.n_cells, len(all_indices))
    selected = random.sample(all_indices, n_take)

    selected_barcodes = [barcodes[i] for i in selected]
    subM = M[:, selected].tocsr()

    print("\nSelected cells:", n_take)
    print("Selected matrix:", subM.shape)

    out_zip = outdir / f"GSE176078_BRCA_ALL_TUMOR_ATLAS_{n_take}cells.mtx.zip"
    write_mtx_zip(subM, genes, selected_barcodes, out_zip)

    if metadata_file is not None:
        meta = pd.read_csv(metadata_file)
        meta.to_csv(outdir / "GSE176078_metadata_preview.csv", index=False)

        print("\nMetadata shape:", meta.shape)
        print("Metadata columns:")
        print(meta.columns.tolist())

        for col in meta.columns:
            vals = meta[col].astype(str)
            if vals.nunique() <= 80:
                print("\nCOLUMN:", col)
                print(vals.value_counts().head(30))


if __name__ == "__main__":
    main()