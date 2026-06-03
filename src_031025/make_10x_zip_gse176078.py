import gzip
import zipfile
from pathlib import Path
import shutil

src = Path("GSE176078_BRCA_ALL_TUMOR_ATLAS_20000cells.mtx")
out_zip = Path("GSE176078_BRCA_ALL_TUMOR_ATLAS_20000cells_10x_standard.zip")

required = ["matrix.mtx", "barcodes.tsv", "features.tsv"]

for f in required:
    if not (src / f).exists():
        raise FileNotFoundError(f"Missing {src / f}")

tmp = Path("GSE176078_BRCA_10x_standard")
tmp.mkdir(exist_ok=True)

# gzip matrix.mtx
with open(src / "matrix.mtx", "rb") as f_in:
    with gzip.open(tmp / "matrix.mtx.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# gzip barcodes.tsv
with open(src / "barcodes.tsv", "rb") as f_in:
    with gzip.open(tmp / "barcodes.tsv.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# gzip features.tsv
with open(src / "features.tsv", "rb") as f_in:
    with gzip.open(tmp / "features.tsv.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# Also create genes.tsv.gz for older parsers
if (src / "genes.tsv").exists():
    with open(src / "genes.tsv", "rb") as f_in:
        with gzip.open(tmp / "genes.tsv.gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

# zip files at root level
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for f in tmp.iterdir():
        z.write(f, arcname=f.name)

print("Saved:", out_zip)
print("Size MB:", round(out_zip.stat().st_size / 1024 / 1024, 2))