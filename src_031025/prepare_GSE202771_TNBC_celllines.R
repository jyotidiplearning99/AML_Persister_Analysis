suppressPackageStartupMessages({
  library(Matrix)
})

# ============================================================
# GSE202771 TNBC breast cancer cell-line scRNA-seq converter
# Purpose:
#   Convert downloaded double-gzipped RData count object into
#   PersisterAI-compatible 10x-style zip files.
#
# Output:
#   *_10000cells_10x_standard.zip
# ============================================================

# -------------------------
# User settings
# -------------------------
n_cells <- 10000
seed <- 42
outdir <- "GSE202771_TNBC_outputs"

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

url <- "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/GSE202771/suppl/GSE202771_TNBC_sc10x_raw_counts.RData.gz"

outer_path <- file.path(outdir, "GSE202771_TNBC_sc10x_raw_counts.RData.gz")
inner_path <- file.path(outdir, "GSE202771_TNBC_sc10x_raw_counts.inner.RData.gz")
rdata_unpacked_path <- file.path(outdir, "GSE202771_TNBC_sc10x_raw_counts.unpacked.RData")

# TNBC cell-line name patterns to try in matrix names or column names
cell_line_patterns <- c(
  "BT20|BT_20|BT-20",
  "BT549|BT_549|BT-549",
  "CAL120|CAL_120|CAL-120",
  "CAL51|CAL_51|CAL-51",
  "DU4475|DU_4475|DU-4475",
  "HCC38|HCC_38|HCC-38",
  "HCC70|HCC_70|HCC-70",
  "HCC1143|HCC_1143|HCC-1143",
  "HCC1187|HCC_1187|HCC-1187",
  "HCC1395|HCC_1395|HCC-1395",
  "HCC1599|HCC_1599|HCC-1599",
  "HCC1806|HCC_1806|HCC-1806",
  "HCC1937|HCC_1937|HCC-1937",
  "HCC2157|HCC_2157|HCC-2157",
  "HCC3153|HCC_3153|HCC-3153",
  "HDQP1|HDQ_P1|HDQ-P1",
  "HS578T|Hs578T|HS_578T|HS-578T",
  "MDAMB231|MDA_MB_231|MDA-MB-231|MDA\\.MB\\.231",
  "MDAMB436|MDA_MB_436|MDA-MB-436|MDA\\.MB\\.436",
  "MDAMB468|MDA_MB_468|MDA-MB-468|MDA\\.MB\\.468",
  "SUM149|SUM149PT|SUM_149|SUM-149",
  "SUM159|SUM159PT|SUM_159|SUM-159",
  "SUM185|SUM185PE|SUM_185|SUM-185"
)

# -------------------------
# Download if needed
# -------------------------
if (!file.exists(outer_path)) {
  message("Downloading GSE202771 raw-count RData.gz...")
  download.file(url, outer_path, mode = "wb")
} else {
  message("Using existing downloaded file: ", outer_path)
}

# -------------------------
# Handle double gzip
# -------------------------
message("Preparing double-gzipped RData...")

if (!file.exists(inner_path)) {
  message("Creating inner gzip file...")
  cmd1 <- paste("gzip -cd", shQuote(outer_path), ">", shQuote(inner_path))
  status1 <- system(cmd1)
  if (status1 != 0) stop("Failed to create inner gzip file.")
} else {
  message("Using existing inner gzip file: ", inner_path)
}

if (!file.exists(rdata_unpacked_path)) {
  message("Creating unpacked RData file...")
  cmd2 <- paste("gzip -cd", shQuote(inner_path), ">", shQuote(rdata_unpacked_path))
  status2 <- system(cmd2)
  if (status2 != 0) stop("Failed to create unpacked RData file.")
} else {
  message("Using existing unpacked RData file: ", rdata_unpacked_path)
}

message("File sizes:")
print(file.info(c(outer_path, inner_path, rdata_unpacked_path))[, c("size", "mtime")])

# -------------------------
# Load RData
# -------------------------
message("Loading unpacked RData...")
env <- new.env()
loaded_objects <- load(rdata_unpacked_path, envir = env)

message("Objects loaded:")
print(loaded_objects)

message("Object classes:")
print(sapply(ls(env), function(n) paste(class(get(n, envir = env)), collapse = ",")))

# -------------------------
# Utility functions
# -------------------------
safe_name <- function(x) {
  x <- gsub("[^A-Za-z0-9_]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  x
}

looks_gene_like <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x)]
  if (length(x) == 0) return(FALSE)

  headx <- head(x, min(1000, length(x)))

  ens <- mean(grepl("^ENSG", headx))
  sym <- mean(grepl("^[A-Za-z][A-Za-z0-9._-]+$", headx))
  ribo <- mean(grepl("^RPL|^RPS|^MT-", headx, ignore.case = FALSE))

  return(ens > 0.2 || sym > 0.5 || ribo > 0.02)
}

is_count_matrix <- function(x) {
  if (!(inherits(x, "Matrix") || inherits(x, "matrix"))) {
    return(FALSE)
  }

  d <- dim(x)
  if (length(d) != 2) return(FALSE)

  # Allow either genes x cells or cells x genes orientation
  if (max(d) < 1000 || min(d) < 100) return(FALSE)

  return(TRUE)
}

matrix_summary <- function(mat, nm) {
  d <- dim(mat)
  rn <- rownames(mat)
  cn <- colnames(mat)

  message("Candidate: ", nm)
  message("  Class: ", paste(class(mat), collapse = ","))
  message("  Dim: ", d[1], " x ", d[2])
  message("  Row names present: ", !is.null(rn))
  message("  Col names present: ", !is.null(cn))

  if (!is.null(rn)) {
    message("  First row names: ", paste(head(rn, 5), collapse = ", "))
    message("  Rows gene-like: ", looks_gene_like(rn))
  }

  if (!is.null(cn)) {
    message("  First col names: ", paste(head(cn, 5), collapse = ", "))
    message("  Cols gene-like: ", looks_gene_like(cn))
  }
}

# -------------------------
# Recursive object walker
# Finds matrices inside lists/S4/environments
# -------------------------
found <- list()
max_depth <- 8

walk_object <- function(x, name_prefix, depth = 0) {
  if (depth > max_depth) return(NULL)

  if (is_count_matrix(x)) {
    found[[name_prefix]] <<- x
    return(NULL)
  }

  # Environment
  if (is.environment(x)) {
    nms <- ls(x)
    for (nm in nms) {
      val <- tryCatch(get(nm, envir = x), error = function(e) NULL)
      if (!is.null(val)) {
        walk_object(val, paste0(name_prefix, "__", nm), depth + 1)
      }
    }
    return(NULL)
  }

  # List
  if (is.list(x)) {
    nms <- names(x)
    if (is.null(nms)) nms <- paste0("item", seq_along(x))

    for (i in seq_along(x)) {
      val <- tryCatch(x[[i]], error = function(e) NULL)
      if (!is.null(val)) {
        walk_object(val, paste0(name_prefix, "__", nms[i]), depth + 1)
      }
    }
    return(NULL)
  }

  # S4 object slots, e.g., Seurat-like objects
  if (isS4(x)) {
    slots <- tryCatch(slotNames(x), error = function(e) character(0))
    if (length(slots) > 0) {
      for (sn in slots) {
        val <- tryCatch(slot(x, sn), error = function(e) NULL)
        if (!is.null(val)) {
          walk_object(val, paste0(name_prefix, "__slot_", sn), depth + 1)
        }
      }
    }
    return(NULL)
  }

  return(NULL)
}

# -------------------------
# Find candidate matrices
# -------------------------
message("Searching loaded object for count matrices...")

for (obj_name in ls(env)) {
  obj <- get(obj_name, envir = env)
  walk_object(obj, obj_name, depth = 0)
}

if (length(found) == 0) {
  stop("No count-like matrix found inside the loaded RData.")
}

message("Candidate count matrices found:")
print(names(found))

for (nm in names(found)) {
  matrix_summary(found[[nm]], nm)
}

# -------------------------
# Prepare matrix orientation
# -------------------------
prepare_matrix <- function(mat, label) {
  mat <- as(mat, "dgCMatrix")

  rn <- rownames(mat)
  cn <- colnames(mat)

  if (is.null(rn) && is.null(cn)) {
    stop(paste("Matrix has no rownames or colnames:", label))
  }

  # If columns look like genes and rows do not, transpose
  if (!is.null(cn) && looks_gene_like(cn) && (is.null(rn) || !looks_gene_like(rn))) {
    message("Transposing matrix because columns look like genes: ", label)
    mat <- t(mat)
    rn <- rownames(mat)
    cn <- colnames(mat)
  }

  rn <- rownames(mat)
  cn <- colnames(mat)

  if (is.null(rn) || !looks_gene_like(rn)) {
    stop(paste("Rows do not look like gene symbols/IDs for:", label))
  }

  if (is.null(cn)) {
    colnames(mat) <- paste0("cell_", seq_len(ncol(mat)))
  }

  return(mat)
}

# -------------------------
# Write 10x zip
# -------------------------
write_10x_zip <- function(mat, label, n_cells = 10000, seed = 42, outdir = ".") {
  set.seed(seed)

  mat <- prepare_matrix(mat, label)

  n_take <- min(n_cells, ncol(mat))
  selected <- sample(seq_len(ncol(mat)), n_take)

  submat <- mat[, selected, drop = FALSE]

  genes <- rownames(submat)
  cells <- colnames(submat)

  if (length(genes) < 5000) {
    stop(paste("Too few genes after preparation:", length(genes), "for", label))
  }

  if (length(cells) < 100) {
    stop(paste("Too few cells after preparation:", length(cells), "for", label))
  }

  safe_label <- safe_name(label)
  sample_dir <- file.path(outdir, paste0(safe_label, "_", n_take, "cells_10x"))

  if (dir.exists(sample_dir)) {
    unlink(sample_dir, recursive = TRUE, force = TRUE)
  }

  dir.create(sample_dir, showWarnings = FALSE, recursive = TRUE)

  Matrix::writeMM(submat, file.path(sample_dir, "matrix.mtx"))

  write.table(
    cells,
    file.path(sample_dir, "barcodes.tsv"),
    quote = FALSE,
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE
  )

  features <- data.frame(
    gene_id = genes,
    gene_symbol = genes,
    feature_type = "Gene Expression"
  )

  write.table(
    features,
    file.path(sample_dir, "features.tsv"),
    quote = FALSE,
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE
  )

  write.table(
    genes,
    file.path(sample_dir, "genes.tsv"),
    quote = FALSE,
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE
  )

  oldwd <- getwd()
  setwd(sample_dir)

  system("gzip -f matrix.mtx")
  system("gzip -f barcodes.tsv")
  system("gzip -f features.tsv")
  system("gzip -f genes.tsv")

  zip_name <- file.path(
    normalizePath(outdir),
    paste0(safe_label, "_", n_take, "cells_10x_standard.zip")
  )

  if (file.exists(zip_name)) file.remove(zip_name)

  system(paste(
    "zip -q -j",
    shQuote(zip_name),
    "matrix.mtx.gz barcodes.tsv.gz features.tsv.gz genes.tsv.gz"
  ))

  setwd(oldwd)

  message("Saved zip: ", zip_name)
  message("  Label: ", label)
  message("  Cells: ", n_take)
  message("  Genes: ", nrow(submat))
  message("  Nonzero entries: ", length(submat@x))

  return(zip_name)
}

# -------------------------
# Write subset by cell-line pattern if possible
# -------------------------
write_cellline_subsets <- function(mat, base_label, patterns, n_cells = 10000, seed = 42, outdir = ".") {
  mat <- prepare_matrix(mat, base_label)
  cn <- colnames(mat)

  made <- character(0)

  for (pat in patterns) {
    idx <- grep(pat, cn, ignore.case = TRUE)

    if (length(idx) >= 500) {
      label <- paste0(base_label, "_", safe_name(pat))
      message("Found cell-line subset: ", label, " cells=", length(idx))

      submat <- mat[, idx, drop = FALSE]

      z <- write_10x_zip(
        mat = submat,
        label = label,
        n_cells = min(n_cells, length(idx)),
        seed = seed,
        outdir = outdir
      )

      made <- c(made, z)
    }
  }

  return(made)
}

# -------------------------
# Select matrices to process
# -------------------------
all_names <- names(found)

# Prefer matrix names suggesting raw/count RNA
priority_names <- all_names[
  grepl("count|counts|raw|RNA|assay|matrix", all_names, ignore.case = TRUE)
]

if (length(priority_names) == 0) {
  priority_names <- all_names
}

# Avoid producing too many files
selected_names <- head(priority_names, 10)

message("Selected matrices for export:")
print(selected_names)

created_zips <- character(0)

for (nm in selected_names) {
  message("====================================================")
  message("Processing matrix: ", nm)

  mat <- found[[nm]]

  # 1. Write an all-cell random TNBC sample
  z_all <- tryCatch({
    write_10x_zip(
      mat = mat,
      label = paste0(nm, "_ALL_TNBC_celllines"),
      n_cells = n_cells,
      seed = seed,
      outdir = outdir
    )
  }, error = function(e) {
    message("Failed all-cell export for ", nm, ": ", e$message)
    NULL
  })

  if (!is.null(z_all)) created_zips <- c(created_zips, z_all)

  # 2. Try to split by cell-line names in column names
  z_subsets <- tryCatch({
    write_cellline_subsets(
      mat = mat,
      base_label = nm,
      patterns = cell_line_patterns,
      n_cells = n_cells,
      seed = seed,
      outdir = outdir
    )
  }, error = function(e) {
    message("Cell-line subset export failed for ", nm, ": ", e$message)
    character(0)
  })

  if (length(z_subsets) > 0) {
    created_zips <- c(created_zips, z_subsets)
  }
}

message("====================================================")
message("Finished.")
message("Created zip files:")
print(created_zips)

message("Files in output folder:")
print(list.files(outdir, pattern = "zip$", full.names = TRUE))