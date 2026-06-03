#!/bin/bash -l
#SBATCH --account=project_2010376
#SBATCH --partition=small
#SBATCH --time=4:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=prepare-gse202771-tnbc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

module --force purge
module load r-env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORKDIR="/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/src_031025"
cd "$WORKDIR"

echo "===================================================="
echo "Preparing GSE202771 TNBC cell-line scRNA-seq"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Workdir: $WORKDIR"
echo "===================================================="

echo "Checking R/container environment"
echo "SING_IMAGE=${SING_IMAGE:-NOT_SET}"
echo "SING_FLAGS=${SING_FLAGS:-NOT_SET}"
which Rscript || true
which apptainer_wrapper || true
which apptainer || true
which singularity || true

# Set TMPDIR
if test -f ~/.Renviron; then
    sed -i '/TMPDIR/d' ~/.Renviron
fi
echo "TMPDIR=/scratch/project_2010376" >> ~/.Renviron

# Preferred route if wrapper exists
if command -v apptainer_wrapper >/dev/null 2>&1; then
    echo "Using apptainer_wrapper"
    srun apptainer_wrapper exec Rscript --no-save --no-restore prepare_GSE202771_TNBC_celllines.R

# Fallback route: use apptainer directly
elif command -v apptainer >/dev/null 2>&1 && [ -n "${SING_IMAGE:-}" ]; then
    echo "Using apptainer directly"
    srun apptainer exec \
        ${SING_FLAGS:-} \
        --bind /scratch:/scratch,/projappl:/projappl,/users:/users \
        "$SING_IMAGE" \
        Rscript --no-save --no-restore prepare_GSE202771_TNBC_celllines.R

# Fallback route: singularity direct
elif command -v singularity >/dev/null 2>&1 && [ -n "${SING_IMAGE:-}" ]; then
    echo "Using singularity directly"
    srun singularity exec \
        ${SING_FLAGS:-} \
        --bind /scratch:/scratch,/projappl:/projappl,/users:/users \
        "$SING_IMAGE" \
        Rscript --no-save --no-restore prepare_GSE202771_TNBC_celllines.R

else
    echo "ERROR: Could not find apptainer_wrapper, apptainer, or singularity."
    echo "Also SING_IMAGE may not be set."
    exit 1
fi

echo "===================================================="
echo "Done: $(date)"
echo "Output files:"
find GSE202771_TNBC_outputs -type f | sort
echo "===================================================="