#!/bin/bash
#SBATCH --job-name=euler_stdp_smoke
#SBATCH --output=logs/euler_stdp_smoke_%j_out.out
#SBATCH --error=logs/euler_stdp_smoke_%j_err.out
#SBATCH --partition=HPC-CPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=8000

set -euo pipefail
set -x

date
hostname
pwd
id

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/euler_stdp_mnist_latest.sif}"
SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
BIND_PATH="${BIND_PATH:-$PROJECT_DIR}"

mkdir -p "$PROJECT_DIR/logs"

"$SINGULARITY_BIN" --version
ls -lh "$SIF_PATH"
ls -lh "$PROJECT_DIR"

"$SINGULARITY_BIN" exec --bind "$BIND_PATH" "$SIF_PATH" /bin/bash "$PROJECT_DIR/hpc/run_container_smoke_test.sh"

rc=$?
echo "singularity rc=$rc"
date
exit $rc
