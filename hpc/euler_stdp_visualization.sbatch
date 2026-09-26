#!/bin/bash
#SBATCH --job-name=euler_stdp_viz
#SBATCH --output=logs/euler_stdp_viz_%j_out.out
#SBATCH --error=logs/euler_stdp_viz_%j_err.out
#SBATCH --partition=HPC-CPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=64000

set -euo pipefail
set -x

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/euler_stdp_mnist_latest.sif}"
SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
BIND_PATH="${BIND_PATH:-$PROJECT_DIR}"

mkdir -p "$PROJECT_DIR/logs"

date
hostname
"$SINGULARITY_BIN" exec --bind "$BIND_PATH" "$SIF_PATH" /bin/bash "$PROJECT_DIR/hpc/run_euler_stdp_hpc.sh" visualization
rc=$?
date
exit $rc
