#!/bin/bash
#SBATCH --job-name=euler_stdp_collect
#SBATCH --output=logs/euler_stdp_collect_%j_out.out
#SBATCH --error=logs/euler_stdp_collect_%j_err.out
#SBATCH --partition=HPC-CPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4000

set -euo pipefail
set -x

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/euler_stdp_mnist_latest.sif}"
SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
BIND_PATH="${BIND_PATH:-$PROJECT_DIR}"
MANIFEST_PATH=$PROJECT_DIR/hpc_30k_manifest.csv
OUT_BASE_DIR=$PROJECT_DIR/runs/long_30k_sweeps_manifest

mkdir -p "$PROJECT_DIR/logs"

"$SINGULARITY_BIN" exec --bind "$BIND_PATH" "$SIF_PATH" /opt/conda/envs/diehl_cook_euler/bin/python "$PROJECT_DIR/hpc/collect_hpc_manifest_summary.py" --manifest "$MANIFEST_PATH" --out-base-dir "$OUT_BASE_DIR"
