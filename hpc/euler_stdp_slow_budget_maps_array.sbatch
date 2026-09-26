#!/bin/bash
#SBATCH --job-name=euler_stdp_sb_maps
#SBATCH --output=logs/euler_stdp_sb_maps_%A_%a_out.out
#SBATCH --error=logs/euler_stdp_sb_maps_%A_%a_err.out
#SBATCH --partition=HPC-CPUs
#SBATCH --array=0-199%40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=16000

set -euo pipefail
set -x

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/euler_stdp_mnist_latest.sif}"
SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
BIND_PATH="${BIND_PATH:-$PROJECT_DIR}"
MANIFEST_PATH=$PROJECT_DIR/hpc_slow_budget_maps_30k_manifest.csv
OUT_BASE_DIR=$PROJECT_DIR/runs/slow_budget_maps_30k_manifest

mkdir -p "$PROJECT_DIR/logs"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

date
hostname
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
"$SINGULARITY_BIN" --version
ls -lh "$SIF_PATH"
ls -lh "$MANIFEST_PATH"

"$SINGULARITY_BIN" exec --bind "$BIND_PATH" "$SIF_PATH" /opt/conda/envs/diehl_cook_euler/bin/python "$PROJECT_DIR/hpc/run_hpc_manifest_job.py" --manifest "$MANIFEST_PATH" --index "$SLURM_ARRAY_TASK_ID" --project-dir "$PROJECT_DIR" --mnist-npz-path "$PROJECT_DIR/data/mnist.npz" --out-base-dir "$OUT_BASE_DIR"

rc=$?
echo "singularity rc=$rc"
date
exit $rc
