#!/bin/bash
#SBATCH --job-name=euler_stdp_30k
#SBATCH --output=logs/euler_stdp_%A_%a_out.out
#SBATCH --error=logs/euler_stdp_%A_%a_err.out
#SBATCH --partition=HPC-CPUs
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem=64000

set -euo pipefail
set -x

date
hostname
pwd
id

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/euler_stdp_mnist_latest.sif}"
SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
BIND_PATH="${BIND_PATH:-$PROJECT_DIR}"

mkdir -p "$PROJECT_DIR/logs"

"$SINGULARITY_BIN" --version
ls -lh "$SIF_PATH"
ls -lh "$PROJECT_DIR"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) RUN_SET=fixed_grid ;;
  1) RUN_SET=compare_mismatched ;;
  2) RUN_SET=compare_matched ;;
  3) RUN_SET=vogels_map ;;
  *) echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

echo "RUN_SET=$RUN_SET"

"$SINGULARITY_BIN" exec --bind "$BIND_PATH" "$SIF_PATH" /bin/bash "$PROJECT_DIR/hpc/run_euler_stdp_hpc.sh" "$RUN_SET"

rc=$?
echo "singularity rc=$rc"
date
exit $rc
