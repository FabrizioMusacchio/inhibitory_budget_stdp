#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MNIST_NPZ_PATH="${MNIST_NPZ_PATH:-$PROJECT_DIR/data/mnist.npz}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/diehl_cook_euler/bin/python}"

cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/.mplconfig" "$PROJECT_DIR/.numba_cache"

export PROJECT_DIR
export MNIST_NPZ_PATH
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PROJECT_DIR/.mplconfig"
export NUMBA_CACHE_DIR="$PROJECT_DIR/.numba_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

echo "Running Euler STDP container smoke test"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "MNIST_NPZ_PATH=$MNIST_NPZ_PATH"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" "$PROJECT_DIR/hpc/hpc_container_smoke_test.py"
