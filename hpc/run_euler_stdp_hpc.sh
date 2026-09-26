#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
RUN_SET="${1:-${RUN_SET:-all}}"
MNIST_NPZ_PATH="${MNIST_NPZ_PATH:-$PROJECT_DIR/data/mnist.npz}"
OUT_BASE_DIR="${OUT_BASE_DIR:-$PROJECT_DIR/runs/long_30k_sweeps}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/diehl_cook_euler/bin/python}"

cd "$PROJECT_DIR"

mkdir -p "$OUT_BASE_DIR" "$PROJECT_DIR/logs" "$PROJECT_DIR/.mplconfig" "$PROJECT_DIR/.numba_cache"

export MPLBACKEND=Agg
export MPLCONFIGDIR="$PROJECT_DIR/.mplconfig"
export NUMBA_CACHE_DIR="$PROJECT_DIR/.numba_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"

if [[ ! -f "$MNIST_NPZ_PATH" ]]; then
  echo "ERROR: MNIST npz not found at $MNIST_NPZ_PATH" >&2
  echo "Copy mnist.npz to $PROJECT_DIR/data/mnist.npz or set MNIST_NPZ_PATH." >&2
  exit 2
fi

echo "PROJECT_DIR=$PROJECT_DIR"
echo "RUN_SET=$RUN_SET"
echo "MNIST_NPZ_PATH=$MNIST_NPZ_PATH"
echo "OUT_BASE_DIR=$OUT_BASE_DIR"
"$PYTHON_BIN" --version

run_fixed_grid() {
  "$PYTHON_BIN" run_baseline_regime_sweep.py --mnist-npz-path "$MNIST_NPZ_PATH" --run-set fixed_grid --out-base-dir "$OUT_BASE_DIR/baseline_regime_sweep" --run-id fixed_grid_30k_1ep --epochs 1 --train-examples 30000 --test-examples 5000 --plot-every 0 --weight-stats-every 5000 --metrics-window 500 --seeds 0,1,2,3,4 --resume
}

run_compare_mismatched() {
  "$PYTHON_BIN" run_baseline_regime_sweep.py --mnist-npz-path "$MNIST_NPZ_PATH" --run-set compare_regime --out-base-dir "$OUT_BASE_DIR/baseline_regime_sweep" --run-id compare_mismatched_budget_30k_1ep --epochs 1 --train-examples 30000 --test-examples 5000 --plot-every 0 --weight-stats-every 5000 --metrics-window 500 --compare-input-intensity 2.0 --compare-w-aiae 10 --compare-w-aeai 10.4 --compare-theta-plus-mV 0.05 --aiae-target-sum 7980 --seeds 0,1,2,3,4 --resume
}

run_compare_matched() {
  "$PYTHON_BIN" run_baseline_regime_sweep.py --mnist-npz-path "$MNIST_NPZ_PATH" --run-set compare_regime --out-base-dir "$OUT_BASE_DIR/baseline_regime_sweep" --run-id compare_matched_budget_30k_1ep --epochs 1 --train-examples 30000 --test-examples 5000 --plot-every 0 --weight-stats-every 5000 --metrics-window 500 --compare-input-intensity 2.0 --compare-w-aiae 10 --compare-w-aeai 10.4 --compare-theta-plus-mV 0.05 --aiae-target-sum 3990 --seeds 0,1,2,3,4 --resume
}

run_vogels_map() {
  "$PYTHON_BIN" run_inhibitory_plasticity_maps.py --mnist-npz-path "$MNIST_NPZ_PATH" --run-set vogels_stability_map --out-base-dir "$OUT_BASE_DIR/inhibitory_plasticity_maps" --run-id vogels_stability_map_30k_1ep --epochs 1 --train-examples 30000 --test-examples 5000 --plot-every 0 --weight-stats-every 5000 --metrics-window 500 --seeds 0,1,2,3,4 --resume
}

run_visualization() {
  "$PYTHON_BIN" Euler_stdp_MNIST_iSTDP.py --mnist-npz-path "$MNIST_NPZ_PATH" --out-dir "$OUT_BASE_DIR/visualization_30k_1ep/fixed_seed0" --epochs 1 --train-examples 30000 --test-examples 5000 --input-intensity 2.0 --theta-plus-mv 0.05 --w-aiae 10 --inhibition-mode fixed --plot-every 5000 --weight-stats-every 5000 --metrics-window 500 --seed 0
  "$PYTHON_BIN" Euler_stdp_MNIST_iSTDP.py --mnist-npz-path "$MNIST_NPZ_PATH" --out-dir "$OUT_BASE_DIR/visualization_30k_1ep/vogels_lowrho_seed0" --epochs 1 --train-examples 30000 --test-examples 5000 --input-intensity 2.0 --theta-plus-mv 0.05 --w-aiae 10 --inhibition-mode istdp --istdp-rule vogels --eta-ie 0.0003 --rho-ie 0.01 --w-ie-max 25 --no-istdp-during-rest --rollback-state-on-retry --plot-every 5000 --weight-stats-every 5000 --metrics-window 500 --seed 0
  "$PYTHON_BIN" Euler_stdp_MNIST_iSTDP.py --mnist-npz-path "$MNIST_NPZ_PATH" --out-dir "$OUT_BASE_DIR/visualization_30k_1ep/vogels_highrho_seed0" --epochs 1 --train-examples 30000 --test-examples 5000 --input-intensity 2.0 --theta-plus-mv 0.05 --w-aiae 10 --inhibition-mode istdp --istdp-rule vogels --eta-ie 0.0003 --rho-ie 0.03 --w-ie-max 25 --no-istdp-during-rest --rollback-state-on-retry --plot-every 5000 --weight-stats-every 5000 --metrics-window 500 --seed 0
  "$PYTHON_BIN" Euler_stdp_MNIST_iSTDP.py --mnist-npz-path "$MNIST_NPZ_PATH" --out-dir "$OUT_BASE_DIR/visualization_30k_1ep/slow_homeostat_seed0" --epochs 1 --train-examples 30000 --test-examples 5000 --input-intensity 2.0 --theta-plus-mv 0.05 --w-aiae 10 --inhibition-mode istdp --istdp-rule slow_homeostat --eta-ie 0.0003 --rho-ie 0.03 --w-ie-max 25 --no-istdp-during-rest --rollback-state-on-retry --plot-every 5000 --weight-stats-every 5000 --metrics-window 500 --seed 0
  "$PYTHON_BIN" Euler_stdp_MNIST_iSTDP.py --mnist-npz-path "$MNIST_NPZ_PATH" --out-dir "$OUT_BASE_DIR/visualization_30k_1ep/normalized_slow_homeostat_seed0" --epochs 1 --train-examples 30000 --test-examples 5000 --input-intensity 2.0 --theta-plus-mv 0.05 --w-aiae 10 --inhibition-mode istdp --istdp-rule normalized_slow_homeostat --eta-ie 0.0003 --rho-ie 0.03 --w-ie-max 25 --normalize-aiae-columns --aiae-target-sum 3990 --aiae-normalize-every 1 --no-istdp-during-rest --rollback-state-on-retry --plot-every 5000 --weight-stats-every 5000 --metrics-window 500 --seed 0
}

case "$RUN_SET" in
  fixed_grid) run_fixed_grid ;;
  compare_mismatched) run_compare_mismatched ;;
  compare_matched) run_compare_matched ;;
  vogels_map) run_vogels_map ;;
  visualization) run_visualization ;;
  all)
    run_fixed_grid
    run_compare_mismatched
    run_compare_matched
    run_vogels_map
    ;;
  *)
    echo "Unknown RUN_SET '$RUN_SET'. Use fixed_grid, compare_mismatched, compare_matched, vogels_map, visualization, or all." >&2
    exit 2
    ;;
esac

echo "Completed RUN_SET=$RUN_SET"
