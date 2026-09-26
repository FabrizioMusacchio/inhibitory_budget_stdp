# SNN STDP MNIST

This repository contains an Euler-based Diehl-Cook MNIST implementation together with switchable inhibitory plasticity variants in `Euler_stdp_MNIST_iSTDP.py`.

## Setup

The currently used Conda environment was created with:

```bash
conda create -n diehl_cook_euler python=3.12 mamba -y
conda activate diehl_cook_euler
mamba install numpy matplotlib numba scikit-learn tensorflow ipykernel -y
```

If `matplotlib` warns about a non-writable config directory, run experiments with a writable `MPLCONFIGDIR`, for example:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python Euler_stdp_MNIST_iSTDP.py --help
```

The iSTDP variant also includes a safeguard against endless resampling/retry loops:

```bash
--max-spike-retries-per-example 25
```

## Experimental Safety Controls

The current inhibitory plasticity runner supports additional safeguards for debugging and parameter sweeps:

```bash
--no-istdp-during-rest
--rollback-state-on-retry
--abort-runaway-spike-threshold 5000
--abort-runaway-consecutive-examples 20
--abort-theta-mean-threshold 200
--abort-aiae-max-fraction-threshold 0.2
```

When an early-abort criterion is triggered, the run writes:

- `logs/run_summary.json`
- `ABORTED_RUN_NOTES.md`

## Inhibitory Plasticity Rules

`Euler_stdp_MNIST_iSTDP.py` now supports the following `Ai->Ae` plasticity rules:

- `--istdp-rule vogels`
- `--istdp-rule centered`
- `--istdp-rule slow_homeostat`
- `--istdp-rule theta_gated`
- `--istdp-rule normalized_slow_homeostat`

Useful rule-specific parameters:

```bash
--theta-gate-ref-mv 20.0
--theta-gate-scale-mv 10.0
--slow-homeostat-target-rate-hz 0.1
--normalize-aiae-columns
--aiae-target-sum 7980
--aiae-normalize-every 1
```

## Reproducible iSTDP Runs

The two larger comparison runs were planned with the following commands:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python Euler_stdp_MNIST_iSTDP.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --epochs 1 --train-examples 5000 --test-examples 1000 \
  --update-interval 500 --weight-stats-every 500 --plot-every 0 \
  --max-spike-retries-per-example 25 \
  --no-istdp-during-rest \
  --rollback-state-on-retry \
  --no-record-spikes \
  --out-dir ./runs/fixed_seed0 \
  --inhibition-mode fixed
```

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python Euler_stdp_MNIST_iSTDP.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --epochs 1 --train-examples 5000 --test-examples 1000 \
  --update-interval 500 --weight-stats-every 500 --plot-every 0 \
  --max-spike-retries-per-example 25 \
  --no-istdp-during-rest \
  --rollback-state-on-retry \
  --no-record-spikes \
  --out-dir ./runs/istdp_seed0 \
  --inhibition-mode istdp
```

## Vogels iSTDP Sweep

To run a small parameter sweep over `eta_ie`, `rho_ie`, and `w_ie_max` with automatic CSV aggregation:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python sweep_vogels_istdp.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --train-examples 3000 \
  --test-examples 500 \
  --metrics-window 250 \
  --plot-every 0
```

The sweep writes one subdirectory per run plus a combined summary CSV in the generated sweep folder.

## Rule Variant Sweep

To compare multiple `Ai->Ae` rules at matched settings:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python sweep_istdp_rule_variants.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --eta-values 3e-4 \
  --rho-values 0.01,0.03 \
  --wmax-values 25 \
  --train-examples 3000 \
  --test-examples 500 \
  --metrics-window 250 \
  --plot-every 0
```

This writes one run directory per rule/parameter combination plus a combined `sweep_summary.csv`.

## Bernstein Follow-up Runs

The accepted Bernstein abstract is supported by the proof-of-concept runs in
`runs/focused_normalized_slow_homeostat_v2_20260630`. Do not overwrite those
folders. For follow-up evidence, use the additive runner below; it writes only
to a new timestamped folder under `runs/bernstein_followup/`.

Run the multi-seed poster conditions:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python run_bernstein_followup.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --run-set poster_multiseed \
  --seeds 0,1,2,3,4 \
  --train-examples 3000 \
  --test-examples 500 \
  --metrics-window 250 \
  --plot-every 0
```

This runs five conditions for each seed:

- `fixed`
- `vogels_stable`: `eta_ie=3e-4`, `rho_ie=0.01`, `w_ie_max=25`
- `vogels_unstable`: `eta_ie=3e-4`, `rho_ie=0.03`, `w_ie_max=25`
- `slow_homeostat_unconstrained`: `eta_ie=3e-4`, `rho_ie=0.03`, `w_ie_max=25`
- `normalized_slow_homeostat`: `eta_ie=3e-4`, `rho_ie=0.03`, `w_ie_max=25`, column-normalized `AiAe`

Run the focused Vogels stability map:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python run_bernstein_followup.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --run-set vogels_stability_map \
  --seeds 0,1,2,3,4 \
  --map-eta-values 1e-5,3e-5,1e-4,3e-4,1e-3 \
  --map-rho-values 0.01,0.03,0.1,0.3 \
  --map-wmax-values 25 \
  --train-examples 3000 \
  --test-examples 500 \
  --metrics-window 250 \
  --plot-every 0
```

Both commands write:

- `combined_summary.csv`
- `<run_set>/summary.csv`
- one timestamped subdirectory per individual run

If a long run is interrupted, restart it by reusing the printed run id:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python run_bernstein_followup.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --run-set poster_multiseed \
  --seeds 0,1,2,3,4 \
  --run-id YYYYMMDD_HHMMSS \
  --resume
```

## Baseline Regime Diagnosis

Before interpreting seed-dependent iSTDP failures, first find a fixed-inhibition
regime that is stable across seeds. This additive runner writes only to
`runs/baseline_regime_sweep/`.

Run the fixed-inhibition stability grid:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python run_baseline_regime_sweep.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --run-set fixed_grid \
  --seeds 0,1,2,3,4 \
  --input-intensities 0.5,1.0,1.5,2.0 \
  --w-aiae-values 10,15,20,25 \
  --w-aeai-values 10.4 \
  --theta-plus-values 0.05 \
  --train-examples 3000 \
  --test-examples 500 \
  --metrics-window 250 \
  --plot-every 0
```

After a stable fixed regime is identified, rerun the rule comparison at that
regime by replacing the three `--compare-*` values:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python run_baseline_regime_sweep.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --run-set compare_regime \
  --seeds 0,1,2,3,4 \
  --compare-input-intensity 1.0 \
  --compare-w-aiae 20.0 \
  --compare-w-aeai 10.4 \
  --compare-theta-plus-mV 0.05 \
  --train-examples 3000 \
  --test-examples 500 \
  --metrics-window 250 \
  --plot-every 0
```

The script also supports `--run-id YYYYMMDD_HHMMSS --resume` for interrupted
runs.
