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

Useful rule-specific parameters:

```bash
--theta-gate-ref-mv 20.0
--theta-gate-scale-mv 10.0
--slow-homeostat-target-rate-hz 0.1
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
