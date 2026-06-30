# SNN STDP MNIST

This repository contains an Euler-based Diehl-Cook MNIST implementation together with an iSTDP variant in `Euler_stdp_MNIST_iSTDP.py`.

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

## Reproducible iSTDP Runs

The two larger comparison runs were planned with the following commands:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python Euler_stdp_MNIST_iSTDP.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --epochs 1 --train-examples 5000 --test-examples 1000 \
  --update-interval 500 --weight-stats-every 500 --plot-every 0 \
  --max-spike-retries-per-example 25 \
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
  --no-record-spikes \
  --out-dir ./runs/istdp_seed0 \
  --inhibition-mode istdp
```
