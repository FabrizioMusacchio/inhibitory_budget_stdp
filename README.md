# Budget-constrained inhibitory plasticity in competitive STDP networks

![GitHub Release](https://img.shields.io/github/v/release/FabrizioMusacchio/inhibitory_budget_stdp) [![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-green.svg)](https://github.com/FabrizioMusacchio/inhibitory_budget_stdp?tab=GPL-3.0-1-ov-file)  [![GitHub last commit](https://img.shields.io/github/last-commit/FabrizioMusacchio/inhibitory_budget_stdp)](https://github.com/FabrizioMusacchio/inhibitory_budget_stdp/commits/main/)   [![GitHub Issues Open](https://img.shields.io/github/issues/FabrizioMusacchio/inhibitory_budget_stdp)](https://github.com/FabrizioMusacchio/inhibitory_budget_stdp/issues) [![GitHub Issues Closed](https://img.shields.io/github/issues-closed/FabrizioMusacchio/inhibitory_budget_stdp?color=53c92e)](https://github.com/FabrizioMusacchio/inhibitory_budget_stdp/issues?q=is%3Aissue%20state%3Aclosed) [![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-pr/FabrizioMusacchio/inhibitory_budget_stdp)](https://github.com/FabrizioMusacchio/inhibitory_budget_stdp/pulls)   ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/fabriziomusacchio/inhibitory_budget_stdp)   [![Example Datasets on Zenodo](https://img.shields.io/badge/Example%20Datasets-10.5281%2Fzenodo.22976539-blue)](https://doi.org/10.5281/zenodo.22976539)   [![Zenodo Archive](https://img.shields.io/badge/Zenodo%20Archive-10.5281%2Fzenodo.22976645-blue)](https://doi.org/10.5281/zenodo.22976645)  [![preprint on bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.09.27.754763-red)](https://doi.org/10.64898/2026.09.27.754763) 


This repository contains an Euler-integrated competitive spiking neural network (SNN) for MNIST, based on the [Diehl-Cook architecture](https://doi.org/10.3389/fncom.2015.00099) and extended with several inhibitory plasticity rules.

In this study, we asked when fixed lateral inhibition can be replaced by adaptive inhibitory synapses without destroying the competitive operating regime that supports unsupervised representation learning. The main simulation script supports fixed inhibition, a Vogels-style inhibitory STDP rule, a slow homeostatic inhibitory rule, and a budget-constrained slow homeostatic rule that conserves the total inhibitory input to each excitatory neuron.

Large run outputs and manuscript files are not stored in this GitHub repository. The datasets needed to reproduce the preprint analyses are stored in a separate Zenodo archive (see below; also see [runs/README.md](runs/README.md)).

Repository Layout:

* `Euler_stdp_MNIST_iSTDP.py`: Main simulator and plotting diagnostics
* `run_baseline_regime_sweep.py`: Fixed-inhibition and matched-rule sweep runner
* `run_inhibitory_plasticity_maps.py`: Vogels-style and inhibitory-plasticity map runner
* `additional_scripts/`: Analysis and preprint figure-generation scripts
* `hpc/`: Docker/Singularity and SLURM helper scripts
* `runs/README.md`: Description of the external Zenodo data package
```

## Setup
Create a Python environment with the packages used for local runs and plotting:

```python
conda create -n diehl_cook_euler python=3.12 mamba -y
conda activate diehl_cook_euler
mamba install numpy matplotlib pandas numba scikit-learn tensorflow ipykernel -y
```

TensorFlow is used only as a convenient local MNIST loader. On HPC systems we usually provide `mnist.npz` directly and use the lighter container environment described in [hpc/README.md](hpc/README.md).

If Matplotlib reports a non-writable configuration directory (for instance on HPC systems), set `MPLCONFIGDIR` to a writable location:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python Euler_stdp_MNIST_iSTDP.py --help
```

## Minimal test run
The command below performs a short fixed-inhibition run and writes its outputs to `runs_smoke/`:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python Euler_stdp_MNIST_iSTDP.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --out-dir runs_smoke/fixed \
  --epochs 1 \
  --train-examples 100 \
  --test-examples 50 \
  --inhibition-mode fixed \
  --plot-every 0 \
  --weight-stats-every 0 \
  --metrics-window 50 \
  --seed 0
```

## Main model options
The simulator exposes the inhibitory mechanisms used in our study through:

```bash
--inhibition-mode fixed
--inhibition-mode istdp --istdp-rule vogels
--inhibition-mode istdp --istdp-rule slow_homeostat
--inhibition-mode istdp --istdp-rule normalized_slow_homeostat
```

Useful sweep and safety options include:

```bash
--input-intensity 2.0
--w-aiae 10.0
--eta-ie 0.0003
--rho-ie 0.03
--slow-homeostat-target-rate-hz 0.03
--aiae-target-sum 3990
--aiae-normalize-every 1
--metrics-window 500
--max-spike-retries-per-example 25
--abort-runaway-spike-threshold 5000
--abort-runaway-consecutive-examples 20
--abort-theta-mean-threshold 200
--abort-aiae-max-fraction-threshold 0.2
```

When an early-abort criterion is triggered, the run writes `logs/run_summary.json` and `ABORTED_RUN_NOTES.md` inside the run directory.

## Reproducing main sweeps
The primary fixed-inhibition grid and matched-rule comparisons are generated with:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python run_baseline_regime_sweep.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --run-set fixed_grid \
  --out-base-dir runs/baseline_regime_sweep \
  --run-id fixed_grid_30k_1ep \
  --epochs 1 \
  --train-examples 30000 \
  --test-examples 5000 \
  --plot-every 0 \
  --weight-stats-every 5000 \
  --metrics-window 500 \
  --seeds 0,1,2,3,4 \
  --resume
```

Matched and mismatched budget comparisons use the same runner with `--run-set compare_regime` and the appropriate `--aiae-target-sum`.

The Vogels-style parameter map is generated with:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python run_inhibitory_plasticity_maps.py \
  --mnist-npz-path "$HOME/.keras/datasets/mnist.npz" \
  --run-set vogels_stability_map \
  --out-base-dir runs/inhibitory_plasticity_maps \
  --run-id vogels_stability_map_30k_1ep \
  --epochs 1 \
  --train-examples 30000 \
  --test-examples 5000 \
  --plot-every 0 \
  --weight-stats-every 5000 \
  --metrics-window 500 \
  --seeds 0,1,2,3,4 \
  --resume
```

For full 30k-example sweeps, the manifest-based SLURM workflow in [hpc/README.md](hpc/README.md) is preferred over running all jobs serially.

## Recreating preprint figures
After downloading the Zenodo data package, place the following folders under `runs/`:

> Musacchio, F. (2026). *Simulation data for inhibitory budget matching in competitive STDP networks* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.22976539

```text
runs/long_30k_sweeps_manifest_HPC/
runs/slow_budget_maps_30k_manifest_HPC/
runs/long_visualization_60k_2ep/
```

Then regenerate the analysis panels with:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python additional_scripts/preprint_generate_figures.py
```

The figure script writes panels to `papers/preprint/figures/` in your local working copy.

## Citation
If you use this code, please cite the following preprint:

> Musacchio F., and Fuhrmann M., *Inhibitory budget matching constrains homeostatic plasticity in competitive spiking networks*. bioRxiv 2026.09.27.754763; doi: https://doi.org/10.64898/2026.09.27.754763

and additionally the archived software version of the analysis scripts:

> Musacchio, F., & Fuhrmann, M. (2026). *Inhibitory Budget STDP: code for budget-constrained inhibitory plasticity in competitive STDP networks* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22976645

The simulation data required to reproduce the preprint figures are archived on Zenodo:

> Musacchio, F. (2026). *Simulation data for inhibitory budget matching in competitive STDP networks* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.22976539
