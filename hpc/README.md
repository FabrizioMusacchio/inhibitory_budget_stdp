# HPC and Container Reproduction

This folder contains the CPU-only Docker/Singularity environment and SLURM wrappers used for the large replicate sweeps reported in the preprint.

The simulation code is CPU-based. GPUs are not required.

## Container
Build the Docker image from the repository root:

```bash
docker build -f hpc/Dockerfile -t euler_stdp_mnist .
```

On clusters that use Singularity/Apptainer, convert the image to a `.sif` container using the method supported by your site, for example:

```bash
singularity build euler_stdp_mnist_latest.sif docker-daemon://euler_stdp_mnist:latest
```

The container installs:

- Python 3.12
- NumPy
- Matplotlib
- pandas
- Numba
- scikit-learn

TensorFlow is intentionally omitted. The scripts expect an explicit Keras-style `mnist.npz` file.

## Cluster project directory
Create a cluster project directory containing at least:

```text
Euler_stdp_MNIST_iSTDP.py
run_baseline_regime_sweep.py
run_inhibitory_plasticity_maps.py
hpc/hpc_container_smoke_test.py
hpc/run_container_smoke_test.sh
hpc/run_euler_stdp_hpc.sh
hpc/run_hpc_manifest_job.py
hpc/collect_hpc_manifest_summary.py
hpc/generate_hpc_run_manifest.py
hpc/generate_slow_budget_map_manifest.py
hpc/*.sbatch
data/mnist.npz
```

The SLURM files use these environment variables:

```bash
export PROJECT_DIR=/path/to/cluster/project
export SIF_PATH=/path/to/euler_stdp_mnist_latest.sif
export SINGULARITY_BIN=singularity
export BIND_PATH=/path/to/bind/root
```

If unset, `PROJECT_DIR` defaults to the SLURM submit directory, `SIF_PATH` defaults to `$PROJECT_DIR/euler_stdp_mnist_latest.sif`, `SINGULARITY_BIN` defaults to `singularity`, and `BIND_PATH` defaults to `$PROJECT_DIR`.

Create the log directory before submitting jobs:

```bash
mkdir -p "$PROJECT_DIR/logs"
```

## Smoke test
Submit a short container and file-layout check:

```bash
sbatch hpc/euler_stdp_smoke_test.sbatch
```

The smoke test imports the required Python packages, checks `data/mnist.npz`, checks required project scripts, writes a small JSON file under `logs/`, and runs `Euler_stdp_MNIST_iSTDP.py --help`.


## Manifest-based sweeps
Generate the main 30k manifest:

```bash
python hpc/generate_hpc_run_manifest.py --out hpc_30k_manifest.csv
```

Submit the array:

```bash
sbatch --array=0-229%40 hpc/euler_stdp_manifest_array.sbatch
```

This manifest contains the fixed-inhibition operating-regime grid, matched and mismatched direct rule comparisons, and the Vogels-derived iSTDP stability map.

After completion, collect summaries:

```bash
sbatch hpc/euler_stdp_manifest_collect.sbatch
```

Generate the slow-homeostatic and budget-constrained maps:

```bash
python hpc/generate_slow_budget_map_manifest.py --out hpc_slow_budget_maps_30k_manifest.csv
sbatch --array=0-199%40 hpc/euler_stdp_slow_budget_maps_array.sbatch
```

All array workers are resumable: if a run directory already contains `logs/run_summary.json`, `hpc/run_hpc_manifest_job.py` skips it unless `--force` is used.

## Legacy convenience array
`hpc/euler_stdp_30k_array.sbatch` and `hpc/run_euler_stdp_hpc.sh` provide an older four-task convenience wrapper for fixed-grid, matched comparison, mismatched comparison, and Vogels-map runs. The manifest-based workflow above is preferred for the final preprint reproduction because it parallelizes individual runs.
