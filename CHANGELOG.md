## Budget-constrained inhibitory plasticity in competitive STDP networks – Changelog

See here for a detailed list of changes made in each release of the
*Budget-constrained inhibitory plasticity in competitive STDP networks* project. Please also refer to the repository
[Releases page](https://github.com/FabrizioMusacchio/inhibitory_budget_stdp/releases).

Each release can be archived on Zenodo for long-term preservation and citation purposes.

<!-- 
---

## 🔜 Inhibitory_Budget_STDP v0.0.3 UPCOMING RELEASE

Unreleased
 -->

---

## 🔜 Inhibitory_Budget_STDP v0.0.2 UPCOMING RELEASE

Unreleased

### 📚 Documentation

- Added the published Zenodo data archive for the simulation outputs required to reproduce the preprint analyses: https://doi.org/10.5281/zenodo.22976539.
- Added the published Zenodo software archive for the v0.0.1 code release: https://doi.org/10.5281/zenodo.22976646.

---

## 🚀 Inhibitory_Budget_STDP v0.0.1

September 26, 2026

This is the first public release of the *Inhibitory Budget STDP* project. The release provides the simulation code, sweep runners, HPC/container workflow, and analysis helpers used to study when fixed lateral inhibition in a competitive STDP network can be replaced by adaptive inhibitory plasticity. The codebase accompanies the forthcoming preprint *Inhibitory budget matching constrains homeostatic plasticity in competitive spiking networks*.

### ✨ Features

- Added the main Euler-integrated competitive spiking network simulator for MNIST in `Euler_stdp_MNIST_iSTDP.py`.
- Implemented the fixed-inhibition reference circuit used to define stable competitive operating regimes.
- Added Vogels-style inhibitory STDP with configurable inhibitory learning rate, target offset, inhibitory bounds, and retry/abort safeguards.
- Added slow homeostatic inhibitory plasticity with stimulus-level updates driven by population-rate deviations.
- Added budget-constrained slow homeostatic inhibition with column-wise inhibitory budget conservation.
- Added training and test diagnostics including receptive fields, confusion matrices, spike-count readout, firing-rate summaries, inhibitory-weight summaries, neuron-usage entropy, dead-neuron statistics, adaptive-threshold summaries, and early-abort notes.
- Added robust early-abort criteria for runaway activity, repeated retry-cap failures, excessive adaptive thresholds, and excessive inhibitory-weight concentration.
- Added `run_baseline_regime_sweep.py` for fixed-inhibition operating-regime sweeps and direct matched/mismatched inhibitory-plasticity comparisons.
- Added `run_inhibitory_plasticity_maps.py` for Vogels-style inhibitory STDP and inhibitory-plasticity stability-map experiments.
- Added `additional_scripts/preprint_generate_figures.py` for regenerating the analysis panels used in the preprint from the archived run outputs.
- Added CPU-only Docker/Singularity and SLURM helper scripts under `hpc/` for manifest-based HPC reproduction.
- Added manifest generation, per-row SLURM execution, smoke testing, and manifest-summary collection utilities for large replicate sweeps.

### 🧩 Changes

- Renamed the former project-specific follow-up runner to the neutral `run_inhibitory_plasticity_maps.py`.
- Removed project-private paths and conference-specific naming from public scripts and documentation.
- Removed the original 2015 Diehl-Cook implementation from the tracked repository while keeping the public codebase focused on the current reproducible implementation.
- Configured `.gitignore` so manuscript drafts, private notes, local smoke runs, and large simulation outputs are excluded from Git.
- Added a Zenodo-oriented `runs/README.md` describing the external data folders required for reproducing the preprint analyses.
- Updated the figure-generation script to rely only on the three intended archived data folders:
  - `runs/long_30k_sweeps_manifest_HPC/`
  - `runs/slow_budget_maps_30k_manifest_HPC/`
  - `runs/long_visualization_60k_2ep/`

### 📚 Documentation

- Rewrote the main `README.md` for the public repository with a project overview, repository layout, setup instructions, smoke-test command, main model options, sweep examples, HPC pointer, figure-regeneration instructions, and citation guidance.
- Added `hpc/README.md` documenting Docker image construction, Singularity/Apptainer conversion, expected cluster layout, required environment variables, smoke testing, manifest generation, array submission, and summary collection.
- Added `runs/README.md` documenting the planned Zenodo data package and how to place archived runs for figure regeneration.
- Added `CITATION.cff` metadata for GitHub citation support and future DOI-based software citation.
- Added GPL-3.0 licensing via `LICENSE`.

### 🔬 Reproducibility notes

- This release contains the code and workflow required to reproduce the simulation analyses, but not the large simulation outputs themselves.
- The preprint figure-generation workflow expects the Zenodo data package to be restored under `runs/`.
