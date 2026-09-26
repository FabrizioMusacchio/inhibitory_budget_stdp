# Zenodo data package for the preprint analyses

Large simulation outputs are intentionally not tracked in Git. To reproduce the preprint analyses and figure panels, download the associated Zenodo archive and place the following folders inside this `runs/` directory:

```text
runs/long_30k_sweeps_manifest_HPC/
runs/slow_budget_maps_30k_manifest_HPC/
runs/long_visualization_60k_2ep/
```

## Included Datasets

`long_30k_sweeps_manifest_HPC/` contains the primary 30,000-training-example, 5,000-test-example HPC sweep results used for the fixed-inhibition reference grid, direct matched-rule comparisons, and Vogels-style inhibitory STDP parameter map.

`slow_budget_maps_30k_manifest_HPC/` contains the additional slow-homeostatic and budget-constrained inhibitory-plasticity parameter maps.

`long_visualization_60k_2ep/` contains longer diagnostic visualization runs used to justify the 30,000-example, one-epoch analysis protocol.

## Reproducing Figure Panels

After placing the folders above under `runs/`, run from the repository root:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n diehl_cook_euler python additional_scripts/preprint_generate_figures.py
```

The script writes the preprint analysis panels to `papers/preprint/figures/` in the local working copy.

## Citation

Zenodo DOI: forthcoming.

Until the DOI is available, please cite the GitHub commit used for code and state that the simulation outputs correspond to the forthcoming Zenodo archive for:

```text
Musacchio F, Fuhrmann M. Inhibitory budget matching constrains homeostatic plasticity in competitive spiking networks. bioRxiv, forthcoming.
```
