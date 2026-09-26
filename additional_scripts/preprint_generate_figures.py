"""Generate figure panels for the inhibitory-plasticity preprint.

Panels are rebuilt from aggregated summaries, saved run summaries, and run-level
logs. Dependencies are limited to the standard library, NumPy, and Matplotlib so
that the script runs in the existing ``diehl_cook_euler`` environment.

author: Fabrizio Musacchio
date:   August 2026
"""
# %% IMPORTS
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
import csv
import json
import math
import shutil
import subprocess

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# set font type for PDF and PS output to TrueType instead of default Type 3:
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42

# set global font to Arial:
mpl.rcParams["font.family"] = "Arial"
# %% PATHS
ROOT_DIR = Path(__file__).resolve().parents[1]
PREPRINT_DIR = ROOT_DIR / "papers" / "preprint"
FIGURES_DIR = PREPRINT_DIR / "figures"
AFFINITY_SAFE_PDFS = {
    ("figure1", "panel_l.pdf"),
    ("figure2", "panel_c.pdf"),
    ("figure2", "panel_d.pdf"),
    ("figure2", "panel_f.pdf"),
    ("figure6", "panel_d.pdf"),
    ("figureS1", "panel_b.pdf"),
    ("figureS1", "panel_d.pdf"),
}

HPC_SWEEP_DIR = ROOT_DIR / "runs" / "long_30k_sweeps_manifest_HPC"
HPC_SWEEP_SUMMARY = HPC_SWEEP_DIR / "reconstructed_summary_from_run_summary.csv"
SLOW_BUDGET_MAP_DIR = ROOT_DIR / "runs" / "slow_budget_maps_30k_manifest_HPC"
FIXED_GRID_DIR = HPC_SWEEP_DIR / "fixed_grid" / "fixed_grid"
COMPARE_BAD_BUDGET_DIR = HPC_SWEEP_DIR / "compare_mismatched" / "compare_regime"
COMPARE_MATCHED_DIR = HPC_SWEEP_DIR / "compare_matched" / "compare_regime"
VOGELS_MAP_DIR = HPC_SWEEP_DIR / "vogels_map" / "vogels_stability_map"
SLOW_MAP_DIR = SLOW_BUDGET_MAP_DIR / "slow_map" / "slow_homeostat_map"
BUDGET_MAP_DIR = SLOW_BUDGET_MAP_DIR / "budget_map" / "normalized_slow_homeostat_budget_map"
LONG_FIXED_RUN = ROOT_DIR / "runs" / "long_visualization_60k_2ep" / "fixed_seed0"
REPRESENTATIVE_FIXED_RUN = COMPARE_MATCHED_DIR / "fixed_input2_wAiAe10_thetaPlus0p05_seed0"
REPRESENTATIVE_VOGELS_RUN = COMPARE_MATCHED_DIR / "vogels_stable_input2_wAiAe10_thetaPlus0p05_seed0"
REPRESENTATIVE_VOGELS_HIGH_RHO_RUN = COMPARE_MATCHED_DIR / "vogels_unstable_input2_wAiAe10_thetaPlus0p05_seed0"
REPRESENTATIVE_SLOW_RUN = COMPARE_MATCHED_DIR / "slow_homeostat_unconstrained_input2_wAiAe10_thetaPlus0p05_seed0"
REPRESENTATIVE_NORMALIZED_RUN = COMPARE_MATCHED_DIR / "normalized_slow_homeostat_input2_wAiAe10_thetaPlus0p05_seed0"
FAILED_VOGELS_LOW_RUN = COMPARE_MATCHED_DIR / "vogels_stable_input2_wAiAe10_thetaPlus0p05_seed2"
FAILED_MISMATCHED_BUDGET_RUN = COMPARE_BAD_BUDGET_DIR / "normalized_slow_homeostat_input2_wAiAe10_thetaPlus0p05_seed0"
FAILED_SLOW_SATURATION_RUN = SLOW_MAP_DIR / "slow_eta0p001_r0hz0p03_seed2"
# %% GLOBAL PLOT SETTINGS
CM_TO_INCH  = 1.0 / 2.54
PANEL_DPI   = 300

FONTSIZE        = 8
LABEL_FONTSIZE  = 7
TICK_FONTSIZE   = 7
LEGEND_FONTSIZE = 7
CONFMAT_LABEL_FONTSIZE = 5
HEATMAP_LABEL_FONTSIZE = 5

BLUE        = "#315c80"
LIGHT_BLUE  = "#79a9d6"
ORANGE      = "#c9742a"
RED         = "#b23b3b"
GREEN       = "#3d7d4a"
GRAY        = "#606060"
LIGHT_GRAY  = "#e8e8e8"

MAP_SPINES  = {"top": False, "right": False, "left": False, "bottom": False}
NO_SPINES   = {"top": False, "right": False, "left": False, "bottom": False}
CMAP_STABILITY = mcolors.LinearSegmentedColormap.from_list(
    "muted_stability",
    ["#f7f7f4", "#d7d2c0", "#8f9398", "#4b5663"])
CMAP_ACCURACY = mcolors.LinearSegmentedColormap.from_list(
    "muted_accuracy",
    ["#f7f7f4", "#dbe8d2", "#9ebfc0", "#667895"])
CMAP_ACTIVITY = mcolors.LinearSegmentedColormap.from_list(
    "muted_activity",
    ["#f8f7f4", "#dccdb8", "#b6816e", "#6e4e5a"])
CMAP_PROCESSED = mcolors.LinearSegmentedColormap.from_list(
    "muted_processed",
    ["#f7f7f4", "#ded8e8", "#a99bc9", "#69548f"])

METHOD_COLORS = {
    "fixed": GRAY,
    "vogels_stable": BLUE,
    "slow_homeostat_unconstrained": ORANGE,
    "normalized_slow_homeostat": GREEN,
    "vogels_unstable": LIGHT_BLUE}
METHOD_LABELS = {
    "fixed": "fixed",
    "vogels_stable": "Vogels\nlow $\\rho$",
    "slow_homeostat_unconstrained": "slow-\nhomeostatic",
    "normalized_slow_homeostat": "budget-\nconstrained",
    "vogels_unstable": "Vogels\nhigh $\\rho$"}
METHOD_LEGEND_LABELS = {
    "fixed": "fixed",
    "vogels_stable": r"Vogels low $\rho$",
    "vogels_unstable": r"Vogels high $\rho$",
    "slow_homeostat_unconstrained": "slow-homeostatic",
    "normalized_slow_homeostat": "budget"}
METHOD_LINESTYLES = {
    "fixed": "-",
    "vogels_stable": "-",
    "vogels_unstable": "--",
    "slow_homeostat_unconstrained": "-",
    "normalized_slow_homeostat": "-"}
METHOD_RUN_PATTERNS = {
    "fixed": "fixed_input2_wAiAe10_thetaPlus0p05_seed*",
    "vogels_stable": "vogels_stable_input2_wAiAe10_thetaPlus0p05_seed*",
    "vogels_unstable": "vogels_unstable_input2_wAiAe10_thetaPlus0p05_seed*",
    "slow_homeostat_unconstrained": "slow_homeostat_unconstrained_input2_wAiAe10_thetaPlus0p05_seed*",
    "normalized_slow_homeostat": "normalized_slow_homeostat_input2_wAiAe10_thetaPlus0p05_seed*"}
# %% CLASSES AND PANEL FUNCTION
@dataclass(frozen=True)
class PanelConfig:
    output_subdir: str
    output_name: str
    figsize_cm: tuple[float, float]
    title: str
    xlabel: str | None = None
    ylabel: str | None = None
    xlim: tuple[float | None, float | None] | None = None
    ylim: tuple[float | None, float | None] | None = None
    cmap: str | mcolors.Colormap | None = None
    colorbar_zero: bool = False
    legend_show: bool = False
    legend_loc: str = "best"
    grid_axis: str | None = "y"
    xrotation: float = 0.0
    xtick_linebreaks: bool = True
    spines: dict[str, bool] = field(default_factory=lambda: {
        "top": False,
        "right": False,
        "left": True,
        "bottom": True,
    })

def panel(
    output_subdir: str,
    output_name: str,
    title: str,
    *,
    figsize_cm: tuple[float, float] = (7.0, 5.0),
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    cmap: str | mcolors.Colormap | None = None,
    colorbar_zero: bool = False,
    legend_show: bool = False,
    legend_loc: str = "best",
    grid_axis: str | None = "y",
    xrotation: float = 0.0,
    xtick_linebreaks: bool = True,
    spines: dict[str, bool] | None = None,
) -> PanelConfig:
    return PanelConfig(
        output_subdir=output_subdir,
        output_name=output_name,
        title=title,
        figsize_cm=figsize_cm,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        cmap=cmap,
        colorbar_zero=colorbar_zero,
        legend_show=legend_show,
        legend_loc=legend_loc,
        grid_axis=grid_axis,
        xrotation=xrotation,
        xtick_linebreaks=xtick_linebreaks,
        spines=spines if spines is not None else {
            "top": False,
            "right": False,
            "left": True,
            "bottom": True,
        },
    )
# %% PANEL CONTROLS
# Edit this dictionary to resize or restyle individual panels before exporting
# them for LaTeX, Affinity, or poster assembly. The drawing functions below
# intentionally read layout, labels, limits, grids, legends, and spine settings
# from these named configs rather than hard-coding them in the plotting logic.

RF_SIZE         = (4.5, 4.5)
CONFMAT_SIZE    = (5.15, 5.5)
HEATMAP_SIZE              = (5.0, 4.2)
HEATMAP_DOUBLE_TITLE_SIZE = (5.9, 4.98)
METHODS_DIAGNOSTIC_LINE_PLOT_SIZE               = (5.0, 3.4)
METHODS_DIAGNOSTIC_LINE_PLOT__DOUBLE_TITLE_SIZE = (5, 3.78)

PREPRINT_PANELS: dict[str, PanelConfig] = {
    # -------------------------------
    # FIGURE 1:
    # -------------------------------
    "representative_receptive_fields": panel(
        "figure1",
        "panel_d.pdf",
        "representative RF", # representative receptive fields
        figsize_cm=RF_SIZE,
        grid_axis=None,
    ),
    "representative_confusion": panel(
        "figure1",
        "panel_e.pdf",
        "confusion matrix", # representative confusion matrix
        figsize_cm=CONFMAT_SIZE,
        grid_axis=None,
        spines=NO_SPINES,
    ),
    "fixed_entropy_dynamics": panel(
        "figure1",
        "panel_i.pdf",
        "neuron usage entropy",
        figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
        xlabel="training examples",
        ylabel="normalized entropy",
        xlim=(0, 30000),
        ylim=(0.0, 1.005), # (0.97, 1.005)
        grid_axis="y",
        legend_show=False,
        legend_loc="upper right",
    ),
    "fixed_spike_dynamics": panel(
        "figure1",
        "panel_j.pdf",
        "rolling population activity",
        figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
        xlabel="training examples",
        ylabel="spikes / image",
        xlim=(0, 30000),
        ylim=(0, 40),
        grid_axis="y",
    ),
    "fixed_gi_dynamics": panel(
        "figure1",
        "panel_k.pdf",
        "rolling inhibitory conductance",
        figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
        xlabel="training examples",
        ylabel=r"mean conduct.",
        xlim=(0, 30000),
        ylim=(0, 1.7),
        grid_axis="y",
    ),
    "fixed_selected_accuracy": panel(
        "figure1",
        "panel_l.pdf",
        "accuracy across replicates",
        figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
        ylabel="test accuracy",
        ylim=(0, 1), # (0.78, 0.88)
        grid_axis="y",
    ),
    "fixed_grid_stability": panel(
        "figure1",
        "panel_f.pdf",
        "stability map",
        figsize_cm=HEATMAP_SIZE,
        xlabel=r"inhibition $w^{IE}$",
        ylabel=r"input gain $g$",
        cmap=CMAP_STABILITY,
        grid_axis=None,
        #xrotation=25,
        spines=MAP_SPINES,
    ),
    "fixed_grid_accuracy": panel(
        "figure1",
        "panel_g.pdf",
        "test accuracy map",
        figsize_cm=HEATMAP_SIZE,
        xlabel=r"inhibition $w^{IE}$",
        ylabel=r"input gain $g$",
        cmap=CMAP_ACCURACY,
        colorbar_zero=True,
        grid_axis=None,
        spines=MAP_SPINES,
    ),
    "fixed_grid_activity": panel(
        "figure1",
        "panel_h.pdf",
        "mean activity map",
        figsize_cm=HEATMAP_SIZE,
        xlabel=r"inhibition $w^{IE}$",
        ylabel=r"input gain $g$",
        cmap=CMAP_ACTIVITY,
        grid_axis=None,
        spines=MAP_SPINES,
    ),
    # -------------------------------
    # FIGURE 2: Vogels-style inhibition
    # -------------------------------
    "supp_vogels_failure_low_bad_rf": panel("figure3", 
                                            "panel_a.pdf", 
                                            r"low $\rho$: RF at abort", 
                                            figsize_cm=RF_SIZE,
                                            grid_axis=None),
    "supp_vogels_failure_low_good_rf": panel("figure3", 
                                             "panel_b.pdf", 
                                             r"low $\rho$: completed RF", 
                                             figsize_cm=RF_SIZE, 
                                             grid_axis=None),
    "supp_vogels_failure_low_confusion": panel("figure3", 
                                               "panel_d.pdf", 
                                               r"low $\rho$: compl. conf. matrix", 
                                               figsize_cm=CONFMAT_SIZE,
                                               grid_axis=None, 
                                               spines=NO_SPINES),
    "supp_vogels_failure_high_good_rf": panel("figure3", 
                                              "panel_c.pdf", 
                                              r"high $\rho$: completed RF", 
                                              figsize_cm=RF_SIZE,
                                              grid_axis=None),
    "supp_vogels_failure_high_confusion": panel("figure3", 
                                                "panel_e.pdf", 
                                                r"high $\rho$: compl. conf. matrix", 
                                                figsize_cm=CONFMAT_SIZE,
                                                grid_axis=None, 
                                                spines=NO_SPINES),
    "supp_vogels_failure_entropy": panel("figure3", 
                                         "panel_f.pdf", 
                                         r"neuron usage entropy", 
                                         figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                         xlabel="training examples", 
                                         ylabel="normalized entropy", 
                                         xlim=(0, 30000), 
                                         ylim=(0.0, 1.005), 
                                         legend_show=True, 
                                         legend_loc="lower right", 
                                         grid_axis="y"),
    "supp_vogels_failure_window_spikes": panel("figure3", 
                                               "panel_g.pdf", 
                                               r"windowed population activity", 
                                               figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                               xlabel="training examples", 
                                               ylabel="spikes / image", 
                                               xlim=(0, 30000), 
                                               ylim=(5, 3000), 
                                               legend_show=False, 
                                               legend_loc="upper right", 
                                               grid_axis="y"),
    "supp_vogels_failure_window_gi": panel("figure3", 
                                           "panel_i.pdf", 
                                           "windowed inhibitory conductance", 
                                           figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                           xlabel="training examples", 
                                           ylabel=r"mean conduct.", 
                                           xlim=(0, 30000), 
                                           ylim=(0.1, 300), 
                                           legend_show=False, 
                                           legend_loc="upper right", 
                                           grid_axis="y"),
    "supp_vogels_failure_spikes": panel("figure3", 
                                        "panel_j.pdf", 
                                        r"rolling spikes", 
                                        figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                        xlabel="training examples", 
                                        ylabel="spikes / image", 
                                        xlim=(0, 30000), 
                                        ylim=(5, 30000), 
                                        legend_show=False, 
                                        legend_loc="upper left", 
                                        grid_axis="y"),
    "supp_vogels_failure_gi": panel("figure3", 
                                    "panel_l.pdf", 
                                    r"rolling inhibitory conductance", 
                                    figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE, 
                                    xlabel="training examples", 
                                    ylabel=r"mean conduct.", 
                                    xlim=(0, 30000), 
                                    ylim=(0.1, 4000), 
                                    legend_show=False, 
                                    legend_loc="upper left", 
                                    grid_axis="y"),
    "vogels_stability_map": panel(
        "figure3",
        "panel_m.pdf",
        "stability map",
        figsize_cm=(5.5, 4.6),
        xlabel=r"learning rate $\eta_I$",
        ylabel=r"target offset $\rho$",
        cmap=CMAP_STABILITY,
        grid_axis=None,
        xrotation=25,
        spines=MAP_SPINES,
    ),
    "vogels_processed_map": panel(
        "figure3",
        "panel_n.pdf",
        "examples processed\nbefore abort",
        figsize_cm=HEATMAP_DOUBLE_TITLE_SIZE,
        xlabel=r"learning rate $\eta_I$",
        ylabel=r"target offset $\rho$",
        cmap=CMAP_PROCESSED,
        grid_axis=None,
        xrotation=25,
        spines=MAP_SPINES,
    ),
    "vogels_seed_counts": panel(
        "figure3",
        "panel_h.pdf",
        "stable runs",
        figsize_cm=(2.3, 3.42),
        xlabel="replicate",
        ylabel="stable param. pairs",
        ylim=(0, 20),
        grid_axis="y",
    ),
    "vogels_outcomes": panel(
        "figure3",
        "panel_k.pdf",
        "outcomes",
        figsize_cm=(2.3, 3.4),
        ylabel="runs",
        ylim=(0, 105),
        xrotation=25,
        grid_axis="y",
    ),
    # -------------------------------
    # FIGURE 3: Inhibitory plasticity depends on the conserved inhibitory budget.
    # -------------------------------
    "budget_rule_stability": panel(
        "figure2",
        "panel_a.pdf",
        "run completion",
        figsize_cm=(5.5, 4.4),
        ylabel="stable replicates",
        ylim=(0, 5.4),
        xrotation=30,
        xtick_linebreaks=False,
        grid_axis="y",
    ),
    "budget_final_weight": panel(
        "figure2",
        "panel_b.pdf",
        "budget-constrained\nfinal mean inhibition",
        figsize_cm=(5.0, 3.57),
        ylabel=r"mean $w^{IE}$",
        ylim=(0, 22),
        grid_axis="y",
    ),
    "budget_matched_accuracy": panel(
        "figure2",
        "panel_c.pdf",
        "classification performance",
        figsize_cm=(5.0, 4.4),
        ylabel="test accuracy",
        ylim=(0.82, 0.861),
        xrotation=30,
        xtick_linebreaks=False,
        grid_axis="y",
    ),
    "budget_matched_activity": panel(
        "figure2",
        "panel_d.pdf",
        "mean excitatory activity",
        figsize_cm=(5.0, 4.4),
        ylabel="spikes / image",
        ylim=(21.35, 21.68),
        xrotation=30,
        xtick_linebreaks=False,
        grid_axis="y",
    ),
    "budget_matched_gi_dynamics": panel(
        "figure2",
        "panel_e.pdf",
        "inhibitory conductance dynamics",
        figsize_cm=(6.15, 3.55),
        xlabel="training examples",
        ylabel=r"mean conduct.",
        xlim=(0, 30000),
        ylim=(0, 1.7),
        legend_show=False,
        legend_loc="lower right",
        grid_axis="y",
    ),
    "budget_matched_final_weight_all": panel(
        "figure2",
        "panel_f.pdf",
        "final inhibitory weight by rule",
        figsize_cm=(5.0, 4.4),
        ylabel=r"mean $w^{IE}$",
        ylim=(9.9, 10.65),
        xrotation=30,
        xtick_linebreaks=False,
        grid_axis="y",
    ),
    # -------------------------------
    # FIGURE 4: Slow-homeostatic inhibition:
    "supp_slow_failure_bad_rf": panel("figure4", 
                                      "panel_a.pdf", 
                                      "saturation example: RF at abort", 
                                      figsize_cm=RF_SIZE,
                                      grid_axis=None),
    "supp_slow_failure_good_rf": panel("figure4", 
                                       "panel_b.pdf", 
                                       "RF (completed)",
                                       figsize_cm=RF_SIZE,
                                       grid_axis=None),
    "supp_slow_failure_confusion": panel("figure4", 
                                         "panel_c.pdf", 
                                         "confusion matrix (completed)",
                                         figsize_cm=CONFMAT_SIZE,
                                         grid_axis=None, 
                                         spines=NO_SPINES),
    "supp_slow_failure_entropy": panel("figure4", 
                                       "panel_d.pdf", 
                                       "neuron usage entropy", 
                                       figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                       xlabel="training examples", 
                                       ylabel="normalized entropy", 
                                       xlim=(0, 30000), 
                                       ylim=(0.0, 1.005), 
                                       legend_show=False, 
                                       legend_loc="lower right", 
                                       grid_axis="y"),
    "supp_slow_failure_window_spikes": panel("figure4", 
                                             "panel_e.pdf", 
                                             "windowed population activity", 
                                             figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                             xlabel="training examples", 
                                             ylabel="spikes / image", 
                                             xlim=(0, 30000), 
                                             ylim=(5, 3000), 
                                             legend_show=False, 
                                             legend_loc="upper left", 
                                             grid_axis="y"),
    "supp_slow_failure_window_gi": panel("figure4", 
                                         "panel_f.pdf", 
                                         "windowed inhibitory conductance",
                                         figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                         xlabel="training examples", 
                                         ylabel=r"mean conduct.", 
                                         xlim=(0, 30000), 
                                         ylim=(0.1, 300), 
                                         legend_show=False, 
                                         legend_loc="upper left", 
                                         grid_axis="y"),
    "supp_slow_failure_spikes": panel("figure4", 
                                      "panel_g.pdf", 
                                      "rolling spikes", 
                                      figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                      xlabel="training examples", 
                                      ylabel="spikes / image", 
                                      xlim=(0, 30000), 
                                      ylim=(5, 30000), 
                                      legend_show=False, 
                                      legend_loc="upper left", 
                                      grid_axis="y"),
    "supp_slow_failure_gi": panel("figure4", 
                                  "panel_h.pdf", 
                                  "rolling inhibitory conductance", 
                                  figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                  xlabel="training examples", 
                                  ylabel=r"mean conduct", 
                                  xlim=(0, 30000), 
                                  ylim=(0.1, 4000), 
                                  legend_show=True, 
                                  legend_loc="upper left", 
                                  grid_axis="y"),
    "slow_stability_map": panel(
        "figure4",
        "panel_i.pdf",
        "stability map",
        figsize_cm=(5.5, 4.65),
        xlabel=r"learning rate $\eta_I$",
        ylabel=r"target rate $r_0$ [Hz]",
        cmap=CMAP_STABILITY,
        grid_axis=None,
        spines=MAP_SPINES,
        xrotation=25,
    ),
    "slow_processed_map": panel(
        "figure4",
        "panel_j.pdf",
        "examples processed\nbefore abort",
        figsize_cm=(5.9, 4.98),
        xlabel=r"learning rate $\eta_I$",
        ylabel=r"target rate $r_0$ [Hz]",
        cmap=CMAP_PROCESSED,
        grid_axis=None,
        spines=MAP_SPINES,
        xrotation=25,
    ),
    # -------------------------------
    # FIGURE 5: Mechanistic diagnostics distinguish stable replacement from failure
    # -------------------------------
    "mechanism_failure_spikes": panel(
        "figure6",
        "panel_a.pdf",
        "failure trajectories: population activity",
        figsize_cm=(8.0, 4.4),
        xlabel="training examples",
        ylabel="spikes / image",
        xlim=(0, 5000),
        ylim=(5, 30000),
        legend_show=False,
        legend_loc="upper left",
        grid_axis="y",
    ),
    "mechanism_failure_gi": panel(
        "figure6",
        "panel_b.pdf",
        "failure trajectories: inhibition",
        figsize_cm=(8.0, 4.4),
        xlabel="training examples",
        ylabel=r"mean inhibitory conductance",
        xlim=(0, 5000),
        ylim=(0.1, 4000),
        legend_show=False,
        legend_loc="upper left",
        grid_axis="y",
    ),
    "mechanism_weight_distribution": panel(
        "figure6",
        "panel_c.pdf",
        r"final $W^{IE}$ weight ranges",
        figsize_cm=(8.0, 4.65),
        xlabel=r"inhibitory weight $w^{IE}_{ij}$",
        ylabel=None,
        grid_axis="x",
    ),
    "mechanism_column_sums": panel(
        "figure6",
        "panel_d.pdf",
        r"final inhibitory budget per excitatory neuron",
        figsize_cm=(8.0, 4.4),
        ylabel=r"column sum $\sum_i w^{IE}_{ij}$",
        ylim=(0,8250), # ylim=(3800, 8250),
        grid_axis="y",
    ),
    # -------------------------------
    # FIGURE 6: Budget-constrained inhibition
    # -------------------------------
    "supp_budget_failure_bad_rf": panel("figure5", 
                                        "panel_a.pdf", 
                                        "mismatched budget failure:\nRF at abort", 
                                        figsize_cm=(4.5, 4.75),
                                        grid_axis=None),
    "supp_budget_failure_good_rf": panel("figure5", 
                                         "panel_b.pdf", 
                                         "matched budget completed:\nRF", 
                                         figsize_cm=(4.5, 4.75),
                                         grid_axis=None),
    "supp_budget_failure_confusion": panel("figure5", 
                                           "panel_c.pdf", 
                                           "confusion matrix (completed)",
                                           figsize_cm=CONFMAT_SIZE,
                                           grid_axis=None, 
                                           spines=NO_SPINES),
    "supp_budget_failure_entropy": panel("figure5", 
                                         "panel_d.pdf", 
                                         "neuron usage entropy", 
                                         figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                         xlabel="training examples", 
                                         ylabel="normalized entropy", 
                                         xlim=(0, 30000), 
                                         ylim=(0.85, 1.005), 
                                         legend_show=True, 
                                         legend_loc="lower right", 
                                         grid_axis="y"),
    "supp_budget_failure_window_spikes": panel("figure5", 
                                               "panel_e.pdf", 
                                               "windowed population activity", 
                                               figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                               xlabel="training examples", 
                                               ylabel="spikes / image", 
                                               xlim=(0, 30000), 
                                               ylim=(5, 3000), 
                                               legend_show=False, 
                                               legend_loc="upper left", 
                                               grid_axis="y"),
    "supp_budget_failure_window_gi": panel("figure5", 
                                           "panel_f.pdf", 
                                           "windowed inhibitory conductance", 
                                           figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                           xlabel="training examples", 
                                           ylabel=r"mean conduct.", 
                                           xlim=(0, 30000), 
                                           ylim=(0.1, 300), 
                                           legend_show=False, 
                                           legend_loc="upper left", 
                                           grid_axis="y"),
    "supp_budget_failure_spikes": panel("figure5", 
                                        "panel_g.pdf", 
                                        "rolling spikes", 
                                        figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                        xlabel="training examples", 
                                        ylabel="spikes / image", 
                                        xlim=(0, 30000), 
                                        ylim=(5, 30000), 
                                        legend_show=False, 
                                        legend_loc="upper left", 
                                        grid_axis="y"),
    "supp_budget_failure_gi": panel("figure5", 
                                    "panel_h.pdf", 
                                    "rolling inhibitory conductance", 
                                    figsize_cm=METHODS_DIAGNOSTIC_LINE_PLOT_SIZE,
                                    xlabel="training examples", 
                                    ylabel=r"mean conduct.", 
                                    xlim=(0, 30000), 
                                    ylim=(0.1, 4000), 
                                    legend_show=False, 
                                    legend_loc="upper left", 
                                    grid_axis="y"),
    "budget_stability_map": panel(
        "figure5",
        "panel_i.pdf",
        "stability map",
        figsize_cm=(5.5, 4.65),
        xlabel=r"learning rate $\eta_I$",
        ylabel=r"inhibitory budget $B_I$",
        cmap=CMAP_STABILITY,
        grid_axis=None,
        spines=MAP_SPINES,
        xrotation=25,
    ),
    "budget_processed_map": panel(
        "figure5",
        "panel_j.pdf",
        "examples processed\nbefore abort",
        figsize_cm=(5.9, 4.98),
        xlabel=r"learning rate $\eta_I$",
        ylabel=r"inhibitory budget $B_I$",
        cmap=CMAP_PROCESSED,
        grid_axis=None,
        spines=MAP_SPINES,
        xrotation=25,
    ),
    # -------------------------------
    # SI-FIGURE 2: 
    # -------------------------------
    "supp_budget_fixed_stability": panel(
        "figureS1",
        "panel_a.pdf",
        r"fixed inhibition: stable replicates at $g=2.0$",
        figsize_cm=(7.2, 5.0),
        xlabel=r"inhibitory budget $B_I$",
        ylabel="stable replicates",
        ylim=(0, 5.4),
        grid_axis="y",
    ),
    "supp_budget_fixed_activity": panel(
        "figureS1",
        "panel_b.pdf",
        r"fixed inhibition: activity at $g=2.0$",
        figsize_cm=(7.2, 5.0),
        xlabel=r"inhibitory budget $B_I$",
        ylabel="spikes / image",
        ylim=(0, 285),
        grid_axis="y",
    ),
    "supp_budget_normalized_stability": panel(
        "figureS1",
        "panel_c.pdf",
        "budget-constrained inhibition:\nmatched vs mismatched budget",
        figsize_cm=(4.5, 5.27),
        ylabel="stable replicates",
        ylim=(0, 5.4),
        grid_axis="y",
    ),
    "supp_budget_normalized_weight": panel(
        "figureS1",
        "panel_d.pdf",
        "budget-constrained inhibition:\nfinal mean $w^{IE}$",
        figsize_cm=(4.5, 5.0),
        ylabel=r"mean $w^{IE}$",
        ylim=(8, 22),
        grid_axis="y",
    ),
    # -------------------------------
    # SI-FIGURE 1: long fixed run diagnostics
    # -------------------------------
    "supp_training_spikes": panel("figureS2", 
                                  "panel_a.pdf", 
                                  "long fixed run activity", 
                                  figsize_cm=(8.0, 4.4),
                                  xlabel="training examples within epoch", 
                                  ylabel="spikes / image", 
                                  xlim=(0, 60000),
                                  ylim=(0,35),
                                  grid_axis="y", 
                                  legend_show=True),
    "supp_training_entropy": panel("figureS2", 
                                   "panel_b.pdf", 
                                   "long fixed run usage entropy", 
                                   figsize_cm=(8.0, 4.4), 
                                   xlabel="training examples within epoch", 
                                   ylabel="normalized entropy", 
                                   xlim=(0, 60000), 
                                   ylim=(0.97, 1.005), 
                                   grid_axis="y", 
                                   legend_show=True),
    "supp_training_accuracy": panel("figureS2", 
                                    "panel_c.pdf", 
                                    "test accuracy after\neach epoch", 
                                    figsize_cm=(4.0, 4.4), 
                                    ylabel="test accuracy", 
                                    ylim=(0.82, 0.88), 
                                    grid_axis="y"),
    "supp_training_weight_change": panel("figureS2", 
                                         "panel_d.pdf", 
                                         "feedforward weights after\nadditional training", 
                                         figsize_cm=(4.0, 4.77), 
                                         ylabel="summary value", 
                                         grid_axis="y"),
}
# %% CORE HELPER FUNCTIONS
def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def std(values: Iterable[float | None]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def sem(values: Iterable[float | None]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / math.sqrt(len(vals)))


def read_summary(path: Path) -> list[dict[str, object]]:
    numeric_columns = {
        "seed", "input_intensity", "w_aiae", "w_aeai", "theta_plus_mV",
        "eta_ie", "rho_ie", "w_ie_max", "accuracy", "train_mean_spikes",
        "train_usage_entropy_norm", "dead_fraction", "firing_rate_cv",
        "gi_mean", "ai_ae_mean", "ai_ae_max", "ai_ae_std", "theta_mean",
        "theta_max", "processed_train_examples", "aiae_target_sum",
    }
    rows: list[dict[str, object]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, object] = dict(raw)
            for col in numeric_columns:
                if col in row:
                    row[col] = to_float(row[col])
            row["aborted_bool"] = to_bool(row.get("aborted", False))
            rows.append(row)
    return rows


def read_hpc_summary() -> list[dict[str, object]]:
    return read_summary(HPC_SWEEP_SUMMARY)


def read_run_directory_summaries(run_root: Path, *, block: str, run_set: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not run_root.exists():
        return rows
    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        config_path = run_dir / "config.json"
        summary_path = run_dir / "logs" / "run_summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue
        config = json.loads(config_path.read_text())
        summary = json.loads(summary_path.read_text())
        row: dict[str, object] = {
            "block": block,
            "run_set": run_set,
            "run_dir": str(run_dir),
            "condition": summary.get("condition", config.get("istdp_rule", "")),
            "seed": to_float(summary.get("seed", config.get("seed"))),
            "eta_ie": to_float(summary.get("eta_ie", config.get("eta_ie"))),
            "rho_ie": to_float(summary.get("rho_ie", config.get("rho_ie"))),
            "slow_homeostat_target_rate_hz": to_float(summary.get("slow_homeostat_target_rate_hz", config.get("slow_homeostat_target_rate_hz"))),
            "aiae_target_sum": to_float(summary.get("aiae_target_sum", config.get("aiae_target_sum"))),
            "accuracy": to_float(summary.get("accuracy")),
            "train_mean_spikes": to_float(summary.get("train_mean_spikes")),
            "train_usage_entropy_norm": to_float(summary.get("train_usage_entropy_norm")),
            "dead_fraction": to_float(summary.get("dead_fraction")),
            "firing_rate_cv": to_float(summary.get("firing_rate_cv")),
            "gi_mean": to_float(summary.get("gi_mean")),
            "ai_ae_mean": to_float(summary.get("ai_ae_mean")),
            "ai_ae_max": to_float(summary.get("ai_ae_max")),
            "ai_ae_std": to_float(summary.get("ai_ae_std")),
            "theta_mean": to_float(summary.get("theta_mean")),
            "theta_max": to_float(summary.get("theta_max")),
            "processed_train_examples": to_float(summary.get("processed_train_examples")),
            "abort_reason": summary.get("abort_reason", ""),
            "aborted_bool": bool(summary.get("aborted", False)),
        }
        rows.append(row)
    return rows


def hpc_rows(*, block: str | None = None, run_set: str | None = None) -> list[dict[str, object]]:
    rows = read_hpc_summary()
    if block is not None:
        rows = [r for r in rows if r.get("block") == block]
    if run_set is not None:
        rows = [r for r in rows if r.get("run_set") == run_set]
    return rows


def fixed_grid_rows() -> list[dict[str, object]]:
    return hpc_rows(block="fixed_grid", run_set="fixed_grid")


def compare_rows(*, budget: str) -> list[dict[str, object]]:
    block = "compare_matched" if budget == "matched" else "compare_mismatched"
    return hpc_rows(block=block, run_set="compare_regime")


def vogels_map_rows() -> list[dict[str, object]]:
    return hpc_rows(block="vogels_map", run_set="vogels_stability_map")


def slow_map_rows() -> list[dict[str, object]]:
    return read_run_directory_summaries(SLOW_MAP_DIR, block="slow_map", run_set="slow_homeostat_map")


def budget_map_rows() -> list[dict[str, object]]:
    return read_run_directory_summaries(BUDGET_MAP_DIR, block="budget_map", run_set="normalized_slow_homeostat_budget_map")


def unique_sorted(rows: list[dict[str, object]], key: str) -> list[float]:
    return sorted({float(r[key]) for r in rows if r.get(key) is not None})


def heatmap_table(
    rows: list[dict[str, object]],
    row_key: str,
    col_key: str,
    value_fn: Callable[[list[dict[str, object]]], float],
) -> tuple[list[float], list[float], np.ndarray]:
    y_values = unique_sorted(rows, row_key)
    x_values = unique_sorted(rows, col_key)
    table = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for yi, y in enumerate(y_values):
        for xi, x in enumerate(x_values):
            subset = [r for r in rows if r.get(row_key) == y and r.get(col_key) == x]
            if subset:
                table[yi, xi] = value_fn(subset)
    return y_values, x_values, table


def apply_panel_style(ax: plt.Axes, cfg: PanelConfig) -> None:
    title = cfg.title
    title_suffix = getattr(ax, "_title_suffix", None)
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title, fontsize=FONTSIZE, pad=5)
    ax.set_axisbelow(True)
    if cfg.xlabel is not None:
        ax.set_xlabel(cfg.xlabel, fontsize=LABEL_FONTSIZE)
    if cfg.ylabel is not None:
        ax.set_ylabel(cfg.ylabel, fontsize=LABEL_FONTSIZE)
    if cfg.xlim is not None:
        ax.set_xlim(*cfg.xlim)
    if cfg.ylim is not None:
        ax.set_ylim(*cfg.ylim)
    for spine, visible in cfg.spines.items():
        ax.spines[spine].set_visible(visible)
    ax.tick_params(labelsize=TICK_FONTSIZE, length=2.5, width=0.8)
    for label in ax.get_xticklabels():
        label.set_rotation(cfg.xrotation)
        if cfg.xrotation:
            label.set_ha("right")
    if cfg.grid_axis is not None:
        ax.grid(axis=cfg.grid_axis, color=LIGHT_GRAY, linewidth=0.6, zorder=0)
    if cfg.legend_show:
        ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc=cfg.legend_loc)


def save_panel(fig: plt.Figure, cfg: PanelConfig) -> None:
    out_dir = FIGURES_DIR / cfg.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(cfg.output_name).stem
    for ext in (".pdf", ".png"):
        out_path = out_dir / f"{stem}{ext}"
        if ext == ".pdf":
            fig.savefig(out_path, dpi=PANEL_DPI, bbox_inches="tight", transparent=True)
            if (cfg.output_subdir, cfg.output_name) in AFFINITY_SAFE_PDFS:
                rewrite_pdf_for_affinity(out_path)
        else: 
            fig.savefig(out_path, dpi=PANEL_DPI, bbox_inches="tight")
    plt.close(fig)


def rewrite_pdf_for_affinity(pdf_path: Path) -> None:
    """Rewrite rare Matplotlib PDFs that trigger Affinity's PDF importer."""
    gs = shutil.which("gs")
    if gs is None:
        print(f"Ghostscript not found; leaving {pdf_path.name} as Matplotlib PDF.")
        return
    tmp_path = pdf_path.with_suffix(".affinity_tmp.pdf")
    cmd = [
        gs,
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dAutoRotatePages=/None",
        "-sOutputFile=" + str(tmp_path),
        str(pdf_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp_path.replace(pdf_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"Could not rewrite {pdf_path.name} for Affinity: {exc}")


def panel_xtick_labels(cfg: PanelConfig, labels: Iterable[str]) -> list[str]:
    labels = list(labels)
    if cfg.xtick_linebreaks:
        return labels
    return [label.replace("\n", " ") for label in labels]


def make_panel(cfg: PanelConfig, draw: Callable[[plt.Axes], None]) -> None:
    fig, ax = plt.subplots(figsize=(cfg.figsize_cm[0] * CM_TO_INCH, cfg.figsize_cm[1] * CM_TO_INCH))
    draw(ax)
    apply_panel_style(ax, cfg)
    fig.tight_layout(pad=0.6)
    save_panel(fig, cfg)


def draw_heatmap(
    ax: plt.Axes,
    y_values: list[float],
    x_values: list[float],
    data: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str | mcolors.Colormap,
    cbar_label: str,
    value_formatter: Callable[[float], str] | None = None,
    colorbar_zero: bool = False,
) -> None:
    if colorbar_zero:
        vmin = 0.0
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower", aspect="auto")
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels([f"{x:g}" for x in x_values])
    ax.set_yticklabels([f"{y:g}" for y in y_values])
    for yi in range(len(y_values)):
        for xi in range(len(x_values)):
            value = data[yi, xi]
            if np.isfinite(value):
                label = value_formatter(value) if value_formatter is not None else format_heatmap_value(value)
                frac = min(max((value - vmin) / max(vmax - vmin, 1e-12), 0.0), 1.0)
                r, g, b, _ = cmap_obj(frac)
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "white" if luminance < 0.45 else "black"
                ax.text(xi, yi, label, ha="center", va="center", fontsize=HEATMAP_LABEL_FONTSIZE, 
                        color=text_color)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE, length=2)
    cbar.set_label(cbar_label, fontsize=LABEL_FONTSIZE)
    cbar.outline.set_linewidth(0)


def format_examples_k(value: float) -> str:
    if value >= 29500:
        return "30k"
    if value >= 1000:
        return f"{value / 1000:.1f}k".replace(".0k", "k")
    return f"{value:.0f}"


def format_heatmap_value(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:.1e}".replace(".0e", "e")
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.0f}"
    if abs_value >= 1:
        return f"{value:.2g}"
    return f"{value:.2g}"


def run_dirs(pattern: str) -> list[Path]:
    return sorted(p for p in COMPARE_MATCHED_DIR.glob(pattern) if p.is_dir())


def completed_method_run_dirs(condition: str) -> list[Path]:
    rows = compare_rows(budget="matched")
    stable_seeds = {
        int(r["seed"])
        for r in rows
        if r["condition"] == condition and r.get("seed") is not None and not r["aborted_bool"]
    }
    dirs: list[Path] = []
    for run_dir in run_dirs(METHOD_RUN_PATTERNS[condition]):
        for seed in stable_seeds:
            if run_dir.name.endswith(f"_seed{seed}"):
                dirs.append(run_dir)
                break
    return sorted(dirs)


def rolling_mean(values: np.ndarray, window: int = 100) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if values.size < window:
        return np.arange(values.size), values
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(values, kernel, mode="valid")
    xs = np.arange(window - 1, values.size)
    return xs, smoothed


def inhibitory_weights_offdiag(run_dir: Path) -> np.ndarray:
    weights = np.load(run_dir / "snapshots" / "W_AiAe_ep0.npy").astype(float)
    if weights.shape[0] == weights.shape[1]:
        mask = ~np.eye(weights.shape[0], dtype=bool)
        return weights[mask]
    return weights.ravel()


def inhibitory_column_sums(run_dir: Path) -> np.ndarray:
    weights = np.load(run_dir / "snapshots" / "W_AiAe_ep0.npy").astype(float)
    return weights.sum(axis=0)


def draw_receptive_fields_from_run(ax: plt.Axes, run_dir: Path) -> None:
    weights = np.load(run_dir / "snapshots" / "W_XeAe_ep0.npy")
    n_show = min(weights.shape[1], 100)
    side = int(math.ceil(math.sqrt(n_show)))
    rf = weights[:, :n_show].T.reshape(n_show, 28, 28)
    rf_min = rf.min(axis=(1, 2), keepdims=True)
    rf_max = rf.max(axis=(1, 2), keepdims=True)
    rf = (rf - rf_min) / np.maximum(rf_max - rf_min, 1e-9)
    canvas = np.ones((side * 28, side * 28), dtype=float)
    for idx in range(n_show):
        row, col = divmod(idx, side)
        canvas[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = rf[idx]
    ax.imshow(canvas, cmap="gray_r", interpolation="nearest")
    ax.set_axis_off()


def color_to_cmap(name: str, color: str) -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(name, ["#ffffff", color])


def draw_confusion_from_run(ax: plt.Axes, run_dir: Path, *, color: str = BLUE) -> None:
    matrix_path = run_dir / "logs" / "confusion_matrix_ep0.npy"
    if not matrix_path.exists():
        img_path = run_dir / "plots" / "confusion_matrix_ep0.png"
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_axis_off()
        return

    matrix = np.load(matrix_path).astype(float)
    metrics_path = run_dir / "logs" / "metrics_ep0.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            ax._title_suffix = f"(overall accuracy={100.0 * float(accuracy):.1f}%)"
    vmax = max(float(matrix.max()), 1.0)
    cmap = color_to_cmap("method_confusion", color)
    ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=vmax)
    threshold = 0.55 * vmax
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(round(matrix[row, col]))
            text_color = "white" if matrix[row, col] > threshold else color
            ax.text(col, row, f"{value}", ha="center", va="center", 
                    fontsize=CONFMAT_LABEL_FONTSIZE, color=text_color)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xlabel("Predicted label", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("True label", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE, length=2.0)
    # remove bottom and left spines:
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def draw_window_metric(
    ax: plt.Axes,
    paths: list[Path],
    metric_key: str,
    *,
    color: str,
    label: str | None = None,
) -> None:
    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    for path in paths:
        data = json.loads(path.read_text())
        xs = np.asarray([float(d["example_end"]) for d in data], dtype=float)
        ys = np.asarray([float(d[metric_key]) for d in data], dtype=float)
        xs_all.append(xs)
        ys_all.append(ys)
        ax.plot(xs, ys, color=color, alpha=0.20, linewidth=0.9)
    if ys_all:
        min_len = min(len(y) for y in ys_all)
        x_mean = xs_all[0][:min_len]
        y_stack = np.vstack([y[:min_len] for y in ys_all])
        mean_y = y_stack.mean(axis=0)
        std_y = y_stack.std(axis=0)
        ax.plot(x_mean, mean_y, color=color, alpha=0.85, linewidth=1.25, label=label or "mean")
        ax.fill_between(x_mean, mean_y - std_y, mean_y + std_y, color=color, alpha=0.08, linewidth=0)


def draw_rolling_npy_metric(
    ax: plt.Axes,
    dirs: list[Path],
    filename: str,
    *,
    color: str,
    label: str | None = None,
    window: int = 100,
) -> None:
    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    for run_dir in dirs:
        path = run_dir / "logs" / filename
        if not path.exists():
            continue
        xs, ys = rolling_mean(np.load(path), window=window)
        xs_all.append(xs)
        ys_all.append(ys)
        ax.plot(xs, ys, color=color, alpha=0.18, linewidth=0.8)
    if ys_all:
        min_len = min(len(y) for y in ys_all)
        x_mean = xs_all[0][:min_len]
        y_stack = np.vstack([y[:min_len] for y in ys_all])
        mean_y = y_stack.mean(axis=0)
        std_y = y_stack.std(axis=0)
        ax.plot(x_mean, mean_y, color=color, alpha=0.85, linewidth=1.2, label=label or "mean")
        ax.fill_between(x_mean, mean_y - std_y, mean_y + std_y, color=color, alpha=0.08, linewidth=0)


def draw_rolling_npy_metric_mean_only(
    ax: plt.Axes,
    dirs: list[Path],
    filename: str,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    window: int = 100,
) -> None:
    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    for run_dir in dirs:
        path = run_dir / "logs" / filename
        if not path.exists():
            continue
        xs, ys = rolling_mean(np.load(path), window=window)
        xs_all.append(xs)
        ys_all.append(ys)
    if ys_all:
        min_len = min(len(y) for y in ys_all)
        x_mean = xs_all[0][:min_len]
        y_stack = np.vstack([y[:min_len] for y in ys_all])
        mean_y = y_stack.mean(axis=0)
        ax.plot(x_mean, mean_y, color=color, alpha=0.70, linewidth=1.2, linestyle=linestyle, label=label)


def draw_method_diagnostic_panels(
    *,
    prefix: str,
    representative_run: Path,
    pattern: str,
    color: str,
    condition: str | None = None,
) -> None:
    dirs = completed_method_run_dirs(condition) if condition is not None else run_dirs(pattern)
    metric_paths = [d / "logs" / "train_window_metrics_ep0.json" for d in dirs]
    metric_paths = [p for p in metric_paths if p.exists()]

    make_panel(PREPRINT_PANELS[f"{prefix}_rf"], lambda ax: draw_receptive_fields_from_run(ax, representative_run))
    make_panel(PREPRINT_PANELS[f"{prefix}_confusion"], lambda ax: draw_confusion_from_run(ax, representative_run, color=color))
    make_panel(PREPRINT_PANELS[f"{prefix}_entropy"], lambda ax: draw_window_metric(ax, metric_paths, "usage_entropy_norm", color=color))
    make_panel(PREPRINT_PANELS[f"{prefix}_spikes"], lambda ax: draw_rolling_npy_metric(ax, dirs, "train_sum_spk_ep0.npy", color=color, window=100))
    make_panel(PREPRINT_PANELS[f"{prefix}_gi"], lambda ax: draw_rolling_npy_metric(ax, dirs, "train_mean_gi_ep0.npy", color=color, window=100))


def figure1_learning_overview() -> None:
    cfg = PREPRINT_PANELS["representative_receptive_fields"]

    def draw_receptive_fields(ax: plt.Axes) -> None:
        draw_receptive_fields_from_run(ax, REPRESENTATIVE_FIXED_RUN)

    make_panel(cfg, draw_receptive_fields)

    cfg = PREPRINT_PANELS["representative_confusion"]

    def draw_confusion(ax: plt.Axes) -> None:
        draw_confusion_from_run(ax, REPRESENTATIVE_FIXED_RUN, color=GRAY)

    make_panel(cfg, draw_confusion)

    cfg = PREPRINT_PANELS["fixed_entropy_dynamics"]

    def draw_entropy(ax: plt.Axes) -> None:
        paths = sorted(COMPARE_MATCHED_DIR.glob("fixed_input2_wAiAe10_thetaPlus0p05_seed*/logs/train_window_metrics_ep0.json"))
        draw_window_metric(ax, paths, "usage_entropy_norm", color=METHOD_COLORS["fixed"], label="fixed inhibition")

    make_panel(cfg, draw_entropy)

    fixed_dirs = run_dirs(METHOD_RUN_PATTERNS["fixed"])

    cfg = PREPRINT_PANELS["fixed_spike_dynamics"]
    make_panel(cfg, lambda ax: draw_rolling_npy_metric(ax, fixed_dirs, "train_sum_spk_ep0.npy", color=METHOD_COLORS["fixed"], label="mean", window=100))

    cfg = PREPRINT_PANELS["fixed_gi_dynamics"]
    make_panel(cfg, lambda ax: draw_rolling_npy_metric(ax, fixed_dirs, "train_mean_gi_ep0.npy", color=METHOD_COLORS["fixed"], label="mean", window=100))


def figure2_baseline_grid() -> None:
    rows = fixed_grid_rows()
    y, x, stability = heatmap_table(rows, "input_intensity", "w_aiae", lambda rs: mean([0.0 if r["aborted_bool"] else 1.0 for r in rs]) or 0.0)
    _, _, accuracy = heatmap_table(rows, "input_intensity", "w_aiae", lambda rs: mean([r["accuracy"] for r in rs if not r["aborted_bool"]]) or np.nan)
    _, _, spikes = heatmap_table(rows, "input_intensity", "w_aiae", lambda rs: mean([r["train_mean_spikes"] for r in rs]) or np.nan)

    cfg = PREPRINT_PANELS["fixed_grid_stability"]
    make_panel(cfg, lambda ax: draw_heatmap(ax, y, x, stability, vmin=0, vmax=1, cmap=cfg.cmap or CMAP_STABILITY, cbar_label="stable fraction"))

    cfg = PREPRINT_PANELS["fixed_grid_accuracy"]
    make_panel(cfg, lambda ax: draw_heatmap(ax, y, x, accuracy, vmin=0.45, vmax=0.88, cmap=cfg.cmap or CMAP_ACCURACY, cbar_label="accuracy", colorbar_zero=cfg.colorbar_zero))

    cfg = PREPRINT_PANELS["fixed_grid_activity"]
    make_panel(cfg, lambda ax: draw_heatmap(ax, y, x, spikes, vmin=0, vmax=300, cmap=cfg.cmap or CMAP_ACTIVITY, cbar_label="spikes / image"))

    best = [r for r in rows if r["input_intensity"] == 2.0 and r["w_aiae"] == 10.0]
    cfg = PREPRINT_PANELS["fixed_selected_accuracy"]

    def draw_best(ax: plt.Axes) -> None:
        xs = np.arange(len(best))
        vals = [float(r["accuracy"]) for r in best]
        ax.plot(
            xs,
            vals,
            linestyle="None",
            marker="o",
            markersize=3.2,
            markerfacecolor=METHOD_COLORS["fixed"],
            markeredgecolor=METHOD_COLORS["fixed"],
            zorder=3,
        )
        ax.axhline(np.mean(vals), color=METHOD_COLORS["fixed"], linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(r["seed"])) for r in best])
        ax.set_xlabel("replicate", fontsize=7)

    make_panel(cfg, draw_best)


def figure3_budget_comparison() -> None:
    bad = compare_rows(budget="mismatched")
    matched = compare_rows(budget="matched")
    for r in bad:
        r["budget_label"] = "mismatched"
    for r in matched:
        r["budget_label"] = "matched"
    rows = bad + matched
    conditions = [
        "fixed",
        "vogels_stable",
        "vogels_unstable",
        "slow_homeostat_unconstrained",
        "normalized_slow_homeostat",
    ]
    labels = [METHOD_LABELS[c] for c in conditions]
    colors = METHOD_COLORS

    cfg = PREPRINT_PANELS["budget_rule_stability"]

    def draw_stability(ax: plt.Axes) -> None:
        plot_conditions = ["fixed", "vogels_stable", "vogels_unstable", "slow_homeostat_unconstrained", "normalized_slow_homeostat", "normalized_slow_homeostat"]
        plot_labels = ["fixed", "Vogels\nlow $\\rho$", "Vogels\nhigh $\\rho$", "slow-\nhomeostatic", "budget-\nmatched", "budget-\nmismatched"]
        plot_budgets = ["matched", "matched", "matched", "matched", "matched", "mismatched"]
        vals = []
        facecolors = []
        edgecolors = []
        for c, budget in zip(plot_conditions, plot_budgets):
            source = matched if budget == "matched" else bad
            sub = [r for r in source if r["condition"] == c]
            vals.append(sum(1 for r in sub if not r["aborted_bool"]))
            facecolors.append(colors[c] if budget == "matched" else "white")
            edgecolors.append(colors[c] if budget == "matched" else colors[c])
        ax.bar(np.arange(len(vals)), vals, color=facecolors, edgecolor=edgecolors, linewidth=1.1, zorder=3)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(panel_xtick_labels(cfg, plot_labels), fontsize=6)

    make_panel(cfg, draw_stability)

    cfg = PREPRINT_PANELS["budget_final_weight"]

    def draw_weight(ax: plt.Axes) -> None:
        sub = [r for r in rows if r["condition"] in {"fixed", "normalized_slow_homeostat"}]
        xs = {"fixed": 0, "normalized_slow_homeostat": 1}
        offsets = {"mismatched": -0.08, "matched": 0.08}
        for r in sub:
            ax.scatter(xs[r["condition"]] + offsets[r["budget_label"]], r["ai_ae_mean"], color=colors[r["condition"]], alpha=0.85, s=16)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(panel_xtick_labels(cfg, ["fixed", "budget-\nconstrained"]), fontsize=6)
        ax.axhline(10, color=GRAY, linestyle=":", linewidth=1)
        ax.axhline(20, color=GRAY, linestyle=":", linewidth=1)
        ax.text(1.15, 10, r"$B_I=3990$", fontsize=6, va="center")
        ax.text(1.15, 20, r"$B_I=7980$", fontsize=6, va="center")

    make_panel(cfg, draw_weight)

    cfg = PREPRINT_PANELS["budget_matched_accuracy"]

    def draw_accuracy(ax: plt.Axes) -> None:
        for xi, c in enumerate(conditions):
            vals = [r["accuracy"] for r in matched if r["condition"] == c and not r["aborted_bool"] and r["accuracy"] is not None]
            ax.scatter(np.repeat(xi, len(vals)), vals, color=colors[c], s=5, zorder=3)
            if vals:
                ax.errorbar(xi, mean(vals), yerr=sem(vals), fmt="_", color="black", capsize=3, markersize=10)
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_xticklabels(panel_xtick_labels(cfg, labels), fontsize=6)

    make_panel(cfg, draw_accuracy)

    cfg = PREPRINT_PANELS["budget_matched_activity"]

    def draw_activity(ax: plt.Axes) -> None:
        for xi, c in enumerate(conditions):
            vals = [r["train_mean_spikes"] for r in matched if r["condition"] == c and not r["aborted_bool"] and r["train_mean_spikes"] is not None]
            ax.scatter(np.repeat(xi, len(vals)), vals, color=colors[c], s=5, zorder=3)
            if vals:
                ax.errorbar(xi, mean(vals), yerr=sem(vals), fmt="_", color="black", capsize=3, markersize=10)
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_xticklabels(panel_xtick_labels(cfg, labels), fontsize=6)

    make_panel(cfg, draw_activity)

    cfg = PREPRINT_PANELS["budget_matched_gi_dynamics"]

    def draw_gi_dynamics(ax: plt.Axes) -> None:
        for condition in conditions:
            dirs = completed_method_run_dirs(condition)
            draw_rolling_npy_metric_mean_only(
                ax,
                dirs,
                "train_mean_gi_ep0.npy",
                color=METHOD_COLORS[condition],
                label=METHOD_LEGEND_LABELS[condition],
                linestyle=METHOD_LINESTYLES[condition],
                window=100,
            )

    make_panel(cfg, draw_gi_dynamics)

    cfg = PREPRINT_PANELS["budget_matched_final_weight_all"]

    def draw_all_weights(ax: plt.Axes) -> None:
        method_conditions = conditions
        for xi, condition in enumerate(method_conditions):
            vals = [
                r["ai_ae_mean"]
                for r in matched
                if r["condition"] == condition and r["ai_ae_mean"] is not None
            ]
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.asarray([0.0])
            ax.scatter(np.repeat(xi, len(vals)) + jitter, vals, color=METHOD_COLORS[condition], s=5, zorder=3)
            if vals:
                ax.errorbar(xi, mean(vals), yerr=sem(vals), fmt="_", color="black", capsize=3, markersize=10)
        ax.axhline(10, color=GRAY, linestyle=":", linewidth=1)
        ax.set_xticks(np.arange(len(method_conditions)))
        ax.set_xticklabels(panel_xtick_labels(cfg, [METHOD_LABELS[c] for c in method_conditions]), fontsize=6)

    make_panel(cfg, draw_all_weights)


def figure4_mechanistic_diagnostics() -> None:
    stable_fixed = REPRESENTATIVE_FIXED_RUN
    stable_budget = REPRESENTATIVE_NORMALIZED_RUN
    failed_vogels = FAILED_VOGELS_LOW_RUN
    failed_budget = FAILED_MISMATCHED_BUDGET_RUN

    trajectory_runs = [
        (stable_fixed, "fixed stable", GRAY, "-"),
        (REPRESENTATIVE_VOGELS_HIGH_RHO_RUN, r"Vogels high $\rho$ stable", LIGHT_BLUE, "-"),
        (REPRESENTATIVE_SLOW_RUN, "slow stable", ORANGE, "-"),
        (stable_budget, "budget matched stable", GREEN, "-"),
        (failed_vogels, r"Vogels low $\rho$ runaway", BLUE, "--"),
        (failed_budget, "mismatched-budget runaway", RED, "--"),
    ]

    cfg = PREPRINT_PANELS["mechanism_failure_spikes"]

    def draw_spike_failures(ax: plt.Axes) -> None:
        for run_dir, label, color, linestyle in trajectory_runs:
            path = run_dir / "logs" / "train_sum_spk_ep0.npy"
            xs, ys = rolling_mean(np.load(path), window=100)
            ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.1, label=label)
        ax.set_yscale("log")

    make_panel(cfg, draw_spike_failures)

    cfg = PREPRINT_PANELS["mechanism_failure_gi"]

    def draw_gi_failures(ax: plt.Axes) -> None:
        for run_dir, label, color, linestyle in trajectory_runs:
            path = run_dir / "logs" / "train_mean_gi_ep0.npy"
            xs, ys = rolling_mean(np.load(path), window=100)
            ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.1, label=label)
        ax.set_yscale("log")

    make_panel(cfg, draw_gi_failures)

    weight_runs = [
        (REPRESENTATIVE_FIXED_RUN, "fixed", GRAY, "-"),
        (REPRESENTATIVE_VOGELS_RUN, r"Vogels low $\rho$", BLUE, "-"),
        (REPRESENTATIVE_VOGELS_HIGH_RHO_RUN, r"Vogels high $\rho$", LIGHT_BLUE, "-"),
        (REPRESENTATIVE_SLOW_RUN, "slow-homeostatic", ORANGE, "-"),
        (REPRESENTATIVE_NORMALIZED_RUN, "budget matched", GREEN, "-"),
        (FAILED_MISMATCHED_BUDGET_RUN, "budget mismatched", RED, "--"),
    ]

    cfg = PREPRINT_PANELS["mechanism_weight_distribution"]

    def draw_weight_distribution(ax: plt.Axes) -> None:
        stable_weight_runs = weight_runs[:-1]
        y_positions = np.arange(len(stable_weight_runs))[::-1]

        for y0, (run_dir, label, color, linestyle) in zip(y_positions, stable_weight_runs):
            vals = inhibitory_weights_offdiag(run_dir)
            q1, q5, median, q95, q99 = np.percentile(vals, [1, 5, 50, 95, 99])
            ax.plot([q1, q99], [y0, y0], color=color, linestyle=linestyle, linewidth=1.1, alpha=0.55)
            ax.plot([q5, q95], [y0, y0], color=color, linestyle=linestyle, linewidth=2.0, alpha=0.95)
            ax.scatter([median], [y0], color=color, s=8, zorder=4)
            sigma_text = "" if label == "fixed" else rf"  $\sigma={vals.std():.3f}$"
            ax.text(9.72, y0 + 0.30, rf"{label}{sigma_text}", ha="left", va="center", fontsize=5.6, color=color)

        ax.axvline(10.0, color=GRAY, linestyle=":", linewidth=0.9, alpha=0.8)
        ax.text(10.02, len(stable_weight_runs) - 0.65, r"reference $\bar w^{IE}=10$", ha="left", va="top", fontsize=5.2, color=GRAY)
        ax.text(10.75, -0.62, "dot: median; thick: 5$-$95%; thin: 1$-$99%", ha="center", va="center", fontsize=5.2, color=GRAY)
        ax.set_xlim(9.65, 11.95)
        ax.set_yticks([])
        ax.set_ylim(-0.95, len(stable_weight_runs) - 0.05)

        mismatch_run, mismatch_label, mismatch_color, mismatch_linestyle = weight_runs[-1]
        mismatch_vals = inhibitory_weights_offdiag(mismatch_run)
        q1, q5, median, q95, q99 = np.percentile(mismatch_vals, [1, 5, 50, 95, 99])
        inset = ax.inset_axes([0.63, 0.58, 0.30, 0.24])
        inset.plot([q1, q99], [0, 0], color=mismatch_color, linestyle=mismatch_linestyle, linewidth=1.1, alpha=0.55)
        inset.plot([q5, q95], [0, 0], color=mismatch_color, linestyle=mismatch_linestyle, linewidth=2.0, alpha=0.95)
        inset.scatter([median], [0], color=mismatch_color, s=8, zorder=4)
        inset.axvline(20.0, color=mismatch_color, linestyle=":", linewidth=0.9, alpha=0.8)
        inset.set_xlim(19.94, 20.06)
        inset.set_ylim(-0.6, 0.6)
        inset.set_yticks([])
        inset.set_title(rf"mismatched $B_I$  $\sigma={mismatch_vals.std():.3f}$", fontsize=5.2, color=mismatch_color, pad=1.5)
        inset.tick_params(axis="x", labelsize=5, length=2, pad=1)
        for spine in ["top", "right", "left"]:
            inset.spines[spine].set_visible(False)

    make_panel(cfg, draw_weight_distribution)

    cfg = PREPRINT_PANELS["mechanism_column_sums"]

    def draw_column_sums(ax: plt.Axes) -> None:
        for xi, (run_dir, label, color, linestyle) in enumerate(weight_runs):
            vals = inhibitory_column_sums(run_dir)
            rng = np.random.default_rng(100 + xi)
            sample = vals if vals.size <= 120 else rng.choice(vals, size=120, replace=False)
            jitter = rng.uniform(-0.11, 0.11, size=sample.size)
            ax.scatter(np.full(sample.size, xi) + jitter, sample, color=color, alpha=0.22, s=7, linewidths=0)
            ax.errorbar(xi, vals.mean(), yerr=vals.std(), fmt="_", color="black", capsize=3, markersize=10)
        ax.axhline(3990, color=GREEN, linestyle=":", linewidth=1.0)
        ax.axhline(7980, color=RED, linestyle=":", linewidth=1.0)
        ax.text(5.35, 3990, r"$B_I=3990$", fontsize=5.5, color=GREEN, va="bottom", ha="right")
        ax.text(5.35, 7980, r"$B_I=7980$", fontsize=5.5, color=RED, va="bottom", ha="right")
        ax.set_xticks(np.arange(len(weight_runs)))
        ax.set_xticklabels(["fixed", "Vogels\nlow", "Vogels\nhigh", "slow-\nhomeostatic", "budget\nmatched", "budget\nmismatched"], fontsize=6)

    make_panel(cfg, draw_column_sums)


def figure_s1_budget_evidence() -> None:
    fixed_rows = [r for r in fixed_grid_rows() if r["input_intensity"] == 2.0]
    bad = [r for r in compare_rows(budget="mismatched") if r["condition"] == "normalized_slow_homeostat"]
    matched = [r for r in compare_rows(budget="matched") if r["condition"] == "normalized_slow_homeostat"]
    budget_scale = 399.0
    weights = sorted({float(r["w_aiae"]) for r in fixed_rows})
    budgets = [budget_scale * w for w in weights]

    cfg = PREPRINT_PANELS["supp_budget_fixed_stability"]

    def draw_fixed_stability(ax: plt.Axes) -> None:
        vals = []
        for w in weights:
            sub = [r for r in fixed_rows if r["w_aiae"] == w]
            vals.append(sum(1 for r in sub if not r["aborted_bool"]))
        colors = [GREEN if w == 10.0 else ORANGE if w == 15.0 else RED for w in weights]
        ax.bar(np.arange(len(weights)), vals, color=colors, alpha=0.85)
        ax.set_xticks(np.arange(len(weights)))
        ax.set_xticklabels([f"{int(b)}" for b in budgets], fontsize=6)
        for xi, w in enumerate(weights):
            ax.text(xi, -0.13, rf"$\bar w^{{IE}}={w:g}$", transform=ax.get_xaxis_transform(), 
                    ha="center", va="top", fontsize=5.5)

    make_panel(cfg, draw_fixed_stability)

    cfg = PREPRINT_PANELS["supp_budget_fixed_activity"]

    def draw_fixed_activity(ax: plt.Axes) -> None:
        for xi, w in enumerate(weights):
            sub = [r for r in fixed_rows if r["w_aiae"] == w]
            vals = [r["train_mean_spikes"] for r in sub if r["train_mean_spikes"] is not None]
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.repeat(xi, len(vals)) + jitter, vals, color=BLUE, s=16, zorder=3)
            ax.errorbar(xi, mean(vals), yerr=sem(vals), fmt="_", color="black", capsize=3, markersize=10)
        ax.set_xticks(np.arange(len(weights)))
        ax.set_xticklabels([f"{int(b)}" for b in budgets], fontsize=6)
        ax.axvspan(-0.4, 0.4, color=GREEN, alpha=0.10, linewidth=0)

    make_panel(cfg, draw_fixed_activity)

    cfg = PREPRINT_PANELS["supp_budget_normalized_stability"]

    def draw_normalized_stability(ax: plt.Axes) -> None:
        vals = [
            sum(1 for r in matched if not r["aborted_bool"]),
            sum(1 for r in bad if not r["aborted_bool"]),
        ]
        ax.bar([0, 1], vals, color=[GREEN, RED], alpha=0.85)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"matched" + "\n" + r"$B_I=3990$", r"mismatched" + "\n" + r"$B_I=7980$"], fontsize=6)

    make_panel(cfg, draw_normalized_stability)

    cfg = PREPRINT_PANELS["supp_budget_normalized_weight"]

    def draw_normalized_weight(ax: plt.Axes) -> None:
        groups = [matched, bad]
        for xi, rows in enumerate(groups):
            vals = [r["ai_ae_mean"] for r in rows if r["ai_ae_mean"] is not None]
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.repeat(xi, len(vals)) + jitter, vals, color=[GREEN, RED][xi], s=16, zorder=3)
            ax.errorbar(xi, mean(vals), yerr=sem(vals), fmt="_", color="black", capsize=3, markersize=10)
        ax.axhline(10, color=GRAY, linestyle=":", linewidth=1)
        ax.axhline(20, color=GRAY, linestyle=":", linewidth=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"$B_I=3990$", r"$B_I=7980$"], fontsize=6)

    make_panel(cfg, draw_normalized_weight)


def figure_supplement_method_diagnostics() -> None:
    draw_method_diagnostic_panels(
        prefix="supp_vogels",
        representative_run=REPRESENTATIVE_VOGELS_RUN,
        pattern=METHOD_RUN_PATTERNS["vogels_stable"],
        color=BLUE,
        condition="vogels_stable",
    )
    draw_method_diagnostic_panels(
        prefix="supp_vogels_high",
        representative_run=REPRESENTATIVE_VOGELS_HIGH_RHO_RUN,
        pattern=METHOD_RUN_PATTERNS["vogels_unstable"],
        color=LIGHT_BLUE,
        condition="vogels_unstable",
    )
    draw_method_diagnostic_panels(
        prefix="supp_slow",
        representative_run=REPRESENTATIVE_SLOW_RUN,
        pattern=METHOD_RUN_PATTERNS["slow_homeostat_unconstrained"],
        color=ORANGE,
        condition="slow_homeostat_unconstrained",
    )
    draw_method_diagnostic_panels(
        prefix="supp_normalized",
        representative_run=REPRESENTATIVE_NORMALIZED_RUN,
        pattern=METHOD_RUN_PATTERNS["normalized_slow_homeostat"],
        color=GREEN,
        condition="normalized_slow_homeostat",
    )


def draw_two_run_trajectory(
    ax: plt.Axes,
    *,
    bad_run: Path,
    good_run: Path,
    filename: str,
    bad_label: str,
    good_label: str,
    color: str,
    good_color: str | None = None,
    window: int = 100,
    log_y: bool = True,
) -> None:
    good_color = good_color or color
    for run_dir, label, linestyle, line_color in [
        (bad_run, bad_label, "--", color),
        (good_run, good_label, "-", good_color),
    ]:
        path = run_dir / "logs" / filename
        xs, ys = rolling_mean(np.load(path), window=window)
        ax.plot(xs, ys, color=line_color, linestyle=linestyle, linewidth=1.1, label=label)
    if log_y:
        ax.set_yscale("log")


def read_run_summary(run_dir: Path) -> dict[str, object]:
    path = run_dir / "logs" / "run_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def format_summary_value(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value_f):
        return "n/a"
    if digits <= 0:
        return f"{int(round(value_f))}"
    return f"{value_f:.{digits}g}"


def draw_two_run_window_metric(
    ax: plt.Axes,
    *,
    bad_run: Path,
    good_run: Path,
    metric_key: str,
    bad_label: str,
    good_label: str,
    bad_color: str,
    good_color: str,
    log_y: bool = False,
) -> None:
    for run_dir, label, linestyle, color in [
        (bad_run, bad_label, "--", bad_color),
        (good_run, good_label, "-", good_color),
    ]:
        path = run_dir / "logs" / "train_window_metrics_ep0.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        xs = np.asarray([float(d["example_end"]) for d in data if metric_key in d], dtype=float)
        ys = np.asarray([float(d[metric_key]) for d in data if metric_key in d], dtype=float)
        if xs.size == 0 or ys.size == 0:
            continue
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.1, label=label)
    if log_y:
        ax.set_yscale("log")


def draw_multi_run_window_metric(
    ax: plt.Axes,
    *,
    runs: list[tuple[Path, str, str, str]],
    metric_key: str,
    log_y: bool = False,
) -> None:
    for run_dir, label, linestyle, color in runs:
        path = run_dir / "logs" / "train_window_metrics_ep0.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        xs = np.asarray([float(d["example_end"]) for d in data if metric_key in d], dtype=float)
        ys = np.asarray([float(d[metric_key]) for d in data if metric_key in d], dtype=float)
        if xs.size == 0 or ys.size == 0:
            continue
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.1, label=label)
    if log_y:
        ax.set_yscale("log")


def draw_multi_run_trajectory(
    ax: plt.Axes,
    *,
    runs: list[tuple[Path, str, str, str]],
    filename: str,
    window: int = 100,
    log_y: bool = True,
) -> None:
    for run_dir, label, linestyle, color in runs:
        path = run_dir / "logs" / filename
        if not path.exists():
            continue
        xs, ys = rolling_mean(np.load(path), window=window)
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.1, label=label)
    if log_y:
        ax.set_yscale("log")


def draw_failure_summary(
    ax: plt.Axes,
    *,
    bad_run: Path,
    good_run: Path,
    bad_label: str,
    good_label: str,
    note: str,
) -> None:
    bad = read_run_summary(bad_run)
    good = read_run_summary(good_run)
    ax.set_axis_off()
    lines = [
        bad_label,
        f"status: aborted ({bad.get('abort_reason', 'n/a') or 'n/a'})",
        f"processed examples: {format_summary_value(bad.get('processed_train_examples'), 0)}",
        f"mean spikes/image: {format_summary_value(bad.get('train_mean_spikes'), 3)}",
        f"mean inhibition: {format_summary_value(bad.get('gi_mean'), 3)}",
        "",
        good_label,
        "status: completed",
        f"processed examples: {format_summary_value(good.get('processed_train_examples'), 0)}",
        f"test accuracy: {format_summary_value(100.0 * float(good.get('accuracy', float('nan'))), 3)}%",
        f"mean spikes/image: {format_summary_value(good.get('train_mean_spikes'), 3)}",
        f"mean inhibition: {format_summary_value(good.get('gi_mean'), 3)}",
        "",
        note,
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#222222",
        linespacing=1.25,
    )


def draw_method_failure_panels(
    *,
    prefix: str,
    bad_run: Path,
    good_run: Path,
    bad_label: str,
    good_label: str,
    bad_color: str,
    good_color: str,
) -> None:
    make_panel(PREPRINT_PANELS[f"{prefix}_bad_rf"], lambda ax: draw_receptive_fields_from_run(ax, bad_run))
    make_panel(PREPRINT_PANELS[f"{prefix}_good_rf"], lambda ax: draw_receptive_fields_from_run(ax, good_run))
    make_panel(PREPRINT_PANELS[f"{prefix}_confusion"], lambda ax: draw_confusion_from_run(ax, good_run, color=good_color))
    make_panel(
        PREPRINT_PANELS[f"{prefix}_entropy"],
        lambda ax: draw_two_run_window_metric(
            ax,
            bad_run=bad_run,
            good_run=good_run,
            metric_key="usage_entropy_norm",
            bad_label=bad_label,
            good_label=good_label,
            bad_color=bad_color,
            good_color=good_color,
        ),
    )
    make_panel(
        PREPRINT_PANELS[f"{prefix}_window_spikes"],
        lambda ax: draw_two_run_window_metric(
            ax,
            bad_run=bad_run,
            good_run=good_run,
            metric_key="mean_e_spikes_per_image",
            bad_label=bad_label,
            good_label=good_label,
            bad_color=bad_color,
            good_color=good_color,
            log_y=True,
        ),
    )
    make_panel(
        PREPRINT_PANELS[f"{prefix}_window_gi"],
        lambda ax: draw_two_run_window_metric(
            ax,
            bad_run=bad_run,
            good_run=good_run,
            metric_key="mean_inhibitory_conductance",
            bad_label=bad_label,
            good_label=good_label,
            bad_color=bad_color,
            good_color=good_color,
            log_y=True,
        ),
    )
    make_panel(
        PREPRINT_PANELS[f"{prefix}_spikes"],
        lambda ax: draw_two_run_trajectory(
            ax,
            bad_run=bad_run,
            good_run=good_run,
            filename="train_sum_spk_ep0.npy",
            bad_label=bad_label,
            good_label=good_label,
            color=bad_color,
            good_color=good_color,
        ),
    )
    make_panel(
        PREPRINT_PANELS[f"{prefix}_gi"],
        lambda ax: draw_two_run_trajectory(
            ax,
            bad_run=bad_run,
            good_run=good_run,
            filename="train_mean_gi_ep0.npy",
            bad_label=bad_label,
            good_label=good_label,
            color=bad_color,
            good_color=good_color,
        ),
    )


def draw_vogels_failure_panels() -> None:
    low_bad = FAILED_VOGELS_LOW_RUN
    low_good = REPRESENTATIVE_VOGELS_RUN
    high_good = REPRESENTATIVE_VOGELS_HIGH_RHO_RUN
    trajectory_runs = [
        (low_bad, r"low $\rho$ failed", "--", BLUE),
        (low_good, r"low $\rho$ compl.", "-", BLUE),
        (high_good, r"high $\rho$ compl.", "-.", LIGHT_BLUE),
    ]

    make_panel(PREPRINT_PANELS["supp_vogels_failure_low_bad_rf"], lambda ax: draw_receptive_fields_from_run(ax, low_bad))
    make_panel(PREPRINT_PANELS["supp_vogels_failure_low_good_rf"], lambda ax: draw_receptive_fields_from_run(ax, low_good))
    make_panel(PREPRINT_PANELS["supp_vogels_failure_low_confusion"], lambda ax: draw_confusion_from_run(ax, low_good, color=BLUE))
    make_panel(PREPRINT_PANELS["supp_vogels_failure_high_good_rf"], lambda ax: draw_receptive_fields_from_run(ax, high_good))
    make_panel(PREPRINT_PANELS["supp_vogels_failure_high_confusion"], lambda ax: draw_confusion_from_run(ax, high_good, color=LIGHT_BLUE))
    make_panel(
        PREPRINT_PANELS["supp_vogels_failure_entropy"],
        lambda ax: draw_multi_run_window_metric(ax, runs=trajectory_runs, metric_key="usage_entropy_norm"),
    )
    make_panel(
        PREPRINT_PANELS["supp_vogels_failure_window_spikes"],
        lambda ax: draw_multi_run_window_metric(ax, runs=trajectory_runs, metric_key="mean_e_spikes_per_image", log_y=True),
    )
    make_panel(
        PREPRINT_PANELS["supp_vogels_failure_window_gi"],
        lambda ax: draw_multi_run_window_metric(ax, runs=trajectory_runs, metric_key="mean_inhibitory_conductance", log_y=True),
    )
    make_panel(
        PREPRINT_PANELS["supp_vogels_failure_spikes"],
        lambda ax: draw_multi_run_trajectory(ax, runs=trajectory_runs, filename="train_sum_spk_ep0.npy"),
    )
    make_panel(
        PREPRINT_PANELS["supp_vogels_failure_gi"],
        lambda ax: draw_multi_run_trajectory(ax, runs=trajectory_runs, filename="train_mean_gi_ep0.npy"),
    )


def figure_supplement_failure_diagnostics() -> None:
    draw_vogels_failure_panels()
    draw_method_failure_panels(
        prefix="supp_slow_failure",
        bad_run=FAILED_SLOW_SATURATION_RUN,
        good_run=REPRESENTATIVE_SLOW_RUN,
        bad_label="saturation example",
        good_label="completed reference",
        bad_color=RED,
        good_color=ORANGE,
    )
    draw_method_failure_panels(
        prefix="supp_budget_failure",
        bad_run=FAILED_MISMATCHED_BUDGET_RUN,
        good_run=REPRESENTATIVE_NORMALIZED_RUN,
        bad_label=r"mismatched $B_I$",
        good_label=r"matched $B_I$",
        bad_color=RED,
        good_color=GREEN,
    )


def figure4_vogels_map() -> None:
    rows = vogels_map_rows()
    y, x, stability = heatmap_table(rows, "rho_ie", "eta_ie", lambda rs: mean([0.0 if r["aborted_bool"] else 1.0 for r in rs]) or 0.0)
    _, _, processed = heatmap_table(rows, "rho_ie", "eta_ie", lambda rs: mean([r["processed_train_examples"] for r in rs]) or np.nan)

    cfg = PREPRINT_PANELS["vogels_stability_map"]
    make_panel(cfg, lambda ax: draw_heatmap(ax, y, x, stability, vmin=0, vmax=1, cmap=cfg.cmap or CMAP_STABILITY, cbar_label="stable fraction"))

    cfg = PREPRINT_PANELS["vogels_processed_map"]
    make_panel(cfg, lambda ax: draw_heatmap(ax, y, x, processed, vmin=0, vmax=30000, cmap=cfg.cmap or CMAP_PROCESSED, cbar_label="training examples", value_formatter=format_examples_k))

    cfg = PREPRINT_PANELS["vogels_seed_counts"]

    def draw_seed(ax: plt.Axes) -> None:
        seed_values = unique_sorted(rows, "seed")
        counts = [sum(1 for r in rows if r["seed"] == seed and not r["aborted_bool"]) for seed in seed_values]
        ax.bar([str(int(s)) for s in seed_values], counts, color=BLUE)

    make_panel(cfg, draw_seed)

    cfg = PREPRINT_PANELS["vogels_outcomes"]

    def draw_outcomes(ax: plt.Axes) -> None:
        n_ok = sum(1 for r in rows if not r["aborted_bool"])
        n_abort = sum(1 for r in rows if r["aborted_bool"])
        ax.bar([0, 1], [n_ok, n_abort], color=[GREEN, RED])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["stable", "aborted"], fontsize=6)

    make_panel(cfg, draw_outcomes)


def figure_slow_budget_maps() -> None:
    slow_rows = slow_map_rows()
    budget_rows = budget_map_rows()

    if slow_rows:
        y, x, stability = heatmap_table(
            slow_rows,
            "slow_homeostat_target_rate_hz",
            "eta_ie",
            lambda rs: mean([0.0 if r["aborted_bool"] else 1.0 for r in rs]) or 0.0,
        )
        _, _, processed = heatmap_table(
            slow_rows,
            "slow_homeostat_target_rate_hz",
            "eta_ie",
            lambda rs: mean([r["processed_train_examples"] for r in rs]) or np.nan,
        )
        cfg = PREPRINT_PANELS["slow_stability_map"]
        make_panel(cfg, lambda ax: draw_heatmap(ax, y, x, stability, vmin=0, vmax=1, cmap=cfg.cmap or CMAP_STABILITY, cbar_label="stable fraction"))
        cfg = PREPRINT_PANELS["slow_processed_map"]
        make_panel(cfg, lambda ax: draw_heatmap(ax, y, x, processed, vmin=0, vmax=30000, cmap=cfg.cmap or CMAP_PROCESSED, cbar_label="training examples", value_formatter=format_examples_k))

    if budget_rows:
        y, x, stability = heatmap_table(
            budget_rows,
            "aiae_target_sum",
            "eta_ie",
            lambda rs: mean([0.0 if r["aborted_bool"] else 1.0 for r in rs]) or 0.0,
        )
        _, _, processed = heatmap_table(
            budget_rows,
            "aiae_target_sum",
            "eta_ie",
            lambda rs: mean([r["processed_train_examples"] for r in rs]) or np.nan,
        )
        cfg = PREPRINT_PANELS["budget_stability_map"]
        make_panel(cfg, lambda ax: draw_heatmap(ax, y, x, stability, vmin=0, vmax=1, cmap=cfg.cmap or CMAP_STABILITY, cbar_label="stable fraction"))
        cfg = PREPRINT_PANELS["budget_processed_map"]
        make_panel(cfg, lambda ax: draw_heatmap(ax, y, x, processed, vmin=0, vmax=30000, cmap=cfg.cmap or CMAP_PROCESSED, cbar_label="training examples", value_formatter=format_examples_k))


def figure_s5_training_budget_justification() -> None:
    run_dir = LONG_FIXED_RUN
    epoch_colors = {0: BLUE, 1: ORANGE}

    cfg = PREPRINT_PANELS["supp_training_spikes"]

    def draw_spikes(ax: plt.Axes) -> None:
        for ep in (0, 1):
            path = run_dir / "logs" / f"train_sum_spk_ep{ep}.npy"
            if not path.exists():
                continue
            xs, ys = rolling_mean(np.load(path), window=500)
            ax.plot(xs, ys, color=epoch_colors[ep], linewidth=1.2, label=f"epoch {ep + 1}")
        ax.axvline(30000, color=GRAY, linestyle=":", linewidth=1)

    make_panel(cfg, draw_spikes)

    cfg = PREPRINT_PANELS["supp_training_entropy"]

    def draw_entropy(ax: plt.Axes) -> None:
        for ep in (0, 1):
            path = run_dir / "logs" / f"train_window_metrics_ep{ep}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            xs = np.asarray([float(d["example_end"]) for d in data], dtype=float)
            ys = np.asarray([float(d["usage_entropy_norm"]) for d in data], dtype=float)
            ax.plot(xs, ys, color=epoch_colors[ep], linewidth=1.2, label=f"epoch {ep + 1}")
        ax.axvline(30000, color=GRAY, linestyle=":", linewidth=1)

    make_panel(cfg, draw_entropy)

    cfg = PREPRINT_PANELS["supp_training_accuracy"]

    def draw_accuracy(ax: plt.Axes) -> None:
        vals = []
        for ep in (0, 1):
            path = run_dir / "logs" / f"metrics_ep{ep}.json"
            vals.append(float(json.loads(path.read_text())["accuracy"]))
        ax.plot([1, 2], vals, color=BLUE, marker="o", linewidth=1.0, markersize=4)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["epoch 1", "epoch 2"], fontsize=6)
        ax.set_xlim(0.5,2.5)

    make_panel(cfg, draw_accuracy)

    cfg = PREPRINT_PANELS["supp_training_weight_change"]

    def draw_weight_change(ax: plt.Axes) -> None:
        w0 = np.load(run_dir / "snapshots" / "W_XeAe_ep0.npy").astype(float)
        w1 = np.load(run_dir / "snapshots" / "W_XeAe_ep1.npy").astype(float)
        corr = float(np.corrcoef(w0.ravel(), w1.ravel())[0, 1])
        rel_delta = float(np.linalg.norm(w1 - w0) / max(np.linalg.norm(w0), 1e-12))
        stats = [corr, rel_delta]
        ax.bar([0, 1], stats, color=[BLUE, ORANGE], alpha=0.85)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["corr.\n$W_1,W_2$", "relative\nchange"], fontsize=6)
        ax.set_ylim(0, 1.05)
        for xi, val in enumerate(stats):
            ax.text(xi, val + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=6)

    make_panel(cfg, draw_weight_change)

# %% MAIN
def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # figure 1:
    figure1_learning_overview()
    figure2_baseline_grid()
    
    # figure 2 Vogels style:
    figure4_vogels_map()
    figure_supplement_failure_diagnostics() # also fpr figure 4 and 6
    
    # figure 3:
    figure3_budget_comparison()
    
    # figure 4 and 6:
    figure_slow_budget_maps()
    
    # figure 5:
    figure4_mechanistic_diagnostics()
    
    # SI figure 2:
    figure_s1_budget_evidence()
    
    # SI figure 1:
    figure_s5_training_budget_justification()
    print(f"Figures written to {FIGURES_DIR}")
# %% MAIN ENTRY POINT
if __name__ == "__main__":
    main()
# %% END
