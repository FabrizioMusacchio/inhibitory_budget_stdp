"""Generate the primary 30k-example HPC manifest.

This script creates the CSV manifest used by the SLURM array workflow for the
main fixed-inhibition operating-regime grid, direct matched and mismatched
inhibitory-plasticity comparisons, and Vogels-style inhibitory STDP map.

author: Fabrizio Musacchio
date:   Jun 2026
"""
# %% IMPORTS
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any
# %% CONSTANTS
SEEDS = [0, 1, 2, 3, 4]
FIXED_INPUT_INTENSITIES = [0.5, 1.0, 1.5, 2.0]
FIXED_W_AIAE_VALUES = [10.0, 15.0, 20.0, 25.0]
W_AEAI = 10.4
THETA_PLUS_MV = 0.05

COMPARE_CONDITIONS: list[dict[str, Any]] = [
    {
        "condition": "fixed",
        "mode": "fixed",
        "istdp_rule": "fixed",
        "eta_ie": "",
        "rho_ie": "",
        "w_ie_max": "",
        "normalize_aiae_columns": "false",
    },
    {
        "condition": "vogels_stable",
        "mode": "istdp",
        "istdp_rule": "vogels",
        "eta_ie": 3e-4,
        "rho_ie": 0.01,
        "w_ie_max": 25.0,
        "normalize_aiae_columns": "false",
    },
    {
        "condition": "vogels_unstable",
        "mode": "istdp",
        "istdp_rule": "vogels",
        "eta_ie": 3e-4,
        "rho_ie": 0.03,
        "w_ie_max": 25.0,
        "normalize_aiae_columns": "false",
    },
    {
        "condition": "slow_homeostat_unconstrained",
        "mode": "istdp",
        "istdp_rule": "slow_homeostat",
        "eta_ie": 3e-4,
        "rho_ie": 0.03,
        "w_ie_max": 25.0,
        "normalize_aiae_columns": "false",
    },
    {
        "condition": "normalized_slow_homeostat",
        "mode": "istdp",
        "istdp_rule": "normalized_slow_homeostat",
        "eta_ie": 3e-4,
        "rho_ie": 0.03,
        "w_ie_max": 25.0,
        "normalize_aiae_columns": "true",
    },
]

VOGELS_ETA_VALUES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
VOGELS_RHO_VALUES = [0.01, 0.03, 0.1, 0.3]
VOGELS_WMAX_VALUES = [25.0]

FIELDNAMES = [
    "job_index",
    "block",
    "run_set",
    "condition",
    "run_name",
    "seed",
    "mode",
    "istdp_rule",
    "input_intensity",
    "w_aiae",
    "w_aeai",
    "theta_plus_mV",
    "eta_ie",
    "rho_ie",
    "w_ie_max",
    "normalize_aiae_columns",
    "aiae_target_sum",
    "aiae_normalize_every",
]
# %% FUNCTIONS
def slugify_float(value: float | str) -> str:
    if value == "":
        return "none"
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def add_fixed_grid(rows: list[dict[str, Any]]) -> None:
    for seed in SEEDS:
        for input_intensity in FIXED_INPUT_INTENSITIES:
            for w_aiae in FIXED_W_AIAE_VALUES:
                rows.append(
                    {
                        "block": "fixed_grid",
                        "run_set": "fixed_grid",
                        "condition": "fixed_grid",
                        "seed": seed,
                        "mode": "fixed",
                        "istdp_rule": "fixed",
                        "input_intensity": input_intensity,
                        "w_aiae": w_aiae,
                        "w_aeai": W_AEAI,
                        "theta_plus_mV": THETA_PLUS_MV,
                        "eta_ie": "",
                        "rho_ie": "",
                        "w_ie_max": "",
                        "normalize_aiae_columns": "false",
                        "aiae_target_sum": 7980.0,
                        "aiae_normalize_every": 1,
                        "run_name": (
                            f"fixed_input{slugify_float(input_intensity)}_"
                            f"wAiAe{slugify_float(w_aiae)}_"
                            f"wAeAi{slugify_float(W_AEAI)}_"
                            f"thetaPlus{slugify_float(THETA_PLUS_MV)}_seed{seed}"
                        ),
                    }
                )


def add_compare(rows: list[dict[str, Any]], *, block: str, aiae_target_sum: float) -> None:
    for seed in SEEDS:
        for condition in COMPARE_CONDITIONS:
            run_name = (
                f"{condition['condition']}_input2_"
                f"wAiAe10_thetaPlus{slugify_float(THETA_PLUS_MV)}_seed{seed}"
            )
            rows.append(
                {
                    "block": block,
                    "run_set": "compare_regime",
                    "condition": condition["condition"],
                    "seed": seed,
                    "mode": condition["mode"],
                    "istdp_rule": condition["istdp_rule"],
                    "input_intensity": 2.0,
                    "w_aiae": 10.0,
                    "w_aeai": W_AEAI,
                    "theta_plus_mV": THETA_PLUS_MV,
                    "eta_ie": condition["eta_ie"],
                    "rho_ie": condition["rho_ie"],
                    "w_ie_max": condition["w_ie_max"],
                    "normalize_aiae_columns": condition["normalize_aiae_columns"],
                    "aiae_target_sum": aiae_target_sum,
                    "aiae_normalize_every": 1,
                    "run_name": run_name,
                }
            )


def add_vogels_map(rows: list[dict[str, Any]]) -> None:
    for seed in SEEDS:
        for eta_ie in VOGELS_ETA_VALUES:
            for rho_ie in VOGELS_RHO_VALUES:
                for w_ie_max in VOGELS_WMAX_VALUES:
                    rows.append(
                        {
                            "block": "vogels_map",
                            "run_set": "vogels_stability_map",
                            "condition": "vogels_map",
                            "seed": seed,
                            "mode": "istdp",
                            "istdp_rule": "vogels",
                            "input_intensity": 2.0,
                            "w_aiae": 10.0,
                            "w_aeai": W_AEAI,
                            "theta_plus_mV": THETA_PLUS_MV,
                            "eta_ie": eta_ie,
                            "rho_ie": rho_ie,
                            "w_ie_max": w_ie_max,
                            "normalize_aiae_columns": "false",
                            "aiae_target_sum": 7980.0,
                            "aiae_normalize_every": 1,
                            "run_name": (
                                f"vogels_eta{slugify_float(eta_ie)}_"
                                f"rho{slugify_float(rho_ie)}_"
                                f"wmax{slugify_float(w_ie_max)}_seed{seed}"
                            ),
                        }
                    )
# %% MAIN FUNCTION
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one-run-per-row HPC manifest for Euler STDP sweeps.")
    parser.add_argument("--out", type=Path, default=Path("hpc_30k_manifest.csv"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    add_fixed_grid(rows)
    add_compare(rows, block="compare_mismatched", aiae_target_sum=7980.0)
    add_compare(rows, block="compare_matched", aiae_target_sum=3990.0)
    add_vogels_map(rows)

    for idx, row in enumerate(rows):
        row["job_index"] = idx

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} manifest rows to {args.out}")
    return 0
# %% MAIN ENTRY POINT
if __name__ == "__main__":
    raise SystemExit(main())
# %% END
