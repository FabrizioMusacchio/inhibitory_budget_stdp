"""Generate the slow-homeostatic and budget-constrained map manifest.

This script writes the CSV manifest for the dedicated HPC parameter maps of
slow homeostatic inhibitory plasticity and budget-constrained slow
homeostatic inhibition at the selected fixed-inhibition reference regime.

author: Fabrizio Musacchio
date:   Jun 2026
"""
# %% IMPORTS
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Union
# %% CONSTANTS
SEEDS = [0, 1, 2, 3, 4]
ETA_VALUES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
SLOW_TARGET_RATE_HZ_VALUES = [0.03, 0.1, 0.3, 1.0]
BUDGET_VALUES = [1995.0, 3990.0, 5985.0, 7980.0]

INPUT_INTENSITY = 2.0
W_AIAE = 10.0
W_AEAI = 10.4
THETA_PLUS_MV = 0.05
RHO_IE = 0.03
W_IE_MAX = 25.0

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
    "slow_homeostat_target_rate_hz",
]

# %% FUNCTIONS
def slugify_float(value: Union[float, str]) -> str:
    if value == "":
        return "none"
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def base_row(seed: int, eta_ie: float) -> Dict[str, Any]:
    return {
        "seed": seed,
        "mode": "istdp",
        "input_intensity": INPUT_INTENSITY,
        "w_aiae": W_AIAE,
        "w_aeai": W_AEAI,
        "theta_plus_mV": THETA_PLUS_MV,
        "eta_ie": eta_ie,
        "rho_ie": RHO_IE,
        "w_ie_max": W_IE_MAX,
        "aiae_normalize_every": 1,
    }


def add_slow_map(rows: List[Dict[str, Any]]) -> None:
    for seed in SEEDS:
        for eta_ie in ETA_VALUES:
            for target_rate_hz in SLOW_TARGET_RATE_HZ_VALUES:
                row = base_row(seed=seed, eta_ie=eta_ie)
                row.update(
                    {
                        "block": "slow_map",
                        "run_set": "slow_homeostat_map",
                        "condition": "slow_homeostat_map",
                        "istdp_rule": "slow_homeostat",
                        "normalize_aiae_columns": "false",
                        "aiae_target_sum": 3990.0,
                        "slow_homeostat_target_rate_hz": target_rate_hz,
                        "run_name": (
                            f"slow_eta{slugify_float(eta_ie)}_"
                            f"r0hz{slugify_float(target_rate_hz)}_seed{seed}"
                        ),
                    }
                )
                rows.append(row)


def add_budget_map(rows: List[Dict[str, Any]]) -> None:
    for seed in SEEDS:
        for eta_ie in ETA_VALUES:
            for budget in BUDGET_VALUES:
                row = base_row(seed=seed, eta_ie=eta_ie)
                row.update(
                    {
                        "block": "budget_map",
                        "run_set": "normalized_slow_homeostat_budget_map",
                        "condition": "normalized_slow_homeostat_budget_map",
                        "istdp_rule": "normalized_slow_homeostat",
                        "normalize_aiae_columns": "true",
                        "aiae_target_sum": budget,
                        "slow_homeostat_target_rate_hz": 0.1,
                        "run_name": (
                            f"budget_eta{slugify_float(eta_ie)}_"
                            f"BI{slugify_float(budget)}_seed{seed}"
                        ),
                    }
                )
                rows.append(row)
# %% MAIN FUNCTION
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Slow/Budget inhibitory-plasticity map manifest.")
    parser.add_argument("--out", type=Path, default=Path("hpc_slow_budget_maps_30k_manifest.csv"))
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    add_slow_map(rows)
    add_budget_map(rows)

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
