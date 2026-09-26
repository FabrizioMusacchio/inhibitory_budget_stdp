"""Collect completed HPC manifest runs into a combined summary table.

This helper reads a manifest CSV and the corresponding per-run
`logs/run_summary.json` files, extracts the simulation metadata and outcome
metrics needed for downstream analyses, and writes aggregate CSV summaries for
the full manifest-based sweep.

author: Fabrizio Musacchio
date:   Jun 2026
"""
# %% IMPORTS
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
# %% CONSTANTS
SUMMARY_COLUMNS = [
    "job_index",
    "block",
    "run_set",
    "condition",
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
    "aborted",
    "abort_reason",
    "accuracy",
    "train_mean_spikes",
    "train_usage_entropy_norm",
    "dead_fraction",
    "firing_rate_cv",
    "gi_mean",
    "ai_ae_mean",
    "ai_ae_max",
    "ai_ae_std",
    "theta_mean",
    "theta_max",
    "processed_train_examples",
    "retry_cap_hits",
    "runaway_detected",
    "collapse_detected",
    "aiae_max_fraction",
    "returncode",
    "skipped",
    "run_dir",
]
# %% FUNCTIONS
def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())

# %% MAIN FUNCTION
def main() -> int:
    parser = argparse.ArgumentParser(description="Collect one-row-per-run manifest summary.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-base-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or (args.out_base_dir / "combined_manifest_summary.csv")
    with args.manifest.open("r", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        run_dir = args.out_base_dir / row["block"] / row["run_set"] / row["run_name"]
        status = load_json(args.out_base_dir / "_manifest_status" / f"job_{int(row['job_index']):04d}.json")
        summary = load_json(run_dir / "logs" / "run_summary.json")
        merged = {
            "job_index": row["job_index"],
            "block": row["block"],
            "run_set": row["run_set"],
            "condition": row["condition"],
            "seed": row["seed"],
            "mode": row["mode"],
            "istdp_rule": row["istdp_rule"],
            "input_intensity": summary.get("input_intensity", row["input_intensity"]),
            "w_aiae": summary.get("w_aiae", row["w_aiae"]),
            "w_aeai": summary.get("w_aeai", row["w_aeai"]),
            "theta_plus_mV": summary.get("theta_plus_mV", row["theta_plus_mV"]),
            "eta_ie": row["eta_ie"],
            "rho_ie": row["rho_ie"],
            "w_ie_max": row["w_ie_max"],
            "normalize_aiae_columns": row["normalize_aiae_columns"],
            "aiae_target_sum": summary.get("aiae_target_sum", row["aiae_target_sum"]),
            "aiae_normalize_every": summary.get("aiae_normalize_every", row["aiae_normalize_every"]),
            "aborted": summary.get("aborted", ""),
            "abort_reason": summary.get("abort_reason", ""),
            "accuracy": summary.get("accuracy", ""),
            "train_mean_spikes": summary.get("train_mean_spikes", ""),
            "train_usage_entropy_norm": summary.get("train_usage_entropy_norm", ""),
            "dead_fraction": summary.get("dead_fraction", ""),
            "firing_rate_cv": summary.get("firing_rate_cv", ""),
            "gi_mean": summary.get("gi_mean", ""),
            "ai_ae_mean": summary.get("ai_ae_mean", ""),
            "ai_ae_max": summary.get("ai_ae_max", ""),
            "ai_ae_std": summary.get("ai_ae_std", ""),
            "theta_mean": summary.get("theta_mean", ""),
            "theta_max": summary.get("theta_max", ""),
            "processed_train_examples": summary.get("processed_train_examples", ""),
            "retry_cap_hits": summary.get("retry_cap_hits", ""),
            "runaway_detected": summary.get("runaway_detected", ""),
            "collapse_detected": summary.get("collapse_detected", ""),
            "aiae_max_fraction": summary.get("aiae_max_fraction", ""),
            "returncode": status.get("returncode", ""),
            "skipped": status.get("skipped", ""),
            "run_dir": str(run_dir),
        }
        rows.append(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0

# %% MAIN ENTRY POINT
if __name__ == "__main__":
    raise SystemExit(main())
# %% END
