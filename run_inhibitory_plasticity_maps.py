from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_SEEDS = "0,1,2,3,4"
DEFAULT_MAP_ETA_VALUES = "1e-5,3e-5,1e-4,3e-4,1e-3"
DEFAULT_MAP_RHO_VALUES = "0.01,0.03,0.1,0.3"
DEFAULT_MAP_WMAX_VALUES = "25"

SUMMARY_COLUMNS = [
    "run_set",
    "condition",
    "seed",
    "mode",
    "istdp_rule",
    "eta_ie",
    "rho_ie",
    "w_ie_max",
    "normalize_aiae_columns",
    "aiae_target_sum",
    "aiae_normalize_every",
    "theta_gate_ref_mV",
    "theta_gate_scale_mV",
    "slow_homeostat_target_rate_hz",
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


POSTER_CONDITIONS: List[Dict[str, Any]] = [
    {
        "condition": "fixed",
        "mode": "fixed",
        "istdp_rule": "fixed",
        "eta_ie": None,
        "rho_ie": None,
        "w_ie_max": None,
        "normalize_aiae_columns": False,
    },
    {
        "condition": "vogels_stable",
        "mode": "istdp",
        "istdp_rule": "vogels",
        "eta_ie": 3e-4,
        "rho_ie": 0.01,
        "w_ie_max": 25.0,
        "normalize_aiae_columns": False,
    },
    {
        "condition": "vogels_unstable",
        "mode": "istdp",
        "istdp_rule": "vogels",
        "eta_ie": 3e-4,
        "rho_ie": 0.03,
        "w_ie_max": 25.0,
        "normalize_aiae_columns": False,
    },
    {
        "condition": "slow_homeostat_unconstrained",
        "mode": "istdp",
        "istdp_rule": "slow_homeostat",
        "eta_ie": 3e-4,
        "rho_ie": 0.03,
        "w_ie_max": 25.0,
        "normalize_aiae_columns": False,
    },
    {
        "condition": "normalized_slow_homeostat",
        "mode": "istdp",
        "istdp_rule": "normalized_slow_homeostat",
        "eta_ie": 3e-4,
        "rho_ie": 0.03,
        "w_ie_max": 25.0,
        "normalize_aiae_columns": True,
    },
]


def parse_int_list(spec: str) -> List[int]:
    values: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def parse_float_list(spec: str) -> List[float]:
    values: List[float] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def slugify_float(value: Optional[float]) -> str:
    if value is None:
        return "none"
    return f"{value:g}".replace(".", "p").replace("-", "m")


def write_summary_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def load_run_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "logs" / "run_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r") as f:
        return json.load(f)


def poster_specs(seeds: List[int]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for seed, condition in itertools.product(seeds, POSTER_CONDITIONS):
        spec = dict(condition)
        spec["seed"] = seed
        spec["run_set"] = "poster_multiseed"
        spec["run_name"] = f"{condition['condition']}_seed{seed}"
        specs.append(spec)
    return specs


def stability_map_specs(args: argparse.Namespace, seeds: List[int]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    eta_values = parse_float_list(args.map_eta_values)
    rho_values = parse_float_list(args.map_rho_values)
    wmax_values = parse_float_list(args.map_wmax_values)
    for seed, eta_ie, rho_ie, w_ie_max in itertools.product(seeds, eta_values, rho_values, wmax_values):
        specs.append(
            {
                "run_set": "vogels_stability_map",
                "condition": "vogels_map",
                "seed": seed,
                "mode": "istdp",
                "istdp_rule": "vogels",
                "eta_ie": eta_ie,
                "rho_ie": rho_ie,
                "w_ie_max": w_ie_max,
                "normalize_aiae_columns": False,
                "run_name": (
                    f"vogels_eta{slugify_float(eta_ie)}_"
                    f"rho{slugify_float(rho_ie)}_"
                    f"wmax{slugify_float(w_ie_max)}_seed{seed}"
                ),
            }
        )
    return specs


def build_base_command(args: argparse.Namespace, script_path: Path, seed: int) -> List[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--mnist-npz-path",
        args.mnist_npz_path,
        "--epochs",
        str(args.epochs),
        "--train-examples",
        str(args.train_examples),
        "--test-examples",
        str(args.test_examples),
        "--update-interval",
        str(args.update_interval),
        "--weight-stats-every",
        str(args.weight_stats_every),
        "--metrics-window",
        str(args.metrics_window),
        "--plot-every",
        str(args.plot_every),
        "--max-spike-retries-per-example",
        str(args.max_spike_retries_per_example),
        "--seed",
        str(seed),
        "--theta-gate-ref-mv",
        str(args.theta_gate_ref_mV),
        "--theta-gate-scale-mv",
        str(args.theta_gate_scale_mV),
        "--slow-homeostat-target-rate-hz",
        str(args.slow_homeostat_target_rate_hz),
        "--aiae-target-sum",
        str(args.aiae_target_sum),
        "--aiae-normalize-every",
        str(args.aiae_normalize_every),
        "--no-record-spikes",
        "--no-istdp-during-rest",
        "--rollback-state-on-retry",
        "--abort-runaway-spike-threshold",
        str(args.abort_runaway_spike_threshold),
        "--abort-runaway-consecutive-examples",
        str(args.abort_runaway_consecutive_examples),
        "--abort-retry-cap-consecutive-examples",
        str(args.abort_retry_cap_consecutive_examples),
        "--abort-theta-mean-threshold",
        str(args.abort_theta_mean_threshold),
        "--abort-aiae-max-fraction-threshold",
        str(args.abort_aiae_max_fraction_threshold),
    ]
    return cmd


def build_command(args: argparse.Namespace, script_path: Path, spec: Dict[str, Any], run_dir: Path) -> List[str]:
    cmd = build_base_command(args, script_path, int(spec["seed"]))
    cmd.extend(["--out-dir", str(run_dir), "--inhibition-mode", spec["mode"]])
    if spec["mode"] == "istdp":
        cmd.extend(
            [
                "--istdp-rule",
                str(spec["istdp_rule"]),
                "--eta-ie",
                str(spec["eta_ie"]),
                "--rho-ie",
                str(spec["rho_ie"]),
                "--w-ie-max",
                str(spec["w_ie_max"]),
            ]
        )
        if spec.get("normalize_aiae_columns", False):
            cmd.append("--normalize-aiae-columns")
        else:
            cmd.append("--no-normalize-aiae-columns")
    return cmd


def row_from_summary(
    spec: Dict[str, Any],
    run_dir: Path,
    returncode: int,
    skipped: bool,
) -> Dict[str, Any]:
    summary = load_run_summary(run_dir)
    return {
        "run_set": spec["run_set"],
        "condition": spec["condition"],
        "seed": spec["seed"],
        "mode": spec["mode"],
        "istdp_rule": spec["istdp_rule"],
        "eta_ie": spec["eta_ie"],
        "rho_ie": spec["rho_ie"],
        "w_ie_max": spec["w_ie_max"],
        "normalize_aiae_columns": spec.get("normalize_aiae_columns", False),
        "aiae_target_sum": summary.get("aiae_target_sum"),
        "aiae_normalize_every": summary.get("aiae_normalize_every"),
        "theta_gate_ref_mV": summary.get("theta_gate_ref_mV"),
        "theta_gate_scale_mV": summary.get("theta_gate_scale_mV"),
        "slow_homeostat_target_rate_hz": summary.get("slow_homeostat_target_rate_hz"),
        "aborted": summary.get("aborted", returncode != 0),
        "abort_reason": summary.get("abort_reason", f"returncode_{returncode}" if returncode != 0 else ""),
        "accuracy": summary.get("accuracy"),
        "train_mean_spikes": summary.get("train_mean_spikes"),
        "train_usage_entropy_norm": summary.get("train_usage_entropy_norm"),
        "dead_fraction": summary.get("dead_fraction"),
        "firing_rate_cv": summary.get("firing_rate_cv"),
        "gi_mean": summary.get("gi_mean"),
        "ai_ae_mean": summary.get("ai_ae_mean"),
        "ai_ae_max": summary.get("ai_ae_max"),
        "ai_ae_std": summary.get("ai_ae_std"),
        "theta_mean": summary.get("theta_mean"),
        "theta_max": summary.get("theta_max"),
        "processed_train_examples": summary.get("processed_train_examples"),
        "retry_cap_hits": summary.get("retry_cap_hits"),
        "runaway_detected": summary.get("runaway_detected"),
        "collapse_detected": summary.get("collapse_detected"),
        "aiae_max_fraction": summary.get("aiae_max_fraction"),
        "returncode": returncode,
        "skipped": skipped,
        "run_dir": str(run_dir),
    }


def selected_specs(args: argparse.Namespace, seeds: List[int]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    if args.run_set in ("poster_multiseed", "all"):
        specs.extend(poster_specs(seeds))
    if args.run_set in ("vogels_stability_map", "all"):
        specs.extend(stability_map_specs(args, seeds))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run inhibitory-plasticity comparison and stability-map experiments "
            "in timestamped output folders."
        )
    )
    parser.add_argument("--mnist-npz-path", type=str, required=True)
    parser.add_argument(
        "--run-set",
        choices=["poster_multiseed", "vogels_stability_map", "all"],
        default="all",
    )
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--out-base-dir", type=str, default="./runs/inhibitory_plasticity_maps")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--map-eta-values", type=str, default=DEFAULT_MAP_ETA_VALUES)
    parser.add_argument("--map-rho-values", type=str, default=DEFAULT_MAP_RHO_VALUES)
    parser.add_argument("--map-wmax-values", type=str, default=DEFAULT_MAP_WMAX_VALUES)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-examples", type=int, default=3000)
    parser.add_argument("--test-examples", type=int, default=500)
    parser.add_argument("--update-interval", type=int, default=250)
    parser.add_argument("--weight-stats-every", type=int, default=250)
    parser.add_argument("--metrics-window", type=int, default=250)
    parser.add_argument("--plot-every", type=int, default=0)
    parser.add_argument("--max-spike-retries-per-example", type=int, default=25)
    parser.add_argument("--abort-runaway-spike-threshold", type=int, default=5000)
    parser.add_argument("--abort-runaway-consecutive-examples", type=int, default=20)
    parser.add_argument("--abort-retry-cap-consecutive-examples", type=int, default=100)
    parser.add_argument("--abort-theta-mean-threshold", type=float, default=200.0)
    parser.add_argument("--abort-aiae-max-fraction-threshold", type=float, default=0.2)
    parser.add_argument("--theta-gate-ref-mV", type=float, default=20.0)
    parser.add_argument("--theta-gate-scale-mV", type=float, default=10.0)
    parser.add_argument("--slow-homeostat-target-rate-hz", type=float, default=0.1)
    parser.add_argument("--aiae-target-sum", type=float, default=7980.0)
    parser.add_argument("--aiae-normalize-every", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    script_path = repo_root / "Euler_stdp_MNIST_iSTDP.py"
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("At least one seed is required.")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    root_dir = (repo_root / args.out_base_dir / run_id).resolve()
    if root_dir.exists() and not args.resume:
        raise SystemExit(
            f"Output directory already exists: {root_dir}\n"
            "Use a new --run-id or pass --resume to continue missing runs."
        )
    root_dir.mkdir(parents=True, exist_ok=True)

    with open(root_dir / "followup_config.json", "w") as f:
        json.dump(vars(args) | {"resolved_run_id": run_id}, f, indent=2)

    specs = selected_specs(args, seeds)
    combined_summary_path = root_dir / "combined_summary.csv"
    rows: List[Dict[str, Any]] = []
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    print(f"Follow-up output directory: {root_dir}")
    print(f"Planned runs: {len(specs)}")

    for idx, spec in enumerate(specs, start=1):
        run_dir = root_dir / spec["run_set"] / spec["run_name"]
        summary_path = run_dir / "logs" / "run_summary.json"
        cmd = build_command(args, script_path, spec, run_dir)
        skipped = False
        returncode = 0

        print(f"[{idx}/{len(specs)}] {spec['run_set']} / {spec['run_name']}")
        if summary_path.exists() and args.resume:
            print("  existing run_summary.json found; skipping")
            skipped = True
        elif args.dry_run:
            print("  " + " ".join(cmd))
        else:
            completed = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
            returncode = int(completed.returncode)

        rows.append(row_from_summary(spec, run_dir, returncode, skipped))
        write_summary_csv(combined_summary_path, rows)
        run_set_summary = root_dir / spec["run_set"] / "summary.csv"
        run_set_rows = [row for row in rows if row["run_set"] == spec["run_set"]]
        run_set_summary.parent.mkdir(parents=True, exist_ok=True)
        write_summary_csv(run_set_summary, run_set_rows)

        if returncode != 0 and args.stop_on_failure:
            raise SystemExit(returncode)

    print(f"Combined summary CSV: {combined_summary_path}")


if __name__ == "__main__":
    main()
