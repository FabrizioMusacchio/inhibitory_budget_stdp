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
from typing import Any, Dict, List


DEFAULT_ETA_VALUES = "1e-5,3e-5,1e-4,3e-4,1e-3"
DEFAULT_RHO_VALUES = "0.01,0.03,0.1,0.3"
DEFAULT_WMAX_VALUES = "20,25,30,40"

SUMMARY_COLUMNS = [
    "mode",
    "eta_ie",
    "rho_ie",
    "w_ie_max",
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
    "theta_mean",
    "theta_max",
    "processed_train_examples",
    "retry_cap_hits",
    "runaway_detected",
    "collapse_detected",
    "aiae_max_fraction",
    "returncode",
    "run_dir",
]


def parse_float_list(spec: str) -> List[float]:
    values: List[float] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def slugify_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_run_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "logs" / "run_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r") as f:
        return json.load(f)


def build_base_command(args: argparse.Namespace, script_path: Path) -> List[str]:
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
        str(args.seed),
        "--no-record-spikes",
        "--no-istdp-during-rest",
        "--rollback-state-on-retry",
        "--abort-runaway-spike-threshold",
        str(args.abort_runaway_spike_threshold),
        "--abort-runaway-consecutive-examples",
        str(args.abort_runaway_consecutive_examples),
        "--abort-theta-mean-threshold",
        str(args.abort_theta_mean_threshold),
        "--abort-aiae-max-fraction-threshold",
        str(args.abort_aiae_max_fraction_threshold),
    ]
    return cmd


def build_run_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    if not args.skip_fixed_baseline:
        specs.append(
            {
                "mode": "fixed",
                "eta_ie": None,
                "rho_ie": None,
                "w_ie_max": None,
                "run_name": f"fixed_seed{args.seed}",
            }
        )

    eta_values = parse_float_list(args.eta_values)
    rho_values = parse_float_list(args.rho_values)
    wmax_values = parse_float_list(args.wmax_values)
    for eta_ie, rho_ie, w_ie_max in itertools.product(eta_values, rho_values, wmax_values):
        specs.append(
            {
                "mode": "istdp",
                "eta_ie": eta_ie,
                "rho_ie": rho_ie,
                "w_ie_max": w_ie_max,
                "run_name": (
                    f"istdp_eta{slugify_float(eta_ie)}_"
                    f"rho{slugify_float(rho_ie)}_"
                    f"wmax{slugify_float(w_ie_max)}_seed{args.seed}"
                ),
            }
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Vogels-style iSTDP parameters")
    parser.add_argument("--mnist-npz-path", type=str, required=True)
    parser.add_argument("--out-base-dir", type=str, default="./runs/vogels_istdp_sweep")
    parser.add_argument("--eta-values", type=str, default=DEFAULT_ETA_VALUES)
    parser.add_argument("--rho-values", type=str, default=DEFAULT_RHO_VALUES)
    parser.add_argument("--wmax-values", type=str, default=DEFAULT_WMAX_VALUES)
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
    parser.add_argument("--abort-theta-mean-threshold", type=float, default=200.0)
    parser.add_argument("--abort-aiae-max-fraction-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-fixed-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    script_path = repo_root / "Euler_stdp_MNIST_iSTDP.py"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_base_dir = (repo_root / args.out_base_dir).resolve()
    sweep_dir = out_base_dir / f"sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    config_path = sweep_dir / "sweep_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    specs = build_run_specs(args)
    summary_csv_path = sweep_dir / "sweep_summary.csv"
    rows: List[Dict[str, Any]] = []
    base_cmd = build_base_command(args, script_path)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    print(f"Sweep output directory: {sweep_dir}")
    print(f"Planned runs: {len(specs)}")

    for idx, spec in enumerate(specs, start=1):
        run_dir = sweep_dir / spec["run_name"]
        cmd = list(base_cmd)
        cmd.extend(["--out-dir", str(run_dir), "--inhibition-mode", spec["mode"]])
        if spec["mode"] == "istdp":
            cmd.extend(
                [
                    "--eta-ie",
                    str(spec["eta_ie"]),
                    "--rho-ie",
                    str(spec["rho_ie"]),
                    "--w-ie-max",
                    str(spec["w_ie_max"]),
                ]
            )

        print(f"[{idx}/{len(specs)}] running {spec['run_name']}")
        if args.dry_run:
            print("  " + " ".join(cmd))
            returncode = 0
        else:
            completed = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
            returncode = int(completed.returncode)

        summary = load_run_summary(run_dir)
        row: Dict[str, Any] = {
            "mode": spec["mode"],
            "eta_ie": spec["eta_ie"],
            "rho_ie": spec["rho_ie"],
            "w_ie_max": spec["w_ie_max"],
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
            "theta_mean": summary.get("theta_mean"),
            "theta_max": summary.get("theta_max"),
            "processed_train_examples": summary.get("processed_train_examples"),
            "retry_cap_hits": summary.get("retry_cap_hits"),
            "runaway_detected": summary.get("runaway_detected"),
            "collapse_detected": summary.get("collapse_detected"),
            "aiae_max_fraction": summary.get("aiae_max_fraction"),
            "returncode": returncode,
            "run_dir": str(run_dir),
        }
        rows.append(row)
        write_summary_csv(summary_csv_path, rows)

    if args.dry_run:
        print(f"Dry run complete. Planned summary CSV: {summary_csv_path}")
        return

    print(f"Finished sweep. Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
