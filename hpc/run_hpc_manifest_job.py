"""Run one row of an HPC simulation manifest.

This worker is called by SLURM array jobs. It reads one manifest row, builds
the corresponding `Euler_stdp_MNIST_iSTDP.py` command, skips completed runs
unless forced, executes the simulation, and writes a compact per-job status
file for later collection.

author: Fabrizio Musacchio
date:   Jun 2026
"""
# %% IMPORTS
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# %% FUNCTIONS
def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def maybe_add(cmd: list[str], flag: str, value: Any) -> None:
    if value is None or str(value).strip() == "":
        return
    cmd.extend([flag, str(value)])


def build_command(row: dict[str, str], project_dir: Path, mnist_npz_path: Path, out_base_dir: Path) -> tuple[list[str], Path]:
    block = row["block"]
    run_set = row["run_set"]
    run_name = row["run_name"]
    run_dir = out_base_dir / block / run_set / run_name

    cmd = [
        sys.executable,
        str(project_dir / "Euler_stdp_MNIST_iSTDP.py"),
        "--mnist-npz-path",
        str(mnist_npz_path),
        "--out-dir",
        str(run_dir),
        "--epochs",
        "1",
        "--train-examples",
        "30000",
        "--test-examples",
        "5000",
        "--update-interval",
        "250",
        "--plot-every",
        "0",
        "--weight-stats-every",
        "5000",
        "--metrics-window",
        "500",
        "--max-spike-retries-per-example",
        "25",
        "--seed",
        row["seed"],
        "--input-intensity",
        row["input_intensity"],
        "--w-aiae",
        row["w_aiae"],
        "--w-aeai",
        row["w_aeai"],
        "--theta-plus-mv",
        row["theta_plus_mV"],
        "--aiae-target-sum",
        row["aiae_target_sum"],
        "--aiae-normalize-every",
        row["aiae_normalize_every"],
        "--no-record-spikes",
        "--no-istdp-during-rest",
        "--rollback-state-on-retry",
        "--abort-runaway-spike-threshold",
        "5000",
        "--abort-runaway-consecutive-examples",
        "20",
        "--abort-retry-cap-consecutive-examples",
        "100",
        "--abort-theta-mean-threshold",
        "200",
        "--abort-aiae-max-fraction-threshold",
        "0.2",
        "--inhibition-mode",
        row["mode"],
    ]

    if row["mode"] == "istdp":
        maybe_add(cmd, "--istdp-rule", row["istdp_rule"])
        maybe_add(cmd, "--eta-ie", row["eta_ie"])
        maybe_add(cmd, "--rho-ie", row["rho_ie"])
        maybe_add(cmd, "--w-ie-max", row["w_ie_max"])
        maybe_add(cmd, "--slow-homeostat-target-rate-hz", row.get("slow_homeostat_target_rate_hz", ""))
        if truthy(row["normalize_aiae_columns"]):
            cmd.append("--normalize-aiae-columns")
        else:
            cmd.append("--no-normalize-aiae-columns")

    return cmd, run_dir
# %% MAIN FUNCTION
def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Euler STDP manifest row.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path(os.environ.get("PROJECT_DIR", ".")))
    parser.add_argument("--mnist-npz-path", type=Path, default=None)
    parser.add_argument("--out-base-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    mnist_npz_path = args.mnist_npz_path or Path(os.environ.get("MNIST_NPZ_PATH", project_dir / "data" / "mnist.npz"))
    out_base_dir = args.out_base_dir or Path(os.environ.get("OUT_BASE_DIR", project_dir / "runs" / "long_30k_sweeps_manifest"))

    with args.manifest.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.index < 0 or args.index >= len(rows):
        raise SystemExit(f"Index {args.index} outside manifest range 0..{len(rows)-1}")

    row = rows[args.index]
    cmd, run_dir = build_command(row, project_dir, mnist_npz_path, out_base_dir)
    run_summary = run_dir / "logs" / "run_summary.json"
    status_dir = out_base_dir / "_manifest_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"job_{args.index:04d}.json"

    print(f"Manifest index: {args.index}")
    print(f"Block: {row['block']}")
    print(f"Run: {row['run_name']}")
    print(f"Run dir: {run_dir}")

    if run_summary.exists() and not args.force:
        print("Existing run_summary.json found; skipping.")
        payload = {"index": args.index, "skipped": True, "returncode": 0, "run_dir": str(run_dir), "row": row}
        status_path.write_text(json.dumps(payload, indent=2))
        return 0

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(project_dir / ".mplconfig"))
    env.setdefault("NUMBA_CACHE_DIR", str(project_dir / ".numba_cache"))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMBA_NUM_THREADS", "1")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    print("Command:")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=project_dir, env=env, check=False)
    payload = {
        "index": args.index,
        "skipped": False,
        "returncode": int(completed.returncode),
        "run_dir": str(run_dir),
        "row": row,
    }
    status_path.write_text(json.dumps(payload, indent=2))
    return int(completed.returncode)
# %% MAIN ENTRY POINT
if __name__ == "__main__":
    raise SystemExit(main())
# %% END
