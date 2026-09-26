"""Validate the HPC container and project layout.

This smoke test checks that the required Python packages import correctly,
that `data/mnist.npz` is available, that the main project scripts are present,
and that the simulator command-line interface is callable inside the
Singularity/Apptainer container.

author: Fabrizio Musacchio
date:   Jun 2026
"""
# %% IMPORTS
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
# %% MAIN FUNCTION
def main() -> int:
    project_dir = Path(os.environ.get("PROJECT_DIR", Path.cwd()))
    mnist_path = Path(os.environ.get("MNIST_NPZ_PATH", project_dir / "data" / "mnist.npz"))

    print("=== Python ===")
    print("executable:", sys.executable)
    print("version:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("cwd:", Path.cwd())
    print("PROJECT_DIR:", project_dir)
    print("MNIST_NPZ_PATH:", mnist_path)

    print("\n=== Imports ===")
    import numpy as np
    import matplotlib
    import numba
    import sklearn

    print("numpy:", np.__version__)
    print("matplotlib:", matplotlib.__version__)
    print("numba:", numba.__version__)
    print("sklearn:", sklearn.__version__)

    print("\n=== MNIST npz ===")
    if not mnist_path.exists():
        raise FileNotFoundError(f"MNIST npz not found: {mnist_path}")
    with np.load(mnist_path) as data:
        keys = sorted(data.files)
        print("keys:", keys)
        for key in keys:
            arr = data[key]
            print(key, arr.shape, arr.dtype)

    print("\n=== Project files ===")
    required = [
        "Euler_stdp_MNIST_iSTDP.py",
        "run_baseline_regime_sweep.py",
        "run_inhibitory_plasticity_maps.py",
    ]
    for name in required:
        path = project_dir / name
        print(name, "OK" if path.exists() else "MISSING")
        if not path.exists():
            raise FileNotFoundError(path)

    print("\n=== Write test ===")
    out_dir = project_dir / "logs" / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "container_smoke_test.json"
    payload = {
        "python": sys.executable,
        "platform": platform.platform(),
        "project_dir": str(project_dir),
        "mnist_path": str(mnist_path),
        "status": "ok",
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print("wrote:", out_path)

    print("\n=== Simulation CLI help smoke test ===")
    import subprocess

    cmd = [sys.executable, str(project_dir / "Euler_stdp_MNIST_iSTDP.py"), "--help"]
    proc = subprocess.run(cmd, cwd=project_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    print(proc.stdout.splitlines()[0] if proc.stdout else "no output")
    if proc.returncode != 0:
        raise RuntimeError(f"CLI help failed with return code {proc.returncode}")

    print("\nSMOKE_TEST_OK")
    return 0
# %% MAIN ENTRY POINT
if __name__ == "__main__":
    raise SystemExit(main())
# %% END
