r"""

Installation
-------------
Create a conda environment with Python 3.12 or later, and install the required packages:

conda create -n diehl_cook_euler python=3.12 mamba -y
conda activate diehl_cook_euler
mamba install numpy matplotlib numba scikit-learn tensorflow ipykernel -y

Execution
-------------
You can run the code via command line or in VS Code's interactive Python environment. To run from command line:

In general:

```bash
python Euler_stdp_MNIST.py --out_dir ./runs/diehl_cook_euler --epochs 1 --train_examples 60000 --test_examples 10000
```

E.g., 

```bash
source /Users/husker/miniforge3/etc/profile.d/conda.sh
conda activate nest
MPLCONFIGDIR=/tmp/mpl python /Users/husker/Science/Python/Projekte/SNN\ STDP\ MNIST/Euler_stdp_MNIST.py \
  --mnist-npz-path /Users/husker/.keras/datasets/mnist.npz \
  --epochs 1 --train-examples 60000 --test-examples 10000 \
  --update-interval 10000 --plot-every 100 \
  --no-record-spikes \
  --w-aiae 20.0 \
  --input-intensity 2.0 \
  --out-dir /Users/husker/Science/Python/Projekte/SNN\ STDP\ MNIST/runs/real_full_60k
```

This will run the full Diehl-Cook training for 1 epoch on all 60k training examples and then test on 
all 10k test examples, with the specified parameters. Adjust the parameters as needed for quicker runs 
or different configurations.


"""
# %% IMPORTS
from __future__ import annotations

import argparse
import copy
import gzip
import math
import json
import struct
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Optional: sklearn for confusion matrix
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# Optional: keras MNIST loader
try:
    from tensorflow.keras.datasets import mnist
    _HAS_KERAS = True
except Exception:
    _HAS_KERAS = False

# set global properties for all plots:
plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# %% CONFIG

@dataclass
class SimConfig:
    seed: int = 0

    # Sizes
    n_input: int = 784
    n_e: int = 400
    n_i: int = 400

    # Discretization
    dt_ms: float = 0.5
    single_example_time_s: float = 0.35
    resting_time_s: float = 0.15

    # LIF and synapse parameters
    v_rest_e_mV: float = -65.0
    v_rest_i_mV: float = -60.0
    v_reset_e_mV: float = -65.0
    v_reset_i_mV: float = -45.0
    v_thresh_e_base_mV: float = -52.0
    v_thresh_i_mV: float = -40.0

    refrac_e_ms: float = 5.0
    refrac_i_ms: float = 2.0

    tau_v_e_ms: float = 100.0   # from dv/dt denominator (100 ms)
    tau_v_i_ms: float = 10.0    # from dv/dt denominator (10 ms)

    tau_ge_ms: float = 1.0
    tau_gi_ms: float = 2.0

    # Reversal potentials used implicitly in Brian equations
    e_inh_e_mV: float = -100.0
    e_inh_i_mV: float = -85.0

    # Adaptive threshold
    offset_mV: float = 20.0
    theta_plus_mV: float = 0.05
    tc_theta_ms: float = 1e7

    # Input encoding
    input_intensity: float = 2.0
    pixel_divisor: float = 8.0  # training['x']/8 * intensity

    # STDP (X->E only by default)
    tc_pre_ms: float = 20.0
    tc_post1_ms: float = 20.0
    tc_post2_ms: float = 40.0
    nu_pre: float = 0.0001
    nu_post: float = 0.01
    wmin: float = 0.0
    wmax: float = 1.0

    # Fixed recurrent inhibition strengths
    w_aeai: float = 10.4
    w_aiae: float = 20.0

    # Inhibitory plasticity (Ai->Ae)
    inhibition_mode: str = "fixed"  # "fixed" or "istdp"
    istdp_rule: str = "vogels"  # "vogels", "centered", "slow_homeostat", "theta_gated", "normalized_slow_homeostat"
    i_trace_tau_ms: float = 20.0
    e_trace_tau_ms: float = 20.0
    eta_ie: float = 0.001
    rho_ie: float = 0.1
    w_ie_min: float = 0.0
    w_ie_max: float = 40.0
    theta_gate_ref_mV: float = 20.0
    theta_gate_scale_mV: float = 10.0
    slow_homeostat_target_rate_hz: float = 0.1
    normalize_aiae_columns: bool = False
    aiae_target_sum: float = 7980.0  # 20.0 * (n_i - 1) for the default n_i=400
    aiae_normalize_every: int = 1

    # Delays
    max_delay_ms: float = 10.0
    use_delays: bool = True

    # Optional extra input pathway (not active in Diehl-Cook baseline run)
    use_xeai_input: bool = False
    xeai_conn_prob: float = 0.1

    # Training schedule
    epochs: int = 1
    train_examples: int = 60000 # max: 60000 for MNIST
    test_examples: int = 10000  # max: 10000 for MNIST
    update_interval: int = 10000  # assignment recompute
    weight_snapshot_interval: int = 1  # epochs
    weight_stats_every: int = 100
    use_input_intensity_protocol: bool = False
    input_intensity_factors: str = "0.5,1.0,2.0,4.0,1.0"
    max_spike_retries_per_example: int = 25
    istdp_during_rest: bool = False
    rollback_state_on_retry: bool = True
    abort_runaway_spike_threshold: int = 5000
    abort_runaway_consecutive_examples: int = 20
    abort_theta_mean_threshold: float = 200.0
    abort_aiae_max_fraction_threshold: float = 0.2

    # Data loading
    mnist_data_dir: str = "."
    mnist_npz_path: str = "./mnist.npz"
    allow_synthetic_data: bool = False
    train_hard_reset_state: bool = False
    test_hard_reset_state: bool = False

    # Logging and evaluation
    out_dir: str = "./runs/diehl_cook_euler"
    record_spikes: bool = True
    record_istdp_metrics: bool = True
    metrics_window: int = 500
    record_v_example_every: int = 0  # 0 disables, else record full v trace for every k-th example
    record_v_subset: int = 20        # always record subset of neurons for v traces
    store_on_disk_threshold_mb: int = 1024  # memmap if larger than this

    # Receptive field visualization
    rf_grid_sqrt: int = 20  # sqrt(400)=20, arrangement for 2d plots
    plot_every: int = 100

# %% FUNCTIONS
# =========================
# Utilities
# =========================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _steps_from_seconds(cfg: SimConfig, t_s: float) -> int:
    return int(round((t_s * 1000.0) / cfg.dt_ms))

def _steps_from_ms(cfg: SimConfig, t_ms: float) -> int:
    return int(round(t_ms / cfg.dt_ms))


# =========================
# Weight init and IO
# =========================

def init_weights(cfg: SimConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Initialize weights with the same spirit as random_conn_generator.py.
    Uses dense matrices, with optional sparsification where needed.
    """
    W = {}

    # X->E: random + 0.01, scaled by ee_input.
    # In spiking_MNIST.py the effective weight['ee_input']=78 but wmax is 1.
    # In the original, the synaptic state is 'ge' and scaling is implicit via units.
    # Here we keep weights normalized to [0, wmax] and treat scaling via input rates.
    W["XeAe"] = (rng.random((cfg.n_input, cfg.n_e)) + 0.01).astype(np.float32)
    W["XeAe"] *= 0.3  # from random_conn_generator weight['ee_input'] = 0.3
    W["XeAe"] = np.clip(W["XeAe"], cfg.wmin, cfg.wmax)

    # E->I: one to one with 10.4 (original random generator).
    W["AeAi"] = np.zeros((cfg.n_e, cfg.n_i), dtype=np.float32)
    m = min(cfg.n_e, cfg.n_i)
    W["AeAi"][np.arange(m), np.arange(m)] = cfg.w_aeai

    # I->E: all to all except diagonal, weight 17.
    W["AiAe"] = (np.ones((cfg.n_i, cfg.n_e), dtype=np.float32) * cfg.w_aiae)
    W["AiAe"] = np.clip(W["AiAe"], cfg.w_ie_min, cfg.w_ie_max)
    m = min(cfg.n_i, cfg.n_e)
    W["AiAe"][np.arange(m), np.arange(m)] = 0.0

    # X->I exists in random generator but is not used in spiking_MNIST default path.
    W["XeAi"] = np.zeros((cfg.n_input, cfg.n_i), dtype=np.float32)
    if cfg.use_xeai_input:
        W["XeAi"] = rng.random((cfg.n_input, cfg.n_i)).astype(np.float32) * 0.2
        mask = (rng.random((cfg.n_input, cfg.n_i)) < cfg.xeai_conn_prob)
        W["XeAi"] *= mask.astype(np.float32)

    return W

def normalize_columns_l1(W: np.ndarray, target_sum: float) -> None:
    col_sums = W.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    factors = target_sum / col_sums
    W *= factors[np.newaxis, :]


def normalize_aiae_columns_l1(W: np.ndarray, target_sum: float, enforce_zero_diag: bool) -> None:
    if target_sum <= 0.0:
        return
    if enforce_zero_diag and W.shape[0] == W.shape[1]:
        diag_idx = np.arange(W.shape[0], dtype=np.int32)
        W[diag_idx, diag_idx] = 0.0
        col_sums = W.sum(axis=0)
        zero_cols = col_sums == 0.0
        if np.any(zero_cols):
            zero_idx = np.where(zero_cols)[0]
            for j in zero_idx:
                W[:, j] = 1.0
                W[j, j] = 0.0
            col_sums = W.sum(axis=0)
        col_sums[col_sums == 0.0] = 1.0
        W *= (target_sum / col_sums)[np.newaxis, :]
        W[diag_idx, diag_idx] = 0.0
        return

    normalize_columns_l1(W, target_sum=target_sum)


# =========================
# Numba kernels
# =========================

@njit(cache=True)
def _poisson_spikes(rates_hz: np.ndarray, dt_s: float, rng_u: np.ndarray) -> np.ndarray:
    """
    Draw Bernoulli approximation to Poisson in small dt.
    p = rate * dt.
    rng_u must be uniform [0,1) same shape as rates.
    """
    p = rates_hz * dt_s
    out = (rng_u < p).astype(np.uint8)
    return out

@njit(cache=True)
def _decay(x: np.ndarray, alpha: float) -> None:
    x *= alpha

@njit(cache=True)
def _lif_step_e(
    v: np.ndarray,
    ge: np.ndarray,
    gi: np.ndarray,
    theta: np.ndarray,
    refrac_count: np.ndarray,
    dt_ms: float,
    v_rest: float,
    tau_v_ms: float,
    v_reset: float,
    v_thresh_base: float,
    offset: float,
    ge_tau_ms: float,
    gi_tau_ms: float,
    e_inh: float
) -> np.ndarray:
    """
    One Euler step for excitatory neurons, returns spike mask (uint8).
    Uses simplified conductance model consistent with the Brian equations.
    Units are treated in mV and dimensionless conductances.
    """
    n = v.shape[0]
    spk = np.zeros(n, dtype=np.uint8)

    # decay conductances
    ge *= math.exp(-dt_ms / ge_tau_ms)
    gi *= math.exp(-dt_ms / gi_tau_ms)

    for i in range(n):
        if refrac_count[i] > 0:
            refrac_count[i] -= 1
            continue

        # synaptic current terms from Brian equations:
        # I_synE = ge * nS * (-v)
        # I_synI = gi * nS * (E_inh - v)
        # dv/dt = ((v_rest - v) + (I_synE + I_synI)/nS) / tau
        # so effectively: dv/dt = ((v_rest - v) + ge*(-v) + gi*(E_inh - v)) / tau
        dv = ((v_rest - v[i]) + ge[i] * (-v[i]) + gi[i] * (e_inh - v[i])) / tau_v_ms
        v[i] += dt_ms * dv

        thr = theta[i] - offset + v_thresh_base
        if v[i] > thr:
            spk[i] = 1
            v[i] = v_reset
            refrac_count[i] = 1  # will be set outside based on refrac_e steps, placeholder here

    return spk

@njit(cache=True)
def _lif_step_i(
    v: np.ndarray,
    ge: np.ndarray,
    gi: np.ndarray,
    refrac_count: np.ndarray,
    dt_ms: float,
    v_rest: float,
    tau_v_ms: float,
    v_reset: float,
    v_thresh: float,
    ge_tau_ms: float,
    gi_tau_ms: float,
    e_inh: float
) -> np.ndarray:
    n = v.shape[0]
    spk = np.zeros(n, dtype=np.uint8)

    ge *= math.exp(-dt_ms / ge_tau_ms)
    gi *= math.exp(-dt_ms / gi_tau_ms)

    for i in range(n):
        if refrac_count[i] > 0:
            refrac_count[i] -= 1
            continue

        dv = ((v_rest - v[i]) + ge[i] * (-v[i]) + gi[i] * (e_inh - v[i])) / tau_v_ms
        v[i] += dt_ms * dv

        if v[i] > v_thresh:
            spk[i] = 1
            v[i] = v_reset
            refrac_count[i] = 1

    return spk

@njit(cache=True)
def _propagate_dense(pre_spk: np.ndarray, W: np.ndarray, post_g: np.ndarray) -> None:
    """
    Dense propagation: post_g += pre_spk @ W
    pre_spk is 0/1 vector. This is a straightforward O(N_in*N_post) kernel.
    For large networks, you often want sparse gather instead.
    """
    n_pre = pre_spk.shape[0]
    n_post = post_g.shape[0]
    for i in range(n_pre):
        if pre_spk[i]:
            for j in range(n_post):
                post_g[j] += W[i, j]

@njit(cache=True)
def _propagate_sparse(pre_idx: np.ndarray, W: np.ndarray, post_g: np.ndarray) -> None:
    """
    Sparse propagation: post_g += sum_{i in pre_idx} W[i,:]
    pre_idx is list of indices that spiked.
    """
    n_post = post_g.shape[0]
    for k in range(pre_idx.shape[0]):
        i = pre_idx[k]
        for j in range(n_post):
            post_g[j] += W[i, j]

@njit(cache=True)
def _stdp_pre_update(
    W: np.ndarray,
    pre_idx: np.ndarray,
    post1_trace: np.ndarray,
    nu_pre: float,
    wmin: float,
    wmax: float
) -> None:
    n_e = W.shape[1]
    for kk in range(pre_idx.shape[0]):
        i = pre_idx[kk]
        for j in range(n_e):
            w = W[i, j] - (nu_pre * post1_trace[j])
            if w < wmin:
                w = wmin
            elif w > wmax:
                w = wmax
            W[i, j] = w


@njit(cache=True)
def _stdp_post_update(
    W: np.ndarray,
    post_idx: np.ndarray,
    pre_trace: np.ndarray,
    post2_trace: np.ndarray,
    nu_post: float,
    wmin: float,
    wmax: float
) -> None:
    n_in = W.shape[0]
    for kk in range(post_idx.shape[0]):
        j = post_idx[kk]
        post2before = post2_trace[j]
        for i in range(n_in):
            w = W[i, j] + (nu_post * pre_trace[i] * post2before)
            if w < wmin:
                w = wmin
            elif w > wmax:
                w = wmax
            W[i, j] = w


@njit(cache=True)
def _istdp_pre_update(
    W: np.ndarray,
    pre_idx: np.ndarray,
    e_post_trace: np.ndarray,
    eta_ie: float,
    rho_ie: float,
    wmin: float,
    wmax: float,
    enforce_zero_diag: bool
) -> None:
    n_e = W.shape[1]
    for kk in range(pre_idx.shape[0]):
        i = pre_idx[kk]
        for j in range(n_e):
            if enforce_zero_diag and i == j:
                W[i, j] = 0.0
                continue
            w = W[i, j] + (eta_ie * (e_post_trace[j] - rho_ie))
            if w < wmin:
                w = wmin
            elif w > wmax:
                w = wmax
            W[i, j] = w


@njit(cache=True)
def _istdp_pre_update_centered(
    W: np.ndarray,
    pre_idx: np.ndarray,
    e_post_trace: np.ndarray,
    eta_ie: float,
    rho_ie: float,
    wmin: float,
    wmax: float,
    enforce_zero_diag: bool
) -> None:
    n_e = W.shape[1]
    mean_post_trace = float(e_post_trace.mean())
    for kk in range(pre_idx.shape[0]):
        i = pre_idx[kk]
        for j in range(n_e):
            if enforce_zero_diag and i == j:
                W[i, j] = 0.0
                continue
            centered_post = e_post_trace[j] - mean_post_trace
            w = W[i, j] + (eta_ie * (centered_post - rho_ie))
            if w < wmin:
                w = wmin
            elif w > wmax:
                w = wmax
            W[i, j] = w


@njit(cache=True)
def _theta_gate_value(theta_value: float, theta_ref: float, theta_scale: float) -> float:
    if theta_scale <= 0.0:
        return 1.0
    delta = theta_value - theta_ref
    if delta <= 0.0:
        return 1.0
    return math.exp(-delta / theta_scale)


@njit(cache=True)
def _istdp_pre_update_theta_gated(
    W: np.ndarray,
    pre_idx: np.ndarray,
    e_post_trace: np.ndarray,
    theta: np.ndarray,
    eta_ie: float,
    rho_ie: float,
    theta_gate_ref_mV: float,
    theta_gate_scale_mV: float,
    wmin: float,
    wmax: float,
    enforce_zero_diag: bool
) -> None:
    n_e = W.shape[1]
    for kk in range(pre_idx.shape[0]):
        i = pre_idx[kk]
        for j in range(n_e):
            if enforce_zero_diag and i == j:
                W[i, j] = 0.0
                continue
            gate = _theta_gate_value(theta[j], theta_gate_ref_mV, theta_gate_scale_mV)
            w = W[i, j] + (eta_ie * (e_post_trace[j] - rho_ie) * gate)
            if w < wmin:
                w = wmin
            elif w > wmax:
                w = wmax
            W[i, j] = w


@njit(cache=True)
def _istdp_post_update(
    W: np.ndarray,
    post_idx: np.ndarray,
    i_pre_trace: np.ndarray,
    eta_ie: float,
    wmin: float,
    wmax: float,
    enforce_zero_diag: bool
) -> None:
    n_i = W.shape[0]
    for kk in range(post_idx.shape[0]):
        j = post_idx[kk]
        for i in range(n_i):
            if enforce_zero_diag and i == j:
                W[i, j] = 0.0
                continue
            w = W[i, j] + (eta_ie * i_pre_trace[i])
            if w < wmin:
                w = wmin
            elif w > wmax:
                w = wmax
            W[i, j] = w


@njit(cache=True)
def _istdp_post_update_theta_gated(
    W: np.ndarray,
    post_idx: np.ndarray,
    i_pre_trace: np.ndarray,
    theta: np.ndarray,
    eta_ie: float,
    theta_gate_ref_mV: float,
    theta_gate_scale_mV: float,
    wmin: float,
    wmax: float,
    enforce_zero_diag: bool
) -> None:
    n_i = W.shape[0]
    for kk in range(post_idx.shape[0]):
        j = post_idx[kk]
        gate = _theta_gate_value(theta[j], theta_gate_ref_mV, theta_gate_scale_mV)
        for i in range(n_i):
            if enforce_zero_diag and i == j:
                W[i, j] = 0.0
                continue
            w = W[i, j] + (eta_ie * i_pre_trace[i] * gate)
            if w < wmin:
                w = wmin
            elif w > wmax:
                w = wmax
            W[i, j] = w


@njit(cache=True)
def _slow_ie_homeostat_update(
    W: np.ndarray,
    i_spike_count: np.ndarray,
    e_spike_count: np.ndarray,
    single_example_time_s: float,
    target_rate_hz: float,
    eta_ie: float,
    wmin: float,
    wmax: float,
    enforce_zero_diag: bool
) -> None:
    n_i = W.shape[0]
    n_e = W.shape[1]
    inv_duration = 1.0 / max(single_example_time_s, 1e-9)
    for i in range(n_i):
        pre_drive = float(i_spike_count[i])
        if pre_drive <= 0.0:
            if enforce_zero_diag and i < n_e:
                W[i, i] = 0.0
            continue
        for j in range(n_e):
            if enforce_zero_diag and i == j:
                W[i, j] = 0.0
                continue
            post_rate_hz = float(e_spike_count[j]) * inv_duration
            w = W[i, j] + (eta_ie * pre_drive * (post_rate_hz - target_rate_hz))
            if w < wmin:
                w = wmin
            elif w > wmax:
                w = wmax
            W[i, j] = w


# =========================
# Assignment and decoding
# =========================

def compute_assignments(result_monitor: np.ndarray, labels: np.ndarray, n_e: int) -> np.ndarray:
    """
    Like get_new_assignments: for each neuron pick the digit that yields maximal mean rate.
    result_monitor shape: (n_samples, n_e)
    labels shape: (n_samples,)
    """
    assignments = np.full(n_e, -1, dtype=np.int32)
    max_rate = np.zeros(n_e, dtype=np.float32)
    for d in range(10):
        idx = np.where(labels == d)[0]
        if idx.size == 0:
            continue
        rate = result_monitor[idx].mean(axis=0)
        better = rate > max_rate
        assignments[better] = d
        max_rate[better] = rate[better]
    return assignments

def rank_digits(assignments: np.ndarray, spike_rates: np.ndarray) -> np.ndarray:
    summed = np.zeros(10, dtype=np.float32)
    counts = np.zeros(10, dtype=np.int32)
    for d in range(10):
        idx = np.where(assignments == d)[0]
        counts[d] = idx.size
        if idx.size > 0:
            summed[d] = spike_rates[idx].mean()
    return np.argsort(summed)[::-1]


# =========================
# Main trainer
# =========================

class DiehlCookEuler:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.dt_s = cfg.dt_ms / 1000.0
        self.steps_example = _steps_from_seconds(cfg, cfg.single_example_time_s)
        self.steps_rest = _steps_from_seconds(cfg, cfg.resting_time_s)
        self.refrac_e_steps = max(1, _steps_from_ms(cfg, cfg.refrac_e_ms))
        self.refrac_i_steps = max(1, _steps_from_ms(cfg, cfg.refrac_i_ms))

        self.rng = np.random.default_rng(cfg.seed)

        # state
        self.v_e = np.full(cfg.n_e, cfg.v_rest_e_mV - 40.0, dtype=np.float32)
        self.v_i = np.full(cfg.n_i, cfg.v_rest_i_mV - 40.0, dtype=np.float32)
        self.ge_e = np.zeros(cfg.n_e, dtype=np.float32)
        self.gi_e = np.zeros(cfg.n_e, dtype=np.float32)
        self.ge_i = np.zeros(cfg.n_i, dtype=np.float32)
        self.gi_i = np.zeros(cfg.n_i, dtype=np.float32)

        self.theta = np.ones(cfg.n_e, dtype=np.float32) * 20.0  # mV, train initial

        self.refrac_e = np.zeros(cfg.n_e, dtype=np.int32)
        self.refrac_i = np.zeros(cfg.n_i, dtype=np.int32)

        # STDP traces
        self.pre_trace = np.zeros(cfg.n_input, dtype=np.float32)
        self.post1 = np.zeros(cfg.n_e, dtype=np.float32)
        self.post2 = np.zeros(cfg.n_e, dtype=np.float32)
        self.i_pre_trace = np.zeros(cfg.n_i, dtype=np.float32)
        self.e_post_trace = np.zeros(cfg.n_e, dtype=np.float32)

        # trace decays
        self.alpha_pre = math.exp(-cfg.dt_ms / cfg.tc_pre_ms)
        self.alpha_post1 = math.exp(-cfg.dt_ms / cfg.tc_post1_ms)
        self.alpha_post2 = math.exp(-cfg.dt_ms / cfg.tc_post2_ms)
        self.alpha_i_trace = math.exp(-cfg.dt_ms / cfg.i_trace_tau_ms)
        self.alpha_e_trace = math.exp(-cfg.dt_ms / cfg.e_trace_tau_ms)
        self.use_istdp = (cfg.inhibition_mode == "istdp")
        self.istdp_rule = cfg.istdp_rule
        self.normalize_aiae_columns = cfg.normalize_aiae_columns or (self.istdp_rule == "normalized_slow_homeostat")
        self.enforce_ie_zero_diag = (cfg.n_i == cfg.n_e)
        if self.use_istdp and self.istdp_rule not in {
            "vogels", "centered", "slow_homeostat", "theta_gated", "normalized_slow_homeostat"
        }:
            raise ValueError(f"Unsupported iSTDP rule: {self.istdp_rule}")

        # weights
        self.W = init_weights(cfg, self.rng)

        # delay buffers for X spikes
        self.max_delay_steps = max(1, _steps_from_ms(cfg, cfg.max_delay_ms))
        self.delay_buf = np.zeros((self.max_delay_steps, cfg.n_input), dtype=np.uint8)
        self.delay_ptr = 0
        self.input_indices = np.arange(cfg.n_input, dtype=np.int32)
        if cfg.use_delays:
            self.input_delay_steps = self.rng.integers(
                low=0,
                high=self.max_delay_steps,
                size=cfg.n_input,
                endpoint=False
            ).astype(np.int32)
        else:
            self.input_delay_steps = np.zeros(cfg.n_input, dtype=np.int32)

    def reset_dynamic_state(self) -> None:
        cfg = self.cfg
        self.v_e.fill(cfg.v_rest_e_mV - 40.0)
        self.v_i.fill(cfg.v_rest_i_mV - 40.0)
        self.ge_e.fill(0.0)
        self.gi_e.fill(0.0)
        self.ge_i.fill(0.0)
        self.gi_i.fill(0.0)
        self.refrac_e.fill(0)
        self.refrac_i.fill(0)
        self.pre_trace.fill(0.0)
        self.post1.fill(0.0)
        self.post2.fill(0.0)
        self.i_pre_trace.fill(0.0)
        self.e_post_trace.fill(0.0)
        self.delay_buf.fill(0)
        self.delay_ptr = 0

    def snapshot_retry_state(self) -> Dict[str, Any]:
        return {
            "v_e": self.v_e.copy(),
            "v_i": self.v_i.copy(),
            "ge_e": self.ge_e.copy(),
            "gi_e": self.gi_e.copy(),
            "ge_i": self.ge_i.copy(),
            "gi_i": self.gi_i.copy(),
            "theta": self.theta.copy(),
            "refrac_e": self.refrac_e.copy(),
            "refrac_i": self.refrac_i.copy(),
            "pre_trace": self.pre_trace.copy(),
            "post1": self.post1.copy(),
            "post2": self.post2.copy(),
            "i_pre_trace": self.i_pre_trace.copy(),
            "e_post_trace": self.e_post_trace.copy(),
            "delay_buf": self.delay_buf.copy(),
            "delay_ptr": int(self.delay_ptr),
            "W_XeAe": self.W["XeAe"].copy(),
            "W_AiAe": self.W["AiAe"].copy(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore_retry_state(self, snapshot: Dict[str, Any]) -> None:
        self.v_e[:] = snapshot["v_e"]
        self.v_i[:] = snapshot["v_i"]
        self.ge_e[:] = snapshot["ge_e"]
        self.gi_e[:] = snapshot["gi_e"]
        self.ge_i[:] = snapshot["ge_i"]
        self.gi_i[:] = snapshot["gi_i"]
        self.theta[:] = snapshot["theta"]
        self.refrac_e[:] = snapshot["refrac_e"]
        self.refrac_i[:] = snapshot["refrac_i"]
        self.pre_trace[:] = snapshot["pre_trace"]
        self.post1[:] = snapshot["post1"]
        self.post2[:] = snapshot["post2"]
        self.i_pre_trace[:] = snapshot["i_pre_trace"]
        self.e_post_trace[:] = snapshot["e_post_trace"]
        self.delay_buf[:] = snapshot["delay_buf"]
        self.delay_ptr = int(snapshot["delay_ptr"])
        self.W["XeAe"][:] = snapshot["W_XeAe"]
        self.W["AiAe"][:] = snapshot["W_AiAe"]
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])

    def _delay_push(self, x_spk: np.ndarray) -> None:
        self.delay_buf[self.delay_ptr, :] = x_spk
        self.delay_ptr = (self.delay_ptr + 1) % self.max_delay_steps

    def _delay_pop_per_input(self) -> np.ndarray:
        # read spikes with a per-input random delay.
        # delay=0 means current step (just pushed), therefore "-1".
        idx = (self.delay_ptr - self.input_delay_steps - 1) % self.max_delay_steps
        return self.delay_buf[idx, self.input_indices]

    def _rates_from_image(self, img_28x28: np.ndarray, intensity: float) -> np.ndarray:
        rates = (img_28x28.reshape(-1).astype(np.float32) / self.cfg.pixel_divisor) * intensity
        # interpret these as Hz directly
        return rates

    def run_one_example(
        self,
        img: np.ndarray,
        label: int,
        training: bool,
        input_intensity: float,
        record_spikes: bool = True,
        record_v: bool = False,
        v_subset_idx: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Simulate one MNIST image presentation plus resting time.
        Returns spike counts and optional traces.
        """
        cfg = self.cfg

        rates_hz = self._rates_from_image(img, input_intensity)

        # In Brian, delay queues drain during the resting period because the
        # simulation clock keeps advancing. Our simplified rest loop does not
        # step input delay propagation, so stale delayed spikes would otherwise
        # leak into later examples and eventually destabilize the network.
        if cfg.use_delays:
            self.delay_buf.fill(0)
            self.delay_ptr = 0

        spike_events_e: List[Tuple[int, int]] = []
        v_trace_subset = None

        if record_v:
            if v_subset_idx is None:
                v_subset_idx = np.arange(min(cfg.record_v_subset, cfg.n_e), dtype=np.int32)
            v_trace_subset = np.zeros((self.steps_example, v_subset_idx.size), dtype=np.float32)

        e_spike_count = np.zeros(cfg.n_e, dtype=np.int32)
        i_spike_count = np.zeros(cfg.n_i, dtype=np.int32)
        mean_gi_e_sum = 0.0
        mean_gi_e_steps = 0
        istdp_active = training and self.use_istdp
        use_spike_based_istdp = istdp_active and self.istdp_rule not in {
            "slow_homeostat", "normalized_slow_homeostat"
        }
        istdp_active_rest = use_spike_based_istdp and cfg.istdp_during_rest

        for t in range(self.steps_example):
            # input poisson
            u = self.rng.random(cfg.n_input, dtype=np.float32)
            x_spk = _poisson_spikes(rates_hz, self.dt_s, u)

            if cfg.use_delays:
                self._delay_push(x_spk)
                x_eff = self._delay_pop_per_input()
            else:
                x_eff = x_spk

            # decay traces
            _decay(self.pre_trace, self.alpha_pre)
            _decay(self.post1, self.alpha_post1)
            _decay(self.post2, self.alpha_post2)
            _decay(self.i_pre_trace, self.alpha_i_trace)
            _decay(self.e_post_trace, self.alpha_e_trace)

            pre_idx = np.where(x_eff > 0)[0].astype(np.int32)
            if training and pre_idx.size > 0:
                _stdp_pre_update(
                    self.W["XeAe"], pre_idx, self.post1,
                    cfg.nu_pre, cfg.wmin, cfg.wmax
                )

            # propagate X->E (excitatory conductance)
            if pre_idx.size > 0:
                _propagate_sparse(pre_idx, self.W["XeAe"], self.ge_e)
                for i in pre_idx:
                    self.pre_trace[i] = 1.0

            if cfg.use_xeai_input and pre_idx.size > 0:
                _propagate_sparse(pre_idx, self.W["XeAi"], self.ge_i)

            # excitatory step
            e_spk = _lif_step_e(
                self.v_e, self.ge_e, self.gi_e, self.theta, self.refrac_e,
                cfg.dt_ms, cfg.v_rest_e_mV, cfg.tau_v_e_ms, cfg.v_reset_e_mV,
                cfg.v_thresh_e_base_mV, cfg.offset_mV, cfg.tau_ge_ms, cfg.tau_gi_ms,
                cfg.e_inh_e_mV
            )
            # fix refrac steps
            e_idx = np.where(e_spk > 0)[0].astype(np.int32)
            for j in e_idx:
                self.refrac_e[j] = self.refrac_e_steps
                e_spike_count[j] += 1
                if record_spikes:
                    spike_events_e.append((t, int(j)))

            post_idx = e_idx
            if training and post_idx.size > 0:
                _stdp_post_update(
                    self.W["XeAe"], post_idx, self.pre_trace, self.post2,
                    cfg.nu_post, cfg.wmin, cfg.wmax
                )
            if use_spike_based_istdp and post_idx.size > 0:
                if self.istdp_rule == "theta_gated":
                    _istdp_post_update_theta_gated(
                        self.W["AiAe"], post_idx, self.i_pre_trace, self.theta,
                        cfg.eta_ie, cfg.theta_gate_ref_mV, cfg.theta_gate_scale_mV,
                        cfg.w_ie_min, cfg.w_ie_max, self.enforce_ie_zero_diag
                    )
                else:
                    _istdp_post_update(
                        self.W["AiAe"], post_idx, self.i_pre_trace,
                        cfg.eta_ie, cfg.w_ie_min, cfg.w_ie_max,
                        self.enforce_ie_zero_diag
                    )
            for j in post_idx:
                self.post1[j] = 1.0
                self.post2[j] = 1.0
                self.e_post_trace[j] = 1.0

            # theta adaptation
            if training:
                for j in post_idx:
                    self.theta[j] += cfg.theta_plus_mV
                # slow decay
                self.theta *= math.exp(-cfg.dt_ms / cfg.tc_theta_ms)

            # propagate E->I
            if e_idx.size > 0:
                _propagate_sparse(e_idx, self.W["AeAi"], self.ge_i)

            # inhibitory step
            i_spk = _lif_step_i(
                self.v_i, self.ge_i, self.gi_i, self.refrac_i,
                cfg.dt_ms, cfg.v_rest_i_mV, cfg.tau_v_i_ms, cfg.v_reset_i_mV,
                cfg.v_thresh_i_mV, cfg.tau_ge_ms, cfg.tau_gi_ms, cfg.e_inh_i_mV
            )
            i_idx = np.where(i_spk > 0)[0].astype(np.int32)
            for j in i_idx:
                self.refrac_i[j] = self.refrac_i_steps
                i_spike_count[j] += 1

            if use_spike_based_istdp and i_idx.size > 0:
                if self.istdp_rule == "centered":
                    _istdp_pre_update_centered(
                        self.W["AiAe"], i_idx, self.e_post_trace,
                        cfg.eta_ie, cfg.rho_ie, cfg.w_ie_min, cfg.w_ie_max,
                        self.enforce_ie_zero_diag
                    )
                elif self.istdp_rule == "theta_gated":
                    _istdp_pre_update_theta_gated(
                        self.W["AiAe"], i_idx, self.e_post_trace, self.theta,
                        cfg.eta_ie, cfg.rho_ie, cfg.theta_gate_ref_mV, cfg.theta_gate_scale_mV,
                        cfg.w_ie_min, cfg.w_ie_max, self.enforce_ie_zero_diag
                    )
                else:
                    _istdp_pre_update(
                        self.W["AiAe"], i_idx, self.e_post_trace,
                        cfg.eta_ie, cfg.rho_ie, cfg.w_ie_min, cfg.w_ie_max,
                        self.enforce_ie_zero_diag
                    )
            # propagate I->E as inhibition (gi)
            if i_idx.size > 0:
                _propagate_sparse(i_idx, self.W["AiAe"], self.gi_e)
                for i in i_idx:
                    self.i_pre_trace[i] = 1.0

            mean_gi_e_sum += float(self.gi_e.mean())
            mean_gi_e_steps += 1

            if record_v and v_trace_subset is not None:
                v_trace_subset[t, :] = self.v_e[v_subset_idx]

        if istdp_active and self.istdp_rule in {"slow_homeostat", "normalized_slow_homeostat"}:
            _slow_ie_homeostat_update(
                self.W["AiAe"], i_spike_count, e_spike_count,
                cfg.single_example_time_s, cfg.slow_homeostat_target_rate_hz,
                cfg.eta_ie, cfg.w_ie_min, cfg.w_ie_max, self.enforce_ie_zero_diag
            )

        # rest period, no input
        for _ in range(self.steps_rest):
            _decay(self.pre_trace, self.alpha_pre)
            _decay(self.post1, self.alpha_post1)
            _decay(self.post2, self.alpha_post2)
            _decay(self.i_pre_trace, self.alpha_i_trace)
            _decay(self.e_post_trace, self.alpha_e_trace)

            e_spk = _lif_step_e(
                self.v_e, self.ge_e, self.gi_e, self.theta, self.refrac_e,
                cfg.dt_ms, cfg.v_rest_e_mV, cfg.tau_v_e_ms, cfg.v_reset_e_mV,
                cfg.v_thresh_e_base_mV, cfg.offset_mV, cfg.tau_ge_ms, cfg.tau_gi_ms,
                cfg.e_inh_e_mV
            )
            e_idx = np.where(e_spk > 0)[0].astype(np.int32)
            for j in e_idx:
                self.refrac_e[j] = self.refrac_e_steps
            if istdp_active_rest and e_idx.size > 0:
                if self.istdp_rule == "theta_gated":
                    _istdp_post_update_theta_gated(
                        self.W["AiAe"], e_idx, self.i_pre_trace, self.theta,
                        cfg.eta_ie, cfg.theta_gate_ref_mV, cfg.theta_gate_scale_mV,
                        cfg.w_ie_min, cfg.w_ie_max, self.enforce_ie_zero_diag
                    )
                else:
                    _istdp_post_update(
                        self.W["AiAe"], e_idx, self.i_pre_trace,
                        cfg.eta_ie, cfg.w_ie_min, cfg.w_ie_max,
                        self.enforce_ie_zero_diag
                    )
                for j in e_idx:
                    self.e_post_trace[j] = 1.0
            if training:
                for j in e_idx:
                    self.theta[j] += cfg.theta_plus_mV
                self.theta *= math.exp(-cfg.dt_ms / cfg.tc_theta_ms)
            if e_idx.size > 0:
                _propagate_sparse(e_idx, self.W["AeAi"], self.ge_i)

            i_spk = _lif_step_i(
                self.v_i, self.ge_i, self.gi_i, self.refrac_i,
                cfg.dt_ms, cfg.v_rest_i_mV, cfg.tau_v_i_ms, cfg.v_reset_i_mV,
                cfg.v_thresh_i_mV, cfg.tau_ge_ms, cfg.tau_gi_ms, cfg.e_inh_i_mV
            )
            i_idx = np.where(i_spk > 0)[0].astype(np.int32)
            for j in i_idx:
                self.refrac_i[j] = self.refrac_i_steps
            if istdp_active_rest and i_idx.size > 0:
                if self.istdp_rule == "centered":
                    _istdp_pre_update_centered(
                        self.W["AiAe"], i_idx, self.e_post_trace,
                        cfg.eta_ie, cfg.rho_ie, cfg.w_ie_min, cfg.w_ie_max,
                        self.enforce_ie_zero_diag
                    )
                elif self.istdp_rule == "theta_gated":
                    _istdp_pre_update_theta_gated(
                        self.W["AiAe"], i_idx, self.e_post_trace, self.theta,
                        cfg.eta_ie, cfg.rho_ie, cfg.theta_gate_ref_mV, cfg.theta_gate_scale_mV,
                        cfg.w_ie_min, cfg.w_ie_max, self.enforce_ie_zero_diag
                    )
                else:
                    _istdp_pre_update(
                        self.W["AiAe"], i_idx, self.e_post_trace,
                        cfg.eta_ie, cfg.rho_ie, cfg.w_ie_min, cfg.w_ie_max,
                        self.enforce_ie_zero_diag
                    )
            if i_idx.size > 0:
                _propagate_sparse(i_idx, self.W["AiAe"], self.gi_e)
                for i in i_idx:
                    self.i_pre_trace[i] = 1.0

            mean_gi_e_sum += float(self.gi_e.mean())
            mean_gi_e_steps += 1

        return {
            "label": int(label),
            "spike_count_e": e_spike_count.astype(np.int32),
            "spike_count_i": i_spike_count.astype(np.int32),
            "spike_events_e": np.array(spike_events_e, dtype=np.int32) if record_spikes else None,
            "v_trace_subset": v_trace_subset,
            "mean_gi_e": float(mean_gi_e_sum / max(1, mean_gi_e_steps)),
        }


# =========================
# Plotting helpers
# =========================

def plot_raster(spike_events: np.ndarray, 
                title: str, 
                out_png: Path,
                cfg: SimConfig) -> None:
    if spike_events is None or spike_events.size == 0:
        return
    t = spike_events[:, 0]
    n = spike_events[:, 1]
    n_events = int(spike_events.shape[0])
    n_active = int(np.unique(n).size)
    
    # convert t into ms for plotting
    t_ms = t * cfg.dt_ms
    
    plt.figure(figsize=(10, 4))
    if n_events < 200:
        point_size = 8
    elif n_events < 2000:
        point_size = 3
    else:
        point_size = 1
    plt.scatter(t_ms, n, s=point_size)
    plt.xlabel("time (ms)")
    plt.ylabel("neuron id")
    plt.ylim(-1, cfg.n_e)
    plt.xlim(0, cfg.single_example_time_s * 1000.0)
    plt.title(f"{title} | events={n_events}, active={n_active}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_image(img: np.ndarray, label: int, title: str, out_png: Path) -> None:
    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(img, interpolation="nearest", cmap="gray")
    plt.title(f"{title} label={label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_receptive_fields(W_xe: np.ndarray, cfg: SimConfig, out_png: Path, title: str) -> None:
    n_e_sqrt = cfg.rf_grid_sqrt
    n_in_sqrt = 28
    big = np.zeros((n_e_sqrt * n_in_sqrt, n_e_sqrt * n_in_sqrt), dtype=np.float32)
    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
            k = i + j * n_e_sqrt
            rf = W_xe[:, k].reshape(n_in_sqrt, n_in_sqrt)
            big[i*n_in_sqrt:(i+1)*n_in_sqrt, j*n_in_sqrt:(j+1)*n_in_sqrt] = rf
    plt.figure(figsize=(10, 10))
    plt.imshow(big, interpolation="nearest", cmap="hot")
    # add a colobar that is not higher than the actual image:
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_weight_stats(stats: Dict[str, List[float]], out_png: Path) -> None:
    epochs = np.arange(len(stats["mean"]))
    if epochs.size == 0:
        return
    plt.figure(figsize=(7, 4))
    ax1 = plt.gca()
    color1 = 'tab:blue'
    color3 = 'tab:green'
    ax1.plot(epochs, stats["l1"], marker="o", label="mean L1 column sum", color=color1)
    ax1.plot(epochs, stats["l2"], marker="o", label="l2 rms per synapse", color=color3)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("l1/l2", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.plot(epochs, stats["mean"], marker="o", label="mean", color=color2)
    ax2.set_ylabel("mean weight", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_weight_stats_iterations(stats: Dict[str, List[float]], out_png: Path) -> None:
    iterations = np.asarray(stats["iteration"], dtype=np.int32)
    if iterations.size == 0:
        return
    plt.figure(figsize=(7, 4))
    ax1 = plt.gca()
    color1 = 'tab:blue'
    color3 = 'tab:green'
    ax1.plot(iterations, stats["l1"], marker="o", label="mean L1 column sum", color=color1)
    ax1.plot(iterations, stats["l2"], marker="o", label="l2 rms per synapse", color=color3)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("l1/l2", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.plot(iterations, stats["mean"], marker="o", label="mean", color=color2)
    ax2.set_ylabel("mean weight", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.0, 1.0)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_accuracy_over_epochs(accuracies: List[float], out_png: Path) -> None:
    acc = np.asarray(accuracies, dtype=np.float32)
    if acc.size == 0:
        return
    epochs = np.arange(1, acc.size + 1, dtype=np.int32)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, acc * 100.0, marker="o")
    if epochs.size == 1:
        plt.xlim(0.5, 1.5)
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.ylim(0.0, 100.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_metric_over_training(
    records: List[Dict[str, Any]],
    key: str,
    ylabel: str,
    out_png: Path,
    title: str
) -> None:
    if len(records) == 0:
        return
    x = np.asarray([rec["example_end"] for rec in records], dtype=np.int32)
    y = np.asarray([rec[key] for rec in records], dtype=np.float32)
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel("training example")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_rate_histogram(firing_rates: np.ndarray, out_png: Path, title: str) -> None:
    if firing_rates.size == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(firing_rates, bins=30)
    plt.xlabel("mean spikes per image")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_ie_weight_stats_iterations(stats: Dict[str, List[float]], out_png: Path) -> None:
    iterations = np.asarray(stats["iteration"], dtype=np.int32)
    if iterations.size == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(iterations, np.asarray(stats["mean"], dtype=np.float32), marker="o", label="mean")
    plt.plot(iterations, np.asarray(stats["std"], dtype=np.float32), marker="o", label="std")
    plt.plot(iterations, np.asarray(stats["min"], dtype=np.float32), linestyle="--", label="min")
    plt.plot(iterations, np.asarray(stats["max"], dtype=np.float32), linestyle="--", label="max")
    plt.xlabel("iteration")
    plt.ylabel("AiAe weight")
    plt.title("AiAe weight statistics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_protocol_metric(
    records: List[Dict[str, Any]],
    key: str,
    ylabel: str,
    out_png: Path,
    title: str
) -> None:
    if len(records) == 0:
        return
    x = np.asarray([rec["intensity_factor_mean"] for rec in records], dtype=np.float32)
    y = np.asarray([rec[key] for rec in records], dtype=np.float32)
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel("input intensity factor")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def weight_statistics(W: np.ndarray) -> Tuple[float, float, float]:
    mean = float(W.mean())
    l1 = float(np.mean(np.sum(np.abs(W), axis=0)))
    l2 = float(np.sqrt(np.mean(W * W)))
    return mean, l1, l2

def ai_ae_weight_statistics(W: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(W.mean()),
        "min": float(W.min()),
        "max": float(W.max()),
        "std": float(W.std()),
    }

def ai_ae_max_fraction(W: np.ndarray, wmax: float, enforce_zero_diag: bool) -> float:
    mask = np.ones(W.shape, dtype=bool)
    if enforce_zero_diag and W.shape[0] == W.shape[1]:
        np.fill_diagonal(mask, False)
    denom = int(np.count_nonzero(mask))
    if denom == 0:
        return 0.0
    num = int(np.count_nonzero(np.logical_and(mask, W >= (wmax - 1e-6))))
    return float(num / denom)

def write_abort_notes(
    run_dir: Path,
    cfg: SimConfig,
    abort_info: Dict[str, Any],
    summary: Dict[str, Any]
) -> None:
    lines = [
        "# Aborted Run Notes",
        "",
        "## Reason",
        "",
        f"- reason: `{abort_info.get('reason', 'unknown')}`",
        f"- phase: `{abort_info.get('phase', 'unknown')}`",
        f"- epoch: `{abort_info.get('epoch', 'unknown')}`",
        f"- example_index: `{abort_info.get('example_index', 'unknown')}`",
        f"- global_iteration: `{abort_info.get('global_iteration', 'unknown')}`",
        "",
        "## Trigger Values",
        "",
        f"- sum_spk: `{abort_info.get('sum_spk', 'unknown')}`",
        f"- theta_mean: `{abort_info.get('theta_mean', 'unknown')}`",
        f"- theta_max: `{abort_info.get('theta_max', 'unknown')}`",
        f"- ai_ae_max: `{abort_info.get('ai_ae_max', 'unknown')}`",
        f"- ai_ae_max_fraction: `{abort_info.get('ai_ae_max_fraction', 'unknown')}`",
        f"- gi_mean: `{abort_info.get('mean_gi_e', 'unknown')}`",
        "",
        "## Config",
        "",
        f"- inhibition_mode: `{cfg.inhibition_mode}`",
        f"- istdp_rule: `{cfg.istdp_rule}`",
        f"- eta_ie: `{cfg.eta_ie}`",
        f"- rho_ie: `{cfg.rho_ie}`",
        f"- w_ie_max: `{cfg.w_ie_max}`",
        f"- theta_gate_ref_mV: `{cfg.theta_gate_ref_mV}`",
        f"- theta_gate_scale_mV: `{cfg.theta_gate_scale_mV}`",
        f"- slow_homeostat_target_rate_hz: `{cfg.slow_homeostat_target_rate_hz}`",
        f"- normalize_aiae_columns: `{cfg.normalize_aiae_columns}`",
        f"- aiae_target_sum: `{cfg.aiae_target_sum}`",
        f"- aiae_normalize_every: `{cfg.aiae_normalize_every}`",
        f"- istdp_during_rest: `{cfg.istdp_during_rest}`",
        f"- rollback_state_on_retry: `{cfg.rollback_state_on_retry}`",
        "",
        "## Summary",
        "",
        f"- aborted: `{summary.get('aborted')}`",
        f"- train_mean_spikes: `{summary.get('train_mean_spikes')}`",
        f"- train_usage_entropy_norm: `{summary.get('train_usage_entropy_norm')}`",
        f"- dead_fraction: `{summary.get('dead_fraction')}`",
        f"- firing_rate_cv: `{summary.get('firing_rate_cv')}`",
        f"- gi_mean: `{summary.get('gi_mean')}`",
        f"- ai_ae_mean: `{summary.get('ai_ae_mean')}`",
        f"- ai_ae_max: `{summary.get('ai_ae_max')}`",
        f"- theta_mean: `{summary.get('theta_mean')}`",
        f"- theta_max: `{summary.get('theta_max')}`",
        f"- retry_cap_hits: `{summary.get('retry_cap_hits')}`",
        f"- runaway_detected: `{summary.get('runaway_detected')}`",
        f"- collapse_detected: `{summary.get('collapse_detected')}`",
    ]
    with open(run_dir / "ABORTED_RUN_NOTES.md", "w") as f:
        f.write("\n".join(lines) + "\n")

def write_run_summary(run_dir: Path, summary: Dict[str, Any]) -> None:
    with open(run_dir / "logs" / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def _parse_intensity_factors(spec: str) -> np.ndarray:
    values: List[float] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if len(values) == 0:
        return np.asarray([1.0], dtype=np.float32)
    return np.asarray(values, dtype=np.float32)

def _example_intensity(
    example_idx: int,
    total_examples: int,
    base_intensity: float,
    protocol_factors: np.ndarray
) -> Tuple[float, int, float]:
    if total_examples <= 0 or protocol_factors.size == 0:
        return float(base_intensity), 0, 1.0
    block_idx = min(protocol_factors.size - 1, int((example_idx * protocol_factors.size) / total_examples))
    factor = float(protocol_factors[block_idx])
    return float(base_intensity * factor), int(block_idx), factor

def compute_activity_metrics(
    spike_counts: np.ndarray,
    mean_gi_e: np.ndarray,
    intensity_factors: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    n_samples = int(spike_counts.shape[0])
    n_e = int(spike_counts.shape[1]) if spike_counts.ndim == 2 else 0
    if n_samples == 0 or n_e == 0:
        empty = {
            "n_samples": 0.0,
            "mean_e_spikes_per_image": 0.0,
            "mean_active_e_neurons_per_image": 0.0,
            "dead_neuron_fraction": 0.0,
            "usage_entropy": 0.0,
            "usage_entropy_norm": 0.0,
            "dominant_neuron_fraction": 0.0,
            "firing_rate_mean": 0.0,
            "firing_rate_std": 0.0,
            "firing_rate_cv": 0.0,
            "mean_inhibitory_conductance": 0.0,
            "intensity_factor_mean": 0.0,
            "intensity_factor_min": 0.0,
            "intensity_factor_max": 0.0,
        }
        return empty, np.zeros(0, dtype=np.float32)

    spikes_per_image = spike_counts.sum(axis=1).astype(np.float64)
    active_per_image = np.count_nonzero(spike_counts, axis=1).astype(np.float64)
    per_neuron_counts = spike_counts.sum(axis=0).astype(np.float64)
    firing_rates = per_neuron_counts / float(n_samples)
    dead_fraction = float(np.mean(per_neuron_counts <= 0.0))

    total_spikes = float(per_neuron_counts.sum())
    usage_entropy = 0.0
    usage_entropy_norm = 0.0
    dominant_fraction = 0.0
    if total_spikes > 0.0:
        p = per_neuron_counts / total_spikes
        nz = p > 0.0
        usage_entropy = float(-np.sum(p[nz] * np.log(p[nz])))
        if n_e > 1:
            usage_entropy_norm = float(usage_entropy / math.log(float(n_e)))
        dominant_fraction = float(p.max())

    firing_rate_mean = float(firing_rates.mean())
    firing_rate_std = float(firing_rates.std())
    firing_rate_cv = float(firing_rate_std / firing_rate_mean) if firing_rate_mean > 0.0 else 0.0

    metrics = {
        "n_samples": float(n_samples),
        "mean_e_spikes_per_image": float(spikes_per_image.mean()),
        "mean_active_e_neurons_per_image": float(active_per_image.mean()),
        "dead_neuron_fraction": dead_fraction,
        "usage_entropy": usage_entropy,
        "usage_entropy_norm": usage_entropy_norm,
        "dominant_neuron_fraction": dominant_fraction,
        "firing_rate_mean": firing_rate_mean,
        "firing_rate_std": firing_rate_std,
        "firing_rate_cv": firing_rate_cv,
        "mean_inhibitory_conductance": float(mean_gi_e.mean()) if mean_gi_e.size > 0 else 0.0,
        "intensity_factor_mean": float(intensity_factors.mean()) if intensity_factors.size > 0 else 0.0,
        "intensity_factor_min": float(intensity_factors.min()) if intensity_factors.size > 0 else 0.0,
        "intensity_factor_max": float(intensity_factors.max()) if intensity_factors.size > 0 else 0.0,
    }
    return metrics, firing_rates.astype(np.float32)

def _open_binary_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def _read_idx_images(path: Path) -> np.ndarray:
    with _open_binary_file(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"invalid IDX image file magic in {path}: {magic}")
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with _open_binary_file(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"invalid IDX label file magic in {path}: {magic}")
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data


def _find_first_existing(base_dir: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = base_dir / name
        if p.exists():
            return p
        gz = base_dir / f"{name}.gz"
        if gz.exists():
            return gz
    return None


def _load_mnist_from_idx_dir(base_dir: Path) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], str]]:
    train_img = _find_first_existing(base_dir, ["train-images-idx3-ubyte", "train-images.idx3-ubyte"])
    train_lbl = _find_first_existing(base_dir, ["train-labels-idx1-ubyte", "train-labels.idx1-ubyte"])
    test_img = _find_first_existing(base_dir, ["t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"])
    test_lbl = _find_first_existing(base_dir, ["t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"])
    if None in (train_img, train_lbl, test_img, test_lbl):
        return None
    x_train = _read_idx_images(train_img)  # type: ignore[arg-type]
    y_train = _read_idx_labels(train_lbl)  # type: ignore[arg-type]
    x_test = _read_idx_images(test_img)    # type: ignore[arg-type]
    y_test = _read_idx_labels(test_lbl)    # type: ignore[arg-type]
    return (x_train, y_train), (x_test, y_test), f"idx:{base_dir}"


def _load_mnist_from_npz(path: Path) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], str]]:
    if not path.exists():
        return None
    data = np.load(path)
    required = {"x_train", "y_train", "x_test", "y_test"}
    if not required.issubset(set(data.keys())):
        return None
    return (data["x_train"], data["y_train"]), (data["x_test"], data["y_test"]), f"npz:{path}"


def _load_mnist(cfg: SimConfig) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], str]:
    candidate_dirs: List[Path] = []
    seen = set()
    for p in [Path(cfg.mnist_data_dir), Path.cwd(), Path("original_implementation_2015")]:
        r = p.resolve()
        if str(r) not in seen:
            seen.add(str(r))
            candidate_dirs.append(r)

    for d in candidate_dirs:
        out = _load_mnist_from_idx_dir(d)
        if out is not None:
            return out

    npz_candidates = [Path(cfg.mnist_npz_path)]
    for d in candidate_dirs:
        npz_candidates.append(d / "mnist.npz")
    for p in npz_candidates:
        out = _load_mnist_from_npz(p)
        if out is not None:
            return out

    if _HAS_KERAS:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (x_train, y_train), (x_test, y_test), "keras"

    if cfg.allow_synthetic_data:
        rng = np.random.default_rng(cfg.seed)
        n_train = max(cfg.train_examples, 1000)
        n_test = max(cfg.test_examples, 200)
        x_train = rng.integers(0, 256, size=(n_train, 28, 28), dtype=np.uint8)
        y_train = rng.integers(0, 10, size=(n_train,), dtype=np.int32)
        x_test = rng.integers(0, 256, size=(n_test, 28, 28), dtype=np.uint8)
        y_test = rng.integers(0, 10, size=(n_test,), dtype=np.int32)
        return (x_train, y_train), (x_test, y_test), "synthetic"

    raise RuntimeError(
        "MNIST dataset not found. Provide IDX files (train/t10k) in --mnist-data-dir, "
        "or provide --mnist-npz-path, or install tensorflow.keras."
    )


def _build_cfg_from_cli() -> SimConfig:
    parser = argparse.ArgumentParser(description="Euler-based Diehl & Cook STDP MNIST")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-examples", type=int, default=None)
    parser.add_argument("--test-examples", type=int, default=None)
    parser.add_argument("--update-interval", type=int, default=None)
    parser.add_argument("--plot-every", type=int, default=None)
    parser.add_argument("--weight-stats-every", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--mnist-data-dir", type=str, default=None)
    parser.add_argument("--mnist-npz-path", type=str, default=None)
    parser.add_argument("--input-intensity", type=float, default=None)
    parser.add_argument("--input-intensity-protocol", action="store_true")
    parser.add_argument("--input-intensity-factors", type=str, default=None)
    parser.add_argument("--max-spike-retries-per-example", type=int, default=None)
    parser.add_argument("--w-aeai", type=float, default=None)
    parser.add_argument("--w-aiae", type=float, default=None)
    parser.add_argument("--inhibition-mode", choices=["fixed", "istdp"], default=None)
    parser.add_argument(
        "--istdp-rule",
        choices=["vogels", "centered", "slow_homeostat", "theta_gated", "normalized_slow_homeostat"],
        default=None,
    )
    parser.add_argument("--i-trace-tau-ms", type=float, default=None)
    parser.add_argument("--e-trace-tau-ms", type=float, default=None)
    parser.add_argument("--eta-ie", type=float, default=None)
    parser.add_argument("--rho-ie", type=float, default=None)
    parser.add_argument("--w-ie-min", type=float, default=None)
    parser.add_argument("--w-ie-max", type=float, default=None)
    parser.add_argument("--theta-gate-ref-mv", type=float, default=None)
    parser.add_argument("--theta-gate-scale-mv", type=float, default=None)
    parser.add_argument("--slow-homeostat-target-rate-hz", type=float, default=None)
    parser.add_argument("--normalize-aiae-columns", action="store_true")
    parser.add_argument("--no-normalize-aiae-columns", action="store_true")
    parser.add_argument("--aiae-target-sum", type=float, default=None)
    parser.add_argument("--aiae-normalize-every", type=int, default=None)
    parser.add_argument("--metrics-window", type=int, default=None)
    parser.add_argument("--istdp-during-rest", action="store_true")
    parser.add_argument("--no-istdp-during-rest", action="store_true")
    parser.add_argument("--rollback-state-on-retry", action="store_true")
    parser.add_argument("--no-rollback-state-on-retry", action="store_true")
    parser.add_argument("--abort-runaway-spike-threshold", type=int, default=None)
    parser.add_argument("--abort-runaway-consecutive-examples", type=int, default=None)
    parser.add_argument("--abort-theta-mean-threshold", type=float, default=None)
    parser.add_argument("--abort-aiae-max-fraction-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-xeai-input", action="store_true")
    parser.add_argument("--no-delays", action="store_true")
    parser.add_argument("--no-record-spikes", action="store_true")
    parser.add_argument("--no-record-istdp-metrics", action="store_true")
    parser.add_argument("--train-hard-reset-state", action="store_true")
    parser.add_argument("--test-hard-reset-state", action="store_true")
    parser.add_argument("--allow-synthetic-data", action="store_true")
    args = parser.parse_args()

    cfg = SimConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.train_examples is not None:
        cfg.train_examples = args.train_examples
    if args.test_examples is not None:
        cfg.test_examples = args.test_examples
    if args.update_interval is not None:
        cfg.update_interval = args.update_interval
    if args.plot_every is not None:
        cfg.plot_every = args.plot_every
    if args.weight_stats_every is not None:
        cfg.weight_stats_every = args.weight_stats_every
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.mnist_data_dir is not None:
        cfg.mnist_data_dir = args.mnist_data_dir
    if args.mnist_npz_path is not None:
        cfg.mnist_npz_path = args.mnist_npz_path
    if args.input_intensity is not None:
        cfg.input_intensity = args.input_intensity
    if args.input_intensity_protocol:
        cfg.use_input_intensity_protocol = True
    if args.input_intensity_factors is not None:
        cfg.input_intensity_factors = args.input_intensity_factors
    if args.max_spike_retries_per_example is not None:
        cfg.max_spike_retries_per_example = args.max_spike_retries_per_example
    if args.w_aeai is not None:
        cfg.w_aeai = args.w_aeai
    if args.w_aiae is not None:
        cfg.w_aiae = args.w_aiae
    if args.inhibition_mode is not None:
        cfg.inhibition_mode = args.inhibition_mode
    if args.istdp_rule is not None:
        cfg.istdp_rule = args.istdp_rule
    if args.i_trace_tau_ms is not None:
        cfg.i_trace_tau_ms = args.i_trace_tau_ms
    if args.e_trace_tau_ms is not None:
        cfg.e_trace_tau_ms = args.e_trace_tau_ms
    if args.eta_ie is not None:
        cfg.eta_ie = args.eta_ie
    if args.rho_ie is not None:
        cfg.rho_ie = args.rho_ie
    if args.w_ie_min is not None:
        cfg.w_ie_min = args.w_ie_min
    if args.w_ie_max is not None:
        cfg.w_ie_max = args.w_ie_max
    if args.theta_gate_ref_mv is not None:
        cfg.theta_gate_ref_mV = args.theta_gate_ref_mv
    if args.theta_gate_scale_mv is not None:
        cfg.theta_gate_scale_mV = args.theta_gate_scale_mv
    if args.slow_homeostat_target_rate_hz is not None:
        cfg.slow_homeostat_target_rate_hz = args.slow_homeostat_target_rate_hz
    if args.normalize_aiae_columns:
        cfg.normalize_aiae_columns = True
    if args.no_normalize_aiae_columns:
        cfg.normalize_aiae_columns = False
    if args.aiae_target_sum is not None:
        cfg.aiae_target_sum = args.aiae_target_sum
    if args.aiae_normalize_every is not None:
        cfg.aiae_normalize_every = args.aiae_normalize_every
    if args.metrics_window is not None:
        cfg.metrics_window = args.metrics_window
    if args.istdp_during_rest:
        cfg.istdp_during_rest = True
    if args.no_istdp_during_rest:
        cfg.istdp_during_rest = False
    if args.rollback_state_on_retry:
        cfg.rollback_state_on_retry = True
    if args.no_rollback_state_on_retry:
        cfg.rollback_state_on_retry = False
    if args.abort_runaway_spike_threshold is not None:
        cfg.abort_runaway_spike_threshold = args.abort_runaway_spike_threshold
    if args.abort_runaway_consecutive_examples is not None:
        cfg.abort_runaway_consecutive_examples = args.abort_runaway_consecutive_examples
    if args.abort_theta_mean_threshold is not None:
        cfg.abort_theta_mean_threshold = args.abort_theta_mean_threshold
    if args.abort_aiae_max_fraction_threshold is not None:
        cfg.abort_aiae_max_fraction_threshold = args.abort_aiae_max_fraction_threshold
    if args.seed is not None:
        cfg.seed = args.seed
    if args.use_xeai_input:
        cfg.use_xeai_input = True
    if args.no_delays:
        cfg.use_delays = False
    if args.no_record_spikes:
        cfg.record_spikes = False
    if args.no_record_istdp_metrics:
        cfg.record_istdp_metrics = False
    if args.train_hard_reset_state:
        cfg.train_hard_reset_state = True
    if args.test_hard_reset_state:
        cfg.test_hard_reset_state = True
    if args.allow_synthetic_data:
        cfg.allow_synthetic_data = True
    if cfg.istdp_rule == "normalized_slow_homeostat":
        cfg.normalize_aiae_columns = True
    return cfg


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    if _HAS_SK:
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    cm = np.zeros((10, 10), dtype=np.int32)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    cfg = _build_cfg_from_cli()
    run_dir = Path(cfg.out_dir)
    _ensure_dir(run_dir)
    _ensure_dir(run_dir / "plots")
    _ensure_dir(run_dir / "snapshots")
    _ensure_dir(run_dir / "logs")

    protocol_factors = _parse_intensity_factors(cfg.input_intensity_factors)
    if not cfg.use_input_intensity_protocol:
        protocol_factors = np.asarray([1.0], dtype=np.float32)
    metrics_window = max(1, cfg.metrics_window)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(run_dir / "logs" / "intensity_protocol.json", "w") as f:
        json.dump(
            {
                "enabled": cfg.use_input_intensity_protocol,
                "base_input_intensity": cfg.input_intensity,
                "factors": [float(x) for x in protocol_factors.tolist()],
            },
            f,
            indent=2,
        )

    (x_train, y_train), (x_test, y_test), source = _load_mnist(cfg)
    x_train = x_train.astype(np.uint8)
    x_test = x_test.astype(np.uint8)
    y_train = y_train.astype(np.int32).reshape(-1)
    y_test = y_test.astype(np.int32).reshape(-1)
    print(f"Loaded MNIST source={source} train={x_train.shape} test={x_test.shape}")
    print(
        f"inhibition_mode={cfg.inhibition_mode} "
        f"input_protocol={'on' if cfg.use_input_intensity_protocol else 'off'} "
        f"protocol_factors={[float(x) for x in protocol_factors.tolist()]}"
    )

    for k in range(min(5, x_train.shape[0], x_test.shape[0])):
        plot_image(x_train[k], y_train[k], f"train example {k}", run_dir / "plots" / f"train_example_{k}.png")
        plot_image(x_test[k], y_test[k], f"test example {k}", run_dir / "plots" / f"test_example_{k}.png")

    trainer = DiehlCookEuler(cfg)
    normalize_columns_l1(trainer.W["XeAe"], target_sum=78.0)
    weight_stats_epoch = {"mean": [], "l1": [], "l2": []}
    weight_stats_iter = {"iteration": [], "mean": [], "l1": [], "l2": []}
    ie_weight_stats_iter = {"iteration": [], "mean": [], "min": [], "max": [], "std": []}
    test_accuracy_over_epochs: List[float] = []
    all_train_window_records: List[Dict[str, Any]] = []
    final_run_summary: Dict[str, Any] = {}

    for ep in range(cfg.epochs):
        print(f"Epoch {ep+1}/{cfg.epochs}")
        train_n = min(cfg.train_examples, x_train.shape[0])
        result_monitor = np.zeros((train_n, cfg.n_e), dtype=np.int32)
        train_labels = np.zeros(train_n, dtype=np.int32)
        train_sum_spk = np.zeros(train_n, dtype=np.int32)
        train_active = np.zeros(train_n, dtype=np.int32)
        train_mean_gi = np.zeros(train_n, dtype=np.float32)
        train_input_intensity = np.zeros(train_n, dtype=np.float32)
        train_input_factor = np.zeros(train_n, dtype=np.float32)
        train_block_idx = np.zeros(train_n, dtype=np.int32)
        train_window_records: List[Dict[str, Any]] = []
        window_start = 0
        processed_train_n = 0
        abort_info: Optional[Dict[str, Any]] = None
        consecutive_runaway_examples = 0
        retry_cap_hits = 0
        runaway_detected = False
        collapse_detected = False
        last_theta_mean = float(trainer.theta.mean())
        last_theta_max = float(trainer.theta.max())
        last_ai_ae_stats = ai_ae_weight_statistics(trainer.W["AiAe"])
        last_ai_ae_max_fraction = ai_ae_max_fraction(
            trainer.W["AiAe"], cfg.w_ie_max, trainer.enforce_ie_zero_diag
        )
        initial_ai_ae_max_fraction = last_ai_ae_max_fraction

        for k in range(train_n):
            if cfg.train_hard_reset_state:
                trainer.reset_dynamic_state()
            img = x_train[k]
            label = int(y_train[k])
            scheduled_intensity, block_idx, _ = _example_intensity(
                k, train_n, cfg.input_intensity, protocol_factors
            )
            retries = 0
            current_input_intensity = scheduled_intensity
            retry_cap_hit = False
            while True:
                normalize_columns_l1(trainer.W["XeAe"], target_sum=78.0)
                retry_snapshot = trainer.snapshot_retry_state() if cfg.rollback_state_on_retry else None
                rec_v = (cfg.record_v_example_every > 0 and (k % cfg.record_v_example_every == 0))
                out = trainer.run_one_example(
                    img=img,
                    label=label,
                    training=True,
                    input_intensity=current_input_intensity,
                    record_spikes=cfg.record_spikes,
                    record_v=rec_v
                )
                sc = out["spike_count_e"]
                sum_spk = int(sc.sum())
                if sum_spk < 5:
                    retries += 1
                    if cfg.rollback_state_on_retry and retry_snapshot is not None:
                        trainer.restore_retry_state(retry_snapshot)
                    if retries >= cfg.max_spike_retries_per_example:
                        retry_cap_hit = True
                        retry_cap_hits += 1
                        collapse_detected = True
                        sc = np.zeros(cfg.n_e, dtype=np.int32)
                        sum_spk = 0
                        out = {
                            "label": int(label),
                            "spike_count_e": sc,
                            "spike_events_e": None,
                            "v_trace_subset": None,
                            "mean_gi_e": 0.0,
                        }
                        print(
                            f"  warning: train ex {k+1} hit retry cap={cfg.max_spike_retries_per_example} "
                            f"at input={current_input_intensity:.2f} with sum_spk={sum_spk}"
                        )
                        break
                    current_input_intensity += 1.0
                    continue
                break

            result_monitor[k, :] = sc
            train_labels[k] = label
            train_sum_spk[k] = sum_spk
            train_active[k] = int(np.count_nonzero(sc))
            train_mean_gi[k] = float(out["mean_gi_e"])
            train_input_intensity[k] = float(current_input_intensity)
            if cfg.input_intensity > 0.0:
                train_input_factor[k] = float(current_input_intensity / cfg.input_intensity)
            train_block_idx[k] = block_idx
            processed_train_n = k + 1

            if trainer.normalize_aiae_columns and (processed_train_n % max(1, cfg.aiae_normalize_every) == 0):
                normalize_aiae_columns_l1(
                    trainer.W["AiAe"], cfg.aiae_target_sum, trainer.enforce_ie_zero_diag
                )

            last_theta_mean = float(trainer.theta.mean())
            last_theta_max = float(trainer.theta.max())
            last_ai_ae_stats = ai_ae_weight_statistics(trainer.W["AiAe"])
            last_ai_ae_max_fraction = ai_ae_max_fraction(
                trainer.W["AiAe"], cfg.w_ie_max, trainer.enforce_ie_zero_diag
            )
            if sum_spk > cfg.abort_runaway_spike_threshold:
                consecutive_runaway_examples += 1
                runaway_detected = True
            else:
                consecutive_runaway_examples = 0

            global_iter = ep * train_n + (k + 1)
            if cfg.weight_stats_every > 0 and ((k + 1) % cfg.weight_stats_every == 0):
                m_i, l1_i, l2_i = weight_statistics(trainer.W["XeAe"])
                weight_stats_iter["iteration"].append(global_iter)
                weight_stats_iter["mean"].append(m_i)
                weight_stats_iter["l1"].append(l1_i)
                weight_stats_iter["l2"].append(l2_i)
                ie_weight_stats_iter["iteration"].append(global_iter)
                ie_weight_stats_iter["mean"].append(last_ai_ae_stats["mean"])
                ie_weight_stats_iter["min"].append(last_ai_ae_stats["min"])
                ie_weight_stats_iter["max"].append(last_ai_ae_stats["max"])
                ie_weight_stats_iter["std"].append(last_ai_ae_stats["std"])

            if cfg.plot_every > 0 and ((k + 1) % cfg.plot_every == 0):
                plot_receptive_fields(
                    trainer.W["XeAe"], cfg,
                    run_dir / "plots" / f"rf_ep{ep}_ex{k+1:05d}.png",
                    f"receptive fields epoch={ep} ex={k+1}"
                )

            if (cfg.record_spikes and k < 25) or (cfg.record_spikes and (k % 1000 == 0)):
                plot_image(img, label, f"train ep{ep} ex{k}", run_dir / "plots" / f"train_img_ep{ep}_ex{k}.png")
                plot_raster(
                    out["spike_events_e"],
                    title=f"train raster ep{ep} ex{k}, shown digit: {label}",
                    out_png=run_dir / "plots" / f"train_raster_ep{ep}_ex{k}.png",
                    cfg=cfg,
                )

            if k == 0 or ((k + 1) % 100 == 0):
                print(
                    f"  train ex {k+1}/{train_n} sum_spk={sum_spk} active={train_active[k]} "
                    f"retries={retries} input={current_input_intensity:.2f} "
                    f"theta_mean={trainer.theta.mean():.3f} theta_max={trainer.theta.max():.3f}"
                )

            if cfg.update_interval > 0 and ((k + 1) % cfg.update_interval == 0):
                w0 = k + 1 - cfg.update_interval
                w1 = k + 1
                s = train_sum_spk[w0:w1]
                a = train_active[w0:w1]
                summary, _ = compute_activity_metrics(
                    result_monitor[w0:w1],
                    train_mean_gi[w0:w1],
                    train_input_factor[w0:w1]
                )
                print(
                    f"  train window [{w0}:{w1}] mean_sum={s.mean():.2f} min_sum={s.min()} max_sum={s.max()} "
                    f"mean_active={a.mean():.2f} usage_entropy={summary['usage_entropy_norm']:.3f} "
                    f"dead_frac={summary['dead_neuron_fraction']:.3f} fr_cv={summary['firing_rate_cv']:.3f} "
                    f"gi_mean={summary['mean_inhibitory_conductance']:.3f}"
                )

            if consecutive_runaway_examples >= cfg.abort_runaway_consecutive_examples:
                abort_info = {
                    "reason": "runaway_spikes_consecutive",
                    "phase": "train",
                    "epoch": int(ep),
                    "example_index": int(k),
                    "global_iteration": int(global_iter),
                    "sum_spk": int(sum_spk),
                    "theta_mean": last_theta_mean,
                    "theta_max": last_theta_max,
                    "ai_ae_max": last_ai_ae_stats["max"],
                    "ai_ae_max_fraction": last_ai_ae_max_fraction,
                    "mean_gi_e": float(out["mean_gi_e"]),
                }
            elif last_theta_mean > cfg.abort_theta_mean_threshold:
                abort_info = {
                    "reason": "theta_mean_threshold",
                    "phase": "train",
                    "epoch": int(ep),
                    "example_index": int(k),
                    "global_iteration": int(global_iter),
                    "sum_spk": int(sum_spk),
                    "theta_mean": last_theta_mean,
                    "theta_max": last_theta_max,
                    "ai_ae_max": last_ai_ae_stats["max"],
                    "ai_ae_max_fraction": last_ai_ae_max_fraction,
                    "mean_gi_e": float(out["mean_gi_e"]),
                }
            elif (
                last_ai_ae_max_fraction > cfg.abort_aiae_max_fraction_threshold and
                last_ai_ae_max_fraction > (initial_ai_ae_max_fraction + 1e-6)
            ):
                abort_info = {
                    "reason": "aiae_max_fraction_threshold",
                    "phase": "train",
                    "epoch": int(ep),
                    "example_index": int(k),
                    "global_iteration": int(global_iter),
                    "sum_spk": int(sum_spk),
                    "theta_mean": last_theta_mean,
                    "theta_max": last_theta_max,
                    "ai_ae_max": last_ai_ae_stats["max"],
                    "ai_ae_max_fraction": last_ai_ae_max_fraction,
                    "mean_gi_e": float(out["mean_gi_e"]),
                }

            if cfg.record_istdp_metrics and (((k + 1) - window_start) >= metrics_window or (k + 1) == train_n):
                w0 = window_start
                w1 = k + 1
                window_metrics, _ = compute_activity_metrics(
                    result_monitor[w0:w1],
                    train_mean_gi[w0:w1],
                    train_input_factor[w0:w1]
                )
                window_metrics.update(
                    {
                        "epoch": int(ep),
                        "phase": "train",
                        "window_start": int(w0),
                        "window_end": int(w1),
                        "example_end": int(global_iter),
                    }
                )
                train_window_records.append(window_metrics)
                window_start = w1

            if abort_info is not None:
                print(
                    f"  aborting run: reason={abort_info['reason']} ex={k+1}/{train_n} "
                    f"sum_spk={sum_spk} theta_mean={last_theta_mean:.3f} "
                    f"ai_ae_max_fraction={last_ai_ae_max_fraction:.3f} retry_cap_hit={retry_cap_hit}"
                )
                break

        if cfg.record_istdp_metrics and window_start < processed_train_n:
            window_metrics, _ = compute_activity_metrics(
                result_monitor[window_start:processed_train_n],
                train_mean_gi[window_start:processed_train_n],
                train_input_factor[window_start:processed_train_n]
            )
            window_metrics.update(
                {
                    "epoch": int(ep),
                    "phase": "train",
                    "window_start": int(window_start),
                    "window_end": int(processed_train_n),
                    "example_end": int(ep * train_n + processed_train_n),
                }
            )
            train_window_records.append(window_metrics)
            window_start = processed_train_n

        if cfg.weight_stats_every > 0:
            last_iter = ep * train_n + processed_train_n
            if len(weight_stats_iter["iteration"]) == 0 or weight_stats_iter["iteration"][-1] != last_iter:
                m_i, l1_i, l2_i = weight_statistics(trainer.W["XeAe"])
                weight_stats_iter["iteration"].append(last_iter)
                weight_stats_iter["mean"].append(m_i)
                weight_stats_iter["l1"].append(l1_i)
                weight_stats_iter["l2"].append(l2_i)
                ie_weight_stats_iter["iteration"].append(last_iter)
                ie_weight_stats_iter["mean"].append(last_ai_ae_stats["mean"])
                ie_weight_stats_iter["min"].append(last_ai_ae_stats["min"])
                ie_weight_stats_iter["max"].append(last_ai_ae_stats["max"])
                ie_weight_stats_iter["std"].append(last_ai_ae_stats["std"])

        result_monitor_used = result_monitor[:processed_train_n]
        train_labels_used = train_labels[:processed_train_n]
        train_sum_spk_used = train_sum_spk[:processed_train_n]
        train_active_used = train_active[:processed_train_n]
        train_mean_gi_used = train_mean_gi[:processed_train_n]
        train_input_intensity_used = train_input_intensity[:processed_train_n]
        train_input_factor_used = train_input_factor[:processed_train_n]
        train_block_idx_used = train_block_idx[:processed_train_n]
        if processed_train_n > 0:
            assignments = compute_assignments(result_monitor_used, train_labels_used, cfg.n_e)
        else:
            assignments = np.full(cfg.n_e, -1, dtype=np.int32)
        np.save(run_dir / "logs" / f"assignments_ep{ep}.npy", assignments)
        np.save(run_dir / "logs" / f"train_sum_spk_ep{ep}.npy", train_sum_spk_used)
        np.save(run_dir / "logs" / f"train_active_ep{ep}.npy", train_active_used)
        np.save(run_dir / "logs" / f"train_mean_gi_ep{ep}.npy", train_mean_gi_used)
        np.save(run_dir / "logs" / f"train_input_intensity_ep{ep}.npy", train_input_intensity_used)
        np.save(run_dir / "logs" / f"train_input_factor_ep{ep}.npy", train_input_factor_used)
        np.save(run_dir / "logs" / f"train_block_idx_ep{ep}.npy", train_block_idx_used)

        train_activity_metrics, train_firing_rates = compute_activity_metrics(
            result_monitor_used, train_mean_gi_used, train_input_factor_used
        )
        train_activity_metrics.update(
            {
                "epoch": int(ep),
                "phase": "train",
                "inhibition_mode": cfg.inhibition_mode,
                "aborted": abort_info is not None,
                "processed_train_examples": int(processed_train_n),
            }
        )
        with open(run_dir / "logs" / f"train_activity_metrics_ep{ep}.json", "w") as f:
            json.dump(train_activity_metrics, f, indent=2)
        np.save(run_dir / "logs" / f"train_firing_rates_ep{ep}.npy", train_firing_rates)
        np.save(run_dir / "logs" / f"train_per_neuron_spike_counts_ep{ep}.npy", result_monitor_used.sum(axis=0))

        if cfg.record_istdp_metrics:
            with open(run_dir / "logs" / f"train_window_metrics_ep{ep}.json", "w") as f:
                json.dump(train_window_records, f, indent=2)
            all_train_window_records.extend(train_window_records)

        m, l1, l2 = weight_statistics(trainer.W["XeAe"])
        weight_stats_epoch["mean"].append(m)
        weight_stats_epoch["l1"].append(l1)
        weight_stats_epoch["l2"].append(l2)
        np.save(run_dir / "snapshots" / f"W_XeAe_ep{ep}.npy", trainer.W["XeAe"])
        np.save(run_dir / "snapshots" / f"W_AiAe_ep{ep}.npy", trainer.W["AiAe"])
        np.save(run_dir / "snapshots" / f"theta_ep{ep}.npy", trainer.theta)
        plot_receptive_fields(trainer.W["XeAe"], cfg, run_dir / "plots" / f"rf_ep{ep}.png", f"receptive fields epoch {ep}")
        plot_weight_stats(weight_stats_epoch, run_dir / "plots" / "weight_stats.png")
        plot_weight_stats_iterations(weight_stats_iter, run_dir / "plots" / "weight_stats_iterations.png")
        plot_ie_weight_stats_iterations(ie_weight_stats_iter, run_dir / "plots" / "aiae_weight_stats_iterations.png")
        plot_rate_histogram(
            train_firing_rates,
            run_dir / "plots" / f"train_firing_rate_distribution_ep{ep}.png",
            f"train E firing-rate distribution ({cfg.inhibition_mode}) epoch {ep}"
        )
        if cfg.record_istdp_metrics and len(all_train_window_records) > 0:
            plot_metric_over_training(
                all_train_window_records,
                "usage_entropy_norm",
                "normalized usage entropy",
                run_dir / "plots" / "train_usage_entropy_over_training.png",
                "train neuron usage entropy"
            )
            plot_metric_over_training(
                all_train_window_records,
                "dead_neuron_fraction",
                "dead neuron fraction",
                run_dir / "plots" / "train_dead_neuron_fraction_over_training.png",
                "train dead neuron fraction"
            )
            plot_metric_over_training(
                all_train_window_records,
                "firing_rate_cv",
                "firing-rate CV",
                run_dir / "plots" / "train_firing_rate_cv_over_training.png",
                "train firing-rate coefficient of variation"
            )

        test_activity_metrics, test_firing_rates = compute_activity_metrics(
            np.zeros((0, cfg.n_e), dtype=np.int32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32)
        )
        test_activity_metrics.update(
            {
                "epoch": int(ep),
                "phase": "test",
                "inhibition_mode": cfg.inhibition_mode,
                "aborted": abort_info is not None,
            }
        )
        y_true = np.zeros(0, dtype=np.int32)
        y_pred = np.zeros(0, dtype=np.int32)
        test_sum_spk = np.zeros(0, dtype=np.int32)
        test_active = np.zeros(0, dtype=np.int32)
        test_mean_gi = np.zeros(0, dtype=np.float32)
        test_input_intensity = np.zeros(0, dtype=np.float32)
        test_input_factor = np.zeros(0, dtype=np.float32)
        test_block_idx = np.zeros(0, dtype=np.int32)
        test_result_monitor = np.zeros((0, cfg.n_e), dtype=np.int32)
        acc: Optional[float] = None

        if abort_info is None:
            test_n = min(cfg.test_examples, x_test.shape[0])
            y_true = np.zeros(test_n, dtype=np.int32)
            y_pred = np.zeros(test_n, dtype=np.int32)
            test_sum_spk = np.zeros(test_n, dtype=np.int32)
            test_active = np.zeros(test_n, dtype=np.int32)
            test_mean_gi = np.zeros(test_n, dtype=np.float32)
            test_input_intensity = np.zeros(test_n, dtype=np.float32)
            test_input_factor = np.zeros(test_n, dtype=np.float32)
            test_block_idx = np.zeros(test_n, dtype=np.int32)
            test_result_monitor = np.zeros((test_n, cfg.n_e), dtype=np.int32)

            for k in range(test_n):
                if cfg.test_hard_reset_state:
                    trainer.reset_dynamic_state()
                img = x_test[k]
                label = int(y_test[k])
                eval_input_intensity, block_idx, _ = _example_intensity(
                    k, test_n, cfg.input_intensity, protocol_factors
                )
                out = trainer.run_one_example(
                    img=img,
                    label=label,
                    training=False,
                    input_intensity=eval_input_intensity,
                    record_spikes=(cfg.record_spikes and k < 25) or (cfg.record_spikes and (k % 1000 == 0)),
                    record_v=False
                )
                sc_i = out["spike_count_e"]
                sc = sc_i.astype(np.float32)
                ranking = rank_digits(assignments, sc)
                y_true[k] = label
                y_pred[k] = int(ranking[0])
                test_result_monitor[k, :] = sc_i
                test_sum_spk[k] = int(sc.sum())
                test_active[k] = int(np.count_nonzero(sc))
                test_mean_gi[k] = float(out["mean_gi_e"])
                test_input_intensity[k] = float(eval_input_intensity)
                if cfg.input_intensity > 0.0:
                    test_input_factor[k] = float(eval_input_intensity / cfg.input_intensity)
                test_block_idx[k] = block_idx

                if (cfg.record_spikes and k < 25) or (cfg.record_spikes and (k % 1000 == 0)):
                    plot_image(img, label, f"test ep{ep} ex{k}", run_dir / "plots" / f"test_img_ep{ep}_ex{k}.png")
                    plot_raster(
                        out["spike_events_e"],
                        title=f"test raster ep{ep} ex{k}",
                        out_png=run_dir / "plots" / f"test_raster_ep{ep}_ex{k}.png",
                        cfg=cfg,
                    )

                if k == 0 or ((k + 1) % 100 == 0):
                    print(
                        f"  test ex {k+1}/{test_n} sum_spk={test_sum_spk[k]} active={test_active[k]} "
                        f"input={eval_input_intensity:.2f} theta_mean={trainer.theta.mean():.3f}"
                    )

            test_activity_metrics, test_firing_rates = compute_activity_metrics(
                test_result_monitor, test_mean_gi, test_input_factor
            )
            test_activity_metrics.update(
                {"epoch": int(ep), "phase": "test", "inhibition_mode": cfg.inhibition_mode, "aborted": False}
            )
            acc = float((y_true == y_pred).mean())
            test_accuracy_over_epochs.append(acc)
            print(f"Epoch {ep} test accuracy: {acc:.4f}")
            print(
                f"Epoch {ep} train spikes mean={train_sum_spk_used.mean():.2f} min={train_sum_spk_used.min()} "
                f"max={train_sum_spk_used.max()} test spikes mean={test_sum_spk.mean():.2f}"
            )
        else:
            print(
                f"Epoch {ep} aborted before test evaluation. "
                f"train spikes mean={train_sum_spk_used.mean():.2f}"
            )

        with open(run_dir / "logs" / f"test_activity_metrics_ep{ep}.json", "w") as f:
            json.dump(test_activity_metrics, f, indent=2)
        np.save(run_dir / "logs" / f"test_firing_rates_ep{ep}.npy", test_firing_rates)
        np.save(run_dir / "logs" / f"test_sum_spk_ep{ep}.npy", test_sum_spk)
        np.save(run_dir / "logs" / f"test_active_ep{ep}.npy", test_active)
        np.save(run_dir / "logs" / f"test_mean_gi_ep{ep}.npy", test_mean_gi)
        np.save(run_dir / "logs" / f"test_input_intensity_ep{ep}.npy", test_input_intensity)
        np.save(run_dir / "logs" / f"test_input_factor_ep{ep}.npy", test_input_factor)
        np.save(run_dir / "logs" / f"test_block_idx_ep{ep}.npy", test_block_idx)

        train_protocol_records: List[Dict[str, Any]] = []
        test_protocol_records: List[Dict[str, Any]] = []
        if cfg.use_input_intensity_protocol and abort_info is None:
            for block_id in range(protocol_factors.size):
                train_idx = np.where(train_block_idx_used == block_id)[0]
                if train_idx.size > 0:
                    block_metrics, _ = compute_activity_metrics(
                        result_monitor_used[train_idx],
                        train_mean_gi_used[train_idx],
                        train_input_factor_used[train_idx]
                    )
                    block_metrics.update(
                        {
                            "epoch": int(ep),
                            "phase": "train",
                            "block_idx": int(block_id),
                            "example_end": int(ep * train_n + train_idx[-1] + 1),
                            "intensity_factor_nominal": float(protocol_factors[block_id]),
                        }
                    )
                    train_protocol_records.append(block_metrics)

                test_idx = np.where(test_block_idx == block_id)[0]
                if test_idx.size > 0:
                    block_metrics, _ = compute_activity_metrics(
                        test_result_monitor[test_idx],
                        test_mean_gi[test_idx],
                        test_input_factor[test_idx]
                    )
                    block_metrics.update(
                        {
                            "epoch": int(ep),
                            "phase": "test",
                            "block_idx": int(block_id),
                            "example_end": int(ep * test_n + test_idx[-1] + 1),
                            "intensity_factor_nominal": float(protocol_factors[block_id]),
                        }
                    )
                    test_protocol_records.append(block_metrics)

            with open(run_dir / "logs" / f"train_protocol_metrics_ep{ep}.json", "w") as f:
                json.dump(train_protocol_records, f, indent=2)
            with open(run_dir / "logs" / f"test_protocol_metrics_ep{ep}.json", "w") as f:
                json.dump(test_protocol_records, f, indent=2)
            plot_protocol_metric(
                train_protocol_records,
                "mean_e_spikes_per_image",
                "mean E spikes per image",
                run_dir / "plots" / f"train_protocol_mean_spikes_ep{ep}.png",
                f"train activity vs intensity epoch {ep}"
            )
            plot_protocol_metric(
                train_protocol_records,
                "dead_neuron_fraction",
                "dead neuron fraction",
                run_dir / "plots" / f"train_protocol_dead_fraction_ep{ep}.png",
                f"train dead neurons vs intensity epoch {ep}"
            )
            plot_protocol_metric(
                test_protocol_records,
                "mean_e_spikes_per_image",
                "mean E spikes per image",
                run_dir / "plots" / f"test_protocol_mean_spikes_ep{ep}.png",
                f"test activity vs intensity epoch {ep}"
            )

        ai_ae_stats = last_ai_ae_stats
        metrics = {
            "epoch": ep,
            "accuracy": acc,
            "inhibition_mode": cfg.inhibition_mode,
            "istdp_rule": cfg.istdp_rule,
            "ai_ae_mean": ai_ae_stats["mean"],
            "ai_ae_min": ai_ae_stats["min"],
            "ai_ae_max": ai_ae_stats["max"],
            "ai_ae_std": ai_ae_stats["std"],
            "ai_ae_max_fraction": last_ai_ae_max_fraction,
            "theta_mean": last_theta_mean,
            "theta_max": last_theta_max,
            "aborted": abort_info is not None,
            "abort_reason": abort_info["reason"] if abort_info is not None else "",
            "processed_train_examples": int(processed_train_n),
            "retry_cap_hits": int(retry_cap_hits),
            "runaway_detected": bool(runaway_detected),
            "collapse_detected": bool(collapse_detected),
            "istdp_during_rest": bool(cfg.istdp_during_rest),
            "rollback_state_on_retry": bool(cfg.rollback_state_on_retry),
            "normalize_aiae_columns": bool(trainer.normalize_aiae_columns),
            "aiae_target_sum": float(cfg.aiae_target_sum),
            "aiae_normalize_every": int(cfg.aiae_normalize_every),
        }
        metrics.update({f"train_{k}": v for k, v in train_activity_metrics.items()})
        metrics.update({f"test_{k}": v for k, v in test_activity_metrics.items()})
        with open(run_dir / "logs" / f"metrics_ep{ep}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if acc is not None:
            _save_confusion_matrix(
                y_true, y_pred,
                run_dir / "plots" / f"confusion_matrix_ep{ep}.png",
                f"confusion matrix epoch {ep}, acc={acc:.3f}"
            )
            plot_accuracy_over_epochs(test_accuracy_over_epochs, run_dir / "plots" / "accuracy_over_epochs.png")
            plot_rate_histogram(
                test_firing_rates,
                run_dir / "plots" / f"test_firing_rate_distribution_ep{ep}.png",
                f"test E firing-rate distribution ({cfg.inhibition_mode}) epoch {ep}"
            )

        final_run_summary = {
            "mode": cfg.inhibition_mode,
            "istdp_rule": cfg.istdp_rule,
            "eta_ie": cfg.eta_ie,
            "rho_ie": cfg.rho_ie,
            "w_ie_max": cfg.w_ie_max,
            "theta_gate_ref_mV": cfg.theta_gate_ref_mV,
            "theta_gate_scale_mV": cfg.theta_gate_scale_mV,
            "slow_homeostat_target_rate_hz": cfg.slow_homeostat_target_rate_hz,
            "aborted": abort_info is not None,
            "abort_reason": abort_info["reason"] if abort_info is not None else "",
            "accuracy": acc,
            "train_mean_spikes": train_activity_metrics["mean_e_spikes_per_image"],
            "train_usage_entropy_norm": train_activity_metrics["usage_entropy_norm"],
            "dead_fraction": train_activity_metrics["dead_neuron_fraction"],
            "firing_rate_cv": train_activity_metrics["firing_rate_cv"],
            "gi_mean": train_activity_metrics["mean_inhibitory_conductance"],
            "ai_ae_mean": ai_ae_stats["mean"],
            "ai_ae_max": ai_ae_stats["max"],
            "ai_ae_std": ai_ae_stats["std"],
            "theta_mean": last_theta_mean,
            "theta_max": last_theta_max,
            "processed_train_examples": int(processed_train_n),
            "retry_cap_hits": int(retry_cap_hits),
            "runaway_detected": bool(runaway_detected),
            "collapse_detected": bool(collapse_detected),
            "aiae_max_fraction": last_ai_ae_max_fraction,
            "initial_aiae_max_fraction": initial_ai_ae_max_fraction,
            "istdp_during_rest": bool(cfg.istdp_during_rest),
            "rollback_state_on_retry": bool(cfg.rollback_state_on_retry),
            "normalize_aiae_columns": bool(trainer.normalize_aiae_columns),
            "aiae_target_sum": float(cfg.aiae_target_sum),
            "aiae_normalize_every": int(cfg.aiae_normalize_every),
            "metrics_window": int(cfg.metrics_window),
            "train_examples_requested": int(cfg.train_examples),
            "test_examples_requested": int(cfg.test_examples),
            "seed": int(cfg.seed),
            "out_dir": str(run_dir),
        }
        write_run_summary(run_dir, final_run_summary)
        if abort_info is not None:
            write_abort_notes(run_dir, cfg, abort_info, final_run_summary)
            break

    if final_run_summary.get("aborted"):
        print(f"Run aborted. Outputs in {run_dir}")
    else:
        print(f"Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
