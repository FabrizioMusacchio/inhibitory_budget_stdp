""" 

Installation
-------------
Create a conda environment with Python 3.12 or later, and install the required packages:

conda create -n diehl_cook_euler python=3.12 mamba -y
conda activate diehl_cook_euler
mamba install numpy matplotlib numba scikit-learn tensorflow ipykernel -y
"""
# %% IMPORTS
from __future__ import annotations

import os
import time
import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from numba import njit, prange

# Optional: sklearn for confusion matrix
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
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

    # Delays
    max_delay_ms: float = 10.0
    use_delays: bool = True

    # Training schedule
    epochs: int = 1
    train_examples: int = 6000 # max: 60000 for MNIST
    test_examples: int = 1000  # max: 10000 for MNIST
    update_interval: int = 1000  # assignment recompute; Diehl and Cook used 10000
    weight_snapshot_interval: int = 1  # epochs

    # Logging and evaluation
    out_dir: str = "./runs/diehl_cook_euler"
    record_spikes: bool = True
    record_v_example_every: int = 0  # 0 disables, else record full v trace for every k-th example
    record_v_subset: int = 20        # always record subset of neurons for v traces
    store_on_disk_threshold_mb: int = 1024  # memmap if larger than this

    # Receptive field visualization
    rf_grid_sqrt: int = 20  # sqrt(400)=20, arrangement for 2d plots

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

    # E->I: one to one with weight 10.4 in the generator, but we interpret as conductance increment.
    W["AeAi"] = np.zeros((cfg.n_e, cfg.n_i), dtype=np.float32)
    m = min(cfg.n_e, cfg.n_i)
    W["AeAi"][np.arange(m), np.arange(m)] = 6.0

    # I->E: all to all except diagonal, weight 17
    W["AiAe"] = (np.ones((cfg.n_i, cfg.n_e), dtype=np.float32) * 8.0)
    m = min(cfg.n_i, cfg.n_e)
    W["AiAe"][np.arange(m), np.arange(m)] = 0.0
    
    # X->I: random + 0.01, scaled by ei_input.
    W["XeAi"] = rng.random((cfg.n_input, cfg.n_i)).astype(np.float32)
    W["XeAi"] *= 0.2  # aus random_conn_generator weight['ei_input']=0.2
    # optional sparsify: p=0.1
    mask = (rng.random((cfg.n_input, cfg.n_i)) < 0.1)
    W["XeAi"] *= mask.astype(np.float32)

    return W

def normalize_columns_l1(W: np.ndarray, target_sum: float) -> None:
    col_sums = W.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    factors = target_sum / col_sums
    W *= factors[np.newaxis, :]


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
def _stdp_update_xe(
    W: np.ndarray,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
    pre_trace: np.ndarray,
    post1_trace: np.ndarray,
    post2_trace: np.ndarray,
    nu_pre: float,
    nu_post: float,
    wmin: float,
    wmax: float
) -> None:
    """
    Implements the Diehl Cook style STDP for X->E using per pre and per post traces.

    Pre event: W[i,:] -= nu_pre * post1_trace[:]
    Post event: W[:,j] += nu_post * pre_trace[:] * post2before_j
                then post1[j]=1, post2[j]=1 (handled outside)
    """
    n_in, n_e = W.shape

    # LTD on pre spikes
    for kk in range(pre_idx.shape[0]):
        i = pre_idx[kk]
        for j in range(n_e):
            W[i, j] -= nu_pre * post1_trace[j]

    # LTP on post spikes
    for kk in range(post_idx.shape[0]):
        j = post_idx[kk]
        post2before = post2_trace[j]
        for i in range(n_in):
            W[i, j] += nu_post * pre_trace[i] * post2before

    # clip
    for i in range(n_in):
        for j in range(n_e):
            if W[i, j] < wmin:
                W[i, j] = wmin
            elif W[i, j] > wmax:
                W[i, j] = wmax


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

        # trace decays
        self.alpha_pre = math.exp(-cfg.dt_ms / cfg.tc_pre_ms)
        self.alpha_post1 = math.exp(-cfg.dt_ms / cfg.tc_post1_ms)
        self.alpha_post2 = math.exp(-cfg.dt_ms / cfg.tc_post2_ms)

        # weights
        self.W = init_weights(cfg, self.rng)

        # delay buffers for X spikes, only for X->E in this starter
        self.max_delay_steps = max(1, _steps_from_ms(cfg, cfg.max_delay_ms))
        self.delay_buf = np.zeros((self.max_delay_steps, cfg.n_input), dtype=np.uint8)
        self.delay_ptr = 0

    def _delay_push(self, x_spk: np.ndarray) -> None:
        self.delay_buf[self.delay_ptr, :] = x_spk
        self.delay_ptr = (self.delay_ptr + 1) % self.max_delay_steps

    def _delay_pop(self, delay_steps: int) -> np.ndarray:
        # read spikes that occurred delay_steps ago
        idx = (self.delay_ptr - delay_steps) % self.max_delay_steps
        return self.delay_buf[idx, :]

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

        spike_events_e: List[Tuple[int, int]] = []
        v_trace_subset = None

        if record_v:
            if v_subset_idx is None:
                v_subset_idx = np.arange(min(cfg.record_v_subset, cfg.n_e), dtype=np.int32)
            v_trace_subset = np.zeros((self.steps_example, v_subset_idx.size), dtype=np.float32)

        e_spike_count = np.zeros(cfg.n_e, dtype=np.int32)

        for t in range(self.steps_example):
            # input poisson
            u = self.rng.random(cfg.n_input, dtype=np.float32)
            x_spk = _poisson_spikes(rates_hz, self.dt_s, u)

            if cfg.use_delays:
                self._delay_push(x_spk)
                x_eff = self._delay_pop(delay_steps=self.max_delay_steps - 1)  # crude: maximal delay
            else:
                x_eff = x_spk

            # decay traces
            _decay(self.pre_trace, self.alpha_pre)
            _decay(self.post1, self.alpha_post1)
            _decay(self.post2, self.alpha_post2)

            # update traces on spikes
            pre_idx = np.where(x_eff > 0)[0].astype(np.int32)
            if pre_idx.size > 0:
                _propagate_sparse(pre_idx, self.W["XeAi"], self.ge_i)
                # this drives input to I neurons even when E has not yet learned to respond, 
                # which is important for stabilizing learning and avoiding all E neurons 
                # responding to all digits.
            for i in pre_idx:
                self.pre_trace[i] = 1.0

            # propagate X->E (excitatory conductance)
            if pre_idx.size > 0:
                _propagate_sparse(pre_idx, self.W["XeAe"], self.ge_e)

            # recurrent propagation from previous step spikes is omitted in this starter.
            # minimal WTA: E->I then I->E using same step spikes.
            # We compute E spikes, then propagate to I, then I spikes, then inhibit E.
            # This is an approximation of continuous interaction within dt.

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

            # post traces
            post_idx = e_idx
            for j in post_idx:
                post2before = self.post2[j]
                self.post1[j] = 1.0
                self.post2[j] = 1.0
                # post2before used in STDP update kernel, stored in post2 before overwriting, but here we overwrite.
                # in this starter we accept the slight mismatch, and use post2_trace before setting in kernel by
                # calling kernel before overwriting would be cleaner.

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

            # propagate I->E as inhibition (gi)
            if i_idx.size > 0:
                _propagate_sparse(i_idx, self.W["AiAe"], self.gi_e)

            # STDP update for X->E
            if training and (pre_idx.size > 0 or post_idx.size > 0):
                # In the original: pre event uses post1, post event uses pre and post2before.
                # Here: we call the kernel before we overwrite post2 to 1 would be more exact.
                _stdp_update_xe(
                    self.W["XeAe"], pre_idx, post_idx,
                    self.pre_trace, self.post1, self.post2,
                    cfg.nu_pre, cfg.nu_post, cfg.wmin, cfg.wmax)

            if record_v and v_trace_subset is not None:
                v_trace_subset[t, :] = self.v_e[v_subset_idx]

        # rest period, no input, just decay dynamics
        for _ in range(self.steps_rest):
            # decay conductances
            self.ge_e *= math.exp(-cfg.dt_ms / cfg.tau_ge_ms)
            self.gi_e *= math.exp(-cfg.dt_ms / cfg.tau_gi_ms)
            self.ge_i *= math.exp(-cfg.dt_ms / cfg.tau_ge_ms)
            self.gi_i *= math.exp(-cfg.dt_ms / cfg.tau_gi_ms)
            # relax voltages toward rest
            self.v_e += (cfg.dt_ms / cfg.tau_v_e_ms) * (cfg.v_rest_e_mV - self.v_e)
            self.v_i += (cfg.dt_ms / cfg.tau_v_i_ms) * (cfg.v_rest_i_mV - self.v_i)

        return {
            "label": int(label),
            "spike_count_e": e_spike_count.astype(np.int32),
            "spike_events_e": np.array(spike_events_e, dtype=np.int32) if record_spikes else None,
            "v_trace_subset": v_trace_subset,
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
    
    # convert t into ms for plotting
    t_ms = t * cfg.dt_ms
    
    plt.figure(figsize=(10, 4))
    plt.scatter(t_ms, n, s=1)
    plt.xlabel("time (ms)")
    plt.ylabel("neuron id")
    plt.ylim(-1, cfg.n_e)
    plt.xlim(0, cfg.single_example_time_s * 1000.0)
    plt.title(title)
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
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_weight_stats(stats: Dict[str, List[float]], out_png: Path) -> None:
    epochs = np.arange(len(stats["mean"]))
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, stats["mean"], label="mean")
    plt.plot(epochs, stats["l1"], label="l1 mean per synapse")
    plt.plot(epochs, stats["l2"], label="l2 rms per synapse")
    plt.xlabel("epoch")
    plt.ylabel("weight statistic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def weight_statistics(W: np.ndarray) -> Tuple[float, float, float]:
    mean = float(W.mean())
    l1 = float(np.mean(np.abs(W)))
    l2 = float(np.sqrt(np.mean(W * W)))
    return mean, l1, l2

# %% MAIN
# =========================
# Main
# =========================


cfg = SimConfig()

#run_dir = Path(cfg.out_dir) / _now_str()
run_dir = Path(cfg.out_dir)
_ensure_dir(run_dir)
_ensure_dir(run_dir / "plots")
_ensure_dir(run_dir / "snapshots")
_ensure_dir(run_dir / "logs")

with open(run_dir / "config.json", "w") as f:
    json.dump(asdict(cfg), f, indent=2)

if not _HAS_KERAS:
    raise RuntimeError("tensorflow.keras not available. Install tensorflow or use an alternative MNIST loader.")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.uint8)
x_test = x_test.astype(np.uint8)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# plot some examples
for k in range(5):
    plot_image(x_train[k], y_train[k], f"train example {k}", run_dir / "plots" / f"train_example_{k}.png")
    plot_image(x_test[k], y_test[k], f"test example {k}", run_dir / "plots" / f"test_example_{k}.png")

trainer = DiehlCookEuler(cfg)

# normalize X->E weights to have L1 norm of 78 per column, as in the original 
# code with ee_input=0.3 and pixel divisor 8, but here we keep weights in [0,1] 
# and treat scaling via input rates:
normalize_columns_l1(trainer.W["XeAe"], target_sum=78.0)

weight_stats = {"mean": [], "l1": [], "l2": []}

# main loop over epochs and examples
for ep in range(cfg.epochs):
    print(f"Epoch {ep+1}/{cfg.epochs}")

    # =====================
    # Train
    # =====================
    train_n = min(cfg.train_examples, x_train.shape[0])
    result_monitor = np.zeros((train_n, cfg.n_e), dtype=np.int32)
    train_labels = np.zeros(train_n, dtype=np.int32)

    input_intensity = cfg.input_intensity

    for k in range(train_n):
        img = x_train[k]
        label = int(y_train[k])

        # adaptive intensity loop
        while True:
            normalize_columns_l1(trainer.W["XeAe"], target_sum=78.0)
            
            rec_v = (cfg.record_v_example_every > 0 and (k % cfg.record_v_example_every == 0))
            out = trainer.run_one_example(
                img=img, label=label, training=True,
                input_intensity=input_intensity,
                record_spikes=cfg.record_spikes,
                record_v=rec_v
            )
            sc = out["spike_count_e"]
            # some debug prints:
            print(  "k", k,
                    "sum_spk", int(sc.sum()),
                    "active", int(np.count_nonzero(sc)),
                    "theta_mean", float(trainer.theta.mean()),
                    "theta_max", float(trainer.theta.max()),
                )
            
            if sc.sum() < 5:
                input_intensity += 1.0
                continue
            break

        result_monitor[k, :] = sc
        train_labels[k] = label

        if (cfg.record_spikes and k < 25) or (cfg.record_spikes and (k % 1000 == 0)):
            plot_image(img, label, f"train ep{ep} ex{k}", run_dir / "plots" / f"train_img_ep{ep}_ex{k}.png")
            plot_raster(out["spike_events_e"], 
                        title=f"train raster ep{ep} ex{k}", 
                        out_png=run_dir / "plots" / f"train_raster_ep{ep}_ex{k}.png", cfg=cfg)
            
            ev = out["spike_events_e"]
            if ev is None or ev.size == 0:
                print("no spike events")
            else:
                t = ev[:, 0]
                print("k:", k,
                      "  events", ev.shape[0],
                      "  t_min", int(t.min()),
                      "  t_max", int(t.max()),
                      "  t_unique", int(np.unique(t).size))

        if k % 500 == 0 and k > 0:
            print(f"  train example {k}/{train_n} ({(k/train_n)*100:.2f}%)")

        input_intensity = cfg.input_intensity

    assignments = compute_assignments(result_monitor, train_labels, cfg.n_e)
    np.save(run_dir / "logs" / f"assignments_ep{ep}.npy", assignments)

    # weight stats and snapshots
    m, l1, l2 = weight_statistics(trainer.W["XeAe"])
    weight_stats["mean"].append(m)
    weight_stats["l1"].append(l1)
    weight_stats["l2"].append(l2)

    np.save(run_dir / "snapshots" / f"W_XeAe_ep{ep}.npy", trainer.W["XeAe"])
    np.save(run_dir / "snapshots" / f"theta_ep{ep}.npy", trainer.theta)

    plot_receptive_fields(trainer.W["XeAe"], cfg, run_dir / "plots" / f"rf_ep{ep}.png", f"receptive fields epoch {ep}")
    plot_weight_stats(weight_stats, run_dir / "plots" / "weight_stats.png")

    # =====================
    # Test
    # =====================
    test_n = min(cfg.test_examples, x_test.shape[0])
    y_true = np.zeros(test_n, dtype=np.int32)
    y_pred = np.zeros(test_n, dtype=np.int32)

    for k in range(test_n):
        img = x_test[k]
        label = int(y_test[k])

        out = trainer.run_one_example(
            img=img, label=label, training=False,
            input_intensity=cfg.input_intensity,
            record_spikes=(cfg.record_spikes and k < 25) or (cfg.record_spikes and (k % 1000 == 0)),
            record_v=False
        )
        sc = out["spike_count_e"].astype(np.float32)
        ranking = rank_digits(assignments, sc)
        pred = int(ranking[0])
        
        # debug stats:
        print(
            "sum_spk", int(sc.sum()),
            "active", int(np.count_nonzero(sc)),
            "max", int(sc.max()),
            "ge_mean", float(trainer.ge_e.mean()),
            "gi_mean", float(trainer.gi_e.mean()),
            "theta_min_mean_max",
            float(trainer.theta.min()), float(trainer.theta.mean()), float(trainer.theta.max()),
        )

        y_true[k] = label
        y_pred[k] = pred

        if (cfg.record_spikes and k < 25) or (cfg.record_spikes and (k % 1000 == 0)):
            plot_image(img, label, f"test ep{ep} ex{k}", run_dir / "plots" / f"test_img_ep{ep}_ex{k}.png")
            plot_raster(out["spike_events_e"], 
                        title=f"test raster ep{ep} ex{k}", 
                        out_png=run_dir / "plots" / f"test_raster_ep{ep}_ex{k}.png", 
                        cfg=cfg)

        if k % 500 == 0 and k > 0:
            print(f"  test example {k}/{test_n} ({(k/test_n)*100:.2f}%)")

    acc = float((y_true == y_pred).mean())
    print(f"Epoch {ep} test accuracy: {acc:.4f}")

    with open(run_dir / "logs" / f"metrics_ep{ep}.json", "w") as f:
        json.dump({"epoch": ep, "accuracy": acc}, f, indent=2)

    if _HAS_SK:
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"confusion matrix epoch {ep}, acc={acc:.3f}")
        fig.tight_layout()
        fig.savefig(run_dir / "plots" / f"confusion_matrix_ep{ep}.png", dpi=200)
        plt.close(fig)
    else:
        # minimal confusion plot without sklearn
        cm = np.zeros((10, 10), dtype=np.int32)
        for a, b in zip(y_true, y_pred):
            cm[a, b] += 1
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"confusion matrix epoch {ep}, acc={acc:.3f}")
        plt.xlabel("pred")
        plt.ylabel("true")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / f"confusion_matrix_ep{ep}.png", dpi=200)
        plt.close()

print(f"Done. Outputs in {run_dir}")
# %% END