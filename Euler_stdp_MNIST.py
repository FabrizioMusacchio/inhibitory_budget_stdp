""" 

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

    # Data loading
    mnist_data_dir: str = "."
    mnist_npz_path: str = "./mnist.npz"
    allow_synthetic_data: bool = False
    train_hard_reset_state: bool = False
    test_hard_reset_state: bool = False

    # Logging and evaluation
    out_dir: str = "./runs/diehl_cook_euler"
    record_spikes: bool = True
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
        self.delay_buf.fill(0)
        self.delay_ptr = 0

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
            for j in post_idx:
                self.post1[j] = 1.0
                self.post2[j] = 1.0

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

            if record_v and v_trace_subset is not None:
                v_trace_subset[t, :] = self.v_e[v_subset_idx]

        # rest period, no input
        for _ in range(self.steps_rest):
            _decay(self.pre_trace, self.alpha_pre)
            _decay(self.post1, self.alpha_post1)
            _decay(self.post2, self.alpha_post2)

            e_spk = _lif_step_e(
                self.v_e, self.ge_e, self.gi_e, self.theta, self.refrac_e,
                cfg.dt_ms, cfg.v_rest_e_mV, cfg.tau_v_e_ms, cfg.v_reset_e_mV,
                cfg.v_thresh_e_base_mV, cfg.offset_mV, cfg.tau_ge_ms, cfg.tau_gi_ms,
                cfg.e_inh_e_mV
            )
            e_idx = np.where(e_spk > 0)[0].astype(np.int32)
            for j in e_idx:
                self.refrac_e[j] = self.refrac_e_steps
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
            if i_idx.size > 0:
                _propagate_sparse(i_idx, self.W["AiAe"], self.gi_e)

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

def weight_statistics(W: np.ndarray) -> Tuple[float, float, float]:
    mean = float(W.mean())
    l1 = float(np.mean(np.sum(np.abs(W), axis=0)))
    l2 = float(np.sqrt(np.mean(W * W)))
    return mean, l1, l2

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
    parser.add_argument("--w-aeai", type=float, default=None)
    parser.add_argument("--w-aiae", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-xeai-input", action="store_true")
    parser.add_argument("--no-delays", action="store_true")
    parser.add_argument("--no-record-spikes", action="store_true")
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
    if args.w_aeai is not None:
        cfg.w_aeai = args.w_aeai
    if args.w_aiae is not None:
        cfg.w_aiae = args.w_aiae
    if args.seed is not None:
        cfg.seed = args.seed
    if args.use_xeai_input:
        cfg.use_xeai_input = True
    if args.no_delays:
        cfg.use_delays = False
    if args.no_record_spikes:
        cfg.record_spikes = False
    if args.train_hard_reset_state:
        cfg.train_hard_reset_state = True
    if args.test_hard_reset_state:
        cfg.test_hard_reset_state = True
    if args.allow_synthetic_data:
        cfg.allow_synthetic_data = True
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

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    (x_train, y_train), (x_test, y_test), source = _load_mnist(cfg)
    x_train = x_train.astype(np.uint8)
    x_test = x_test.astype(np.uint8)
    y_train = y_train.astype(np.int32).reshape(-1)
    y_test = y_test.astype(np.int32).reshape(-1)
    print(f"Loaded MNIST source={source} train={x_train.shape} test={x_test.shape}")

    for k in range(min(5, x_train.shape[0], x_test.shape[0])):
        plot_image(x_train[k], y_train[k], f"train example {k}", run_dir / "plots" / f"train_example_{k}.png")
        plot_image(x_test[k], y_test[k], f"test example {k}", run_dir / "plots" / f"test_example_{k}.png")

    trainer = DiehlCookEuler(cfg)
    normalize_columns_l1(trainer.W["XeAe"], target_sum=78.0)
    weight_stats_epoch = {"mean": [], "l1": [], "l2": []}
    weight_stats_iter = {"iteration": [], "mean": [], "l1": [], "l2": []}
    test_accuracy_over_epochs: List[float] = []

    for ep in range(cfg.epochs):
        print(f"Epoch {ep+1}/{cfg.epochs}")
        train_n = min(cfg.train_examples, x_train.shape[0])
        result_monitor = np.zeros((train_n, cfg.n_e), dtype=np.int32)
        train_labels = np.zeros(train_n, dtype=np.int32)
        train_sum_spk = np.zeros(train_n, dtype=np.int32)
        train_active = np.zeros(train_n, dtype=np.int32)

        input_intensity = cfg.input_intensity
        for k in range(train_n):
            if cfg.train_hard_reset_state:
                trainer.reset_dynamic_state()
            img = x_train[k]
            label = int(y_train[k])
            retries = 0
            while True:
                normalize_columns_l1(trainer.W["XeAe"], target_sum=78.0)
                rec_v = (cfg.record_v_example_every > 0 and (k % cfg.record_v_example_every == 0))
                out = trainer.run_one_example(
                    img=img,
                    label=label,
                    training=True,
                    input_intensity=input_intensity,
                    record_spikes=cfg.record_spikes,
                    record_v=rec_v
                )
                sc = out["spike_count_e"]
                sum_spk = int(sc.sum())
                if sum_spk < 5:
                    retries += 1
                    input_intensity += 1.0
                    continue
                break

            result_monitor[k, :] = sc
            train_labels[k] = label
            train_sum_spk[k] = sum_spk
            train_active[k] = int(np.count_nonzero(sc))

            global_iter = ep * train_n + (k + 1)
            if cfg.weight_stats_every > 0 and ((k + 1) % cfg.weight_stats_every == 0):
                m_i, l1_i, l2_i = weight_statistics(trainer.W["XeAe"])
                weight_stats_iter["iteration"].append(global_iter)
                weight_stats_iter["mean"].append(m_i)
                weight_stats_iter["l1"].append(l1_i)
                weight_stats_iter["l2"].append(l2_i)

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
                    f"retries={retries} theta_mean={trainer.theta.mean():.3f} theta_max={trainer.theta.max():.3f}"
                )

            if cfg.update_interval > 0 and ((k + 1) % cfg.update_interval == 0):
                w0 = k + 1 - cfg.update_interval
                w1 = k + 1
                s = train_sum_spk[w0:w1]
                a = train_active[w0:w1]
                print(
                    f"  train window [{w0}:{w1}] mean_sum={s.mean():.2f} min_sum={s.min()} max_sum={s.max()} "
                    f"mean_active={a.mean():.2f}"
                )

            input_intensity = cfg.input_intensity

        if cfg.weight_stats_every > 0:
            last_iter = ep * train_n + train_n
            if len(weight_stats_iter["iteration"]) == 0 or weight_stats_iter["iteration"][-1] != last_iter:
                m_i, l1_i, l2_i = weight_statistics(trainer.W["XeAe"])
                weight_stats_iter["iteration"].append(last_iter)
                weight_stats_iter["mean"].append(m_i)
                weight_stats_iter["l1"].append(l1_i)
                weight_stats_iter["l2"].append(l2_i)

        assignments = compute_assignments(result_monitor, train_labels, cfg.n_e)
        np.save(run_dir / "logs" / f"assignments_ep{ep}.npy", assignments)
        np.save(run_dir / "logs" / f"train_sum_spk_ep{ep}.npy", train_sum_spk)
        np.save(run_dir / "logs" / f"train_active_ep{ep}.npy", train_active)

        m, l1, l2 = weight_statistics(trainer.W["XeAe"])
        weight_stats_epoch["mean"].append(m)
        weight_stats_epoch["l1"].append(l1)
        weight_stats_epoch["l2"].append(l2)
        np.save(run_dir / "snapshots" / f"W_XeAe_ep{ep}.npy", trainer.W["XeAe"])
        np.save(run_dir / "snapshots" / f"theta_ep{ep}.npy", trainer.theta)
        plot_receptive_fields(trainer.W["XeAe"], cfg, run_dir / "plots" / f"rf_ep{ep}.png", f"receptive fields epoch {ep}")
        plot_weight_stats(weight_stats_epoch, run_dir / "plots" / "weight_stats.png")
        plot_weight_stats_iterations(weight_stats_iter, run_dir / "plots" / "weight_stats_iterations.png")

        test_n = min(cfg.test_examples, x_test.shape[0])
        y_true = np.zeros(test_n, dtype=np.int32)
        y_pred = np.zeros(test_n, dtype=np.int32)
        test_sum_spk = np.zeros(test_n, dtype=np.int32)
        test_active = np.zeros(test_n, dtype=np.int32)

        for k in range(test_n):
            if cfg.test_hard_reset_state:
                trainer.reset_dynamic_state()
            img = x_test[k]
            label = int(y_test[k])
            out = trainer.run_one_example(
                img=img,
                label=label,
                training=False,
                input_intensity=cfg.input_intensity,
                record_spikes=(cfg.record_spikes and k < 25) or (cfg.record_spikes and (k % 1000 == 0)),
                record_v=False
            )
            sc = out["spike_count_e"].astype(np.float32)
            ranking = rank_digits(assignments, sc)
            y_true[k] = label
            y_pred[k] = int(ranking[0])
            test_sum_spk[k] = int(sc.sum())
            test_active[k] = int(np.count_nonzero(sc))

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
                    f"theta_mean={trainer.theta.mean():.3f}"
                )

        acc = float((y_true == y_pred).mean())
        test_accuracy_over_epochs.append(acc)
        print(f"Epoch {ep} test accuracy: {acc:.4f}")
        print(
            f"Epoch {ep} train spikes mean={train_sum_spk.mean():.2f} min={train_sum_spk.min()} max={train_sum_spk.max()} "
            f"test spikes mean={test_sum_spk.mean():.2f}"
        )

        metrics = {
            "epoch": ep,
            "accuracy": acc,
            "train_spike_sum_mean": float(train_sum_spk.mean()),
            "train_spike_sum_min": int(train_sum_spk.min()),
            "train_spike_sum_max": int(train_sum_spk.max()),
            "train_active_mean": float(train_active.mean()),
            "test_spike_sum_mean": float(test_sum_spk.mean()),
            "test_spike_sum_min": int(test_sum_spk.min()),
            "test_spike_sum_max": int(test_sum_spk.max()),
            "test_active_mean": float(test_active.mean()),
        }
        with open(run_dir / "logs" / f"metrics_ep{ep}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        _save_confusion_matrix(
            y_true, y_pred,
            run_dir / "plots" / f"confusion_matrix_ep{ep}.png",
            f"confusion matrix epoch {ep}, acc={acc:.3f}"
        )
        plot_accuracy_over_epochs(test_accuracy_over_epochs, run_dir / "plots" / "accuracy_over_epochs.png")

    print(f"Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
