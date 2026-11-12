"""Whole-brain leaky integrate-and-fire model of the adult Drosophila brain.

This is a faithful port of the model of Shiu et al., Nature 2024
("A Drosophila computational brain model reveals sensorimotor processing"),
following the PyTorch backend in the Eon Systems `fly-brain` repository
(code/run_pytorch.py), which was itself validated against the original Brian2
code (github.com/philshiu/Drosophila_brain_model).

Same equations and parameters as the paper:
    dv/dt = (v_0 - v + g) / t_mbr        (LIF membrane, t_mbr = 20 ms)
    dg/dt = -g / tau                     (synaptic conductance, tau = 5 ms)
    v_th = -45 mV, v_0 = v_rst = -52 mV, refractory 2.2 ms, synaptic delay 1.8 ms
    every synapse: w = 0.275 mV x (signed synapse count from FlyWire v783)
    stimulated neurons receive Poisson input (default 200 Hz) with weight
    250 x 0.275 mV (enough to make them spike) and have no refractory period.
    Silenced neurons have all their outgoing synapses set to 0.

The only difference from run_pytorch.py is *how* the recurrent input is
computed: instead of a dense/sparse matrix product over all 138,639 neurons
every 0.1 ms, we propagate only the spikes that actually happened (event
driven).  This is mathematically identical but ~50x faster on a laptop CPU.
The delay line is a ring buffer instead of torch.roll (also identical).

The wiring is fixed. Nothing here learns.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PATH_COMP = DATA / "Completeness_783.csv"
PATH_CON = DATA / "Connectivity_783.parquet"
PATH_CSR = DATA / "weights_pre_csr.npz"

# Parameters identical to Shiu et al. 2024 / fly-brain run_pytorch.py
PARAMS = {
    "tauSyn": 5.0,  # ms
    "tDelay": 1.8,  # ms
    "v0": -52.0,  # mV
    "vReset": -52.0,  # mV
    "vRest": -52.0,  # mV
    "vThreshold": -45.0,  # mV
    "tauMem": 20.0,  # ms
    "tRefrac": 2.2,  # ms
    "scalePoisson": 250,
    "wScale": 0.275,  # mV per synapse
}
DT = 0.1  # ms


@dataclass
class Stimulus:
    """Poisson drive applied to a set of neurons during [t0, t1) seconds."""

    indices: list[int]
    rate_hz: float = 200.0
    t0: float = 0.0
    t1: float | None = None  # None -> until end


@dataclass
class SimResult:
    duration_s: float
    n_trials: int
    counts: np.ndarray  # (n_trials, N) spike counts
    spike_step: np.ndarray  # (S,) int32 step index of each spike
    spike_trial: np.ndarray  # (S,) int16
    spike_neuron: np.ndarray  # (S,) int32
    wall_s: float
    meta: dict = field(default_factory=dict)

    @property
    def rates(self) -> np.ndarray:
        """Mean firing rate (Hz) per neuron across trials."""
        return self.counts.mean(0) / self.duration_s

    def rate_std(self) -> np.ndarray:
        return (self.counts / self.duration_s).std(0)

    def binned_rates(self, indices, bin_s: float = 0.02) -> np.ndarray:
        """Population mean rate (Hz per neuron, averaged over trials) of `indices` in time bins."""
        n_bins = int(round(self.duration_s / bin_s))
        idx = np.asarray(list(indices), dtype=np.int64)
        if len(idx) == 0:
            return np.zeros(n_bins)
        m = np.isin(self.spike_neuron, idx)
        t = self.spike_step[m] * DT / 1000.0
        b = np.minimum((t / bin_s).astype(int), n_bins - 1)
        h = np.bincount(b, minlength=n_bins).astype(float)
        return h / (len(idx) * self.n_trials * bin_s)


class Brain:
    """Loads the FlyWire v783 connectome once and runs LIF simulations."""

    rest_eps = 1e-4  # mV; see _lif_numba.run_one_trial_sparse
    workers = max(1, (os.cpu_count() or 4) - 2)

    def __init__(self, device: str | None = None, threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        t = time.perf_counter()
        comp = pd.read_csv(PATH_COMP, index_col=0)
        self.root_ids = comp.index.to_numpy(np.int64)
        self.N = len(self.root_ids)
        self.id2i = {int(r): i for i, r in enumerate(self.root_ids)}
        indptr, post, w = self._load_csr()
        self.indptr = torch.from_numpy(indptr).to(self.device)
        self.post = torch.from_numpy(post).to(self.device)
        self.w = torch.from_numpy(w).to(self.device)  # already x wScale (mV)
        self.n_synapse_pairs = len(post)
        self.load_s = time.perf_counter() - t

    def _load_csr(self):
        if PATH_CSR.exists():
            z = np.load(PATH_CSR)
            return z["indptr"], z["post"], z["w"]
        df = pd.read_parquet(PATH_CON, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        pre = df["Presynaptic_Index"].to_numpy(np.int64)
        post = df["Postsynaptic_Index"].to_numpy(np.int64)
        w = df["Excitatory x Connectivity"].to_numpy(np.float32) * PARAMS["wScale"]
        order = np.argsort(pre, kind="stable")
        pre, post, w = pre[order], post[order], w[order]
        indptr = np.zeros(self.N + 1, dtype=np.int64)
        np.cumsum(np.bincount(pre, minlength=self.N), out=indptr[1:])
        np.savez(PATH_CSR, indptr=indptr, post=post, w=w)
        return indptr, post, w

    def indices(self, root_ids) -> list[int]:
        return [self.id2i[int(r)] for r in root_ids if int(r) in self.id2i]

    def run(
        self,
        stimuli: list[Stimulus],
        silence: list[int] | None = None,
        duration_s: float = 0.5,
        n_trials: int = 8,
        seed: int = 0,
        backend: str = "numba",
        progress: bool = False,
    ) -> SimResult:
        """Simulate `n_trials` independent trials. backend: 'numba' (fast) or 'torch' (reference)."""
        if backend == "torch":
            return self.run_torch(stimuli, silence, duration_s, n_trials, seed, progress)
        return self.run_numba(stimuli, silence, duration_s, n_trials, seed)

    def run_numba(self, stimuli, silence=None, duration_s=0.5, n_trials=8, seed=0, max_events=3_000_000, dense=False):
        from ._lif_numba import run_one_trial_sparse, run_trials

        p, N, B = PARAMS, self.N, n_trials
        n_steps = int(round(duration_s * 1000.0 / DT))
        if not hasattr(self, "_np_csr"):
            self._np_csr = (self.indptr.cpu().numpy(), self.post.cpu().numpy().astype(np.int32), self.w.cpu().numpy())
        indptr, post, w = self._np_csr
        stim_idx = np.array(sorted({i for s in stimuli for i in s.indices}), dtype=np.int64)
        pos = {int(n): k for k, n in enumerate(stim_idx)}
        rate = np.zeros((n_steps, len(stim_idx)), np.float32)  # Hz per step & stimulated neuron
        for s in stimuli:
            a = int(round(s.t0 * 1000 / DT))
            b = n_steps if s.t1 is None else min(n_steps, int(round(s.t1 * 1000 / DT)))
            ks = [pos[i] for i in s.indices]
            rate[a:b, ks] = np.maximum(rate[a:b, ks], s.rate_hz)
        rng = np.random.default_rng(seed)
        stim_spikes = rng.random((B, n_steps, len(stim_idx)), dtype=np.float32) < (rate * DT / 1000.0)[None]
        refrac_steps = np.full(N, int(round(p["tRefrac"] / DT)), np.int32)
        refrac_steps[stim_idx] = 0
        silenced = np.zeros(N, np.bool_)
        if silence:
            silenced[np.asarray(sorted(set(silence)), dtype=np.int64)] = True
        L = int(p["tDelay"] / DT) + 1
        t0 = time.perf_counter()
        args = (
            indptr, post, w, N, n_steps, L, refrac_steps, silenced, stim_idx, stim_spikes,
            np.float32(p["wScale"] * p["scalePoisson"]), np.float32(1.0 - DT / p["tauSyn"]),
            np.float32(DT / p["tauMem"]), np.float32(p["v0"]), np.float32(p["vRest"]),
            np.float32(p["vThreshold"]), np.float32(p["vReset"]), max_events,
        )
        if dense:
            counts, ev_step, ev_neu, n_ev = run_trials(*args)
            steps = [ev_step[b, : n_ev[b]] for b in range(B)]
            neus = [ev_neu[b, : n_ev[b]] for b in range(B)]
            mean_active = float(N)
        else:
            from concurrent.futures import ThreadPoolExecutor

            def one(b):
                a = list(args)
                a[9] = stim_spikes[b]
                return run_one_trial_sparse(*a, np.float32(self.rest_eps))

            with ThreadPoolExecutor(max_workers=min(B, self.workers)) as ex:
                outs = list(ex.map(one, range(B)))
            counts = np.stack([o[0] for o in outs])
            steps = [o[1] for o in outs]
            neus = [o[2] for o in outs]
            n_ev = np.array([o[3] for o in outs])
            mean_active = float(np.mean([o[4] for o in outs]))
        wall = time.perf_counter() - t0
        trials = np.concatenate([np.full(len(steps[b]), b, np.int16) for b in range(B)])
        steps = np.concatenate(steps)
        neus = np.concatenate(neus)
        return SimResult(
            duration_s=duration_s, n_trials=B, counts=counts, spike_step=steps, spike_trial=trials,
            spike_neuron=neus, wall_s=wall,
            meta={"backend": "numba-dense" if dense else "numba-active-set", "n_steps": n_steps, "dt_ms": DT, "seed": seed,
                  "events_truncated": bool((n_ev >= max_events).any()),
                  "mean_active_set": mean_active},
        )

    @torch.no_grad()
    def run_torch(self, stimuli, silence=None, duration_s=0.5, n_trials=4, seed=0, progress=False) -> SimResult:
        p = PARAMS
        dev, N, B = self.device, self.N, n_trials
        n_steps = int(round(duration_s * 1000.0 / DT))
        gen = torch.Generator(device=dev).manual_seed(seed)

        # --- stimulus schedule -------------------------------------------------
        stim_idx = sorted({i for s in stimuli for i in s.indices})
        sidx = torch.tensor(stim_idx, dtype=torch.long, device=dev)
        pos = {n: k for k, n in enumerate(stim_idx)}
        # boundaries where the rate vector changes
        segs = []
        for s in stimuli:
            a = int(round(s.t0 * 1000 / DT))
            b = n_steps if s.t1 is None else int(round(s.t1 * 1000 / DT))
            segs.append((a, b, [pos[i] for i in s.indices], s.rate_hz))
        bounds = sorted({0, n_steps, *[x for a, b, _, _ in segs for x in (a, b)]})

        def rate_vec(step):
            r = torch.zeros(len(stim_idx), device=dev)
            for a, b, ks, hz in segs:
                if a <= step < b:
                    r[ks] = torch.maximum(r[ks], torch.tensor(hz, device=dev))
            return r * (DT / 1000.0)

        # --- silencing: drop outgoing spikes of these neurons -------------------
        silenced = torch.zeros(N, dtype=torch.bool, device=dev)
        if silence:
            silenced[torch.tensor(sorted(set(silence)), device=dev)] = True

        # --- state --------------------------------------------------------------
        v = torch.full((B, N), p["v0"], device=dev)
        g = torch.zeros((B, N), device=dev)
        spikes = torch.zeros((B, N), dtype=torch.bool, device=dev)
        L = int(p["tDelay"] / DT) + 1  # delay-line length (same as run_pytorch.py)
        buf = torch.zeros((L, B, N), device=dev)
        refrac_steps = torch.full((N,), int(round(p["tRefrac"] / DT)), dtype=torch.int32, device=dev)
        if len(stim_idx):
            refrac_steps[sidx] = 0
        refrac = refrac_steps.unsqueeze(0).repeat(B, 1).clone()
        syn_decay = 1.0 - DT / p["tauSyn"]
        mem = DT / p["tauMem"]
        kick = p["wScale"] * p["scalePoisson"]
        counts = torch.zeros((B, N), dtype=torch.int32, device=dev)
        ev_step, ev_trial, ev_neu = [], [], []
        rec = torch.zeros(B * N, device=dev)

        t_start = time.perf_counter()
        rv = rate_vec(0)
        for step in range(n_steps):
            if step in bounds:
                rv = rate_vec(step)
            # 1) recurrent input from last step's spikes (event driven)
            rec.zero_()
            nz = spikes.nonzero()
            if nz.numel():
                if silence:
                    nz = nz[~silenced[nz[:, 1]]]
                if nz.numel():
                    b_i, pre = nz[:, 0], nz[:, 1]
                    starts = self.indptr[pre]
                    cnt = self.indptr[pre + 1] - starts
                    tot = int(cnt.sum())
                    if tot:
                        rep_b = torch.repeat_interleave(b_i, cnt)
                        base = torch.repeat_interleave(starts - (torch.cumsum(cnt, 0) - cnt), cnt)
                        e = base + torch.arange(tot, device=dev)
                        rec.index_add_(0, rep_b * N + self.post[e], self.w[e])
            # 2) refractory counter
            refrac = torch.where(spikes, torch.zeros_like(refrac), refrac + 1)
            # 3) synapse: delayed input gated by refractoriness
            slot = step % L
            g_new = g * syn_decay + buf[slot] * (refrac >= refrac_steps).float()
            buf[slot] = rec.view(B, N)
            # 4) membrane (uses the old conductance, as in run_pytorch.py)
            if len(stim_idx):
                pois = torch.bernoulli(rv.expand(B, -1), generator=gen) * kick
                v[:, sidx] += pois
            v += mem * (g - (v - p["vRest"]))
            spikes = v > p["vThreshold"]
            v = torch.where(spikes, torch.full_like(v, p["vReset"]), v)
            g = torch.where(spikes, torch.zeros_like(g_new), g_new)
            # 5) record
            if spikes.any():
                counts += spikes.int()
                nz2 = spikes.nonzero()
                ev_trial.append(nz2[:, 0].to(torch.int16).cpu())
                ev_neu.append(nz2[:, 1].to(torch.int32).cpu())
                ev_step.append(torch.full((len(nz2),), step, dtype=torch.int32))
            if progress and step % 1000 == 0:
                print(f"  step {step}/{n_steps}  {time.perf_counter() - t_start:.1f}s", flush=True)

        wall = time.perf_counter() - t_start
        cat = lambda xs, dt: torch.cat(xs).numpy() if xs else np.zeros(0, dt)
        return SimResult(
            duration_s=duration_s,
            n_trials=B,
            counts=counts.cpu().numpy(),
            spike_step=cat(ev_step, np.int32),
            spike_trial=cat(ev_trial, np.int16),
            spike_neuron=cat(ev_neu, np.int32),
            wall_s=wall,
            meta={"backend": "torch", "device": str(dev), "n_steps": n_steps, "dt_ms": DT, "seed": seed},
        )


class BrainStepper:
    """One trial of the whole-brain model advanced chunk by chunk (closed loop with a simulated world).

    `inputs` = {name: [neuron indices]}: the sensory/command populations whose Poisson rate the world sets
    before every chunk (e.g. left/right LPLC1 from the car's eyes). Same equations, parameters and kernel
    as Brain.run_numba (active-set), only the state is kept between calls."""

    def __init__(self, brain: Brain, inputs: dict[str, list[int]], silence=None, seed: int = 0,
                 max_events: int = 2_000_000, kernel: str = "dense_par"):
        p, N = PARAMS, brain.N
        if not hasattr(brain, "_np_csr"):
            brain._np_csr = (brain.indptr.cpu().numpy(), brain.post.cpu().numpy().astype(np.int32), brain.w.cpu().numpy())
        self.b = brain
        self.names = list(inputs)
        self.stim_idx = np.array(sorted({i for v in inputs.values() for i in v}), dtype=np.int64)
        pos = {int(n): k for k, n in enumerate(self.stim_idx)}
        self.cols = {k: np.array([pos[i] for i in v], dtype=np.int64) for k, v in inputs.items()}
        self.refrac_steps = np.full(N, int(round(p["tRefrac"] / DT)), np.int32)
        self.refrac_steps[self.stim_idx] = 0
        self.silenced = np.zeros(N, np.bool_)
        if silence:
            self.silenced[np.asarray(sorted(set(silence)), dtype=np.int64)] = True
        self.L = int(p["tDelay"] / DT) + 1
        self.consts = (np.float32(p["wScale"] * p["scalePoisson"]), np.float32(1.0 - DT / p["tauSyn"]),
                       np.float32(DT / p["tauMem"]), np.float32(p["v0"]), np.float32(p["vRest"]),
                       np.float32(p["vThreshold"]), np.float32(p["vReset"]))
        L = self.L
        self.state = (np.full(N, -1, np.int32), np.zeros(N, np.int32), np.zeros(N, np.float32),
                      np.zeros(N, np.float32), np.zeros(N, np.float32), np.zeros(N, np.float32),
                      np.zeros(N, np.int32), np.zeros(N, np.int32), np.zeros(N, np.int32),
                      np.zeros(N, np.bool_), np.zeros((N, L), np.float32), np.zeros(N, np.int32),
                      np.zeros(N, np.int32), np.zeros(2, np.int64))
        # dense_par: multithreaded dense kernel (fast when many neurons are active, e.g. visual input);
        # sparse: single-threaded active-set kernel (fast when activity is local)
        self.kernel = kernel
        self.dstate = (np.full(N, p["v0"], np.float32), np.zeros(N, np.float32), np.zeros((L, N), np.float32),
                       np.zeros(N, np.float32), self.refrac_steps.copy(), np.zeros(N, np.bool_),
                       np.zeros(N, np.int32), np.zeros(1, np.int64))
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.max_events = max_events

    def step(self, rates_hz: dict[str, float], n_steps: int, stim_spikes: np.ndarray | None = None):
        """Advance n_steps (0.1 ms each) with the given Poisson rate per input population.
        Returns (counts per neuron in this chunk, spike steps, spike neurons)."""
        from ._lif_numba import run_chunk_dense_par, run_chunk_sparse

        if stim_spikes is None:
            rate = np.zeros(len(self.stim_idx), np.float32)
            for k, r in rates_hz.items():
                rate[self.cols[k]] = np.maximum(rate[self.cols[k]], r)
            stim_spikes = self.rng.random((n_steps, len(self.stim_idx)), dtype=np.float32) < (rate * DT / 1000.0)[None]
        indptr, post, w = self.b._np_csr
        if self.kernel == "dense_par":
            kick, syn_decay, mem, _v0, v_rest, v_th, v_reset = self.consts
            counts, ev_s, ev_n = run_chunk_dense_par(
                indptr, post, w, self.b.N, n_steps, self.L, self.t, self.refrac_steps, self.silenced, self.stim_idx,
                stim_spikes, kick, syn_decay, mem, v_rest, v_th, v_reset, self.max_events, *self.dstate)
            self.t += n_steps
            return counts, ev_s, ev_n
        counts, ev_s, ev_n = run_chunk_sparse(
            indptr, post, w, self.b.N, self.t, n_steps, self.L, self.refrac_steps, self.silenced, self.stim_idx,
            stim_spikes, *self.consts, self.max_events, np.float32(self.b.rest_eps), *self.state)
        self.t += n_steps
        return counts, ev_s, ev_n


_BRAIN: Brain | None = None


def get_brain() -> Brain:
    """Process-wide singleton (loading takes a few seconds)."""
    global _BRAIN
    if _BRAIN is None:
        _BRAIN = Brain()
    return _BRAIN
