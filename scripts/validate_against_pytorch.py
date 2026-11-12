"""Validate FlyLab's fast LIF kernel against the reference PyTorch model.

Imports the model classes from Eon Systems' fly-brain repository
(external/fly-brain/code/run_pytorch.py: TorchModel = Poisson + sparse matmul + AlphaLIF)
and feeds BOTH implementations exactly the same Poisson input spikes, then
compares spike counts neuron by neuron.

Usage: python scripts/validate_against_pytorch.py [duration_s]
"""
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CODE = REPO / "external" / "fly-brain" / "code"

# run_pytorch.py imports helpers from benchmark.py; stub them so we only need the model classes
stub = types.ModuleType("benchmark")
for name in ["T_RUN_VALUES_SEC", "N_RUN_VALUES", "path_comp", "path_con", "path_wt", "get_experiment",
             "get_spike_output_path", "print_summary_table", "save_result_csv", "spike_io_enabled"]:
    setattr(stub, name, None)
sys.modules["benchmark"] = stub
sys.path.insert(0, str(CODE))
import run_pytorch as ref  # noqa: E402

from flylab.brain import DT, Brain, Stimulus  # noqa: E402
from flylab.groups import load_groups  # noqa: E402


def main(duration_s=0.1):
    brain = Brain()
    groups = load_groups()
    sugar = brain.indices(groups["sugar_GRNs"]["root_ids"])
    n_steps = int(round(duration_s * 1000 / DT))

    # identical Poisson input for both models
    rng = np.random.default_rng(0)
    stim = rng.random((1, n_steps, len(sugar)), dtype=np.float32) < (200.0 * DT / 1000.0)

    # ---- reference: Eon Systems TorchModel (dense-state, sparse matmul) ----
    N = brain.N
    pre = np.repeat(np.arange(N), np.diff(brain.indptr.numpy()))
    W = torch.sparse_coo_tensor(
        np.vstack([brain.post.numpy(), pre]), brain.w.numpy() / ref.MODEL_PARAMS["wScale"], (N, N)
    ).coalesce().to_sparse_csr()
    model = ref.TorchModel(1, N, DT, ref.MODEL_PARAMS, W, exc_indices=sugar, device="cpu")
    state = model.state_init()
    step_i = {"t": 0}
    sug_t = torch.tensor(sugar)

    def fake_poisson(rates, generator=None):  # replay our pre-drawn input spikes
        out = torch.zeros(1, N)
        out[0, sug_t] = torch.from_numpy(stim[0, step_i["t"]].astype(np.float32))
        return out * model.poisson.scale

    model.poisson.forward = fake_poisson
    counts_ref = np.zeros(N, np.int64)
    t0 = time.perf_counter()
    with torch.no_grad():
        for t in range(n_steps):
            step_i["t"] = t
            state = model(torch.zeros(1, N), *state)
            counts_ref += state[2][0].numpy().astype(np.int64)
    t_ref = time.perf_counter() - t0

    # ---- FlyLab fast kernel with the same input ----
    from flylab._lif_numba import run_one_trial_sparse
    p = ref.MODEL_PARAMS
    indptr = brain.indptr.numpy()
    post = brain.post.numpy().astype(np.int32)
    w = brain.w.numpy()
    rs = np.full(N, int(round(p["tRefrac"] / DT)), np.int32)
    rs[sugar] = 0
    t0 = time.perf_counter()
    counts_fast = run_one_trial_sparse(
        indptr, post, w, N, n_steps, int(p["tDelay"] / DT) + 1, rs, np.zeros(N, np.bool_),
        np.array(sugar, np.int64), stim[0], np.float32(p["wScale"] * p["scalePoisson"]),
        np.float32(1 - DT / p["tauSyn"]), np.float32(DT / p["tauMem"]), np.float32(p["v0"]),
        np.float32(p["vRest"]), np.float32(p["vThreshold"]), np.float32(p["vReset"]), 1_000_000,
        np.float32(brain.rest_eps),
    )[0]
    t_fast = time.perf_counter() - t0

    diff = np.abs(counts_ref - counts_fast)
    print(f"duration {duration_s}s  reference {t_ref:.1f}s  fast {t_fast:.2f}s")
    print(f"total spikes: reference {counts_ref.sum()}  fast {counts_fast.sum()}")
    print(f"neurons active: reference {(counts_ref > 0).sum()}  fast {(counts_fast > 0).sum()}")
    print(f"neurons with different spike count: {(diff > 0).sum()}  (max diff {diff.max()})")
    mn9 = brain.indices(groups["MN9"]["root_ids"])
    print(f"MN9 spikes: reference {counts_ref[mn9].tolist()}  fast {counts_fast[mn9].tolist()}")
    return counts_ref, counts_fast


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.1)
