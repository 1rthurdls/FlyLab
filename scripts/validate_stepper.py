"""Check that BrainStepper (chunked, closed-loop kernel) is spike-identical to the validated one-shot kernel.

Same input spikes (P9 + left LPLC1, 0.5 s) -> run once with run_one_trial_sparse, and again as 25 chunks of
20 ms with BrainStepper (both kernels: active-set 'sparse' and multithreaded 'dense_par'). Prints spike counts and the number of neurons whose spike trains differ (expect 0).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab import lab  # noqa: E402
from flylab._lif_numba import run_one_trial_sparse  # noqa: E402
from flylab.brain import DT, PARAMS, BrainStepper, get_brain  # noqa: E402

B = get_brain()
p9, _ = lab.resolve("P9")
lp, _ = lab.resolve("LPLC1_L")
st = BrainStepper(B, {"P9": p9, "LPLC1_L": lp})
n_steps, chunk = 5000, 200
rate = np.zeros(len(st.stim_idx), np.float32)
rate[st.cols["P9"]] = 100.0
rate[st.cols["LPLC1_L"]] = 50.0
spk = np.random.default_rng(3).random((n_steps, len(st.stim_idx)), dtype=np.float32) < (rate * DT / 1000)[None]

indptr, post, w = B._np_csr
c_ref, s_ref, n_ref, _, _ = run_one_trial_sparse(
    indptr, post, w, B.N, n_steps, st.L, st.refrac_steps, st.silenced, st.stim_idx, spk, *st.consts,
    3_000_000, np.float32(B.rest_eps))
import time  # noqa: E402

ok = True
for kernel in ("sparse", "dense_par"):
    st = BrainStepper(B, {"P9": p9, "LPLC1_L": lp}, kernel=kernel)
    S, Nn = [], []
    t0 = time.perf_counter()
    for k in range(0, n_steps, chunk):
        _, s, n = st.step({}, chunk, stim_spikes=spk[k:k + chunk])
        S.append(s); Nn.append(n)
    dt = time.perf_counter() - t0
    S, Nn = np.concatenate(S), np.concatenate(Nn)
    same = len(S) == len(s_ref) and np.array_equal(S, s_ref) and np.array_equal(Nn, n_ref)
    ok &= same
    print(f"[{kernel}] one-shot: {len(s_ref)} spikes; chunked: {len(S)} spikes; identical spike trains: {same}; "
          f"{dt / 0.5:.1f} s wall per simulated s")
sys.exit(0 if ok else 1)
