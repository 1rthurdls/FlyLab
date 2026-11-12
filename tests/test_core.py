"""Unit tests: groups, search, name resolution, pathway tracing, interface, kernel equivalence.
Run: python -m pytest -q tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab import connectome, interface, lab  # noqa: E402
from flylab.groups import load_groups  # noqa: E402


def test_groups_have_sources_and_members():
    G = load_groups()
    for name in ["sugar_GRNs", "bitter_GRNs", "MN9", "P9", "MDN", "DNa02_left", "JO_grooming"]:
        assert G[name]["n"] > 0 and G[name]["source"], name
    assert G["sugar_GRNs"]["n"] == 21
    assert G["MN9"]["annotation_summary"]["cell_type"] == {"CB0701": 2}
    assert G["P9"]["annotation_summary"]["cell_type"] == {"DNp09": 2}


def test_search_finds_known_types():
    assert {m["cell_type"] for m in connectome.search_neurons("MDN")["matches"]} == {"MDN"}
    assert connectome.search_neurons("bitter")["n_neurons"] >= 42
    assert connectome.search_neurons("zzzz-no-such-thing")["n_neurons"] == 0


def test_resolve_names():
    assert len(lab.resolve("MN9")[0]) == 2
    assert len(lab.resolve("CB0553")[0]) == 2
    assert len(lab.resolve("CB0553_L")[0]) == 1
    assert lab.resolve("720575940660219265")[0] == lab.resolve("CB0701_R")[0]
    with pytest.raises(ValueError):
        lab.resolve("not_a_neuron")


def test_trace_sugar_to_mn9_is_short_and_excitatory():
    r = lab.trace_pathway("sugar_GRNs", "MN9")
    best = r["paths"][0]
    assert best["hops"] <= 4 and best["net_sign"] == "excitatory"
    assert best["path"][-1].startswith("CB0701")


def test_interface_mapping():
    n = 10
    ts = {g: np.zeros(n) for g in ["P9", "MDN", "DNa01_left", "DNa02_left", "DNa01_right", "DNa02_right", "MN9"]}
    ts["P9"][:] = 100
    ts["DNa02_left"][:] = 50
    c = interface.motor_commands(ts, 0.02)
    assert c["forward"][-1] == pytest.approx(1.0) and c["turn"][-1] > 0 and not c["feeding_state"].any()
    ts["MN9"][:] = 80
    c = interface.motor_commands(ts, 0.02)
    assert c["feeding_state"].all() and c["forward"].max() == 0  # feeding suppresses walking


def test_active_set_kernel_matches_dense_kernel():
    from flylab.brain import Stimulus, get_brain
    b = get_brain()
    st = [Stimulus(lab.resolve("sugar_GRNs")[0], 200)]
    a = b.run_numba(st, duration_s=0.05, n_trials=2, seed=3)
    d = b.run_numba(st, duration_s=0.05, n_trials=2, seed=3, dense=True)
    assert np.array_equal(a.counts, d.counts) and a.counts.sum() > 0
