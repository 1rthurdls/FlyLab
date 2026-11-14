"""Data for the dashboard's 3D brain view.

* positions(): one 3D point per model neuron (FlyWire v783 annotation 'pos_x/y/z', a point on
  the neuron inside the brain; voxel size 4 x 4 x 40 nm), normalised to roughly [-1, 1].
* activity(run_id): spike times of ONE trial (trial 0) for a simulation or a behaviour run,
  plus the role of the neurons to colour (stimulated / output / silenced). For a behaviour with
  a sugar-contact event the two brain simulations are spliced at the contact time, exactly as
  the body received them.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
POS_NPY = DATA / "neuron_positions.npy"
VOXEL_NM = np.array([4.0, 4.0, 40.0])
DT_MS = 0.1
OUTPUT_LABEL_GROUPS = ["MN9", "P9", "DNa01_left", "DNa01_right", "DNa02_left", "DNa02_right", "MDN", "giant_fiber"]


@lru_cache(maxsize=1)
def positions() -> np.ndarray:
    """(N, 3) float32 in model index order; x = left-right, y = dorsal-up, z = anterior-posterior."""
    if POS_NPY.exists():
        return np.load(POS_NPY)
    comp = pd.read_csv(DATA / "Completeness_783.csv", index_col=0)
    a = pd.read_csv(DATA / "Supplemental_file1_neuron_annotations.tsv", sep="\t",
                    usecols=["root_id", "pos_x", "pos_y", "pos_z"], dtype={"root_id": "int64"})
    a = a.drop_duplicates("root_id").set_index("root_id").reindex(comp.index.astype("int64"))
    xyz = a[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM  # nm
    ok = np.isfinite(xyz).all(1)
    c = np.nanmedian(xyz[ok], axis=0)
    s = np.nanmax(np.abs(xyz[ok] - c))
    p = (xyz - c) / s
    p[:, 1] *= -1  # image y grows ventrally -> flip so dorsal is up
    p[:, 2] *= -1
    p[~ok] = np.nan
    p = p.astype(np.float32)
    np.save(POS_NPY, p)
    return p


def positions_bytes() -> bytes:
    p = positions().copy()
    p[~np.isfinite(p)] = 1e6  # far away = invisible
    return p.tobytes()


def _roles(sim_results: list[dict], silenced: list[str]) -> dict:
    from .lab import _group_indices, _resolve_many, resolve

    stim = set()
    for r in sim_results:
        prm = r.get("params", {})
        stim |= set(_resolve_many(prm.get("activate", []))[0])
        for s in prm.get("schedule") or []:
            stim |= set(_resolve_many(s["neurons"])[0])
    sil = set(_resolve_many(silenced)[0]) if silenced else set()
    gi = _group_indices()
    out, labels = [], {}
    for g in OUTPUT_LABEL_GROUPS:
        for i in gi.get(g, []):
            out.append(i)
            labels[int(i)] = "DNa" if g.startswith("DNa") else ("GF" if g == "giant_fiber" else g.split("_")[0])
    return {"stim": sorted(int(i) for i in stim), "output": sorted(set(int(i) for i in out)),
            "silenced": sorted(int(i) for i in sil), "output_labels": labels}


def _spikes(run_id: str, trial: int = 0):
    z = np.load(cache.RUNS / run_id / "spikes.npz")
    m = z["trial"] == trial
    return z["step"][m].astype(np.int64), z["neuron"][m].astype(np.int64), float(z["duration_s"])


MAX_SPIKES_3D = 120_000   # cap for the dashboard's 3D brain (it animates spikes; it does not need them all)

def activity(run_id: str) -> dict:
    res = cache.load_result(run_id)
    if res is None:
        raise ValueError(f"unknown run {run_id}")
    if res.get("kind") == "behavior":
        sims = [cache.load_result(r) for r in res["brain_runs"]]
        steps, neus, dur = _spikes(res["brain_runs"][0])
        contact = None
        if len(res["brain_runs"]) > 1 and len(res.get("phases", [])) > 1:
            contact = float(res["phases"][1]["t"][0])
            c_step = int(round(contact * 1000 / DT_MS))
            s2, n2, dur = _spikes(res["brain_runs"][1])
            keep1, keep2 = steps < c_step, s2 >= c_step
            steps = np.concatenate([steps[keep1], s2[keep2]])
            neus = np.concatenate([neus[keep1], n2[keep2]])
        roles = _roles(sims, res.get("silenced") or [])
        label = res["behavior"]
        play_speed = res.get("video_play_speed", 0.25)
    else:
        steps, neus, dur = _spikes(run_id)
        contact = None
        roles = _roles([res], res.get("params", {}).get("silence", []))
        label = res.get("label", run_id)
        play_speed = 0.25
    o = np.argsort(steps, kind="stable")
    n_total = int(len(o))
    if n_total > MAX_SPIKES_3D:   # long closed-loop runs (25 s) have ~10^5-10^6 spikes: send an even subsample
        o = o[np.linspace(0, n_total - 1, MAX_SPIKES_3D).astype(np.int64)]
    return {"run_id": run_id, "kind": res.get("kind"), "label": label, "duration_s": dur, "contact_s": contact,
            "play_speed": play_speed, "step_ms": DT_MS, "n_spikes": n_total, "n_spikes_sent": int(len(o)),
            "subsampled": n_total > MAX_SPIKES_3D,
            "n_neurons_firing": int(len(np.unique(neus))),
            "t": steps[o].tolist(), "n": neus[o].tolist(), **roles}


if __name__ == "__main__":
    p = positions()
    print(p.shape, np.nanmin(p, 0), np.nanmax(p, 0))
    latest = json.loads((cache.RUNS / "latest.json").read_text())
    a = activity(latest["run_id"])
    print(a["label"], a["n_spikes"], a["n_neurons_firing"], len(a["stim"]), len(a["output"]))
