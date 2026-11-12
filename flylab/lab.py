"""FlyLab experiment layer: everything the MCP tools do, as plain Python functions.

All functions return compact JSON-able dicts and write full outputs to runs/<run_id>/.
Results are cached by parameters (see cache.py), so repeated calls are instant.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import cache, connectome
from .brain import DT, SimResult, Stimulus, get_brain
from .groups import OUTPUT_GROUPS, load_groups

REPO = Path(__file__).resolve().parents[1]
BIN_S = 0.02  # time bin for group time series / body interface


# ----------------------------------------------------------------------------- names
def resolve(name: str) -> tuple[list[int], str]:
    """Name -> model indices. Accepts: a curated group name ('sugar_GRNs'), a FlyWire cell
    type ('CB0616'), a cell type with side suffix ('CB0616_L'), or a FlyWire root id."""
    groups = load_groups()
    a = connectome.annotations()
    s = str(name).strip()
    for g in groups:
        if g.lower() == s.lower():
            id2i = get_brain().id2i
            return [id2i[r] for r in groups[g]["root_ids"] if r in id2i], g
    if s.isdigit():
        i = get_brain().id2i.get(int(s))
        if i is None:
            raise ValueError(f"root id {s} is not in the model")
        return [i], connectome.short_label(i)
    side = None
    base = s
    if s[-2:] in ("_L", "_R"):
        base, side = s[:-2], {"_L": "left", "_R": "right"}[s[-2:]]
    m = a.label == base
    if side:
        m &= a.side == side
    if not m.any():
        lower = a.label.str.lower() == base.lower()
        m = lower & ((a.side == side) if side else True)
    if not m.any():
        raise ValueError(f"unknown neuron group / cell type / root id: '{name}'. "
                         f"Try search_neurons() or list_neuron_groups().")
    return a.loc[m, "idx"].astype(int).tolist(), s


def _resolve_many(names) -> tuple[list[int], dict]:
    idx, info = [], {}
    for n in names or []:
        ii, lab = resolve(n)
        idx += ii
        info[lab] = len(ii)
    return sorted(set(idx)), info


# ------------------------------------------------------------------------ simulation
def _group_indices() -> dict[str, list[int]]:
    groups = load_groups()
    id2i = get_brain().id2i
    return {g: [id2i[r] for r in groups[g]["root_ids"] if r in id2i] for g in groups}


def run_simulation(activate: list[str], silence: list[str] | None = None, stim_rate_hz: float = 200.0,
                   duration_s: float = 1.0, n_trials: int = 8, seed: int = 0,
                   schedule: list[dict] | None = None, label: str | None = None) -> dict:
    """Activate (Poisson drive) and/or silence neuron sets, simulate the whole brain.

    schedule (optional, advanced): list of {"neurons": [names], "rate_hz": .., "t0": s, "t1": s}
    for time-varying stimulation; used by run_behavior."""
    silence = silence or []
    params = {"activate": sorted(activate), "silence": sorted(silence), "rate": stim_rate_hz,
              "duration_s": duration_s, "n_trials": n_trials, "seed": seed, "schedule": schedule, "v": 4}
    lab = label or ("+".join(activate) or "baseline") + (("__silence-" + "+".join(silence)) if silence else "")
    run_id = cache.run_id_for("sim", lab, params)
    cached = cache.load_result(run_id)
    if cached is not None:
        cache.save_result(run_id, cached)  # bump 'latest' for the dashboard
        return cached

    brain = get_brain()
    act_idx, act_info = _resolve_many(activate)
    sil_idx, sil_info = _resolve_many(silence)
    stimuli = []
    if act_idx:
        stimuli.append(Stimulus(act_idx, stim_rate_hz))
    for s in schedule or []:
        ii, info = _resolve_many(s["neurons"])
        act_info.update({f"{k} [{s.get('t0', 0)}-{s.get('t1', 'end')} s]": v for k, v in info.items()})
        act_idx = sorted(set(act_idx) | set(ii))
        stimuli.append(Stimulus(ii, s.get("rate_hz", stim_rate_hz), s.get("t0", 0.0), s.get("t1")))
    res = brain.run(stimuli, silence=sil_idx, duration_s=duration_s, n_trials=n_trials, seed=seed)
    out = _summarize(run_id, res, act_idx, act_info, sil_idx, sil_info, params, lab)
    cache.save_result(run_id, out)
    return out


def _summarize(run_id, res: SimResult, act_idx, act_info, sil_idx, sil_info, params, lab) -> dict:
    from .plots import raster_figure

    d = cache.run_dir(run_id)
    rates = res.rates
    gi = _group_indices()
    np.savez_compressed(d / "spikes.npz", rates=rates.astype(np.float32), step=res.spike_step,
                        trial=res.spike_trial, neuron=res.spike_neuron, duration_s=res.duration_s)
    # Effective output: a silenced neuron's outgoing synapses are zero, so its output is 0 even if it
    # still spikes (relevant when an output neuron such as MN9 itself is silenced).
    sil_set = set(sil_idx)
    eff = rates.copy()
    if sil_idx:
        eff[list(sil_idx)] = 0.0
    out_groups = {}
    for g in OUTPUT_GROUPS:
        ii = gi[g]
        r = eff[ii]
        out_groups[g] = {"mean_rate_hz": round(float(r.mean()), 2), "max_neuron_rate_hz": round(float(r.max()), 2),
                         "n_neurons": len(ii), "n_active": int((r > 0).sum())}
        if len(ii) <= 4:
            labs = [connectome.short_label(i) for i in ii]
            labs = [f"{l}#{labs[:k].count(l) + 1}" if labs.count(l) > 1 else l for k, l in enumerate(labs)]
            out_groups[g]["per_neuron_hz"] = {l: round(float(eff[i]), 1) for l, i in zip(labs, ii)}
        if sil_set & set(ii):
            out_groups[g]["note"] = "contains silenced neurons: their effective output is counted as 0 Hz"
    def _ts(ii):
        keep = [i for i in ii if i not in sil_set]
        x = res.binned_rates(keep, BIN_S) * (len(keep) / max(len(ii), 1)) if keep else             np.zeros(int(round(res.duration_s / BIN_S)))
        return np.round(x, 2).tolist()
    timeseries = {g: _ts(gi[g]) for g in OUTPUT_GROUPS}
    stim_set = set(act_idx)
    order = np.argsort(-rates)
    top = []
    for i in order:
        if rates[i] <= 0 or len(top) >= 15:
            break
        if i in stim_set:
            continue
        desc = connectome.describe(i)
        top.append({"neuron": connectome.short_label(i), "root_id": desc["root_id"], "class": desc["class"],
                    "nt": desc["nt"], "rate_hz": round(float(rates[i]), 1)})
    # raster figure (trial 0): up to 5 stimulated neurons, one row per output group (all its neurons
    # pooled), and the top 10 responders
    m0 = res.spike_trial == 0
    st, sn = res.spike_step[m0] * DT / 1000.0, res.spike_neuron[m0]
    rows, spikes_by_row, shown = [], [], set()
    for i in sorted(act_idx, key=lambda i: -rates[i])[:5]:
        rows.append((connectome.short_label(i), "stim")); spikes_by_row.append(st[sn == i]); shown.add(i)
    for g in ["MN9", "P9", "DNa01_left", "DNa01_right", "DNa02_left", "DNa02_right", "MDN"]:
        ii = [i for i in gi[g] if i not in sil_set]
        rows.append((f"{g} ({len(gi[g])})", "output"))
        spikes_by_row.append(st[np.isin(sn, ii)] if ii else np.zeros(0)); shown.update(gi[g])
    for t in top:
        if sum(r[1] in ("other", "silenced") for r in rows) >= 10:
            break
        i = resolve(str(t["root_id"]))[0][0]
        if i not in shown:
            rows.append((t["neuron"], "silenced" if i in sil_set else "other")); spikes_by_row.append(st[sn == i])
    bars = [(g, out_groups[g]["mean_rate_hz"]) for g in ["MN9", "P9", "DNa01_left", "DNa01_right", "DNa02_left",
                                                          "DNa02_right", "MDN", "DNg12"]]
    title = f"Activate: {', '.join(act_info) or 'nothing'}" + (f"   |   Silence: {', '.join(sil_info)}" if sil_info else "")
    from .interface import MN9_THRESHOLD_HZ
    png = raster_figure(d / "raster.png", title, rows, spikes_by_row, res.duration_s, bars,
                        threshold_lines=[("MN9", MN9_THRESHOLD_HZ)])
    return {
        "run_id": run_id, "kind": "simulation", "label": lab,
        "params": params, "activated": act_info, "silenced": sil_info,
        "sim": {"duration_s": res.duration_s, "n_trials": res.n_trials, "wall_s": round(res.wall_s, 1),
                "n_active_neurons": int((rates > 0).sum()), "total_spikes": int(res.counts.sum()),
                "backend": res.meta.get("backend")},
        "output_groups": out_groups,
        "top_responders": top,
        "timeseries_bin_s": BIN_S,
        "timeseries": timeseries,
        "raster_png": str(Path(png).relative_to(REPO)).replace("\\", "/"),
        "note": "No spontaneous activity in this model: unstimulated neurons fire 0 Hz, so any rate > 0 is evoked.",
    }


def _rates(run_id: str) -> np.ndarray:
    return np.load(cache.RUNS / run_id / "spikes.npz")["rates"]


def compare_conditions(run_id_a: str, run_id_b: str, top_k: int = 12) -> dict:
    A, B = cache.load_result(run_id_a), cache.load_result(run_id_b)
    if A is None or B is None:
        raise ValueError("unknown run id")
    groups = {}
    for g in OUTPUT_GROUPS:
        if g not in A["output_groups"] or g not in B["output_groups"]:
            continue  # runs cached before this group existed
        ra, rb = A["output_groups"][g]["mean_rate_hz"], B["output_groups"][g]["mean_rate_hz"]
        if ra == 0 and rb == 0:
            continue
        groups[g] = {"a_hz": ra, "b_hz": rb, "change_hz": round(rb - ra, 2),
                     "change_pct": round(100 * (rb - ra) / ra, 1) if ra > 0 else None}
    out = {"a": run_id_a, "b": run_id_b, "a_label": A["label"], "b_label": B["label"], "output_groups": groups}
    try:
        xa, xb = _rates(run_id_a), _rates(run_id_b)
        diff = xb - xa
        o = np.argsort(diff)
        out["most_decreased"] = [{"neuron": connectome.short_label(i), "a_hz": round(float(xa[i]), 1),
                                  "b_hz": round(float(xb[i]), 1)} for i in o[:top_k] if diff[i] < 0]
        out["most_increased"] = [{"neuron": connectome.short_label(i), "a_hz": round(float(xa[i]), 1),
                                  "b_hz": round(float(xb[i]), 1)} for i in o[::-1][:top_k] if diff[i] > 0]
        out["n_neurons_changed_gt_5hz"] = int((np.abs(diff) > 5).sum())
    except FileNotFoundError:
        pass
    return out


# --------------------------------------------------------------------------- pathways
def trace_pathway(source: str, target: str, max_hops: int = 4, functional_run_id: str | None = None,
                  k: int = 8) -> dict:
    src, _ = resolve(source)
    tgt, _ = resolve(target)
    mask = None
    if functional_run_id:
        mask = _rates(functional_run_id) > 0
    r = connectome.trace_pathway(src, tgt, max_hops=max_hops, k=k, restrict_to=mask)
    r.pop("_intermediate_idx", None)
    r["source"], r["target"] = source, target
    r["mode"] = "functional (only neurons active in run %s)" % functional_run_id if functional_run_id else "structural"
    r["how_to_read"] = ("strength = product of input fractions along the path (synapses from pre / all input "
                        "synapses of post), edges >= 5 synapses. net_sign multiplies edge signs "
                        "(inhibitory = GABA/glutamate predicted).")
    return r


def lesion_screen(activate: list[str], target: str, candidates: list[str] | None = None, n_candidates: int = 8,
                  stim_rate_hz: float = 200.0, duration_s: float = 1.0, n_trials: int = 8,
                  by_cell_type: bool = True) -> dict:
    """Silence candidates one at a time and measure the drop in the target group's (effective) rate.

    If candidates is None they are chosen automatically: the key intermediates of the functional
    pathway (neurons active in the control run that lie on the strongest source->target paths).
    by_cell_type=True silences whole cell types (both hemispheres), like a genetic driver line would."""
    ctrl = run_simulation(activate, [], stim_rate_hz, duration_s, n_trials)
    tgt_idx, _ = resolve(target)
    base = float(_rates(ctrl["run_id"])[tgt_idx].mean())
    if candidates is None:
        mask = _rates(ctrl["run_id"]) > 0
        src = set(_resolve_many(activate)[0])
        tr = connectome.trace_pathway(sorted(src), tgt_idx, max_hops=4, k=50, restrict_to=mask)
        cand_idx = [i for i in tr["_intermediate_idx"] if i not in src and i not in set(tgt_idx)]
        if by_cell_type:
            a = connectome.annotations()
            candidates = list(dict.fromkeys(a.label.iloc[i] for i in cand_idx if a.label.iloc[i] != "unannotated"))
        else:
            candidates = [str(connectome.describe(i)["root_id"]) for i in cand_idx]
        candidates = candidates[:n_candidates]
    rows = []
    for c in candidates:
        r = run_simulation(activate, [c], stim_rate_hz, duration_s, n_trials)
        ii, lab = resolve(c)
        x = _rates(r["run_id"]).copy()
        x[ii] = 0.0  # silenced neurons have no effective output
        v = float(x[tgt_idx].mean())
        rows.append({"silenced": connectome.short_label(ii[0]) if len(ii) == 1 else lab,
                     "n_neurons": len(ii), "target_rate_hz": round(v, 1),
                     "change_pct": round(100 * (v - base) / base, 1) if base > 0 else None,
                     "run_id": r["run_id"]})
    rows.sort(key=lambda x: x["target_rate_hz"])
    return {"control_run_id": ctrl["run_id"], "target": target, "control_rate_hz": round(base, 1),
            "results": rows,
            "interpretation": "Candidates whose silencing produces the largest drop are candidate necessary nodes "
                              "(model prediction; check against literature before claiming)."}


# ---------------------------------------------------------------------------- misc
def list_neuron_groups() -> dict:
    G = load_groups()
    return {g: {"n": v["n"], "role": v["role"], "description": v["description"]} for g, v in G.items()}


def get_neuron_group(name: str) -> dict:
    G = load_groups()
    for g, v in G.items():
        if g.lower() == name.lower():
            out = {k: v[k] for k in ("n", "role", "description", "source", "annotation_summary")}
            out["name"] = g
            out["root_ids"] = v["root_ids"][:30]
            if v["n"] > 30:
                out["root_ids_truncated"] = True
            return out
    raise ValueError(f"no group '{name}'. Available: {list(G)}")


def get_literature_facts(topic: str = "") -> dict:
    text = (REPO / "data" / "literature.md").read_text(encoding="utf8")
    blocks = [b.strip() for b in text.split("\n## ") if b.strip()]
    if topic:
        t = topic.lower()
        hits = [b for b in blocks if t in b.lower()]
        if not hits:  # fall back to word overlap
            words = [w for w in t.split() if len(w) > 2]
            hits = [b for b in blocks if any(w in b.lower() for w in words)]
        blocks = hits or blocks
    return {"topic": topic, "source_file": "data/literature.md", "sections": ["## " + b if not b.startswith("#") else b
                                                                             for b in blocks]}
