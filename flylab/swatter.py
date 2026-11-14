"""Fly-swatter escape game: a swatter comes down on a walking fly; does the brain get it out in time?

HONEST FRAMING
* Real: the whole-brain model (FlyWire v783, Shiu et al. LIF), run as ONE continuous trial in 5 ms chunks
  (BrainStepper). P9 is held at 100 Hz so the fly walks. The swatter's looming image drives the looming detectors
  LPLC2 and LC4 (both sides); whatever the giant fiber (DNp01) does after that is the connectome's response.
  Literature: LPLC2 and LC4 feed looming information to the giant fiber, which drives the fast escape take-off
  (data/literature.md: vonReyn2014, Klapoetke2017, vonReyn2017, Ache2019).
* Designed (documented here and in looming.py):
    - the eye model: looming expansion speed -> LPLC2/LC4 Poisson rate (looming.py);
    - the decision rule: escape when the giant fiber's population rate over the last 20 ms exceeds
      interface.GF_THRESHOLD_HZ (50 Hz, the same threshold as run_behavior("takeoff")), plus a fixed
      TAKEOFF_LATENCY_S from that moment until the fly is off the ground;
    - the swatter (size, start distance, speed) and the outcome rule: escaped if the fly is off the ground before
      the swatter reaches it, otherwise SPLAT;
    - the take-off itself is the SCRIPTED animation of FlyBody.scripted_takeoff (NeuroMechFly cannot fly).
* Real flies also have a slower, giant-fiber-independent escape (vonReyn2014); it is not modelled here, so a
  giant-fiber lesion predicts "no escape" only within this model.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from . import cache, interface, lab, looming
from .arena import record_brain
from .brain import BrainStepper, get_brain

CHUNK_S = 0.005
STEPS = int(round(CHUNK_S * 1000 / 0.1))
P9_HZ = 100.0
SWAT_HALF_MM = 20.0          # 4 cm swatter pad
START_MM = 150.0             # swatter starts 15 cm from the fly
SWING_T0 = 0.3               # s: the fly walks alone first
SPEEDS = {"slow": 250.0, "normal": 500.0, "fast": 1500.0, "lightning": 3000.0}   # mm/s
GF_WINDOW_S = 0.02
TAKEOFF_LATENCY_S = 0.010    # designed: giant fiber command -> legs off the ground
TAIL_S = 0.06
PAD_HALF_THICK = 0.8
_d = np.array([-0.35, 0.55, 1.0])
SWAT_DIR = _d / np.linalg.norm(_d)  # swatter comes down from above, from behind the fly (camera side opposite)


def _props(wb):
    sw = wb.add("body", name="swatter", mocap=True, pos=(0, 0, 400))
    kw = dict(contype=0, conaffinity=0)
    sw.add("geom", name="swat_pad", type="box", size=(SWAT_HALF_MM, SWAT_HALF_MM * 0.85, PAD_HALF_THICK),
           rgba=(0.85, 0.16, 0.12, 0.92), **kw)
    for k in range(-4, 5):   # mesh holes look
        sw.add("geom", name=f"swat_gx{k}", type="box", size=(SWAT_HALF_MM, 0.35, PAD_HALF_THICK + 0.05),
               pos=(0, k * SWAT_HALF_MM * 0.85 / 5, 0), rgba=(0.55, 0.08, 0.06, 1), **kw)
        sw.add("geom", name=f"swat_gy{k}", type="box", size=(0.35, SWAT_HALF_MM * 0.85, PAD_HALF_THICK + 0.05),
               pos=(k * SWAT_HALF_MM / 5, 0, 0), rgba=(0.55, 0.08, 0.06, 1), **kw)
    h = SWAT_DIR[:2] / np.linalg.norm(SWAT_DIR[:2])
    a = (h[0] * SWAT_HALF_MM, h[1] * SWAT_HALF_MM, 0)
    b = (a[0] + h[0] * 110, a[1] + h[1] * 110, 35)
    sw.add("geom", name="swat_handle", type="capsule", fromto=(*a, *b), size=(1.6,), rgba=(0.15, 0.15, 0.17, 1), **kw)


def _brain_trial(speed_mm_s, silence, seed):
    from .groups import OUTPUT_GROUPS
    brain = get_brain()
    inputs = {"P9": lab.resolve("P9")[0], "LPLC2": lab.resolve("LPLC2")[0], "LC4": lab.resolve("LC4")[0]}
    sil_idx, sil_info = lab._resolve_many(silence)
    st = BrainStepper(brain, inputs, silence=sil_idx, seed=seed)
    gi = lab._group_indices()
    sil = set(sil_idx)
    n_gf = max(len(gi["giant_fiber"]), 1)
    gf_eff = [i for i in gi["giant_fiber"] if i not in sil]   # a silenced neuron's output counts as 0
    t_hit = SWING_T0 + START_MM / speed_mm_s
    w = int(round(GF_WINDOW_S / CHUNK_S))
    ts = {g: [] for g in OUTPUT_GROUPS}
    loom_log, gf_win, gf_counts, all_s, all_n = [], [], [], [], []
    counts = np.zeros(brain.N, np.int64)
    t_esc = None
    t_wall = time.perf_counter()
    k = 0
    while True:
        tm = (k + 0.5) * CHUNK_S
        d = START_MM - speed_mm_s * (tm - SWING_T0) if tm >= SWING_T0 else START_MM
        rate = looming.loom_rate_hz(d, speed_mm_s, SWAT_HALF_MM) if (tm >= SWING_T0 and d > 0) else 0.0
        c, s_, n_ = st.step({"P9": P9_HZ, "LPLC2": rate, "LC4": rate}, STEPS)
        counts += c
        all_s.append(s_); all_n.append(n_)
        for g in OUTPUT_GROUPS:
            ii = [i for i in gi[g] if i not in sil]
            ts[g].append(float(c[ii].sum()) / max(len(gi[g]), 1) / CHUNK_S)
        gf_counts.append(int(c[gf_eff].sum()) if gf_eff else 0)
        win_hz = sum(gf_counts[-w:]) / n_gf / GF_WINDOW_S
        gf_win.append(win_hz)
        loom_log.append((round(max(d, 0.0), 1), round(rate, 1)))
        k += 1
        t = k * CHUNK_S
        if t_esc is None and win_hz > interface.GF_THRESHOLD_HZ:
            t_esc = round(t, 4)
        if (t_esc is not None and t >= t_esc + TAIL_S) or t >= t_hit + TAIL_S:
            break
    D = round(k * CHUNK_S, 4)
    act_info = {"P9 (100 Hz)": len(inputs["P9"]), "LPLC2 (looming, closed loop)": len(inputs["LPLC2"]),
                "LC4 (looming, closed loop)": len(inputs["LC4"])}
    sim_params = {"activate": ["P9", "LPLC2", "LC4"], "silence": sorted(silence), "closed_loop": True,
                  "swatter_mm_s": speed_mm_s, "seed": seed, "v": 1}
    lab_ = "swatter" + (("__silence-" + "+".join(silence)) if silence else "")
    sim_id, sim = record_brain(lab_, D, counts, all_s, all_n, t_wall, inputs, act_info, sil_idx, sil_info,
                               sim_params)
    return dict(ts=ts, loom=loom_log, gf_win=gf_win, t_esc=t_esc, t_hit=t_hit, D=D, sim_id=sim_id, sim=sim)


def swat_fly(speed: str = "normal", silence: list[str] | None = None, seed: int = 0, render: bool = True) -> dict:
    if speed not in SPEEDS:
        raise ValueError(f"unknown speed '{speed}'. Use one of {list(SPEEDS)}")
    silence = silence or []
    v = SPEEDS[speed]
    params = {"speed": v, "silence": sorted(silence), "seed": seed, "size": SWAT_HALF_MM, "start": START_MM,
              "loom": [looming.LOOM_MAX_HZ, looming.LOOM_SAT_DEG_S], "lat": TAKEOFF_LATENCY_S, "v": 2}
    lab_ = f"swat_{speed}" + (("__silence-" + "+".join(silence)) if silence else "")
    run_id = cache.run_id_for("beh", lab_, params)
    cached = cache.load_result(run_id)
    if cached is not None and (cached.get("video") and (cache.REPO / cached["video"]).exists() or not render):
        cache.save_result(run_id, cached)
        return cached
    t0w = time.perf_counter()
    B = _brain_trial(v, silence, seed)
    t_esc, t_hit = B["t_esc"], B["t_hit"]
    escaped = t_esc is not None and t_esc + TAKEOFF_LATENCY_S < t_hit
    margin_ms = round((t_hit - (t_esc + TAKEOFF_LATENCY_S)) * 1000) if t_esc is not None else None
    cmd = interface.motor_commands({g: np.array(x) for g, x in B["ts"].items()}, CHUNK_S)
    i_esc = int(round(t_esc / CHUNK_S)) - 1 if t_esc is not None else None
    d_esc = B["loom"][i_esc][0] if i_esc is not None else None
    loom_rates = np.array([r for _, r in B["loom"]])
    out_stats = {
        "swatter_speed_mm_s": v, "swatter_size_mm": 2 * SWAT_HALF_MM, "swing_start_s": SWING_T0,
        "swatter_arrives_s": round(t_hit, 3),
        "giant_fiber_threshold_s": t_esc,
        "swatter_distance_at_gf_mm": d_esc,
        "swatter_angle_at_gf_deg": round(looming.angle_deg(d_esc, SWAT_HALF_MM), 1) if d_esc is not None else None,
        "looming_rate_at_gf_hz": float(loom_rates[i_esc]) if i_esc is not None else None,
        "giant_fiber_peak_hz": round(float(max(B["gf_win"])), 1),
        "looming_peak_hz": round(float(loom_rates.max()), 1),
        "margin_ms": margin_ms,
    }
    if escaped:
        verdict = (f"ESCAPED: giant fiber crossed {interface.GF_THRESHOLD_HZ:.0f} Hz at t = {t_esc:.3f} s, when the "
                   f"swatter was {d_esc:.0f} mm away ({out_stats['swatter_angle_at_gf_deg']:.0f} deg wide); "
                   f"off the ground {margin_ms} ms before the swatter landed")
    elif t_esc is not None:
        verdict = (f"SPLAT: the giant fiber fired at t = {t_esc:.3f} s but {-margin_ms} ms too late "
                   f"(swatter arrived at {t_hit:.3f} s)")
    else:
        verdict = (f"SPLAT: the giant fiber never crossed {interface.GF_THRESHOLD_HZ:.0f} Hz "
                   f"(peak {out_stats['giant_fiber_peak_hz']:.0f} Hz) before the swatter arrived at {t_hit:.3f} s")
    d = cache.run_dir(run_id)
    video = None
    body_info = None
    if render:
        body_info = _render(d, cmd, B, escaped, t_esc, t_hit, v)
        video = f"runs/{run_id}/video.mp4"
    from .behaviors import _save_cmds
    cmd_s = dict(cmd)
    cmd_s["rates"] = dict(cmd["rates"]) | {"looming": loom_rates, "GF": np.array(B["gf_win"])}
    _save_cmds(d, [cmd_s])
    out = {"run_id": run_id, "kind": "behavior", "behavior": f"swat_{speed}", "label": lab_,
           "description": "Fly swatter: looming swatter -> LPLC2/LC4 -> whole brain -> giant fiber; escape if it "
                          "fires in time (scripted take-off animation), otherwise splat.",
           "silenced": silence, "outcome": "escaped" if escaped else "splat", **out_stats,
           "brain_runs": [B["sim_id"]], "raster_png": B["sim"]["raster_png"], "video": video,
           "video_play_speed": 0.1, "body": body_info, "phases": [],
           "summary": (f"LESION: silenced {', '.join(silence)}. " if silence else "") + verdict + ".",
           "wall_s": round(time.perf_counter() - t0w, 1),
           "framing": "Brain is real (closed loop, 5 ms chunks); looming eye model, 50 Hz giant-fiber threshold, "
                      "take-off latency and the take-off animation are designed. Nothing learns."}
    cache.save_result(run_id, out)
    return out


def _render(d, cmd, B, escaped, t_esc, t_hit, v):
    from .behaviors import _clean
    from .body import FlyBody

    body = FlyBody(props=_props, play_speed=0.1, cam_distance=45.0, cam_elevation_deg=35.0,
                   cam_azimuth_deg=-65.0, cam_fovy=55.0)
    body._cam_z = 8.0   # aim above the fly so the incoming swatter is in view
    ph = body.physics
    mid = int(ph.model.body_mocapid[ph.model.name2id("swatter", "body")])
    S = {"target": None}
    gf_win, loom = B["gf_win"], B["loom"]

    def hook(b):
        t = b.time
        hit_t = (t_esc + TAKEOFF_LATENCY_S) if escaped else t_hit
        if S["target"] is None or t < hit_t - 1e-9 and not (escaped and t >= t_esc):
            S["target"] = b.position.copy()
        dist = START_MM if t < SWING_T0 else max(START_MM - v * (t - SWING_T0), 0.0)
        rest_z = PAD_HALF_THICK + (0.0 if escaped else 0.9)
        p = np.array([S["target"][0], S["target"][1], rest_z]) + SWAT_DIR * dist
        ph.data.mocap_pos[mid] = p
        i = min(int(t / CHUNK_S), len(gf_win) - 1)
        if not escaped and t >= t_hit:
            b._hud_text = "SPLAT  -  the giant fiber did not fire in time"
        elif escaped and t >= t_esc:
            b._hud_text = f"ESCAPE: giant fiber {gf_win[min(i, int(t_esc / CHUNK_S))]:.0f} Hz > 50 Hz -> take-off (scripted)"
        else:
            b._hud_text = (f"swatter {dist:5.0f} mm | LPLC2/LC4 {loom[i][1]:3.0f} Hz | giant fiber "
                           f"{gf_win[i]:3.0f} Hz")

    body.frame_hooks.append(hook)
    b_end = int(round(((t_esc + TAKEOFF_LATENCY_S) if escaped else t_hit) / CHUNK_S))
    r1 = body.run(interface.to_program(cmd, 0, b_end))
    info = {"walk": _clean(r1)}
    if escaped:
        r2 = body.scripted_takeoff(duration_s=0.3, climb_mm=14.0, forward_mm=16.0)
        info["takeoff"] = _clean(r2)
        rest = t_hit + 0.05 - body.time
        if rest > 0:
            body.hold(rest)
    else:
        body.hold(0.12)
    body.save_video(d / "video.mp4")
    body.close()
    return info


if __name__ == "__main__":
    import sys
    r = swat_fly(sys.argv[1] if len(sys.argv) > 1 else "normal", sys.argv[2:] or None)
    print(r["summary"], "| wall", r["wall_s"], "s")
