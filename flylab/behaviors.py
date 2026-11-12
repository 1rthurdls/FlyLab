"""Catalogue of behaviours and the end-to-end brain -> body runner.

Each behaviour = a stimulus protocol on the whole-brain model. The resulting output-neuron
rates go through the documented interface (interface.py) into the flygym body (body.py).
The "find_sugar" hero runs a coarse closed loop: the BODY detects contact with the sugar drop
and that switches on the sugar GRN stimulus in the BRAIN.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import cache, interface, lab

N_TRIALS = 4  # trials averaged for behaviour time series (rates are smooth enough at 4)
SUGAR_POS = (9.0, 0.0)  # mm, straight ahead of the fly's start position

BEHAVIORS = {
    "walk_forward": {
        "stimulus": [("P9", 100.0)], "duration_s": 2.0,
        "description": "Activate P9 (DNp09) descending neurons -> forward walking.",
        "validated_in_model": "P9 ~100 Hz when stimulated (directly driven); also recruits DNa02 (R>L) -> slight right curve.",
    },
    "turn_left": {
        "stimulus": [("P9", 100.0), ("DNa01_left", 100.0), ("DNa02_left", 100.0)], "duration_s": 2.0,
        "description": "P9 for walking + left DNa01/DNa02 steering neurons -> walking turn to the left.",
        "validated_in_model": "left DNa ~100 Hz each; right DNa only weakly recruited.",
    },
    "turn_right": {
        "stimulus": [("P9", 100.0), ("DNa01_right", 100.0), ("DNa02_right", 100.0)], "duration_s": 2.0,
        "description": "P9 + right DNa01/DNa02 -> walking turn to the right.",
        "validated_in_model": "right DNa ~100 Hz each.",
    },
    "walk_backward": {
        "stimulus": [("MDN", 100.0)], "duration_s": 2.0,
        "description": "Activate moonwalker descending neurons (MDN) -> backward walking.",
        "validated_in_model": "MDN ~100 Hz; no P9 recruitment.",
    },
    "feed": {
        "stimulus": [("sugar_GRNs", 200.0)], "duration_s": 1.5, "sugar_under_fly": True,
        "description": "Sugar GRNs (Shiu et al. 2024 set) at 200 Hz -> MN9 -> proboscis extension.",
        "validated_in_model": "MN9 ~78 Hz (0 Hz at baseline).",
    },
    "feed_with_bitter": {
        "stimulus": [("sugar_GRNs", 200.0), ("bitter_GRNs", 200.0)], "duration_s": 1.5, "sugar_under_fly": True,
        "description": "Sugar + bitter GRNs together -> bitter suppresses the MN9 feeding output.",
        "validated_in_model": "MN9 ~11 Hz (vs ~78 Hz sugar alone) -> below feeding threshold.",
    },
    "groom": {
        "stimulus": [("JO_grooming", 200.0)], "duration_s": 1.0, "brain_only": True,
        "description": "Johnston's organ (JO-F, 'grooming') mechanosensory neurons. BRAIN ONLY: flygym has no "
                       "grooming controller, so no body video; inspect the raster/top responders instead.",
        "validated_in_model": "activates ~300 neurons incl. DNg84, AN_GNG_65; no walking/feeding outputs.",
    },
    "takeoff": {
        "stimulus": [("P9", 100.0)], "looming": [("LPLC2", 50.0, 0.6)], "duration_s": 1.2,
        "description": "Escape take-off: the fly walks (P9); at 0.6 s a looming stimulus drives the LPLC2 looming "
                       "detectors (50 Hz). If the giant fiber (DNp01) fires above 50 Hz the fly takes off. RESTRICTED: "
                       "the take-off/flight is a single SCRIPTED animation (no aerodynamics); only the trigger comes "
                       "from the brain. Lesion with silence=['giant_fiber'] -> no take-off.",
        "validated_in_model": "LPLC2 50 Hz -> giant fiber ~95-118 Hz; sugar GRNs -> giant fiber 0 Hz.",
    },
    "find_sugar": {
        "stimulus": [("P9", 100.0)], "on_sugar_contact": [("sugar_GRNs", 200.0)], "duration_s": 3.0,
        "description": "HERO: P9 drives the fly toward a sugar drop; when the body touches the sugar the sugar GRNs "
                       "are stimulated in the brain; MN9 rises -> the fly stops and feeds. Lesion it with silence=[...].",
        "validated_in_model": "see runs/ for latest; with silence=['MN9'] the fly walks over the sugar without feeding.",
    },
}


def list_behaviors() -> dict:
    return {k: {"description": v["description"], "stimulus": [f"{g} @ {r:.0f} Hz" for g, r in v["stimulus"]],
                "duration_s": v["duration_s"], "body_video": not v.get("brain_only", False),
                "validated_in_model": v["validated_in_model"]}
            for k, v in BEHAVIORS.items()} | {"_interface": interface.MAPPING_TEXT.strip().splitlines()[0:3]}


def _sim(stim, extra_activate, silence, duration_s, label, extra_sched=None):
    sched = [{"neurons": [g], "rate_hz": r, "t0": 0.0, "t1": None} for g, r in stim]
    sched += [{"neurons": [g], "rate_hz": 200.0, "t0": 0.0, "t1": None} for g in extra_activate]
    sched += extra_sched or []
    return lab.run_simulation([], silence, 200.0, duration_s, N_TRIALS, schedule=sched, label=label)


def run_behavior(behavior: str, activate: list[str] | None = None, silence: list[str] | None = None,
                 duration_s: float | None = None, render: bool = True) -> dict:
    if behavior not in BEHAVIORS:
        raise ValueError(f"unknown behaviour '{behavior}'. Available: {list(BEHAVIORS)}")
    B = BEHAVIORS[behavior]
    activate, silence = activate or [], silence or []
    D = float(duration_s or B["duration_s"])
    params = {"behavior": behavior, "activate": sorted(activate), "silence": sorted(silence), "D": D,
              "sugar": SUGAR_POS, "v": 3}
    if behavior == "find_sugar":
        from .body import CONTACT_PARTS
        params["contact_parts"] = list(CONTACT_PARTS)
    lab_ = behavior + (("__silence-" + "+".join(silence)) if silence else "")
    run_id = cache.run_id_for("beh", lab_, params)
    cached = cache.load_result(run_id)
    if cached is not None and (not cached.get("video") or (cache.REPO / cached["video"]).exists()):
        cache.save_result(run_id, cached)
        return cached
    t_start = time.perf_counter()
    d = cache.run_dir(run_id)
    phases = []
    loom = [{"neurons": [g], "rate_hz": r, "t0": t0, "t1": None} for g, r, t0 in B.get("looming", [])]
    sim1 = _sim(B["stimulus"], activate, silence, D, f"{lab_}_brain", loom or None)
    cmd = interface.motor_commands(sim1["timeseries"], sim1["timeseries_bin_s"])
    out = {"run_id": run_id, "kind": "behavior", "behavior": behavior, "label": lab_,
           "description": B["description"], "silenced": silence, "extra_activated": activate,
           "brain_runs": [sim1["run_id"]], "raster_png": sim1["raster_png"]}
    if B.get("brain_only") or not render:
        phases.append({"t": [0, D], "stimulus": [g for g, _ in B["stimulus"]] + activate,
                       "brain_to_body": interface.describe(cmd)})
        out.update(phases=phases, video=None, body=None,
                   summary=f"{behavior}: brain-only. " + interface.describe(cmd))
        _save_cmds(d, [cmd])
        cache.save_result(run_id, out)
        return out

    from .body import FlyBody
    sugar_pos = (0.0, 0.0) if B.get("sugar_under_fly") else (SUGAR_POS if behavior == "find_sugar" else None)
    body = FlyBody(sugar_pos=sugar_pos)
    if "looming" in B:  # escape take-off
        hits = np.flatnonzero(cmd["takeoff_state"])
        loom_t = min(t0 for _, _, t0 in B["looming"])
        if len(hits):
            b_to = int(hits[0])
            t_to = b_to * cmd["bin_s"]
            r1 = body.run(interface.to_program(cmd, 0, b_to))
            phases.append({"t": [0, round(t_to, 2)], "stimulus": [g for g, _ in B["stimulus"]] + activate +
                           [f"{g} from {t0} s" for g, _, t0 in B["looming"]],
                           "brain_to_body": interface.describe(cmd, 0, b_to), "body": _clean(r1)})
            r2 = body.scripted_takeoff()
            phases.append({"t": [round(t_to, 2), round(t_to + r2.get("t_end_s", 0) - r2.get("t_start_s", 0), 2)],
                           "event": f"giant fiber > {interface.GF_THRESHOLD_HZ:.0f} Hz "
                                    f"{(t_to - loom_t) * 1000:.0f} ms after looming onset -> TAKE-OFF "
                                    "(scripted flight animation)",
                           "brain_to_body": interface.describe(cmd, b_to, min(b_to + 5, cmd['n_bins'])),
                           "body": _clean(r2)})
        else:
            r1 = body.run(interface.to_program(cmd))
            phases.append({"t": [0, D], "stimulus": [g for g, _ in B["stimulus"]] + activate,
                           "event": "looming from %.1f s, giant fiber never crossed threshold -> NO take-off" % loom_t,
                           "brain_to_body": interface.describe(cmd), "body": _clean(r1)})
        video = body.save_video(d / "video.mp4")
        _save_cmds(d, [cmd])
        out.update(phases=phases, video=f"runs/{run_id}/video.mp4", wall_s=round(time.perf_counter() - t_start, 1),
                   video_play_speed=body.play_speed, takeoff=bool(len(hits)),
                   video_note="slow motion: video time x play_speed = simulated time; flight is scripted")
        out["summary"] = _summary(behavior, phases, silence)
        cache.save_result(run_id, out)
        return out
    hero = "on_sugar_contact" in B
    r1 = body.run(interface.to_program(cmd), stop_on_sugar_contact=hero)
    tc = r1.get("sugar_contact_time_s") if hero else None
    phases.append({"t": [0, round(tc if tc is not None else D, 2)], "stimulus": [g for g, _ in B["stimulus"]] + activate,
                   "brain_to_body": interface.describe(cmd, 0, int((tc or D) / cmd["bin_s"])), "body": _clean(r1)})
    cmds = [cmd]
    if hero and tc is not None and tc < D - 0.1:
        sched2 = [{"neurons": [g], "rate_hz": r, "t0": round(tc, 3), "t1": None} for g, r in B["on_sugar_contact"]]
        sim2 = _sim(B["stimulus"], activate, silence, D, f"{lab_}_brain_after_contact", sched2)
        cmd2 = interface.motor_commands(sim2["timeseries"], sim2["timeseries_bin_s"])
        b0 = int(round(tc / cmd2["bin_s"]))
        r2 = body.run(interface.to_program(cmd2, b0))
        phases.append({"t": [round(tc, 2), D], "event": "sugar contact -> sugar GRNs stimulated",
                       "stimulus": [g for g, _ in B["stimulus"]] + [g for g, _ in B["on_sugar_contact"]] + activate,
                       "brain_to_body": interface.describe(cmd2, b0), "body": _clean(r2)})
        out["brain_runs"].append(sim2["run_id"])
        out["raster_png"] = sim2["raster_png"]
        cmds.append(cmd2)
    video = body.save_video(d / "video.mp4")
    _save_cmds(d, cmds)
    out.update(phases=phases, video=f"runs/{run_id}/video.mp4", wall_s=round(time.perf_counter() - t_start, 1),
               video_play_speed=getattr(body, "play_speed", 1.0),
               video_note="slow motion: video time x play_speed = simulated time")
    out["summary"] = _summary(behavior, phases, silence)
    cache.save_result(run_id, out)
    return out


def _clean(r: dict) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in (r or {}).items()
            if isinstance(v, (int, float, str, bool, list, tuple, type(None)))}


def _summary(behavior, phases, silence) -> str:
    s = []
    if silence:
        s.append(f"LESION: silenced {', '.join(silence)}.")
    for p in phases:
        ev = f" [{p['event']}]" if "event" in p else ""
        b = p.get("body") or {}
        if b.get("scripted"):
            mv = f" Body: scripted take-off, climbed {b.get('climb_mm', 0):.0f} mm (animation, no aerodynamics)"
        else:
            mv = f" Body: walked {b.get('distance_walked_mm', 0):.1f} mm, heading change " \
                 f"{b.get('heading_change_deg', 0):+.0f} deg" if b else ""
        s.append(f"t={p['t'][0]}-{p['t'][1]} s{ev}: {p['brain_to_body']}.{mv}")
    return " ".join(s)


def _save_cmds(d, cmds):
    """Per-bin motor commands (also usable to animate a stylised fly in the dashboard)."""
    def enc(v):
        v = np.asarray(v)
        return v.astype(int).tolist() if v.dtype == bool else v.round(3).tolist()

    ser = [{k: (enc(v) if isinstance(v, np.ndarray) else v)
            for k, v in c.items() if k != "rates"} | {"rates": {k: np.asarray(v).round(1).tolist()
                                                                for k, v in c["rates"].items()}} for c in cmds]
    (d / "commands.json").write_text(json.dumps(ser), encoding="utf8")
