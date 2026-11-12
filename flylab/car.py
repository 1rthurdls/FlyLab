"""The fly brain drives a car.

HONEST FRAMING
* What is real: the whole-brain model (FlyWire connectome, Shiu et al. LIF). Each manoeuvre stimulates real
  descending neurons with Poisson input, exactly like run_simulation:
      forward       -> P9 (forward-walking command)          100 Hz
      left / right  -> P9 + DNa01 + DNa02 of that side        100 Hz
      reverse       -> MDN (backward-walking command)         100 Hz
      reverse_left / reverse_right -> MDN + that side's DNa01/DNa02
      stop          -> nothing
  The car is driven by whatever those output neurons ACTUALLY fire in the simulation (including side effects,
  e.g. P9 also recruits DNa02_right more than DNa02_left, so the car pulls slightly to the right).
* What is designed: the car link reuses interface.py (P9 -> throttle, MDN -> reverse, DNa left-right ->
  steering wheel) feeding a simple kinematic bicycle-model car. Who chooses the manoeuvres is Claude (or the
  small planner below for goal = "station"). The fly does not learn to drive; this is an analogy that shows the
  brain's real steering/speed commands controlling a vehicle instead of legs. Others have shown flies "driving"
  online, but we found no primary source; this is FlyLab's own transparent version.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from . import cache, interface, lab

MANEUVERS = {
    "forward": [("P9", 100.0)],
    "left": [("P9", 100.0), ("DNa01_left", 100.0), ("DNa02_left", 100.0)],
    "right": [("P9", 100.0), ("DNa01_right", 100.0), ("DNa02_right", 100.0)],
    "reverse": [("MDN", 100.0)],
    "reverse_left": [("MDN", 100.0), ("DNa01_left", 100.0), ("DNa02_left", 100.0)],
    "reverse_right": [("MDN", 100.0), ("DNa01_right", 100.0), ("DNa02_right", 100.0)],
    "stop": [],
}
# car (metres, seconds)
WHEELBASE = 2.6
V_FWD, V_REV = 8.0, 3.5
MAX_STEER = math.radians(32)
TAU_V, TAU_S = 0.5, 0.12
STATION = (26.0, 16.0)
STATION_RADIUS = 2.5
N_TRIALS = 4
PRESETS = {
    "u_turn": [("forward", 1.2), ("left", 2.4), ("forward", 1.2)],
    "slalom": [("forward", 0.6), ("left", 0.5), ("right", 1.0), ("left", 1.0), ("right", 1.0), ("left", 0.5),
               ("forward", 0.6)],
    "parallel_park": [("forward", 1.0), ("stop", 0.4), ("reverse_right", 1.0), ("reverse_left", 1.0), ("stop", 0.5)],
    "figure_eight": [("left", 3.0), ("right", 3.0)],
}


def _schedule(maneuvers):
    sched, t = [], 0.0
    for m in maneuvers:
        a, dur = m["action"], float(m["seconds"])
        if a not in MANEUVERS:
            raise ValueError(f"unknown action '{a}'. Use one of {list(MANEUVERS)}")
        for g, r in MANEUVERS[a]:
            sched.append({"neurons": [g], "rate_hz": r, "t0": round(t, 3), "t1": round(t + dur, 3)})
        t += dur
    return sched, t


def _drive(cmd: dict) -> list[list[float]]:
    """Integrate a kinematic bicycle model from per-bin commands. Returns [t, x, y, heading, v, steer] rows."""
    dt = cmd["bin_s"]
    x = y = th = v = st = 0.0
    out = [[0.0, x, y, th, v, st]]
    for i in range(cmd["n_bins"]):
        v_goal = V_FWD * cmd["forward"][i] - V_REV * cmd["backward"][i]
        s_goal = MAX_STEER * cmd["turn"][i]
        v += (v_goal - v) * min(1.0, dt / TAU_V)
        st += (s_goal - st) * min(1.0, dt / TAU_S)
        th += v / WHEELBASE * math.tan(st) * dt
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        out.append([round((i + 1) * dt, 3), round(x, 3), round(y, 3), round(th, 4), round(v, 3), round(st, 4)])
    return out


def _calibration() -> dict:
    """Mean motor command each manoeuvre produces in the brain model (used only by the goal planner)."""
    cal = {}
    for a, stim in MANEUVERS.items():
        if not stim:
            cal[a] = (0.0, 0.0, 0.0)
            continue
        sched = [{"neurons": [g], "rate_hz": r, "t0": 0.0, "t1": None} for g, r in stim]
        r = lab.run_simulation([], [], 200.0, 0.5, N_TRIALS, schedule=sched, label=f"car_cal_{a}")
        c = interface.motor_commands(r["timeseries"], r["timeseries_bin_s"])
        k = slice(5, None)  # skip onset
        cal[a] = (float(np.mean(c["forward"][k])), float(np.mean(c["turn"][k])), float(np.mean(c["backward"][k])))
    return cal


STEP_S = 0.25


def _decide(x, y, th, v, target):
    dx, dy = target[0] - x, target[1] - y
    if math.hypot(dx, dy) < STATION_RADIUS * 0.3 + abs(v) * TAU_V:  # leave room to roll to a stop
        return "stop"
    err = (math.atan2(dy, dx) - th + math.pi) % (2 * math.pi) - math.pi
    return "left" if err > math.radians(8) else "right" if err < -math.radians(8) else "forward"


def _greedy(state, cal, target, t0, max_s=12.0):
    """Greedy STEP_S decisions from `state`, predicting the car with the brain's calibrated mean response."""
    x, y, th, v, st, cf, ct, cb = state
    acts, t = [], t0
    while t < max_s:
        a = _decide(x, y, th, v, target)
        acts.append(a)
        if a == "stop":
            break
        f, tn, b = cal[a]
        for _ in range(int(round(STEP_S / 0.02))):
            k = 0.02 / 0.06  # brain latency + interface smoothing
            cf += (f - cf) * k; ct += (tn - ct) * k; cb += (b - cb) * k
            v += (V_FWD * cf - V_REV * cb - v) * (0.02 / TAU_V)
            st += (MAX_STEER * ct - st) * min(1.0, 0.02 / TAU_S)
            th += v / WHEELBASE * math.tan(st) * 0.02
            x += v * math.cos(th) * 0.02
            y += v * math.sin(th) * 0.02
        t += STEP_S
    return acts


def _merge(acts):
    plan = []
    for a in acts:
        dur = 0.75 if a == "stop" else STEP_S
        if plan and plan[-1]["action"] == a:
            plan[-1]["seconds"] = round(plan[-1]["seconds"] + dur, 2)
        else:
            plan.append({"action": a, "seconds": dur})
    return plan


def _run_real(maneuvers, silence, label):
    sched, D = _schedule(maneuvers)
    sim = lab.run_simulation([], silence, 200.0, round(D, 3), N_TRIALS, schedule=sched, label=label)
    cmd = interface.motor_commands(sim["timeseries"], sim["timeseries_bin_s"])
    return sim, cmd, _drive(cmd)


def plan_to(target=STATION, silence=None, max_iter=4) -> tuple[list[dict], list[dict]]:
    """Closed-loop planning on the REAL brain output: plan greedily, simulate the whole brain, replay the
    decisions along the car's actual path, re-plan from the first divergence, repeat."""
    silence = silence or []
    cal = _calibration()
    acts = _greedy((0, 0, 0, 0, 0, 0, 0, 0), cal, target, 0.0)
    log = []
    best = None
    for it in range(max_iter):
        plan = _merge(acts)
        sim, cmd, traj = _run_real(plan, silence, "drive_car_station_plan")
        T = np.array(traj)
        d_end = math.hypot(target[0] - T[-1, 1], target[1] - T[-1, 2])
        log.append({"iteration": it + 1, "plan": plan, "end_distance_m": round(d_end, 1)})
        if best is None or d_end < best[0]:
            best = (d_end, plan)
        if d_end < STATION_RADIUS:
            break
        # first step where the decision taken along the REAL trajectory differs from the plan
        for k, a in enumerate(acts):
            i = min(int(round(k * STEP_S / 0.02)), len(T) - 1)
            _, x, y, th, v, st = T[i]
            j = max(i - 1, 0)
            a_real = _decide(x, y, th, v, target)
            if a_real != a:
                state = (x, y, th, v, st, cmd["forward"][j], cmd["turn"][j], cmd["backward"][j])
                acts = acts[:k] + _greedy(state, cal, target, k * STEP_S)
                break
        else:
            break
    return best[1], log


def drive_car(maneuvers: list[dict] | None = None, preset: str | None = None, goal: str | None = None,
              silence: list[str] | None = None) -> dict:
    """maneuvers: [{"action": "forward"|"left"|"right"|"reverse"|"reverse_left"|"reverse_right"|"stop",
    "seconds": float}, ...]  or preset (u_turn, slalom, parallel_park, figure_eight) or goal="station"."""
    silence = silence or []
    plan_log = None
    if goal == "station":
        maneuvers, plan_log = plan_to(STATION, silence)
        label = "station"
    elif preset:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset '{preset}'. Available: {list(PRESETS)}")
        maneuvers = [{"action": a, "seconds": s} for a, s in PRESETS[preset]]
        label = preset
    elif maneuvers:
        label = "custom"
    else:
        raise ValueError("give maneuvers, a preset or goal='station'")
    maneuvers = [{"action": m["action"], "seconds": float(m["seconds"])} for m in maneuvers]
    if sum(m["seconds"] for m in maneuvers) > 15:
        raise ValueError("total drive longer than 15 s; split it up")
    params = {"maneuvers": maneuvers, "silence": sorted(silence), "v": 1}
    lab_ = f"drive_car_{label}" + (("__silence-" + "+".join(silence)) if silence else "")
    run_id = cache.run_id_for("beh", lab_, params)
    cached = cache.load_result(run_id)
    if cached is not None:
        cache.save_result(run_id, cached)
        return cached
    t0 = time.perf_counter()
    sched, D = _schedule(maneuvers)
    sim = lab.run_simulation([], silence, 200.0, round(D, 3), N_TRIALS, schedule=sched, label=f"{lab_}_brain")
    cmd = interface.motor_commands(sim["timeseries"], sim["timeseries_bin_s"])
    traj = _drive(cmd)
    T = np.array(traj)
    segs, t = [], 0.0
    for m in maneuvers:
        b0, b1 = int(round(t / cmd["bin_s"])), int(round((t + m["seconds"]) / cmd["bin_s"]))
        rows = T[(T[:, 0] >= t) & (T[:, 0] <= t + m["seconds"])]
        dist = float(np.sum(np.hypot(np.diff(rows[:, 1]), np.diff(rows[:, 2])))) if len(rows) > 1 else 0.0
        dh = float(np.degrees(rows[-1, 3] - rows[0, 3])) if len(rows) > 1 else 0.0
        segs.append({"t": [round(t, 2), round(t + m["seconds"], 2)], "action": m["action"],
                     "neurons_stimulated": [g for g, _ in MANEUVERS[m["action"]]],
                     "brain_to_car": interface.describe(cmd, b0, max(b1, b0 + 1)),
                     "car": {"distance_m": round(dist, 1), "heading_change_deg": round(dh, 0)}})
        t += m["seconds"]
    end = T[-1]
    d_station = math.hypot(STATION[0] - end[1], STATION[1] - end[2])
    d = cache.run_dir(run_id)
    (d / "car.json").write_text(json.dumps({"trajectory": traj, "station": STATION, "station_radius": STATION_RADIUS,
                                            "maneuvers": maneuvers}), encoding="utf8")
    from .behaviors import _save_cmds
    _save_cmds(d, [cmd])
    summ = "; ".join(f"{s['t'][0]}-{s['t'][1]} s {s['action'].upper()}: car {s['car']['distance_m']} m, "
                     f"turned {s['car']['heading_change_deg']:+.0f} deg" for s in segs)
    out = {"run_id": run_id, "kind": "behavior", "behavior": "drive_car", "scene": "car", "label": lab_,
           "description": "The fly brain's steering/speed neurons drive a car (hand-designed car link).",
           "silenced": silence, "maneuvers": maneuvers, "phases": [], "segments": segs,
           "brain_runs": [sim["run_id"]], "raster_png": sim["raster_png"], "video": None,
           "car": f"runs/{run_id}/car.json",
           "end_position_m": [round(float(end[1]), 1), round(float(end[2]), 1)],
           "station_m": list(STATION), "distance_to_station_m": round(d_station, 1),
           "reached_station": d_station < STATION_RADIUS,
           "summary": (f"LESION: silenced {', '.join(silence)}. " if silence else "") + summ +
                      f". End {d_station:.1f} m from the sugar station"
                      + (" -> ARRIVED." if d_station < STATION_RADIUS else "."),
           "wall_s": round(time.perf_counter() - t0, 1),
           "planning_log": plan_log,
           "framing": "Brain is real (whole-brain LIF); car link + manoeuvre choice are designed, nothing learns."}
    cache.save_result(run_id, out)
    return out


if __name__ == "__main__":
    import sys
    r = drive_car(preset=sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "station" else drive_car(goal="station")
    print(r["summary"])
