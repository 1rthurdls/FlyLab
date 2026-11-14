"""The fly rides a bicycle, using the read-out of a real neuroscience rig: the fly walking on a floating ball.

HONEST FRAMING
* Real: the whole-brain model (same manoeuvre stimuli as drive_car: forward = P9, left/right = P9 + that side's
  DNa01/DNa02, ...), the documented brain -> body interface (interface.py) and the physics-simulated NeuroMechFly
  body (flygym) whose legs actually walk. Unlike the car, the vehicle does NOT read the neurons: it reads the
  LEGS. Whatever the six legs achieve in MuJoCo (including stumbles and flygym's slight leftward drift) is what
  the bike does.
* Rig equivalence: in a ball rig the fly is tethered and the ball turns under its feet; the ball's rotation equals
  the walking the fly would do on the ground (forward speed and yaw rate). FlyLab simulates the free-walking fly
  and reads its forward speed and yaw rate in its own body frame, which is that same signal.
* Designed: the bike link (GEAR: bike m/s per fly mm/s; YAW_GAIN), the smoothing (100 ms, removes the stride
  sway), and the bike kinematics. The pedals are animated from the bike speed. Nothing learns; Claude or a
  preset chooses the manoeuvres.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from . import cache, interface, lab
from .car import MANEUVERS, N_TRIALS, _schedule

GEAR = 0.45          # bike speed (m/s) per fly walking speed (mm/s): 13.5 mm/s -> ~22 km/h
YAW_GAIN = 0.6       # bike yaw rate per fly yaw rate
SMOOTH_S = 0.1
WHEELBASE = 1.05     # m, for the handlebar angle shown on the dashboard
PRESETS = {
    "loop": [("forward", 0.8), ("left", 1.6), ("forward", 0.8)],
    "slalom": [("forward", 0.5), ("left", 0.7), ("right", 1.0), ("left", 0.7), ("forward", 0.5)],
    "sprint": [("forward", 2.5)],
    "reverse": [("reverse", 1.5)],
}


def _ball_readout(traj, dt_out=0.02):
    """Body trajectory rows (t, x mm, y mm, heading deg, ...) every ~10 ms -> per-20 ms ball signals:
    forward speed (mm/s, body frame) and yaw rate (deg/s), smoothed over SMOOTH_S."""
    T = np.array([r[:4] for r in traj], float)
    t, x, y, h = T[:, 0], T[:, 1], T[:, 2], np.radians(T[:, 3])
    grid = np.arange(0.0, t[-1] + 1e-9, dt_out)
    xi, yi, hi = (np.interp(grid, t, a) for a in (x, y, h))
    dx, dy, dh = np.diff(xi), np.diff(yi), np.diff(hi)
    hm = (hi[1:] + hi[:-1]) / 2
    v_fwd = (dx * np.cos(hm) + dy * np.sin(hm)) / dt_out
    yaw = np.degrees(dh) / dt_out
    n = max(1, int(round(SMOOTH_S / dt_out)))
    k = np.ones(n) / n
    pad = lambda a: np.concatenate([np.full(n - 1, a[0]), a])  # noqa: E731  causal smoothing
    return grid[1:], np.convolve(pad(v_fwd), k, "valid"), np.convolve(pad(yaw), k, "valid")


def _ride(t, v_fly, yaw_fly):
    x = y = th = 0.0
    rows = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    for i in range(len(t)):
        dt = t[i] - (t[i - 1] if i else 0.0)
        v = GEAR * v_fly[i]
        w = math.radians(YAW_GAIN * yaw_fly[i])
        th += w * dt
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        steer = math.atan(WHEELBASE * w / v) if abs(v) > 0.3 else 0.0
        rows.append([round(float(t[i]), 3), round(x, 3), round(y, 3), round(th, 4), round(v, 3),
                     round(max(-0.6, min(0.6, steer)), 4)])
    return rows


def ride_bike(preset: str | None = "loop", maneuvers: list[dict] | None = None,
              silence: list[str] | None = None) -> dict:
    silence = silence or []
    if maneuvers:
        label = "custom"
    else:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset '{preset}'. Available: {list(PRESETS)}")
        maneuvers = [{"action": a, "seconds": s} for a, s in PRESETS[preset]]
        label = preset
    maneuvers = [{"action": m["action"], "seconds": float(m["seconds"])} for m in maneuvers]
    if sum(m["seconds"] for m in maneuvers) > 5:
        raise ValueError("total ride longer than 5 s (the leg physics costs ~1.5 min per simulated second)")
    params = {"maneuvers": maneuvers, "silence": sorted(silence), "gear": GEAR, "yaw": YAW_GAIN, "v": 1}
    lab_ = f"bike_{label}" + (("__silence-" + "+".join(silence)) if silence else "")
    run_id = cache.run_id_for("beh", lab_, params)
    cached = cache.load_result(run_id)
    if cached is not None and cached.get("video") and (cache.REPO / cached["video"]).exists():
        cache.save_result(run_id, cached)
        return cached
    t0 = time.perf_counter()
    sched, D = _schedule(maneuvers)
    sim = lab.run_simulation([], silence, 200.0, round(D, 3), N_TRIALS, schedule=sched, label=f"{lab_}_brain")
    cmd = interface.motor_commands(sim["timeseries"], sim["timeseries_bin_s"])
    from .behaviors import _clean, _save_cmds
    from .body import FlyBody
    body = FlyBody(window_size=(800, 600), cam_distance=9.0)
    r = body.run(interface.to_program(cmd))
    d = cache.run_dir(run_id)
    body.save_video(d / "video.mp4")
    body.close()
    t, v_fly, yaw_fly = _ball_readout(body.trajectory)
    traj = _ride(t, v_fly, yaw_fly)
    T = np.array(traj)
    segs, acc = [], 0.0
    for m in maneuvers:
        sel = (t >= acc) & (t < acc + m["seconds"])
        rows = T[(T[:, 0] >= acc) & (T[:, 0] <= acc + m["seconds"])]
        dist = float(np.sum(np.hypot(np.diff(rows[:, 1]), np.diff(rows[:, 2])))) if len(rows) > 1 else 0.0
        b0, b1 = int(round(acc / cmd["bin_s"])), int(round((acc + m["seconds"]) / cmd["bin_s"]))
        segs.append({"t": [round(acc, 2), round(acc + m["seconds"], 2)], "action": m["action"],
                     "neurons_stimulated": [g for g, _ in MANEUVERS[m["action"]]],
                     "brain_to_legs": interface.describe(cmd, b0, max(b1, b0 + 1)),
                     "ball": {"fly_speed_mm_s": round(float(np.mean(v_fly[sel])), 1) if sel.any() else 0.0,
                              "fly_yaw_deg_s": round(float(np.mean(yaw_fly[sel])), 0) if sel.any() else 0.0},
                     "bike": {"distance_m": round(dist, 1),
                              "heading_change_deg": round(float(np.degrees(rows[-1, 3] - rows[0, 3])), 0)
                              if len(rows) > 1 else 0.0}})
        acc += m["seconds"]
    (d / "car.json").write_text(json.dumps({
        "mode": "bike", "trajectory": traj, "maneuvers": maneuvers,
        "ball": [[round(float(a), 3), round(float(b), 2), round(float(c), 1)] for a, b, c in zip(t, v_fly, yaw_fly)],
        "video": f"runs/{run_id}/video.mp4", "video_play_speed": body.play_speed, "gear": GEAR}), encoding="utf8")
    _save_cmds(d, [cmd])
    dist_total = float(np.sum(np.hypot(np.diff(T[:, 1]), np.diff(T[:, 2]))))
    summ = "; ".join(f"{s['t'][0]}-{s['t'][1]} s {s['action'].upper()}: legs {s['ball']['fly_speed_mm_s']} mm/s, "
                     f"yaw {s['ball']['fly_yaw_deg_s']:+.0f} deg/s -> bike {s['bike']['distance_m']} m, turned "
                     f"{s['bike']['heading_change_deg']:+.0f} deg" for s in segs)
    out = {"run_id": run_id, "kind": "behavior", "behavior": f"bike_{label}", "scene": "car", "label": lab_,
           "description": "Brain -> legs (flygym physics) -> ball-rig read-out (forward speed, yaw) -> bicycle.",
           "silenced": silence, "maneuvers": maneuvers, "segments": segs, "phases": [],
           "brain_runs": [sim["run_id"]], "raster_png": sim["raster_png"], "video": f"runs/{run_id}/video.mp4",
           "video_play_speed": body.play_speed, "car": f"runs/{run_id}/car.json",
           "distance_m": round(dist_total, 1), "top_speed_kmh": round(float(np.max(np.abs(T[:, 4]))) * 3.6, 0),
           "legs": {"distance_walked_mm": r.get("distance_walked_mm"), "heading_change_deg": r.get("heading_change_deg")},
           "body": _clean(r),
           "summary": (f"LESION: silenced {', '.join(silence)}. " if silence else "") + summ +
                      f". Total {dist_total:.0f} m.",
           "wall_s": round(time.perf_counter() - t0, 1),
           "framing": "Brain and legs are real simulations; the ball read-out is the fly's own walking; the bike "
                      "link (gear, yaw gain) and the bike are designed. Nothing learns."}
    cache.save_result(run_id, out)
    return out


if __name__ == "__main__":
    import sys
    r = ride_bike(sys.argv[1] if len(sys.argv) > 1 else "loop", silence=sys.argv[2:] or None)
    print(r["summary"], "| wall", r["wall_s"], "s")
