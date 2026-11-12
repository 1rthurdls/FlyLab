"""Closed-loop obstacle courses: the fly brain steers a car between SOLID walls, with no planner at all.

HONEST FRAMING
* What is real: the whole-brain model (FlyWire v783, Shiu et al. LIF), run as ONE continuous trial advanced
  in 20 ms chunks (BrainStepper, spike-identical to the one-shot kernel). Every steering decision is whatever the
  brain's DNa01/DNa02 neurons fire in response to its own visual input. Nobody chooses manoeuvres.
* What is designed (documented here):
    - the "eyes": rays cast from the car's nose in each visual hemifield measure how near the walls are;
      nearness -> Poisson rate of the LPLC1 visual projection neurons of THAT side (0-100 Hz).
    - the walking drive: P9 (forward-walking command) is held at 100 Hz the whole time.
    - the car link: interface.py (P9 -> throttle, DNa L-R -> steering), the same as drive_car.
    - the course geometry. Walls are solid: touching one ends the run ("CRASH"). No corner can be cut.
* Why LPLC1: measured in this model (P9 100 Hz + one-sided LPLC1, 0.5 s, 4 trials): LPLC1 on the LEFT at
  25/50 Hz drives right DNa01 31/46 Hz and left DNa01 0 Hz -> turn away (right); mirror-symmetric for the right
  side; equal input on both sides cancels (turn -0.07); the giant fiber stays ~0 Hz (no escape take-off).
  Turning away from the nearer wall is therefore a property of the connectome, not of this file.
* Nothing learns. The same brain with LPLC1 silenced is the control.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from . import cache, interface, lab
from .brain import BrainStepper, SimResult, get_brain

CHUNK_S = 0.02  # brain <-> world exchange every 20 ms (= interface bin)
STEPS = int(round(CHUNK_S * 1000 / 0.1))
P9_HZ = 100.0
EYE_MAX_HZ = 100.0
RAY_RANGE = 14.0  # m: walls farther than this are invisible
RAY_NEAR = 2.0  # m: nearness saturates here
RAY_ANGLES = np.radians(np.arange(4.0, 90.0, 6.0))  # per hemifield, from straight ahead to the side
CAR_HALF_LEN, CAR_RADIUS = 1.7, 0.95  # car = capsule (spine +-1.7 m, radius 0.95 m)
V_FWD = 7.0  # m/s at full P9 drive
WHEELBASE, MAX_STEER, TAU_V, TAU_S = 2.6, math.radians(32), 0.5, 0.12

# Courses: centerline pieces ("S", length) straight or ("A", radius, degrees) arc (+ = left), corridor width.
COURSES = {
    "canyon": {"width": 10.0, "pieces": [("S", 16), ("A", 15, 90), ("S", 8), ("A", 15, -90), ("S", 6),
                                         ("A", 15, -70), ("A", 15, 70), ("S", 14)],
               "description": "Winding canyon: left bend, right bend, then an S-chicane. ~125 m of solid walls."},
    "hairpin": {"width": 12.0, "pieces": [("S", 14), ("A", 13, 90), ("S", 4), ("A", 13, 90), ("S", 16)],
                "description": "Two left bends making a 180 degree hairpin."},
    "slalom": {"width": 10.0, "pieces": [("S", 10)] + [("A", 16, s * 55) for s in (1, -1, -1, 1, 1, -1)] + [("S", 12)],
               "description": "Tight serpentine corridor, alternating left and right bends."},
}
INPUTS = {"P9": "P9", "eye_L": "LPLC1_L", "eye_R": "LPLC1_R"}


# ----------------------------------------------------------------------------- geometry
def build_course(name: str) -> dict:
    C = COURSES[name]
    x = y = th = 0.0
    pts = [(x, y, th)]
    for p in C["pieces"]:
        if p[0] == "S":
            n = max(1, int(p[1] / 0.5))
            for _ in range(n):
                x += math.cos(th) * p[1] / n
                y += math.sin(th) * p[1] / n
                pts.append((x, y, th))
        else:
            R, ang = p[1], math.radians(p[2])
            n = max(2, int(abs(ang) * R / 0.5))
            for _ in range(n):
                th += ang / n
                x += math.cos(th - ang / n / 2) * R * abs(ang) / n
                y += math.sin(th - ang / n / 2) * R * abs(ang) / n
                pts.append((x, y, th))
    P = np.array(pts)
    h = C["width"] / 2
    nrm = np.stack([-np.sin(P[:, 2]), np.cos(P[:, 2])], 1)
    left = P[:, :2] + h * nrm
    right = P[:, :2] - h * nrm
    start_back = P[0, :2] - 6 * np.array([math.cos(P[0, 2]), math.sin(P[0, 2])])
    lb = start_back + h * nrm[0]
    rb = start_back - h * nrm[0]
    left = np.vstack([lb, left])
    right = np.vstack([rb, right])
    segs = np.vstack([np.hstack([left[:-1], left[1:]]), np.hstack([right[:-1], right[1:]]),
                      [[lb[0], lb[1], rb[0], rb[1]]]])  # back wall closes the start
    seglen = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    s_cum = np.concatenate([[0], np.cumsum(seglen)])
    return {"name": name, "width": C["width"], "centerline": P, "s": s_cum, "length": float(s_cum[-1]),
            "segments": segs, "left_wall": left, "right_wall": right,
            "finish": [left[-1].tolist(), right[-1].tolist()], "description": C["description"]}


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def raycast(ox, oy, angles, segs, rmax=RAY_RANGE):
    """Distance from (ox, oy) along each world angle to the nearest wall (rmax if none)."""
    dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]
    x1, y1, x2, y2 = (segs[:, k][None] for k in range(4))
    ex, ey = x2 - x1, y2 - y1
    den = _cross(dx, dy, ex, ey)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(x1 - ox, y1 - oy, ex, ey) / den
        u = _cross(x1 - ox, y1 - oy, dx, dy) / den
    ok = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf)
    return np.minimum(t.min(1), rmax)


def _pt_seg(px, py, segs):
    x1, y1, x2, y2 = segs.T
    ex, ey = x2 - x1, y2 - y1
    L2 = ex * ex + ey * ey
    u = np.clip(((px - x1) * ex + (py - y1) * ey) / np.maximum(L2, 1e-12), 0, 1)
    return np.hypot(x1 + u * ex - px, y1 + u * ey - py)


def collides(x, y, th, segs) -> bool:
    c, s = math.cos(th), math.sin(th)
    ax, ay = x - CAR_HALF_LEN * c, y - CAR_HALF_LEN * s
    bx, by = x + CAR_HALF_LEN * c, y + CAR_HALF_LEN * s
    for k in np.linspace(0, 1, 5):
        if _pt_seg(ax + k * (bx - ax), ay + k * (by - ay), segs).min() < CAR_RADIUS:
            return True
    return False


def eyes(x, y, th, segs):
    """Designed eye model: per hemifield, nearness = max over rays of cos(angle) * clip((RANGE - d)/(RANGE - NEAR)).
    Returns (rate_left_Hz, rate_right_Hz, ray distances left, ray distances right)."""
    nx, ny = x + CAR_HALF_LEN * math.cos(th), y + CAR_HALF_LEN * math.sin(th)
    dl = raycast(nx, ny, th + RAY_ANGLES, segs)
    dr = raycast(nx, ny, th - RAY_ANGLES, segs)
    w = np.cos(RAY_ANGLES)
    near = lambda d: np.clip((RAY_RANGE - d) / (RAY_RANGE - RAY_NEAR), 0, 1)  # noqa: E731
    return (EYE_MAX_HZ * float((w * near(dl)).max()), EYE_MAX_HZ * float((w * near(dr)).max()), dl, dr)


def progress(course, x, y) -> float:
    P = course["centerline"]
    i = int(np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)))
    return float(course["s"][i])


# ----------------------------------------------------------------------------- closed loop
def run_course(course: str = "canyon", silence: list[str] | None = None, max_s: float = 25.0, seed: int = 0) -> dict:
    if course not in COURSES:
        raise ValueError(f"unknown course '{course}'. Available: {list(COURSES)}")
    silence = silence or []
    params = {"course": course, "pieces": COURSES[course]["pieces"], "width": COURSES[course]["width"],
              "silence": sorted(silence), "max_s": max_s, "seed": seed, "eye": [RAY_RANGE, RAY_NEAR, EYE_MAX_HZ],
              "v": 1}
    lab_ = f"course_{course}" + (("__silence-" + "+".join(silence)) if silence else "")
    run_id = cache.run_id_for("beh", lab_, params)
    cached = cache.load_result(run_id)
    if cached is not None:
        cache.save_result(run_id, cached)
        return cached
    t_wall = time.perf_counter()
    W = build_course(course)
    segs = W["segments"]
    brain = get_brain()
    inputs = {k: lab.resolve(v)[0] for k, v in INPUTS.items()}
    sil_idx, sil_info = lab._resolve_many(silence)
    st = BrainStepper(brain, inputs, silence=sil_idx, seed=seed)
    gi = lab._group_indices()
    from .groups import OUTPUT_GROUPS
    ts = {g: [] for g in OUTPUT_GROUPS}
    eye_log, traj, all_s, all_n = [], [], [], []
    counts = np.zeros(brain.N, np.int64)
    x = y = th = v = steer = 0.0
    traj.append([0.0, x, y, th, v, steer])
    outcome, t_end, crash_at = "timeout", max_s, None
    n_chunks = int(round(max_s / CHUNK_S))
    sm = interface.SMOOTH_BINS
    for k in range(n_chunks):
        rl, rr, _, _ = eyes(x, y, th, segs)
        eye_log.append([round(rl, 1), round(rr, 1)])
        c, s_, n_ = st.step({"P9": P9_HZ, "eye_L": rl, "eye_R": rr}, STEPS)
        counts += c
        all_s.append(s_); all_n.append(n_)
        for g in OUTPUT_GROUPS:
            ts[g].append(float(c[gi[g]].sum()) / max(len(gi[g]), 1) / CHUNK_S)
        # same smoothing/mapping as interface.motor_commands (causal 3-bin mean)
        mean = lambda g: float(np.mean(([ts[g][0]] * sm + ts[g])[-sm:]))  # noqa: E731
        fwd = min(max(mean("P9") / interface.P9_FULL_HZ, 0), 1)
        Lh = mean("DNa01_left") + mean("DNa02_left")
        Rh = mean("DNa01_right") + mean("DNa02_right")
        turn = min(max((Lh - Rh) / interface.TURN_FULL_HZ, -1), 1)
        # car moves during this chunk (sub-steps of 5 ms), walls are solid
        crashed = False
        for _ in range(4):
            dt = CHUNK_S / 4
            v += (V_FWD * fwd - v) * min(1.0, dt / TAU_V)
            steer += (MAX_STEER * turn - steer) * min(1.0, dt / TAU_S)
            th2 = th + v / WHEELBASE * math.tan(steer) * dt
            x2, y2 = x + v * math.cos(th2) * dt, y + v * math.sin(th2) * dt
            if collides(x2, y2, th2, segs):
                crashed = True
                v = 0.0
                break
            x, y, th = x2, y2, th2
        t = round((k + 1) * CHUNK_S, 3)
        traj.append([t, round(x, 3), round(y, 3), round(th, 4), round(v, 3), round(steer, 4)])
        if crashed and crash_at is None:
            crash_at = (t, round(x, 2), round(y, 2))
            outcome, t_end = "crash", t
        if crash_at and t >= crash_at[0] + 0.6:  # keep the brain running briefly after impact, then stop
            break
        if not crash_at and progress(W, x, y) >= W["length"] - 1.0:
            outcome, t_end = "finished", t
            break
    D = round(len(ts["P9"]) * CHUNK_S, 3)
    # ---- brain run record (same format as run_simulation, so raster/3D brain/dashboard work unchanged)
    S = np.concatenate(all_s); Nn = np.concatenate(all_n)
    res = SimResult(duration_s=D, n_trials=1, counts=counts[None].astype(np.int32), spike_step=S,
                    spike_trial=np.zeros(len(S), np.int16), spike_neuron=Nn, wall_s=time.perf_counter() - t_wall,
                    meta={"backend": "numba-active-set closed-loop (BrainStepper)"})
    sim_params = {"activate": ["P9", "LPLC1_L", "LPLC1_R"], "silence": sorted(silence), "closed_loop": True,
                  "course": course, "run": run_id}
    sim_id = cache.run_id_for("sim", f"{lab_}_brain", sim_params)
    act_idx = sorted({i for v_ in inputs.values() for i in v_})
    act_info = {"P9 (100 Hz)": len(inputs["P9"]), "LPLC1_L (eye, closed loop)": len(inputs["eye_L"]),
                "LPLC1_R (eye, closed loop)": len(inputs["eye_R"])}
    sim = lab._summarize(sim_id, res, act_idx, act_info, sil_idx, sil_info, sim_params, f"{lab_}_brain")
    cache.save_result(sim_id, sim)
    cmd = interface.motor_commands({g: np.array(v_) for g, v_ in ts.items()}, CHUNK_S)
    # ---- summary numbers
    T = np.array(traj)
    E = np.array(eye_log)
    prog = progress(W, x, y)
    dist = float(np.sum(np.hypot(np.diff(T[:, 1]), np.diff(T[:, 2]))))
    clear = [float(_pt_seg(r[1], r[2], segs).min()) - CAR_RADIUS for r in traj[::5]]
    asym = E[:, 0] - E[:, 1]
    tn = np.asarray(cmd["turn"])
    m = np.abs(asym) > 10
    agree = float(np.mean(np.sign(tn[m]) == -np.sign(asym[m]))) if m.any() else float("nan")
    d = cache.run_dir(run_id)
    (d / "car.json").write_text(json.dumps({
        "trajectory": traj, "walls": segs.round(3).tolist(), "finish": W["finish"],
        "centerline": W["centerline"][::4, :2].round(2).tolist(), "eyes_hz": eye_log, "maneuvers": [],
        "mode": "course", "course": course, "outcome": outcome, "crash": crash_at,
        "ray_angles_deg": np.degrees(RAY_ANGLES).round(1).tolist(), "ray_range": RAY_RANGE}), encoding="utf8")
    from .behaviors import _save_cmds
    cmd_s = dict(cmd)
    cmd_s["rates"] = dict(cmd["rates"]) | {"eye_L": E[:, 0], "eye_R": E[:, 1]}
    _save_cmds(d, [cmd_s])
    pct = 100 * prog / W["length"]
    if outcome == "finished":
        verdict = f"FINISHED the {course} course ({W['length']:.0f} m) in {t_end:.1f} s without touching a wall"
    elif outcome == "crash":
        verdict = f"CRASHED into a wall at t = {crash_at[0]:.1f} s after {prog:.0f} m ({pct:.0f}% of the course)"
    else:
        verdict = f"did not finish in {max_s:.0f} s ({prog:.0f} m, {pct:.0f}% of the course)"
    out = {"run_id": run_id, "kind": "behavior", "behavior": f"course_{course}", "scene": "car", "label": lab_,
           "description": "Closed loop, no planner: wall nearness in each eye -> LPLC1 of that side -> whole brain "
                          "-> DNa01/DNa02 -> steering; P9 held on for forward drive. Walls are solid.",
           "silenced": silence, "course": course, "course_length_m": round(W["length"], 1),
           "outcome": outcome, "time_s": t_end, "progress_m": round(prog, 1), "progress_pct": round(pct, 0),
           "distance_driven_m": round(dist, 1), "crash": crash_at,
           "min_wall_clearance_m": round(min(clear), 2),
           "eyes_mean_hz": {"LPLC1_L": round(float(E[:, 0].mean()), 1), "LPLC1_R": round(float(E[:, 1].mean()), 1)},
           "steer_away_agreement": round(agree, 2) if agree == agree else None,
           "brain": {"DNa_left_mean_hz": round(float(np.mean(cmd["rates"]["DNa_left"])), 1),
                     "DNa_right_mean_hz": round(float(np.mean(cmd["rates"]["DNa_right"])), 1),
                     "P9_mean_hz": round(float(np.mean(cmd["rates"]["P9"])), 1),
                     "giant_fiber_mean_hz": round(float(np.mean(cmd["rates"]["GF"])), 1),
                     "active_neurons": sim["sim"]["n_active_neurons"], "spikes": sim["sim"]["total_spikes"]},
           "phases": [], "segments": [], "maneuvers": [], "brain_runs": [sim_id], "raster_png": sim["raster_png"],
           "video": None, "car": f"runs/{run_id}/car.json",
           "summary": (f"LESION: silenced {', '.join(silence)}. " if silence else "") + verdict +
                      f". Eyes (LPLC1 L/R) mean {E[:, 0].mean():.0f}/{E[:, 1].mean():.0f} Hz; brain DNa L/R mean "
                      f"{np.mean(cmd['rates']['DNa_left']):.0f}/{np.mean(cmd['rates']['DNa_right']):.0f} Hz; " +
                      (f"when one eye saw the nearer wall (>10 Hz difference) the brain steered away "
                       f"{100 * agree:.0f}% of the time." if agree == agree else "."),
           "wall_s": round(time.perf_counter() - t_wall, 1),
           "framing": "Brain is real and runs closed loop (no planner, nobody picks manoeuvres); eye model, P9 drive, "
                      "car link and course are designed. Nothing learns."}
    cache.save_result(run_id, out)
    return out


if __name__ == "__main__":
    import sys
    r = run_course(sys.argv[1] if len(sys.argv) > 1 else "canyon", sys.argv[2:] or None)
    print(r["summary"], "| wall", r["wall_s"], "s")
