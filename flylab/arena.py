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

from . import cache, interface, lab, looming
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
    # ---- themed fly-throughs (obstacles = solid pillars: (arc length s, lateral offset to the left, radius), m)
    "cave": {"width": 14.0, "theme": "cave", "avatar": "fly", "speed": 6.0,
             "pieces": [("S", 14), ("A", 16, 60), ("S", 8), ("A", 16, -100), ("S", 8), ("A", 18, 50), ("S", 18)],
             # pillars only on straights: in a bend a pillar either cancels the bend's own visual cue (inside)
             # or ends up dead ahead where both eyes see it equally (outside); measured crashes in both cases
             # (a pillar right after a bend must sit on the bend's INSIDE: the fly exits a bend angled outward)
             "obstacles": [(9, 3.0, 1.0), (35, -3.0, 1.0), (72, -3.0, 1.0), (97, -3.0, 1.1)],
             "description": "Stalactite cave: winding tunnel with rock pillars alternating left and right."},
    "forest": {"width": 18.0, "theme": "forest", "avatar": "fly", "speed": 6.0,
               "pieces": [("S", 18), ("A", 30, 45), ("S", 10), ("A", 30, -70), ("S", 10), ("A", 30, 25), ("S", 16)],
               "obstacles": [(16, 4.0, 0.8), (26, -4.2, 0.9), (36, -4.6, 0.7), (48, -4.0, 0.8), (60, 4.2, 0.9),
                             (72, 4.4, 0.8), (84, 4.0, 0.7), (96, -4.2, 0.9), (106, -4.4, 0.8), (118, 4.0, 0.8)],
               "description": "Forest trail: tree trunks alternating on either side of a wide winding path."},
    # ---- sugar maze: hedge corridor with sugar and poisoned (sugar + bitter) pellets on the path
    "garden": {"width": 11.0, "theme": "garden", "avatar": "fly", "speed": 5.0,
               "pieces": [("S", 12), ("A", 14, 90), ("S", 8), ("A", 14, -90), ("S", 8), ("A", 14, -90), ("S", 8),
                          ("A", 14, 90), ("S", 12)],
               "pellets": [(8, 0.0, "sugar"), (38, 0.0, "poison"), (68, 0.0, "sugar"), (98, 0.0, "poison"),
                           (128, 0.0, "sugar")],
               "description": "Hedge maze corridor with 3 sugar pellets and 2 poisoned (sugar + bitter) pellets."},
}
INPUTS = {"P9": "P9", "eye_L": "LPLC1_L", "eye_R": "LPLC1_R"}
# bat (fly-throughs): head-on swoop; looming -> LPLC2/LC4 (looming.py); giant fiber > 50 Hz in a 20 ms chunk = dodge
BAT_D0, BAT_V, BAT_R, BAT_CATCH_M = 40.0, 16.0, 1.6, 1.5
# a dodge only counts while the bat really looms (drive >= BAT_MIN_LOOM_HZ). Measured: with both eyes seeing walls
# the giant fiber also fires short wall-driven bursts (forest: mean 5.8 Hz, some 20 ms chunks > 50 Hz); those are
# counted separately (gf_bursts_without_bat) instead of being credited to the bat.
BAT_MIN_LOOM_HZ = 5.0
# pellets (sugar maze): nose within PELLET_R = taste contact; GRNs at TASTE_HZ; eaten after EAT_S of feeding state
PELLET_R, TASTE_HZ, EAT_S, TAU_BRAKE = 2.4, 200.0, 0.6, 0.08


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


def record_brain(lab_, duration_s, counts, all_s, all_n, t_wall, inputs, act_info, sil_idx, sil_info, sim_params):
    """Store a closed-loop BrainStepper trial in the same format as run_simulation (raster, 3D brain, dashboard)."""
    S = np.concatenate(all_s) if all_s else np.zeros(0, np.int32)
    Nn = np.concatenate(all_n) if all_n else np.zeros(0, np.int32)
    res = SimResult(duration_s=duration_s, n_trials=1, counts=counts[None].astype(np.int32), spike_step=S,
                    spike_trial=np.zeros(len(S), np.int16), spike_neuron=Nn, wall_s=time.perf_counter() - t_wall,
                    meta={"backend": "numba closed-loop (BrainStepper)"})
    sim_id = cache.run_id_for("sim", f"{lab_}_brain", sim_params)
    act_idx = sorted({i for v_ in inputs.values() for i in v_})
    sim = lab._summarize(sim_id, res, act_idx, act_info, sil_idx, sil_info, sim_params, f"{lab_}_brain")
    cache.save_result(sim_id, sim)
    return sim_id, sim


# ----------------------------------------------------------------------------- closed loop
def _place(W, s, lateral):
    """World (x, y) of a point at arc length s along the centerline, `lateral` m to the left."""
    P, S = W["centerline"], W["s"]
    i = int(np.clip(np.searchsorted(S, s), 1, len(S) - 1))
    u = (s - S[i - 1]) / max(S[i] - S[i - 1], 1e-9)
    x = P[i - 1, 0] + u * (P[i, 0] - P[i - 1, 0])
    y = P[i - 1, 1] + u * (P[i, 1] - P[i - 1, 1])
    th = P[i, 2]
    return float(x - lateral * math.sin(th)), float(y + lateral * math.cos(th))


def _circle_segs(cx, cy, r, n=10):
    a = np.linspace(0, 2 * np.pi, n + 1)
    px, py = cx + r * np.cos(a), cy + r * np.sin(a)
    return np.stack([px[:-1], py[:-1], px[1:], py[1:]], 1)


def run_course(course: str = "canyon", silence: list[str] | None = None, max_s: float = 25.0, seed: int = 0,
               bat_at_s: float | None = None) -> dict:
    """Closed loop on a course. Optional extras (only in the courses that define them, or when asked):
    obstacles (solid pillars/trees), pellets (sugar / poisoned sugar: touching one stimulates the taste GRNs;
    MN9 above threshold = the fly stops and eats), and a bat (bat_at_s: a bat swoops in head-on at that time;
    its looming image drives LPLC2/LC4; giant fiber above threshold = dodge (scripted), otherwise caught)."""
    if course not in COURSES:
        raise ValueError(f"unknown course '{course}'. Available: {list(COURSES)}")
    silence = silence or []
    C = COURSES[course]
    params = {"course": course, "pieces": C["pieces"], "width": C["width"],
              "silence": sorted(silence), "max_s": max_s, "seed": seed, "eye": [RAY_RANGE, RAY_NEAR, EYE_MAX_HZ],
              "v": 1}
    extras = {k: C[k] for k in ("obstacles", "pellets", "speed") if k in C}
    if bat_at_s is not None:
        extras["bat"] = [bat_at_s, BAT_D0, BAT_V, BAT_R, looming.LOOM_SAT_DEG_S, BAT_MIN_LOOM_HZ]
    if extras:
        params["extras"] = extras
    lab_ = f"course_{course}" + ("_bat" if bat_at_s is not None else "") + \
           (("__silence-" + "+".join(silence)) if silence else "")
    run_id = cache.run_id_for("beh", lab_, params)
    cached = cache.load_result(run_id)
    if cached is not None:
        cache.save_result(run_id, cached)
        return cached
    t_wall = time.perf_counter()
    W = build_course(course)
    obstacles = [(*_place(W, s, lat), r) for s, lat, r in C.get("obstacles", [])]
    segs = W["segments"]
    if obstacles:
        segs = np.vstack([segs] + [_circle_segs(ox, oy, r) for ox, oy, r in obstacles])
    pellets = [{"xy": _place(W, s, lat), "kind": kind, "state": "waiting", "eat_s": 0.0, "touched": False}
               for s, lat, kind in C.get("pellets", [])]
    v_max = C.get("speed", V_FWD)
    brain = get_brain()
    names = dict(INPUTS)
    if bat_at_s is not None:
        names |= {"LPLC2": "LPLC2", "LC4": "LC4"}
    if pellets:
        names |= {"sugar": "sugar_GRNs", "bitter": "bitter_GRNs"}
    inputs = {k: lab.resolve(v)[0] for k, v in names.items()}
    sil_idx, sil_info = lab._resolve_many(silence)
    st = BrainStepper(brain, inputs, silence=sil_idx, seed=seed)
    gi = lab._group_indices()
    sil = set(sil_idx)
    gi_eff = {g: [i for i in ii if i not in sil] for g, ii in gi.items()}  # silenced = no output
    from .groups import OUTPUT_GROUPS
    ts = {g: [] for g in OUTPUT_GROUPS}
    eye_log, traj, all_s, all_n, taste_log, events = [], [], [], [], [], []
    counts = np.zeros(brain.N, np.int64)
    x = y = th = v = steer = 0.0
    traj.append([0.0, x, y, th, v, steer])
    outcome, t_end, crash_at = "timeout", max_s, None
    bat = None if bat_at_s is None else {"t0": bat_at_s, "d": BAT_D0, "state": "waiting", "log": []}
    gf_other = 0
    n_chunks = int(round(max_s / CHUNK_S))
    sm = interface.SMOOTH_BINS
    for k in range(n_chunks):
        t_now = k * CHUNK_S
        rl, rr, _, _ = eyes(x, y, th, segs)
        eye_log.append([round(rl, 1), round(rr, 1)])
        rates = {"P9": P9_HZ, "eye_L": rl, "eye_R": rr}
        # ---- bat: head-on swoop, looming -> LPLC2 + LC4 (looming.py)
        if bat is not None:
            loom = 0.0
            if bat["state"] == "waiting" and t_now >= bat["t0"]:
                bat.update(state="flying", hx=math.cos(th), hy=math.sin(th))
            if bat["state"] in ("flying", "dodged"):
                bat["d"] -= BAT_V * CHUNK_S
                if bat["state"] == "flying" and bat["d"] > 0:
                    loom = looming.loom_rate_hz(bat["d"], BAT_V, BAT_R)
                bd = max(bat["d"], -8.0)
                bat["log"].append([round(t_now, 3), round(x + bat["hx"] * bd, 2), round(y + bat["hy"] * bd, 2)])
                if bat["d"] < -8.0:
                    bat["state"] = "gone"
            rates |= {"LPLC2": loom, "LC4": loom}
        # ---- pellets: contact = car nose within PELLET_R
        nx, ny = x + CAR_HALF_LEN * math.cos(th), y + CAR_HALF_LEN * math.sin(th)
        touching = None
        for j, p in enumerate(pellets):
            if p["state"] != "eaten" and math.hypot(nx - p["xy"][0], ny - p["xy"][1]) < PELLET_R:
                touching = j
                if not p["touched"]:
                    p["touched"] = True
                    events.append([round(t_now, 2), "touch", j, p["kind"]])
            elif p["touched"] and p["state"] == "waiting":
                p["state"] = "passed"
                events.append([round(t_now, 2), "passed", j, p["kind"]])
        if pellets:
            sug = TASTE_HZ if touching is not None else 0.0
            bit = TASTE_HZ if touching is not None and pellets[touching]["kind"] == "poison" else 0.0
            rates |= {"sugar": sug, "bitter": bit}
            taste_log.append([sug, bit])
        c, s_, n_ = st.step(rates, STEPS)
        counts += c
        all_s.append(s_); all_n.append(n_)
        for g in OUTPUT_GROUPS:
            ts[g].append(float(c[gi_eff[g]].sum()) / max(len(gi[g]), 1) / CHUNK_S)
        # same smoothing/mapping as interface.motor_commands (causal 3-bin mean)
        mean = lambda g: float(np.mean(([ts[g][0]] * sm + ts[g])[-sm:]))  # noqa: E731
        feeding = mean("MN9") > interface.MN9_THRESHOLD_HZ
        fwd = 0.0 if feeding else min(max(mean("P9") / interface.P9_FULL_HZ, 0), 1)
        Lh = mean("DNa01_left") + mean("DNa02_left")
        Rh = mean("DNa01_right") + mean("DNa02_right")
        turn = 0.0 if feeding else min(max((Lh - Rh) / interface.TURN_FULL_HZ, -1), 1)
        t = round((k + 1) * CHUNK_S, 3)
        if touching is not None:
            p = pellets[touching]
            if feeding:
                if p["eat_s"] == 0.0:
                    events.append([t, "eating", touching, p["kind"]])
                p["eat_s"] += CHUNK_S
                if p["eat_s"] >= EAT_S:
                    p["state"] = "eaten"
                    events.append([t, "eaten", touching, p["kind"]])
        gf = float(c[gi_eff["giant_fiber"]].sum()) / max(len(gi["giant_fiber"]), 1) / CHUNK_S
        bat_looms = bat is not None and bat["state"] == "flying" and loom >= BAT_MIN_LOOM_HZ
        if gf > interface.GF_THRESHOLD_HZ and not bat_looms:
            gf_other += 1
        if bat is not None and bat["state"] == "flying":
            if gf > interface.GF_THRESHOLD_HZ and bat_looms:
                bat.update(state="dodged", t_dodge=t, d_dodge=round(bat["d"], 1), gf_hz=round(gf, 1))
                events.append([t, "dodge", None, None])
            elif bat["d"] <= BAT_CATCH_M:
                bat.update(state="caught", t_caught=t)
                events.append([t, "caught", None, None])
                outcome, t_end = "caught", t
        # car moves during this chunk (sub-steps of 5 ms), walls are solid
        crashed = False
        tau_v = TAU_BRAKE if feeding else TAU_V
        if outcome != "caught":
            for _ in range(4):
                dt = CHUNK_S / 4
                v += (v_max * fwd - v) * min(1.0, dt / tau_v)
                steer += (MAX_STEER * turn - steer) * min(1.0, dt / TAU_S)
                th2 = th + v / WHEELBASE * math.tan(steer) * dt
                x2, y2 = x + v * math.cos(th2) * dt, y + v * math.sin(th2) * dt
                if collides(x2, y2, th2, segs):
                    crashed = True
                    v = 0.0
                    break
                x, y, th = x2, y2, th2
        else:
            v = 0.0
        traj.append([t, round(x, 3), round(y, 3), round(th, 4), round(v, 3), round(steer, 4)])
        if outcome == "caught":
            if t >= t_end + 0.6:
                break
            continue
        if crashed and crash_at is None:
            crash_at = (t, round(x, 2), round(y, 2))
            outcome, t_end = "crash", t
        if crash_at and t >= crash_at[0] + 0.6:  # keep the brain running briefly after impact, then stop
            break
        if not crash_at and progress(W, x, y) >= W["length"] - 1.0:
            outcome, t_end = "finished", t
            break
    D = round(len(ts["P9"]) * CHUNK_S, 3)
    act_info = {"P9 (100 Hz)": len(inputs["P9"]), "LPLC1_L (eye, closed loop)": len(inputs["eye_L"]),
                "LPLC1_R (eye, closed loop)": len(inputs["eye_R"])}
    sim_params = {"activate": ["P9", "LPLC1_L", "LPLC1_R"], "silence": sorted(silence), "closed_loop": True,
                  "course": course, "run": run_id}
    if bat is not None:
        act_info |= {"LPLC2 (bat looming)": len(inputs["LPLC2"]), "LC4 (bat looming)": len(inputs["LC4"])}
        sim_params["activate"] += ["LPLC2", "LC4"]
    if pellets:
        act_info |= {"sugar_GRNs (pellet contact)": len(inputs["sugar"]),
                     "bitter_GRNs (poisoned pellet contact)": len(inputs["bitter"])}
        sim_params["activate"] += ["sugar_GRNs", "bitter_GRNs"]
    sim_id, sim = record_brain(lab_, D, counts, all_s, all_n, t_wall, inputs, act_info, sil_idx, sil_info,
                               sim_params)
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
    car_json = {
        "trajectory": traj, "walls": W["segments"].round(3).tolist(), "finish": W["finish"],
        "centerline": W["centerline"][::4, :2].round(2).tolist(), "eyes_hz": eye_log, "maneuvers": [],
        "mode": "course", "course": course, "outcome": outcome, "crash": crash_at,
        "ray_angles_deg": np.degrees(RAY_ANGLES).round(1).tolist(), "ray_range": RAY_RANGE,
        "theme": C.get("theme", "canyon"), "avatar": C.get("avatar", "car"),
        "obstacles": [[round(a, 2), round(b, 2), r] for a, b, r in obstacles],
        "pellets": [{"xy": [round(p["xy"][0], 2), round(p["xy"][1], 2)], "kind": p["kind"]} for p in pellets],
        "pellet_radius": PELLET_R, "events": events}
    if bat is not None:
        car_json["bat"] = {k: bat.get(k) for k in ("t0", "state", "t_dodge", "d_dodge", "t_caught", "gf_hz")} | \
                          {"track": bat["log"], "radius": BAT_R}
    (d / "car.json").write_text(json.dumps(car_json), encoding="utf8")
    from .behaviors import _save_cmds
    cmd_s = dict(cmd)
    extra_rates = {"eye_L": E[:, 0], "eye_R": E[:, 1]}
    if pellets:
        TL = np.array(taste_log)
        extra_rates |= {"sugar_in": TL[:, 0], "bitter_in": TL[:, 1]}
    cmd_s["rates"] = dict(cmd["rates"]) | extra_rates
    _save_cmds(d, [cmd_s])
    pct = 100 * prog / W["length"]
    if outcome == "finished":
        verdict = f"FINISHED the {course} course ({W['length']:.0f} m) in {t_end:.1f} s without touching a wall"
    elif outcome == "crash":
        verdict = f"CRASHED into a wall at t = {crash_at[0]:.1f} s after {prog:.0f} m ({pct:.0f}% of the course)"
    elif outcome == "caught":
        verdict = f"CAUGHT by the bat at t = {t_end:.1f} s after {prog:.0f} m: the giant fiber never fired"
    else:
        verdict = f"did not finish in {max_s:.0f} s ({prog:.0f} m, {pct:.0f}% of the course)"
    extra = ""
    bat_out = None
    if bat is not None:
        if bat.get("t_dodge"):
            extra += (f" Bat at t = {bat_at_s:.1f} s: giant fiber {bat['gf_hz']:.0f} Hz at "
                      f"t = {bat['t_dodge']:.2f} s with the bat {bat['d_dodge']:.0f} m away -> DODGE (scripted).")
        elif bat["state"] == "waiting":
            extra += " The bat never came (run ended first)."
        bat_out = {"released_s": bat_at_s, "result": bat["state"], "dodge_s": bat.get("t_dodge"),
                   "bat_distance_at_dodge_m": bat.get("d_dodge"), "giant_fiber_hz": bat.get("gf_hz"),
                   "gf_bursts_without_bat": gf_other}
        extra += (f" Giant fiber also crossed 50 Hz in {gf_other} 20-ms chunks without a looming bat "
                  f"(wall-driven, not counted as dodges).")
    pel_out = None
    if pellets:
        eaten = [p["kind"] for p in pellets if p["state"] == "eaten"]
        touched = [p["kind"] for p in pellets if p["touched"]]
        pel_out = {"sugar_eaten": eaten.count("sugar"), "sugar_total": sum(p["kind"] == "sugar" for p in pellets),
                   "poison_eaten": eaten.count("poison"),
                   "poison_total": sum(p["kind"] == "poison" for p in pellets),
                   "touched": len(touched), "events": events}
        extra += (f" Pellets: ate {pel_out['sugar_eaten']}/{pel_out['sugar_total']} sugar, "
                  f"{pel_out['poison_eaten']}/{pel_out['poison_total']} poisoned (touched {len(touched)}); "
                  f"MN9 peak {max(cmd['rates']['MN9']):.0f} Hz.")
    out = {"run_id": run_id, "kind": "behavior", "behavior": f"course_{course}" + ("_bat" if bat else ""),
           "scene": "car", "label": lab_,
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
                     "MN9_peak_hz": round(float(np.max(cmd["rates"]["MN9"])), 1),
                     "active_neurons": sim["sim"]["n_active_neurons"], "spikes": sim["sim"]["total_spikes"]},
           "phases": [], "segments": [], "maneuvers": [], "brain_runs": [sim_id], "raster_png": sim["raster_png"],
           "video": None, "car": f"runs/{run_id}/car.json",
           "summary": (f"LESION: silenced {', '.join(silence)}. " if silence else "") + verdict +
                      f". Eyes (LPLC1 L/R) mean {E[:, 0].mean():.0f}/{E[:, 1].mean():.0f} Hz; brain DNa L/R mean "
                      f"{np.mean(cmd['rates']['DNa_left']):.0f}/{np.mean(cmd['rates']['DNa_right']):.0f} Hz; " +
                      (f"when one eye saw the nearer wall (>10 Hz difference) the brain steered away "
                       f"{100 * agree:.0f}% of the time." if agree == agree else ".") + extra,
           "wall_s": round(time.perf_counter() - t_wall, 1),
           "framing": "Brain is real and runs closed loop (no planner, nobody picks manoeuvres); eye model, P9 drive, "
                      "car link and course are designed. Nothing learns."}
    if bat_out:
        out["bat"] = bat_out
        out["framing"] += " Bat: looming eye model + 50 Hz giant-fiber threshold are designed; the dodge is scripted."
    if pel_out:
        out["pellets"] = pel_out
        out["framing"] += (" Pellets: touching one stimulates sugar GRNs (+ bitter GRNs if poisoned) at 200 Hz; "
                           "MN9 above 30 Hz = stop and eat (interface.py rule).")
    cache.save_result(run_id, out)
    return out


if __name__ == "__main__":
    import sys
    r = run_course(sys.argv[1] if len(sys.argv) > 1 else "canyon", sys.argv[2:] or None)
    print(r["summary"], "| wall", r["wall_s"], "s")
