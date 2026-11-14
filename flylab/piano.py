"""Fly piano band. There is no circuit for music: these are two honest ways to make sound from the model.

1. keyboard (fly_piano): a gang of flies walks across a giant piano keyboard on the floor. Each fly is one
   behaviour (default: walk_forward, turn_left, walk_backward), i.e. a whole-brain simulation driving the
   physics-simulated body (same pipeline as run_behavior). Every time a FRONT foot touches down (tarsal contact
   force crosses a threshold in MuJoCo) the key under that foot sounds. Rhythm = the fly's real stepping
   (~12 Hz CPG of flygym), melody = where its feet land. Each fly starts at a different place on the keyboard
   (different register); the flies are simulated one at a time and shown together.
   Designed: keyboard layout (keys 1.2 mm wide, C-major pentatonic so any chord is consonant), front legs only
   (all six legs at 12 Hz is a blur), the audio is played 8x slowed down (same as the video).

2. orchestra (brain_orchestra): three musician flies, brain only. We (Claude) wrote the score, i.e. WHEN each
   stimulus is applied; the brain decides WHETHER and HOW LOUD each note sounds:
     - bass   : sugar GRN pulses on beats 1 and 3 -> MN9 rate in each eighth note -> bass note (chord root), loud
                when MN9 is high. Bars 5-6 add bitter GRNs to the pulses -> MN9 is suppressed -> the bass drops out.
     - melody : P9 held on, left/right DNa01+DNa02 alternately stimulated -> the brain's turn signal (DNa L - R)
                picks a tone of the bar's chord (strong left = high fifth ... right = low fifth).
     - drums  : LPLC2 looming pulses on beats 2 and 4, LC4 pulses on the off-beats -> giant fiber peak rate
                (strongest 20 ms bin of each eighth note): > 50 Hz = snare, 10-50 Hz = hi-hat.
   Sonification of real model output; the musical mapping and the score are designed. Lesions are audible
   (e.g. silence=["giant_fiber"] -> no drums, silence=["CB0553"] -> weak bass).
"""
from __future__ import annotations

import json
import math
import time
import wave

import numpy as np

from . import cache, interface, lab

SR = 22050
PENTA = [0, 2, 4, 7, 9]
# ----------------------------------------------------------------------------- keyboard
KEY_W = 1.2            # mm per key (along x, the walking direction)
KEY0_X = -16.8         # left edge of key 0 (mm)
N_KEYS = 48
KEY_LEN = 44.0         # mm (keys span y = -22..22)
BASE_MIDI = 36         # key 0 = C2
AUDIO_SLOW = 8.0       # audio (and video) slowed 8x
PIANO_PLAY_SPEED = 1 / AUDIO_SLOW
F_ON, F_OFF, MIN_SWING_S = 2.5, 1.0, 0.015
DEFAULT_PLAYERS = [("walk_forward", -10.0, 0.0), ("turn_left", 4.0, -10.0), ("walk_backward", 26.0, 11.0)]
PLAYER_DUR = 1.5


def key_of(x: float) -> int:
    return int(np.clip(math.floor((x - KEY0_X) / KEY_W), 0, N_KEYS - 1))


def midi_of(k: int) -> int:
    return BASE_MIDI + 12 * (k // 5) + PENTA[k % 5]


def _keyboard_props(x_off, y_off):
    def add(wb):
        kw = dict(contype=0, conaffinity=0)
        for k in range(N_KEYS):
            x0 = KEY0_X + k * KEY_W - x_off
            c = 0.97 if k % 5 else 0.86   # a slightly darker key marks each octave's C
            wb.add("geom", name=f"key{k}", type="box", size=(KEY_W / 2 - 0.05, KEY_LEN / 2, 0.01),
                   pos=(x0 + KEY_W / 2, -y_off, 0.012), rgba=(c, c, c * 0.97, 1), **kw)
            if k % 5 in (1, 3):   # decorative black keys at the far end
                wb.add("geom", name=f"bkey{k}", type="box", size=(KEY_W * 0.3, KEY_LEN * 0.18, 0.15),
                       pos=(x0, -y_off + KEY_LEN * 0.32, 0.15), rgba=(0.08, 0.08, 0.09, 1), **kw)
    return add


def _footfall_hook(body, notes, x_off, y_off, player, legs=("LF", "RF")):
    ph = body.physics
    from .body import _CONTACT_SENSORS
    n = body.fly.name
    idx = {l: [i for i, s in enumerate(_CONTACT_SENSORS) if s.startswith(l + "Tarsus")] for l in legs}
    bid = {l: ph.model.name2id(f"{n}/{l}Tarsus5", "body") for l in legs}
    st = {l: {"on": True, "off_since": None} for l in legs}

    def hook(b):
        f = np.asarray(b.obs["contact_forces"])
        for l in legs:
            F = float(np.linalg.norm(f[idx[l]], axis=-1).sum())
            s = st[l]
            if s["on"] and F < F_OFF:
                s["on"], s["off_since"] = False, b.time
            elif not s["on"] and F > F_ON:
                s["on"] = True
                if s["off_since"] is not None and b.time - s["off_since"] >= MIN_SWING_S:
                    x, y = ph.data.xpos[bid[l], :2]
                    xw, yw = float(x) + x_off, float(y) + y_off
                    k = key_of(xw)
                    on_keys = KEY0_X <= xw <= KEY0_X + N_KEYS * KEY_W and abs(yw) <= KEY_LEN / 2
                    if on_keys:
                        notes.append({"t": round(b.time, 4), "player": player, "leg": l, "key": k,
                                      "midi": midi_of(k), "x": round(xw, 2), "y": round(yw, 2),
                                      "force": round(F, 2)})
    return hook


# ----------------------------------------------------------------------------- synthesis
def _tone(midi, dur, vel, kind="piano"):
    f = 440.0 * 2 ** ((midi - 69) / 12)
    n = int(dur * SR)
    t = np.arange(n) / SR
    if kind == "piano":
        y = sum(np.sin(2 * np.pi * f * h * t) * (1 / h ** 1.4) * np.exp(-t * 2.5 * h ** 0.6) for h in range(1, 7))
    elif kind == "bass":
        y = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)) * np.exp(-t * 2.2)
    elif kind == "lead":
        y = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(6 * np.pi * f * t)) * np.exp(-t * 4.0)
    elif kind == "epiano":   # soft electric piano: warm fundamental + bell-like partial, medium decay
        y = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t * 6)
             + 0.12 * np.sin(2 * np.pi * f * 3.98 * t) * np.exp(-t * 14)) * np.exp(-t * 1.6)
    elif kind == "organ":   # sustained pipe-organ-like tone: octave + fifth partials, slow swell, soft release
        y = sum(a_ * np.sin(2 * np.pi * f * h * t) for h, a_ in ((0.5, 0.35), (1, 1.0), (2, 0.55), (3, 0.3), (4, 0.2)))
        y = y * np.minimum(1.0, t / 0.09) * np.where(t < 0.75 * dur, 1.0, np.exp(-(t - 0.75 * dur) * 9))
    elif kind == "snare":
        rng = np.random.default_rng(int(midi))
        y = rng.standard_normal(n) * np.exp(-t * 18) * 0.6 + np.sin(2 * np.pi * 185 * t) * np.exp(-t * 25)
    else:  # hihat
        rng = np.random.default_rng(int(midi) + 7)
        w = rng.standard_normal(n)
        y = np.diff(w, prepend=0) * np.exp(-t * 45) * 0.35
    att = np.minimum(1.0, t / 0.004)
    return vel * y * att


def _render_wav(notes, path, time_scale=1.0, tail=1.5):
    if not notes:
        total = 1.0
    else:
        total = max(n["t"] for n in notes) * time_scale + tail
    buf = np.zeros(int(total * SR) + 1)
    for n in notes:
        y = _tone(n["midi"], n.get("dur", 1.2), n.get("vel", 0.7), n.get("inst", "piano"))
        i0 = int(n["t"] * time_scale * SR)
        m = min(len(y), len(buf) - i0)
        if m > 0:
            buf[i0:i0 + m] += y[:m]
    peak = np.max(np.abs(buf)) or 1.0
    pcm = (buf / peak * 0.85 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return round(total, 2)


def _mux(video, audio, out):
    import subprocess
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio), "-c:v", "copy",
                        "-c:a", "aac", "-shortest", str(out)], capture_output=True, text=True)
    return r.returncode == 0


# ----------------------------------------------------------------------------- 1. keyboard
def fly_piano(players: list[str] | None = None, silence: list[str] | None = None) -> dict:
    from .behaviors import BEHAVIORS, _clean, _sim
    from .body import FlyBody
    silence = silence or []
    spec = DEFAULT_PLAYERS if not players else \
        [(p, DEFAULT_PLAYERS[i % 3][1], DEFAULT_PLAYERS[i % 3][2]) for i, p in enumerate(players[:3])]
    for b, _, _ in spec:
        if b not in BEHAVIORS or BEHAVIORS[b].get("brain_only") or b in ("feed", "feed_with_bitter", "takeoff",
                                                                          "find_sugar"):
            raise ValueError(f"'{b}' is not a walking behaviour. Use walk_forward, turn_left, turn_right, walk_backward")
    params = {"players": spec, "silence": sorted(silence), "dur": PLAYER_DUR, "keys": [KEY_W, KEY0_X, N_KEYS],
              "f": [F_ON, F_OFF, MIN_SWING_S], "v": 1}
    lab_ = "piano_keyboard" + (("__silence-" + "+".join(silence)) if silence else "")
    run_id = cache.run_id_for("beh", lab_, params)
    cached = cache.load_result(run_id)
    if cached is not None and cached.get("video") and (cache.REPO / cached["video"]).exists():
        cache.save_result(run_id, cached)
        return cached
    t0 = time.perf_counter()
    d = cache.run_dir(run_id)
    notes, flies, brain_runs, cmds = [], [], [], []
    for pi, (beh, x_off, y_off) in enumerate(spec):
        B = BEHAVIORS[beh]
        sim = _sim(B["stimulus"], [], silence, PLAYER_DUR, f"{beh}" + (("__silence-" + "+".join(silence))
                                                                       if silence else "") + "_brain")
        brain_runs.append(sim["run_id"])
        cmd = interface.motor_commands(sim["timeseries"], sim["timeseries_bin_s"])
        cmds.append(cmd)
        render = pi == 0
        body = FlyBody(render=render, props=_keyboard_props(x_off, y_off) if render else None,
                       play_speed=PIANO_PLAY_SPEED, window_size=(960, 600), cam_distance=13.0,
                       cam_elevation_deg=42.0, cam_azimuth_deg=-80.0)
        mine = []
        body.step_hooks.append(_footfall_hook(body, mine, x_off, y_off, pi))
        r = body.run(interface.to_program(cmd))
        if render:
            body.save_video(d / "video_silent.mp4")
        notes += mine
        tr = [[round(row[0], 3), round(row[1] + x_off, 2), round(row[2] + y_off, 2), round(row[3], 1)]
              for row in body.trajectory]
        flies.append({"behavior": beh, "start": [x_off, y_off], "trajectory": tr, "n_notes": len(mine),
                      "keys_played": sorted({n["key"] for n in mine}), "body": _clean(r),
                      "brain": interface.describe(cmd)})
        body.close()
    notes.sort(key=lambda n: n["t"])
    timbre = ["piano", "lead", "bass"]
    for n in notes:
        n.update(inst=timbre[n["player"] % 3], vel=0.75 if n["player"] == 0 else 0.6, dur=1.4)
    dur_audio = _render_wav(notes, d / "piano.wav", time_scale=AUDIO_SLOW)
    ok = _mux(d / "video_silent.mp4", d / "piano.wav", d / "video.mp4")
    if not ok:
        (d / "video_silent.mp4").replace(d / "video.mp4")
    (d / "piano.json").write_text(json.dumps({
        "mode": "keyboard", "notes": notes, "flies": flies, "key_w": KEY_W, "key0_x": KEY0_X, "n_keys": N_KEYS,
        "key_len": KEY_LEN, "midi": [midi_of(k) for k in range(N_KEYS)], "audio": f"runs/{run_id}/piano.wav",
        "time_scale": AUDIO_SLOW, "video": f"runs/{run_id}/video.mp4", "video_play_speed": PIANO_PLAY_SPEED}),
        encoding="utf8")
    from .behaviors import _save_cmds
    _save_cmds(d, [cmds[0]])
    per = "; ".join(f"fly {i + 1} ({f['behavior']}): {f['n_notes']} notes on keys {min(f['keys_played'], default='-')}"
                    f"-{max(f['keys_played'], default='-')}, walked {f['body'].get('distance_walked_mm', 0):.1f} mm"
                    for i, f in enumerate(flies))
    out = {"run_id": run_id, "kind": "behavior", "behavior": "piano_keyboard", "scene": "piano", "label": lab_,
           "description": "A gang of flies walks on a piano keyboard; each front-foot touchdown (MuJoCo contact) "
                          "plays the key under it.",
           "silenced": silence, "players": [s[0] for s in spec], "n_notes": len(notes),
           "notes_per_fly": [f["n_notes"] for f in flies], "brain_runs": brain_runs,
           "raster_png": cache.load_result(brain_runs[0])["raster_png"], "video": f"runs/{run_id}/video.mp4",
           "video_play_speed": PIANO_PLAY_SPEED, "audio": f"runs/{run_id}/piano.wav", "audio_s": dur_audio,
           "piano": f"runs/{run_id}/piano.json", "phases": [],
           "summary": (f"LESION: silenced {', '.join(silence)}. " if silence else "") + per +
                      f". {len(notes)} notes in {PLAYER_DUR} s of walking ({dur_audio:.0f} s of music, 8x slowed).",
           "wall_s": round(time.perf_counter() - t0, 1),
           "framing": "Brains and legs are real simulations; footfalls come from MuJoCo contact forces. The keyboard, "
                      "the pentatonic tuning, front-legs-only and the 8x slow-down are designed. The flies are not "
                      "making music; we listen to their walking."}
    cache.save_result(run_id, out)
    return out


# ----------------------------------------------------------------------------- 2. orchestra
BPM = 100.0
EIGHTH = 60.0 / BPM / 2
BARS = 8
CHORDS = [48, 45, 41, 43, 48, 45, 41, 43]       # C  Am F  G  C  Am F  G   (bass roots, MIDI)
BITTER_BARS = {4, 5}                            # 0-based: bars 5-6 get bitter with the sugar


def _scores():
    beat = 2 * EIGHTH
    bar = 4 * beat
    bass, melody, drums = [], [], []
    for b in range(BARS):
        t = b * bar
        for k in (0, 2):        # sugar on beats 1 and 3
            bass.append({"neurons": ["sugar_GRNs"], "rate_hz": 200.0, "t0": round(t + k * beat, 3),
                         "t1": round(t + k * beat + 0.25, 3)})
            if b in BITTER_BARS:
                bass.append({"neurons": ["bitter_GRNs"], "rate_hz": 200.0, "t0": round(t + k * beat, 3),
                             "t1": round(t + k * beat + 0.25, 3)})
        side = ["left", "left", "right", "left", "right", "right", "left", "right"][b]
        for k in range(4):      # steer pattern: 2 beats one way, then the other (shifted every bar)
            sd = side if k < 2 else ("right" if side == "left" else "left")
            for g in (f"DNa01_{sd}", f"DNa02_{sd}"):
                melody.append({"neurons": [g], "rate_hz": 60.0 + 20.0 * (k % 2), "t0": round(t + k * beat, 3),
                               "t1": round(t + k * beat + beat * 0.8, 3)})
        for k in (1, 3):        # looming on beats 2 and 4
            drums.append({"neurons": ["LPLC2"], "rate_hz": 40.0, "t0": round(t + k * beat, 3),
                          "t1": round(t + k * beat + 0.08, 3)})
        for k in range(4):      # LC4 on the off-beats
            drums.append({"neurons": ["LC4"], "rate_hz": 35.0, "t0": round(t + k * beat + EIGHTH, 3),
                          "t1": round(t + k * beat + EIGHTH + 0.08, 3)})
    melody.append({"neurons": ["P9"], "rate_hz": 100.0, "t0": 0.0, "t1": None})
    return bass, melody, drums


def _stepper_sim(sched, D, silence, label, seed=0, chunk_s=0.02):
    """One trial of a scheduled stimulus run with BrainStepper (same equations as run_simulation; dense parallel
    kernel). Returns the fields brain_orchestra needs (timeseries per output group, raster, run_id)."""
    from .arena import record_brain
    from .brain import BrainStepper, get_brain
    from .groups import OUTPUT_GROUPS
    names = sorted({n for s in sched for n in s["neurons"]})
    sim_params = {"activate": names, "silence": sorted(silence), "schedule": sched, "stepper": True, "seed": seed}
    hit = cache.load_result(cache.run_id_for("sim", f"{label}_brain", sim_params))
    if hit is not None:
        return {"run_id": hit["run_id"], "timeseries": hit["timeseries"], "timeseries_bin_s": hit["timeseries_bin_s"],
                "raster_png": hit["raster_png"]}
    inputs = {n: lab.resolve(n)[0] for n in names}
    sil_idx, sil_info = lab._resolve_many(silence)
    brain = get_brain()
    st = BrainStepper(brain, inputs, silence=sil_idx, seed=seed)
    gi = lab._group_indices()
    sil = set(sil_idx)
    keep = {g: [i for i in gi[g] if i not in sil] for g in OUTPUT_GROUPS}
    steps = int(round(chunk_s * 1000 / 0.1))
    ts = {g: [] for g in OUTPUT_GROUPS}
    counts = np.zeros(brain.N, np.int64)
    all_s, all_n = [], []
    t_wall = time.perf_counter()
    n = int(round(D / chunk_s))
    for k in range(n):
        tm = (k + 0.5) * chunk_s
        rates = {nm: 0.0 for nm in names}
        for s_ in sched:
            if s_["t0"] <= tm < (s_["t1"] if s_["t1"] is not None else 1e9):
                for nm in s_["neurons"]:
                    rates[nm] = max(rates[nm], s_["rate_hz"])
        c, sp, ne = st.step(rates, steps)
        counts += c
        all_s.append(sp); all_n.append(ne)
        for g in OUTPUT_GROUPS:
            ts[g].append(float(c[keep[g]].sum()) / max(len(gi[g]), 1) / chunk_s)
    act_info = {f"{nm} (score pulses)": len(inputs[nm]) for nm in names}
    sim_id, sim = record_brain(label, round(n * chunk_s, 3), counts, all_s, all_n, t_wall, inputs, act_info,
                               sil_idx, sil_info, sim_params)
    return {"run_id": sim_id, "timeseries": ts, "timeseries_bin_s": chunk_s, "raster_png": sim["raster_png"]}


def brain_orchestra(silence: list[str] | None = None) -> dict:
    silence = silence or []
    params = {"bpm": BPM, "bars": BARS, "chords": CHORDS, "bitter": sorted(BITTER_BARS),
              "silence": sorted(silence), "v": 2}
    lab_ = "piano_orchestra" + (("__silence-" + "+".join(silence)) if silence else "")
    run_id = cache.run_id_for("beh", lab_, params)
    cached = cache.load_result(run_id)
    if cached is not None:
        cache.save_result(run_id, cached)
        return cached
    t0 = time.perf_counter()
    D = round(BARS * 8 * EIGHTH, 3)
    sb, sm, sd = _scores()
    sims = {}
    sfx = ("__silence-" + "+".join(silence)) if silence else ""
    for name, sched in (("bass", sb), ("melody", sm)):
        sims[name] = lab.run_simulation([], silence, 200.0, D, 1, schedule=sched, label=f"orchestra_{name}{sfx}")
    # drums drive ~300 visual neurons: the multithreaded stepper kernel is ~10x faster than the active-set kernel
    sims["drums"] = _stepper_sim(sd, D, silence, f"orchestra_drums{sfx}")
    bin_s = sims["bass"]["timeseries_bin_s"]
    per = int(round(EIGHTH / bin_s))
    n_slots = int(round(D / EIGHTH))

    def slot(ts, i):
        return float(np.mean(ts[i * per:(i + 1) * per])) if len(ts) >= (i + 1) * per else 0.0

    def peak(ts, i):   # strongest 20 ms bin in the slot (a giant-fiber burst is shorter than an eighth note)
        seg = ts[i * per:(i + 1) * per]
        return float(np.max(seg)) if len(seg) else 0.0

    T = {k: {g: np.asarray(v) for g, v in s["timeseries"].items()} for k, s in sims.items()}
    notes = []
    rows = {"bass_MN9": [], "melody_turn": [], "drums_GF": []}
    for i in range(n_slots):
        t = round(i * EIGHTH, 3)
        bar = i // 8
        mn9 = slot(T["bass"]["MN9"], i)
        rows["bass_MN9"].append(round(mn9, 1))
        if mn9 > 20.0:
            notes.append({"t": t, "midi": CHORDS[bar] - 12 + (12 if i % 4 == 2 else 0), "vel": min(1.0, mn9 / 80),
                          "inst": "bass", "dur": 0.9, "musician": "bass", "hz": round(mn9, 1)})
        L = slot(T["melody"]["DNa01_left"], i) + slot(T["melody"]["DNa02_left"], i)
        R = slot(T["melody"]["DNa01_right"], i) + slot(T["melody"]["DNa02_right"], i)
        p9 = slot(T["melody"]["P9"], i)
        turn = float(np.clip((L - R) / interface.TURN_FULL_HZ, -1, 1))
        rows["melody_turn"].append(round(turn, 2))
        if p9 > 50.0:
            # the brain's turn signal picks a tone of the bar's chord: strong left = high fifth, left = third,
            # straight = root, right = low fifth (chords and this mapping are ours; the turn signal is the brain's)
            root = CHORDS[bar] + 12
            minor = CHORDS[bar] % 12 in (9, 4, 2)      # Am (and Em/Dm if used) take a minor third
            third = 3 if minor else 4
            idx = 3 if turn > 0.6 else 2 if turn > 0.15 else 1 if turn > -0.15 else 0
            pitch = [root - 5, root, root + third, root + 7][idx]
            notes.append({"t": t, "midi": pitch, "vel": 0.35 + 0.45 * abs(turn),
                          "inst": "lead", "dur": 0.5, "musician": "melody", "turn": round(turn, 2)})
        gf = peak(T["drums"]["giant_fiber"], i)
        rows["drums_GF"].append(round(gf, 1))
        if gf > interface.GF_THRESHOLD_HZ:
            notes.append({"t": t, "midi": 38, "vel": min(1.0, gf / 120), "inst": "snare", "dur": 0.3,
                          "musician": "drums", "hz": round(gf, 1)})
        elif gf > 10.0:
            notes.append({"t": t, "midi": 42, "vel": 0.3 + gf / 100, "inst": "hihat", "dur": 0.2,
                          "musician": "drums", "hz": round(gf, 1)})
    d = cache.run_dir(run_id)
    dur_audio = _render_wav(notes, d / "orchestra.wav")
    cnt = {m: sum(n["musician"] == m for n in notes) for m in ("bass", "melody", "drums")}
    bars_mn9 = [round(float(np.mean(rows["bass_MN9"][b * 8:(b + 1) * 8])), 1) for b in range(BARS)]
    (d / "piano.json").write_text(json.dumps({
        "mode": "orchestra", "notes": notes, "eighth_s": EIGHTH, "bars": BARS, "slots": rows,
        "bitter_bars": sorted(BITTER_BARS), "audio": f"runs/{run_id}/orchestra.wav", "time_scale": 1.0,
        "duration_s": D}), encoding="utf8")
    from .behaviors import _save_cmds
    _save_cmds(d, [interface.motor_commands(sims["melody"]["timeseries"], bin_s)])
    sugar_bars = [m for b, m in enumerate(bars_mn9) if b not in BITTER_BARS]
    bitter_bars = [m for b, m in enumerate(bars_mn9) if b in BITTER_BARS]
    out = {"run_id": run_id, "kind": "behavior", "behavior": "piano_orchestra", "scene": "piano", "label": lab_,
           "description": "Three musician flies (brain only): sugar/bitter -> MN9 = bass, P9 + DNa steering = "
                          "melody, looming -> giant fiber = drums. Score designed; notes decided by the brain.",
           "silenced": silence, "notes": cnt, "mn9_per_bar_hz": bars_mn9,
           "brain_runs": [s["run_id"] for s in sims.values()], "raster_png": sims["drums"]["raster_png"],
           "video": None, "audio": f"runs/{run_id}/orchestra.wav", "audio_s": dur_audio,
           "piano": f"runs/{run_id}/piano.json", "phases": [],
           "summary": (f"LESION: silenced {', '.join(silence)}. " if silence else "") +
                      f"{BARS} bars at {BPM:.0f} BPM: bass {cnt['bass']} notes, melody {cnt['melody']}, drums "
                      f"{cnt['drums']}. Bass (MN9) mean {np.mean(sugar_bars):.0f} Hz in sugar bars vs "
                      f"{np.mean(bitter_bars):.0f} Hz in the bitter bars 5-6 (bitter silences the bass).",
           "wall_s": round(time.perf_counter() - t0, 1),
           "framing": "Sonification: the brain's real output rates decide which notes sound and how loud; the score "
                      "(stimulus timing), chords and note mapping are designed. The flies are not composing."}
    cache.save_result(run_id, out)
    return out


def play_piano(mode: str = "keyboard", players: list[str] | None = None, silence: list[str] | None = None,
               song=None, p9_hz: float = 100.0, bpm: float = 100.0, bitter_notes: list[int] | None = None) -> dict:
    if mode == "keyboard":
        return fly_piano(players, silence)
    if mode == "orchestra":
        return brain_orchestra(silence)
    from . import song as _song
    if mode in ("song", "song_road"):
        return _song.song_road(song or "ode_to_joy", p9_hz, silence)
    if mode == "song_orchestra":
        return _song.song_orchestra(song or "ode_to_joy", bpm, bitter_notes, silence)
    raise ValueError("mode must be 'keyboard', 'orchestra', 'song_road' or 'song_orchestra'")


if __name__ == "__main__":
    import sys
    r = play_piano(sys.argv[1] if len(sys.argv) > 1 else "keyboard", silence=sys.argv[2:] or None)
    print(r["summary"], "| wall", r["wall_s"], "s")
