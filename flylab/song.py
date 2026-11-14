"""Make the flies play a SPECIFIC song. No fly circuit knows a melody, so the song always comes from a score that
we (Claude / the user) provide. Two honest versions:

1. song_road ("musical road"): the song's notes are laid on the floor as a road of keys, in order, each key as long
   as its note (MM_PER_BEAT per beat). A fly driven by its real forward-walking command (P9 -> interface -> flygym
   legs) walks along it; a key sounds the FIRST time any foot touches down on it (MuJoCo tarsal contact forces).
   Melody = the score. Rhythm = the fly's own walking: faster stepping = faster song, a stumble = a hiccup, a key a
   stride steps over = a missed note (reported). A walking fly drifts (~-30 deg/s with P9 alone), so each key is
   laid just ahead of the fly along its CURRENT heading when the fly comes within LOOKAHEAD_MM of it (keys are mocap
   bodies that never collide with or steer the fly): the road unrolls in front of it.
   Controls: p9_hz lower -> slower walk -> slower tempo; silence=["P9"] -> no walking -> no song.
2. song_orchestra: a "singer" fly gets one sugar-GRN pulse per note; the note sounds only if MN9 (feeding motor
   neuron) fires above 20 Hz in that note, louder when it fires harder. bitter_notes = indices of notes that also get
   bitter GRNs -> MN9 is suppressed -> those notes go silent. silence=["CB0553"] weakens the singer.

Song format: a built-in name (public-domain tunes in SONGS) or note text "E4 E4 F4 G4:2 R:1 C#5:0.5" (note+octave,
optional :beats, R = rest). For copyrighted songs the user supplies the notes.
"""
from __future__ import annotations

import json
import math
import re
import time

import numpy as np

from . import cache, interface, lab
from .piano import _mux, _render_wav

SONGS = {
    "ode_to_joy": {"title": "Ode to Joy (Beethoven, public domain)",
                   "notes": "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 E4:1.5 D4:0.5 D4:2 "
                            "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 D4:1.5 C4:0.5 C4:2"},
    "twinkle": {"title": "Twinkle Twinkle Little Star (traditional, public domain)",
                "notes": "C4 C4 G4 G4 A4 A4 G4:2 F4 F4 E4 E4 D4 D4 C4:2 "
                         "G4 G4 F4 F4 E4 E4 D4:2 G4 G4 F4 F4 E4 E4 D4:2"},
    "frere_jacques": {"title": "Frere Jacques (traditional, public domain)",
                      "notes": "C4 D4 E4 C4 C4 D4 E4 C4 E4 F4 G4:2 E4 F4 G4:2 "
                               "G4:0.5 A4:0.5 G4:0.5 F4:0.5 E4 C4 G4:0.5 A4:0.5 G4:0.5 F4:0.5 E4 C4 "
                               "C4 G3 C4:2 C4 G3 C4:2"},
}
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

MM_PER_BEAT = 1.2          # key length on the road per beat
ROAD_START_MM = 2.5        # road starts this far along the fly's path (front feet are ~1.5 mm ahead of the thorax)
ROAD_HALF_W = 3.5          # touchdowns farther than this from the road centre line do not count
ROAD_VIS_HALF_W = 2.6      # drawn half-width of the keys
WALK_SPEED_GUESS = 11.0    # mm/s, to size the walk (the real speed comes out of the physics)
AUDIO_SLOW = 8.0           # song is heard 8x slowed down (same as the video)
LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
F_ON, F_OFF, MIN_SWING_S = 2.5, 1.0, 0.015


def parse_song(song) -> tuple[str, list[tuple[int | None, float]]]:
    if isinstance(song, (list, tuple)):
        return "custom", [(None if m is None else int(m), float(b)) for m, b in song]
    key = str(song).strip()
    if key.lower() in SONGS:
        title, text = SONGS[key.lower()]["title"], SONGS[key.lower()]["notes"]
    else:
        title, text = "custom", key
    out = []
    for tok in text.replace("|", " ").split():
        name, _, beats = tok.partition(":")
        b = float(beats) if beats else 1.0
        if name.upper() in ("R", "REST", "-"):
            out.append((None, b))
            continue
        m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", name)
        if not m:
            raise ValueError(f"bad note '{tok}': use e.g. C4, F#4:0.5, Bb3:2, R:1 (rest), or a built-in song "
                             f"{list(SONGS)}")
        acc = 1 if m[2] == "#" else -1 if m[2] == "b" else 0
        out.append((12 * (int(m[3]) + 1) + _PC[m[1].upper()] + acc, b))
    if not out:
        raise ValueError("empty song")
    if sum(b for _, b in out) > 64:
        raise ValueError("song longer than 64 beats; send a shorter excerpt")
    return title, out


def _name(m):
    return "rest" if m is None else ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"][m % 12] + str(m // 12 - 1)


# ----------------------------------------------------------------------------- road
LOOKAHEAD_MM = 7.0         # a key is laid when the fly is this far (along its walk) from the key's start


def _key_rgba(m):
    import colorsys
    if m is None:
        return (0.25, 0.25, 0.27, 1)
    return (*colorsys.hsv_to_rgb(((m - 60) % 12) / 12.0, 0.75, 0.95), 1.0)   # one colour per pitch class


def _road_props(notes):
    """One movable (mocap) flat key per note, parked under the floor until laid. Keys never collide."""
    def add(wb):
        kw = dict(contype=0, conaffinity=0)
        for i, (m, b) in enumerate(notes):
            k = wb.add("body", name=f"songkey{i}", mocap=True, pos=(0, 0, -60.0 - i))
            k.add("geom", name=f"songkey{i}_g", type="box", rgba=_key_rgba(m),
                  size=(max(b * MM_PER_BEAT / 2 - 0.09, 0.05), ROAD_VIS_HALF_W, 0.01), **kw)
    return add


def _road_layer(body, notes, edges, laid):
    """Step hook: track the distance walked (thorax, 50 ms low-pass) and lay each key ahead of the fly along its
    CURRENT heading when it comes within LOOKAHEAD_MM. The keys do not influence the fly (no collision, no input)."""
    ph = body.physics
    mids = [int(ph.model.body_mocapid[ph.model.name2id(f"songkey{i}", "body")]) for i in range(len(notes))]
    st = {"p": body.position.copy(), "s": 0.0}

    def hook(b):
        p = b.position
        lp = st["p"] + 0.02 * (p - st["p"])       # 1 ms step, tau 50 ms
        st["s"] += float(np.hypot(*(lp - st["p"])))
        st["p"] = lp
        h = math.radians(b.heading)
        for i in range(len(notes)):
            if i in laid or edges[i] > st["s"] + LOOKAHEAD_MM:
                continue
            mid = (edges[i] + edges[i + 1]) / 2 - st["s"]
            cx, cy = float(p[0] + mid * math.cos(h)), float(p[1] + mid * math.sin(h))
            ph.data.mocap_pos[mids[i]] = (cx, cy, 0.012)
            ph.data.mocap_quat[mids[i]] = (math.cos(h / 2), 0, 0, math.sin(h / 2))
            laid[i] = (cx, cy, h, round(b.time, 4))
    return hook


def _on_key(x, y, key, half_len):
    cx, cy, h, _ = key
    dx, dy = x - cx, y - cy
    along = dx * math.cos(h) + dy * math.sin(h)
    across = -dx * math.sin(h) + dy * math.cos(h)
    return abs(along) <= half_len and abs(across) <= ROAD_HALF_W


def _touchdown_hook(body, out):
    from .body import _CONTACT_SENSORS
    ph = body.physics
    n = body.fly.name
    idx = {l: [i for i, s in enumerate(_CONTACT_SENSORS) if s.startswith(l + "Tarsus")] for l in LEGS}
    bid = {l: ph.model.name2id(f"{n}/{l}Tarsus5", "body") for l in LEGS}
    st = {l: {"on": True, "off": None} for l in LEGS}

    def hook(b):
        f = np.asarray(b.obs["contact_forces"])
        for l in LEGS:
            F = float(np.linalg.norm(f[idx[l]], axis=-1).sum())
            s_ = st[l]
            if s_["on"] and F < F_OFF:
                s_["on"], s_["off"] = False, b.time
            elif not s_["on"] and F > F_ON:
                s_["on"] = True
                if s_["off"] is not None and b.time - s_["off"] >= MIN_SWING_S:
                    x, y = ph.data.xpos[bid[l], :2]
                    out.append((round(b.time, 4), l, float(x), float(y)))
    return hook


# ----------------------------------------------------------------------------- 1. musical road
def song_road(song="ode_to_joy", p9_hz: float = 100.0, silence: list[str] | None = None) -> dict:
    from .behaviors import _clean, _save_cmds
    from .body import FlyBody
    silence = silence or []
    title, notes = parse_song(song)
    total_mm = ROAD_START_MM + sum(b for _, b in notes) * MM_PER_BEAT
    D = round(min(6.0, total_mm / WALK_SPEED_GUESS + 0.6), 2)
    params = {"notes": notes, "p9": p9_hz, "silence": sorted(silence), "mm_beat": MM_PER_BEAT, "D": D,
              "look": LOOKAHEAD_MM, "v": 3}
    lab_ = "song_road" + (("__silence-" + "+".join(silence)) if silence else "")
    run_id = cache.run_id_for("beh", lab_, params)
    cached = cache.load_result(run_id)
    if cached is not None and cached.get("video") and (cache.REPO / cached["video"]).exists():
        cache.save_result(run_id, cached)
        return cached
    t0 = time.perf_counter()
    sched = [{"neurons": ["P9"], "rate_hz": p9_hz, "t0": 0.0, "t1": None}]
    sim = lab.run_simulation([], silence, 200.0, D, 4, schedule=sched, label=f"{lab_}_brain")
    cmd = interface.motor_commands(sim["timeseries"], sim["timeseries_bin_s"])
    prog = interface.to_program(cmd)
    d = cache.run_dir(run_id)
    edges = np.concatenate([[0.0], np.cumsum([b * MM_PER_BEAT for _, b in notes])]) + ROAD_START_MM
    attempts = []
    for seed in (0, 1, 2, 3, 4):
        body = FlyBody(render=True, props=_road_props(notes), play_speed=1 / AUDIO_SLOW, window_size=(960, 600),
                       cam_distance=12.0, cam_elevation_deg=55.0, cam_azimuth_deg=-90.0, seed=seed)
        td, laid = [], {}
        body.step_hooks.append(_road_layer(body, notes, edges, laid))
        body.step_hooks.append(_touchdown_hook(body, td))
        r = body.run(prog)
        if r["physics_error"] and len(laid) < len(notes) and seed < 4:
            attempts.append({"seed": seed, "t_end": r["t_end_s"], "error": r["physics_error"]})
            body.close()
            continue
        body.save_video(d / "video_silent.mp4")
        body.close()
        break
    # touchdowns -> keys (first touch of each laid key plays its note)
    played = {}
    for t, leg, x, y in td:
        for k, key in laid.items():
            if k not in played and t >= key[3] and _on_key(x, y, key, notes[k][1] * MM_PER_BEAT / 2):
                played[k] = (t, leg)
                break
    seq = []
    for k, (m, b) in enumerate(notes):
        e = {"i": k, "midi": m, "name": _name(m), "beats": b, "played": k in played and m is not None,
             "t": played[k][0] if k in played else None, "leg": played[k][1] if k in played else None,
             "laid": k in laid}
        seq.append(e)
    # notes sound in ROAD order: a foot that lands on a later key first only plays it in its turn? No - every note
    # sounds when it is first touched (honest), so a stumble can swap two notes; report how many are out of order.
    hits = sorted([e for e in seq if e["played"]], key=lambda e: e["t"])
    out_of_order = sum(1 for a, b_ in zip(hits, hits[1:]) if b_["i"] < a["i"])
    audio_notes = []
    for a, b_ in zip(hits, hits[1:] + [None]):
        dur = ((b_["t"] - a["t"]) * AUDIO_SLOW + 0.15) if b_ else 1.5
        audio_notes.append({"t": a["t"], "midi": a["midi"], "vel": 0.8, "inst": "piano",
                            "dur": max(0.25, min(dur, 2.5))})
    dur_audio = _render_wav(audio_notes, d / "song.wav", time_scale=AUDIO_SLOW)
    if not _mux(d / "video_silent.mp4", d / "song.wav", d / "video.mp4"):
        (d / "video_silent.mp4").replace(d / "video.mp4")
    sounding = [e for e in seq if e["midi"] is not None]
    n_hit = len(hits)
    tempo = None
    if n_hit >= 2:
        first, last = min(hits, key=lambda e: e["i"]), max(hits, key=lambda e: e["i"])
        beats = sum(e["beats"] for e in seq[first["i"]:last["i"]])
        span = (last["t"] - first["t"]) * AUDIO_SLOW
        tempo = round(60.0 * beats / span, 0) if beats and span > 0 else None
    walked = r.get("distance_walked_mm", 0.0)
    (d / "piano.json").write_text(json.dumps({
        "mode": "song", "title": title, "seq": seq, "time_scale": AUDIO_SLOW, "video": f"runs/{run_id}/video.mp4",
        "audio": f"runs/{run_id}/song.wav", "touchdowns": [[t, l] for t, l, _, _ in td]}), encoding="utf8")
    _save_cmds(d, [cmd])
    out = {"run_id": run_id, "kind": "behavior", "behavior": "song_road", "scene": "piano", "label": lab_,
           "description": "Musical road: the song's keys are laid just ahead of a P9-driven fly along its heading as "
                          "it walks; each key sounds when a foot first lands on it.",
           "song": title, "silenced": silence, "p9_hz": p9_hz, "notes_total": len(sounding), "notes_played": n_hit,
           "notes_missed": [e["name"] + f"#{e['i']}" for e in sounding if not e["played"]],
           "notes_out_of_order": out_of_order, "tempo_bpm_heard": tempo, "walked_mm": walked,
           "road_mm": round(total_mm, 1), "cpg_seed": seed, "physics_retries": attempts,
           "physics_error": r.get("physics_error"), "brain_runs": [sim["run_id"]], "raster_png": sim["raster_png"],
           "video": f"runs/{run_id}/video.mp4", "video_play_speed": 1 / AUDIO_SLOW, "audio": f"runs/{run_id}/song.wav",
           "audio_s": dur_audio, "piano": f"runs/{run_id}/piano.json", "phases": [], "body": _clean(r),
           "summary": (f"LESION: silenced {', '.join(silence)}. " if silence else "") +
                      f"'{title}': the fly (P9 {p9_hz:.0f} Hz -> {interface.describe(cmd).split('; ')[1]}) walked "
                      f"{walked:.1f} mm of a {total_mm:.0f} mm road and played {n_hit}/{len(sounding)} notes"
                      + (f" at ~{tempo:.0f} BPM (8x slowed, set by its own stepping)" if tempo else "")
                      + (f", {out_of_order} out of order" if out_of_order else "") + ".",
           "wall_s": round(time.perf_counter() - t0, 1),
           "framing": "The melody is the score we lay on the floor (each key placed just ahead of the fly along its "
                      "current heading; keys never touch or steer it); the brain only drives walking (P9), the legs "
                      "are physics, and the rhythm and missed notes come from the fly's real steps. Nothing learns."}
    cache.save_result(run_id, out)
    return out


# ----------------------------------------------------------------------------- 2. singer fly (brain only)
def song_orchestra(song="ode_to_joy", bpm: float = 100.0, bitter_notes: list[int] | None = None,
                   silence: list[str] | None = None) -> dict:
    from .behaviors import _save_cmds
    silence = silence or []
    title, notes = parse_song(song)
    bitter = set(bitter_notes or [])
    beat = 60.0 / bpm
    sched, t, spans = [], 0.0, []
    for i, (m, b) in enumerate(notes):
        dur = b * beat
        spans.append((round(t, 3), round(t + dur, 3)))
        if m is not None:
            on = round(t + min(0.8 * dur, 0.35), 3)
            sched.append({"neurons": ["sugar_GRNs"], "rate_hz": 200.0, "t0": round(t, 3), "t1": on})
            if i in bitter:
                sched.append({"neurons": ["bitter_GRNs"], "rate_hz": 200.0, "t0": round(t, 3), "t1": on})
        t += dur
    D = round(t + 0.3, 3)
    params = {"notes": notes, "bpm": bpm, "bitter": sorted(bitter), "silence": sorted(silence), "v": 1}
    lab_ = "song_orchestra" + (("__silence-" + "+".join(silence)) if silence else "")
    run_id = cache.run_id_for("beh", lab_, params)
    cached = cache.load_result(run_id)
    if cached is not None:
        cache.save_result(run_id, cached)
        return cached
    t0 = time.perf_counter()
    sim = lab.run_simulation([], silence, 200.0, D, 1, schedule=sched, label=f"{lab_}_brain")
    mn9 = np.asarray(sim["timeseries"]["MN9"])
    bin_s = sim["timeseries_bin_s"]
    seq, audio = [], []
    for i, ((m, b), (a, z)) in enumerate(zip(notes, spans)):
        hz = float(np.mean(mn9[int(a / bin_s):max(int(z / bin_s), int(a / bin_s) + 1)])) if m is not None else 0.0
        ok = m is not None and hz > 20.0
        seq.append({"i": i, "midi": m, "name": _name(m), "beats": b, "t": a, "played": ok, "mn9_hz": round(hz, 1),
                    "bitter": i in bitter})
        if ok:
            audio.append({"t": a, "midi": m, "vel": min(1.0, 0.25 + hz / 90), "inst": "lead",
                          "dur": max(0.25, (z - a) * 1.1)})
    d = cache.run_dir(run_id)
    dur_audio = _render_wav(audio, d / "song.wav")
    (d / "piano.json").write_text(json.dumps({"mode": "song", "title": title, "seq": seq, "time_scale": 1.0,
                                              "audio": f"runs/{run_id}/song.wav", "video": None}), encoding="utf8")
    _save_cmds(d, [interface.motor_commands(sim["timeseries"], bin_s)])
    sounding = [e for e in seq if e["midi"] is not None]
    plain = [e["mn9_hz"] for e in sounding if not e["bitter"]]
    bit = [e["mn9_hz"] for e in sounding if e["bitter"]]
    out = {"run_id": run_id, "kind": "behavior", "behavior": "song_orchestra", "scene": "piano", "label": lab_,
           "description": "Singer fly: one sugar pulse per note; a note sounds only if MN9 fires > 20 Hz.",
           "song": title, "silenced": silence, "bpm": bpm, "notes_total": len(sounding),
           "notes_played": sum(e["played"] for e in sounding),
           "mn9_mean_hz_sugar_notes": round(float(np.mean(plain)), 1) if plain else None,
           "mn9_mean_hz_bitter_notes": round(float(np.mean(bit)), 1) if bit else None,
           "brain_runs": [sim["run_id"]], "raster_png": sim["raster_png"], "video": None,
           "audio": f"runs/{run_id}/song.wav", "audio_s": dur_audio, "piano": f"runs/{run_id}/piano.json",
           "phases": [],
           "summary": (f"LESION: silenced {', '.join(silence)}. " if silence else "") +
                      f"'{title}' at {bpm:.0f} BPM: {sum(e['played'] for e in sounding)}/{len(sounding)} notes sang "
                      f"(MN9 mean {np.mean(plain) if plain else 0:.0f} Hz on sugar notes"
                      + (f", {np.mean(bit):.0f} Hz on the {len(bit)} bitter notes" if bit else "") + ").",
           "wall_s": round(time.perf_counter() - t0, 1),
           "framing": "The score (which note, when) is ours; the brain decides whether each note sounds (MN9 > 20 Hz) "
                      "and how loud. Sonification, not musicianship."}
    cache.save_result(run_id, out)
    return out
