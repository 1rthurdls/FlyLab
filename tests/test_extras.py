"""Unit tests for the extra activities' pure functions (no brain / body simulation).
Run: python -m pytest -q tests
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab import arena, bike, looming, piano  # noqa: E402


def test_looming_expansion_grows_as_object_approaches():
    r, v = 20.0, 500.0
    rates = [looming.loom_rate_hz(d, v, r) for d in (150, 100, 60, 30)]
    assert all(a < b for a, b in zip(rates, rates[1:]))
    assert rates[-1] == looming.LOOM_MAX_HZ                     # saturates near contact
    assert looming.loom_rate_hz(50, 0.0, r) == 0.0              # a still object does not loom
    assert abs(looming.angle_deg(r, r) - 90.0) < 1e-6           # at d = r the object spans 90 deg


def test_piano_keys_are_pentatonic_and_ordered():
    assert piano.key_of(piano.KEY0_X + 0.1) == 0
    assert piano.key_of(piano.KEY0_X + piano.KEY_W * 7.5) == 7
    assert piano.key_of(-1e6) == 0 and piano.key_of(1e6) == piano.N_KEYS - 1
    m = [piano.midi_of(k) for k in range(piano.N_KEYS)]
    assert all(a < b for a, b in zip(m, m[1:]))
    assert {x % 12 for x in m} <= {0, 2, 4, 7, 9}               # C-major pentatonic only


def test_ball_readout_of_straight_and_turning_walk():
    t = np.arange(0, 1.0, 0.01)
    straight = [(ti, 10.0 * ti, 0.0, 0.0) for ti in t]           # 10 mm/s along +x, heading 0
    _, v, yaw = bike._ball_readout(straight)
    assert abs(np.median(v) - 10.0) < 0.5 and abs(np.median(yaw)) < 1.0
    turning = [(ti, 0.0, 0.0, 90.0 * ti) for ti in t]           # turning on the spot, +90 deg/s
    _, v, yaw = bike._ball_readout(turning)
    assert abs(np.median(yaw) - 90.0) < 3.0 and abs(np.median(v)) < 0.5


def test_bike_kinematics_follow_the_legs():
    t = np.arange(1, 51) * 0.02
    rows = bike._ride(t, np.full(50, 10.0), np.zeros(50))
    assert abs(rows[-1][1] - bike.GEAR * 10.0 * 1.0) < 1e-6 and abs(rows[-1][2]) < 1e-9


def test_new_courses_place_pellets_and_obstacles_inside_the_corridor():
    for name in ("cave", "forest", "garden"):
        W = arena.build_course(name)
        C = arena.COURSES[name]
        for s, lat, *_ in C.get("obstacles", []) + C.get("pellets", []):
            x, y = arena._place(W, s, lat)
            P = W["centerline"]
            d = np.min(np.hypot(P[:, 0] - x, P[:, 1] - y))
            assert d <= C["width"] / 2 - 0.5, (name, s, lat, d)
    # the default course keeps its old parameters (cached runs stay valid)
    assert "obstacles" not in arena.COURSES["canyon"] and "pellets" not in arena.COURSES["canyon"]
    assert math.isclose(arena.COURSES["canyon"]["width"], 10.0)


def test_song_parser_and_builtin_songs():
    from flylab import song
    title, notes = song.parse_song("C4 C#4:0.5 Bb3:2 R:1")
    assert notes == [(60, 1.0), (61, 0.5), (58, 2.0), (None, 1.0)]
    for name in song.SONGS:
        t, n = song.parse_song(name)
        assert ("public domain" in t or "original" in t) and 20 <= len(n) and sum(b for _, b in n) <= 64
    import pytest
    with pytest.raises(ValueError):
        song.parse_song("H4 X")
