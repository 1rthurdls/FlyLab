"""Shared DESIGNED looming eye model (used by the fly swatter and the bat in the fly-throughs).

An object of half-size r approaching the eye at speed v, at distance d, subtends an angle
theta = 2 atan(r / d); it expands at d(theta)/dt = 2 r v / (d^2 + r^2) (rad/s).
FlyLab drives the looming-detector visual projection neurons (LPLC2 and LC4, both sides) with a Poisson rate
proportional to that expansion speed, saturating at LOOM_SAT_DEG_S:

    rate_Hz = LOOM_MAX_HZ * clip(d(theta)/dt / LOOM_SAT_DEG_S, 0, 1)

This mapping is a hand-designed eye model (the optic lobe is not simulated). What happens next is the
connectome: LPLC2/LC4 -> ... -> giant fiber (DNp01). Measured in this model (0.5 s, 4 trials, no walking):
LPLC2 at 10/20/30/50 Hz -> giant fiber 33/64/85/105 Hz; LC4 alone at 20/30/50 Hz -> 8/26/47 Hz.
"""
from __future__ import annotations

import math

LOOM_MAX_HZ = 100.0
LOOM_SAT_DEG_S = 500.0
LOOM_GROUPS = ("LPLC2", "LC4")


def expansion_deg_s(d: float, v: float, r: float) -> float:
    """Angular expansion speed (deg/s) of an object of half-size r at distance d approaching at speed v."""
    d = max(d, 0.0)
    return math.degrees(2.0 * r * v / (d * d + r * r)) if v > 0 else 0.0


def angle_deg(d: float, r: float) -> float:
    return math.degrees(2.0 * math.atan2(r, max(d, 1e-9)))


def loom_rate_hz(d: float, v: float, r: float) -> float:
    return LOOM_MAX_HZ * min(max(expansion_deg_s(d, v, r) / LOOM_SAT_DEG_S, 0.0), 1.0)
