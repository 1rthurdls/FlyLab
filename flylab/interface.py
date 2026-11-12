"""The brain -> body interface. HAND-DESIGNED, documented here in one place.

The connectome model ends at brain output neurons (descending neurons and brain motor neurons).
The real fly's ventral nerve cord, which turns descending commands into leg/proboscis movements,
is NOT simulated. Instead FlyLab maps the firing rates of a few well-studied output neurons onto
the high-level commands of flygym's walking controller (NeuroMechFly v2):

  forward  = clip(P9_rate / P9_FULL_HZ, 0, 1)
             P9 (FlyWire DNp09) initiates forward walking (Bidaye et al. 2020).
  backward = clip(MDN_rate / MDN_FULL_HZ, 0, 1)
             MDN activation drives backward walking (Bidaye et al. 2014).
  turn     = clip((L - R) / TURN_FULL_HZ, -1, 1),   L = DNa01_left + DNa02_left, R = same on the right
             DNa02 predicts ipsilateral turning (Rayshubskiy et al. 2020); positive = turn left.
  feeding  = clip(MN9_rate / MN9_FULL_HZ, 0, 1); the fly is in the FEEDING state when the
             MN9 rate exceeds MN9_THRESHOLD_HZ. While feeding, walking is suppressed
             (forward = backward = turn = 0): a design rule standing in for the VNC/behavioural
             arbitration that the model does not contain.
             MN9 = FlyWire CB0701, proboscis motor neuron (Shiu et al. 2024).

  takeoff  = the giant fiber (FlyWire DNp01, escape take-off command neuron) exceeds GF_THRESHOLD_HZ.
             The body then plays a SCRIPTED take-off/flight animation (no aerodynamics; the NeuroMechFly
             walking model cannot fly). Only the trigger comes from the brain.
             Measured: looming detectors LPLC2 at 50 Hz -> GF ~95-118 Hz; sugar -> GF 0 Hz.

Rates are population means (Hz per neuron, averaged over trials) in 20 ms bins, smoothed with a
60 ms moving average. The constants below were set from measured model rates
(see PROGRESS.md, "Interface calibration"):
  * directly stimulated DNs (100 Hz Poisson input) fire ~100 Hz -> FULL = 100 Hz
  * MN9: 0 Hz at baseline (the model has no spontaneous activity), ~78 Hz with sugar GRNs at 200 Hz,
    ~11 Hz with sugar + bitter GRNs -> threshold 30 Hz (between the two), FULL 80 Hz.
"""
from __future__ import annotations

import numpy as np

P9_FULL_HZ = 100.0
MDN_FULL_HZ = 100.0
TURN_FULL_HZ = 100.0
MN9_FULL_HZ = 80.0
MN9_THRESHOLD_HZ = 30.0
GF_THRESHOLD_HZ = 50.0
SMOOTH_BINS = 3

MAPPING_TEXT = __doc__


def _smooth(x, n=SMOOTH_BINS):
    x = np.asarray(x, float)
    if n <= 1 or len(x) == 0:
        return x
    k = np.ones(n) / n
    pad = np.concatenate([np.full(n - 1, x[0]), x])
    return np.convolve(pad, k, mode="valid")


def motor_commands(timeseries: dict, bin_s: float) -> dict:
    """timeseries: {group: [rate per bin]} as produced by lab.run_simulation. Returns per-bin commands."""
    g = {k: _smooth(v) for k, v in timeseries.items()}
    n = len(g["P9"])
    L = g["DNa01_left"] + g["DNa02_left"]
    R = g["DNa01_right"] + g["DNa02_right"]
    forward = np.clip(g["P9"] / P9_FULL_HZ, 0, 1)
    backward = np.clip(g["MDN"] / MDN_FULL_HZ, 0, 1)
    turn = np.clip((L - R) / TURN_FULL_HZ, -1, 1)
    feeding = np.clip(g["MN9"] / MN9_FULL_HZ, 0, 1)
    feeding_state = g["MN9"] > MN9_THRESHOLD_HZ
    forward = np.where(feeding_state, 0.0, forward)
    backward = np.where(feeding_state, 0.0, backward)
    turn = np.where(feeding_state, 0.0, turn)
    feeding = np.where(feeding_state, feeding, 0.0)
    gf = g.get("giant_fiber", np.zeros(n))
    takeoff_state = gf > GF_THRESHOLD_HZ
    return {"bin_s": bin_s, "n_bins": n, "forward": forward, "backward": backward, "turn": turn,
            "feeding": feeding, "feeding_state": feeding_state, "takeoff_state": takeoff_state,
            "rates": {"P9": g["P9"], "MDN": g["MDN"], "DNa_left": L, "DNa_right": R, "MN9": g["MN9"], "GF": gf}}


def describe(cmd: dict, t0_bin: int = 0, t1_bin: int | None = None) -> str:
    """One-line plain-language summary of a stretch of commands, with the numbers that caused it."""
    s = slice(t0_bin, t1_bin)
    r = {k: float(np.mean(v[s])) for k, v in cmd["rates"].items()}
    parts = []
    feed_frac = float(np.mean(cmd["feeding_state"][s]))
    parts.append(f"MN9 {r['MN9']:.0f} Hz -> " + ("FEEDING (proboscis extension)" if feed_frac > 0.5
                                                 else f"no feeding (threshold {MN9_THRESHOLD_HZ:.0f} Hz)"))
    fw, bw, tn = (float(np.mean(cmd[k][s])) for k in ("forward", "backward", "turn"))
    parts.append(f"P9 {r['P9']:.0f} Hz -> forward drive {fw:.2f}")
    parts.append(f"MDN {r['MDN']:.0f} Hz -> backward drive {bw:.2f}")
    parts.append(f"DNa L/R {r['DNa_left']:.0f}/{r['DNa_right']:.0f} Hz -> turn {tn:+.2f} ({'left' if tn > 0.02 else 'right' if tn < -0.02 else 'straight'})")
    if r.get("GF", 0) > 1:
        parts.append(f"giant fiber {r['GF']:.0f} Hz -> " + ("TAKE-OFF" if r["GF"] > GF_THRESHOLD_HZ else
                                                            f"no take-off (threshold {GF_THRESHOLD_HZ:.0f} Hz)"))
    return "; ".join(parts)


def to_program(cmd: dict, t0_bin: int = 0, t1_bin: int | None = None):
    from .body import MotorProgram
    s = slice(t0_bin, t1_bin)
    return MotorProgram(bin_s=cmd["bin_s"], forward=cmd["forward"][s], turn=cmd["turn"][s],
                        backward=cmd["backward"][s], feeding=cmd["feeding"][s])
