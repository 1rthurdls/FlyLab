"""Render 4 short test clips of the FlyLab body into runs/body_test/.

    .venv/Scripts/python.exe scripts/test_body.py            # 4 clips in parallel
    .venv/Scripts/python.exe scripts/test_body.py --serial   # one after another
    .venv/Scripts/python.exe scripts/test_body.py --only feeding

Clips (simulated time; video plays at 0.25x):
  forward.mp4    forward=1                                  1.5 s
  turn_left.mp4  forward=0.8, turn=+0.6                     1.5 s
  backward.mp4   backward=1                                 1.5 s
  feeding.mp4    sugar drop ~9 mm ahead; forward=0.7 until a front leg / the
                 proboscis is within sugar_radius of the drop (run() with
                 stop_on_sugar_contact=True, max 1.4 s), then feeding=1
                 for 1.0 s (walking is suppressed, proboscis extends).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "runs" / "body_test"
BIN = 0.02
SUGAR = (9.0, 2.0)


def _clip(name: str) -> dict:
    from flylab.body import FlyBody, MotorProgram, render_program

    t0 = time.perf_counter()
    out = OUT / f"{name}.mp4"
    if name == "forward":
        s = render_program(MotorProgram.constant(1.5, BIN, forward=1.0), out)
    elif name == "turn_left":
        s = render_program(MotorProgram.constant(1.5, BIN, forward=0.8, turn=0.6), out)
    elif name == "backward":
        s = render_program(MotorProgram.constant(1.5, BIN, backward=1.0), out)
    elif name == "feeding":
        body = FlyBody(sugar_pos=SUGAR)
        approach = body.run(MotorProgram.constant(1.4, BIN, forward=0.7),
                            stop_on_sugar_contact=True)
        feed = body.run(MotorProgram.constant(1.0, BIN, feeding=1.0))
        body.save_video(out)
        s = {"approach": approach, "feeding": feed, "video": str(out)}
        body.close()
    else:
        raise ValueError(name)
    s.pop("trajectory", None)
    s["clip"] = name
    s["clip_wall_s"] = round(time.perf_counter() - t0, 1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    names = args.only or ["forward", "turn_left", "backward", "feeding"]
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.serial or len(names) == 1:
        results = [_clip(n) for n in names]
    else:
        with ProcessPoolExecutor(max_workers=len(names)) as ex:
            results = list(ex.map(_clip, names))
    for r in results:
        print(f"\n=== {r['clip']}  (wall-clock {r['clip_wall_s']} s)")
        print(json.dumps(r, indent=1))
    print(f"\ntotal wall-clock {time.perf_counter() - t0:.1f} s")
    for n in names:
        p = OUT / f"{n}.mp4"
        size = p.stat().st_size if p.exists() else 0
        print(f"{p}: {size / 1e3:.0f} kB {'OK' if size > 20_000 else 'MISSING/TRIVIAL'}")


if __name__ == "__main__":
    main()
