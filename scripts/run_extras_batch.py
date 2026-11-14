"""Compute (or refresh from cache) every extra-activity result, one job at a time.
Usage: .venv/Scripts/python.exe -u scripts/run_extras_batch.py > runs/extras_batch.log
Each finished job prints one line: OK <name> [seconds] <run_id> | <summary>   (or FAIL ...)."""
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab import bike, olympics, piano, swatter  # noqa: E402

JOBS = [
    ("cave+bat", lambda: olympics.fly_through("cave", True)),
    ("forest+bat", lambda: olympics.fly_through("forest", True)),
    ("olympics canyon", lambda: olympics.run_olympics("canyon")),
    ("cave+bat, giant fiber silenced", lambda: olympics.fly_through("cave", True, ["giant_fiber"])),
    ("forest+bat, giant fiber silenced", lambda: olympics.fly_through("forest", True, ["giant_fiber"])),
    # cached (instant) unless parameters changed
    ("swatter", lambda: swatter.swat_fly("normal")),
    ("swatter, giant fiber silenced", lambda: swatter.swat_fly("normal", ["giant_fiber"])),
    ("bike", lambda: bike.ride_bike("loop")),
    ("bike, left DNa silenced", lambda: bike.ride_bike("loop", silence=["DNa01_left", "DNa02_left"])),
    ("piano keyboard", lambda: piano.play_piano("keyboard")),
    ("orchestra", lambda: piano.play_piano("orchestra")),
    ("song road", lambda: piano.play_piano("song_road", song="ode_to_joy")),
    ("singer", lambda: piano.play_piano("song_orchestra", song="ode_to_joy")),
    ("singer, bitter phrase 2", lambda: piano.play_piano("song_orchestra", song="ode_to_joy",
                                                         bitter_notes=list(range(15, 30)))),
    ("maze", lambda: olympics.run_maze()),
    ("maze, bitter GRNs silenced", lambda: olympics.run_maze(["bitter_GRNs"])),
]
only = set(sys.argv[1:])
for name, fn in JOBS:
    if only and name not in only:
        continue
    t = time.time()
    try:
        r = fn()
        print(f"OK {name} [{time.time() - t:.0f}s] {r['run_id']} | {r['summary']}", flush=True)
    except Exception:
        print(f"FAIL {name}: {traceback.format_exc()[-1500:]}", flush=True)
print("BATCH DONE", flush=True)
