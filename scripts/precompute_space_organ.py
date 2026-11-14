"""Precompute the Time Pulse piece (singer fly ~5 min, then the musical road ~30 min) so the dashboard and the
ready-made prompt are instant. Usage: .venv/Scripts/python.exe -u scripts/precompute_space_organ.py"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab import piano  # noqa: E402

for name, kw in [("singer", dict(mode="song_orchestra", song="time_pulse", bpm=60)),
                 ("singer + bitter on the climb (notes 16-27)", dict(mode="song_orchestra", song="time_pulse", bpm=60,
                                                       bitter_notes=list(range(16, 28)))),
                 ("musical road", dict(mode="song_road", song="time_pulse"))]:
    t = time.time()
    r = piano.play_piano(**kw)
    print(f"OK {name} [{time.time() - t:.0f}s] {r['run_id']} | {r['summary']}", flush=True)
print("DONE", flush=True)
