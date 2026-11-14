"""Precompute 22 Questions (user-supplied arrangement): melody+bass duet (~10-20 min), duet with bitter on the
chorus (~10-20 min), then the musical road video (~30-40 min). Usage: .venv/Scripts/python.exe -u scripts/precompute_22q.py"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab import piano  # noqa: E402

for name, kw in [("duet", dict(mode="song_orchestra", song="twenty_two_questions", bpm=90)),
                 ("duet + bitter chorus", dict(mode="song_orchestra", song="twenty_two_questions", bpm=90,
                                               bitter_notes=list(range(15, 28)))),
                 ("musical road", dict(mode="song_road", song="twenty_two_questions"))]:
    t = time.time()
    r = piano.play_piano(**kw)
    print(f"OK {name} [{time.time() - t:.0f}s] {r['run_id']} | {r['summary']}", flush=True)
print("DONE", flush=True)
