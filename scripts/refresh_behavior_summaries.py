"""Recompute the text summary of cached behaviour runs (no re-rendering), e.g. after a summary-format fix."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from flylab.behaviors import _summary  # noqa: E402

for p in sorted((REPO / "runs").glob("beh_*/result.json")):
    r = json.loads(p.read_text(encoding="utf8"))
    if not r.get("video"):
        continue
    r["summary"] = _summary(r["behavior"], r["phases"], r.get("silenced") or [])
    r.setdefault("video_play_speed", 0.25)
    r.setdefault("video_note", "slow motion: video time x play_speed = simulated time")
    p.write_text(json.dumps(r, indent=1, default=float), encoding="utf8")
    print(p.parent.name, "|", r["summary"][:140])
