"""Connectome Olympics: the same brain with different lesions races the same closed-loop course.

Every contestant is run_course(course, silence=...) (closed loop, no planner; see arena.py). The ranking is
finished runs by time, then everyone else by distance covered. It is a lesion experiment presented as a race:
each result is a model prediction about what that neuron population contributes to steering.
Also hosts the wrappers fly_through (themed courses + bat) and run_maze (sugar maze).
"""
from __future__ import annotations

import json
import time

from . import cache
from .arena import COURSES, run_course

DEFAULT_CONTESTANTS = [
    {"name": "Intact brain", "silence": []},
    {"name": "Left eye's LPLC1 off", "silence": ["LPLC1_L"]},
    {"name": "No left steering (DNa01+DNa02 left)", "silence": ["DNa01_left", "DNa02_left"]},
    {"name": "Blind to walls (all LPLC1)", "silence": ["LPLC1"]},
]
BAT_AT_S = 3.0
COLORS = ["#3987e5", "#199e70", "#c98500", "#d55181", "#d95926", "#8a6fd1"]


def run_olympics(course: str = "canyon", contestants: list[dict] | None = None) -> dict:
    if course not in COURSES or "pellets" in COURSES[course]:
        raise ValueError(f"pick a race course: {[c for c in COURSES if 'pellets' not in COURSES[c]]}")
    cs = contestants or DEFAULT_CONTESTANTS
    cs = [{"name": c.get("name") or ("Intact brain" if not c.get("silence") else "Silenced " + "+".join(c["silence"])),
           "silence": list(c.get("silence") or [])} for c in cs][:6]
    params = {"course": course, "contestants": cs, "v": 1}
    run_id = cache.run_id_for("beh", f"olympics_{course}", params)
    cached = cache.load_result(run_id)
    if cached is not None:
        cache.save_result(run_id, cached)
        return cached
    t0 = time.perf_counter()
    racers = []
    for i, c in enumerate(cs):
        r = run_course(course, c["silence"])
        car = json.loads((cache.REPO / r["car"]).read_text(encoding="utf8"))
        racers.append({"name": c["name"], "silence": c["silence"], "color": COLORS[i % len(COLORS)],
                       "run_id": r["run_id"], "outcome": r["outcome"], "time_s": r["time_s"],
                       "progress_m": r["progress_m"], "progress_pct": r["progress_pct"], "crash": r["crash"],
                       "brain": r["brain"], "trajectory": car["trajectory"], "eyes_hz": car["eyes_hz"],
                       "walls": car["walls"], "finish": car["finish"]})
    order = sorted(range(len(racers)), key=lambda i: (racers[i]["outcome"] != "finished",
                                                     racers[i]["time_s"] if racers[i]["outcome"] == "finished"
                                                     else -racers[i]["progress_m"]))
    for rank, i in enumerate(order, 1):
        racers[i]["rank"] = rank
    d = cache.run_dir(run_id)
    first = racers[0]
    (d / "car.json").write_text(json.dumps({
        "mode": "race", "course": course, "walls": first["walls"], "finish": first["finish"],
        "racers": [{k: v for k, v in r.items() if k not in ("walls", "finish", "brain")} for r in racers],
        "trajectory": max((r["trajectory"] for r in racers), key=len), "maneuvers": []}), encoding="utf8")
    board = [{"rank": racers[i]["rank"], "name": racers[i]["name"], "silence": racers[i]["silence"],
              "outcome": racers[i]["outcome"], "time_s": racers[i]["time_s"], "progress_pct": racers[i]["progress_pct"],
              "DNa_L_R_mean_hz": [racers[i]["brain"]["DNa_left_mean_hz"], racers[i]["brain"]["DNa_right_mean_hz"]],
              "run_id": racers[i]["run_id"]} for i in order]
    line = "; ".join(f"#{b['rank']} {b['name']}: " + (f"finished in {b['time_s']:.1f} s" if b["outcome"] == "finished"
                                                     else f"{b['outcome']} at {b['progress_pct']:.0f}%")
                     for b in board)
    from .behaviors import _save_cmds  # noqa: F401
    first_cmd = cache.RUNS / racers[order[0]]["run_id"] / "commands.json"
    if first_cmd.exists():
        (d / "commands.json").write_text(first_cmd.read_text(encoding="utf8"), encoding="utf8")
    winner = cache.load_result(racers[order[0]]["run_id"])
    out = {"run_id": run_id, "kind": "behavior", "behavior": f"olympics_{course}", "scene": "car",
           "label": f"olympics_{course}", "description": "Same brain, different lesions, same closed-loop course.",
           "silenced": [], "course": course, "leaderboard": board, "phases": [], "segments": [], "maneuvers": [],
           "brain_runs": [r["run_id"] for r in racers], "raster_png": winner.get("raster_png"), "video": None,
           "car": f"runs/{run_id}/car.json", "summary": f"Connectome Olympics on the {course}: {line}.",
           "wall_s": round(time.perf_counter() - t0, 1),
           "framing": "Each contestant is a closed-loop run of the same connectome with a different lesion; results "
                      "are model predictions. Nothing learns."}
    cache.save_result(run_id, out)
    return out


def fly_through(place: str = "cave", bat: bool = True, silence: list[str] | None = None) -> dict:
    """Themed closed-loop fly-through (cave / forest). Flight is NOT simulated: the fly avatar glides at a fixed
    height and is steered by the brain exactly like run_course. With bat=True a bat swoops in head-on at BAT_AT_S."""
    if place not in ("cave", "forest"):
        raise ValueError("place must be 'cave' or 'forest'")
    return run_course(place, silence or [], 25.0, 0, BAT_AT_S if bat else None)


def run_maze(silence: list[str] | None = None) -> dict:
    """Sugar maze: hedge corridor with sugar and poisoned (sugar + bitter) pellets, closed loop."""
    return run_course("garden", silence or [], 40.0)


if __name__ == "__main__":
    import sys
    r = run_olympics(sys.argv[1] if len(sys.argv) > 1 else "canyon")
    print(r["summary"], "| wall", r["wall_s"], "s")
