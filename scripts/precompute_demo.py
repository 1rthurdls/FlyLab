"""Pre-run and cache every scenario in DEMO.md so the recorded take is fast.

Uses exactly the same functions and default parameters as the MCP tools, so when Claude calls
the tools during the recording the results come straight from runs/ (cache hits).

Usage: python scripts/precompute_demo.py [--brain-only] [--clear-feed]
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from flylab import behaviors, lab  # noqa: E402

BRAIN_ONLY = "--brain-only" in sys.argv
T0 = time.time()


def step(name, fn, *a, **k):
    t = time.time()
    r = fn(*a, **k)
    extra = ""
    if isinstance(r, dict):
        if "output_groups" in r and "MN9" in r["output_groups"] and isinstance(r["output_groups"]["MN9"], dict):
            og = r["output_groups"]
            extra = "MN9 %.1f Hz" % og["MN9"].get("mean_rate_hz", og["MN9"].get("b_hz", 0))
        if "summary" in r:
            extra = r["summary"][:160]
        if "results" in r:
            extra = ", ".join(f"{x['silenced']} {x['change_pct']}%" for x in r["results"][:4])
    print(f"[{time.time() - T0:6.0f}s] {name:48s} {time.time() - t:6.1f}s  {extra}", flush=True)
    return r


# ---- hero: "Make the fly eat when it finds sugar, then prove which neuron is responsible"
step("trace sugar->MN9", lab.trace_pathway, "sugar_GRNs", "MN9")
ctrl = step("sim sugar", lab.run_simulation, ["sugar_GRNs"])
step("functional trace", lab.trace_pathway, "sugar_GRNs", "MN9", 4, ctrl["run_id"])
screen = step("lesion screen sugar->MN9", lab.lesion_screen, ["sugar_GRNs"], "MN9")
hit = screen["results"][0]["silenced"]
les = step(f"sim sugar, silence {hit}", lab.run_simulation, ["sugar_GRNs"], [hit])
step("compare control vs lesion", lab.compare_conditions, ctrl["run_id"], les["run_id"])
step("sim sugar, silence MN9", lab.run_simulation, ["sugar_GRNs"], ["MN9"])

# ---- other scenarios from the skill's worked examples
step("trace bitter->MN9", lab.trace_pathway, "bitter_GRNs", "MN9")
step("sim sugar+bitter", lab.run_simulation, ["sugar_GRNs", "bitter_GRNs"])
step("sim bitter", lab.run_simulation, ["bitter_GRNs"])
step("sim MDN 100Hz", lab.run_simulation, ["MDN"], [], 100.0)
step("sim P9 100Hz", lab.run_simulation, ["P9"], [], 100.0)
step("sim JO grooming", lab.run_simulation, ["JO_grooming"])
step("trace P9->DNa02_right", lab.trace_pathway, "P9", "DNa02_right")

if not BRAIN_ONLY:
    for b, sil in [("find_sugar", []), ("find_sugar", [hit]), ("feed", []), ("feed_with_bitter", []),
                   ("walk_forward", []), ("walk_backward", []), ("walk_backward", ["MDN"]), ("turn_left", []),
                   ("turn_right", []), ("groom", [])]:
        step(f"behaviour {b}" + (f" silence {sil}" if sil else ""), behaviors.run_behavior, b, [], sil, None)

# ---- extra activities: escape take-off (restricted, scripted flight) and the fly-driven car
from flylab import car  # noqa: E402
step("sim LPLC2 50Hz (looming -> giant fiber)", lab.run_simulation, ["LPLC2"], [], 50.0)
step("trace LPLC2->giant_fiber", lab.trace_pathway, "LPLC2", "giant_fiber")
for p_ in ["u_turn", "slalom", "parallel_park", "figure_eight"]:
    step(f"car preset {p_}", car.drive_car, None, p_)
step("car u_turn, left steering silenced", car.drive_car, None, "u_turn", None, ["DNa01_left", "DNa02_left"])
step("car goal=station", car.drive_car, None, None, "station")
if not BRAIN_ONLY:
    step("behaviour takeoff", behaviors.run_behavior, "takeoff", [], [], None)
    step("behaviour takeoff silence giant_fiber", behaviors.run_behavior, "takeoff", [], ["giant_fiber"], None)

# leave the dashboard on the hero scenario, intact brain
if not BRAIN_ONLY:
    behaviors.run_behavior("find_sugar")
lab.run_simulation(["sugar_GRNs"])
if "--clear-feed" in sys.argv:
    ev = REPO / "runs" / "events.jsonl"
    if ev.exists():
        ev.rename(REPO / "runs" / f"events_archive_{int(time.time())}.jsonl")
    lr = REPO / "runs" / "latest_report.json"
    if lr.exists():
        lr.unlink()
print(f"done in {time.time() - T0:.0f}s; lesion hit = {hit}")
(REPO / "runs" / "demo_cache.json").write_text(json.dumps({"lesion_hit": hit, "control": ctrl["run_id"],
                                                           "lesion": les["run_id"]}, indent=1))
