"""Show one of the extra activities on the dashboard (instant when cached; otherwise it runs the simulation).

Usage:  .venv/Scripts/python.exe scripts/demo_extras.py <name>      (no name = list)
Then look at http://localhost:8000 (python dashboard/server.py). The dashboard shows the most recent result.
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEMOS = {
    "swatter": ("Fly swatter, intact brain (escapes)", lambda m: m["swatter"].swat_fly("normal")),
    "swatter_splat": ("Fly swatter, giant fiber silenced (splat)", lambda m: m["swatter"].swat_fly("normal", ["giant_fiber"])),
    "bike": ("Bicycle via ball-rig read-out, loop", lambda m: m["bike"].ride_bike("loop")),
    "bike_lesion": ("Bicycle, left DNa01/DNa02 silenced", lambda m: m["bike"].ride_bike("loop", silence=["DNa01_left", "DNa02_left"])),
    "piano": ("Fly piano: 3 flies walk on a keyboard (video with sound)", lambda m: m["piano"].play_piano("keyboard")),
    "orchestra": ("Brain orchestra: bass / melody / drums flies", lambda m: m["piano"].play_piano("orchestra")),
    "song_road": ("Musical road: a walking fly plays Ode to Joy", lambda m: m["piano"].play_piano("song_road", song="ode_to_joy")),
    "singer": ("Singer fly: Ode to Joy via MN9", lambda m: m["piano"].play_piano("song_orchestra", song="ode_to_joy")),
    "singer_bitter": ("Singer fly with bitter on phrase 2 (goes silent)",
                      lambda m: m["piano"].play_piano("song_orchestra", song="ode_to_joy", bitter_notes=list(range(15, 30)))),
    "maze": ("Sugar maze: eats sugar, refuses poison", lambda m: m["olympics"].run_maze()),
    "maze_poison": ("Sugar maze, bitter GRNs silenced (eats poison)", lambda m: m["olympics"].run_maze(["bitter_GRNs"])),
    "cave": ("Cave fly-through with a bat", lambda m: m["olympics"].fly_through("cave", True)),
    "forest": ("Forest fly-through with a bat", lambda m: m["olympics"].fly_through("forest", True)),
    "olympics": ("Connectome Olympics on the canyon", lambda m: m["olympics"].run_olympics("canyon")),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in DEMOS:
        print(__doc__)
        for k, (d, _) in DEMOS.items():
            print(f"  {k:14s} {d}")
        return
    from flylab import bike, olympics, piano, swatter
    mods = {"bike": bike, "olympics": olympics, "piano": piano, "swatter": swatter}
    t = time.time()
    r = DEMOS[sys.argv[1]][1](mods)
    # put this run's own brain simulation on the brain panel too, then the behaviour back on top
    from flylab import cache
    for rid in (r.get("brain_runs") or [])[:1]:
        sim = cache.load_result(rid)
        if sim is not None:
            cache.save_result(rid, sim)
    cache.save_result(r["run_id"], r)
    print(f"[{time.time() - t:.1f} s] {r['run_id']}\n{r['summary']}")


if __name__ == "__main__":
    main()
