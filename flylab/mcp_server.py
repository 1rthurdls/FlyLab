"""FlyLab MCP server (stdio). Exposes the whole-brain fly simulation as tools for Claude.

Run: python -m flylab.mcp_server   (registered for Claude Code in .mcp.json)
Built on the official Python MCP SDK v2 (`MCPServer`, formerly `FastMCP`).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anyio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from mcp.server.mcpserver import MCPServer  # noqa: E402

import threading  # noqa: E402


class _LazyLab:
    """flylab.lab pulls in torch/pandas/the connectome (~10-20 s cold). Import it lazily so the MCP handshake is
    instant, and warm it up in a background thread right after start."""

    def __getattr__(self, name):
        from flylab import lab as _lab
        return getattr(_lab, name)


lab = _LazyLab()


def _warmup():
    try:
        from flylab import connectome, lab as _lab  # noqa: F401
        connectome.annotations()
        _lab.load_groups()
    except Exception:
        pass


threading.Thread(target=_warmup, daemon=True).start()

INSTRUCTIONS = """FlyLab: a whole-brain spiking model of the adult fruit fly (FlyWire v783 connectome, 138,639
neurons, ~15M connections, LIF model of Shiu et al. 2024) wired to a physics-simulated body (NeuroMechFly/flygym).
The wiring is fixed and never learns: you find and drive circuits that already exist.
Typical loop: search_neurons / list_neuron_groups -> trace_pathway -> run_simulation -> run_behavior
-> lesion_screen / run_simulation(silence=...) + compare_conditions -> get_literature_facts -> save_report.
Names accepted everywhere: curated group names (see list_neuron_groups), FlyWire cell types (e.g. 'CB0616'),
cell type + side ('CB0616_L') or a FlyWire root id."""

server = MCPServer(name="flylab", instructions=INSTRUCTIONS)


def _j(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


async def _run(fn, *a, **k) -> str:
    try:
        return _j(await anyio.to_thread.run_sync(lambda: fn(*a, **k)))
    except Exception as e:  # report errors as data so Claude can recover
        return _j({"error": f"{type(e).__name__}: {e}"})


@server.tool()
async def search_neurons(query: str, limit: int = 20) -> str:
    """Search the FlyWire annotations (cell type, class, sub-class, super-class, hemilineage,
    neurotransmitter, synonyms), e.g. 'sugar', 'bitter', 'MDN', 'descending', 'proboscis', 'grooming'.
    Returns matching cell types with neuron counts, sides and example root ids."""
    from flylab import connectome
    return await _run(connectome.search_neurons, query, limit)


@server.tool()
async def list_neuron_groups() -> str:
    """List curated neuron groups (sensory inputs and motor/descending outputs) resolved from data."""
    return await _run(lab.list_neuron_groups)


@server.tool()
async def get_neuron_group(name: str) -> str:
    """Details of one curated group: description, data source, cell types, sides, root ids."""
    return await _run(lab.get_neuron_group, name)


@server.tool()
async def trace_pathway(source: str, target: str, max_hops: int = 4, functional_run_id: str | None = None) -> str:
    """Strongest synaptic paths from a source group/cell type to a target group/cell type.
    Path strength = product of input fractions (edges >= 5 synapses). net_sign tells whether the path is
    excitatory or inhibitory overall. Pass functional_run_id (a run_simulation run_id) to restrict
    intermediates to neurons that were actually active in that simulation."""
    return await _run(lab.trace_pathway, source, target, max_hops, functional_run_id)


@server.tool()
async def run_simulation(activate: list[str], silence: list[str] | None = None, stim_rate_hz: float = 200.0,
                         duration_s: float = 1.0, n_trials: int = 8) -> str:
    """Simulate the whole brain: drive `activate` neurons with Poisson input (stim_rate_hz) and set the outgoing
    synapses of `silence` neurons to zero. Returns firing rates of every curated output group, top responding
    neurons and a raster PNG path. Cached by parameters. ~10-20 s per simulated second on a laptop CPU."""
    return await _run(lab.run_simulation, activate, silence or [], stim_rate_hz, duration_s, n_trials)


@server.tool()
async def compare_conditions(run_id_a: str, run_id_b: str) -> str:
    """Compare two simulation runs (e.g. control vs lesion): change of each output group and the
    neurons whose rate decreased / increased most."""
    return await _run(lab.compare_conditions, run_id_a, run_id_b)


@server.tool()
async def lesion_screen(activate: list[str], target: str, candidates: list[str] | None = None,
                        n_candidates: int = 8, stim_rate_hz: float = 200.0, duration_s: float = 1.0) -> str:
    """Silence candidate neurons one at a time while activating `activate`, and rank them by how much the
    `target` group's firing drops. If candidates is omitted, they are picked automatically from the
    functional pathway (active neurons on the strongest source->target paths). Takes ~10-20 s per candidate."""
    return await _run(lab.lesion_screen, activate, target, candidates, n_candidates, stim_rate_hz, duration_s)


@server.tool()
async def list_behaviors() -> str:
    """Catalogue of behaviours FlyLab can show on the 3D body, with the neurons they stimulate and
    whether they were validated."""
    from flylab import behaviors
    return await _run(behaviors.list_behaviors)


@server.tool()
async def run_behavior(behavior: str, silence: list[str] | None = None, activate: list[str] | None = None,
                       duration_s: float | None = None) -> str:
    """Run a behaviour end to end: whole-brain simulation -> output-neuron rates -> documented brain->body
    interface (flylab/interface.py) -> physics-simulated fly (flygym) -> mp4 video.
    Use `silence` for lesion experiments (e.g. silence=['MN9']) and `activate` to add extra stimulation.
    Behaviours: walk_forward, turn_left, turn_right, walk_backward, feed, feed_with_bitter, find_sugar,
    takeoff (looming -> giant fiber -> RESTRICTED scripted take-off animation; the walking body cannot really fly),
    groom (brain only). Body videos take 2-8 min unless cached.
    Returns the video path and a plain-language summary of what the body did and why."""
    from flylab import behaviors
    return await _run(behaviors.run_behavior, behavior, activate or [], silence or [], duration_s)


@server.tool()
async def drive_car(maneuvers: list[dict] | None = None, preset: str | None = None, goal: str | None = None,
                    silence: list[str] | None = None) -> str:
    """The fly brain drives a car (shown as a 3D scene on the dashboard). Each manoeuvre stimulates real
    descending neurons in the whole-brain model: forward = P9; left/right = P9 + that side's DNa01/DNa02;
    reverse = MDN; reverse_left/reverse_right = MDN + DNa of that side; stop = nothing. The car follows what
    those neurons actually fire (hand-designed car link: P9 -> throttle, MDN -> reverse, DNa L-R -> steering).
    Use ONE of: maneuvers=[{"action": "left", "seconds": 0.5}, ...] (max 15 s total),
    preset = "u_turn" | "slalom" | "parallel_park" | "figure_eight", or goal = "station" (drive to the sugar
    station; a planner re-plans on the real brain output until the car arrives, ~1-3 min).
    silence = neurons to lesion (e.g. ["DNa01_left", "DNa02_left"] -> the car cannot turn left).
    Returns per-manoeuvre distance/turn, end position and distance to the sugar station."""
    from flylab import car
    return await _run(car.drive_car, maneuvers, preset, goal, silence or [])


@server.tool()
async def run_course(course: str = "canyon", silence: list[str] | None = None, max_s: float = 25.0) -> str:
    """CLOSED-LOOP obstacle course, no planner: the car must get through a corridor of SOLID walls (touching one =
    CRASH, no corner can be cut). Every 20 ms, rays from the car's nose measure wall nearness in each visual
    hemifield -> Poisson drive of LPLC1 (looming/collision visual neurons) of that side; P9 is held at 100 Hz for
    forward drive; the whole brain runs as one continuous trial and its DNa01/DNa02 output steers the car
    (hand-designed eye model + car link; the turn-away-from-the-wall reflex itself comes from the connectome).
    Courses: "canyon" (128 m, bends + chicane), "hairpin" (180 deg), "slalom" (serpentine).
    silence = lesion (e.g. ["LPLC1"] -> blind to walls -> expect a crash). ~2-8 min per run unless cached.
    Returns outcome (finished / crash / timeout), time, % of course, LPLC1 and DNa rates."""
    from flylab import arena
    return await _run(arena.run_course, course, silence or [], max_s)


@server.tool()
async def swat_fly(speed: str = "normal", silence: list[str] | None = None) -> str:
    """FLY-SWATTER escape game (body video, 10x slow motion). A swatter comes down on a walking fly (P9 on).
    Its looming image drives LPLC2 + LC4 (designed eye model, flylab/looming.py) in a closed-loop whole-brain trial
    (5 ms chunks). If the giant fiber (DNp01) exceeds 50 Hz in time, the fly takes off (SCRIPTED animation, no
    aerodynamics) and the swatter hits the floor; otherwise SPLAT. speed = "slow" | "normal" | "fast" | "lightning".
    Controls: silence=["giant_fiber"] -> splat; silence=["LPLC2"] -> escape relies on LC4 and comes later.
    Returns outcome, giant-fiber time, swatter distance/angle at that moment, margin in ms. ~2-4 min unless cached."""
    from flylab import swatter
    return await _run(swatter.swat_fly, speed, silence or [])


@server.tool()
async def ride_bike(preset: str | None = "loop", maneuvers: list[dict] | None = None,
                    silence: list[str] | None = None) -> str:
    """The fly rides a BICYCLE through a ball-rig read-out: manoeuvre stimuli (same as drive_car: forward = P9,
    left/right = P9 + DNa01/DNa02 of that side, reverse = MDN) -> whole brain -> interface -> the physics-simulated
    legs walk -> their forward speed and yaw (what a ball under a tethered fly would measure) drive the bike
    (designed gear/yaw link). The bike reads the LEGS, not the neurons. preset = "loop" | "slalom" | "sprint" |
    "reverse", or maneuvers=[{"action": "left", "seconds": 1.0}, ...] (max 5 s). Lesion demo:
    silence=["DNa01_left","DNa02_left"] with preset="loop". ~4-8 min unless cached (leg physics)."""
    from flylab import bike
    return await _run(bike.ride_bike, preset, maneuvers, silence or [])


@server.tool()
async def play_piano(mode: str = "keyboard", players: list[str] | None = None, silence: list[str] | None = None,
                     song: str | None = None, p9_hz: float = 100.0, bpm: float = 100.0,
                     bitter_notes: list[int] | None = None) -> str:
    """FLY PIANO BAND. No circuit makes music; two honest sonifications:
    mode="keyboard": a gang of up to 3 flies (walking behaviours, default walk_forward, turn_left, walk_backward)
    walks across a giant pentatonic keyboard; every front-foot touchdown in the physics sim plays the key under it
    (rhythm = real stepping, 8x slowed). Video of fly 1 with sound + all notes. ~6-10 min unless cached.
    mode="orchestra": three musician flies, brain only, ~2 min: sugar pulses -> MN9 = bass (bitter in bars 5-6
    mutes it), P9 + left/right DNa stimulation -> turn signal = melody, looming LPLC2/LC4 -> giant fiber = drums.
    The score is designed; the brain decides which notes sound. Lesions are audible (silence=["giant_fiber"]).
    A SPECIFIC SONG (song = "ode_to_joy" | "twinkle" | "frere_jacques" or note text "E4 E4 F4 G4:2 R:1 C#5:0.5"):
    mode="song_road": a musical road of the song's keys laid along the path a P9-driven fly really walks; each key
    sounds when a foot first lands on it (rhythm = the fly's stepping; p9_hz lower = slower tempo; silence=["P9"] =
    no song). ~10-20 min unless cached. mode="song_orchestra": one sugar pulse per note to a singer fly; a note sounds
    if MN9 > 20 Hz (bitter_notes = note indices that also get bitter -> silent). ~1-3 min. The melody is always OUR
    score; for copyrighted songs ask the user for the notes."""
    from flylab import piano
    return await _run(piano.play_piano, mode, players, silence or [], song, p9_hz, bpm, bitter_notes)


@server.tool()
async def run_olympics(course: str = "canyon", contestants: list[dict] | None = None) -> str:
    """CONNECTOME OLYMPICS: the same brain with different lesions races the same closed-loop course (run_course).
    contestants = [{"name": "Intact", "silence": []}, {"name": "No LPLC1", "silence": ["LPLC1"]}, ...] (max 6);
    default: intact, left LPLC1 off, left DNa01+DNa02 off, all LPLC1 off. Returns a leaderboard (finish time or
    % of course before crashing) and shows all cars racing on the dashboard. Results are model predictions.
    Each new contestant costs a run_course (~2-8 min); cached ones are instant."""
    from flylab import olympics
    return await _run(olympics.run_olympics, course, contestants)


@server.tool()
async def fly_through(place: str = "cave", bat: bool = True, silence: list[str] | None = None) -> str:
    """Themed closed-loop FLY-THROUGH: place = "cave" (stalactite pillars) | "forest" (tree trunks). Same engine as
    run_course (eyes -> LPLC1 L/R -> whole brain -> DNa steering, P9 on, solid walls and obstacles). Flight is NOT
    simulated: the fly avatar glides at a fixed height. bat=True: at 3 s a bat swoops in head-on; its looming drives
    LPLC2/LC4; giant fiber > 50 Hz -> DODGE (scripted), else CAUGHT. Controls: silence=["giant_fiber"] (caught),
    silence=["LPLC1"] (crashes into the rock). ~2-8 min unless cached."""
    from flylab import olympics
    return await _run(olympics.fly_through, place, bat, silence or [])


@server.tool()
async def run_maze(silence: list[str] | None = None) -> str:
    """SUGAR MAZE (closed loop): a hedge corridor with 3 sugar pellets and 2 poisoned pellets (sugar + bitter).
    Steering as in run_course. Touching a pellet stimulates sugar GRNs (+ bitter GRNs if poisoned) at 200 Hz;
    if MN9 exceeds 30 Hz the fly stops and eats (0.6 s eats the pellet). Expected: eats sugar, skips poison because
    bitter suppresses MN9. Note: bitter alone does NOT drive backing up (MDN 0 Hz in this model).
    Controls: silence=["bitter_GRNs"] -> eats the poison too; silence=["CB0553"] -> eats nothing.
    Returns pellets eaten, MN9 peak, outcome. ~3-8 min unless cached."""
    from flylab import olympics
    return await _run(olympics.run_maze, silence or [])


@server.tool()
async def get_literature_facts(topic: str = "") -> str:
    """Curated, cited facts from the literature (Shiu et al. 2024, FlyWire papers, descending-neuron papers)
    stored in data/literature.md. Use to check claims; anything not here is unverified."""
    return await _run(lab.get_literature_facts, topic)


@server.tool()
async def save_report(title: str, markdown: str, run_ids: list[str] | None = None) -> str:
    """Save a lab report (markdown) to runs/reports/ and attach it to the given runs. Shown on the dashboard."""
    from flylab import reports
    return await _run(reports.save_report, title, markdown, run_ids or [])


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
