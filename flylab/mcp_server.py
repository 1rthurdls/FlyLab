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
