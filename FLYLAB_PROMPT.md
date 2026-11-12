# FlyLab — overnight build brief for Claude Code

You are building a hackathon project autonomously overnight. I will NOT be available to answer questions until morning. Never stop to ask me anything: make the most reasonable decision, write it down in `PROGRESS.md`, and keep going. Your job is to have a working, demo-ready project when I wake up.

## 1. The project in one sentence

**FlyLab: Claude as an autonomous neuroscientist operating a real, connectome-based simulation of the entire fruit fly brain (FlyWire, ~138k neurons), driving a physics-simulated 3D fly body.**

The user types a task in plain English into Claude Code (e.g. "make the fly eat when it finds sugar"). Claude, using a FlyLab MCP server plus a skill and subagents, then:
1. searches the real connectome for the relevant neurons,
2. traces the pathway from sensory neurons to motor/descending neurons,
3. runs the whole-brain spiking simulation (activate / silence neurons),
4. maps the brain's output neurons to a 3D fly body and renders a video of the behaviour,
5. runs a follow-up "lesion" experiment (silence a key neuron → behaviour disappears) and writes a short lab report checked against the published literature.

The final deliverable is a **recorded video demo** (2–3 min), so nothing has to run in real time. Slow simulations are fine, as long as results are cached and the render looks good.

## 2. Hackathon judging context (optimise for this)

- Judged on project quality + "wow". Projects that **combine a real dataset with Claude skills, agents/subagents, MCP** score highest.
- So the project must visibly use: **an MCP server, a SKILL.md, at least one subagent, a CLAUDE.md, and a hook.** Each must do real work, not be decorative.
- Honesty matters: the connectome wiring is fixed and does not learn. Never claim the fly "learned" a task. The framing is: *Claude finds and drives the circuits that already exist for a behaviour, and wires the brain's outputs to a body.* The brain→body mapping is a documented, hand-designed interface; say so in the README.

## 3. Tech stack (use these unless they are truly broken)

- Python 3.10 or 3.11, a `venv` in the repo (`.venv`). Pin everything in `requirements.txt`.
- **Brain model:** the leaky integrate-and-fire whole-brain model from Shiu et al., *Nature* 2024 ("A Drosophila computational brain model reveals sensorimotor processing").
  - Primary source: `https://github.com/eonsystemspbc/fly-brain` (has a PyTorch backend: `code/run_pytorch.py`, plus `model.py`, `utils.py`, `example.ipynb`). Use its **PyTorch** implementation; do NOT go down the Brian2CUDA / NEST GPU / GeNN install path.
  - Fallback source: `https://github.com/philshiu/Drosophila_brain_model` (original Brian2 code + data files).
  - Data: FlyWire v783 `Completeness_783.csv` and `Connectivity_783.parquet` (shipped with those repos, or on Zenodo). Do not invent data.
  - Neuron identities (cell types like sugar GRNs, MN9, P9, DNa01/DNa02, MDN, antennal grooming neurons): get them from the repos' example notebooks and from the FlyWire annotation file at `https://github.com/flyconnectome/flywire_annotations` (`supplemental_files/Supplemental_file1_neuron_annotations.tsv`). **Never hard-code neuron IDs from memory.** Look them up in data and save the resolved lists to `data/neuron_groups.json` with their source.
  - Use GPU (CUDA or Apple MPS) if available, otherwise CPU. Benchmark a 0.5 s and 1 s simulation early and record timings in `PROGRESS.md`.
- **Body:** `flygym` (NeuroMechFly v2, MuJoCo). `pip install flygym`. Check the installed version's API (read its source or docs) before writing code. Its APIs changed between versions. Use its existing walking/turning controllers (e.g. a hybrid turning controller / CPG) rather than writing locomotion from scratch. For headless rendering set `MUJOCO_GL=egl` (or `osmesa`) and save `.mp4`.
- **MCP server:** official Python MCP SDK (`pip install mcp`, `FastMCP`), stdio transport, registered in `.mcp.json` at repo root so Claude Code picks it up automatically.
- **Dashboard:** FastAPI (or plain `http.server`) serving one static HTML page. No React build step. Vanilla JS, optionally three.js from a CDN.
- Use `git`. Commit after every milestone with a clear message.

## 4. Repository layout to create

```
flylab/
  CLAUDE.md                     # project context + demo rules for Claude
  README.md                     # what it is, how to run, honest limitations, credits
  PROGRESS.md                   # your running log: decisions, timings, problems, status
  DEMO.md                       # exact step-by-step script for recording the video
  requirements.txt
  .mcp.json                     # registers the flylab MCP server
  .claude/
    settings.json               # hooks config
    skills/fly-neuroscientist/SKILL.md
    agents/literature-checker.md
    agents/experiment-designer.md   (optional)
    hooks/log_tool_event.py     # PostToolUse hook → appends to runs/events.jsonl
  flylab/
    brain.py                    # load connectome, LIF sim (PyTorch), activate/silence, rates
    connectome.py               # neuron search, annotations, pathway tracing (graph)
    groups.py                   # named neuron groups (resolved from data, cached JSON)
    body.py                     # flygym body, brain→body interface, render mp4
    interface.py                # documented mapping: output-neuron rates → motor commands
    behaviors.py                # catalogue of tested behaviours (feeding, walking, turning, backing up, grooming)
    cache.py                    # hash(params) → cached results on disk
    mcp_server.py               # FastMCP server exposing the tools below
  dashboard/
    server.py
    index.html                  # split view: 3D fly video | brain activity | live tool-call feed
  scripts/
    setup.sh                    # one-command setup
    smoke_test.py               # end-to-end check, must pass
    precompute_demo.py          # pre-runs and caches every demo scenario
  data/                         # connectome files + neuron_groups.json
  runs/                         # per-run outputs: result.json, raster.png, video.mp4
```

## 5. MCP tools to expose (FastMCP)

Every tool returns compact, human-readable JSON (top-N lists, not 138k numbers) and writes full outputs to `runs/<run_id>/`.

- `search_neurons(query: str, limit=20)` → match cell type / class / super-class / hemilineage / neurotransmitter in annotations (e.g. "sugar", "MN9", "descending", "bitter").
- `get_neuron_group(name)` / `list_neuron_groups()` → curated named groups (sugar_GRNs, bitter_GRNs, water_GRNs, MN9_proboscis, P9_forward, DNa01_DNa02_turning, MDN_backward, antennal_grooming, …) with counts and source.
- `trace_pathway(source_group, target_group, max_hops=4)` → strongest paths through the connectome (weighted by synapse counts), hop count, intermediate cell types.
- `run_simulation(activate: list[str], silence: list[str] = [], stim_rate_hz=200, duration_s=0.5)` → top responding neurons, firing rates of every curated output group, and a raster/heatmap PNG path. Cached by params.
- `run_behavior(behavior: str, activate=[], silence=[], duration_s=…)` → runs the brain sim, converts output-group rates to body commands via `interface.py`, renders the fly in flygym, returns the video path plus a summary ("MN9 at 82 Hz → proboscis extension; P9 at 3 Hz → no walking").
- `compare_conditions(run_id_a, run_id_b)` → which output groups changed and by how much (used for lesion experiments).
- `get_literature_facts(topic)` → returns a small curated, cited fact sheet (from the Shiu et al. 2024 paper abstract/results and FlyWire papers) stored in `data/literature.md`. Write that file yourself from what the papers say; mark every fact with its source. Keep it short and accurate. If unsure, leave it out.

## 6. Claude Code components (must do real work)

- **CLAUDE.md:** what FlyLab is, the honest framing (fixed wiring, no learning), how to use the MCP tools, house rules for the demo (always trace a pathway before simulating; always run a lesion control; always end with a 5-line lab report with numbers).
- **Skill `fly-neuroscientist/SKILL.md`:** the experimental method Claude follows when the user asks for a behaviour: (1) translate the request into sensory input + expected motor output, (2) find groups, (3) trace pathway, (4) simulate, (5) run behaviour + render, (6) lesion/control experiment, (7) send results to `literature-checker`, (8) write the lab report into `runs/<id>/report.md`. Include worked examples for "eat sugar", "avoid bitter", "walk backwards", "clean antennae".
- **Subagent `literature-checker`:** read-only. Given a lab report, it checks each claim against `data/literature.md` and flags anything unsupported. Its verdict is appended to the report.
- **Hook:** a `PostToolUse` hook (matcher: the flylab MCP tools) runs `.claude/hooks/log_tool_event.py`, which appends `{time, tool, args, short_result}` to `runs/events.jsonl`. The dashboard tails this file to show Claude's actions live next to the fly. This is the visible "agent at work" feed in the video.

## 7. Behaviours: realistic targets

Pick behaviours that the brain model is known to support AND that the body can show. Verify each one; drop any that don't work.
1. **Feeding / proboscis extension on sugar:** sugar GRNs → MN9 is the headline result of Shiu et al. Check whether flygym's body has an actuated proboscis/head joints. If not, show it with a clear visual stand-in (e.g. stop walking + head dips toward the sugar drop) and say so honestly in README.
2. **Bitter suppresses feeding:** activate sugar + bitter → MN9 activity drops. Good "control" result.
3. **Walking forward (P9) / turning (DNa01/DNa02) / backing up (MDN):** activate these descending neurons and drive the flygym walking controller from their rates. These are the most reliable for a visible 3D body.
4. **Antennal grooming** from mechanosensory (JO) activation: second headline result of Shiu et al. Show in body only if flygym supports it; otherwise show the brain activity side only.
5. **Hero sequence for the video:** "fly walks toward a sugar drop, stops, feeds; Claude silences the key neuron and the fly no longer feeds."

`interface.py` must document the mapping clearly, e.g. forward speed ∝ P9 rate, turn ∝ (left − right DNa rates), backward ∝ MDN rate, feeding state when MN9 rate > threshold. Thresholds come from your measured baseline vs stimulated rates (record them).

## 8. Milestones (do them in order, commit after each, timebox and fall back)

**M0 — Setup (≤45 min).** Repo skeleton, venv, `requirements.txt`, `scripts/setup.sh`, clone/copy the brain-model code and data, git init. Log machine info (CPU/GPU/RAM) in `PROGRESS.md`.

**M1 — Brain works (≤2.5 h). Most important gate.** Load connectome, run the PyTorch LIF model, reproduce the sugar GRN → MN9 activation result qualitatively (MN9 clearly fires when sugar GRNs are stimulated, not at baseline). Benchmark timings. Save `runs/m1_sugar/` with result + raster. If PyTorch backend is broken, port the minimal LIF equations yourself from the repo's model (same parameters as the paper), and validate against the Brian2 version on a short run.

**M2 — Connectome tools (≤1.5 h).** Annotations loading, `search_neurons`, curated groups saved to `data/neuron_groups.json`, `trace_pathway` with networkx (or scipy sparse BFS on the strongest edges). Unit tests.

**M3 — MCP server (≤1.5 h).** All tools from section 5 (body tool can be a stub until M4). `.mcp.json`. Verify by launching the server and calling every tool from a Python MCP client script (`scripts/test_mcp.py`). Also verify `claude mcp list` sees it if the CLI is available.

**M4 — Body (≤3 h).** flygym installed, headless rendering works, walking controller driven by descending-neuron rates, render `.mp4` for forward / turn / backward / feeding stand-in. **Fallback if flygym can't render or install after ~90 min of real effort:** build a three.js 3D stylized fly in `dashboard/` (body, 6 legs with a simple tripod gait animation, proboscis, antennae) that is animated from a per-frame JSON of motor commands exported by `interface.py`. Record that choice in `PROGRESS.md`. A working stylized 3D fly beats a broken realistic one.

**M5 — Claude layer (≤1.5 h).** CLAUDE.md, SKILL.md, subagent(s), hook, `data/literature.md`.

**M6 — Dashboard (≤2 h).** One page, designed to look great in a screen recording (dark background, large type, 16:9): left = live tool-call feed from `runs/events.jsonl`; centre/right = latest fly video (auto-refreshes on new run); bottom = brain activity heatmap/raster of the latest run with the curated groups highlighted and labelled; small header with "FlyWire connectome · 138,639 neurons · whole-brain LIF". Auto-picks up new runs (poll every 1 s).

**M7 — Demo hardening (≤1.5 h).** `scripts/precompute_demo.py` runs and caches every scenario in the demo script so the recorded run is fast. `scripts/smoke_test.py` runs the full chain end to end. Write `DEMO.md`: exact commands, exact prompts to type into Claude Code, expected outputs, and a 2–3 min shot list (hook → ask → Claude working → fly behaves → lesion surprise → how it works → close).

**M8 — Polish if time remains.** Better visuals, second hero scenario ("avoid bitter"), README diagram, a 1-slide architecture image (`docs/architecture.png`).

## 9. Working rules for tonight

- **Never block on me.** Decide, log the decision and reason in `PROGRESS.md`, continue.
- **Keep `PROGRESS.md` current.** At the top: a "Morning summary" section with status of each milestone (done / partial / failed + why), what works right now, the exact commands to see it, open problems, and what you'd do next. Update it after every milestone, not just at the end.
- **Verify everything by running it.** No milestone is done until its script runs green. Never fake outputs, mock simulation results, or hand-write numbers that should come from the model.
- **Timebox.** If something burns past its budget, take the documented fallback and move on. Prefer a smaller thing that works end to end over a bigger thing that is half-built.
- **Long jobs:** run simulations in the background with logs, keep durations short (0.25–1 s of simulated time) and cache results.
- **Don't break working things.** Commit before risky changes. Keep `smoke_test.py` passing.
- **Credit sources** in README: FlyWire (Dorkenwald et al. 2024, Schlegel et al. 2024), Shiu et al. 2024, NeuroMechFly v2 (Wang-Chen et al. 2024), Eon Systems fly-brain repo. Respect their licenses.
- Stay inside this repo folder. Don't touch anything else on the machine.

## 10. Definition of done (what I want to see in the morning)

1. `bash scripts/setup.sh && python scripts/smoke_test.py` passes.
2. Opening Claude Code in the repo shows the flylab MCP server; typing *"Make the fly eat when it finds sugar, then prove which neuron is responsible"* makes Claude run the full loop on its own (search → trace → simulate → render → lesion → literature check → report).
3. The dashboard at `http://localhost:8000` shows Claude's tool calls live, the fly's 3D video, and brain activity, and looks good enough to screen-record.
4. `DEMO.md` lets me record the video in under an hour.
5. `PROGRESS.md` morning summary tells me honestly what works and what doesn't.

Start now with M0. Go.
