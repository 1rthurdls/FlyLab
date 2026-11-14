# FlyLab — Claude as an autonomous fly neuroscientist

**Claude operates a real, connectome-based simulation of the entire fruit-fly brain (FlyWire, 138,639 neurons,
15.1 M connections) and drives a physics-simulated 3D fly body with it.**

You type a task in plain English into Claude Code, e.g.
*"Make the fly eat when it finds sugar, then prove which neuron is responsible."*
Claude, using the FlyLab MCP server, a skill, subagents and a hook:

1. searches the real connectome for the relevant neurons (sugar gustatory receptor neurons, the MN9 proboscis motor neuron),
2. traces the strongest synaptic pathways from sensory to motor neurons,
3. runs the whole-brain spiking simulation (activate / silence neurons),
4. maps the brain's output neurons onto a NeuroMechFly body and renders a video of the behaviour,
5. runs a lesion screen, silences the key neuron, shows the behaviour disappear, and
6. writes a short lab report that a `literature-checker` subagent checks against cited facts.

## What is real and what is designed (honest framing)

| Part | Status |
|---|---|
| Wiring (138,639 neurons, 15,091,983 connected pairs, synapse counts and signs) | **Real data**: FlyWire v783 connectome (Dorkenwald et al. 2024; Schlegel et al. 2024) |
| Neuron dynamics | **Published model**: leaky integrate-and-fire model of Shiu et al. 2024, identical parameters. FlyLab's fast kernel is validated **spike-for-spike** against the Eon Systems PyTorch implementation (`scripts/validate_against_pytorch.py`) |
| Learning | **None.** The weights are fixed by the connectome. Claude finds and drives circuits that already exist; the fly does not learn. |
| Brain → body mapping | **Hand-designed interface** (`flylab/interface.py`): P9 rate → forward walking, MDN → backward, left−right DNa01/DNa02 → turning, MN9 > 30 Hz → feeding (walking suppressed). The ventral nerve cord is not simulated. |
| Body | NeuroMechFly v2 (flygym 1.2.1, MuJoCo physics) with flygym's own `HybridTurningController`. The proboscis is not articulated in the stock model: FlyLab adds hinge joints + actuators as a **visual stand-in** for proboscis extension |
| Sugar contact (hero scenario) | Coarse closed loop: the body detects contact with the sugar drop, and that switches on sugar GRN stimulation in the brain |
| Lesion results (e.g. CB0553) | **Model predictions**, flagged as such; not literature facts |

## Results produced by the model (1 s simulated, 8 trials)

| Condition | MN9 (Hz) |
|---|---|
| no stimulus | 0 (the model has no spontaneous activity) |
| sugar GRNs, 200 Hz | **78** (440 neurons active) |
| sugar + bitter GRNs | **11** (bitter suppresses feeding output, −86%) |
| sugar GRNs, CB0553 silenced (2 neurons) | **18.5** (−76%; top hit of an automatic lesion screen) |

Details, timings and every decision: [PROGRESS.md](PROGRESS.md).

Media: [hero, intact brain](docs/hero_find_sugar_intact.mp4) · [hero, CB0553 silenced](docs/hero_find_sugar_CB0553_lesion.mp4) ·
[dashboard](docs/dashboard.png) · [architecture slide](docs/architecture.png) · rasters: [sugar](docs/raster_sugar.png),
[sugar + CB0553 lesion](docs/raster_sugar_CB0553_lesion.png).

## Extra activities (restricted, honest versions of what others show online)

| Activity | What is real | What is designed |
|---|---|---|
| **Escape take-off** (`run_behavior("takeoff")`) | Looming detectors (LPLC2, 50 Hz) drive the giant fiber (DNp01) to ~100 Hz in the whole-brain model; the fly takes off ~20 ms after looming onset. Silencing the giant fiber abolishes take-off. | The flight itself is a single **scripted animation** (NeuroMechFly cannot fly; no aerodynamics). |
| **The fly drives a car** (`drive_car` tool) | Each manoeuvre stimulates real descending neurons (P9 forward, DNa01/DNa02 steering, MDN reverse); the car follows what they actually fire, incl. P9's slight pull to the right. Silencing the left DNa neurons -> the car cannot turn left. | Car link (throttle/steering/reverse) and manoeuvre choice (Claude or a planner that re-plans on the real brain output). Nothing learns. Online "fly drives a car" claims had no primary source. |
| **Fly swatter** (`swat_fly`) | A swatter comes down on a walking fly; its looming image drives LPLC2 + LC4 in a closed-loop whole-brain trial (5 ms chunks). Intact: the giant fiber crosses 50 Hz while the swatter is still 134 mm away and the fly is off the ground **255 ms before impact**. Giant fiber silenced: **splat**. LPLC2 silenced: escape relies on LC4 and comes 120 ms later. [escape](docs/swatter_intact_escape.mp4) · [splat](docs/swatter_giant_fiber_silenced_splat.mp4) | Looming eye model (`flylab/looming.py`), 50 Hz threshold + 10 ms take-off latency, swatter size/speed, the **scripted** take-off. Real flies also have a slower GF-independent escape (not modelled). |
| **Bicycle** (`ride_bike`) | The bike reads the **legs**, not the neurons: manoeuvre stimuli -> whole brain -> interface -> physics-simulated legs -> their forward speed and yaw (what a ball under a tethered fly measures) -> bike. "Left" turned the legs +190 deg/s and the bike +186 deg; with left DNa01/DNa02 silenced the same manoeuvre turned it **-47 deg** (right). [video](docs/bike_loop_legs.mp4) | Gear (0.45 m/s per mm/s), yaw gain, the bike. |
| **Fly piano band** (`play_piano`) | *keyboard*: three flies walk across a giant pentatonic keyboard; every front-foot touchdown (MuJoCo contact force) plays the key under it: 93 notes in 1.5 s of walking. *orchestra*: three brain-only musician flies; sugar -> MN9 = bass (bitter in bars 5-6 mutes it: 17 -> 3 Hz), P9 + DNa steering = melody, looming -> giant-fiber bursts = snare/hi-hat. [keyboard](docs/piano_keyboard_with_sound.mp4) · [orchestra](docs/brain_orchestra.m4a) | Sonification: keyboard layout, scale, chords, score and note mapping are ours. The flies are not making music. |
| **A specific song** (`play_piano(mode="song_road" \| "song_orchestra", song=...)`) | *Musical road*: the song's keys are laid just ahead of a P9-driven fly as it walks; each key sounds when a foot first lands on it. Ode to Joy: **28/30 notes**, ~73 BPM set by its own stepping (8x slowed). *Singer fly*: one sugar pulse per note, a note sounds if MN9 > 20 Hz during the pulse: 30/30 notes; bitter on phrase 2 -> those 15 notes go silent (MN9 8 Hz). With a bass line, a second (bass) fly plays it; the dashboard shows the flies at a piano, tapping each key with the proboscis when MN9 fires. [road](docs/song_road_ode_to_joy.mp4) · [singer](docs/singer_ode_to_joy.m4a) · [singer + bitter](docs/singer_ode_to_joy_bitter_phrase2.m4a) | The melody is always our score (built-in public-domain tunes or note text like `E4 E4 F4 G4:2 R:1`). |
| **Sugar maze** (`run_maze`) | Closed loop like `run_course`; touching a pellet drives sugar GRNs (+ bitter GRNs if poisoned); MN9 > 30 Hz = stop and eat. Intact: ate **3/3 sugar, 0/2 poisoned**, finished 136 m. Bitter GRNs silenced: ate 2/2 poisoned too. Note: bitter alone does **not** make the model back up (MDN 0 Hz). | Pellet contact radius, eating time, corridor. |
| **Connectome Olympics** (`run_olympics`) | The same brain with different lesions races the closed-loop canyon: #1 intact (finished, 20.3 s), #2 all LPLC1 silenced (crash at 13%), #3 left DNa01+DNa02 silenced (crash at 8%), #4 only the LEFT eye's LPLC1 silenced (crash at 5%: half-blind is worse than blind, because the remaining eye's wall signal biases steering one way). | A lesion experiment presented as a race; results are model predictions. |
| **Fly-throughs with a bat** (`fly_through`) | Same closed loop with rock pillars (cave) or trees (forest): cave 108 m in 20.7 s, forest 127 m in 24.4 s, nothing touched. A bat swoops in head-on; its looming drives LPLC2/LC4; the giant fiber fired 75 Hz with the bat 6-8 m away -> dodge. Wall input alone also gives occasional giant-fiber bursts (4-7 per run); they are reported, not credited to the bat. | **Flight is not simulated** (fixed-height glide); dodge is scripted; looming eye model, bat size/speed. Pillars had to be placed on straights: in bends they cancel the two-eye wall cue. |
| **Obstacle course, closed loop** (`run_course` tool) | **No planner.** The brain runs as one continuous trial, exchanging data with the world every 20 ms (`BrainStepper`, spike-identical to the one-shot kernel). Wall nearness in each eye drives that side's LPLC1; in the connectome LPLC1 excites the *opposite* DNa01, so the car turns away from the nearer wall. Canyon (128 m of solid walls): **finished in 20.3 s, no wall touched**. Same brain with LPLC1 silenced: **crash at 3.4 s, 17 m in**. [figure](docs/course_canyon.png) | Eye model (rays -> LPLC1 rate), constant P9 drive, car link, course geometry. Literature: LPLC1 is needed for collision avoidance (slowing; Tanaka & Clark 2022); turning away is a model prediction. |

## Ask from the dashboard

The left column has a chat box: each message runs **Claude Code headless** (the same program and login as the
VS Code extension) in this folder, with the flylab MCP server, CLAUDE.md, the skill, subagents and the hook.
The conversation continues across messages. Only flylab tools, subagents, the skill and read-only file tools are
allowed (no shell, no file edits). Requires `.claude/settings.json` (copy of `settings.flylab.json`).

## Architecture

```mermaid
flowchart LR
  U[User prompt in Claude Code] --> C[Claude + CLAUDE.md]
  C -- skill --> S[fly-neuroscientist SKILL.md]
  C -- subagents --> L[literature-checker] & E[experiment-designer]
  C -- MCP stdio --> M[flylab MCP server]
  M --> CN[connectome.py: search, pathways]
  M --> B[brain.py: whole-brain LIF, numba]
  M --> BH[behaviors.py -> interface.py -> body.py flygym]
  B & BH --> R[(runs/: result.json, raster.png, video.mp4)]
  C -- PostToolUse hook --> EV[(runs/events.jsonl)]
  EV & R --> D[dashboard :8000]
```

## Quick start

```bash
bash scripts/setup.sh                      # venv, deps, data, caches (Windows: run in Git Bash)
.venv/Scripts/python.exe scripts/smoke_test.py      # (macOS/Linux: .venv/bin/python)
cp .claude/settings.flylab.json .claude/settings.json   # enable the hook that feeds the dashboard
.venv/Scripts/python.exe dashboard/server.py        # http://localhost:8000
claude                                       # in this folder; the flylab MCP server loads from .mcp.json
```

Then ask: *"Make the fly eat when it finds sugar, then prove which neuron is responsible."*
Or try the extras: *"Swat the fly, then cut the giant fiber"*, *"Let the fly ride a bike"*, *"Make the flies play
Twinkle Twinkle"*, *"Run the sugar maze with poisoned pellets"*, *"Run the Connectome Olympics"*.

Show any precomputed extra on the dashboard without Claude (instant when cached):
```bash
.venv/Scripts/python.exe scripts/demo_extras.py              # list: swatter, bike, piano, orchestra, song_road, singer, maze, cave, forest, olympics ...
.venv/Scripts/python.exe scripts/demo_extras.py song_road
.venv/Scripts/python.exe -u scripts/run_extras_batch.py > runs/extras_batch.log   # (re)compute all extras, ~2-4 h on a laptop CPU
```
For recording the video, see [DEMO.md](DEMO.md) (precompute everything first with `scripts/precompute_demo.py`).

## Claude Code components

| Component | File | What it does |
|---|---|---|
| MCP server | `flylab/mcp_server.py`, `.mcp.json` | 11 tools: `search_neurons`, `list_neuron_groups`, `get_neuron_group`, `trace_pathway`, `run_simulation`, `compare_conditions`, `lesion_screen`, `list_behaviors`, `run_behavior`, `get_literature_facts`, `save_report` |
| CLAUDE.md | `CLAUDE.md` | Framing (fixed wiring, hand-designed interface) and house rules: trace before simulating, always a control, always a body video, literature check, 5-line report |
| Skill | `.claude/skills/fly-neuroscientist/SKILL.md` | The 8-step experimental method with worked examples (eat sugar, avoid bitter, walk backwards, clean antennae) |
| Subagents | `.claude/agents/literature-checker.md`, `experiment-designer.md` | Read-only fact checker of lab reports against `data/literature.md`; protocol planner |
| Hook | `.claude/hooks/log_tool_event.py` (config in `.claude/settings.flylab.json`) | Pre/PostToolUse on flylab tools, subagents and skills → `runs/events.jsonl` → dashboard live feed |

## Repository layout

```
flylab/        brain.py, _lif_numba.py (model) · connectome.py, groups.py (data) · lab.py (experiments)
               interface.py, body.py, behaviors.py (body) · mcp_server.py · plots.py, cache.py, reports.py
               extras: car.py, arena.py (closed-loop courses, maze, fly-throughs, bat) · looming.py · swatter.py
               bike.py · piano.py (keyboard, orchestra) · song.py (musical road, singer) · olympics.py
dashboard/     server.py (FastAPI) + index.html (vanilla JS) + car3d.js (3D scenes) + piano.js (music views)
               + band3d.js (fly band at a piano: proboscis taps driven by each fly's MN9; the tap plays the note)
scripts/       setup.sh, smoke_test.py, precompute_demo.py, test_mcp.py, test_body.py, validate_against_pytorch.py
               demo_extras.py (show a cached extra on the dashboard), run_extras_batch.py (compute all extras)
data/          neuron_groups.json (resolved from data, with sources), literature.md (cited fact sheet);
               connectome + annotation files are downloaded by setup.sh
runs/          one folder per run: result.json, raster.png, video.mp4, commands.json, report.md
```

## Performance note

The reference PyTorch backend multiplies a 138,639 × 138,639 sparse matrix every 0.1 ms; on a laptop CPU that is
~12 minutes per simulated second. FlyLab uses the same equations in an event-driven numba kernel that only updates
neurons that are away from rest (~11,000 of 138,639 during sugar stimulation), giving **1 s of whole-brain activity
in ~2 s per trial** on an i7 laptop, with identical spikes (see PROGRESS.md for the validation).

## Limitations

- The brain→body interface is hand-designed; thresholds come from measured model rates (PROGRESS.md).
- One-sided sugar stimulus set (as in the Shiu et al. example); 9 of its 21 neurons are now annotated in FlyWire as
  LB3d/LB4b rather than sugar types. An annotation-based alternative group (`sugar_GRNs_flywire_all`) is provided.
- Grooming is brain-only: flygym has no grooming controller.
- Water GRNs (LB3a, one side) do not drive MN9 in our runs; we do not use them in demos.
- Simulations average 8 trials (the paper used 30).

## Credits and licenses

- **FlyWire connectome**: Dorkenwald et al., *Nature* 2024; Schlegel et al., *Nature* 2024; annotations from
  github.com/flyconnectome/flywire_annotations. Neurotransmitter predictions: Eckstein et al., *Cell* 2024.
- **Whole-brain model**: Shiu et al., *Nature* 2024, github.com/philshiu/Drosophila_brain_model (MIT license).
- **PyTorch backend**: Eon Systems, github.com/eonsystemspbc/fly-brain (GPL-2.0). FlyLab does not copy or
  redistribute its code: `setup.sh` clones it into `external/` (git-ignored) and `scripts/validate_against_pytorch.py`
  imports it only to validate FlyLab's own implementation of the published equations.
- **Body**: NeuroMechFly v2 / flygym, Wang-Chen et al., *Nature Methods* 2024 (Apache-2.0).
- Descending-neuron literature: Bidaye et al. 2014 (MDN), Bidaye et al. 2020 (P9), Rayshubskiy et al. 2020 (DNa02),
  Hampel et al. 2015 (grooming). Full list in `data/literature.md`.
