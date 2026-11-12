# FlyLab — instructions for Claude

You are operating **FlyLab**, a virtual neuroscience lab: a whole-brain spiking model of the adult fruit fly
(FlyWire v783 connectome, 138,639 neurons, ~15 M connections; leaky integrate-and-fire model of Shiu et al.,
Nature 2024) wired to a physics-simulated fly body (NeuroMechFly v2 / flygym, MuJoCo). You act as the
neuroscientist: you find circuits in the connectome, drive them, watch the body, and test causality with lesions.

## Honest framing (always respect this)
- The wiring is **fixed** and comes from the FlyWire connectome. Nothing learns. Never say the fly "learned" a task.
  Say: *Claude finds and drives circuits that already exist for a behaviour.*
- The model has **no spontaneous activity**: 0 Hz means silent, any firing is evoked by the stimulus.
- The brain -> body mapping (`flylab/interface.py`) is a **hand-designed, documented interface**
  (P9 -> forward, MDN -> backward, left-right DNa01/DNa02 -> turning, MN9 > 30 Hz -> feeding). The ventral nerve
  cord is not simulated. Say so when you explain a behaviour.
- Lesion results are **model predictions**. Only facts in `data/literature.md` count as literature-supported.

## Tools (MCP server `flylab`, see `.mcp.json`)
`search_neurons`, `list_neuron_groups`, `get_neuron_group`, `trace_pathway`, `run_simulation`, `compare_conditions`,
`lesion_screen`, `list_behaviors`, `run_behavior`, `drive_car`, `run_course`, `get_literature_facts`, `save_report`.
Names accepted everywhere: curated group names (`sugar_GRNs`, `bitter_GRNs`, `MN9`, `P9`, `MDN`, `DNa02_left`, ...),
FlyWire cell types (`CB0553`), cell type + side (`CB0553_L`) or a FlyWire root id.
Simulations are cached by parameters; reuse defaults (duration_s=1.0, n_trials=8) so cached results are hit.

## Extra activities (restricted, always state the limits)
- **Take-off / flight**: `run_behavior("takeoff")`: looming (LPLC2) -> giant fiber (DNp01) -> a single SCRIPTED take-off
  animation. The body cannot really fly (no aerodynamics); only the trigger comes from the brain. Control:
  `silence=["giant_fiber"]` -> no take-off. Never claim the fly flies by itself.
- **Driving a car**: `drive_car(...)` maps real P9/MDN/DNa outputs to a simple car (hand-designed link). You choose the
  manoeuvres (or `goal="station"`); the fly does not learn to drive. Lesion demo: `silence=["DNa01_left","DNa02_left"]`
  with `preset="u_turn"` -> the car cannot turn left.
- **Obstacle course (closed loop, no planner)**: `run_course(course="canyon"|"hairpin"|"slalom")`: solid walls, touching
  one = crash. Wall nearness per eye -> LPLC1 of that side (designed eye model); P9 held on; the brain runs as ONE
  continuous trial in 20 ms chunks and its DNa output steers. Nobody picks manoeuvres. Turning away from the nearer wall
  is a model property (LPLC1 -> contralateral DNa01); the literature only shows LPLC1 -> slowing/collision avoidance.
  Control: `silence=["LPLC1"]` -> blind to walls -> crash.
- Requests with no circuit (e.g. "play chess", "learn to like bitter"): say so, offer the closest real experiment.

## House rules for experiments (and for the demo)
1. For any behaviour request, use the **fly-neuroscientist skill** (`.claude/skills/fly-neuroscientist/SKILL.md`).
2. **Always trace a pathway before simulating** (structural first; functional with `functional_run_id` after a run).
3. **Always run a control**: a lesion (silence a key node) or a counter-stimulus (e.g. add bitter to sugar).
4. **Always show the body** when the behaviour has one (`run_behavior`), and run the lesioned version too.
5. **Always send the report to the `literature-checker` subagent** before saving it; append its verdict.
6. **Always end with a 5-line lab report with numbers** (Hz, % change, neuron counts), saved with `save_report`.
7. Keep tool outputs short in chat: quote the 2-3 numbers that matter, not whole JSON blobs.

## Repo map
- `flylab/brain.py`, `flylab/_lif_numba.py` — the LIF model (validated spike-for-spike against the Eon PyTorch model)
- `flylab/connectome.py` — annotations, search, pathway tracing; `flylab/groups.py` — named groups from data
- `flylab/lab.py` — experiments (what the MCP tools call); `flylab/behaviors.py` + `interface.py` + `body.py` — body
- `dashboard/` — live view (tool feed from `runs/events.jsonl`, fly video, brain raster); `runs/` — all outputs
- `PROGRESS.md` — build log and measured results; `DEMO.md` — recording script
