# FlyLab — PROGRESS

## Morning summary (kept current)

| Milestone | Status | Notes |
|---|---|---|
| M0 setup | done | Python 3.11 venv via `uv` (needed `--native-tls`: TLS interception on this machine) |
| M1 brain | done | Spike-for-spike identical to Eon's PyTorch reference; ~25-50x faster |
| M2 connectome tools | done | search, groups (from data), strongest-path tracing, functional tracing |
| M3 MCP server | done | 11 tools, `scripts/test_mcp.py` all green |
| M4 body | done | flygym 1.2.1 NeuroMechFly renders on Windows (no fallback needed); forward/turn/backward/feeding + sugar-contact closed loop |
| M5 Claude layer | done | CLAUDE.md, skill, 2 subagents, hook script. Hook config shipped as `.claude/settings.flylab.json` (see below) |
| M6 dashboard | done | `dashboard/server.py` + `index.html`, live feed, video, synced brain readout chart, raster, report |
| M7 demo hardening | done | `precompute_demo.py` caches all 26 demo steps (replay in ~9 s); `smoke_test.py` **8/8 PASS**; DEMO.md |
| M8 polish | partial | architecture slide `docs/architecture.png`, second scenario (bitter) + backward/turn/lesion controls rendered; hero clips in `docs/` |

### What works right now
- Whole-brain LIF model of all 138,639 FlyWire neurons, **spike-for-spike identical** to the Eon PyTorch model,
  1 s of brain activity x 8 trials in ~6-15 s on this laptop.
- Hero: P9 makes the fly walk to a sugar drop; the proboscis touches it at 0.67 s; sugar GRNs switch on; MN9 rises to
  78 Hz; the fly stops and extends its proboscis. **Lesion**: an automatic screen ranks CB0553 first (-76%); with CB0553
  silenced MN9 only reaches 18 Hz and the fly walks 33 mm straight over the sugar without feeding.
  Clips: `docs/hero_find_sugar_intact.mp4`, `docs/hero_find_sugar_CB0553_lesion.mp4`. Dashboard: `docs/dashboard.png`.
- Also rendered and cached: feed, feed_with_bitter (MN9 11 Hz, no feeding), walk_forward, walk_backward (+ MDN lesion:
  fly stays put), turn_left, turn_right; groom (brain only).

### Commands to see it
```bash
.venv/Scripts/python.exe scripts/smoke_test.py          # 8/8, ~5 min (renders one short clip)
.venv/Scripts/python.exe dashboard/server.py            # http://localhost:8000 (shows the cached hero)
cp .claude/settings.flylab.json .claude/settings.json   # once: hook -> live feed
# then open Claude Code in this folder and type the DEMO.md prompt
```

### Open problems / caveats
- Not yet run: a real Claude Code session driving the MCP tools end to end (needs you: approve the server + hooks).
  Everything it needs was tested via the Python MCP client and the hook script directly.
- Physics is slow (~100 s wall per simulated second with rendering). Any behaviour Claude requests with *new*
  parameters takes 3-8 min. The demo prompts are all precomputed; the skill tells Claude to use defaults.
- flygym's controller drifts ~20 deg left over 1.5 s of "straight" walking; P9->DNa02 asymmetry partially cancels it.
- Water GRNs (one side) don't drive MN9 in our runs (the paper reports water GRNs drive feeding; our group choice may
  differ from theirs). Not used in the demo.

### What I would do next
1. Record a dry run in Claude Code and tune SKILL.md wording if Claude skips steps.
2. Show the lesion screen as an animated bar chart on the dashboard.
3. Map aBN/aDN grooming neurons onto FlyWire types and add a grooming leg-motion clip (flygym has no grooming controller).

(Details and decisions below.)

## Extra activities v2 (2026-11-12/13): swatter, bicycle, piano band, songs, Olympics, fly-throughs, maze

All six are MCP tools (`swat_fly`, `ride_bike`, `play_piano`, `run_olympics`, `fly_through`, `run_maze`), have a
dashboard view (`car3d.js` scenes, `piano.js`), a control/lesion, and cached results (`scripts/demo_extras.py <name>`).
Numbers are in README ("Extra activities") and `data/literature.md` ([FlyLab-data] facts).

Measured before building (changed the design):
- Bitter GRNs alone drive NO output (MDN 0 Hz): bitter does not make the model back up; in the maze it only blocks
  eating. (My initial pitch said otherwise; corrected.)
- LPLC2 10/20/30/50 Hz -> giant fiber 33/64/85/105 Hz; LC4 alone 20/30/50 Hz -> 8/26/47 Hz.

Decisions / fixes worth knowing:
- `looming.py`: one designed eye model (rate = 100 Hz x clip(expansion / 500 deg/s)) for swatter and bat.
- Swatter: closed loop in 5 ms chunks; escape = GF > 50 Hz over 20 ms + 10 ms latency before impact. Videos at 0.1x.
- Bat: the giant fiber also fires short wall-driven bursts in closed loop (LPLC1 input on both eyes). The first forest
  run credited such a burst to the bat (bat 35 m away, looming 0.5 Hz). Now a dodge counts only while the bat's looming
  drive is >= 5 Hz; other bursts are reported (`gf_bursts_without_bat`).
- Course geometry: obstacles in bends crash the fly in the two-eye model (inside: cancels the bend cue; outside / right
  after a bend: dead ahead, both eyes equal). Cave pillars live on straights or the inside after a bend.
- Garden maze: 9 m corridor with 10 m bends was too tight (crash); widened to 11 m / 14 m like the proven hairpin.
- Bicycle reads the LEGS (ball-rig equivalent: forward speed + yaw from the body trajectory), not the neurons.
- Piano keyboard: footfalls = tarsal contact force hysteresis (on 2.5, off 1.0, >= 15 ms swing); front legs only.
- Song road: first version replayed the physics twice (path pass + filmed pass) and drifted up to 5 mm / blew up
  (mjWARN_BADQACC) in the filmed pass. v3 lays each key (mocap body) just ahead of the fly along its current heading
  in ONE pass; physics errors retry with another CPG seed. Ode to Joy: 28/30 notes, ~73 BPM.
- Orchestra drums (300+ visual neurons stimulated for 19 s) took > 2 h on the active-set kernel; now one
  BrainStepper (dense parallel) trial, 14 min, cached. Snare = peak 20 ms giant-fiber bin per eighth note.
- Dashboard: the first `/api/brain/activity` request imported torch + connectome inside the request (20-60 s) and
  blocked the server; the page stayed empty. Server now warms up in a background thread; `/api/latest` is memoised
  on `runs/latest.json`; the page never overlaps polls; 3D brain caps at 120k spikes. `piano.js` is scoped
  (a global `loop` name clashed with index.html). Hook matcher anchored (`Task` matched `TaskStop`).
- `.mcp.json` now starts the server from this folder (it pointed to an older copy in Downloads/Flylab).

Costs on this laptop (CPU only, often with two jobs at once): closed-loop course 10-25 min, swatter video 3-5 min,
bike 10-50 min, piano keyboard 25 min, song road 30 min, orchestra 15 min, singer 4 min.

## ACTION NEEDED in the morning (1 command)
The auto-mode safety classifier blocked me from writing `.claude/settings.json` (it would have changed the hooks of
my own running session). The hook config is ready as `.claude/settings.flylab.json`. To enable the live feed:
`cp .claude/settings.flylab.json .claude/settings.json` (it also pre-allows the `mcp__flylab` tools).
Also: the `claude` CLI is not installed on this machine (VS Code extension only), so `claude mcp list` was not run;
the MCP server was verified end to end with the official Python MCP client (`scripts/test_mcp.py`, all 11 tools OK).

## Machine
- Windows 11, Intel i7-1360P (12 cores / 16 threads; 4P+8E), 15.6 GB RAM, Intel Iris Xe (no CUDA, no MPS) -> **CPU only**.
- Python 3.11.14 in `.venv` (system Python is 3.12; uv downloaded 3.11).

## Key results so far (all from the model, 1 s simulated, 8 trials, Poisson stimulus)
| Condition | MN9 (Hz) | Other |
|---|---|---|
| baseline (no stimulus) | 0 | the model has no spontaneous activity: all neurons 0 Hz |
| sugar GRNs 200 Hz | **78.1** | 440 active neurons (paper notebook: "about 400") |
| sugar GRNs 150 / 100 / 50 Hz | 70.2 / 56.1 / 17.9 | graded with stimulus rate |
| sugar + bitter GRNs | **11.0** | bitter suppresses the feeding output (-86%) |
| bitter GRNs alone | 0 | |
| water GRNs (15 LB3a, one side) | 0.06 | water does NOT drive MN9 with this group; not used |
| P9 (DNp09) 100 Hz | 0 | P9 ~100 Hz; recruits DNa02_R 19.5 Hz, DNa02_L 5.5 Hz (asymmetric) |
| MDN 100 Hz | 0 | MDN ~100 Hz |
| JO_grooming (JO-F) 200 Hz | 0 | 312 active; top: DNg84, CB1231b, AN_GNG_65; no walking/feeding outputs |

**Lesion screen** (sugar GRNs -> MN9, automatic candidates from the functional pathway, silencing whole cell types):
CB0553 -76% (78 -> 18.5 Hz), DNge080 -17%, CB0616 -15%, DNge062 -10%, CB0880 -1%,
CB0192 +4%, CB0499 +10%, CB0465 +23% (these two lie on an inhibitory path: silencing them disinhibits MN9).
Single-neuron (one side) lesions: CB0553_R -41%, CB0553_L -33%.
=> **Hero lesion: silence CB0553 (2 neurons)**. This is a *model prediction*; it is not in `data/literature.md`.

## Decisions log
- **Python 3.11 via uv** with `--native-tls` (uv's bundled certs fail on this network: "invalid peer certificate: UnknownIssuer").
- **flygym**: unpinned resolution picked ancient 0.2.5; forced `flygym>=1.0` -> 1.2.1 (+ mujoco 3.2.7, numpy 2.0.2).
- **mcp 2.3.0**: `FastMCP` is renamed `MCPServer` in SDK v2 (`from mcp.server.mcpserver import MCPServer`). Used it.
- **Brain backend**: The Eon PyTorch backend (sparse matmul over all 138,639 neurons every 0.1 ms) took 72.7 s per
  0.1 s simulated on this CPU (~12 min per simulated second per trial). Too slow for an interactive demo. I kept its
  exact equations/parameters/update order and wrote:
  1. `Brain.run_torch`: event-driven PyTorch port (only spikes that happened are propagated; ring-buffer delay line).
  2. `flylab/_lif_numba.py`: dense numba kernel (same math) and an **active-set** kernel: neurons exactly at rest are
     skipped; active neurons are stored in compact "slots" (L2-cache friendly); spikes are sorted each step so float
     sums happen in the same order as the dense kernel.
  - Validation (`scripts/validate_against_pytorch.py`): Eon `TorchModel` and FlyLab kernel fed the *same* Poisson
    input spikes for 0.1 s of sugar stimulation: **1539 vs 1539 spikes, 317 vs 317 active neurons, 0 neurons with any
    difference**, MN9 spikes identical. Active-set vs dense numba kernel: identical spike counts.
  - Only approximation: a neuron returns to the resting set when |v - v_rest| < 1e-4 mV and |g| < 1e-4 mV
    (float32 resolution at -52 mV is ~4e-6 mV, so exact equality never happens). Validated bit-identical anyway.
- **Silencing semantics** = Shiu et al.: outgoing synapses set to 0. For output neurons (e.g. silencing MN9 itself),
  readouts report the *effective* output (silenced neurons count as 0 Hz) because their target (muscle) is not modelled.
- **Neuron groups** are parsed from data/code, never typed from memory (`flylab/groups.py` -> `data/neuron_groups.json`):
  - sugar_GRNs: the 21-neuron Shiu et al. set as shipped (v783 IDs) in `external/fly-brain/code/benchmark.py`.
    FlyWire annotations label them LB3c (10), LB3d (5), LB4b (4), LB3b (2) on the left side -> note that 9 of the
    paper's "sugar" neurons are now annotated as other gustatory types (LB3d high_salt, LB4b putative_attractive).
    An annotation-based alternative (`sugar_GRNs_flywire_all`, 57 neurons) is also available.
  - MN9 = the notebook's `id_mn9` (720575940660219265) -> FlyWire type CB0701 -> both CB0701 neurons.
  - P9 = `benchmark.py` EXPERIMENTS['p9'] ("P9 left/right") -> FlyWire type DNp09.
  - DNa01/DNa02 left/right, MDN, bitter (LB1a-d, same side as sugar), water (LB3a), JO_grooming (sub-class 'grooming'),
    JO_CE, proboscis_motor_neurons, DNg12 from FlyWire annotations.
- **Grooming**: brain-only (flygym has no grooming controller). aBN1/aDN names from Hampel 2015 are not mapped to
  FlyWire types in our data, so we only report top responders.
- **Interface thresholds** (interface.py): MN9 threshold 30 Hz (between sugar+bitter 11 Hz / CB0553-lesion 18.5 Hz and
  sugar 78 Hz), FULL = 80 Hz; P9/MDN/turn FULL = 100 Hz (directly stimulated DNs fire ~100 Hz at 100 Hz input).

## Body (M4) facts
- Controller: flygym `HybridTurningController` (CPG + stumbling/retraction reflexes), action = [d_left, d_right];
  |d| = amplitude, sign = step direction (negative drive = backward walking, works: ~8 mm/s).
- Mapping (body.py): net = forward - backward; d_L = net - 0.6 turn; d_R = net + 0.6 turn; zero while feeding > 0.5;
  50 ms low-pass. Proboscis: stock model has Rostrum/Haustellum welded to the head; the body agent added hinge joints +
  position actuators (visual stand-in for proboscis extension, angles chosen by eye) and a head-pitch actuator.
- Sugar drop: visual only (no collision); contact = proboscis tip or a front tarsus within 1.5 mm (x,y), checked every ms.
- Video 960x720, 30 fps, **0.25x slow motion** (the dashboard maps video time x 0.25 to simulated time).
- Speed: ~75-125 s wall per simulated second with rendering (MuJoCo mesh collisions + flygym Python), ~13 s model build.
  => all demo clips are pre-rendered (`scripts/precompute_demo.py`).
- Known: flygym's controller drifts ~+20 deg (left) in 1.5 s of straight walking; P9 activation in the brain recruits
  DNa02_R > DNa02_L, giving turn = -0.14 (right), which partly cancels it.
- Hero `find_sugar`: brain sim 1 = P9 (100 Hz) for 3 s -> body walks until contact time tc; brain sim 2 = P9 + sugar GRNs
  from tc (re-simulated from t=0 with the new schedule, statistically identical up to tc) -> body continues from its state.
  Note: once contact happened the sugar stimulus stays on, even if a lesioned fly walks past the drop.

## Timings (CPU, numba active-set kernel; note: other jobs were often running concurrently)
- Connectome load: ~2.5 s (cached CSR npz). Annotations: ~6 s first time, then parquet cache.
- numba JIT: ~25-35 s first time ever, cached afterwards (`cache=True`).
- Sugar GRNs, 1 trial x 1 s: ~2 s. 8 trials x 1 s: ~11-16 s. 8 trials x 0.5 s: ~6-8 s.
- Eon reference PyTorch (sparse matmul, CPU): 72.7 s for 1 trial x 0.1 s.
- Lesion screen with 8 cell-type candidates (9 sims of 1 s x 8 trials): ~3 min.

## Log
- 22:06 start. ~22:45 venv + data. ~23:10 kernel validated vs Eon PyTorch. ~23:35 MCP server green. ~23:45 lesion screen,
  Claude layer, dashboard. ~00:06 body agent done (flygym renders). 00:10-01:05 precompute (all clips).
  ~01:20 hero re-rendered with proboscis-tip contact. ~01:35 smoke test 8/8, docs, final commit.
- Git: `runs/` is a local cache (~80 MB, git-ignored); hero clips + figures are copied to `docs/`.
