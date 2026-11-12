# FlyLab — demo recording script (2–3 min video)

Goal: record Claude Code autonomously running the full loop (search → trace → simulate → body video → lesion →
literature check → report) next to the live dashboard. Everything is precomputed, so tool calls return in seconds.

## 0. One-time preparation (≈15–30 min, mostly unattended)

All commands from the repo root in **Git Bash** (on macOS/Linux replace `.venv/Scripts/python.exe` with `.venv/bin/python`).

```bash
bash scripts/setup.sh                                   # if not done yet
cp .claude/settings.flylab.json .claude/settings.json   # enables the hook -> dashboard live feed (+ auto-allows flylab tools)
.venv/Scripts/python.exe scripts/precompute_demo.py --clear-feed   # caches every scenario (~15-25 min with videos)
```

Check: `runs/demo_cache.json` exists and names the lesion hit (expected: `CB0553`).

## 1. Screen layout for recording (1920×1080)

- **Left 40%**: terminal with Claude Code (`claude` started in the repo folder), font size ≥ 16.
- **Right 60%**: browser at `http://localhost:8000` (dashboard), zoom so it fills the area (Ctrl+– to ~80% if needed).

Start the dashboard in a second terminal:
```bash
.venv/Scripts/python.exe dashboard/server.py
```
Fresh feed for the take: `curl -X POST http://localhost:8000/api/clear_events` (or rerun precompute with `--clear-feed`).

In Claude Code, check the server is loaded: `/mcp` should list **flylab** (11 tools). If prompted, approve the
project MCP server and the hook config. The VS Code extension works the same way (open the `Flylab` folder as
the workspace; `.mcp.json` and `.claude/` are picked up from the workspace root). The `claude` CLI is not
installed on this machine, so the server was verified with the Python MCP client (`scripts/test_mcp.py`).

## 1b. Simplest way: type in the dashboard

The dashboard (`http://localhost:8000`) has an **Ask Claude** box (bottom left). Type a prompt, press Enter, and
watch the feed, video/car, 3D brain and report update. Wait until "Claude is working…" disappears before the next
prompt (follow-ups continue the same conversation). "new chat" starts over; "stop" cancels.

Extra precomputed prompts: *"Make the fly take off when something looms, then prove the giant fiber is needed."* ·
*"Drive the car to the sugar station."* · *"Do a U-turn with the car, then silence the left steering neurons and try again."*
· *"Do a slalom / parallel park / figure eight with the car."*

## 2. The prompt to type (hero take)

```
Make the fly eat when it finds sugar, then prove which neuron is responsible.
```

What Claude does on its own (CLAUDE.md + fly-neuroscientist skill), and what you should see:

| # | Claude's tool calls | Dashboard |
|---|---|---|
| 1 | `list_behaviors`, `get_neuron_group sugar_GRNs`, `get_neuron_group MN9` | feed cards: 21 sugar GRNs, MN9 = CB0701 (2 neurons) |
| 2 | `trace_pathway sugar_GRNs → MN9` | path card: e.g. LB3d → CB0880 → DNge062 → CB0701, 3 hops, excitatory |
| 3 | `run_simulation [sugar_GRNs]` | raster: MN9 ~78 Hz, ~440 neurons active, everything else 0 |
| 4 | `trace_pathway ... functional_run_id` | functional path through active neurons (often LB3c → CB0616 → CB0553 → CB0701) |
| 5 | `run_behavior find_sugar` | **video: fly walks to the sugar drop, stops, extends proboscis**; readout chart: P9 drive, contact line, MN9 crosses 30 Hz |
| 6 | `lesion_screen [sugar_GRNs] → MN9` | card: CB0553 −76%, DNge080 −17%, CB0616 −15% … |
| 7 | `run_simulation silence CB0553`, `compare_conditions` | raster with SILENCED badge: MN9 78 → 18 Hz |
| 8 | `run_behavior find_sugar silence [CB0553]` | **video with red LESION badge: fly reaches sugar but does not feed** |
| 9 | Agent → `literature-checker` | subagent card; verdict: sugar→MN9 supported [Shiu2024]; CB0553 = model prediction |
| 10 | `save_report` | lab report panel fills in (5 numbered lines + literature check) |

If Claude skips a step (e.g. the body run of the lesion), nudge with: *"Now show me the lesioned fly."*

Backup prompts (all precomputed):
- *"Can you make it reject food when there's something bitter in it?"* (sugar vs sugar+bitter, MN9 78 → 11 Hz)
- *"Make the fly walk backwards."* (MDN) · *"Make it turn left."* · *"What happens when it cleans its antennae?"* (brain only)

## 3. Shot list (2:30)

| Time | Shot | Voice-over (suggested) |
|---|---|---|
| 0:00–0:12 | Dashboard idle, header visible | "This is a simulation of every neuron in a fruit fly's brain — 138,639 neurons wired exactly as in the FlyWire connectome — driving a physics-simulated body." |
| 0:12–0:20 | Type the prompt in Claude Code | "I just tell Claude what I want the fly to do." |
| 0:20–0:50 | Feed fills: groups, pathway, simulation; raster appears | "Claude searches the real connectome, finds the sugar-sensing neurons and the proboscis motor neuron MN9, traces the 3-synapse pathway between them, and runs the whole-brain model: MN9 fires at 78 Hz." |
| 0:50–1:15 | Video: fly walks to sugar, stops, feeds; chart playhead crosses threshold | "Then it wires the brain's outputs to the body. P9 neurons drive walking; when the legs touch the sugar, the sugar neurons fire, MN9 crosses threshold, and the fly stops to feed." |
| 1:15–1:45 | Lesion screen card → lesion video with red badge | "Now the experiment. Claude silences candidate neurons one cell type at a time. Silencing just two neurons — CB0553 — drops MN9 by 76 percent. Same fly, same sugar: it walks right up to it and doesn't eat." |
| 1:45–2:05 | Subagent card + report panel | "A literature-checker subagent verifies each claim: the sugar-to-MN9 result matches Shiu et al. 2024; the CB0553 result is flagged as a new model prediction." |
| 2:05–2:30 | Architecture (README diagram) | "Under the hood: an MCP server with 11 tools, a skill that encodes the scientific method, two subagents, and a hook that streams Claude's actions to this dashboard. Nothing here learns: Claude finds and drives circuits that already exist." |

## 4. Troubleshooting

- **Tool call slow (> 30 s)**: the parameters differ from the precomputed ones (cache miss). It still works
  (~10–20 s per simulated second); keep recording or retake.
- **No cards in the feed**: hook not active → `cp .claude/settings.flylab.json .claude/settings.json`, restart `claude`.
  Hooks call `python` (any Python 3; the hook uses only the standard library).
- **Video doesn't update**: the dashboard picks the most recently written `runs/*/result.json` of kind `behavior`;
  reload the page.
- **flylab server missing in `/mcp`**: start `claude` from the repo root (the `.mcp.json` path is relative).
