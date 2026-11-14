---
name: fly-neuroscientist
description: Experimental method for FlyLab. Use whenever the user asks the fly to DO something (eat, avoid a taste, walk, turn, back up, groom) or asks which neurons cause a behaviour. Turns a plain-English request into connectome search -> pathway tracing -> whole-brain simulation -> body video -> lesion control -> literature check -> lab report, using the flylab MCP tools.
---

# Fly neuroscientist: the FlyLab experimental method

You are running experiments on a fixed, connectome-derived whole-brain model (nothing learns). Follow these
steps in order. Narrate each step in one short sentence to the user (it is being screen-recorded), quoting numbers.

## 1. Translate the request
Write down: **sensory input** (which neurons would sense this?), **expected motor output** (which output
neurons should respond?), and the **behaviour** to show on the body. Use `list_behaviors` and
`list_neuron_groups` first; use `search_neurons` for anything not curated.

| Request | Input group(s) | Output group | Behaviour |
|---|---|---|---|
| eat / feed / sugar | `sugar_GRNs` | `MN9` (proboscis) | `feed`, hero: `find_sugar` |
| avoid bitter | `sugar_GRNs` + `bitter_GRNs` | `MN9` drop | `feed_with_bitter` |
| walk forward | `P9` (descending) | walking | `walk_forward` |
| turn | `P9` + `DNa01/DNa02_<side>` | turning | `turn_left` / `turn_right` |
| walk backwards | `MDN` | backward walking | `walk_backward` |
| clean antennae | `JO_grooming` | (brain only) | `groom` |
| fly away / escape | `LPLC2` (looming) | `giant_fiber` (DNp01) | `takeoff` (scripted flight, say so) |
| drive a car | P9 / DNa / MDN | car (analogy) | `drive_car` tool (presets or goal="station") |
| obstacle course / avoid walls | eyes -> LPLC1 L/R -> DNa01/02 | car, CLOSED LOOP | `run_course` (canyon, hairpin, slalom); control `silence=["LPLC1"]` |
| swat it / escape a swatter | looming -> `LPLC2` + `LC4` | `giant_fiber` | `swat_fly`; control `silence=["giant_fiber"]` |
| ride a bike | P9 / DNa / MDN -> real legs | ball read-out -> bike | `ride_bike`; control `silence=["DNa01_left","DNa02_left"]` |
| play music / piano | walking legs or MN9 / DNa / GF | sonification | `play_piano` (keyboard / orchestra); say we composed the score |
| play a specific song | P9 walking (road) or sugar -> MN9 (singer) | notes | `play_piano(mode="song_road"/"song_orchestra", song=...)`; control `bitter_notes` / `silence=["P9"]` |
| race / compare lesions | eyes -> LPLC1 -> DNa | race leaderboard | `run_olympics` |
| fly through a cave / forest, bat | LPLC1 (walls) + LPLC2/LC4 (bat) | DNa + giant fiber | `fly_through`; control `silence=["giant_fiber"]` |
| maze / find food, avoid poison | sugar_GRNs (+ bitter_GRNs) on contact | MN9 (stop + eat) | `run_maze`; control `silence=["bitter_GRNs"]` |

## 2. Find the neurons
`get_neuron_group` for each group: report count, cell types, data source. For anything else,
`search_neurons("<keyword>")`.

## 3. Trace the pathway (before simulating!)
`trace_pathway(source, target)`. Report hop count, the best 1-2 paths (cell types) and whether they are net
excitatory or inhibitory. Inhibitory routes explain suppression (e.g. bitter -> MN9).

## 4. Simulate the brain
`run_simulation(activate=[...])` (defaults: 200 Hz, 1 s, 8 trials). Report the output-group rates and number of
active neurons. Remind: baseline is 0 Hz everywhere (no spontaneous activity). Then
`trace_pathway(..., functional_run_id=<run_id>)` to show the path restricted to neurons that actually fired.

## 5. Show the behaviour on the body
`run_behavior(<behaviour>)`. Report what the body did and the numbers that drove it (from `summary`), and say
the brain->body mapping is a documented hand-designed interface.

## 6. Control / lesion experiment (always)
- Find a necessary node: `lesion_screen(activate=[...], target=<output group>)`; pick the top hit.
- Confirm in the brain: `run_simulation(activate=[...], silence=[<hit>])` then
  `compare_conditions(control_run_id, lesion_run_id)`.
- Confirm in the body: `run_behavior(<behaviour>, silence=[<hit>])`. The behaviour should disappear.
- Or a counter-stimulus control (e.g. add `bitter_GRNs`).

## 7. Literature check
Draft the report (step 8 format), then launch the **literature-checker** subagent with the draft. It labels every
claim SUPPORTED / MODEL PREDICTION / UNSUPPORTED. Fix or relabel anything UNSUPPORTED.

## 8. Lab report
`save_report(title, markdown, run_ids=[...])` with exactly this shape, then show it to the user:
```
**Question:** <one line>
1. Pathway: <source> -> <key intermediates> -> <target> (<hops> hops, net <sign>).
2. Activation: <input> at <rate> Hz -> <output> <X> Hz (baseline 0 Hz); <N> neurons active.
3. Behaviour: <what the body did> (<driving rates>; hand-designed interface).
4. Lesion: silencing <node> (<n> neurons) -> <output> <Y> Hz (<change>%); behaviour <gone/kept>.
5. Verdict: <one-line conclusion>, literature: <supported facts>; <model predictions flagged as such>.
---
Literature check: <verdict from literature-checker>
```

## Worked examples

### "Make the fly eat when it finds sugar, then prove which neuron is responsible"
1. Input `sugar_GRNs` (21 GRNs, Shiu et al. set), output `MN9` (CB0701, 2 neurons), behaviour `find_sugar`.
2. `trace_pathway("sugar_GRNs","MN9")` -> 3-hop excitatory paths, e.g. LB3c -> CB0616 -> CB0553 -> CB0701.
3. `run_simulation(["sugar_GRNs"])` -> MN9 ~78 Hz, ~440 active neurons. Functional trace with that run_id.
4. `run_behavior("find_sugar")` -> fly walks (P9) to the drop; contact switches on sugar GRNs; MN9 crosses 30 Hz
   -> fly stops and extends its proboscis.
5. `lesion_screen(["sugar_GRNs"], "MN9")` -> top hit CB0553 (~-76%). `run_simulation(["sugar_GRNs"], silence=["CB0553"])`,
   `compare_conditions`, then `run_behavior("find_sugar", silence=["CB0553"])` -> the fly reaches the sugar but
   does not feed.
6. literature-checker: MN9 as feeding readout and sugar->MN9 are supported (Shiu 2024); CB0553 necessity is a
   MODEL PREDICTION.

### "Make it avoid bitter food"
`trace_pathway("bitter_GRNs","MN9")` -> net inhibitory paths (e.g. via CB0219/CB0493).
`run_simulation(["sugar_GRNs","bitter_GRNs"])` vs sugar alone -> MN9 ~78 -> ~11 Hz (-86%).
`run_behavior("feed_with_bitter")` vs `run_behavior("feed")`: no proboscis extension with bitter.

### "Walk backwards"
`get_neuron_group("MDN")` (4 neurons). `run_simulation(["MDN"], stim_rate_hz=100)`; `run_behavior("walk_backward")`.
Control: `run_behavior("walk_forward")` (P9) for contrast; `run_behavior("walk_backward", silence=["MDN"])` -> no movement.

### "Clean your antennae"
`get_neuron_group("JO_grooming")` (187 JO-F neurons). `run_simulation(["JO_grooming"])` -> report top responders
(descending/ascending neurons). Body: brain-only (flygym has no grooming controller) — say so plainly.
Literature: Shiu 2024 reports a small antennal-grooming circuit activated by mechanosensory input; FlyLab has not
mapped the aBN/aDN names, so name-level claims are unverified.

### "Make the fly fly away" / "take off"
`trace_pathway("LPLC2","giant_fiber")`; `run_simulation(["LPLC2"], stim_rate_hz=50)` -> giant fiber ~100 Hz (sugar -> 0 Hz).
`run_behavior("takeoff")` -> fly walks, looming at 0.6 s, giant fiber crosses 50 Hz ~20 ms later, SCRIPTED take-off.
Control: `run_behavior("takeoff", silence=["giant_fiber"])` -> no take-off. Always say the flight itself is an animation.

### "Make the fly drive a car"
Explain the analogy first (real steering/speed neurons -> hand-designed car link). `drive_car(goal="station")` (arrives
after re-planning on the real brain output), `drive_car(preset="u_turn")`, then the lesion
`drive_car(preset="u_turn", silence=["DNa01_left","DNa02_left"])` -> the car can no longer turn left.
