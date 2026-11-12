"""End-to-end smoke test: brain -> connectome tools -> lesion -> body video -> MCP server -> hook -> dashboard.

Usage: python scripts/smoke_test.py [--no-body] [--no-mcp]
Exit code 0 = all checks passed.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
results = []


def check(name, fn):
    t = time.perf_counter()
    try:
        msg = fn()
        results.append((True, name))
        print(f"PASS  {name:42s} {time.perf_counter() - t:6.1f}s  {msg or ''}", flush=True)
    except Exception as e:
        results.append((False, name))
        print(f"FAIL  {name:42s} {time.perf_counter() - t:6.1f}s  {type(e).__name__}: {e}", flush=True)


def t_brain():
    from flylab.brain import Brain, Stimulus
    from flylab.groups import load_groups
    G = load_groups()
    b = Brain()
    mn9 = b.indices(G["MN9"]["root_ids"])
    base = b.run([], duration_s=0.1, n_trials=2)
    assert base.counts.sum() == 0, "baseline should be silent"
    r = b.run([Stimulus(b.indices(G["sugar_GRNs"]["root_ids"]), 200)], duration_s=0.3, n_trials=4, seed=7)
    mn = r.rates[mn9].mean()
    assert mn > 20, f"MN9 only {mn:.1f} Hz"
    return f"baseline silent; sugar -> MN9 {mn:.0f} Hz, {(r.rates > 0).sum()} active, {r.wall_s:.1f}s"


def t_search():
    from flylab import connectome
    r = connectome.search_neurons("sugar")
    assert r["n_neurons"] > 0
    return f"'sugar' -> {[m['cell_type'] for m in r['matches']]}"


def t_trace():
    from flylab import lab
    r = lab.trace_pathway("sugar_GRNs", "MN9")
    assert r["paths"], "no path"
    return " -> ".join(r["paths"][0]["path"])


def t_lesion():
    from flylab import lab
    a = lab.run_simulation(["sugar_GRNs"], duration_s=0.3, n_trials=4, seed=11)
    b = lab.run_simulation(["sugar_GRNs"], ["CB0553"], duration_s=0.3, n_trials=4, seed=11)
    c = lab.compare_conditions(a["run_id"], b["run_id"])
    ra, rb = c["output_groups"]["MN9"]["a_hz"], c["output_groups"]["MN9"]["b_hz"]
    assert rb < 0.6 * ra, f"lesion effect too small: {ra} -> {rb}"
    assert Path(REPO / a["raster_png"]).exists()
    return f"MN9 {ra} -> {rb} Hz with CB0553 silenced"


def t_body():
    from flylab import behaviors
    r = behaviors.run_behavior("walk_forward", duration_s=1.0)
    p = REPO / r["video"]
    assert p.exists() and p.stat().st_size > 10_000, "video missing"
    return f"{r['video']} ({p.stat().st_size // 1024} kB); {r['summary'][:90]}"


def t_mcp():
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "test_mcp.py"), "--quick"], capture_output=True,
                         text=True, timeout=1800, cwd=REPO)
    assert "ALL OK" in out.stdout, out.stdout[-800:] + out.stderr[-800:]
    return "all tools answered over stdio"


def t_hook():
    ev = {"hook_event_name": "PostToolUse", "tool_name": "mcp__flylab__run_simulation",
          "tool_input": {"activate": ["sugar_GRNs"]},
          "tool_response": [{"type": "text", "text": json.dumps({"run_id": "smoke", "output_groups": {"MN9": {"mean_rate_hz": 78}}})}]}
    events = REPO / "runs" / "events.jsonl"
    before = events.read_text(encoding="utf8") if events.exists() else None
    subprocess.run([sys.executable, str(REPO / ".claude" / "hooks" / "log_tool_event.py")], input=json.dumps(ev),
                   text=True, check=True, timeout=30)
    last = json.loads(events.read_text(encoding="utf8").splitlines()[-1])
    assert last["tool"] == "run_simulation" and last["short_result"]["rates_hz"]["MN9"] == 78
    if before is None:
        events.unlink()
    else:
        events.write_text(before, encoding="utf8")
    return "hook appends to runs/events.jsonl"


def t_dashboard():
    from fastapi.testclient import TestClient
    sys.path.insert(0, str(REPO / "dashboard"))
    import server
    c = TestClient(server.app)
    assert c.get("/").status_code == 200
    d = c.get("/api/latest").json()
    assert d["sim"] is not None
    return f"latest sim {d['sim']['run_id']}, latest behaviour {(d['behavior'] or {}).get('run_id')}"


if __name__ == "__main__":
    check("brain model (LIF, sugar -> MN9)", t_brain)
    check("connectome search", t_search)
    check("pathway tracing", t_trace)
    check("lesion + compare_conditions", t_lesion)
    if "--no-body" not in sys.argv:
        check("body: brain -> interface -> flygym -> mp4", t_body)
    if "--no-mcp" not in sys.argv:
        check("MCP server over stdio", t_mcp)
    check("PostToolUse hook script", t_hook)
    check("dashboard API", t_dashboard)
    ok = all(r[0] for r in results)
    print("\nSMOKE TEST", "PASSED" if ok else "FAILED", f"({sum(r[0] for r in results)}/{len(results)})")
    sys.exit(0 if ok else 1)
