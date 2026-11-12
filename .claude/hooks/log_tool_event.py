"""Claude Code hook: append FlyLab tool activity to runs/events.jsonl (the dashboard's live feed).

Registered in .claude/settings.json for PreToolUse and PostToolUse on the flylab MCP tools,
subagent launches and skills. Reads the hook JSON from stdin. Standard library only, never
blocks Claude (always exits 0).
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVENTS = REPO / "runs" / "events.jsonl"
KEEP = ("run_id", "kind", "label", "video", "raster_png", "summary", "saved", "control_run_id", "error",
        "n_targets_reached", "query", "n_neurons", "behavior")


def _text(resp):
    """Extract the text payload of a tool response (MCP content blocks, dict or str)."""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        return "\n".join(_text(x) for x in resp)
    if isinstance(resp, dict):
        if "text" in resp and isinstance(resp["text"], str):
            return resp["text"]
        if "content" in resp:
            return _text(resp["content"])
        if "result" in resp:
            return _text(resp["result"])
        return json.dumps(resp)
    return str(resp)


def _short(tool, text):
    try:
        data = json.loads(text)
    except Exception:
        return {"text": text[:300]}
    # Claude Code wraps MCP results, e.g. {"result": "<json string>"} or [{"type": "text", "text": ...}]
    for _ in range(3):
        if isinstance(data, dict) and "result" in data and len(data) <= 2 and isinstance(data["result"], str):
            try:
                data = json.loads(data["result"])
                continue
            except Exception:
                break
        if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
            try:
                data = json.loads(data[0]["text"])
                continue
            except Exception:
                break
        break
    if not isinstance(data, dict):
        return {"text": text[:300]}
    out = {k: data[k] for k in KEEP if k in data}
    og = data.get("output_groups")
    if isinstance(og, dict):
        if any(isinstance(v, dict) and "a_hz" in v for v in og.values()):  # compare_conditions
            out["changes"] = {g: f"{v['a_hz']} -> {v['b_hz']} Hz" for g, v in og.items()
                              if isinstance(v, dict) and "a_hz" in v}
        else:  # headline rates of a simulation
            out["rates_hz"] = {g: v.get("mean_rate_hz") for g, v in og.items()
                               if isinstance(v, dict) and (v.get("mean_rate_hz") or 0) > 0}
    if "sim" in data:
        out["active_neurons"] = data["sim"].get("n_active_neurons")
    if "results" in data and isinstance(data["results"], list):  # lesion screen
        out["top_hits"] = [f"{r.get('silenced')}: {r.get('change_pct')}%" for r in data["results"][:4]]
    if "paths" in data and data["paths"]:
        out["best_path"] = " -> ".join(data["paths"][0].get("path", []))
    if "matches" in data:
        out["matches"] = [m.get("cell_type") for m in data["matches"][:5]]
    if not out:
        if "description" in data:  # neuron group
            out["text"] = f"{data.get('name', '')} · {data.get('n', '?')} neurons · {data['description']}"
        elif "sections" in data:  # literature facts
            out["text"] = f"{len(data['sections'])} cited fact section(s) from data/literature.md"
        else:
            out["text"] = json.dumps(data)[:200]
    return out


def main():
    try:
        ev = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return
    tool = ev.get("tool_name", "?")
    phase = "start" if ev.get("hook_event_name") == "PreToolUse" else "end"
    args = ev.get("tool_input", {})
    if tool in ("Agent", "Task"):
        args = {"subagent": args.get("subagent_type"), "description": args.get("description")}
    rec = {"time": time.strftime("%H:%M:%S"), "ts": time.time(), "phase": phase,
           "tool": tool.replace("mcp__flylab__", ""), "args": args}
    if phase == "end":
        resp = ev.get("tool_response")
        if tool in ("Agent", "Task"):
            rec["short_result"] = {"text": _text(resp)[:400]}
        else:
            rec["short_result"] = _short(tool, _text(resp))
    EVENTS.parent.mkdir(exist_ok=True)
    line = json.dumps(rec, default=str)
    if len(line) > 6000:
        rec["args"] = str(args)[:500]
        line = json.dumps(rec, default=str)
    with EVENTS.open("a", encoding="utf8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
