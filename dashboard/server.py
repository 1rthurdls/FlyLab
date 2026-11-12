"""FlyLab dashboard server: python dashboard/server.py  ->  http://localhost:8000"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
RUNS = REPO / "runs"
RUNS.mkdir(exist_ok=True)
EVENTS = RUNS / "events.jsonl"

app = FastAPI(title="FlyLab dashboard")
app.mount("/runs", StaticFiles(directory=str(RUNS)), name="runs")
app.mount("/vendor", StaticFiles(directory=str(REPO / "dashboard" / "vendor")), name="vendor")
sys.path.insert(0, str(REPO))
_ACT_CACHE: dict = {}


@app.get("/dashboard/{name}.js")
def dashboard_js(name: str):
    if name not in ("brain3d", "car3d"):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(REPO / "dashboard" / f"{name}.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})


@app.get("/api/brain/positions")
def brain_positions():
    """Float32 xyz per neuron (model index order) for the 3D brain view."""
    from flylab import brain3d
    return Response(brain3d.positions_bytes(), media_type="application/octet-stream",
                    headers={"Cache-Control": "max-age=86400"})


@app.get("/api/brain/activity")
def brain_activity(run_id: str):
    """Trial-0 spike times + neuron roles of a simulation or behaviour run (cached by result mtime)."""
    from flylab import brain3d
    p = RUNS / run_id / "result.json"
    if not p.exists():
        return JSONResponse({"error": "unknown run"}, status_code=404)
    key = (run_id, p.stat().st_mtime)
    if key not in _ACT_CACHE:
        _ACT_CACHE.clear() if len(_ACT_CACHE) > 20 else None
        _ACT_CACHE[key] = brain3d.activity(run_id)
    return JSONResponse(_ACT_CACHE[key])


@app.get("/")
def index():
    return FileResponse(REPO / "dashboard" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/events")
def events(after: int = 0):
    if not EVENTS.exists():
        return {"n": 0, "events": []}
    lines = EVENTS.read_text(encoding="utf8", errors="replace").splitlines()
    out = []
    for ln in lines[after:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return {"n": len(lines), "events": out}


_KIND_CACHE: dict[str, tuple[float, str]] = {}  # path -> (mtime, kind)


def _latest(kind: str):
    """Most recently written result.json of a given kind (kinds cached by mtime; only the winner is parsed)."""
    cands = []
    for p in RUNS.glob("*/result.json"):
        try:
            t = p.stat().st_mtime
        except OSError:
            continue
        k = _KIND_CACHE.get(str(p))
        if k is None or k[0] != t:
            try:
                head = p.read_text(encoding="utf8")
                kk = json.loads(head).get("kind")
            except Exception:
                continue
            _KIND_CACHE[str(p)] = (t, kk)
            k = (t, kk)
        if k[1] == kind:
            cands.append((t, p))
    if not cands:
        return None
    t, p = max(cands)
    best = json.loads(p.read_text(encoding="utf8"))
    best["_mtime"] = t
    return best


def _compact_sim(d):
    if not d:
        return None
    return {k: d.get(k) for k in ("run_id", "label", "activated", "silenced", "sim", "output_groups", "top_responders",
                                  "raster_png", "timeseries", "timeseries_bin_s", "_mtime")}


@app.get("/api/latest")
def latest():
    beh = _latest("behavior")
    if beh:
        cmd_p = RUNS / beh["run_id"] / "commands.json"
        beh["commands"] = json.loads(cmd_p.read_text(encoding="utf8")) if cmd_p.exists() else None
    rep = None
    lr = RUNS / "latest_report.json"
    if lr.exists():
        info = json.loads(lr.read_text(encoding="utf8"))
        p = REPO / info["path"]
        if p.exists():
            rep = {"title": info["title"], "path": info["path"], "markdown": p.read_text(encoding="utf8"),
                   "_mtime": p.stat().st_mtime}
    return JSONResponse({"sim": _compact_sim(_latest("simulation")), "behavior": beh, "report": rep},
                        headers={"Cache-Control": "no-store"})


from fastapi import Request  # noqa: E402

from chat import RUNNER  # noqa: E402  (dashboard/chat.py)


@app.get("/api/chat")
def chat_state(after: int = 0):
    return JSONResponse(RUNNER.state(after), headers={"Cache-Control": "no-store"})


@app.post("/api/chat")
async def chat_send(req: Request):
    data = await req.json()
    try:
        RUNNER.send(data.get("message", ""))
    except (RuntimeError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    return {"ok": True}


@app.post("/api/chat/reset")
def chat_reset():
    try:
        RUNNER.reset()
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    return {"ok": True}


@app.post("/api/chat/stop")
def chat_stop():
    RUNNER.stop()
    return {"ok": True}


@app.post("/api/clear_events")
def clear_events():
    """Start a fresh feed for a recording take (old events are archived)."""
    if EVENTS.exists():
        EVENTS.rename(RUNS / f"events_archive_{int(EVENTS.stat().st_mtime)}.jsonl")
    return {"ok": True}


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
