"""Deterministic run IDs: hash(params) -> runs/<run_id>/ (results are reused if present)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
RUNS.mkdir(exist_ok=True)


def params_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n] or "x"


def run_id_for(kind: str, label: str, params: dict) -> str:
    return f"{kind}_{slug(label)}_{params_hash({'kind': kind, **params})}"


def run_dir(run_id: str) -> Path:
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_result(run_id: str) -> dict | None:
    p = RUNS / run_id / "result.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf8"))
    return None


def save_result(run_id: str, result: dict) -> None:
    p = run_dir(run_id) / "result.json"
    p.write_text(json.dumps(result, indent=1, default=float), encoding="utf8")
    (RUNS / "latest.json").write_text(json.dumps({"run_id": run_id, "kind": result.get("kind")}), encoding="utf8")
