"""Lab reports written by Claude (via the save_report MCP tool)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import cache

REPORTS = cache.RUNS / "reports"


def save_report(title: str, markdown: str, run_ids: list[str]) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{cache.slug(title)}.md"
    path = REPORTS / name
    path.write_text(f"# {title}\n\n{markdown.strip()}\n", encoding="utf8")
    for rid in run_ids:
        d = cache.RUNS / rid
        if d.exists():
            (d / "report.md").write_text(path.read_text(encoding="utf8"), encoding="utf8")
    (cache.RUNS / "latest_report.json").write_text(json.dumps({"path": f"runs/reports/{name}", "title": title}),
                                                   encoding="utf8")
    return {"saved": f"runs/reports/{name}", "attached_to": run_ids}
