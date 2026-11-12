"""Dashboard chat: each message runs Claude Code headless (`claude -p`) in the FlyLab folder.

Same program and login as the Claude Code VS Code extension, same .mcp.json / CLAUDE.md / skill /
subagents / hooks, so tool calls still stream into the live feed through the hook. The conversation
continues across messages (--resume <session_id>). Allowed tools are restricted to the flylab MCP tools,
subagents, the skill and read-only file tools; shell and file editing are disallowed.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOWED = ["mcp__flylab", "Agent", "Task", "Skill", "Read", "Grep", "Glob"]
DISALLOWED = ["Bash", "PowerShell", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch"]


def claude_bin() -> str | None:
    env = os.environ.get("FLYLAB_CLAUDE_BIN")
    if env and Path(env).exists():
        return env
    found = shutil.which("claude")
    if found:
        return found
    home = Path.home()
    cands = glob.glob(str(home / ".vscode" / "extensions" / "anthropic.claude-code-*" / "resources" / "native-binary" / "claude*"))
    cands = [c for c in cands if Path(c).is_file()]
    return max(cands, key=os.path.getmtime) if cands else None


class ChatRunner:
    def __init__(self):
        self.messages: list[dict] = []
        self.busy = False
        self.session_id: str | None = None
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def state(self, after: int = 0) -> dict:
        return {"busy": self.busy, "n": len(self.messages), "messages": self.messages[after:],
                "session_id": self.session_id, "claude_found": claude_bin() is not None}

    def reset(self):
        if self.busy:
            raise RuntimeError("Claude is still working; wait for it to finish (or stop it).")
        self.messages, self.session_id = [], None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def send(self, text: str):
        text = (text or "").strip()
        if not text:
            raise ValueError("empty message")
        with self.lock:
            if self.busy:
                raise RuntimeError("Claude is still working on the previous request.")
            self.busy = True
        self.messages.append({"role": "user", "text": text, "t": time.time()})
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _add(self, role, text):
        self.messages.append({"role": role, "text": text, "t": time.time()})

    def _run(self, text: str):
        try:
            exe = claude_bin()
            if not exe:
                self._add("error", "Claude Code program not found (set FLYLAB_CLAUDE_BIN to claude.exe).")
                return
            cmd = [exe, "-p", text, "--output-format", "stream-json", "--verbose",
                   "--allowedTools", *ALLOWED, "--disallowedTools", *DISALLOWED]
            if self.session_id:
                cmd += ["--resume", self.session_id]
            # MCP_TIMEOUT: allow the flylab server up to 2 min to start (cold Python imports on a busy laptop)
            env = dict(os.environ, PYTHONIOENCODING="utf-8", MCP_TIMEOUT="120000", MCP_TOOL_TIMEOUT="1800000")
            self.proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                         env=env)
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("session_id"):
                    self.session_id = ev["session_id"]
                if ev.get("type") == "assistant":
                    for blk in ev.get("message", {}).get("content", []):
                        if blk.get("type") == "text" and blk.get("text", "").strip():
                            self._add("assistant", blk["text"].strip())
                elif ev.get("type") == "result" and ev.get("is_error"):
                    self._add("error", str(ev.get("result") or ev.get("subtype") or "error")[:500])
            rc = self.proc.wait()
            if rc not in (0, None):
                err = (self.proc.stderr.read() or "").strip()[-600:]
                self._add("error", f"Claude Code exited with code {rc}. {err}")
        except Exception as e:  # never leave the UI stuck in 'busy'
            self._add("error", f"{type(e).__name__}: {e}")
        finally:
            self.busy = False
            self.proc = None


RUNNER = ChatRunner()
