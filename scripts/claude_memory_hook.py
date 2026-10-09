#!/usr/bin/env python3
"""Claude Code hook → personal memory API. Stdlib only; never blocks or fails the session.

UserPromptSubmit: on the FIRST prompt of a session, inject recall(prompt + project) as context.
PreCompact:       save everything not yet saved (the context is about to be summarised away).
SessionEnd:       save the rest.

While a session runs, transcript_sync.py (launchd, every 10 minutes) saves it chunk by chunk;
both share one offset file, so nothing is sent twice.

Env: MEMORY_API (default http://127.0.0.1:8001), ADMIN_TOKEN (optional bearer).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import transcript_sync  # noqa: E402

API = os.environ.get("MEMORY_API", "http://127.0.0.1:8001").rstrip("/")


def _post(path: str, body: dict, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if os.environ.get("ADMIN_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['ADMIN_TOKEN']}"
    req = urllib.request.Request(API + path, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def save(event: dict) -> None:
    """Flush this session's unsaved conversation (same offsets as transcript_sync, so no repeats)."""
    path = Path(event.get("transcript_path") or "")
    if path.is_file():
        with transcript_sync.offsets() as data:
            transcript_sync.sync_file(path, "claude-code", transcript_sync.claude_entry, False, data, final=True)


def _first_prompt(session_id: str) -> bool:
    marker = Path(tempfile.gettempdir()) / f"claude-memory-{session_id}"
    if marker.exists():
        return False
    marker.touch()
    return True


def main() -> None:
    event = json.load(sys.stdin)
    name = event.get("hook_event_name")
    sid = event.get("session_id") or "unknown"
    project = Path(event.get("cwd") or ".").name

    if name == "UserPromptSubmit" and _first_prompt(sid):
        out = _post("/api/memory/recall", {"situation": f"{event.get('prompt', '')}\n(project: {project})",
                                           "session_id": sid, "budget_tokens": 800, "fast": True,
                                           "agent": "claude-code-hook"}, timeout=4)
        if out.get("items") or "so far" in out.get("brief", ""):
            print(json.dumps({"hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "Personal memory about the user (from the Engram MCP server; "
                                     "call its recall/remember tools for more):\n\n" + out["brief"]}}))
    elif name in ("PreCompact", "SessionEnd"):
        save(event)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # memory is best-effort; a down server must never break Claude Code
        print(f"memory hook skipped: {e}", file=sys.stderr)
    sys.exit(0)
