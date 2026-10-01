#!/usr/bin/env python3
"""Claude Code hook → personal memory API. Stdlib only; never blocks or fails the session.

UserPromptSubmit: on the FIRST prompt of a session, inject recall(prompt + project) as context.
SessionEnd:       send the conversation's user/assistant text to /api/memory/ingest.

Env: MEMORY_API (default http://127.0.0.1:8001), ADMIN_TOKEN (optional bearer).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

API = os.environ.get("MEMORY_API", "http://127.0.0.1:8001").rstrip("/")
MAX_TRANSCRIPT_CHARS = 24_000
MIN_TRANSCRIPT_CHARS = 400  # skip trivial sessions


def _post(path: str, body: dict, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if os.environ.get("ADMIN_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['ADMIN_TOKEN']}"
    req = urllib.request.Request(API + path, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


def transcript_text(path: str) -> str:
    """User/assistant prose only (no tool calls/results, no injected system reminders), newest kept."""
    lines = []
    for raw in Path(path).read_text(errors="ignore").splitlines():
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if entry.get("type") not in ("user", "assistant") or entry.get("isMeta"):
            continue
        msg = entry.get("message") or {}
        text = _text(msg.get("content")).strip()
        if not text or text.startswith(("<system-reminder>", "<command-", "<local-command", "Caveat:")):
            continue
        lines.append(f"{msg.get('role', entry['type'])}: {text}")
    return "\n".join(lines)[-MAX_TRANSCRIPT_CHARS:]


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
                                           "session_id": sid, "budget_tokens": 800, "fast": True}, timeout=4)
        if out.get("items") or "so far" in out.get("brief", ""):
            print(json.dumps({"hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "Personal memory about the user (from the Engram MCP server; "
                                     "call its recall/remember tools for more):\n\n" + out["brief"]}}))
    elif name == "SessionEnd" and event.get("transcript_path"):
        text = transcript_text(event["transcript_path"])
        if len(text) >= MIN_TRANSCRIPT_CHARS:
            _post("/api/memory/ingest", {"text": f"[Claude Code session in project {project}]\n{text}",
                                         "agent": "claude-code", "session_id": sid, "session_end": True}, timeout=4)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # memory is best-effort; a down server must never break Claude Code
        print(f"memory hook skipped: {e}", file=sys.stderr)
    sys.exit(0)
