#!/usr/bin/env python3
"""Save coding-assistant conversations into Engram as they happen. Stdlib only.

Reads each tool's local transcripts, keeps only user/assistant prose (no tool calls or output),
and sends new text to /api/memory/ingest in chunks, each stamped with the agent and the time of
its last message. A shared offset file (~/.config/engram/sync-offsets.json) records how far every
transcript has been read, so the Claude Code hook and this script never send the same text twice.

    transcript_sync.py            sync every source (launchd runs this every 10 minutes)
    transcript_sync.py --init     mark everything already on disk as read (first install)

Sources: Claude Code (~/.claude/projects), Codex CLI (~/.codex/sessions), Gemini CLI
(~/.gemini/tmp/*/chats). A partial chunk is held back until its transcript has been idle for
IDLE_FLUSH_S, so an open session isn't cut mid-thought.
Env: MEMORY_API (default http://127.0.0.1:8001), ADMIN_TOKEN (optional bearer).
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

API = os.environ.get("MEMORY_API", "http://127.0.0.1:8001").rstrip("/")
HOME = Path.home()
OFFSETS = HOME / ".config/engram/sync-offsets.json"
MAX_CHUNK_CHARS = 12_000  # 24k-char chunks took ~2 min each to extract and sometimes timed out
MIN_CHUNK_CHARS = 400
IDLE_FLUSH_S = 30 * 60


# -- parsers: one transcript entry -> (prose line or "", ISO timestamp or None) ----------------

def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") in ("text", "input_text", "output_text"))
    return ""


_NOISE = ("<system-reminder>", "<command-", "<local-command", "Caveat:", "<environment_context>",
          "<user_instructions>", "# AGENTS.md")


def _line(role: str, text: str) -> str:
    text = text.strip()
    return "" if not text or text.startswith(_NOISE) else f"{role}: {text}"


def claude_entry(e: dict) -> tuple[str, str | None]:
    if e.get("type") not in ("user", "assistant") or e.get("isMeta"):
        return "", None
    msg = e.get("message") or {}
    return _line(msg.get("role", e["type"]), _text(msg.get("content"))), e.get("timestamp")


def codex_entry(e: dict) -> tuple[str, str | None]:
    p = e.get("payload") or {}
    if e.get("type") != "response_item" or p.get("type") != "message" or p.get("role") not in ("user", "assistant"):
        return "", None
    return _line(p["role"], _text(p.get("content"))), e.get("timestamp")


def gemini_entry(m: dict) -> tuple[str, str | None]:
    role = {"user": "user", "gemini": "assistant", "model": "assistant"}.get(m.get("type") or m.get("role"))
    return (_line(role, _text(m.get("content"))), m.get("timestamp")) if role else ("", None)


# (agent, glob under HOME, parser, whole-file JSON?)
SOURCES = [
    ("claude-code", ".claude/projects/*/*.jsonl", claude_entry, False),
    ("codex", ".codex/sessions/**/*.jsonl", codex_entry, False),
    ("gemini-cli", ".gemini/tmp/*/chats/*.json", gemini_entry, True),
]


# -- shared offsets --------------------------------------------------------------------------

def _key(path: Path) -> str:
    # The file name (a session UUID), not the full path: Claude Code reported one session's
    # folder as "-dev-technsure" while the folder on disk is "-dev-TechNSure" (same file on a
    # case-insensitive disk), and the path-keyed offset made the hook resend 60 chunks of history.
    return path.name


@contextmanager
def offsets():
    OFFSETS.parent.mkdir(parents=True, exist_ok=True)
    with open(OFFSETS.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # the hook and launchd may run at once
        try:
            data = json.loads(OFFSETS.read_text())
        except (OSError, ValueError):
            data = {}
        merged: dict[str, int] = {}
        for k, v in data.items():  # older files were keyed by full path, sometimes twice per session
            merged[Path(k).name] = max(v, merged.get(Path(k).name, 0))
        data = merged
        yield data
        tmp = OFFSETS.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(OFFSETS)


def _entries(path: Path, whole_json: bool, start: int):
    """Yield (entry, position after it). Position is a byte offset, or a message index for JSON."""
    if whole_json:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        msgs = data.get("messages", []) if isinstance(data, dict) else data
        for i in range(start, len(msgs)):
            yield msgs[i], i + 1
        return
    with open(path, "rb") as f:
        f.seek(start)
        for raw in iter(f.readline, b""):
            if not raw.endswith(b"\n"):  # still being written
                return
            try:
                entry = json.loads(raw)
            except ValueError:
                entry = {}
            yield entry, f.tell()


def end_position(path: Path, whole_json: bool) -> int:
    if not whole_json:
        return path.stat().st_size
    pos = 0
    for _, pos in _entries(path, True, 0):
        pass
    return pos


def unsaved_chunks(path: Path, parser, whole_json: bool, start: int, final: bool):
    """([(text, timestamp)], new offset). A partial last chunk waits unless `final`."""
    chunks, buf, size, last_ts, done, pos = [], [], 0, None, start, start
    for entry, pos in _entries(path, whole_json, start):
        line, ts = parser(entry if isinstance(entry, dict) else {})
        if line:
            buf.append(line[:MAX_CHUNK_CHARS])
            size += len(buf[-1]) + 1
            last_ts = ts or last_ts
        if size >= MAX_CHUNK_CHARS:
            chunks.append(("\n".join(buf), last_ts))
            buf, size, done = [], 0, pos
        elif not buf:
            done = pos
    if final and buf:
        chunks.append(("\n".join(buf), last_ts))
        done = pos
    return [(t, ts) for t, ts in chunks if len(t) >= MIN_CHUNK_CHARS and ts], done


def _token() -> str | None:
    if os.environ.get("ADMIN_TOKEN"):
        return os.environ["ADMIN_TOKEN"]
    try:
        return (HOME / ".config/engram/admin_token").read_text().strip() or None
    except OSError:
        return None


def _post(body: dict) -> None:
    headers = {"Content-Type": "application/json"}
    if token := _token():
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(API + "/api/memory/ingest", json.dumps(body).encode(), headers)
    urllib.request.urlopen(req, timeout=10).read()


def sync_file(path: Path, agent: str, parser, whole_json: bool, data: dict, final: bool) -> int:
    """Send this transcript's unsaved text; the offset only advances once Engram accepted it."""
    key = _key(path)
    chunks, done = unsaved_chunks(path, parser, whole_json, data.get(key, 0), final)
    label = path.parent.name if agent == "claude-code" else path.stem
    for i, (text, ts) in enumerate(chunks):
        _post({"text": f"[{agent} session {label}]\n{text}", "agent": agent, "occurred_at": ts,
               "session_id": path.stem, "session_end": final and i == len(chunks) - 1})
    data[key] = done
    return len(chunks)


def run(init: bool = False) -> int:
    sent = 0
    with offsets() as data:
        for agent, pattern, parser, whole in SOURCES:
            for path in HOME.glob(pattern):
                if init:
                    data.setdefault(_key(path), end_position(path, whole))
                    continue
                idle = time.time() - path.stat().st_mtime > IDLE_FLUSH_S
                sent += sync_file(path, agent, parser, whole, data, final=idle)
    return sent


if __name__ == "__main__":
    try:
        n = run(init="--init" in sys.argv)
        print(f"engram sync: {n} chunk(s) sent")
    except Exception as e:  # Engram down: offsets didn't advance, the next run retries
        print(f"engram sync skipped: {e}", file=sys.stderr)
