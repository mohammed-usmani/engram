import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "transcript_sync", Path(__file__).parent.parent / "scripts" / "transcript_sync.py")
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


def _claude(*texts, ts="2026-10-03T10:00:00Z"):
    return "".join(json.dumps({"type": "user", "timestamp": ts, "message": {"role": "user", "content": t}}) + "\n"
                   for t in texts)


def test_parsers_keep_prose_only():
    assert sync.claude_entry({"type": "user", "timestamp": "t", "message": {"role": "user", "content": "Hi"}}) == ("user: Hi", "t")
    assert sync.claude_entry({"type": "user", "isMeta": True, "message": {"content": "x"}})[0] == ""
    assert sync.claude_entry({"type": "user", "message": {"content": "<system-reminder>x</system-reminder>"}})[0] == ""
    assert sync.claude_entry({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "text", "text": "Done"}, {"type": "tool_use", "name": "Bash"}]}})[0] == "assistant: Done"
    assert sync.claude_entry({"type": "user", "message": {"content": [{"type": "tool_result", "content": "x"}]}})[0] == ""
    assert sync.codex_entry({"timestamp": "t", "type": "response_item", "payload": {
        "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Fixed"}]}}) == ("assistant: Fixed", "t")
    assert sync.codex_entry({"type": "response_item", "payload": {"type": "function_call"}})[0] == ""
    assert sync.gemini_entry({"type": "gemini", "content": "Sure", "timestamp": "t"}) == ("assistant: Sure", "t")


def test_long_session_is_saved_in_full_once(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "MAX_CHUNK_CHARS", 1000)
    monkeypatch.setattr(sync, "MIN_CHUNK_CHARS", 10)
    monkeypatch.setattr(sync, "OFFSETS", tmp_path / "offsets.json")
    sent = []
    monkeypatch.setattr(sync, "_post", sent.append)
    p = tmp_path / "s1.jsonl"
    p.write_text(_claude(*["a" * 300] * 5))  # one full chunk + a partial one

    with sync.offsets() as data:
        assert sync.sync_file(p, "claude-code", sync.claude_entry, False, data, final=False) == 1
    assert sent[0]["agent"] == "claude-code" and sent[0]["occurred_at"] == "2026-10-03T10:00:00Z"
    with sync.offsets() as data:  # the partial chunk waits for more conversation
        assert sync.sync_file(p, "claude-code", sync.claude_entry, False, data, final=False) == 0

    with p.open("a") as f:
        f.write(_claude("the end", ts="2026-10-03T11:00:00Z") + '{"type": "user", "mess')  # half-written line
    with sync.offsets() as data:
        assert sync.sync_file(p, "claude-code", sync.claude_entry, False, data, final=True) == 1
    assert sent[-1]["text"].endswith("user: the end") and sent[-1]["session_end"] is True
    with sync.offsets() as data:  # nothing is sent twice
        assert sync.sync_file(p, "claude-code", sync.claude_entry, False, data, final=True) == 0


def test_offsets_survive_a_failed_post(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "MIN_CHUNK_CHARS", 1)
    monkeypatch.setattr(sync, "OFFSETS", tmp_path / "offsets.json")
    p = tmp_path / "s.jsonl"
    p.write_text(_claude("hello"))

    def down(body):
        raise OSError("engram down")
    monkeypatch.setattr(sync, "_post", down)
    try:
        with sync.offsets() as data:
            sync.sync_file(p, "claude-code", sync.claude_entry, False, data, final=True)
    except OSError:
        pass
    sent = []
    monkeypatch.setattr(sync, "_post", sent.append)
    with sync.offsets() as data:
        sync.sync_file(p, "claude-code", sync.claude_entry, False, data, final=True)
    assert len(sent) == 1  # retried, not lost


def test_same_session_under_another_spelling_is_not_resent(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "MIN_CHUNK_CHARS", 1)
    monkeypatch.setattr(sync, "OFFSETS", tmp_path / "offsets.json")
    sent = []
    monkeypatch.setattr(sync, "_post", sent.append)
    (tmp_path / "TechNSure").mkdir()
    p = tmp_path / "TechNSure" / "abc.jsonl"
    p.write_text(_claude("hello"))
    with sync.offsets() as data:
        sync.sync_file(p, "claude-code", sync.claude_entry, False, data, final=True)
    other_spelling = tmp_path / "other" / "abc.jsonl"  # how the hook may name the same session
    other_spelling.parent.mkdir()
    other_spelling.write_text(_claude("hello"))
    with sync.offsets() as data:
        sync.sync_file(other_spelling, "claude-code", sync.claude_entry, False, data, final=True)
    assert len(sent) == 1
