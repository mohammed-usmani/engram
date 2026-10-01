import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "claude_memory_hook", Path(__file__).parent.parent / "scripts" / "claude_memory_hook.py")
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def test_transcript_text_keeps_prose_only(tmp_path):
    rows = [
        {"type": "user", "message": {"role": "user", "content": "I got the Acme offer!"}},
        {"type": "user", "isMeta": True, "message": {"role": "user", "content": "meta noise"}},
        {"type": "user", "message": {"role": "user", "content": "<system-reminder>ignore</system-reminder>"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Congrats!"}, {"type": "tool_use", "name": "Bash", "input": {}}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "secret output"}]}},
        {"type": "summary", "summary": "x"},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    assert hook.transcript_text(str(p)) == "user: I got the Acme offer!\nassistant: Congrats!"


def test_first_prompt_only_once(tmp_path, monkeypatch):
    monkeypatch.setattr(hook.tempfile, "gettempdir", lambda: str(tmp_path))
    assert hook._first_prompt("abc") is True
    assert hook._first_prompt("abc") is False
