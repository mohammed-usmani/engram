import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "claude_memory_hook", Path(__file__).parent.parent / "scripts" / "claude_memory_hook.py")
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def test_first_prompt_only_once(tmp_path, monkeypatch):
    monkeypatch.setattr(hook.tempfile, "gettempdir", lambda: str(tmp_path))
    assert hook._first_prompt("abc") is True
    assert hook._first_prompt("abc") is False
