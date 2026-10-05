"""Every tool error must tell the calling agent what was wrong with its input and how to fix it."""
import pytest

from tests.test_mcp_tools import _call, mcp_db  # noqa: F401  (fixture)

import src.mcp_server as srv


def _is_input_error(out: dict) -> None:
    assert isinstance(out, dict) and "error" in out and "fix" in out, out
    assert "internal" not in out["error"].lower(), out


async def test_save_new_doc_with_slug_prefix_that_is_not_a_type(mcp_db):
    out = await _call(srv.save_document)(slug="interview/redfox-2026-10-05", agent="claude-code",
                                         title="Redfox screening", content="Summary:\nWent well.\n")
    _is_input_error(out)
    assert "'interview'" in out["error"] and "slug" in out["error"].lower()
    assert "experience" in out["valid_types"]
    assert "type=" in out["fix"] or "`type`" in out["fix"]
    assert "slug prefix" in out["how_it_works"].lower()


async def test_save_with_invalid_explicit_type_lists_valid_types(mcp_db):
    out = await _call(srv.save_document)(slug="project/x", agent="chatgpt", title="X", content="x", type="Projects")
    _is_input_error(out)
    assert "'Projects'" in out["error"] and "project" in out["valid_types"]


async def test_save_new_doc_missing_title_names_the_field(mcp_db):
    out = await _call(srv.save_document)(slug="project/new-thing", agent="chatgpt", content="x")
    _is_input_error(out)
    assert "title" in out["error"]


async def test_write_without_agent(mcp_db):
    out = await _call(srv.save_document)(slug="skill/python", agent="", content="x")
    _is_input_error(out)
    assert "agent" in out["error"] and "chatgpt" in out["fix"]


async def test_get_unknown_slug_suggests_close_matches(mcp_db):
    out = await _call(srv.get_document)(slug="skills/python")
    _is_input_error(out)
    assert "skill/python" in out["did_you_mean"]


async def test_edit_old_text_not_found_and_ambiguous(mcp_db):
    out = await _call(srv.edit_document)(slug="skill/python", old_text="java", new_text="x", agent="chatgpt")
    _is_input_error(out)
    assert "0 times" in out["error"] and "get_document" in out["fix"]
    out = await _call(srv.edit_document)(slug="experience/freelance", old_text="s", new_text="x", agent="chatgpt")
    _is_input_error(out)
    assert "times" in out["error"] and "surrounding" in out["fix"]


async def test_list_documents_invalid_type_filter(mcp_db):
    out = await _call(srv.list_documents)(type="projects")
    _is_input_error(out)
    assert "project" in out["valid_types"]


async def test_remember_bad_occurred_at(mcp_db):
    out = await _call(srv.remember)(text="Interviewed at Acme", agent="chatgpt", occurred_at="yesterday evening")
    _is_input_error(out)
    assert "occurred_at" in out["error"] and "+05:30" in out["fix"]


async def test_expand_bad_and_unknown_ids(mcp_db):
    out = await _call(srv.expand)(item_id="12")
    _is_input_error(out)
    assert "e:" in out["fix"] and "f:" in out["fix"]
    out = await _call(srv.expand)(item_id="e:999999")
    _is_input_error(out)
    assert "recall" in out["fix"]


async def test_timeline_bad_entity_and_since(mcp_db):
    out = await _call(srv.timeline)(entity="redfox")
    _is_input_error(out)
    assert "kind:name" in out["error"] or "kind:name" in out["fix"]
    out = await _call(srv.timeline)(entity="topic:job-search", since="last week")
    _is_input_error(out)
    assert "since" in out["error"]


async def test_unexpected_failure_says_it_is_not_the_agents_fault(mcp_db, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("db exploded")
    monkeypatch.setattr(srv, "_recall", boom)
    out = await _call(srv.recall)(situation="anything")
    assert "internal" in out["error"].lower() and "not your input" in out["fix"].lower()
