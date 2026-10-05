"""Open document types (any lowercase type, registered with a description) and per-document privacy."""
import pytest

from tests.test_mcp_tools import _call, mcp_db  # noqa: F401  (fixture)

import src.mcp_server as srv
from src.privacy import REQUEST_IS_REMOTE


@pytest.fixture
def remote():
    tok = REQUEST_IS_REMOTE.set(True)
    yield
    REQUEST_IS_REMOTE.reset(tok)


async def _save(**kw):
    kw.setdefault("agent", "claude-code")
    return await _call(srv.save_document)(**kw)


# ---------------------------------------------------------------- open types

async def test_new_type_needs_a_description(mcp_db):
    out = await _save(slug="recipe/dal-makhani", title="Dal makhani", content="Ingredients:\n- urad dal\n")
    assert "type_description" in out["fix"] and "recipe" in out["error"]
    out = await _save(slug="recipe/dal-makhani", title="Dal makhani", content="Ingredients:\n- urad dal\n",
                      type_description="Recipes I cook")
    assert out["type"] == "recipe"
    types = {t["name"]: t for t in await _call(srv.list_document_types)()}
    assert types["recipe"]["description"] == "Recipes I cook" and types["recipe"]["count"] == 1
    assert types["project"]["count"] == 1 and "interview" in types  # built-ins exist


async def test_invalid_and_near_duplicate_type_names(mcp_db):
    out = await _save(slug="x/y", type="Recipes!", title="t", content="c", type_description="d")
    assert "type" in out["error"] and "lowercase" in out["fix"]
    out = await _save(slug="projects/y", title="t", content="c", type_description="my projects")
    assert "project" in out["did_you_mean"]


async def test_interview_type_is_built_in(mcp_db):
    out = await _save(slug="interview/redfox", title="Redfox screening", content="Outcome: advanced\n")
    assert out["type"] == "interview"


# ---------------------------------------------------------------- privacy

async def test_private_docs_stay_out_of_search_and_listing_shows_flag(mcp_db):
    await _save(slug="health/blood-report", title="Blood report", content="HbA1c 5.4 fasting glucose normal",
                type="health", privacy="private")
    hits = await _call(srv.search_context)(query="glucose")
    assert all(h["slug"] != "health/blood-report" for h in hits)
    doc = await _call(srv.get_document)(slug="health/blood-report")
    assert doc["privacy"] == "private" and "glucose" in doc["content"]
    listed = {d["slug"]: d for d in await _call(srv.list_documents)()}
    assert listed["health/blood-report"]["privacy"] == "private"


async def test_invalid_privacy_value(mcp_db):
    out = await _save(slug="note/x", title="x", content="x", privacy="secret")
    assert "privacy" in out["error"] and "local-only" in out["fix"]


async def test_local_only_is_invisible_and_untouchable_remotely(mcp_db, remote):
    REQUEST_IS_REMOTE.set(False)
    await _save(slug="finance/bank-accounts", title="Bank accounts", content="HDFC savings, SBI salary",
                type="finance", privacy="local-only")
    REQUEST_IS_REMOTE.set(True)
    out = await _call(srv.get_document)(slug="finance/bank-accounts")
    assert "error" in out and "Bank" not in str(out.get("did_you_mean"))
    assert all(d["slug"] != "finance/bank-accounts" for d in await _call(srv.list_documents)())
    assert all(h["slug"] != "finance/bank-accounts" for h in await _call(srv.search_context)(query="HDFC"))
    out = await _save(slug="finance/bank-accounts", title="x", content="overwrite", type="finance")
    assert "error" in out
    out = await _save(slug="finance/new", title="x", content="x", type="finance", privacy="local-only")
    assert "local-only" in out["error"]
    REQUEST_IS_REMOTE.set(False)
    assert (await _call(srv.get_document)(slug="finance/bank-accounts"))["content"].startswith("HDFC")


async def test_remote_request_flag_set_by_middleware(client):
    """Through the app, a proxied (tunnel) request is remote; a direct local one is not."""
    from src.privacy import REQUEST_IS_REMOTE as flag
    seen = {}
    from main import app

    from fastapi.routing import APIRoute

    async def _probe():
        seen["remote"] = flag.get()
        return {}
    app.router.routes.insert(0, APIRoute("/api/_test_remote_flag", _probe))  # ahead of the MCP mount at "/"
    await client.get("/api/_test_remote_flag", headers={"host": "127.0.0.1:8001"})
    local = seen["remote"]
    await client.get("/api/_test_remote_flag", headers={"host": "127.0.0.1:8001", "X-Forwarded-For": "1.2.3.4"})
    assert local is False and seen["remote"] is True
