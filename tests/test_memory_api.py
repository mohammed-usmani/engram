import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.database import get_session
from src.memory import api


async def test_remember_recall_roundtrip(client):
    r = await client.post("/api/memory/remember", json={"text": "I applied to Acme today", "agent": "test", "occurred_at": "2026-10-03T10:00:00+05:30"})
    assert r.status_code == 202 and r.json()["job_id"] > 0 and r.json()["queued"] is True
    again = await client.post("/api/memory/remember", json={"text": "I applied to Acme today", "agent": "test", "occurred_at": "2026-10-03T10:00:00+05:30"})
    assert again.json()["job_id"] == r.json()["job_id"] and again.json()["queued"] is False

    r = await client.post("/api/memory/note", json={"session_id": "s1", "key": "deadline", "value": "Friday"})
    assert r.status_code == 200
    r = await client.post("/api/memory/recall", json={"situation": "what am I working on", "session_id": "s1", "fast": True})
    assert r.status_code == 200 and "Friday" in r.json()["brief"]

    r = await client.post("/api/memory/teach", json={"name": "deploy cityfix", "steps": ["test", "build", "ship"], "agent": "test"})
    fid = r.json()["id"]
    assert (await client.get(f"/api/memory/item/f:{fid}")).json()["memory"].startswith("How to deploy cityfix")
    assert (await client.delete(f"/api/memory/item/f:{fid}")).json() == {"deleted": True}
    assert (await client.get(f"/api/memory/item/f:{fid}")).status_code == 404

    jobs = (await client.get("/api/memory/jobs")).json()
    assert jobs[0]["status"] == "pending"


async def test_ingest_hook_payload(client):
    r = await client.post("/api/memory/ingest", json={"text": "user: hi\nassistant: hello", "agent": "claude-code",
                                                       "occurred_at": "2026-10-03T10:00:00+05:30",
                                                       "session_id": "abc", "session_end": True})
    assert r.status_code == 202


async def test_writes_must_say_who_and_when(client):
    no_date = await client.post("/api/memory/remember", json={"text": "Applied to Acme", "agent": "chatgpt"})
    no_agent = await client.post("/api/memory/remember", json={"text": "Applied to Acme",
                                                               "occurred_at": "2026-10-03T10:00:00+05:30"})
    no_teacher = await client.post("/api/memory/teach", json={"name": "deploy", "steps": ["ship"]})
    assert no_date.status_code == no_agent.status_code == no_teacher.status_code == 422


async def test_bearer_required_when_token_set(client, monkeypatch):
    monkeypatch.setattr(api.settings, "admin_token", "sekret")
    assert (await client.post("/api/memory/recall", json={"situation": "x", "fast": True})).status_code == 401
    ok = await client.post("/api/memory/recall", json={"situation": "x", "fast": True},
                           headers={"Authorization": "Bearer sekret"})
    assert ok.status_code == 200
    assert (await client.post("/mcp", json={})).status_code == 401


async def test_mcp_exposes_memory_tools():
    from src.mcp_server import mcp
    names = {t.name for t in await mcp.list_tools()}
    assert {"recall", "remember", "note", "teach", "expand", "timeline", "forget"} <= names


async def test_admin_shortcut_redirects(client):
    r = await client.get("/admin", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/api/admin"


async def test_mcp_tool_call_without_session_id(client):
    """ChatGPT's connector sometimes sends a tool call without the MCP session header; it must still work."""
    from src.mcp_server import mcp
    async with mcp.session_manager.run():
        r = await client.post("/mcp", headers={"Accept": "application/json, text/event-stream"},
                              json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                    "params": {"name": "list_document_types", "arguments": {}}})
    assert r.status_code == 200, r.text
    assert "resume" in r.text
