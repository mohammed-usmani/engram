import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.database import get_session
from src.memory import api


@pytest.fixture
async def client(engine, mem0_store, monkeypatch):
    from main import app
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def _session():
        async with Session() as s:
            yield s
    app.dependency_overrides[get_session] = _session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_remember_recall_roundtrip(client):
    r = await client.post("/api/memory/remember", json={"text": "I applied to Acme today", "agent": "test"})
    assert r.status_code == 202 and r.json()["job_id"] > 0 and r.json()["queued"] is True
    again = await client.post("/api/memory/remember", json={"text": "I applied to Acme today", "agent": "test"})
    assert again.json()["job_id"] == r.json()["job_id"] and again.json()["queued"] is False

    r = await client.post("/api/memory/note", json={"session_id": "s1", "key": "deadline", "value": "Friday"})
    assert r.status_code == 200
    r = await client.post("/api/memory/recall", json={"situation": "what am I working on", "session_id": "s1", "fast": True})
    assert r.status_code == 200 and "Friday" in r.json()["brief"]

    r = await client.post("/api/memory/teach", json={"name": "deploy cityfix", "steps": ["test", "build", "ship"]})
    fid = r.json()["id"]
    assert (await client.get(f"/api/memory/item/f:{fid}")).json()["memory"].startswith("How to deploy cityfix")
    assert (await client.delete(f"/api/memory/item/f:{fid}")).json() == {"deleted": True}
    assert (await client.get(f"/api/memory/item/f:{fid}")).status_code == 404

    jobs = (await client.get("/api/memory/jobs")).json()
    assert jobs[0]["status"] == "pending"


async def test_ingest_hook_payload(client):
    r = await client.post("/api/memory/ingest", json={"text": "user: hi\nassistant: hello", "agent": "claude-code",
                                                       "session_id": "abc", "session_end": True})
    assert r.status_code == 202


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
