"""Ask Engram (/chat, /api/ask) and global search (/search). No real LLM: llm.make is faked."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.memory import facts, llm
from src.memory.models import Entity, Episode, IngestJob, Reflection
from src.models import ChatMessage, ChatSession, ContextDocument

LOCAL = {"host": "127.0.0.1:8001"}


class FakeLLM:
    calls: list[dict] = []

    def __init__(self, answer):
        self.answer = answer

    async def generate(self, message, system="", history=None, json=False):
        FakeLLM.calls.append({"message": message, "system": system, "history": history})
        return self.answer


@pytest.fixture
def fake(monkeypatch):
    FakeLLM.calls = []
    made = []

    def make(provider, model=None):
        made.append((provider, model))
        return FakeLLM("You applied to Acme [e:1]. <script>alert(1)</script>")
    monkeypatch.setattr(llm, "make", make)

    async def recall(session, situation, budget_tokens=1500, session_id=None, fast=False, explain=False):
        assert fast and explain and budget_tokens == 2000
        return {"brief": "## Relevant history\n- [e:1] Applied to Acme", "items": ["e:1"],
                "explain": {"candidates": [
                    {"id": "e:1", "kind": "episode", "text": "Applied to Acme", "in_brief": True},
                    {"id": "e:2", "kind": "episode", "text": "Not used", "in_brief": False}]}}
    monkeypatch.setattr("src.ui.ask.rc.recall", recall)
    return made


async def test_ask_stores_conversation_queues_memory_and_renders_escaped(client, session, fake):
    r = await client.post("/api/ask", json={"message": "Where did I apply?", "provider": "ollama", "model": "m1"},
                          headers=LOCAL)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["provider"] == "ollama" and out["model"] == "m1"
    assert out["used"] == [{"id": "e:1", "kind": "episode", "text": "Applied to Acme", "href": "/events?focus=1"}]
    assert "[e:1] Applied to Acme" in FakeLLM.calls[0]["system"]
    cid = out["conversation_id"]

    conv = await session.get(ChatSession, cid)
    assert conv.title == "Where did I apply?" and conv.provider == "ollama" and conv.model == "m1"
    job = (await session.execute(select(IngestJob))).scalar_one()
    assert job.text == "user (Ask Engram): Where did I apply?" and job.agent == "engram-chat"
    assert job.session_id == f"ask-{cid}"

    # follow-up: same conversation, its provider, and the earlier turns as history
    r = await client.post("/api/ask", json={"message": "And then?", "conversation_id": cid}, headers=LOCAL)
    assert r.status_code == 200, r.text
    assert fake[-1] == ("ollama", "m1")
    assert [h["role"] for h in FakeLLM.calls[1]["history"]] == ["user", "assistant"]
    assert FakeLLM.calls[1]["history"][0]["content"] == "Where did I apply?"

    page = await client.get(f"/chat/{cid}", headers=LOCAL)
    assert page.status_code == 200
    assert "<script>alert(1)</script>" not in page.text and "&lt;script&gt;" in page.text
    assert "Memories used" in page.text and "No longer in memory." in page.text  # e:1 isn't in the test DB

    listing = await client.get("/chat", headers=LOCAL)
    assert f'href="/chat/{cid}"' in listing.text

    r = await client.delete(f"/api/ask/conversations/{cid}", headers=LOCAL)
    assert r.status_code == 200
    session.expire_all()
    assert await session.get(ChatSession, cid) is None
    assert not (await session.execute(select(ChatMessage))).scalars().all()
    assert (await client.delete(f"/api/ask/conversations/{cid}", headers=LOCAL)).status_code == 404


async def test_ask_rejects_provider_without_key_and_empty_message(client, fake, monkeypatch):
    monkeypatch.setattr(llm, "key_source", lambda name: "local" if name == "ollama" else None)
    r = await client.post("/api/ask", json={"message": "hi", "provider": "openai"}, headers=LOCAL)
    assert r.status_code == 400 and "Settings" in r.json()["detail"]
    r = await client.post("/api/ask", json={"message": "   "}, headers=LOCAL)
    assert r.status_code == 400
    assert not fake


async def test_chat_page_suggests_from_real_topics(client, session, fake):
    session.add(Entity(slug="company:acme", kind="company", name="Acme Corp", aliases=[]))
    now = datetime.now(timezone.utc)
    session.add_all([Episode(occurred_at=now - timedelta(days=i), kind="applied", summary=f"Acme step {i}",
                             entities=["company:acme"]) for i in range(3)])
    await session.commit()
    page = await client.get("/chat", headers=LOCAL)
    assert page.status_code == 200
    assert "What&#39;s the latest on Acme Corp?" in page.text
    assert "Answer with" in page.text and 'href="/settings"' in page.text


async def test_search_every_section(client, session, monkeypatch):
    session.add_all([
        ContextDocument(type="project", slug="projects/zephyr", title="Zephyr", content="The zephyr engine design.",
                        source="manual", privacy="normal"),
        ContextDocument(type="project", slug="secret", title="Zephyr notes", content="zephyr secret",
                        source="manual", privacy="local-only"),
        Entity(slug="project:zephyr", kind="project", name="Zephyr", aliases=["zeph"]),
        Episode(occurred_at=datetime.now(timezone.utc), kind="milestone", summary="Shipped zephyr v2",
                entities=["project:zephyr"]),
        Reflection(lesson="Zephyr releases go smoother on Mondays", evidence=[], entities=[]),
    ])
    await session.commit()
    await facts.add_fact("Zephyr is written in Rust", "fact", ["project:zephyr"])

    async def down(*a, **k):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(facts, "similar", down)  # exercises the SQL fallback

    page = await client.get("/search", params={"q": "zephyr"}, headers=LOCAL)
    assert page.status_code == 200
    t = page.text
    assert 'href="/api/admin/projects/zephyr"' in t and 'href="/api/admin/secret"' in t  # local sees local-only
    assert 'href="/topics/project:zephyr"' in t
    assert "Shipped zephyr v2" in t and "/events?focus=" in t
    assert "Zephyr is written in Rust" in t and "/facts?q=Zephyr" in t
    assert "Zephyr releases go smoother" in t
    assert 'value="zephyr"' in t  # sidebar box keeps the query

    remote = await client.get("/search", params={"q": "zephyr"})
    assert 'href="/api/admin/secret"' not in remote.text

    empty = await client.get("/search", params={"q": "qwertyuiop"}, headers=LOCAL)
    assert "Nothing matches" in empty.text
