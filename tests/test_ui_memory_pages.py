"""Events, Facts and Lessons pages and their JSON edits. Uses the test DB and local Ollama embeddings."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src import settings_store as store
from src.memory import USER_TZ, consolidate, facts
from src.memory.models import Entity, Episode, Reflection
from src.services.ollama_client import embed

H = {"host": "127.0.0.1:8001"}
NOW = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    monkeypatch.setattr(store, "_cache", {})


async def _events(session):
    session.add(Entity(slug="topic:job-search", kind="topic", name="Job search", aliases=[], dirty=False))
    a = Episode(occurred_at=NOW - timedelta(hours=1), kind="interview", summary="Acme onsite went well",
                outcome="advanced", importance=4, entities=["topic:job-search"], source_agent="chatgpt",
                payload={"job": 7})
    b = Episode(occurred_at=NOW - timedelta(days=40), kind="task_done", summary="Shipped VoiceDiary release",
                entities=[], source_agent="claude-code")
    session.add_all([a, b])
    await session.commit()
    return a, b


async def test_events_page_filters_panel_csv(client, session):
    a, b = await _events(session)
    html = (await client.get("/events", headers=H)).text
    assert "Acme onsite went well" in html and "Shipped VoiceDiary" in html and "k-interview" in html
    assert "Last 7 days by kind" in html and "chatgpt · 1" in html
    html = (await client.get("/events", params={"period": "7"}, headers=H)).text
    assert "Acme onsite" in html and "VoiceDiary" not in html
    html = (await client.get("/events", params={"q": "voicediary"}, headers=H)).text
    assert "VoiceDiary" in html and "Acme onsite" not in html
    html = (await client.get("/events", params={"topic": "topic:job-search", "focus": a.id}, headers=H)).text
    assert "VoiceDiary" not in html and "/queue#job-7" in html and 'href="/topics/topic%3Ajob-search"' in html
    panel = await client.get(f"/events/{a.id}/panel", headers=H)
    assert panel.status_code == 200 and "Edit event" in panel.text and "<html" not in panel.text
    csv = (await client.get("/api/events.csv", params={"agent": "chatgpt"}, headers=H)).text
    assert csv.splitlines()[0].startswith("id,occurred_at") and "Acme onsite" in csv and "VoiceDiary" not in csv


async def test_patch_event(client, session):
    a, _ = await _events(session)
    r = await client.patch(f"/api/events/{a.id}", headers=H, json={
        "summary": "Acme onsite went great", "outcome": "", "kind": "milestone", "importance": 5,
        "occurred_at": "2026-03-01T09:30"})
    assert r.status_code == 200, r.text
    await session.refresh(a)
    assert (a.summary, a.outcome, a.kind, a.importance) == ("Acme onsite went great", None, "milestone", 5)
    assert a.occurred_at.astimezone(USER_TZ).strftime("%Y-%m-%d %H:%M") == "2026-03-01 09:30"
    assert a.embedding is not None
    ent = (await session.execute(select(Entity))).scalar_one()
    await session.refresh(ent)
    assert ent.dirty
    assert (await client.patch(f"/api/events/{a.id}", headers=H, json={"kind": "nope"})).status_code == 400
    assert (await client.patch("/api/events/999999", headers=H, json={"importance": 2})).status_code == 404


async def test_facts_flow(client, session):
    r = await client.post("/api/facts", headers=H, json={"text": "Lives in Bangalore", "kind": "fact",
                                                          "importance": 2, "topics": "Bangalore, Home"})
    assert r.status_code == 200, r.text
    home = r.json()["id"]
    assert sorted(r.json()["entities"]) == ["topic:bangalore", "topic:home"]
    f = await facts.get_fact(home)
    assert f["metadata"]["agent"] == "admin" and f["metadata"]["importance"] == 2

    # edit keeps other metadata and re-embeds
    r = await client.patch(f"/api/facts/{home}", headers=H, json={"text": "Lives in Bengaluru, India", "importance": 3})
    assert r.status_code == 200
    f = await facts.get_fact(home)
    assert f["memory"] == "Lives in Bengaluru, India" and f["metadata"]["importance"] == 3
    assert f["metadata"]["entities"] == ["topic:bangalore", "topic:home"] and f["metadata"]["agent"] == "admin"
    assert (await facts.similar("Which city does he live in"))[0]["id"] == home

    # pinned facts open the profile even when less important
    big = await facts.add_fact("Is a backend engineer", "fact", ["topic:work"], importance=5)
    assert (await facts.profile_rows())[0]["id"] == big
    assert (await client.post(f"/api/facts/{home}/pin", headers=H, json={"pinned": True})).status_code == 200
    assert (await facts.profile_rows())[0]["id"] == home
    assert (await facts.profile()).startswith("- Lives in Bengaluru")

    page = (await client.get("/facts", headers=H)).text
    assert "Your profile" in page and "Lives in Bengaluru" in page and "Unpin" in page

    # combine: new fact gets union of topics and max importance; old ones are replaced, then one restored
    r = await client.post("/api/facts/combine", headers=H, json={"ids": [home, big],
                                                                 "text": "Backend engineer living in Bengaluru"})
    assert r.status_code == 200, r.text
    new = r.json()["id"]
    nf = await facts.get_fact(new)
    assert nf["metadata"]["importance"] == 5 and nf["metadata"]["pinned"] is True
    assert set(nf["metadata"]["entities"]) == {"topic:bangalore", "topic:home", "topic:work"}
    assert (await facts.get_fact(home))["metadata"]["superseded_by"] == new
    assert [x["id"] for x in await facts.profile_rows()] == [new]

    replaced = (await client.get("/facts", params={"replaced": 1}, headers=H)).text
    assert "Lives in Bengaluru" in replaced and "→ Backend engineer living in Bengaluru" in replaced
    assert (await client.post(f"/api/facts/{home}/restore", headers=H)).status_code == 200
    assert not (await facts.get_fact(home))["metadata"].get("superseded_by")
    assert home in {x["id"] for x in await facts.similar("Which city does he live in")}
    assert (await client.post(f"/api/facts/{home}/restore", headers=H)).status_code == 400
    assert (await client.patch("/api/facts/not-a-uuid", headers=H, json={"importance": 2})).status_code == 404


async def _lesson(session, text="System design is the weak spot in onsites"):
    eps = [Episode(occurred_at=NOW - timedelta(days=i), kind="interview", summary=f"Onsite {i}: system design weak",
                   entities=["topic:job-search"]) for i in (1, 2)]
    session.add_all(eps)
    await session.flush()
    r = Reflection(lesson=text, evidence=[e.id for e in eps] + [999999], confidence=0.5,
                   entities=["topic:job-search"], embedding=await embed(text))
    session.add(r)
    await session.commit()
    return r


async def test_lessons_page_and_edits(client, session):
    r = await _lesson(session)
    html = (await client.get("/lessons", params={"sort": "least"}, headers=H)).text
    assert "System design is the weak spot" in html and "Confidence of all 1 lessons" in html
    ev = (await client.get(f"/lessons/{r.id}/evidence", headers=H)).text
    assert "Onsite 1: system design weak" in ev and "1 supporting event was forgotten" in ev

    assert (await client.post(f"/api/lessons/{r.id}/confirm", headers=H)).status_code == 200
    await session.refresh(r)
    assert r.confidence == 1.0
    r2 = await client.patch(f"/api/lessons/{r.id}", headers=H, json={"lesson": "Onsites keep exposing weak system design"})
    assert r2.status_code == 200
    await session.refresh(r)
    assert r.lesson == "Onsites keep exposing weak system design"

    assert (await client.post(f"/api/lessons/{r.id}/reject", headers=H)).status_code == 200
    assert await session.scalar(select(Reflection).where(Reflection.id == r.id).execution_options(populate_existing=True)) is None
    assert store.get("rejected_lessons") == ["Onsites keep exposing weak system design"]


async def test_consolidate_skips_rejected(session, mem0_store, monkeypatch):
    monkeypatch.setattr(store, "_cache", {"rejected_lessons": ["Onsites keep exposing weak system design"]})
    session.add(Entity(slug="topic:job-search", kind="topic", name="Job search", aliases=[], dirty=True))
    for i in range(2):
        session.add(Episode(occurred_at=NOW - timedelta(days=i), kind="interview", summary=f"Onsite {i}",
                            entities=["topic:job-search"]))
    await session.commit()

    async def fake(prompt, system="", task="default"):
        ids = [e.id for e in (await session.execute(select(Episode))).scalars()]
        return {"digest": "Interviewing.", "reflections": [
            {"lesson": "Onsites keep exposing weak system design.", "evidence": ids, "confidence": 0.8},
            {"lesson": "Interviews cluster at the start of the week", "evidence": ids, "confidence": 0.6}]}
    monkeypatch.setattr(consolidate, "complete_json", fake)
    res = await consolidate.consolidate(session)
    lessons = [x.lesson for x in (await session.execute(select(Reflection))).scalars()]
    assert lessons == ["Interviews cluster at the start of the week"] and res["reflections"] == 1
