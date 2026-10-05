"""Overview, Topics, Queue, Recall inspector, Settings and Documents pages: render real data, and
every action they offer is a working API call."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.memory import facts
from src.memory.models import Entity, Episode, IngestJob, Reflection, SessionState
from src.models import ContextDocument, Source

NOW = datetime.now(timezone.utc)
LOCAL = {"host": "127.0.0.1:8001", "accept": "text/html"}


async def _seed(engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        s.add_all([
            Entity(slug="topic:job-search", kind="topic", name="Job search", aliases=[], dirty=False,
                   digest="Applying steadily; system design is the weak spot."),
            Entity(slug="project:voicediary", kind="project", name="VoiceDiary", aliases=[], dirty=False),
            Episode(occurred_at=NOW - timedelta(days=2), kind="applied", summary="Applied to Acme",
                    entities=["topic:job-search"], payload={"job": 1}),
            Episode(occurred_at=NOW - timedelta(days=1), kind="interview", summary="Acme onsite went poorly",
                    outcome="pending", entities=["topic:job-search"]),
            Episode(occurred_at=NOW, kind="milestone", summary="Released VoiceDiary 1.3.0",
                    entities=["project:voicediary"]),
            Reflection(lesson="System design is the weak spot", entities=["topic:job-search"],
                       evidence=[2], confidence=0.8),
            SessionState(session_id="s1", key="deadline", value="CV to Acme by Friday",
                         expires_at=NOW + timedelta(days=2)),
            IngestJob(id=1, content_hash="h0", text="applied to acme", status="done", attempts=1, occurred_at=NOW,
                      agent="claude-code", result={"episodes": 1, "facts": 0, "fact_ids": [], "entities": ["topic:job-search"]}),
            IngestJob(id=2, content_hash="h1", text="boom", status="failed", attempts=3, error="all down",
                      occurred_at=NOW, agent="claude-code"),
            ContextDocument(type="interview", slug="interview/acme", title="Acme onsite", tags=[], sections={},
                            source=Source.MANUAL, doc_metadata={"Company": "Acme"},
                            content="Company: Acme\n\nTake-home:\n- Deadline: Wednesday 8 Oct 2026.\n"),
        ])
        await s.commit()
    await facts.add_fact("Prefers Material Rounded icons", kind="preference", entities=["project:voicediary"])


async def test_every_page_renders(client, engine):
    await _seed(engine)
    for path in ("/memory", "/topics", "/topics/topic:job-search", "/queue", "/queue?status=failed", "/recall",
                 "/settings", "/admin", "/api/admin/interview/acme", "/api/admin/new?template=passport"):
        r = await client.get(path, headers=LOCAL, follow_redirects=True)
        assert r.status_code == 200, (path, r.text[:300])
    overview = (await client.get("/memory", headers=LOCAL)).text
    for expected in ("Job search", "Released VoiceDiary 1.3.0", "Untracked date in Acme onsite", "Thursday"):
        assert expected in overview, expected
    assert "CV to Acme by Friday" in (await client.get("/queue", headers=LOCAL)).text
    assert "Applying steadily" in (await client.get("/topics/topic:job-search", headers=LOCAL)).text
    assert (await client.get("/memory?tab=events", headers=LOCAL)).status_code == 301


async def test_user_text_is_escaped(client, engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        s.add(Episode(occurred_at=NOW, kind="other", summary="<script>alert(1)</script>", entities=[]))
        await s.commit()
    html = (await client.get("/memory", headers=LOCAL)).text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


async def test_track_date_rename_topic_undo_job_and_note(client, engine):
    await _seed(engine)
    r = await client.post("/api/documents/interview/acme/track-date", headers=LOCAL, json={"field": "Due", "date": "2026-10-07"})
    assert r.status_code == 200, r.text
    bad = await client.post("/api/documents/interview/acme/track-date", headers=LOCAL, json={"field": "Lunch", "date": "2026-10-07"})
    assert bad.status_code == 422 and "isn't a tracked field" in bad.json()["detail"]
    r = await client.patch("/api/topics/topic:job-search", headers=LOCAL, json={"name": "Career"})
    assert r.json()["name"] == "Career" and "Job search" in r.json()["aliases"]
    assert [t["slug"] for t in (await client.get("/api/topics?q=career", headers=LOCAL)).json()] == ["topic:job-search"]
    detail = (await client.get("/api/queue/jobs/1", headers=LOCAL)).json()
    assert detail["episodes"][0]["summary"] == "Applied to Acme"
    assert (await client.post("/api/queue/jobs/1/undo", headers=LOCAL)).json()["episodes"] == 1
    r = await client.request("DELETE", "/api/session-notes", headers=LOCAL, json={"session_id": "s1", "key": "deadline"})
    assert r.status_code == 200
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        doc = (await s.execute(select(ContextDocument).where(ContextDocument.slug == "interview/acme"))).scalar_one()
        assert doc.doc_metadata["Due"] == "2026-10-07"
        assert (await s.execute(select(Episode).where(Episode.summary == "Applied to Acme"))).first() is None
        assert (await s.get(IngestJob, 1)).status == "undone"


async def test_recall_inspector_explains_and_respects_settings(client, engine, monkeypatch):
    await _seed(engine)
    from src import settings_store
    monkeypatch.setitem(settings_store._cache, "recall", {"self_entity": "topic:job-search", "pinned_cap": 0.3})
    r = await client.post("/api/recall/inspect", headers=LOCAL, json={"situation": "how is my job search going", "budget_tokens": 800})
    assert r.status_code == 200, r.text
    x = r.json()
    assert "topic:job-search" not in x["plan"]["entities"] and x["plan"]["excluded"] == ["topic:job-search"]
    assert x["explain"]["pinned_budget"] == 240
    assert sum(s["tokens"] for s in x["explain"]["sections"] if s["pinned"]) <= 240
    assert all({"score", "rrf", "recency", "in_brief"} <= set(c) for c in x["explain"]["candidates"])
