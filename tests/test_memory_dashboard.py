from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import async_sessionmaker

from src.memory import facts
from src.memory.models import Entity, Episode, IngestJob, Reflection, SessionState

NOW = datetime.now(timezone.utc)


async def _seed(engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        s.add_all([
            Entity(slug="topic:job-search", kind="topic", name="Job search", aliases=[], dirty=False,
                   digest="Applying steadily; system design is the weak spot."),
            Entity(slug="project:voicediary", kind="project", name="VoiceDiary", aliases=[], dirty=False),
            Episode(occurred_at=NOW - timedelta(days=2), kind="applied", summary="Applied to Acme",
                    entities=["topic:job-search"]),
            Episode(occurred_at=NOW - timedelta(days=1), kind="interview", summary="Acme onsite went poorly",
                    outcome="pending", entities=["topic:job-search"]),
            Episode(occurred_at=NOW, kind="milestone", summary="Released VoiceDiary 1.3.0",
                    entities=["project:voicediary"]),
            Reflection(lesson="System design is the weak spot", entities=["topic:job-search"],
                       evidence=[2], confidence=0.8),
            SessionState(session_id="s1", key="deadline", value="CV to Acme by Friday",
                         expires_at=NOW + timedelta(days=2)),
            IngestJob(content_hash="h1", text="boom", status="failed", attempts=3, error="all down",
                      occurred_at=NOW, agent="claude-code"),
        ])
        await s.commit()
    await facts.add_fact("Prefers Material Rounded icons", kind="preference", entities=["project:voicediary"])


async def test_overview_shows_every_layer(client, engine):
    await _seed(engine)
    r = await client.get("/memory")
    assert r.status_code == 200
    html = r.text
    for expected in ("Job search", "VoiceDiary", "<b>3</b> events", "<b>1</b> lesson<", "<b>1</b> failed",
                     "Applying steadily; system design is the weak spot."):
        assert expected in html, expected


async def test_tabs_and_filters(client, engine):
    await _seed(engine)
    events = (await client.get("/memory", params={"tab": "events", "entity": "topic:job-search"})).text
    assert "Applied to Acme" in events and "Released VoiceDiary" not in events
    assert "Released VoiceDiary" in (await client.get("/memory", params={"tab": "events", "q": "voicediary"})).text
    assert "Prefers Material Rounded icons" in (await client.get("/memory", params={"tab": "facts"})).text
    assert "System design is the weak spot" in (await client.get("/memory", params={"tab": "lessons"})).text
    assert "CV to Acme by Friday" in (await client.get("/memory", params={"tab": "notes"})).text
    queue = (await client.get("/memory", params={"tab": "queue"})).text
    assert "all down" in queue and "/api/memory/jobs/" in queue  # retry button wired


async def test_recall_preview_returns_brief_html(client, engine):
    await _seed(engine)
    r = await client.get("/memory/recall-preview", params={"situation": "how is my job search going"})
    assert r.status_code == 200 and "<pre" in r.text and "applied 1" in r.text


async def test_user_text_is_escaped(client, engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        s.add(Episode(occurred_at=NOW, kind="other", summary="<script>alert(1)</script>", entities=[]))
        await s.commit()
    html = (await client.get("/memory", params={"tab": "events"})).text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
