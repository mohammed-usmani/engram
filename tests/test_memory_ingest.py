import pytest
from datetime import datetime, timezone

from sqlalchemy import select

from src.memory import extract, facts, ingest
from src.memory.models import Entity, Episode, IngestJob, SessionState

WHEN = datetime(2026, 9, 29, 18, tzinfo=timezone.utc)

EXTRACTION = {
    "entities": [{"name": "Acme", "kind": "company"}, {"name": "Job search", "kind": "topic"}],
    "episodes": [{"kind": "interview", "summary": "Had Acme onsite; system design round went poorly",
                  "when": "2026-09-29", "entities": ["Acme", "Job search"], "outcome": "pending",
                  "sentiment": -0.4, "importance": 4}],
    "facts": [{"text": "Wants backend or AI engineering roles", "kind": "preference",
               "entities": ["Job search"], "importance": 4}],
    "procedures": [],
    "session_notes": [{"key": "follow_up", "value": "Email Acme recruiter Friday"}],
}



@pytest.fixture(autouse=True)
def _placeholder_text_is_grounded(request, monkeypatch):
    """These tests feed placeholder text ("...") to check field clean-up; grounding has its own test."""
    if "recaps" not in request.node.name:
        from src.memory import extract as _ex
        monkeypatch.setattr(_ex, "_grounded", lambda ep, *a, **k: ep)

def _patch(monkeypatch, extraction=EXTRACTION, reconcile=None):
    async def fake_extract(prompt, system="", task="default"):
        return extraction
    monkeypatch.setattr(extract, "complete_json", fake_extract)

    async def fake_reconcile(prompt, system="", task="default"):
        return reconcile or []
    monkeypatch.setattr(ingest, "complete_json", fake_reconcile)


async def test_enqueue_is_idempotent(session):
    a, created_a = await ingest.enqueue(session, "same text", agent="claude-code", session_id="s1")
    b, created_b = await ingest.enqueue(session, "same text", agent="claude-code", session_id="s1")
    assert a == b and created_a and not created_b


async def test_process_job_writes_all_layers(session, mem0_store, monkeypatch):
    _patch(monkeypatch)
    job_id, _ = await ingest.enqueue(session, "I had my Acme onsite today...", agent="claude-code",
                                     session_id="s1", occurred_at=WHEN)
    assert await ingest.process_pending(session) == 1

    job = await session.get(IngestJob, job_id)
    assert job.status == "done", job.error
    ep = (await session.execute(select(Episode))).scalar_one()
    assert ep.kind == "interview" and set(ep.entities) == {"company:acme", "topic:job-search"}
    assert ep.source_agent == "claude-code" and ep.embedding is not None
    ents = (await session.execute(select(Entity))).scalars().all()
    assert all(e.dirty for e in ents)
    note = (await session.execute(select(SessionState))).scalar_one()
    assert note.key == "follow_up"
    found = await facts.search_facts("what roles am I targeting", kinds=["preference"])
    assert found and found[0]["metadata"]["entities"] == ["topic:job-search"]

    # the same event mentioned again in another session is not double-counted
    await ingest.enqueue(session, "Recap: Acme onsite happened", session_id="s2", occurred_at=WHEN)
    await ingest.process_pending(session)
    assert len((await session.execute(select(Episode))).scalars().all()) == 1


async def test_reconcile_supersedes(session, mem0_store, monkeypatch):
    old = await facts.add_fact("Prefers NestJS for backend work", kind="preference", entities=["tech:nestjs"])
    newer = {**EXTRACTION, "episodes": [], "session_notes": [],
             "facts": [{"text": "Now prefers FastAPI over NestJS for backend work", "kind": "preference",
                        "entities": [], "importance": 3}]}
    _patch(monkeypatch, extraction=newer, reconcile=[{"pair": 0, "verdict": "supersedes"}])
    await ingest.enqueue(session, "switched to FastAPI", occurred_at=WHEN)
    await ingest.process_pending(session)
    ids = [f["id"] for f in await facts.search_facts("backend framework preference")]
    assert old not in ids
    assert (await facts.get_fact(old))["metadata"]["superseded_by"] in ids


async def test_bad_payload_retries_then_fails(session, monkeypatch):
    # (a provider *outage* doesn't burn attempts — see test_outage_does_not_consume_attempts)
    async def down(prompt, system="", task="default"):
        raise ValueError("all down")
    monkeypatch.setattr(extract, "complete_json", down)
    job_id, _ = await ingest.enqueue(session, "keep me", occurred_at=WHEN)
    for _ in range(3):
        await ingest.process_job(session, await session.get(IngestJob, job_id))
    job = await session.get(IngestJob, job_id)
    assert job.status == "failed" and job.attempts == 3 and job.text == "keep me"
    assert "all down" in job.error


async def test_note_upserts(session):
    await ingest.note(session, "s9", "format", "PDF")
    await ingest.note(session, "s9", "format", "DOCX")
    rows = (await session.execute(select(SessionState))).scalars().all()
    assert [(r.key, r.value) for r in rows] == [("format", "DOCX")]


async def test_split_events_for_different_companies_are_not_deduped(session, mem0_store, monkeypatch):
    two = {"entities": [{"name": "Zomato", "kind": "company"}, {"name": "Swiggy", "kind": "company"}],
           "episodes": [{"kind": "applied", "summary": "Applied to Zomato and Swiggy for SDE-2 roles",
                         "entities": ["Zomato", "Swiggy"], "when": "2026-09-28"}],
           "facts": [], "procedures": [], "session_notes": []}
    _patch(monkeypatch, extraction=two)
    await ingest.enqueue(session, "applied to zomato and swiggy", occurred_at=WHEN)
    await ingest.process_pending(session)
    eps = (await session.execute(select(Episode))).scalars().all()
    assert sorted(tuple(sorted(e.entities)) for e in eps) == [
        ("company:swiggy", "topic:job-search"), ("company:zomato", "topic:job-search")]


async def test_one_failed_job_does_not_break_the_rest_of_the_batch(session, mem0_store, monkeypatch):
    _patch(monkeypatch)
    real = extract.complete_json

    async def first_fails(prompt, system="", task="default"):
        if "broken" in prompt:
            raise ValueError("bad output")
        return await real(prompt, system, task)
    monkeypatch.setattr(extract, "complete_json", first_fails)
    bad, _ = await ingest.enqueue(session, "broken payload", agent="t", occurred_at=WHEN)
    good, _ = await ingest.enqueue(session, "I had my Acme onsite today", agent="t", occurred_at=WHEN)
    assert await ingest.process_pending(session, limit=2) == 2
    assert (await session.get(IngestJob, bad)).status == "pending"
    assert (await session.get(IngestJob, good)).status == "done"


async def test_worker_slots_extract_concurrently(engine, mem0_store, monkeypatch):
    import asyncio
    import time
    from sqlalchemy.ext.asyncio import async_sessionmaker
    _patch(monkeypatch)
    real = extract.complete_json

    async def slow(prompt, system="", task="default"):
        await asyncio.sleep(0.5)
        return await real(prompt, system, task)
    monkeypatch.setattr(extract, "complete_json", slow)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        for i in range(4):
            await ingest.enqueue(s, f"job {i}: I had my Acme onsite", agent="t", occurred_at=WHEN)

    async def slot():
        async with Session() as s:
            return await ingest.process_pending(s)
    t = time.monotonic()
    assert sum(await asyncio.gather(*(slot() for _ in range(4)))) == 4  # each slot took a different job
    assert time.monotonic() - t < 1.5  # four 0.5 s extractions overlapped, not 2 s in a row
    async with Session() as s:
        assert (await s.execute(select(IngestJob.status))).scalars().all() == ["done"] * 4


async def test_reapplying_to_the_same_company_within_60_days_is_one_application(session):
    from datetime import datetime, timedelta, timezone
    from src.memory.ingest import _is_duplicate_episode
    from src.memory.models import Episode
    now = datetime(2026, 10, 6, tzinfo=timezone.utc)
    session.add(Episode(occurred_at=now - timedelta(days=15), kind="applied", summary="Applied to Acme",
                        entities=["company:acme", "topic:job-search"], embedding=[0.1] * 768))
    await session.commit()
    vec = [0.2] * 768
    assert await _is_duplicate_episode(session, "applied", now, vec, ["company:acme", "topic:job-search"])
    assert not await _is_duplicate_episode(session, "applied", now, vec, ["company:globex", "topic:job-search"])
    assert not await _is_duplicate_episode(session, "applied", now + timedelta(days=90), vec, ["company:acme"])
