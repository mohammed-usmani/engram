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
