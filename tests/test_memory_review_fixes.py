"""Regression tests for the final-review findings (#1–#8)."""
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text

from src.memory import api, extract, facts, ingest, recall
from src.memory.llm import LLMUnavailable
from src.memory.models import Episode, IngestJob, SessionState

WHEN = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)


# #1 — token protects the whole app once tunnelled; local loopback UI still works
async def test_auth_covers_every_path_for_forwarded_requests(monkeypatch):
    from main import app
    monkeypatch.setattr(api.settings, "admin_token", "sekret")
    fwd = {"X-Forwarded-For": "203.0.113.9"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.get("/api/chat/sessions", headers=fwd)).status_code == 401
        assert (await c.get("/api/settings/providers", headers=fwd)).status_code == 401
        assert (await c.get("/api/health", headers=fwd)).status_code == 200
        assert (await c.post("/api/memory/recall", json={"situation": "x"}, headers=fwd)).status_code == 401
        assert (await c.post("/mcp", json={}, headers=fwd)).status_code == 401
    local = {b"host": b"127.0.0.1:8001"}
    assert api.needs_token("/api/memories/3", local, ("127.0.0.1", 5)) is False  # local admin UI / local Claude Code
    assert api.needs_token("/mcp", local, ("127.0.0.1", 5)) is False
    assert api.needs_token("/mcp", {b"host": b"evil.example"}, ("127.0.0.1", 5)) is True  # DNS rebinding
    assert api.needs_token("/api/memories/3", {**local, b"cf-connecting-ip": b"1.2.3.4"}, ("127.0.0.1", 5)) is True
    assert api.needs_token("/admin", local, ("192.168.1.5", 5)) is True
    assert api.needs_token("/api/health", {}, ("8.8.8.8", 5)) is False


def _fail_extract(monkeypatch, exc):
    async def down(prompt, system="", task="default"):
        raise exc
    monkeypatch.setattr(extract, "complete_json", down)


# #2 — a crash mid-job leaves 'processing'; it must come back
async def test_stuck_processing_jobs_are_requeued(session, monkeypatch):
    _fail_extract(monkeypatch, RuntimeError("boom"))
    job_id, _ = await ingest.enqueue(session, "stuck", occurred_at=WHEN)
    await session.execute(text("UPDATE ingest_jobs SET status='processing', updated_at=now()-interval '20 minutes'"))
    await session.commit()
    assert await ingest.process_pending(session) == 1
    job = await session.get(IngestJob, job_id)
    await session.refresh(job)
    assert job.attempts == 1


# #3 — a retry after partial failure must not duplicate episodes/facts
async def test_retry_after_partial_failure_does_not_duplicate(session, mem0_store, monkeypatch):
    calls = {"n": 0}

    async def fake(prompt, system="", task="default"):
        calls["n"] += 1
        return {"entities": [], "episodes": [{"kind": "applied", "summary": f"Applied to Acme (v{calls['n']})",
                                              "entities": [], "when": "2026-09-29"}],
                "facts": [{"text": "Likes Rust", "kind": "fact"}, {"text": "Lives in Pune", "kind": "fact"}]}
    monkeypatch.setattr(extract, "complete_json", fake)
    real_add = facts.add_fact
    state = {"fail": True}

    async def flaky_add(text_, **kw):
        if text_ == "Lives in Pune" and state["fail"]:
            state["fail"] = False
            raise RuntimeError("embed timeout")
        return await real_add(text_, **kw)
    monkeypatch.setattr(facts, "add_fact", flaky_add)

    job_id, _ = await ingest.enqueue(session, "applied to acme", occurred_at=WHEN)
    job = await session.get(IngestJob, job_id)
    await ingest.process_job(session, job)                      # fails after episode + first fact
    await ingest.process_job(session, await session.get(IngestJob, job_id))
    assert (await session.get(IngestJob, job_id)).status == "done"
    assert len((await session.execute(select(Episode))).scalars().all()) == 1
    assert sorted(f["memory"] for f in await facts.all_facts()) == ["Likes Rust", "Lives in Pune"]


# #4 — dates resolve in the user's timezone, even in the early-morning IST window
def test_when_uses_user_timezone_for_date_only_and_naive():
    ref = datetime(2026, 9, 30, 20, 30, tzinfo=timezone.utc)   # 02:00 IST on 1 Oct
    from src.memory import local_date
    assert local_date(extract._when("2026-09-30", ref)) == "2026-09-30"
    naive = extract._when("2026-09-30T18:00:00", ref)
    assert naive.astimezone(timezone.utc).hour == 12 and naive.astimezone(timezone.utc).minute == 30


async def test_naive_occurred_at_is_user_local(session):
    job_id, _ = await ingest.enqueue(session, "x", occurred_at=datetime(2026, 9, 30, 18, 0))
    job = await session.get(IngestJob, job_id)
    assert job.occurred_at.astimezone(timezone.utc).hour == 12


# #5 — same sentence on another day is a new memory
async def test_same_text_different_day_is_new_job(session):
    a, _ = await ingest.enqueue(session, "Went to the gym today", occurred_at=WHEN)
    b, created = await ingest.enqueue(session, "Went to the gym today", occurred_at=WHEN + timedelta(days=1))
    assert created and a != b


# #6 — secrets never hit the DB, more formats caught, no epoch false positive
async def test_redaction_before_storage(session):
    await ingest.enqueue(session, 'my password is hunter2 and "api_key": "abc123def"', occurred_at=WHEN)
    await ingest.note(session, "s", "db", "postgres://u:supersecret@host/db")
    job = (await session.execute(select(IngestJob))).scalar_one()
    note = (await session.execute(select(SessionState))).scalar_one()
    assert "hunter2" not in job.text and "abc123def" not in job.text
    assert "supersecret" not in note.value


def test_redact_more_formats():
    for s in ["AKIAIOSFODNN7EXAMPLE", "xoxb-1234567890-abcdefghij", "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----",
              "4111 1111 1111 1111"]:
        assert "[REDACTED]" in extract.redact(s), s
    assert extract.redact("at 1727712000000 ms") == "at 1727712000000 ms"  # epoch ms is not a card


# #7 — an LLM outage doesn't burn attempts; failed jobs can be retried
async def test_outage_does_not_consume_attempts(session, monkeypatch):
    _fail_extract(monkeypatch, LLMUnavailable("offline"))
    job_id, _ = await ingest.enqueue(session, "offline memory", occurred_at=WHEN)
    for _ in range(5):
        await ingest.process_job(session, await session.get(IngestJob, job_id))
    job = await session.get(IngestJob, job_id)
    assert job.status == "pending" and job.attempts == 0 and job.result.get("_outages") == 5
    job.status, job.attempts = "failed", 3
    await session.commit()
    assert await ingest.retry_job(session, job_id) is True
    await session.refresh(job)
    assert job.status == "pending" and job.attempts == 0


# #8 — fast recall survives Ollama being down
async def test_recall_without_embeddings(session, monkeypatch):
    session.add(Episode(occurred_at=WHEN, kind="applied", summary="Applied to Acme", entities=["topic:job-search"]))
    from src.memory.models import Entity
    session.add(Entity(slug="topic:job-search", kind="topic", name="Job search", aliases=[]))
    await session.commit()

    async def down(text_):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(recall, "embed", down)

    async def no_facts(q, top_k=15):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(facts, "similar", no_facts)
    out = await recall.recall(session, "how is my job search going", fast=True)
    assert "applied 1" in out["brief"]
