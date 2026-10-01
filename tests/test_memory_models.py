from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from src.memory.models import Entity, Episode, IngestJob, MemoryBlock, Reflection, SessionState


async def test_models_roundtrip(session):
    now = datetime.now(timezone.utc)
    session.add_all([
        Entity(slug="company:acme", kind="company", name="Acme", aliases=["acme corp"]),
        Episode(occurred_at=now, kind="applied", summary="Applied to Acme backend role",
                entities=["company:acme", "topic:job-search"], importance=3),
        Reflection(lesson="Tailored CVs get more callbacks", entities=["topic:job-search"],
                   evidence=[1], confidence=0.7),
        SessionState(session_id="s1", key="deadline", value="Friday", expires_at=now + timedelta(days=7)),
        MemoryBlock(name="profile", content="Backend dev"),
        IngestJob(content_hash="abc", text="hello", occurred_at=now),
    ])
    await session.commit()

    ep = (await session.execute(select(Episode))).scalar_one()
    assert ep.entities == ["company:acme", "topic:job-search"]
    assert ep.access_count == 0
    job = (await session.execute(select(IngestJob))).scalar_one()
    assert job.status == "pending" and job.attempts == 0
    # generated FTS column works
    hit = (await session.execute(text(
        "select count(*) from episodes where search_vector @@ plainto_tsquery('english','backend')"
    ))).scalar()
    assert hit == 1


async def test_ingest_job_hash_unique(session):
    now = datetime.now(timezone.utc)
    session.add(IngestJob(content_hash="dup", text="a", occurred_at=now))
    await session.commit()
    session.add(IngestJob(content_hash="dup", text="a", occurred_at=now))
    with pytest.raises(IntegrityError):
        await session.commit()
