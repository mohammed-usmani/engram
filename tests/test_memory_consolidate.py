import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from src.memory import consolidate, facts, worker
from src.memory.llm import LLMUnavailable
from src.memory.models import Entity, Episode, MemoryBlock, Reflection, SessionState
from src.services.ollama_client import embed

NOW = datetime.now(timezone.utc)


async def _seed(session):
    session.add(Entity(slug="topic:job-search", kind="topic", name="Job search", aliases=[], dirty=True))
    session.add(Entity(slug="company:solo", kind="company", name="Solo", aliases=[], dirty=True))
    for i, (kind, s) in enumerate([("applied", "Applied to Acme"), ("interview", "Acme onsite: system design weak"),
                                   ("interview", "Beta onsite: system design weak again")]):
        session.add(Episode(occurred_at=NOW - timedelta(days=5 - i), kind=kind, summary=s,
                            entities=["topic:job-search"], embedding=await embed(s)))
    session.add(Episode(occurred_at=NOW, kind="other", summary="Mentioned Solo once", entities=["company:solo"]))
    session.add(SessionState(session_id="old", key="format", value="PDF", expires_at=NOW - timedelta(hours=1)))
    session.add(SessionState(session_id="live", key="format", value="DOCX", expires_at=NOW + timedelta(days=1)))
    await session.commit()


async def test_consolidate(session, mem0_store, monkeypatch):
    await _seed(session)
    await facts.add_fact("Lives in Bangalore", kind="fact", entities=[], importance=5)
    await facts.add_fact("Likes dark mode", kind="preference", entities=[], importance=2)
    calls = []

    async def fake(prompt, system="", task="default"):
        calls.append(task)
        ids = [e.id for e in (await session.execute(select(Episode).where(Episode.kind == "interview"))).scalars()]
        return {"digest": "Applying for 5 days; system design is the recurring weak spot.",
                "reflections": [{"lesson": "System design is the weak spot in onsites", "evidence": ids, "confidence": 0.8},
                                {"lesson": "Unsupported hunch about things", "evidence": [ids[0]], "confidence": 0.9},
                                {"lesson": "a specific pattern you see in these events", "evidence": ids, "confidence": 0.5}]}
    monkeypatch.setattr(consolidate, "complete_json", fake)

    res = await consolidate.consolidate(session)
    js = (await session.execute(select(Entity).where(Entity.slug == "topic:job-search"))).scalar_one()
    solo = (await session.execute(select(Entity).where(Entity.slug == "company:solo"))).scalar_one()
    assert not js.dirty and js.digest.startswith("Applying for 5 days")  # prose only; numbers are live
    assert not solo.dirty and calls == ["consolidate"]        # single-episode entity needs no LLM
    refl = (await session.execute(select(Reflection))).scalars().all()
    assert [r.lesson for r in refl] == ["System design is the weak spot in onsites"]  # ≥2 evidence only
    prof = await session.get(MemoryBlock, "profile")
    assert prof.content.index("Bangalore") < prof.content.index("dark mode")
    notes = (await session.execute(select(SessionState))).scalars().all()
    assert [n.session_id for n in notes] == ["live"]
    assert (await session.execute(select(Episode).where(Episode.kind == "session_note"))).scalar_one()
    assert res["entities"] == 2 and res["reflections"] == 1

    # re-running with the same lesson updates instead of duplicating
    js.dirty = True
    await session.commit()
    await consolidate.consolidate(session)
    assert len((await session.execute(select(Reflection))).scalars().all()) == 1


async def test_consolidate_llm_down_keeps_dirty(session, mem0_store, monkeypatch):
    await _seed(session)

    async def down(prompt, system="", task="default"):
        raise LLMUnavailable("down")
    monkeypatch.setattr(consolidate, "complete_json", down)
    await consolidate.consolidate(session)
    js = (await session.execute(select(Entity).where(Entity.slug == "topic:job-search"))).scalar_one()
    assert js.dirty and js.digest is None  # retried next run; recall shows live numbers meanwhile


async def test_worker_processes_and_stops(monkeypatch):
    seen = []

    async def fake_pending(session, limit=5):
        seen.append("p")
        return 0

    async def fake_consolidate(session):
        seen.append("c")
        return {}
    monkeypatch.setattr(worker, "process_pending", fake_pending)
    monkeypatch.setattr(worker, "consolidate", fake_consolidate)

    async def no_batches(session):
        seen.append("b")
    monkeypatch.setattr(worker.batch, "tick", no_batches)
    stop = asyncio.Event()
    worker.request_consolidation()
    task = asyncio.create_task(worker.run_worker(stop, idle_sleep=0.01))
    await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(task, 2)
    assert "p" in seen and "c" in seen and "b" in seen
