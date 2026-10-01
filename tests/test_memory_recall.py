"""Golden recall test: real Ollama embeddings, no LLM (fast=True)."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from src.memory import facts, local_date, recall
from src.memory.models import Entity, Episode, MemoryBlock, Reflection, SessionState
from src.services.ollama_client import embed

START = datetime.now(timezone.utc) - timedelta(days=16)
COMPANIES = ["Acme", "Beta Labs", "Gamma AI", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]


async def _seed(session):
    js = Entity(slug="topic:job-search", kind="topic", name="Job search", aliases=[],
                embedding=await embed("topic: Job search"), dirty=False,
                digest="Job search since 16 days ago; DSA rounds go well, system design is the weak spot.")
    acme = Entity(slug="company:acme", kind="company", name="Acme", aliases=[], embedding=await embed("company: Acme"))
    fit = Entity(slug="topic:fitness", kind="topic", name="Fitness", aliases=[], embedding=await embed("topic: Fitness"))
    session.add_all([js, acme, fit])
    for i, c in enumerate(COMPANIES):
        s = f"Applied to {c} for a backend engineer role"
        session.add(Episode(occurred_at=START + timedelta(days=i), kind="applied", summary=s, importance=3,
                            entities=["topic:job-search"] + (["company:acme"] if c == "Acme" else []),
                            embedding=await embed(s)))
    for d, s, out, ents in [
        (12, "Acme onsite: system design round went poorly", "pending", ["topic:job-search", "company:acme"]),
        (13, "Beta Labs recruiter screen went fine", "rejected", ["topic:job-search"]),
    ]:
        session.add(Episode(occurred_at=START + timedelta(days=d), kind="interview", summary=s, outcome=out,
                            importance=4, entities=ents, embedding=await embed(s)))
    s = "Went for a 5k run"
    session.add(Episode(occurred_at=START, kind="milestone", summary=s, entities=["topic:fitness"], embedding=await embed(s)))
    lesson = "System design is the weak spot: flagged in both onsite interviews"
    session.add(Reflection(lesson=lesson, entities=["topic:job-search"], evidence=[11, 12], confidence=0.8,
                           embedding=await embed(lesson)))
    session.add(MemoryBlock(name="profile", content="Backend/AI engineer in Bangalore. Prefers concise answers."))
    session.add(SessionState(session_id="s1", key="deadline", value="CV to Acme by Friday",
                             expires_at=datetime.now(timezone.utc) + timedelta(days=2)))
    await session.commit()
    await facts.add_fact("Targets backend and AI engineering roles, remote or Bangalore", kind="preference",
                         entities=["topic:job-search"], importance=4)
    await facts.add_fact("How to apply for a job: 1. cv send <company> → 2. log it in applications.md → 3. follow up in 7 days",
                         kind="procedure", entities=["topic:job-search"], importance=4)


async def test_job_search_recall(session, mem0_store):
    await _seed(session)
    out = await recall.recall(session, "I'm applying for jobs", budget_tokens=1500, session_id="s1", fast=True)
    brief = out["brief"]
    assert "topic:job-search" in out["plan"]["entities"]
    assert "applied 10" in brief and "interview 2" in brief
    assert local_date(START) in brief                             # first date, user's timezone
    assert "rejected 1" in brief                                   # outcome breakdown
    assert "System design is the weak spot" in brief               # reflection
    assert "Bangalore" in brief                                    # profile
    assert "CV to Acme by Friday" in brief                         # session state
    assert "cv send" in brief                                      # procedure
    assert "5k run" not in brief                                   # unrelated area stays out
    assert recall.estimate_tokens(brief) <= 1500
    ep = (await session.execute(select(Episode).where(Episode.kind == "interview").order_by(Episode.id))).scalars().first()
    assert ep.access_count >= 1                                    # served items learn


async def test_graph_expansion_and_budget(session, mem0_store):
    await _seed(session)
    out = await recall.recall(session, "prepping for my Acme interview tomorrow", budget_tokens=300, fast=True)
    assert {"company:acme", "topic:job-search"} <= set(out["plan"]["entities"])
    assert recall.estimate_tokens(out["brief"]) <= 300
    assert "Bangalore" in out["brief"]  # pinned profile survives a tight budget


async def test_empty_store(session, mem0_store):
    out = await recall.recall(session, "anything at all", fast=True)
    assert out["brief"] and out["items"] == []


async def test_expand_timeline_forget(session, mem0_store):
    await _seed(session)
    tl = await recall.timeline(session, "topic:job-search")
    assert len(tl) == 12 and tl[0]["occurred_at"] <= tl[-1]["occurred_at"]
    first = f"e:{tl[0]['id']}"
    assert (await recall.expand(session, first))["summary"].startswith("Applied to Acme")
    assert await recall.forget(session, first) is True
    assert await recall.expand(session, first) is None
