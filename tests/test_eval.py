"""Evaluation: traces for every recall and extraction, health checks, golden questions, extraction test set."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.eval.models import EvalRun, Trace
from src.memory.models import Entity, Episode, IngestJob

LOCAL = {"host": "127.0.0.1:8001"}
NOW = datetime.now(timezone.utc)


async def _seed(engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        s.add_all([
            Entity(slug="topic:job-search", kind="topic", name="Job search", aliases=[], dirty=False),
            Episode(occurred_at=NOW - timedelta(days=1), kind="applied", summary="Applied to Acme",
                    entities=["topic:job-search"]),
        ])
        await s.commit()


def test_number_matching_is_exact():
    from src.eval.recall_eval import _contains
    assert _contains("CGPA 8.27/10", "8.27") and not _contains("CGPA 8.27/10", "8.2")
    assert not _contains("CGPA 8.07", "8.0") and _contains("CGPA of 8.0.", "8.0")
    assert _contains("Due:  2026-10-07", "due: 2026-10-07")


async def test_every_recall_is_traced_with_source_and_flags(client, engine, mem0_store):
    await _seed(engine)
    r = await client.post("/api/memory/recall", headers=LOCAL,
                          json={"situation": "job search status", "fast": True, "agent": "claude-code-hook"})
    tid = r.json()["trace_id"]
    t = (await client.get(f"/api/traces/{tid}", headers=LOCAL)).json()
    assert t["kind"] == "recall" and t["source"] == "rest:claude-code-hook"
    assert {"plan", "items", "sections", "timings", "brief"} <= set(t["data"])
    assert t["data"]["plan"]["planner"] == "off"
    listed = (await client.get("/api/traces?kind=recall", headers=LOCAL)).json()
    assert listed[0]["id"] == tid and "glance" in listed[0]
    fb = await client.post(f"/api/traces/{tid}/feedback", headers=LOCAL, json={"value": -1, "note": "missed Acme"})
    assert fb.json()["feedback"] == -1
    assert (await client.get("/api/traces?feedback=-1", headers=LOCAL)).json()[0]["id"] == tid


async def test_extraction_is_traced_with_drops(session, mem0_store, monkeypatch):
    from src.memory import extract, ingest
    data = {"entities": [{"name": "Globex", "kind": "company"}],
            "episodes": [{"kind": "applied", "summary": "Applied to Globex", "when": "2026-09-01", "entities": ["Globex"]}]}

    async def fake(prompt, system="", task="default"):
        return data
    monkeypatch.setattr(extract, "complete_json", fake)

    async def no_reconcile(*a, **k):
        return []
    monkeypatch.setattr(ingest, "complete_json", no_reconcile)
    job = IngestJob(content_hash="x1", text="Globex — Backend Engineer. Strong fit, tailored resume.",
                    occurred_at=NOW, agent="chatgpt", result={})
    session.add(job)
    await session.commit()
    await ingest.process_job(session, job)
    t = (await session.execute(select(Trace).where(Trace.kind == "extract"))).scalar_one()
    assert t.source == "chatgpt" and "dropped_events" in t.flags
    assert t.data["dropped"][0]["reason"] == "application not stated in text"


async def test_health_and_recall_runs_are_kept(client, engine, mem0_store, monkeypatch):
    await _seed(engine)
    from src.eval import health

    async def no_judge(session, pairs):
        for p in pairs:
            p["verdict"] = None
    monkeypatch.setattr(health, "_judge", no_judge)
    r = (await client.post("/api/evals/run", headers=LOCAL, json={"kind": "health", "wait": True})).json()
    assert 0 <= r["score"] <= 100 and {c["id"] for c in r["details"]} >= {"contradictions", "duplicate_events", "backup"}

    bad = await client.post("/api/evals/questions", headers=LOCAL, json={"situation": "anything"})
    assert bad.status_code == 422 and "must" in bad.json()["detail"]
    q = (await client.post("/api/evals/questions", headers=LOCAL,
                           json={"situation": "how is my job search going", "must": ["Acme"], "must_not": ["Globex"]})).json()
    one = (await client.post(f"/api/evals/questions/{q['id']}/check", headers=LOCAL)).json()
    assert one["passed"] and one["trace_id"]
    r = (await client.post("/api/evals/run", headers=LOCAL, json={"kind": "recall", "wait": True})).json()
    assert r["score"] == 100.0 and r["summary"]["questions"] == 1
    runs = (await client.get("/api/evals/runs", headers=LOCAL)).json()
    assert [x["kind"] for x in runs][:2] == ["recall", "health"]
    # recall evaluations don't count as items being served
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        assert (await s.execute(select(Episode.access_count))).scalar_one() == 0


async def test_extraction_set_runs_against_a_forced_model(monkeypatch):
    from src.eval import extraction
    from src.memory import extract, llm
    seen = []

    async def fake(prompt, system="", task="default"):
        seen.append(llm.steps(task))
        return {"episodes": [], "facts": []}
    monkeypatch.setattr(extract, "complete_json", fake)
    score, results, summary = await extraction.run("ollama", "tiny-model")
    assert seen and all(s == [("ollama", "tiny-model")] for s in seen)
    assert llm.FORCE.get() is None                     # live configuration untouched afterwards
    assert 0 < score < 100 and summary["total"] == sum(r["total"] for r in results)
    assert any(r["case"].startswith("REG") for r in results)


async def test_unknown_eval_and_bad_question(client):
    r = await client.post("/api/evals/run", headers=LOCAL, json={"kind": "vibes"})
    assert r.status_code == 422 and "health" in r.json()["detail"]
    assert (await client.get("/api/traces/999999", headers=LOCAL)).status_code == 404
