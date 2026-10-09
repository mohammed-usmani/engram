"""Evaluation and Traces pages: render empty and with data, escape what users and models wrote."""
from datetime import datetime, timezone

from src.eval.models import EvalRun, GoldenQuestion, Trace
from src.memory import llm

LOCAL = {"host": "127.0.0.1:8001"}
NOW = datetime.now(timezone.utc)
XSS = "<script>alert(1)</script>"


async def test_pages_render_empty(client):
    ev = await client.get("/evals", headers=LOCAL)
    assert ev.status_code == 200, ev.text
    assert "No golden questions yet" in ev.text and "No health run yet" in ev.text and "No extraction run yet" in ev.text
    assert 'href="/evals"' in ev.text and 'href="/traces"' in ev.text
    tr = await client.get("/traces", headers=LOCAL)
    assert tr.status_code == 200 and "No traces yet" in tr.text
    assert (await client.get("/traces/999", headers=LOCAL)).status_code == 404


async def seed(session):
    contradiction = {"a": "CGPA is 8.0", "b": "CGPA is 8.27", "a_id": "aaa", "b_id": "bbb", "similarity": 0.9,
                     "why": "different numbers", "verdict": True, "judge": "two CGPAs for one degree"}
    q = GoldenQuestion(situation="what is my CGPA", must=["8.27"], must_not=["8.0"])
    session.add(q)
    await session.flush()
    session.add_all([
        EvalRun(kind="health", score=95, summary={"ok": 2}, finished_at=NOW, details=[]),
        EvalRun(kind="health", score=80, summary={"ok": 1, "warn": 0, "bad": 1}, finished_at=NOW, details=[
            {"id": "contradictions", "title": "Facts that contradict each other", "status": "bad", "value": 1,
             "detail": "1 confirmed", "items": [contradiction], "href": "/facts"},
            {"id": "two_names", "title": "Same thing under two topic names", "status": "ok", "value": 0,
             "detail": "0 groups", "items": [{"names": ["Redfox", "Redfox Cyber"]}], "href": "/cleanup#merge"}]),
        EvalRun(kind="recall", score=0, finished_at=NOW, summary={"questions": 1, "passed": 0, "must_hit": 0,
                "forbidden_hits": 1, "pinned_share": 10, "median_ms": 40, "max_ms": 40},
                details=[{"id": q.id, "situation": q.situation, "passed": False, "missing": ["8.27"], "forbidden": ["8.0"],
                          "must": ["8.27"], "must_not": ["8.0"], "used": 100, "pinned": 10, "items": 3, "ms": 40,
                          "trace_id": None}]),
        EvalRun(kind="extract", score=50, finished_at=NOW, summary={"provider": "ollama", "model": "qwen", "passed": 1,
                "total": 2, "errors": 0, "median_s": 1.5, "max_s": 2},
                details=[{"case": "REG plan is not an application", "passed": 1, "total": 2, "failed": ["no applied event"],
                          "error": None, "seconds": 1.5,
                          "output": {"episodes": ["applied 2026-10-01 " + XSS], "facts": [], "dropped": []}}]),
        Trace(kind="recall", source="mcp:claude", input="prep me " + XSS, flags=["empty_brief"], duration_ms=120,
              data={"budget": 1500, "used": 40, "pinned": 30, "items": ["e:7", "f:0a1b2c3d-1111"],
              "plan": {"entities": ["company:acme"], "excluded": [], "weights": {"episode": 1.0}, "planner": "off"},
              "sections": [{"title": "Profile", "tokens": 30, "pinned": True}, {"title": "History", "tokens": 10, "pinned": False}],
              "timings": {"embed_ms": 5, "plan_ms": 0, "pinned_ms": 3, "rank_ms": 9},
              "brief": "## History\n- [e:7] Applied to Acme\n- [f:0a1b2c3d-1111] CGPA is 8.27"}),
        Trace(kind="extract", source="worker", input="Applied to Globex", flags=["dropped_events"], duration_ms=900,
              data={"job_id": 12, "attempt": 1, "batched": False, "provider": "ollama", "model": "qwen",
                    "provider_errors": [], "error": None, "episodes": [{"kind": "applied", "summary": "Applied to Globex", "date": "2026-10-01"}],
                    "facts": ["Lives in Pune"], "dropped": [{"summary": "Invented interview", "kind": "interview", "reason": "no date in text"}],
                    "stored": {"episodes": 1, "facts": 1}}),
        Trace(kind="recall", source="eval", input="golden run situation", data={}, duration_ms=10),
    ])
    await session.commit()


async def test_evals_page_with_runs_and_questions(client, session):
    await seed(session)
    r = await client.get("/evals", headers=LOCAL)
    assert r.status_code == 200, r.text
    t = r.text
    assert "two CGPAs for one degree" in t and "Not a conflict" in t and "/api/evals/health/verdict" in t
    assert "/facts?q=CGPA%20is%208.0" in t
    assert "Missing: 8.27" in t and "Found forbidden: 8.0" in t and "/recall?situation=what%20is%20my%20CGPA" in t
    assert "REG</span> plan is not an application" in t and "no applied event" in t
    assert XSS not in t and "&lt;script&gt;" in t
    assert 'aria-label="Last 2 scores: 95, 80"' in t   # trend is oldest to newest from real runs


async def test_traces_list_filters_and_escapes(client, session):
    await seed(session)
    t = (await client.get("/traces", headers=LOCAL)).text
    assert "prep me &lt;script&gt;" in t and XSS not in t
    assert "golden run situation" not in t        # eval traces hidden unless chosen
    assert "golden run situation" in (await client.get("/traces?source=eval", headers=LOCAL)).text
    only = (await client.get("/traces?kind=extract", headers=LOCAL)).text
    assert "Applied to Globex" in only and "prep me" not in only
    assert "prep me" in (await client.get("/traces?flag=empty_brief", headers=LOCAL)).text
    assert "Applied to Globex" not in (await client.get("/traces?flag=empty_brief", headers=LOCAL)).text


async def test_trace_detail_pages(client, session):
    await seed(session)
    ids = {t["input"][:7]: t["id"] for t in (await client.get("/api/traces?limit=10", headers=LOCAL)).json()}
    rec = (await client.get(f"/traces/{ids['prep me']}", headers=LOCAL)).text
    assert XSS not in rec and "&lt;script&gt;" in rec
    assert 'href="/events?focus=7"' in rec and "/facts?q=CGPA+is+8.27" in rec
    assert "Save as golden question" in rec and "company:acme" in rec and "cut from" not in rec
    ext = (await client.get(f"/traces/{ids['Applied']}", headers=LOCAL)).text
    assert "/queue#job-12" in ext and "no date in text" in ext and "Lives in Pune" in ext


async def test_ask_returns_recall_trace_id(client, monkeypatch):
    class Fake:
        async def generate(self, message, system="", history=None, json=False):
            return "nothing in memory"
    monkeypatch.setattr(llm, "make", lambda provider, model=None: Fake())
    r = await client.post("/api/ask", headers=LOCAL, json={"message": "anything?", "provider": "ollama", "model": "m"})
    assert r.status_code == 200, r.text
    tid = r.json()["trace_id"]
    assert (await client.get(f"/api/traces/{tid}", headers=LOCAL)).json()["source"] == "ask"
