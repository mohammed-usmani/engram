"""JSON endpoints for evaluation: runs, golden questions, traces and feedback."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import AsyncSessionLocal, get_session
from src.eval import runner
from src.eval.models import EvalRun, GoldenQuestion, Trace

router = APIRouter(prefix="/api", tags=["evaluation"])
KINDS = ("health", "recall", "extract")


def run_out(r: EvalRun, details: bool = False) -> dict:
    out = {"id": r.id, "kind": r.kind, "trigger": r.trigger, "score": r.score, "summary": r.summary,
           "started_at": r.started_at, "finished_at": r.finished_at}
    if details:
        out["details"] = r.details
    return out


class RunIn(BaseModel):
    kind: str
    provider: str | None = None
    model: str | None = None
    wait: bool = False   # tests and the CLI wait; the page starts it and polls


@router.post("/evals/run")
async def start_run(body: RunIn, session: AsyncSession = Depends(get_session)):
    if body.kind not in KINDS:
        raise HTTPException(422, f"Unknown evaluation '{body.kind}'. Use one of: {', '.join(KINDS)}.")
    if body.kind in runner._running:
        raise HTTPException(409, f"A {body.kind} evaluation is already running. Wait for it to finish.")
    if body.kind == "extract" and body.provider:
        from src.memory import llm
        if not llm.key_source(body.provider):
            raise HTTPException(422, f"{body.provider} has no API key. Add it in Settings first.")
    if body.wait:
        return run_out(await runner.run(session, body.kind, "manual", body.provider, body.model), details=True)

    async def bg():
        async with AsyncSessionLocal() as s:
            await runner.run(s, body.kind, "manual", body.provider, body.model)
    asyncio.create_task(bg())
    return {"started": body.kind}


@router.get("/evals/status")
async def status():
    return {"running": sorted(runner._running)}


@router.get("/evals/runs")
async def list_runs(kind: str | None = None, limit: int = 30, session: AsyncSession = Depends(get_session)):
    q = select(EvalRun).order_by(EvalRun.id.desc()).limit(max(1, min(limit, 200)))
    if kind:
        q = q.where(EvalRun.kind == kind)
    return [run_out(r) for r in (await session.execute(q)).scalars()]


@router.get("/evals/runs/{run_id}")
async def get_run(run_id: int, session: AsyncSession = Depends(get_session)):
    r = await session.get(EvalRun, run_id)
    if r is None:
        raise HTTPException(404, "No such evaluation run.")
    return run_out(r, details=True)


# ---------------------------------------------------------------- golden questions

class QuestionIn(BaseModel):
    situation: str = Field(..., min_length=3, max_length=2000)
    must: list[str] = Field(default_factory=list, max_length=20)
    must_not: list[str] = Field(default_factory=list, max_length=20)
    budget: int = Field(1500, ge=200, le=8000)
    note: str | None = Field(None, max_length=500)
    trace_id: int | None = None
    active: bool = True


def q_out(q: GoldenQuestion) -> dict:
    return {"id": q.id, "situation": q.situation, "must": q.must, "must_not": q.must_not, "budget": q.budget,
            "note": q.note, "trace_id": q.trace_id, "active": q.active, "created_at": q.created_at}


def _clean(xs: list[str]) -> list[str]:
    return [x.strip() for x in xs if x and x.strip()]


@router.get("/evals/questions")
async def list_questions(session: AsyncSession = Depends(get_session)):
    return [q_out(q) for q in (await session.execute(select(GoldenQuestion).order_by(GoldenQuestion.id))).scalars()]


@router.post("/evals/questions")
async def add_question(body: QuestionIn, session: AsyncSession = Depends(get_session)):
    if not _clean(body.must) and not _clean(body.must_not):
        raise HTTPException(422, "Add at least one thing the brief must contain or must not contain, "
                                 "otherwise the question can never fail.")
    q = GoldenQuestion(situation=body.situation.strip(), must=_clean(body.must), must_not=_clean(body.must_not),
                       budget=body.budget, note=body.note, trace_id=body.trace_id, active=body.active)
    session.add(q)
    await session.commit()
    return q_out(q)


@router.put("/evals/questions/{qid}")
async def edit_question(qid: int, body: QuestionIn, session: AsyncSession = Depends(get_session)):
    q = await session.get(GoldenQuestion, qid)
    if q is None:
        raise HTTPException(404, "No such question.")
    if not _clean(body.must) and not _clean(body.must_not):
        raise HTTPException(422, "Keep at least one must or must-not item.")
    q.situation, q.must, q.must_not = body.situation.strip(), _clean(body.must), _clean(body.must_not)
    q.budget, q.note, q.active = body.budget, body.note, body.active
    await session.commit()
    return q_out(q)


@router.delete("/evals/questions/{qid}")
async def delete_question(qid: int, session: AsyncSession = Depends(get_session)):
    q = await session.get(GoldenQuestion, qid)
    if q is None:
        raise HTTPException(404, "No such question.")
    await session.delete(q)
    await session.commit()
    return {"deleted": qid}


@router.post("/evals/questions/{qid}/check")
async def check_question(qid: int, session: AsyncSession = Depends(get_session)):
    from src.eval.recall_eval import check
    q = await session.get(GoldenQuestion, qid)
    if q is None:
        raise HTTPException(404, "No such question.")
    return await check(session, q)


# ---------------------------------------------------------------- contradiction verdicts

class VerdictIn(BaseModel):
    a_id: str
    b_id: str
    contradict: bool


@router.post("/evals/health/verdict")
async def set_verdict(body: VerdictIn, session: AsyncSession = Depends(get_session)):
    """The user's call on a pair overrides the LLM's: 'not a conflict' stops it being reported."""
    from src import settings_store as store
    cache = store.get("health_verdicts") or {}
    cache[f"{body.a_id}|{body.b_id}"] = {"c": body.contradict, "why": "your decision", "user": True}
    await store.put(session, "health_verdicts", cache)
    return {"ok": True}


# ---------------------------------------------------------------- traces

def t_out(t: Trace, full: bool = False) -> dict:
    d = t.data or {}
    out = {"id": t.id, "kind": t.kind, "source": t.source, "input": t.input if full else (t.input or "")[:240],
           "flags": t.flags, "duration_ms": t.duration_ms, "feedback": t.feedback, "note": t.note,
           "created_at": t.created_at}
    if full:
        out["data"] = d
    elif t.kind == "recall":
        out["glance"] = {"items": len(d.get("items") or []), "used": d.get("used"), "pinned": d.get("pinned"),
                         "planner": (d.get("plan") or {}).get("planner")}
    else:
        out["glance"] = {"provider": d.get("provider"), "model": d.get("model"),
                         "episodes": len(d.get("episodes") or []), "facts": len(d.get("facts") or []),
                         "dropped": len(d.get("dropped") or []), "error": d.get("error")}
    return out


@router.get("/traces")
async def list_traces(kind: str | None = None, source: str | None = None, flag: str | None = None,
                      feedback: int | None = None, q: str | None = None, limit: int = 50, offset: int = 0,
                      session: AsyncSession = Depends(get_session)):
    stmt = select(Trace).order_by(Trace.id.desc()).limit(max(1, min(limit, 200))).offset(max(0, offset))
    if kind:
        stmt = stmt.where(Trace.kind == kind)
    if source:
        stmt = stmt.where(Trace.source.like(f"{source}%"))
    if flag:
        stmt = stmt.where(Trace.flags.op("?")(flag))
    if feedback is not None:
        stmt = stmt.where(Trace.feedback == feedback)
    if q:
        stmt = stmt.where(Trace.input.ilike(f"%{q}%"))
    return [t_out(t) for t in (await session.execute(stmt)).scalars()]


@router.get("/traces/stats")
async def trace_stats(days: int = 7, session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(sql("""
        select kind, source, count(*) n, percentile_cont(0.5) within group (order by duration_ms) p50,
               percentile_cont(0.95) within group (order by duration_ms) p95
        from eval_traces where created_at > now() - make_interval(days => :d) group by kind, source order by n desc"""),
        {"d": max(1, min(days, 30))})).mappings().all()
    flags = (await session.execute(sql("""
        select kind, f flag, count(*) n from eval_traces, jsonb_array_elements_text(flags) f
        where created_at > now() - make_interval(days => :d) group by kind, f order by n desc"""),
        {"d": max(1, min(days, 30))})).mappings().all()
    return {"by_source": [dict(r) for r in rows], "flags": [dict(r) for r in flags]}


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: int, session: AsyncSession = Depends(get_session)):
    t = await session.get(Trace, trace_id)
    if t is None:
        raise HTTPException(404, "No such trace (traces are kept 30 days).")
    return t_out(t, full=True)


class FeedbackIn(BaseModel):
    value: int = Field(..., ge=-1, le=1)   # 1 good, -1 bad, 0 clears
    note: str | None = Field(None, max_length=1000)


@router.post("/traces/{trace_id}/feedback")
async def feedback(trace_id: int, body: FeedbackIn, session: AsyncSession = Depends(get_session)):
    t = await session.get(Trace, trace_id)
    if t is None:
        raise HTTPException(404, "No such trace (traces are kept 30 days).")
    t.feedback = body.value or None
    if body.note is not None:
        t.note = body.note.strip() or None
    await session.commit()
    return {"id": t.id, "feedback": t.feedback, "note": t.note}
