"""Evaluation and Traces pages. They only read; every change goes through src/eval/api.py."""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select, text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.eval import runner
from src.eval.api import t_out
from src.eval.models import GoldenQuestion, Trace
from src.memory import llm
from src.ui.ask import _composer, link_for
from src.ui.templating import render

router = APIRouter(include_in_schema=False)
BRIEF_LINE = re.compile(r"^- \[([^\]]+)\] (.*)$", re.M)


@router.get("/evals", response_class=HTMLResponse)
async def evals_page(request: Request, session: AsyncSession = Depends(get_session)):
    from src.eval.extraction import CASES
    layers = {}
    for kind in ("health", "recall", "extract"):
        runs = await runner.latest(session, kind, 20)
        layers[kind] = {"runs": runs, "last": runs[0] if runs else None,
                        "trend": [r.score for r in reversed(runs) if r.score is not None]}
    questions = [{"id": q.id, "situation": q.situation, "must": q.must, "must_not": q.must_not, "budget": q.budget,
                  "note": q.note, "trace_id": q.trace_id, "active": q.active}
                 for q in (await session.execute(select(GoldenQuestion).order_by(GoldenQuestion.id))).scalars()]
    last_recall = layers["recall"]["last"]
    results = {d["id"]: d for d in (last_recall.details if last_recall else []) if isinstance(d, dict)}
    steps = llm.steps("extract")
    return await render(request, "evals.html", session, "evals", layers=layers, questions=questions,
                        results=results, running=sorted(runner._running), cases=len(CASES),
                        composer=await _composer(session, steps[0] if steps else None))


@router.get("/traces", response_class=HTMLResponse)
async def traces_page(request: Request, kind: str = "", source: str = "", flag: str = "", feedback: str = "",
                      q: str = "", limit: int = 50, session: AsyncSession = Depends(get_session)):
    from src.eval.api import trace_stats
    limit = max(1, min(limit, 1000))
    stmt = select(Trace).order_by(Trace.id.desc()).limit(limit + 1)
    if kind:
        stmt = stmt.where(Trace.kind == kind)
    # eval runs make dozens of traces per run; they'd bury real traffic unless asked for
    stmt = stmt.where(Trace.source.like(f"{source}%")) if source else stmt.where(Trace.source != "eval")
    if flag:
        stmt = stmt.where(Trace.flags.op("?")(flag))
    if feedback in ("1", "-1"):
        stmt = stmt.where(Trace.feedback == int(feedback))
    if q:
        stmt = stmt.where(Trace.input.ilike(f"%{q}%"))
    rows = [t_out(t) for t in (await session.execute(stmt)).scalars()]
    sources = (await session.scalars(sql("select distinct source from eval_traces order by source"))).all()
    return await render(request, "traces.html", session, "traces", traces=rows[:limit], more=len(rows) > limit,
                        stats=await trace_stats(7, session), sources=sources, limit=limit,
                        f={"kind": kind, "source": source, "flag": flag, "feedback": feedback, "q": q})


@router.get("/traces/{trace_id}", response_class=HTMLResponse)
async def trace_page(trace_id: int, request: Request, session: AsyncSession = Depends(get_session)):
    t = await session.get(Trace, trace_id)
    if t is None:
        raise HTTPException(404, "No such trace (traces are kept 30 days).")
    d = t.data or {}
    items = []
    if t.kind == "recall":
        texts = dict(BRIEF_LINE.findall(d.get("brief") or ""))
        items = [{"id": i, "text": texts.get(i, ""), "href": link_for(i, texts.get(i, ""))} for i in d.get("items") or []]
    return await render(request, "trace.html", session, "traces", t=t, d=d, items=items)
