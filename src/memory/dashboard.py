"""/memory — a browser view of everything Engram knows, with delete/retry and a recall preview."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, or_, select, text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.memory import USER_TZ, facts, local_date
from src.memory.models import Entity, Episode, IngestJob, Reflection, SessionState
from src.memory.recall import recall

log = logging.getLogger(__name__)
router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))
templates.env.filters["local_date"] = local_date
templates.env.filters["local_time"] = lambda dt: dt.astimezone(USER_TZ).strftime("%Y-%m-%d %H:%M")

TABS = ["overview", "topics", "events", "facts", "lessons", "notes", "queue"]


def _last_backup() -> datetime | None:
    folder = Path(os.environ.get("ENGRAM_BACKUP_DIR", Path.home() / "dev/personal/engram-data/backups"))
    archives = sorted(folder.glob("engram-*.tar.age"), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
    if not archives:
        return None
    return datetime.fromtimestamp(archives[-1].stat().st_mtime, timezone.utc)


async def _topics(session: AsyncSession, limit: int = 200) -> list[dict]:
    rows = (await session.execute(sql("""
        select e.slug, e.name, e.kind, e.digest, e.dirty, count(ep.id) as events, max(ep.occurred_at) as last
        from entities e left join episodes ep on ep.entities ? e.slug
        group by e.id order by count(ep.id) desc, e.updated_at desc limit :n"""), {"n": limit})).mappings().all()
    return [dict(r) for r in rows]


async def _facts(q: str | None) -> tuple[list[dict], str | None]:
    try:
        rows = await facts.all_facts()
    except Exception as e:  # mem0/pgvector unavailable shouldn't take the whole page down
        log.warning("dashboard facts unavailable: %s", e)
        return [], str(e)
    if q:
        rows = [f for f in rows if q.lower() in f["memory"].lower()]
    order = {"preference": 0, "fact": 1, "procedure": 2}
    rows.sort(key=lambda f: (order.get(f["metadata"].get("kind"), 3), -int(f["metadata"].get("importance", 3))))
    return rows, None


@router.get("/memory", response_class=HTMLResponse)
async def memory_page(request: Request, tab: str = "overview", q: str | None = None, entity: str | None = None,
                      kind: str | None = None, session: AsyncSession = Depends(get_session)):
    tab = tab if tab in TABS else "overview"
    now = datetime.now(timezone.utc)
    ctx: dict = {"request": request, "active_page": "memory", "tab": tab, "tabs": TABS, "q": q or "",
                 "entity": entity or "", "kind": kind or ""}

    count = lambda model, *where: session.scalar(select(func.count()).select_from(model).where(*where))  # noqa: E731
    jobs_by_status = dict((await session.execute(
        select(IngestJob.status, func.count()).group_by(IngestJob.status))).all())
    fact_rows, facts_error = await _facts(q if tab == "facts" else None)
    ctx["counts"] = {
        "events": await count(Episode),
        "topics": await count(Entity),
        "lessons": await count(Reflection),
        "notes": await count(SessionState, SessionState.expires_at > now),
        "facts": sum(f["metadata"].get("kind") == "fact" for f in fact_rows),
        "preferences": sum(f["metadata"].get("kind") == "preference" for f in fact_rows),
        "procedures": sum(f["metadata"].get("kind") == "procedure" for f in fact_rows),
        "pending": jobs_by_status.get("pending", 0) + jobs_by_status.get("processing", 0),
        "failed": jobs_by_status.get("failed", 0),
        "dirty": await count(Entity, Entity.dirty),
    }
    ctx["last_backup"] = _last_backup()
    ctx["facts_error"] = facts_error

    if tab in ("overview", "topics"):
        ctx["topics"] = await _topics(session, limit=12 if tab == "overview" else 200)
    if tab == "overview":
        from src.dates import upcoming
        ctx["upcoming"], _ = await upcoming(session, days=60)
        ctx["recent"] = (await session.execute(
            select(Episode).order_by(Episode.occurred_at.desc()).limit(8))).scalars().all()
    if tab == "events":
        stmt = select(Episode).order_by(Episode.occurred_at.desc()).limit(300)
        if entity:
            stmt = stmt.where(Episode.entities.has_key(entity))
        if kind:
            stmt = stmt.where(Episode.kind == kind)
        if q:
            stmt = stmt.where(or_(Episode.summary.ilike(f"%{q}%"), Episode.outcome.ilike(f"%{q}%")))
        ctx["events"] = (await session.execute(stmt)).scalars().all()
        ctx["kinds"] = (await session.execute(select(Episode.kind).distinct().order_by(Episode.kind))).scalars().all()
    if tab == "facts":
        ctx["facts"] = fact_rows
    if tab == "lessons":
        ctx["lessons"] = (await session.execute(
            select(Reflection).order_by(Reflection.confidence.desc(), Reflection.updated_at.desc()))).scalars().all()
    if tab == "notes":
        ctx["notes"] = (await session.execute(
            select(SessionState).where(SessionState.expires_at > now).order_by(SessionState.created_at.desc())
        )).scalars().all()
    if tab == "queue":
        from src.memory import batch
        ctx["jobs"] = (await session.execute(
            select(IngestJob).order_by(IngestJob.id.desc()).limit(100))).scalars().all()
        ctx["batch"] = {"on": await batch.is_on(session), "available": batch.available(),
                        "model": batch.model(), "counts": await batch.counts(session)}
    return templates.TemplateResponse(request, "memory.html", ctx)


@router.get("/memory/recall-preview", response_class=HTMLResponse)
async def recall_preview(request: Request, situation: str = "", budget_tokens: int = 1500,
                         session: AsyncSession = Depends(get_session)):
    """Exactly the brief an assistant would get (fast mode: no LLM call)."""
    if not situation.strip():
        return HTMLResponse("<p class='muted'>Type what you'd say to an assistant.</p>")
    out = await recall(session, situation, budget_tokens=max(100, min(budget_tokens, 8000)), fast=True)
    return templates.TemplateResponse(request, "memory_recall.html", {"out": out})
