"""JSON endpoints behind the Overview, Documents, Topics, Queue and Recall-inspector pages."""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from src import dates
from src.database import get_session
from src.memory import facts
from src.memory import recall as rc
from src.memory.models import Entity, IngestJob
from src.models import ContextDocument
from src.privacy import readable
from src.ui.templating import invalidate

router = APIRouter(prefix="/api", tags=["ui"])


async def _doc(session: AsyncSession, slug: str) -> ContextDocument:
    doc = (await session.execute(select(ContextDocument).where(
        ContextDocument.slug == slug, readable(ContextDocument.privacy)))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, f"No document '{slug}'.")
    return doc


class TrackIn(BaseModel):
    field: str = Field("Due", min_length=2, max_length=40)
    date: date


@router.post("/documents/{slug:path}/track-date")
async def track_date(slug: str, body: TrackIn, session: AsyncSession = Depends(get_session)):
    """Adds `Field: YYYY-MM-DD` to the document's top fields, so it shows under Coming up."""
    from src.routers.admin import _refresh_derived
    k = body.field.strip().lower()
    if k not in dates.ONE_OFF and k not in dates.YEARLY:
        raise HTTPException(422, f"'{body.field}' isn't a tracked field. Use one of: Due, Deadline, Expires, Renews, "
                                 "Appointment, Next Service, Matures, Birthday, Anniversary.")
    doc = await _doc(session, slug)
    doc.content = dates.add_field(doc.content, doc.doc_metadata, body.field.strip().title(), body.date.isoformat())
    doc.updated_by = "admin"
    await _refresh_derived(doc)
    await session.commit()
    return {"slug": slug, "field": body.field.title(), "date": body.date.isoformat()}


# ---------------------------------------------------------------- topics

class TopicIn(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    kind: str | None = Field(None, pattern=r"^[a-z][a-z_-]{1,40}$")


@router.get("/topics")
async def find_topics(q: str = "", limit: int = 20, session: AsyncSession = Depends(get_session)):
    """Typeahead for topic pickers: name, slug or alias match, busiest first."""
    rows = (await session.execute(sql("""
        select e.slug, e.name, e.kind, coalesce(c.n, 0) n from entities e
        left join (select x.e, count(*) n from episodes, jsonb_array_elements_text(entities) x(e) group by 1) c on c.e = e.slug
        where e.name ilike :q or e.slug ilike :q or e.aliases::text ilike :q
        order by coalesce(c.n, 0) desc limit :lim"""), {"q": f"%{q.strip()}%", "lim": max(1, min(limit, 50))})).mappings().all()
    return [dict(r) for r in rows]


@router.patch("/topics/{slug}")
async def edit_topic(slug: str, body: TopicIn, session: AsyncSession = Depends(get_session)):
    e = (await session.execute(select(Entity).where(Entity.slug == slug))).scalar_one_or_none()
    if e is None:
        raise HTTPException(404, f"No topic '{slug}'.")
    if body.name and body.name != e.name:
        if e.name not in e.aliases:
            e.aliases = [*e.aliases, e.name]  # the old name keeps matching in text
        e.name = body.name.strip()
    await session.commit()
    invalidate()
    return {"slug": e.slug, "name": e.name, "aliases": e.aliases}


@router.post("/topics/{slug}/summarize")
async def summarize_topic(slug: str, session: AsyncSession = Depends(get_session)):
    """Marks the topic for the next sleep pass and asks for one soon."""
    from src.memory.worker import request_consolidation
    e = (await session.execute(select(Entity).where(Entity.slug == slug))).scalar_one_or_none()
    if e is None:
        raise HTTPException(404, f"No topic '{slug}'.")
    e.dirty = True
    await session.commit()
    request_consolidation()
    return {"queued": True}


# ---------------------------------------------------------------- queue

@router.get("/queue/jobs/{job_id}")
async def job_detail(job_id: int, session: AsyncSession = Depends(get_session)):
    j = await session.get(IngestJob, job_id)
    if j is None:
        raise HTTPException(404, "No such job.")
    episodes = (await session.execute(sql(
        "select id, kind, summary, outcome from episodes where payload->>'job' = :j order by id"), {"j": str(job_id)})).mappings().all()
    written = []
    for fid in (j.result or {}).get("fact_ids", [])[:20]:
        f = await facts.get_fact(fid)
        if f:
            written.append({"id": f["id"], "text": f["memory"], "kind": f["metadata"].get("kind"),
                            "replaced": bool(f["metadata"].get("superseded_by"))})
    return {"id": j.id, "status": j.status, "agent": j.agent, "session_id": j.session_id, "attempts": j.attempts,
            "error": j.error, "result": j.result, "text": j.text, "created_at": j.created_at,
            "occurred_at": j.occurred_at, "episodes": [dict(e) for e in episodes], "facts": written}


@router.post("/queue/jobs/{job_id}/undo")
async def undo_job(job_id: int, session: AsyncSession = Depends(get_session)):
    """Removes everything one job wrote (its events and facts). The job stays, marked undone."""
    j = await session.get(IngestJob, job_id)
    if j is None:
        raise HTTPException(404, "No such job.")
    if j.status in ("pending", "processing", "batched"):
        raise HTTPException(409, f"Job #{job_id} is {j.status}; wait until it finishes.")
    n_ep = (await session.execute(sql("DELETE FROM episodes WHERE payload->>'job' = :j"), {"j": str(job_id)})).rowcount
    fact_ids = (j.result or {}).get("fact_ids", [])
    for fid in fact_ids:
        await facts.delete_fact(fid)
    touched = (j.result or {}).get("entities", [])
    if touched:
        await session.execute(sql("UPDATE entities SET dirty = true WHERE slug = ANY(:s)"), {"s": touched})
    j.status, j.result = "undone", {**(j.result or {}), "undone": {"episodes": n_ep, "facts": len(fact_ids)}}
    await session.commit()
    invalidate()
    return {"episodes": n_ep, "facts": len(fact_ids)}


class NoteKey(BaseModel):
    session_id: str
    key: str


@router.delete("/session-notes")
async def delete_note(body: NoteKey, session: AsyncSession = Depends(get_session)):
    n = (await session.execute(sql("delete from session_state where session_id = :s and key = :k"),
                               {"s": body.session_id, "k": body.key})).rowcount
    await session.commit()
    if not n:
        raise HTTPException(404, "No such note.")
    return {"deleted": n}


# ---------------------------------------------------------------- recall inspector

class InspectIn(BaseModel):
    situation: str = Field(..., min_length=1, max_length=4000)
    budget_tokens: int = Field(1500, ge=200, le=8000)
    fast: bool = True


@router.post("/recall/inspect")
async def inspect(body: InspectIn, session: AsyncSession = Depends(get_session)):
    return await rc.recall(session, body.situation, body.budget_tokens, fast=body.fast, explain=True)
