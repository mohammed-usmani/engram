"""Events, Facts and Lessons: the three memory lists, their detail fragments and the JSON edits behind them."""
from __future__ import annotations

import csv
import io
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select, text as sql, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import settings_store as store
from src.database import get_session
from src.memory import USER_TZ, facts
from src.memory.entities import resolve
from src.memory.extract import EPISODE_KINDS
from src.memory.models import Entity, Episode, Reflection
from src.services.ollama_client import embed
from src.ui.templating import invalidate, render, templates

router = APIRouter(include_in_schema=False)

PAGE = 50
LESSON_PAGE = 20
PERIODS = {"7": 7, "30": 30, "all": None}
KIND_ORDER = sorted(EPISODE_KINDS)


def _fragment(template: str, block: str, **ctx) -> HTMLResponse:
    """Render one {% block %} of a page — the htmx partials reuse the page's own markup."""
    t = templates.env.get_template(template)
    return HTMLResponse("".join(t.blocks[block](t.new_context(ctx))))


def _dots(n) -> str:
    n = max(1, min(5, int(n or 3)))
    return "●" * n + "○" * (5 - n)


templates.env.filters.setdefault("dots", _dots)


async def _names(session: AsyncSession, slugs) -> dict[str, str]:
    slugs = sorted(set(slugs))
    if not slugs:
        return {}
    rows = await session.execute(select(Entity.slug, Entity.name).where(Entity.slug.in_(slugs)))
    return {s: n for s, n in rows}


def _iso(s: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(s) if s else None
    except ValueError:
        return None


# ================================================================ events

def _event_where(q: str, kind: str, topic: str, agent: str, period: str) -> list:
    w = []
    if q:
        like = f"%{q}%"
        w.append(or_(Episode.search_vector.op("@@")(func.plainto_tsquery("english", q)),
                     Episode.summary.ilike(like), Episode.outcome.ilike(like)))
    if kind:
        w.append(Episode.kind == kind)
    if topic:
        w.append(Episode.entities.has_key(topic))
    if agent:
        w.append(Episode.source_agent == agent)
    if days := PERIODS.get(period):
        w.append(Episode.occurred_at >= datetime.now(timezone.utc) - timedelta(days=days))
    return w


def _day_label(d) -> str:
    today = datetime.now(USER_TZ).date()
    label = d.strftime("%A %-d %B") + ("" if d.year == today.year else d.strftime(" %Y"))
    return {0: "Today · ", 1: "Yesterday · "}.get((today - d).days, "") + label


async def _event_groups(session, where, offset: int, prev_day: str = "") -> tuple[list[dict], bool]:
    rows = (await session.execute(select(Episode).where(*where).order_by(Episode.occurred_at.desc(), Episode.id.desc())
                                  .offset(offset).limit(PAGE + 1))).scalars().all()
    more = len(rows) > PAGE
    rows = rows[:PAGE]
    local = func.date(func.timezone(USER_TZ.key, Episode.occurred_at))
    days = {rows[-1].occurred_at.astimezone(USER_TZ).date(), rows[0].occurred_at.astimezone(USER_TZ).date()} if rows else set()
    counts = dict((await session.execute(
        select(local, func.count()).where(*where, local.between(min(days), max(days))).group_by(local)
    )).all()) if days else {}
    groups: list[dict] = []
    for e in rows:
        d = e.occurred_at.astimezone(USER_TZ).date()
        if not groups or groups[-1]["day"] != d.isoformat():
            groups.append({"day": d.isoformat(), "label": _day_label(d), "n": counts.get(d, 0),
                           "header": d.isoformat() != prev_day, "events": []})
        groups[-1]["events"].append(e)
    return groups, more


async def _event_panel_ctx(session, ep: Episode) -> dict:
    return {"ep": ep, "ep_topics": await _names(session, ep.entities),
            "local_input": ep.occurred_at.astimezone(USER_TZ).strftime("%Y-%m-%dT%H:%M"), "tz": USER_TZ.key}


@router.get("/events", response_class=HTMLResponse)
async def events_page(request: Request, q: str = "", kind: str = "", topic: str = "", agent: str = "",
                      period: str = "all", offset: int = 0, prev_day: str = "", focus: int | None = None,
                      session: AsyncSession = Depends(get_session)):
    period = period if period in PERIODS else "all"
    filters = {k: v for k, v in dict(q=q, kind=kind, topic=topic, agent=agent, period=period).items() if v}
    qs = urlencode(filters)
    where = _event_where(q, kind, topic, agent, period)
    groups, more = await _event_groups(session, where, max(0, offset), prev_day)
    page = dict(groups=groups, more=more, next_offset=offset + PAGE, qs=qs)
    if request.headers.get("hx-request") and offset:
        return _fragment("events.html", "rows", **page)

    week = datetime.now(timezone.utc) - timedelta(days=7)
    mix = (await session.execute(select(Episode.kind, func.count()).where(Episode.occurred_at >= week)
                                 .group_by(Episode.kind).order_by(func.count().desc()))).all()
    topics = (await session.execute(sql("""
        select e.slug, e.name, count(x.id) n from entities e join episodes x on x.entities ? e.slug
        group by e.slug, e.name order by n desc limit 80"""))).all()
    agents = (await session.execute(select(Episode.source_agent, func.count()).where(Episode.source_agent.isnot(None))
                                    .group_by(Episode.source_agent).order_by(func.count().desc()))).all()
    ep = await session.get(Episode, focus) if focus else None
    return await render(
        request, "events.html", session, "events", **page,
        f=dict(q=q, kind=kind, topic=topic, agent=agent, period=period),
        total=await session.scalar(select(func.count()).select_from(Episode)),
        matching=await session.scalar(select(func.count()).select_from(Episode).where(*where)),
        mix=mix, mix_total=sum(n for _, n in mix), kinds=KIND_ORDER, topics=topics, agents=agents,
        focus=focus, **(await _event_panel_ctx(session, ep) if ep else {}),
    )


@router.get("/events/{event_id}/panel", response_class=HTMLResponse)
async def event_panel(event_id: int, session: AsyncSession = Depends(get_session)):
    ep = await session.get(Episode, event_id)
    if ep is None:
        raise HTTPException(404, "That event no longer exists.")
    return _fragment("events.html", "panel", kinds=KIND_ORDER, **await _event_panel_ctx(session, ep))


@router.get("/api/events.csv")
async def events_csv(q: str = "", kind: str = "", topic: str = "", agent: str = "", period: str = "all",
                     session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(select(Episode).where(*_event_where(q, kind, topic, agent, period))
                                  .order_by(Episode.occurred_at.desc()))).scalars()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "occurred_at", "kind", "summary", "outcome", "importance", "topics", "source_agent"])
    for e in rows:
        w.writerow([e.id, e.occurred_at.astimezone(USER_TZ).isoformat(), e.kind, e.summary, e.outcome or "",
                    e.importance, ";".join(e.entities), e.source_agent or ""])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="engram-events.csv"'})


class EventPatch(BaseModel):
    summary: str | None = Field(None, min_length=1, max_length=5000)
    outcome: str | None = Field(None, max_length=100)
    kind: str | None = None
    importance: int | None = Field(None, ge=1, le=5)
    occurred_at: str | None = None


@router.patch("/api/events/{event_id}")
async def patch_event(event_id: int, body: EventPatch, session: AsyncSession = Depends(get_session)):
    ep = await session.get(Episode, event_id)
    if ep is None:
        raise HTTPException(404, "That event no longer exists.")
    sent = body.model_fields_set
    if body.kind is not None:
        if body.kind not in EPISODE_KINDS:
            raise HTTPException(400, f"Unknown kind “{body.kind}”. Pick one from the list.")
        ep.kind = body.kind
    if body.summary is not None and body.summary.strip() != ep.summary:
        ep.summary = body.summary.strip()
        ep.embedding = await embed(ep.summary)
    if "outcome" in sent:
        ep.outcome = (body.outcome or "").strip() or None
    if body.importance is not None:
        ep.importance = body.importance
    if body.occurred_at:
        try:
            when = datetime.fromisoformat(body.occurred_at)
        except ValueError:
            raise HTTPException(400, "That date isn't valid. Use the date picker.")
        ep.occurred_at = when if when.tzinfo else when.replace(tzinfo=USER_TZ)
    if ep.entities:  # topic digests and lessons are rebuilt from events on the next sleep pass
        await session.execute(update(Entity).where(Entity.slug.in_(ep.entities)).values(dirty=True))
    await session.commit()
    return {"id": ep.id, "kind": ep.kind, "summary": ep.summary, "outcome": ep.outcome,
            "importance": ep.importance, "occurred_at": ep.occurred_at.isoformat()}


# ================================================================ facts

def _fact_row(fid: str, p: dict, names: dict[str, str]) -> dict:
    return {"id": fid, "text": p.get("data", ""), "kind": p.get("kind") or "fact",
            "importance": int(p.get("importance") or 3), "agent": p.get("agent"),
            "topics": [(s, names.get(s, s.split(":", 1)[-1])) for s in p.get("entities") or []],
            "created": _iso(p.get("created_at")), "pinned": bool(p.get("pinned")),
            "superseded_by": p.get("superseded_by"), "valid_to": _iso(p.get("valid_to"))}


async def _fact_rows(session, kind: str, q: str, replaced: bool, offset: int) -> tuple[list[dict], bool]:
    t = facts.table()
    raw = (await session.execute(sql(f"""
        select id::text id, payload - 'text_lemmatized' p from {t}
        where coalesce(payload->>'kind', 'fact') = :kind
          and (payload->>'superseded_by' is not null) = :replaced
          and (:q = '' or payload->>'data' ilike :like)
        order by coalesce(payload->>'valid_to', payload->>'created_at') desc, id
        limit :lim offset :off"""), {"kind": kind, "replaced": replaced, "q": q, "like": f"%{q}%",
                                     "lim": PAGE + 1, "off": max(0, offset)})).all()
    more = len(raw) > PAGE
    raw = raw[:PAGE]
    names = await _names(session, [s for _, p in raw for s in p.get("entities") or []])
    rows = [_fact_row(i, p, names) for i, p in raw]
    by = [r["superseded_by"] for r in rows if r["superseded_by"]]
    if by:
        repl = dict((await session.execute(sql(f"select id::text, payload->>'data' from {t} where id::text = any(:ids)"),
                                           {"ids": by})).all())
        for r in rows:
            r["replacement"] = repl.get(r["superseded_by"])
    return rows, more


async def _combine_suggestion(session, prof: list[dict], names: dict[str, str]) -> dict | None:
    """Profile slots are precious: flag ≥3 facts about one topic, or facts that say nearly the same thing."""
    tok = {f["id"]: len(f["memory"]) // 4 + 1 for f in prof}
    shared = Counter(s for f in prof for s in set(f["metadata"].get("entities") or []))
    slug, n = shared.most_common(1)[0] if shared else (None, 0)
    if n >= 3:
        ids = [f["id"] for f in prof if slug in (f["metadata"].get("entities") or [])]
        why = f"{n} of the {len(prof)} are about {names.get(slug, slug)}"
    else:
        pairs = (await session.execute(sql(f"""
            select a.id::text, b.id::text from {facts.table()} a join {facts.table()} b on a.id < b.id
            where a.id::text = any(:ids) and b.id::text = any(:ids) and 1 - (a.vector <=> b.vector) >= 0.9
            order by a.vector <=> b.vector"""), {"ids": list(tok)})).all()
        if not pairs:
            return None
        # ponytail: the closest pair plus anything near either of them; no full clustering of 15 facts
        seed = set(pairs[0])
        ids = [i for i in tok if i in seed or any(i in p and seed & set(p) for p in pairs)]
        why = f"{len(ids)} of the {len(prof)} say nearly the same thing"
    return {"ids": ids, "why": why, "saves": sum(tok[i] for i in ids) - max(tok[i] for i in ids)}


@router.get("/facts", response_class=HTMLResponse)
async def facts_page(request: Request, kind: str = "fact", q: str = "", replaced: bool = False, offset: int = 0,
                     session: AsyncSession = Depends(get_session)):
    kind = kind if kind in facts.KINDS else "fact"
    rows, more = await _fact_rows(session, kind, q, replaced, offset)
    qs = urlencode({k: v for k, v in dict(kind=kind, q=q, replaced="1" if replaced else "").items() if v})
    page = dict(rows=rows, more=more, next_offset=offset + PAGE, qs=qs, replaced=replaced)
    if request.headers.get("hx-request") and offset:
        return _fragment("facts.html", "rows", **page)

    t = facts.table()
    counts = dict((await session.execute(sql(f"""select coalesce(payload->>'kind', 'fact'), count(*) from {t}
        where payload->>'superseded_by' is null group by 1"""))).all())
    n_replaced = await session.scalar(sql(f"""select count(*) from {t} where payload->>'superseded_by' is not null
        and coalesce(payload->>'kind', 'fact') = :kind"""), {"kind": kind})
    prof = await facts.profile_rows(15)
    names = await _names(session, [s for f in prof for s in f["metadata"].get("entities") or []])
    return await render(
        request, "facts.html", session, "facts", **page, kind=kind, q=q, kinds=facts.KINDS, counts=counts,
        n_replaced=n_replaced, profile=prof,
        profile_tokens=len("\n".join(f"- {f['memory']}" for f in prof)) // 4 + 1,
        suggest=await _combine_suggestion(session, prof, names),
    )


class FactIn(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000)
    kind: str = "fact"
    importance: int = Field(3, ge=1, le=5)
    topics: str = ""


class FactPatch(BaseModel):
    text: str | None = Field(None, min_length=3, max_length=2000)
    importance: int | None = Field(None, ge=1, le=5)


class PinIn(BaseModel):
    pinned: bool


class CombineIn(BaseModel):
    ids: list[str] = Field(..., min_length=2)
    text: str = Field(..., min_length=3, max_length=2000)
    importance: int | None = Field(None, ge=1, le=5)


async def _fact_or_404(fact_id: str) -> dict:
    try:
        uuid.UUID(fact_id)
    except ValueError:
        raise HTTPException(404, "That fact no longer exists.")
    f = await facts.get_fact(fact_id)
    if f is None:
        raise HTTPException(404, "That fact no longer exists.")
    return f


@router.post("/api/facts")
async def add_fact(body: FactIn, session: AsyncSession = Depends(get_session)):
    if body.kind not in facts.KINDS:
        raise HTTPException(400, f"Kind must be one of: {', '.join(facts.KINDS)}.")
    names = [n.strip() for n in body.topics.split(",") if n.strip()]
    slugs = list((await resolve(session, [{"name": n, "kind": "topic"} for n in names])).values()) if names else []
    await session.commit()
    fid = await facts.add_fact(body.text.strip(), body.kind, slugs, body.importance, agent="admin")
    invalidate()
    return {"id": fid, "entities": slugs}


@router.patch("/api/facts/{fact_id}")
async def patch_fact(fact_id: str, body: FactPatch):
    f = await _fact_or_404(fact_id)
    text = body.text.strip() if body.text and body.text.strip() != f["memory"] else None
    meta = {"importance": body.importance} if body.importance else {}
    if text or meta:
        await facts.update_fact(fact_id, text, **meta)
    return {"id": fact_id, "updated": bool(text or meta)}


@router.post("/api/facts/{fact_id}/restore")
async def restore_fact(fact_id: str):
    f = await _fact_or_404(fact_id)
    if not f["metadata"].get("superseded_by"):
        raise HTTPException(400, "That fact is already current.")
    await facts.restore(fact_id)
    invalidate()
    return {"id": fact_id, "restored": True}


@router.post("/api/facts/{fact_id}/pin")
async def pin_fact(fact_id: str, body: PinIn):
    f = await _fact_or_404(fact_id)
    if body.pinned and f["metadata"].get("superseded_by"):
        raise HTTPException(400, "A replaced fact can't be pinned. Restore it first.")
    await facts.update_fact(fact_id, pinned=body.pinned)
    return {"id": fact_id, "pinned": body.pinned}


@router.post("/api/facts/combine")
async def combine_facts(body: CombineIn):
    old = [await _fact_or_404(i) for i in dict.fromkeys(body.ids)]
    if len(old) < 2:
        raise HTTPException(400, "Pick at least two facts to combine.")
    metas = [f["metadata"] for f in old]
    kind = Counter(m.get("kind") or "fact" for m in metas).most_common(1)[0][0]
    entities = sorted({s for m in metas for s in m.get("entities") or []})
    importance = body.importance or max(int(m.get("importance") or 3) for m in metas)
    new_id = await facts.add_fact(body.text.strip(), kind, entities, importance, agent="admin")
    if any(m.get("pinned") for m in metas):
        await facts.update_fact(new_id, pinned=True)
    for f in old:
        await facts.supersede(f["id"], new_id)
    invalidate()
    return {"id": new_id, "replaced": [f["id"] for f in old], "importance": importance}


# ================================================================ lessons

BUCKETS = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"]
SORTS = {"evidence": "Most evidence", "newest": "Newest", "least": "Least confident"}


@router.get("/lessons", response_class=HTMLResponse)
async def lessons_page(request: Request, topic: str = "", sort: str = "evidence", page: int = 1,
                       session: AsyncSession = Depends(get_session)):
    sort = sort if sort in SORTS else "evidence"
    page = max(1, page)
    bucket = func.least(func.floor(Reflection.confidence * 5), 4)
    hist = dict((await session.execute(select(bucket, func.count()).group_by(bucket))).all())
    hist = [{"label": b, "n": hist.get(i, 0)} for i, b in enumerate(BUCKETS)]
    topics = (await session.execute(sql("""
        select t.slug, coalesce(e.name, t.slug) name, count(*) n
        from reflections r cross join jsonb_array_elements_text(r.entities) t(slug)
        left join entities e on e.slug = t.slug group by t.slug, e.name order by n desc limit 80"""))).all()
    where = [Reflection.entities.has_key(topic)] if topic else []
    order = {"evidence": (func.jsonb_array_length(Reflection.evidence).desc(), Reflection.id.desc()),
             "newest": (Reflection.updated_at.desc(), Reflection.id.desc()),
             "least": (Reflection.confidence.asc(), Reflection.id.desc())}[sort]
    matching = await session.scalar(select(func.count()).select_from(Reflection).where(*where))
    lessons = (await session.execute(select(Reflection).where(*where).order_by(*order)
                                     .offset((page - 1) * LESSON_PAGE).limit(LESSON_PAGE))).scalars().all()
    names = await _names(session, [s for r in lessons for s in r.entities])
    qs = urlencode({k: v for k, v in dict(topic=topic, sort=sort).items() if v})
    return await render(
        request, "lessons.html", session, "lessons", hist=hist, hist_max=max([h["n"] for h in hist] + [1]),
        total=sum(h["n"] for h in hist), topics=topics, topic=topic, sort=sort, sorts=SORTS, lessons=lessons,
        names=names, matching=matching, page=page, pages=max(1, -(-matching // LESSON_PAGE)), qs=qs,
    )


@router.get("/lessons/{lesson_id}/evidence", response_class=HTMLResponse)
async def lesson_evidence(lesson_id: int, session: AsyncSession = Depends(get_session)):
    r = await session.get(Reflection, lesson_id)
    if r is None:
        raise HTTPException(404, "That lesson no longer exists.")
    eps = (await session.execute(select(Episode).where(Episode.id.in_(r.evidence or [-1]))
                                 .order_by(Episode.occurred_at.desc()))).scalars().all()
    return _fragment("lessons.html", "evidence", r=r, eps=eps, missing=len(set(r.evidence)) - len(eps))


class LessonPatch(BaseModel):
    lesson: str = Field(..., min_length=10, max_length=2000)


async def _lesson_or_404(session, lesson_id: int) -> Reflection:
    r = await session.get(Reflection, lesson_id)
    if r is None:
        raise HTTPException(404, "That lesson no longer exists.")
    return r


@router.post("/api/lessons/{lesson_id}/confirm")
async def confirm_lesson(lesson_id: int, session: AsyncSession = Depends(get_session)):
    r = await _lesson_or_404(session, lesson_id)
    r.confidence = 1.0
    await session.commit()
    return {"id": r.id, "confidence": 1.0}


@router.patch("/api/lessons/{lesson_id}")
async def reword_lesson(lesson_id: int, body: LessonPatch, session: AsyncSession = Depends(get_session)):
    r = await _lesson_or_404(session, lesson_id)
    r.lesson = body.lesson.strip()
    r.embedding = await embed(r.lesson)
    await session.commit()
    return {"id": r.id, "lesson": r.lesson}


@router.post("/api/lessons/{lesson_id}/reject")
async def reject_lesson(lesson_id: int, session: AsyncSession = Depends(get_session)):
    r = await _lesson_or_404(session, lesson_id)
    text = r.lesson
    await session.delete(r)
    await store.put(session, "rejected_lessons", [*(store.get("rejected_lessons") or []), text])  # commits both
    invalidate()
    return {"id": lesson_id, "rejected": True}
