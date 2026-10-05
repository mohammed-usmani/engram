"""HTML pages of the admin UI. Pages only read; every change goes through a JSON endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from src import settings_store as store
from src.database import get_session
from src.memory import USER_TZ, llm
from src.ui.templating import render

router = APIRouter(include_in_schema=False)


# ---------------------------------------------------------------- settings

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, session: AsyncSession = Depends(get_session)):
    from src.memory import batch
    from src.routers import settings as api
    from src.ui import system_api

    providers = await api.list_providers(session)
    options = []
    for p in providers:
        p["models"] = await api._models_for(p["provider"])
        if p["effective_model"] and p["effective_model"] not in p["models"]:
            p["models"].insert(0, p["effective_model"])
        if p["api_key_set"]:
            options += [{"value": f"{p['provider']}|{m}", "label": f"{p['label']} · {m}"} for m in p["models"]]
    people = (await session.execute(sql("""
        select e.slug, e.name, count(x.id) n from entities e left join episodes x on x.entities ? e.slug
        where e.kind = 'person' group by e.id order by n desc limit 25"""))).mappings().all()
    return await render(
        request, "settings.html", session, "settings",
        providers=providers, options=options, tasks=api._models_view()["tasks"],
        batch={"on": await batch.is_on(session), "available": batch.available(), "model": batch.model()},
        recall=store.get("recall"), worker=store.get("worker"), flags=store.get("flags"), people=people,
        link=await system_api.public_link(), apps=await system_api.connected_apps(),
    )


# ---------------------------------------------------------------- overview

STARTERS = [
    {"title": "Passport", "field": "Expires: 2031-04-12", "template": "passport"},
    {"title": "Vehicle insurance", "field": "Renews: 14/03/2027", "template": "insurance"},
    {"title": "Someone close to you", "field": "Birthday: 10 Oct", "template": "person"},
]


@router.get("/memory", response_class=HTMLResponse)
async def overview(request: Request, tab: str | None = None, session: AsyncSession = Depends(get_session)):
    if tab:  # old /memory?tab=… links
        target = {"topics": "/topics", "events": "/events", "facts": "/facts", "lessons": "/lessons",
                  "queue": "/queue", "notes": "/queue"}.get(tab)
        if target:
            return RedirectResponse(target, 301)
    from src import dates
    from src.memory import batch, facts
    from src.ui import system_api

    me = (store.get("recall") or {}).get("self_entity")
    stats = (await session.execute(sql(f"""
        select (select count(*) from episodes) events,
               (select count(*) from episodes where occurred_at > now() - interval '7 days') events_week,
               (select count(*) from reflections) lessons,
               (select count(*) from entities) topics,
               (select count(*) from entities e left join (select x.e, count(*) n from episodes,
                    jsonb_array_elements_text(entities) x(e) group by 1) c on c.e = e.slug
                where coalesce(c.n, 0) <= 1) topics_once,
               (select count(*) from {facts.table()} where payload->>'superseded_by' is null) facts,
               (select count(*) from {facts.table()} where payload->>'superseded_by' is null and payload->>'kind' = 'preference') prefs,
               (select count(*) from {facts.table()} where payload->>'superseded_by' is null and payload->>'kind' = 'procedure') procs
        """))).mappings().one()
    weeks = (await session.execute(sql("""
        select to_char(w, 'Mon DD') label, coalesce(n, 0) n
        from generate_series(date_trunc('week', now()) - interval '11 weeks', date_trunc('week', now()), interval '1 week') w
        left join (select date_trunc('week', occurred_at) wk, count(*) n from episodes group by 1) c on c.wk = w
        order by w"""))).mappings().all()
    topics = (await session.execute(sql("""
        select e.slug, e.name, e.kind, c.n from entities e join (select x.e, count(*) n from episodes,
            jsonb_array_elements_text(entities) x(e) group by 1) c on c.e = e.slug
        where e.slug <> coalesce(:me, '') order by c.n desc limit 7"""), {"me": me})).mappings().all()
    recent = (await session.execute(sql(
        "select id, kind, summary, outcome, occurred_at from episodes order by occurred_at desc limit 6"))).mappings().all()
    jobs = dict((await session.execute(sql("select status, count(*) from ingest_jobs group by status"))).all())
    docs = await session.scalar(sql("select count(*) from context_documents"))
    types_used = await session.scalar(sql("select count(distinct type) from context_documents"))

    soon, unreadable = await dates.upcoming(session, 60)
    hidden = await dates.untracked(session, limit=3)
    attention = []
    try:
        from src.memory import cleanup
        n = await cleanup.badge(session)
        if n:
            attention.append({"count": n, "title": "Cleanup suggestions", "href": "/cleanup",
                              "detail": "Topics with two names, duplicate events and one-off topics to fold."})
    except Exception:
        pass
    if stats["topics_once"]:
        attention.append({"count": stats["topics_once"], "title": "Topics mentioned once", "href": "/cleanup#once",
                          "detail": "One-off topics such as file names make search and recall noisier."})
    if jobs.get("failed"):
        attention.append({"count": jobs["failed"], "title": "Failed queue jobs", "href": "/queue?status=failed",
                          "detail": "Text that couldn't be turned into memory. Retry or inspect them."})
    if unreadable:
        attention.append({"count": len(unreadable), "title": "Unreadable dates", "href": f"/api/admin/{unreadable[0]['slug']}",
                          "detail": "Date fields Engram couldn't parse, so they aren't tracked."})
    if not store.get("recall").get("self_entity"):
        attention.append({"count": "1", "title": "Tell Engram which topic is you", "href": "/settings#h-recall",
                          "detail": "Otherwise your own name boosts every result and wastes brief space."})
    if not store.get("flags").get("recovery_key_saved"):
        attention.append({"count": "!", "title": "Recovery key not confirmed", "href": "/settings#h-backup",
                          "detail": "Without ~/.config/engram/backup.key, backups can't be restored."})
    extract = llm.steps("extract")
    return await render(
        request, "overview.html", session, "overview",
        stats=stats, weeks=weeks, max_week=max([w["n"] for w in weeks] + [1]), topics=topics,
        max_topic=max([t["n"] for t in topics] + [1]), recent=recent, jobs=jobs, docs=docs, types_used=types_used,
        soon=soon, hidden=hidden, starters=STARTERS, attention=attention[:5],
        extract=[f"{n} · {m or llm.default_model(n)}" for n, m in extract],
        batch_on=await batch.is_on(session), link=await system_api.public_link(),
    )


# ---------------------------------------------------------------- topics

_COUNTS = "(select x.e, count(*) n, max(occurred_at) last from episodes, jsonb_array_elements_text(entities) x(e) group by 1)"
KINDS = ["company", "person", "project", "topic", "place", "tool", "org"]


@router.get("/topics", response_class=HTMLResponse)
async def topics_page(request: Request, q: str = "", kind: str = "", sort: str = "events", page: int = 1,
                      session: AsyncSession = Depends(get_session)):
    where, args = ["true"], {}
    if q.strip():
        where.append("(e.name ilike :q or e.slug ilike :q or e.aliases::text ilike :q)")
        args["q"] = f"%{q.strip()}%"
    if kind == "other":
        where.append("e.kind <> all(:kinds)")
        args["kinds"] = KINDS
    elif kind:
        where.append("e.kind = :kind")
        args["kind"] = kind
    order = {"events": "coalesce(c.n,0) desc, e.name", "recent": "c.last desc nulls last", "name": "lower(e.name)"}.get(sort, "coalesce(c.n,0) desc")
    per = 30
    args.update(lim=per + 1, off=(max(page, 1) - 1) * per)
    rows = (await session.execute(sql(f"""
        select e.slug, e.name, e.kind, e.digest, e.dirty, coalesce(c.n, 0) n, c.last from entities e
        left join {_COUNTS} c on c.e = e.slug where {' and '.join(where)} order by {order} limit :lim offset :off"""), args)).mappings().all()
    kinds = (await session.execute(sql("select kind, count(*) n from entities group by kind order by n desc"))).all()
    by_kind = {k: n for k, n in kinds}
    chips = [("", "All", sum(by_kind.values()))] + [(k, k, by_kind.get(k, 0)) for k in KINDS if by_kind.get(k)] \
        + [("other", "other kinds", sum(n for k, n in by_kind.items() if k not in KINDS))]
    once = await session.scalar(sql(f"select count(*) from entities e left join {_COUNTS} c on c.e = e.slug where coalesce(c.n,0) <= 1"))
    top = max([r["n"] for r in rows] + [1])
    return await render(request, "topics.html", session, "topics", rows=rows[:per], more=len(rows) > per, page=page,
                        q=q, kind=kind, sort=sort, chips=chips, once=once, top=top,
                        me=(store.get("recall") or {}).get("self_entity"))


@router.get("/topics/{slug}", response_class=HTMLResponse)
async def topic_page(request: Request, slug: str, kind: str = "", session: AsyncSession = Depends(get_session)):
    from fastapi import HTTPException
    from src.memory.recall import _aggregates
    e = (await session.execute(sql("select * from entities where slug = :s"), {"s": slug})).mappings().one_or_none()
    if e is None:
        raise HTTPException(404, "No such topic")
    counts = (await session.execute(sql("""select kind, count(*) n, min(occurred_at) first, max(occurred_at) last
        from episodes where entities ? :s group by kind order by n desc"""), {"s": slug})).mappings().all()
    total = sum(c["n"] for c in counts)
    days = (await session.execute(sql("""
        select to_char(d, 'YYYY-MM-DD') as dkey, to_char(d, 'Mon DD') as label, x.kind, count(x.id) n
        from generate_series((now() at time zone :tz)::date - 14, (now() at time zone :tz)::date, interval '1 day') d
        left join episodes x on x.entities ? :s and (x.occurred_at at time zone :tz)::date = d::date
        group by d, x.kind order by d"""), {"s": slug, "tz": str(USER_TZ)})).mappings().all()
    timeline = {}
    for r in days:
        t = timeline.setdefault(r["dkey"], {"label": r["label"], "blocks": []})
        if r["kind"]:
            t["blocks"] += [r["kind"]] * r["n"]
    in_window = sum(len(t["blocks"]) for t in timeline.values())
    q = "select id, kind, summary, outcome, occurred_at, importance from episodes where entities ? :s"
    args = {"s": slug}
    if kind:
        q += " and kind = :k"
        args["k"] = kind
    events = (await session.execute(sql(q + " order by occurred_at desc limit 40"), args)).mappings().all()
    lessons = (await session.execute(sql("""select id, lesson, confidence, jsonb_array_length(evidence) ev, updated_at
        from reflections where entities ? :s order by confidence desc, jsonb_array_length(evidence) desc limit 8"""), args)).mappings().all()
    related = (await session.execute(sql("""
        select e.slug, e.name, e.kind, count(*) n from episodes x, jsonb_array_elements_text(x.entities) r(slug)
        join entities e on e.slug = r.slug where x.entities ? :s and r.slug <> :s group by e.id order by n desc limit 8"""), args)).mappings().all()
    return await render(request, "topic.html", session, "topics", e=e, counts=counts, total=total,
                        first=min((c["first"] for c in counts), default=None), last=max((c["last"] for c in counts), default=None),
                        timeline=list(timeline.values()), hidden=total - in_window, events=events, lessons=lessons,
                        related=related, kind=kind, agg=await _aggregates(session, slug),
                        me=store.get("recall").get("self_entity"), cap=store.get("recall").get("pinned_cap"))


# ---------------------------------------------------------------- queue

@router.get("/queue", response_class=HTMLResponse)
async def queue_page(request: Request, status: str = "", agent: str = "", page: int = 1,
                     session: AsyncSession = Depends(get_session)):
    from src.memory import batch
    where, args = ["true"], {}
    if status:
        where.append("status = :st")
        args["st"] = status
    if agent:
        where.append("agent = :ag")
        args["ag"] = agent
    per = 40
    args.update(lim=per + 1, off=(max(page, 1) - 1) * per)
    jobs = (await session.execute(sql(f"""select id, status, agent, attempts, error, result, created_at, updated_at,
        length(text) chars, left(text, 200) preview from ingest_jobs where {' and '.join(where)}
        order by id desc limit :lim offset :off"""), args)).mappings().all()
    counts = dict((await session.execute(sql("select status, count(*) from ingest_jobs group by status"))).all())
    agents = (await session.execute(sql("select agent, count(*) n from ingest_jobs group by agent order by n desc"))).all()
    notes = (await session.execute(sql("""select session_id, key, value, expires_at from session_state
        where expires_at > now() order by created_at desc limit 20"""))).mappings().all()
    return await render(request, "queue.html", session, "queue", jobs=jobs[:per], more=len(jobs) > per, page=page,
                        counts=counts, agents=agents, status=status, agent=agent, notes=notes,
                        batch={"on": await batch.is_on(session), "model": batch.model(), "available": batch.available()},
                        extract=[f"{n} · {m or llm.default_model(n)}" for n, m in llm.steps("extract")])


# ---------------------------------------------------------------- recall inspector

@router.get("/recall", response_class=HTMLResponse)
async def recall_page(request: Request, situation: str = "", session: AsyncSession = Depends(get_session)):
    return await render(request, "recall.html", session, "recall", situation=situation, recall=store.get("recall"))


# ---------------------------------------------------------------- old addresses

@router.get("/memories")
async def legacy_memories():
    return RedirectResponse("/facts", 301)
