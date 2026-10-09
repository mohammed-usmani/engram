"""Ask Engram (a chat that answers from the whole memory) and global search."""
from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlencode
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from src import settings_store as store
from src.database import get_session
from src.memory import facts, ingest, llm
from src.memory import recall as rc
from src.memory.models import Episode, Reflection
from src.models import ChatMessage, ChatSession, ContextDocument, MessageRole
from src.privacy import readable
from src.ui.templating import render

log = logging.getLogger(__name__)
router = APIRouter(include_in_schema=False)

ID_RE = re.compile(r"\[((?:e|r):\d+|f:[0-9a-fA-F-]{8,}|d:[^\]\s]+)\]")
SYSTEM = """You are Engram, answering the user's questions about their own life and work.
Use ONLY the memory brief below; it is everything you know about them. When a statement relies on a
memory item, cite its id in square brackets exactly as it appears in the brief, e.g. [e:12], [f:<id>],
[r:3], [d:<slug>]. If the brief doesn't contain the answer, say plainly that memory doesn't have it;
never guess or invent. Be concise.

# Memory brief
{brief}"""


# ---------------------------------------------------------------- helpers

def link_for(item_id: str, text: str) -> str:
    """The page an item lives on."""
    prefix, _, key = item_id.partition(":")
    if prefix == "f":
        return "/facts?" + urlencode({"q": " ".join(text.split()[:5])})
    return {"e": f"/events?focus={key}", "r": "/lessons", "d": f"/api/admin/{key}"}.get(prefix, "")


async def _composer(session: AsyncSession, current: tuple[str, str | None] | None) -> dict:
    """Providers with a key and their models (for the "Answer with" select), and the rest."""
    from src.routers import settings as api
    usable, missing = [], []
    for p in await api.list_providers(session):
        if not p["api_key_set"]:
            missing.append(p["label"])
            continue
        models = [m for m in await api._models_for(p["provider"]) if "embed" not in m]  # embedders can't chat
        if p["effective_model"] and p["effective_model"] not in models:
            models.insert(0, p["effective_model"])
        usable.append({"provider": p["provider"], "label": p["label"], "default": p["effective_model"],
                       "models": models or [p["effective_model"] or ""]})
    cur = current or _default()
    selected = f"{cur[0]}|{cur[1] or llm.default_model(cur[0]) or ''}"
    return {"usable": usable, "missing": missing, "selected": selected}


def _default() -> tuple[str, str | None]:
    steps = llm.steps("chat")
    return steps[0] if steps else ("ollama", None)


async def _suggestions(session: AsyncSession) -> list[str]:
    """Four questions from what's actually in memory: dates coming up, then the busiest topics."""
    from src.dates import upcoming
    out = []
    try:
        soon, _ = await upcoming(session, days=60)
    except Exception:  # a bad date field must not break the page
        soon = []
    for i in soon[:2]:
        out.append(f"What do I need to sort out before {i['title']} ({i['label']} {i['date']})?")
    me = (store.get("recall") or {}).get("self_entity")
    rows = (await session.execute(sql("""
        select e.name, e.kind from entities e join episodes x on x.entities ? e.slug
        where x.occurred_at > now() - interval '30 days' and e.slug is distinct from :me
        group by e.id order by count(*) desc limit 8"""), {"me": me})).all()
    templates = ["What's the latest on {}?", "What have I decided about {} so far?",
                 "What did I do on {} recently?", "Summarize everything I know about {}."]
    for i, (name, _kind) in enumerate(rows):
        if len(out) >= 4:
            break
        out.append(templates[i % len(templates)].format(name))
    return out


async def _conversations(session: AsyncSession) -> list[ChatSession]:
    return list((await session.execute(
        select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(30))).scalars())


async def _cited(session: AsyncSession, answer: str) -> list[dict]:
    """Items an old answer cites, looked up again (only the live answer has the full used list)."""
    out = []
    for item_id in dict.fromkeys(ID_RE.findall(answer)):
        try:
            row = await rc.expand(session, item_id)
        except Exception as e:  # fact store down: still list the id
            log.info("expand %s failed: %s", item_id, e)
            row = {}
        if row is None:
            out.append({"id": item_id, "kind": "gone", "text": "No longer in memory.", "href": ""})
            continue
        text = (row.get("summary") or row.get("lesson") or row.get("memory") or row.get("title") or "")
        kind = {"e": "event", "r": "lesson", "f": "fact", "d": "document"}[item_id[0]]
        out.append({"id": item_id, "kind": kind, "text": text, "href": link_for(item_id, text)})
    return out


# ---------------------------------------------------------------- pages

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, session: AsyncSession = Depends(get_session)):
    return await render(request, "ask.html", session, "chat", conversation=None, messages=[],
                        conversations=await _conversations(session), suggestions=await _suggestions(session),
                        composer=await _composer(session, None))


@router.get("/chat/{conversation_id}", response_class=HTMLResponse)
async def conversation_page(conversation_id: int, request: Request, session: AsyncSession = Depends(get_session)):
    conv = await session.get(ChatSession, conversation_id)
    if conv is None:
        raise HTTPException(404, "That conversation doesn't exist (it may have been deleted).")
    msgs = (await session.execute(select(ChatMessage).where(ChatMessage.session_id == conv.id)
                                  .order_by(ChatMessage.created_at, ChatMessage.id))).scalars().all()
    messages = [{"role": m.role.value, "content": m.content, "at": m.created_at,
                 "used": await _cited(session, m.content) if m.role == MessageRole.ASSISTANT else []}
                for m in msgs]
    return await render(request, "ask.html", session, "chat", conversation=conv, messages=messages,
                        conversations=await _conversations(session), suggestions=[],
                        composer=await _composer(session, (conv.provider, conv.model) if conv.provider else None))


# ---------------------------------------------------------------- API

class AskIn(BaseModel):
    message: str = Field(..., max_length=20_000)
    conversation_id: int | None = None
    provider: str | None = None
    model: str | None = None


@router.post("/api/ask")
async def ask(body: AskIn, session: AsyncSession = Depends(get_session)):
    from src.services.providers import list_provider_names
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "Type a question first.")
    conv = None
    if body.conversation_id is not None:
        conv = await session.get(ChatSession, body.conversation_id)
        if conv is None:
            raise HTTPException(404, "That conversation doesn't exist (it may have been deleted).")

    if body.provider:
        provider, model = body.provider, body.model or None
    elif conv and conv.provider:
        provider, model = conv.provider, conv.model
    else:
        provider, model = _default()
    if provider not in list_provider_names():
        raise HTTPException(400, f"Unknown provider '{provider}'.")
    if not llm.key_source(provider):
        raise HTTPException(400, f"{provider} has no API key. Add one in Settings, or pick another model.")
    model = model or llm.default_model(provider)

    history = []
    if conv:
        last = (await session.execute(select(ChatMessage).where(ChatMessage.session_id == conv.id)
                                      .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                                      .limit(10))).scalars().all()
        history = [{"role": m.role.value, "content": m.content} for m in reversed(last)]

    result = await rc.recall(session, message, budget_tokens=2000, fast=True, explain=True, source="ask")
    used = [{"id": c["id"], "kind": c["kind"], "text": c["text"]}
            for c in result["explain"]["candidates"] if c["in_brief"]]
    try:
        answer = await asyncio.wait_for(
            llm.make(provider, model).generate(message, system=SYSTEM.format(brief=result["brief"]), history=history),
            timeout=180)
    except Exception as e:
        raise HTTPException(502, f"{provider} · {model} didn't answer: {(str(e) or type(e).__name__)[:300]}")
    answer = (answer or "").strip() or "(The model returned an empty answer.)"

    if conv is None:
        conv = ChatSession(title=message if len(message) <= 80 else message[:79] + "…", message_count=0)
        session.add(conv)
    conv.provider, conv.model = provider, model
    conv.message_count = (conv.message_count or 0) + 2
    conv.updated_at = datetime.now(timezone.utc)
    await session.flush()
    session.add_all([ChatMessage(session_id=conv.id, role=MessageRole.USER, content=message),
                     ChatMessage(session_id=conv.id, role=MessageRole.ASSISTANT, content=answer)])
    await session.commit()
    await ingest.enqueue(session, "user (Ask Engram): " + message, agent="engram-chat",
                         session_id=f"ask-{conv.id}", occurred_at=datetime.now(timezone.utc))
    return {"conversation_id": conv.id, "title": conv.title, "answer": answer, "provider": provider, "model": model,
            "used": [{**u, "href": link_for(u["id"], u["text"])} for u in used],
            "trace_id": result.get("trace_id")}


@router.delete("/api/ask/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, session: AsyncSession = Depends(get_session)):
    await session.execute(delete(ChatMessage).where(ChatMessage.session_id == conversation_id))
    n = (await session.execute(delete(ChatSession).where(ChatSession.id == conversation_id))).rowcount
    await session.commit()
    if not n:
        raise HTTPException(404, "That conversation doesn't exist.")
    return {"deleted": conversation_id}


# ---------------------------------------------------------------- search

@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: str = "", session: AsyncSession = Depends(get_session)):
    q = q.strip()
    res = {"documents": [], "topics": [], "events": [], "facts": [], "lessons": []}
    if q:
        like = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        tsq = func.websearch_to_tsquery("english", q)
        res["documents"] = (await session.execute(
            select(ContextDocument.slug, ContextDocument.title, ContextDocument.type, ContextDocument.privacy,
                   func.ts_headline("english", ContextDocument.content, tsq,
                                    'StartSel="",StopSel="",MaxWords=30,MinWords=12').label("snippet"))
            .where(ContextDocument.search_vector.op("@@")(tsq), readable(ContextDocument.privacy))
            .order_by(func.ts_rank(ContextDocument.search_vector, tsq).desc()).limit(20))).mappings().all()
        res["topics"] = (await session.execute(sql("""
            select e.slug, e.name, e.kind, count(x.id) n from entities e left join episodes x on x.entities ? e.slug
            where e.name ilike :p or e.slug ilike :p or e.aliases::text ilike :p
            group by e.id order by n desc, e.name limit 20"""), {"p": like})).mappings().all()
        res["events"] = (await session.execute(
            select(Episode.id, Episode.kind, Episode.summary, Episode.occurred_at)
            .where(Episode.search_vector.op("@@")(tsq))
            .order_by(func.ts_rank(Episode.search_vector, tsq).desc(), Episode.occurred_at.desc())
            .limit(20))).mappings().all()
        res["lessons"] = (await session.execute(
            select(Reflection.id, Reflection.lesson, Reflection.confidence)
            .where(Reflection.lesson.ilike(like)).order_by(Reflection.confidence.desc()).limit(20))).mappings().all()
        try:
            hits = [h for h in await facts.similar(q, top_k=10) if (h["score"] or 0) >= rc.MIN_SIM]
            res["facts"] = [{"id": h["id"], "text": h["memory"], "kind": h["metadata"].get("kind", "fact")}
                            for h in hits]
        except Exception as e:  # Ollama down: plain text match instead
            log.info("fact search falling back to SQL: %s", e)
            res["facts"] = [dict(r) for r in (await session.execute(sql(
                f"select id::text, payload->>'data' as text, coalesce(payload->>'kind', 'fact') as kind "
                f"from {facts.table()} where payload->>'superseded_by' is null and payload->>'data' ilike :p "
                f"limit 20"), {"p": like})).mappings().all()]
        for f in res["facts"]:
            f["href"] = link_for("f:" + f["id"], f["text"])
    total = sum(len(v) for v in res.values())
    return await render(request, "search.html", session, "", q=q, q_global=q, res=res, total=total)
