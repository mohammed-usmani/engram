"""The "sleep" pass: episodes → entity digests + reflections; facts → profile; expire session state.

Nothing here deletes memory; expired session notes are promoted to episodes first.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.memory import facts, local_date
from src.memory.llm import complete_json
from src.memory.models import Entity, Episode, MemoryBlock, Reflection, SessionState
from src.memory.recall import _aggregates
from src.services.ollama_client import embed

log = logging.getLogger(__name__)

MIN_EPISODES_FOR_LLM = 2
MIN_EVIDENCE = 2
REFLECTION_MERGE_SIM = 0.85
PROFILE_FACTS = 15

_PROMPT = """You maintain long-term memory about the user for their AI assistants.
Entity: {name} ({kind})
Numbers: {agg}
Existing lessons: {lessons}
Recent events (id · date · kind · summary · outcome):
{events}

Return JSON with keys "digest" (2-3 sentences: where things stand now, the trend, what matters next)
and "reflections" (list of objects with "lesson" = a specific pattern you see in these events,
"evidence" = list of supporting event ids, "confidence" = 0 to 1).
Only include lessons supported by at least {min_ev} events and specific to these events — no generic advice.
Re-state existing lessons if still true. Use [] when there is no real pattern."""

_TEMPLATE_ECHO = ("specific pattern", "pattern or conclusion", "2-3 sentences", "where things stand")


def _echo(text: str) -> bool:
    low = text.lower()
    return len(low) < 15 or any(t in low for t in _TEMPLATE_ECHO)


async def _entity(session: AsyncSession, e: Entity) -> int:
    agg = await _aggregates(session, e.slug)
    eps = (await session.execute(
        select(Episode).where(Episode.entities.has_key(e.slug)).order_by(Episode.occurred_at.desc()).limit(25)
    )).scalars().all()
    if len(eps) < MIN_EPISODES_FOR_LLM:
        e.dirty = False  # numbers are computed live at recall; nothing to summarise yet
        return 0
    lessons = (await session.execute(select(Reflection).where(Reflection.entities.has_key(e.slug)))).scalars().all()
    try:
        out = await complete_json(_PROMPT.format(
            name=e.name, kind=e.kind, agg=agg, min_ev=MIN_EVIDENCE,
            lessons="; ".join(r.lesson for r in lessons) or "none",
            events="\n".join(f"{x.id} · {local_date(x.occurred_at)} · {x.kind} · {x.summary} · {x.outcome or ''}" for x in eps),
        ), task="consolidate")
    except Exception as ex:
        log.warning("consolidate %s deferred: %s", e.slug, ex)
        return 0  # stays dirty so the LLM part retries next run; recall still shows live numbers

    digest = str(out.get("digest") or "").strip()
    e.digest = digest if digest and not _echo(digest) else e.digest
    e.dirty = False
    valid_ids = {x.id for x in eps}
    made = 0
    for r in out.get("reflections") or []:
        if not isinstance(r, dict) or not isinstance(r.get("lesson"), str) or _echo(r["lesson"]):
            continue
        ev = sorted({i for i in r.get("evidence") or [] if isinstance(i, int) and i in valid_ids})
        if len(ev) < MIN_EVIDENCE:
            continue
        conf = r.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) and 0 <= conf <= 1 else 0.5
        vec = await embed(r["lesson"])
        same = (await session.execute(
            select(Reflection, 1 - Reflection.embedding.cosine_distance(vec))
            .where(Reflection.embedding.isnot(None)).order_by(Reflection.embedding.cosine_distance(vec)).limit(1)
        )).first()
        if same and float(same[1]) >= REFLECTION_MERGE_SIM:
            ref = same[0]
            ref.lesson, ref.confidence, ref.embedding = r["lesson"], conf, vec
            ref.evidence = sorted(set(ref.evidence) | set(ev))
            ref.entities = sorted(set(ref.entities) | {e.slug})
        else:
            session.add(Reflection(lesson=r["lesson"], evidence=ev, confidence=conf, entities=[e.slug], embedding=vec))
        made += 1
    return made


async def _profile(session: AsyncSession) -> None:
    rows = await facts.all_facts(kinds=["fact", "preference"])
    rows.sort(key=lambda f: (int(f["metadata"].get("importance", 3)), f.get("updated_at") or ""), reverse=True)
    if not rows:
        return
    content = "\n".join(f"- {f['memory']}" for f in rows[:PROFILE_FACTS])
    block = await session.get(MemoryBlock, "profile")
    if block:
        block.content = content
    else:
        session.add(MemoryBlock(name="profile", content=content))


async def _expire_sessions(session: AsyncSession) -> int:
    now = datetime.now(timezone.utc)
    old = (await session.execute(select(SessionState).where(SessionState.expires_at <= now))).scalars().all()
    for n in old:
        session.add(Episode(occurred_at=n.created_at, kind="session_note", importance=2, entities=[],
                            summary=f"Session note ({n.session_id}) — {n.key}: {n.value}"))
        await session.delete(n)
    return len(old)


async def consolidate(session: AsyncSession, max_entities: int = 25) -> dict:
    dirty = (await session.execute(select(Entity).where(Entity.dirty).limit(max_entities))).scalars().all()
    made = 0
    for e in dirty:
        made += await _entity(session, e)
        await session.commit()
    await _profile(session)
    expired = await _expire_sessions(session)
    await session.commit()
    return {"entities": len(dirty), "reflections": made, "expired_notes": expired}
