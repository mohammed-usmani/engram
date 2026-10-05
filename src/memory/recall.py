"""recall(situation): blend every memory layer into one token-budgeted brief.

plan (entities, weights) → channels in sequence over one DB session →
rank (RRF × kind weight × importance × recency × confidence × entity boost) →
pack pinned sections first, then greedy by score under the budget.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.ext.asyncio import AsyncSession

from src.memory import facts, local_date
from src.memory.entities import match_text
from src.memory.llm import complete_json
from src.memory.models import Entity, Episode, MemoryBlock, Reflection, SessionState
from src.services.ollama_client import embed

from src.dates import format_line, upcoming as upcoming_dates
from src import settings_store

log = logging.getLogger(__name__)

ENTITY_SIM = 0.6          # situation ↔ entity embedding; measured: hits ≥0.64, misses ≤0.51
RRF_K = 60
MAX_PER_EPISODE_KIND = 3  # counts live in the aggregates; list only a few examples per kind
MIN_SIM = 0.5             # vector hits below this are noise, not memory about the situation
HALF_LIFE_DAYS = {"episode": 30, "reflection": 90}
DEFAULT_WEIGHTS = {"episode": 1.0, "fact": 1.0, "preference": 1.0, "procedure": 0.8, "reflection": 1.2, "document": 0.6}
SECTIONS = [("reflection", "Lessons"), ("episode", "Relevant history"), ("fact", "What you know"),
            ("preference", "Preferences"), ("procedure", "How you do it"), ("document", "Documents")]


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


@dataclass
class Item:
    id: str
    kind: str
    text: str
    importance: int = 3
    when: datetime | None = None
    confidence: float = 1.0
    entities: list[str] = field(default_factory=list)
    rrf: float = 0.0
    score: float = 0.0
    text_kind: str = ""  # episode kind (applied, interview…) for per-kind caps


# ---------------------------------------------------------------- planning

_plan_cache: dict[str, dict] = {}

_PLAN_PROMPT = """The user said this to their AI assistant: "{situation}"
Known memory entities: {known}
Which entities are relevant, and how much should each memory kind matter here (0.5 = less useful, 1 = normal, 2 = crucial)?
Return JSON: {{"entities": [exact names from the known list], "weights": {{"episode": n, "fact": n, "preference": n,
"procedure": n, "reflection": n, "document": n}}}}"""


async def profile_params(session: AsyncSession, budget_tokens: int = 1500) -> tuple[list[str], int]:
    """(the user's own names, pinned budget) — shared by recall and the Facts page's profile preview."""
    cap = (settings_store.get("recall") or {}).get("pinned_cap")
    pinned_budget = int(budget_tokens * cap) if cap else budget_tokens
    me = (settings_store.get("recall") or {}).get("self_entity")
    names: list[str] = []
    if me:
        row = (await session.execute(select(Entity.name, Entity.aliases).where(Entity.slug == me))).first()
        names = [row[0], *(row[1] or [])] if row else []
    return names, pinned_budget


def _sane_weights(raw) -> dict:
    """The planner only nudges ranking. It can't switch a kind off: a weight of 0 zeroes every score of that
    kind (seen: all zeros, so the brief got no memories at all). Clamp to 0.3–2; ignore an answer with no signal."""
    if not isinstance(raw, dict):
        return {}
    w = {k: float(v) for k, v in raw.items() if k in DEFAULT_WEIGHTS and isinstance(v, (int, float))}
    if not w or max(w.values()) <= 0:
        return {}
    return {k: min(2.0, max(0.3, v)) for k, v in w.items()}


async def _plan(session: AsyncSession, situation: str, vec: list[float], fast: bool) -> dict:
    slugs = set(await match_text(session, situation))
    rows = [] if vec is None else (await session.execute(
        select(Entity.slug, 1 - Entity.embedding.cosine_distance(vec))
        .where(Entity.embedding.isnot(None)).order_by(Entity.embedding.cosine_distance(vec)).limit(3)
    )).all()
    slugs |= {s for s, sim in rows if float(sim) >= ENTITY_SIM}
    weights = dict(DEFAULT_WEIGHTS)

    if not fast:
        key = hashlib.sha256(situation.encode()).hexdigest()
        if key not in _plan_cache:
            known = dict((await session.execute(select(Entity.name, Entity.slug).order_by(Entity.updated_at.desc()).limit(80))).all())
            try:
                p = await complete_json(_PLAN_PROMPT.format(situation=situation, known=", ".join(known) or "none"), task="plan")
                _plan_cache[key] = {"entities": [known[n] for n in p.get("entities", []) if n in known],
                                    "weights": _sane_weights(p.get("weights"))}
            except Exception as e:  # planner is an optimisation; the fast plan still works
                log.info("planner skipped: %s", e)
                _plan_cache[key] = {"entities": [], "weights": {}}
        slugs |= set(_plan_cache[key]["entities"])
        weights.update(_plan_cache[key]["weights"])

    # 1-hop graph expansion: entities that co-occur with matched ones in ≥2 episodes
    if slugs:
        co = Counter()
        me = (settings_store.get("recall") or {}).get("self_entity")
        for (ents,) in (await session.execute(
            select(Episode.entities).where(Episode.entities.has_any(array(list(slugs))))
        )).all():
            co.update(e for e in ents if e not in slugs and e != me)
        slugs |= {e for e, n in co.most_common(2) if n >= 2}
    me = (settings_store.get("recall") or {}).get("self_entity")
    excluded = [me] if me and me in slugs else []
    slugs -= set(excluded)  # the user is in nearly every event: boosting or summarizing them is noise
    return {"entities": sorted(slugs), "weights": weights, "fast": fast, "excluded": excluded}


# ---------------------------------------------------------------- channels

def _rrf(ranked: list[Item], into: dict[str, Item]) -> None:
    for rank, it in enumerate(ranked):
        cur = into.setdefault(it.id, it)
        cur.rrf += 1.0 / (RRF_K + rank + 1)


async def _aggregates(session: AsyncSession, slug: str) -> str | None:
    """Counts, first/last dates and outcome split for one entity — the numbers vectors can't give."""
    has = Episode.entities.has_key(slug)
    first, last, total = (await session.execute(
        select(func.min(Episode.occurred_at), func.max(Episode.occurred_at), func.count()).where(has))).one()
    if not total:
        return None
    by_kind = (await session.execute(
        select(Episode.kind, func.count()).where(has).group_by(Episode.kind).order_by(func.count().desc()))).all()
    outcomes = (await session.execute(
        select(Episode.outcome, func.count()).where(has, Episode.outcome.isnot(None))
        .group_by(Episode.outcome).order_by(func.count().desc()))).all()
    days = (datetime.now(timezone.utc) - first).days
    line = (f"{total} events since {local_date(first)} ({days}d), last {local_date(last)} · "
            + " · ".join(f"{k} {n}" for k, n in by_kind))
    if outcomes:
        line += " · outcomes: " + ", ".join(f"{o} {n}" for o, n in outcomes)
    return line


async def _episodes(session: AsyncSession, situation: str, vec, slugs: list[str]) -> list[list[Item]]:
    def item(e: Episode) -> Item:
        return Item(f"e:{e.id}", "episode", f"{local_date(e.occurred_at)} {e.summary}"
                    + (f" (outcome: {e.outcome})" if e.outcome else ""),
                    e.importance, e.occurred_at, 1.0, e.entities, text_kind=e.kind)

    by_vec = []
    if vec is not None:
        dist = Episode.embedding.cosine_distance(vec)
        by_vec = (await session.execute(
            select(Episode).where(Episode.embedding.isnot(None), dist <= 1 - MIN_SIM)
            .order_by(dist).limit(15))).scalars().all()
    tsq = func.plainto_tsquery("english", situation)
    by_fts = (await session.execute(
        select(Episode).where(Episode.search_vector.op("@@")(tsq))
        .order_by(func.ts_rank(Episode.search_vector, tsq).desc()).limit(15))).scalars().all()
    lists = [[item(e) for e in by_vec], [item(e) for e in by_fts]]
    if slugs:
        recent = (await session.execute(
            select(Episode).where(Episode.entities.has_any(array(slugs)))
            .order_by(Episode.importance.desc(), Episode.occurred_at.desc()).limit(15))).scalars().all()
        lists.append([item(e) for e in recent])
    return lists


async def _reflections(session: AsyncSession, vec, slugs: list[str]) -> list[list[Item]]:
    def item(r: Reflection) -> Item:
        return Item(f"r:{r.id}", "reflection", r.lesson, 4, r.updated_at, r.confidence, r.entities)
    by_vec = []
    if vec is not None:
        dist = Reflection.embedding.cosine_distance(vec)
        by_vec = (await session.execute(
            select(Reflection).where(Reflection.embedding.isnot(None), dist <= 1 - MIN_SIM)
            .order_by(dist).limit(8))).scalars().all()
    lists = [[item(r) for r in by_vec]]
    if slugs:
        tagged = (await session.execute(
            select(Reflection).where(Reflection.entities.has_any(array(slugs)))
            .order_by(Reflection.confidence.desc()).limit(8))).scalars().all()
        lists.append([item(r) for r in tagged])
    return lists


async def _facts(situation: str) -> list[list[Item]]:
    try:
        hits = await facts.similar(situation, top_k=15)
    except Exception as e:
        log.warning("fact channel failed: %s", e)
        return []
    out: dict[str, list[Item]] = defaultdict(list)
    for h in hits:
        if (h["score"] or 0) < MIN_SIM:
            continue
        md = h["metadata"]
        kind = md.get("kind", "fact")
        out[kind].append(Item(f"f:{h['id']}", kind, h["memory"], int(md.get("importance", 3)), None, 1.0,
                              md.get("entities") or []))
    return list(out.values())


async def _documents(session: AsyncSession, situation: str) -> list[list[Item]]:
    try:
        from src.services.retrieval import retrieve
        docs = await retrieve(situation, session, limit=3)
    except Exception as e:
        log.info("document channel skipped: %s", e)
        return []
    def first_line(text: str) -> str:
        lines = [l.strip() for l in text.splitlines() if l.strip() and not set(l.strip()) <= set("=-#*")]
        body = next((l for l in lines if len(l) > 40 and ":" not in l[:20]), lines[0] if lines else "")
        return body[:160]
    return [[Item(f"d:{d.slug}", "document", f"{d.title} ({d.type}) — {first_line(d.content)}", 3)
             for d in docs if d.score >= 0.5]]


# ---------------------------------------------------------------- ranking + packing

def _score(it: Item, weights: dict, slugs: set[str], now: datetime) -> float:
    s = it.rrf * weights.get(it.kind, 1.0) * (it.importance / 3) * it.confidence
    half = HALF_LIFE_DAYS.get(it.kind)
    if half and it.when:
        s *= 0.5 ** (max((now - it.when).total_seconds(), 0) / 86400 / half)
    if slugs and set(it.entities) & slugs:
        s *= 1.5
    return s


def _similar_text(a: str, b: str) -> bool:
    ta, tb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) > 0.8


async def recall(session: AsyncSession, situation: str, budget_tokens: int = 1500,
                 session_id: str | None = None, fast: bool = False, explain: bool = False) -> dict:
    """explain=True (the Recall inspector) also returns where the budget went and every candidate's
    score factors, and doesn't count as the items being served."""
    now = datetime.now(timezone.utc)
    try:
        vec = await embed(situation)
    except Exception as e:  # Ollama down: keep going on profile, notes, aggregates and full-text
        log.warning("recall without embeddings: %s", e)
        vec = None
    plan = await _plan(session, situation, vec, fast)
    slugs = plan["entities"]

    # pinned: profile, session state, per-entity digest + aggregates
    pinned: list[str] = []
    me_names, pinned_budget = await profile_params(session, budget_tokens)
    try:
        # the profile gets at most half the pinned space, so dates and topic summaries always fit too
        live = await facts.profile(names=me_names, max_tokens=pinned_budget // 2)
    except Exception as e:  # fact store down: fall back to the last snapshot from the sleep pass
        log.warning("live profile unavailable: %s", e)
        live = ""
    prof = await session.get(MemoryBlock, "profile")
    if live and (prof is None or prof.content != live):
        # keep the fallback snapshot as fresh as the last successful recall
        if prof is None:
            session.add(MemoryBlock(name="profile", content=live))
        else:
            prof.content = live
        await session.commit()
    pinned.append("## You\n" + (live or (prof.content if prof else "(no profile yet — memory is still learning about you)")))
    try:
        soon, _ = await upcoming_dates(session, days=60)
    except Exception as e:  # never let a bad date field break recall
        log.warning("upcoming dates unavailable: %s", e)
        soon = []
    if soon:
        pinned.append("## Coming up\n" + "\n".join(format_line(i) for i in soon[:8]))
    if session_id:
        notes = (await session.execute(select(SessionState).where(
            SessionState.session_id == session_id, SessionState.expires_at > now))).scalars().all()
        if notes:
            pinned.append("## Current task\n" + "\n".join(f"- {n.key}: {n.value}" for n in notes))
    for slug in slugs:
        e = (await session.execute(select(Entity).where(Entity.slug == slug))).scalar_one_or_none()
        agg = await _aggregates(session, slug)
        if e and (e.digest or agg):
            pinned.append(f"## {e.name} so far\n" + "\n".join(x for x in (agg, e.digest) if x))

    pool: dict[str, Item] = {}
    for ranked in [*await _episodes(session, situation, vec, slugs), *await _reflections(session, vec, slugs),
                   *await _facts(situation), *await _documents(session, situation)]:
        _rrf(ranked, pool)
    for it in pool.values():
        it.score = _score(it, plan["weights"], set(slugs), now)

    # pack: pinned first, up to the pinned cap (Settings: room is left for ranked memories),
    # then best score per token
    used = 0
    kept_pinned = []
    sections = []
    for block in pinned:
        full = estimate_tokens(block)
        cost = full
        if used + cost > pinned_budget:
            # only the profile is cut to fit; a summary that doesn't fit whole is left out, not chopped mid-sentence
            block = block[: max(0, (pinned_budget - used) * 4 - 8)] if not kept_pinned else ""
            cost = estimate_tokens(block) if block else 0
        if explain:
            sections.append({"title": pinned_title(block or pinned[len(sections)]), "tokens": cost, "full": full,
                             "pinned": True})
        if block:
            kept_pinned.append(block)
            used += cost
    chosen: list[Item] = []
    for it in sorted(pool.values(), key=lambda i: i.score / estimate_tokens(i.text), reverse=True):
        if it.score <= 0 or any(_similar_text(it.text, c.text) for c in chosen):
            continue
        if it.kind == "episode":
            ep_kind = it.text_kind
            if sum(1 for c in chosen if c.kind == "episode" and c.text_kind == ep_kind) >= MAX_PER_EPISODE_KIND:
                continue
        cost = estimate_tokens(f"- [{it.id}] {it.text}\n") + (6 if not any(c.kind == it.kind for c in chosen) else 0)
        if used + cost > budget_tokens:
            continue
        chosen.append(it)
        used += cost

    parts = list(kept_pinned)
    for kind, title in SECTIONS:
        rows = [c for c in chosen if c.kind == kind]
        if kind == "episode":  # history reads best newest-first
            rows.sort(key=lambda c: c.when, reverse=True)
        else:
            rows.sort(key=lambda c: c.score, reverse=True)
        if rows:
            parts.append(f"## {title}\n" + "\n".join(f"- [{c.id}] {c.text}" for c in rows))
            if explain:
                sections.append({"title": title, "tokens": estimate_tokens(parts[-1]), "pinned": False})
    brief = "\n\n".join(parts)

    out = {"brief": brief, "items": [c.id for c in chosen], "plan": plan}
    if explain:
        picked = {c.id for c in chosen}
        ranked = sorted(pool.values(), key=lambda i: i.score, reverse=True)[:40]
        out["explain"] = {
            "budget": budget_tokens, "pinned_budget": pinned_budget, "used": estimate_tokens(brief),
            "sections": sections,
            "candidates": [{**_factors(it, plan, set(slugs), now), "in_brief": it.id in picked} for it in ranked],
        }
        return out
    await _touch(session, [c.id for c in chosen])
    return out


def pinned_title(block: str) -> str:
    return block.split("\n", 1)[0].removeprefix("## ").strip()


def _factors(it: Item, plan: dict, slugs: set[str], now: datetime) -> dict:
    half = HALF_LIFE_DAYS.get(it.kind)
    decay = 0.5 ** (max((now - it.when).total_seconds(), 0) / 86400 / half) if half and it.when else 1.0
    return {"id": it.id, "kind": it.kind, "text": it.text, "score": round(it.score, 5), "rrf": round(it.rrf, 5),
            "weight": plan["weights"].get(it.kind, 1.0), "importance": it.importance,
            "confidence": round(it.confidence, 2), "recency": round(decay, 2),
            "boost": bool(slugs and set(it.entities) & slugs), "tokens": estimate_tokens(it.text)}


async def _touch(session: AsyncSession, ids: list[str]) -> None:
    """Served items get access stats; consolidation uses them to rank the profile and archive."""
    now = datetime.now(timezone.utc)
    for model, prefix in ((Episode, "e:"), (Reflection, "r:")):
        nums = [int(i[2:]) for i in ids if i.startswith(prefix)]
        if nums:
            await session.execute(update(model).where(model.id.in_(nums))
                                  .values(access_count=model.access_count + 1, last_accessed=now))
    await session.commit()


# ---------------------------------------------------------------- item tools

def _row(obj) -> dict:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns if c.name not in ("embedding", "search_vector")}


async def expand(session: AsyncSession, item_id: str) -> dict | None:
    prefix, _, key = item_id.partition(":")
    if prefix == "e" and key.isdigit():
        e = await session.get(Episode, int(key))
        return _row(e) if e else None
    if prefix == "r" and key.isdigit():
        r = await session.get(Reflection, int(key))
        if not r:
            return None
        ev = (await session.execute(select(Episode).where(Episode.id.in_(r.evidence)))).scalars().all()
        return {**_row(r), "evidence_episodes": [_row(e) for e in ev]}
    if prefix == "f":
        return await facts.get_fact(key)
    if prefix == "d":
        from src.models import ContextDocument
        from src.privacy import readable
        d = (await session.execute(select(ContextDocument).where(
            ContextDocument.slug == key, readable(ContextDocument.privacy)))).scalar_one_or_none()
        return {"slug": d.slug, "title": d.title, "type": getattr(d.type, "value", d.type), "privacy": d.privacy,
                "content": d.content, "sections": d.sections} if d else None
    return None


async def timeline(session: AsyncSession, entity: str, since: datetime | None = None, limit: int = 100) -> list[dict]:
    q = select(Episode).where(Episode.entities.has_key(entity))
    if since:
        q = q.where(Episode.occurred_at >= since)
    rows = (await session.execute(q.order_by(Episode.occurred_at).limit(limit))).scalars().all()
    return [_row(e) for e in rows]


async def forget(session: AsyncSession, item_id: str) -> bool:
    """Hard delete — it's the user's data. Documents are managed via the admin UI instead."""
    prefix, _, key = item_id.partition(":")
    if prefix in ("e", "r") and key.isdigit():
        model = Episode if prefix == "e" else Reflection
        n = (await session.execute(delete(model).where(model.id == int(key)))).rowcount
        await session.commit()
        return bool(n)
    if prefix == "f" and await facts.get_fact(key):
        await facts.delete_fact(key)
        return True
    return False
