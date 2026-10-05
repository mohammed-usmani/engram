"""Cleanup suggestions (same topic under two names, duplicate events, stray kinds, one-off topics)
and the confirmed fixes. Suggestions only read; every fix keeps events and records old names as aliases."""
from __future__ import annotations

import difflib
import json
import re
import time
from collections import defaultdict

from fastapi import HTTPException
from sqlalchemy import select, text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from src import settings_store as store
from src.memory import USER_TZ, facts
from src.memory.entities import CANON_KINDS, slugify
from src.memory.models import Entity

TOOLISH = {"device", "platform", "model", "service", "script", "language", "account"}
PROJECTISH = {"feature", "product", "app", "widget", "page"}
DUPE_SIM = 0.84  # spec said 0.85; the known Redfox duplicate pair (e:897, e:912) scores 0.847


def suggest_kind(kind: str) -> str:
    return "project" if kind in PROJECTISH else "tool" if kind in TOOLISH else "topic"


def _tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.lower())


def _uniq(xs):
    return list(dict.fromkeys(xs))


# ---------------------------------------------------------------- reading

async def _load(session: AsyncSession) -> tuple[dict[str, dict], dict[str, set[int]]]:
    ents = {r["slug"]: dict(r) for r in (await session.execute(
        sql("select slug, kind, name, aliases from entities order by slug"))).mappings()}
    eps: dict[str, set[int]] = defaultdict(set)
    for eid, slugs in (await session.execute(sql("select id, entities from episodes"))).all():
        for s in slugs or []:
            eps[s].add(eid)
    return ents, eps


def _same(a: str, b: str) -> str | None:
    """Why two names of one kind look like the same thing, or None."""
    ta, tb = _tokens(a), _tokens(b)
    na, nb = "".join(ta), "".join(tb)
    if not na or not nb:
        return None
    if na == nb:
        return "Same name apart from spaces and capitals."
    short, long_ = sorted((ta, tb), key=len)
    if long_[:len(short)] == short:
        return "One name starts with the other."
    sm = difflib.SequenceMatcher(None, na, nb)
    if sm.real_quick_ratio() >= .9 and sm.quick_ratio() >= .9 and sm.ratio() >= .9:
        return "Names differ by a letter or two."
    return None


async def merge_groups(session: AsyncSession, data=None) -> list[dict]:
    ents, eps = data or await _load(session)
    dismissed = set(store.get("cleanup_dismissed") or [])
    by_kind: dict[str, list[str]] = defaultdict(list)
    for s, e in ents.items():
        by_kind[e["kind"]].append(s)
    parent: dict[str, str] = {}
    why: dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    # ponytail: O(n²) per kind with difflib prefilters; ~1k entities takes well under a second
    for slugs in by_kind.values():
        for i, a in enumerate(slugs):
            for b in slugs[i + 1:]:
                reason = _same(ents[a]["name"], ents[b]["name"])
                if reason:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
                    why.setdefault(ra, reason)
    groups: dict[str, list[str]] = defaultdict(list)
    for s in parent.keys() | set(parent.values()):
        groups[find(s)].append(s)
    out = []
    for root, members in groups.items():
        key = "merge:" + ",".join(sorted(members))
        if key in dismissed:
            continue
        items = sorted(({"slug": s, "name": ents[s]["name"], "kind": ents[s]["kind"], "n": len(eps.get(s, ()))}
                        for s in members), key=lambda x: (-x["n"], x["name"].lower()))
        shared = set.intersection(*(eps.get(s, set()) for s in members))
        note = why.get(root) or next((why[find(s)] for s in members if find(s) in why), "")
        if shared:
            note += f" They share {len(shared)} event{'s' if len(shared) != 1 else ''}."
        out.append({"key": key, "items": items, "keep": items[0]["slug"], "why": note.strip(),
                    "total": sum(i["n"] for i in items)})
    out.sort(key=lambda g: -g["total"])
    return out


async def dupe_groups(session: AsyncSession) -> list[dict]:
    pairs = (await session.execute(sql("""
        select a.id, b.id from episodes a join episodes b
          on a.id < b.id and a.kind = b.kind
         and (a.occurred_at at time zone :tz)::date = (b.occurred_at at time zone :tz)::date
         and a.entities ?| array(select jsonb_array_elements_text(b.entities))
         and a.embedding is not null and b.embedding is not null
         and 1 - (a.embedding <=> b.embedding) >= :sim"""), {"tz": str(USER_TZ), "sim": DUPE_SIM})).all()
    parent: dict[int, int] = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = defaultdict(list)
    for x in parent.keys() | set(parent.values()):
        groups[find(x)].append(x)
    if not groups:
        return []
    dismissed = set(store.get("cleanup_dismissed") or [])
    rows = {r["id"]: dict(r) for r in (await session.execute(sql("""
        select id, kind, occurred_at, summary, outcome, entities, source_agent from episodes
        where id = any(:ids)"""), {"ids": [i for g in groups.values() for i in g]})).mappings()}
    out = []
    for ids in groups.values():
        key = "dupe:" + ",".join(map(str, sorted(ids)))
        if key in dismissed:
            continue
        items = sorted((rows[i] for i in ids), key=lambda r: r["occurred_at"])
        out.append({"key": key, "items": items, "kind": items[0]["kind"], "day": items[0]["occurred_at"],
                    "agents": sorted({r["source_agent"] or "unknown" for r in items}),
                    "keep": max(items, key=lambda r: (len(r["summary"]), -r["id"]))["id"]})
    out.sort(key=lambda g: g["day"], reverse=True)
    return out


async def kinds(session: AsyncSession) -> list[dict]:
    rows = (await session.execute(sql("select kind, count(*) n from entities group by kind order by n desc, kind"))).all()
    return [{"kind": k, "n": n, "canon": k in CANON_KINDS, "to": suggest_kind(k)} for k, n in rows]


async def once(session: AsyncSession, data=None) -> list[dict]:
    """Topics with at most one event, each with the parent it would fold into (or None)."""
    ents, eps = data or await _load(session)
    me = (store.get("recall") or {}).get("self_entity")
    n = {s: len(eps.get(s, ())) for s in ents}
    by_name: dict[str, list[str]] = defaultdict(list)  # "word word" -> slugs with that name/alias
    for s, e in ents.items():
        for name in {e["name"], *(e["aliases"] or [])}:
            t = " ".join(_tokens(name))
            if len(t) > 2:
                by_name[t].append(s)
    ep_ents: dict[int, list[str]] = defaultdict(list)
    for s, ids in eps.items():
        for i in ids:
            ep_ents[i].append(s)
    out = []
    for s, e in ents.items():
        if n[s] > 1:
            continue
        toks = _tokens(e["name"])
        cands = []  # (words matched, events, slug): the longest contained name wins
        for size in range(min(len(toks), 6), 0, -1):
            for i in range(len(toks) - size + 1):
                for p in by_name.get(" ".join(toks[i:i + size]), ()):
                    if p != s and p != me and n[p] > n[s]:
                        cands.append((size, n[p], p))
        parent = max(cands)[2] if cands else None
        if parent is None:
            co = [p for i in eps.get(s, ()) for p in ep_ents[i] if p not in (s, me) and n[p] > 1]
            parent = max(co, key=lambda p: (n[p], p)) if co else None
        out.append({"slug": s, "name": e["name"], "kind": e["kind"], "n": n[s],
                    "into": parent, "into_name": ents[parent]["name"] if parent else None})
    out.sort(key=lambda o: (o["into"] is None, (o["into_name"] or "").lower(), o["name"].lower()))
    return out


async def self_suggestion(session: AsyncSession) -> dict | None:
    if (store.get("recall") or {}).get("self_entity"):
        return None
    row = (await session.execute(sql("""
        select e.slug, e.name, count(x.id) n from entities e left join episodes x on x.entities ? e.slug
        where e.kind = 'person' group by e.id order by n desc, e.slug limit 1"""))).mappings().first()
    return dict(row) if row and row["n"] else None


_cache: dict[str, tuple[float, int]] = {}


async def badge(session: AsyncSession) -> int:
    """Open suggestion groups (merges + duplicate events), cached for a minute."""
    hit = _cache.get("badge")
    if hit and time.monotonic() - hit[0] < 60:
        return hit[1]
    n = len(await merge_groups(session)) + len(await dupe_groups(session))
    _cache["badge"] = (time.monotonic(), n)
    return n


def invalidate() -> None:
    _cache.clear()
    from src.ui import templating
    templating.invalidate()


# ---------------------------------------------------------------- fixing

def _swap(slugs: list[str], mapping: dict[str, str]) -> list[str]:
    return _uniq(mapping.get(s, s) for s in slugs)


async def _repoint(session: AsyncSession, mapping: dict[str, str]) -> dict:
    """Rewrite entity references (events, lessons, fact metadata) from old slugs to new ones."""
    olds = list(mapping)
    out = {}
    for table in ("episodes", "reflections"):
        rows = (await session.execute(sql(f"select id, entities from {table} where entities ?| :o"), {"o": olds})).all()
        if rows:
            await session.execute(sql(f"update {table} set entities = cast(:e as jsonb) where id = :id"),
                                  [{"id": i, "e": json.dumps(_swap(ents, mapping))} for i, ents in rows])
        out[table] = len(rows)
    # Facts live in mem0's pgvector table (same database). AsyncMemory.update merges metadata into the
    # payload but re-embeds the text and can't join our transaction, so the entities key is set in SQL.
    t = facts.table()
    out["facts"] = 0
    if await session.scalar(sql("select to_regclass(:t) is not null"), {"t": t}):
        rows = (await session.execute(sql(
            f"select id::text, payload->'entities' from {t} where payload->'entities' ?| :o"), {"o": olds})).all()
        if rows:
            await session.execute(sql(
                f"update {t} set payload = jsonb_set(payload, '{{entities}}', cast(:e as jsonb)) where id = cast(:id as uuid)"),
                [{"id": i, "e": json.dumps(_swap(ents, mapping))} for i, ents in rows])
        out["facts"] = len(rows)
    return out


async def _entities(session: AsyncSession, slugs: list[str]) -> dict[str, Entity]:
    got = {e.slug: e for e in (await session.execute(select(Entity).where(Entity.slug.in_(slugs)))).scalars()}
    missing = [s for s in slugs if s not in got]
    if missing:
        raise HTTPException(404, f"No topic {', '.join(missing)}. Reload the page; it may already be merged.")
    return got


async def _merge(session: AsyncSession, keep: str, merge: list[str], recall: dict) -> dict:
    merge = _uniq(s for s in merge if s != keep)
    if not merge:
        raise HTTPException(422, "Pick at least one other topic to merge into the one you keep.")
    ents = await _entities(session, [keep, *merge])
    counts = await _repoint(session, {m: keep for m in merge})
    k = ents[keep]
    aliases = list(k.aliases or [])
    seen = {k.name.lower(), *(a.lower() for a in aliases)}
    for m in merge:
        for name in [ents[m].name, *(ents[m].aliases or [])]:
            if name.lower() not in seen:
                seen.add(name.lower())
                aliases.append(name)
        await session.delete(ents[m])
    k.aliases = aliases
    k.dirty = True
    if recall.get("self_entity") in merge:
        recall["self_entity"] = keep
    await session.flush()
    return {**counts, "merged": len(merge)}


async def _commit(session: AsyncSession, recall: dict) -> None:
    if recall != store.get("recall"):
        await store.put(session, "recall", recall)  # commits the whole transaction
    else:
        await session.commit()
    invalidate()


async def merge(session: AsyncSession, keep: str, slugs: list[str]) -> dict:
    recall = store.get("recall") or {}
    out = await _merge(session, keep, slugs, recall)
    await _commit(session, recall)
    return out


async def fold(session: AsyncSession, items: list[dict]) -> dict:
    if not items:
        raise HTTPException(422, "Select at least one topic to fold.")
    recall = store.get("recall") or {}
    folded: dict[str, str] = {}
    total = {"episodes": 0, "reflections": 0, "facts": 0, "merged": 0}
    for it in items:
        slug, into = it.get("slug"), it.get("into")
        if not slug or not into:
            raise HTTPException(422, "Each item needs a topic and a parent.")
        while into in folded:  # its parent was folded earlier in this batch
            into = folded[into]
        if slug == into:
            continue
        for k, v in (await _merge(session, into, [slug], recall)).items():
            total[k] += v
        folded[slug] = into
    await _commit(session, recall)
    return total


async def apply_kinds(session: AsyncSession, mapping: dict[str, str]) -> dict:
    bad = {k: v for k, v in mapping.items() if v not in CANON_KINDS}
    if bad:
        raise HTTPException(422, f"Kinds can only map to {', '.join(CANON_KINDS)}.")
    mapping = {k: v for k, v in mapping.items() if k != v}
    recall = store.get("recall") or {}
    moved = merged = 0
    rows = (await session.execute(select(Entity).where(Entity.kind.in_(list(mapping))).order_by(Entity.id))).scalars().all()
    for e in rows:
        new = slugify(mapping[e.kind], e.name)
        if await session.scalar(select(Entity.id).where(Entity.slug == new)):
            await _merge(session, new, [e.slug], recall)
            merged += 1
        else:
            old = e.slug
            e.slug, e.kind, e.dirty = new, mapping[e.kind], True
            await session.flush()
            await _repoint(session, {old: new})
            if recall.get("self_entity") == old:
                recall["self_entity"] = new
            moved += 1
    kind_map = {**(store.get("kind_map") or {}), **mapping}
    if recall != store.get("recall"):
        await store.put(session, "recall", recall)
    await store.put(session, "kind_map", kind_map)
    invalidate()
    return {"moved": moved, "merged": merged, "kinds": len(mapping)}


async def dedupe(session: AsyncSession, keep: int, remove: list[int]) -> dict:
    remove = _uniq(i for i in remove if i != keep)
    if not remove:
        raise HTTPException(422, "Nothing to remove: pick the event to keep and at least one other.")
    rows = {r["id"]: dict(r) for r in (await session.execute(sql(
        "select id, outcome, entities from episodes where id = any(:ids)"), {"ids": [keep, *remove]})).mappings()}
    missing = [i for i in [keep, *remove] if i not in rows]
    if missing:
        raise HTTPException(404, f"No event {', '.join(f'e:{i}' for i in missing)}. Reload the page.")
    k = rows[keep]
    outcome = k["outcome"] or next((rows[i]["outcome"] for i in remove if rows[i]["outcome"]), None)
    ents = _uniq([*k["entities"], *(s for i in remove for s in rows[i]["entities"])])
    await session.execute(sql("update episodes set outcome = :o, entities = cast(:e as jsonb) where id = :id"),
                          {"o": outcome, "e": json.dumps(ents), "id": keep})
    # lessons cite events as evidence: point them at the survivor
    gone = set(remove)
    # ponytail: scans every lesson's evidence; a few hundred rows, use a jsonb index if it grows
    refs = [(rid, ev) for rid, ev in (await session.execute(sql("select id, evidence from reflections"))).all()
            if gone & {int(i) for i in ev or []}]
    for rid, ev in refs:
        new = _uniq(keep if int(i) in gone else i for i in ev)
        await session.execute(sql("update reflections set evidence = cast(:e as jsonb) where id = :id"),
                              {"e": json.dumps(new), "id": rid})
    await session.execute(sql("delete from episodes where id = any(:ids)"), {"ids": remove})
    await session.execute(sql("update entities set dirty = true where slug = any(:s)"), {"s": ents})
    await session.commit()
    invalidate()
    return {"kept": keep, "removed": len(remove), "outcome": outcome}


async def dismiss(session: AsyncSession, key: str) -> dict:
    if not re.fullmatch(r"(merge|dupe):\S+", key or ""):
        raise HTTPException(422, "Unknown suggestion.")
    keys = store.get("cleanup_dismissed") or []
    if key not in keys:
        await store.put(session, "cleanup_dismissed", [*keys, key])
    invalidate()
    return {"dismissed": key}
