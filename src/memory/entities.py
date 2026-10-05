"""Entity resolution: the join key that lets one situation pull every memory layer."""
from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.memory.models import Entity
from src.services.ollama_client import embed

_SIM_THRESHOLD = 0.9
CANON_KINDS = ("company", "person", "project", "topic", "place", "tool", "org")


def canon_kind(kind: str | None) -> str:
    """Apply the Cleanup page's kind map; anything still unknown becomes a plain topic."""
    from src import settings_store  # late: settings_store pulls in the LLM layer
    kind = (kind or "topic").strip().lower()
    kind = (settings_store.get("kind_map") or {}).get(kind, kind)
    return kind if kind in CANON_KINDS else "topic"


def slugify(kind: str, name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{(kind or 'topic').lower()}:{s or 'unknown'}"


def _names(e: Entity) -> set[str]:
    return {e.name.lower(), *(a.lower() for a in e.aliases)}


async def resolve(session: AsyncSession, items: list[dict]) -> dict[str, str]:
    """Map extractor entity names to canonical slugs, creating entities as needed."""
    # ponytail: loads every entity; fine for one person's few thousand, index aliases if it grows
    known = list((await session.execute(select(Entity))).scalars().all())
    by_slug = {e.slug: e for e in known}
    out: dict[str, str] = {}
    for item in items:
        name = (item.get("name") or "").strip()
        if not name or name in out:
            continue
        kind = canon_kind(item.get("kind"))
        slug = slugify(kind, name)
        hit = by_slug.get(slug) or next((e for e in known if name.lower() in _names(e)), None)
        vec = None
        if hit is None:
            vec = await embed(f"{kind}: {name}")
            if any(vec):
                row = (await session.execute(
                    select(Entity, 1 - Entity.embedding.cosine_distance(vec))
                    .where(Entity.kind == kind, Entity.embedding.isnot(None))
                    .order_by(Entity.embedding.cosine_distance(vec)).limit(1)
                )).first()
                if row and float(row[1]) >= _SIM_THRESHOLD:
                    hit = row[0]
                    hit.aliases = [*hit.aliases, name]
        if hit is None:
            hit = Entity(slug=slug, kind=kind, name=name, aliases=[], embedding=vec, dirty=True)
            session.add(hit)
            known.append(hit)
            by_slug[slug] = hit
        out[name] = hit.slug
    await session.flush()
    return out


async def match_text(session: AsyncSession, text: str) -> list[str]:
    """Entities whose name or alias appears in the text (whole-word, case-insensitive)."""
    low = text.lower()
    hits = []
    for e in (await session.execute(select(Entity))).scalars():
        if any(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", low) for n in _names(e) if len(n) > 2):
            hits.append(e.slug)
    return sorted(hits)
