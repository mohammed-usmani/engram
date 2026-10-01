from __future__ import annotations

import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import ContextDocument
from src.services.ollama_client import embed, embed_batch

log = logging.getLogger(__name__)

# nomic-embed-text context is 8192 tokens (~6000 chars safe limit)
_MAX_EMBED_CHARS = 6000


def _truncate(text: str) -> str:
    return text[:_MAX_EMBED_CHARS] if len(text) > _MAX_EMBED_CHARS else text


async def embed_all_documents(session: AsyncSession, force: bool = True) -> dict[str, int]:
    stmt = select(ContextDocument)
    if not force:
        stmt = stmt.where(ContextDocument.embedding.is_(None))
    rows = (await session.execute(stmt.order_by(ContextDocument.id))).scalars().all()

    if not rows:
        return {"embedded": 0, "errors": 0}

    summary = {"embedded": 0, "errors": 0}

    # Embed one at a time to handle per-doc errors gracefully
    for row in rows:
        text = _truncate(f"{row.title}\n{row.content}")
        try:
            row.embedding = await embed(text)
            summary["embedded"] += 1
        except Exception as e:
            log.warning("embed failed for %s: %s", row.slug, e)
            summary["errors"] += 1

    return summary
