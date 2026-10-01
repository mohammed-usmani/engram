from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import ContextDocument, Source
from src.seed.parser import parse_file

log = logging.getLogger(__name__)


async def run_seed(session: AsyncSession, data_dir: Path) -> dict[str, int]:
    summary = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}
    files = sorted(Path(data_dir).rglob("*.txt"))
    for path in files:
        try:
            payload = parse_file(path)
        except Exception as e:
            log.warning("parse failed for %s: %s", path, e)
            summary["errors"] += 1
            continue

        slug = payload["slug"]
        existing = (
            await session.execute(select(ContextDocument).where(ContextDocument.slug == slug))
        ).scalar_one_or_none()

        if existing is None:
            session.add(ContextDocument(
                type=payload["type"],
                slug=slug,
                title=payload["title"],
                tags=payload["tags"],
                content=payload["content"],
                sections=payload["sections"],
                source=Source.FILE,
                doc_metadata=payload["metadata"],
            ))
            summary["inserted"] += 1
        elif existing.source == Source.FILE:
            existing.title = payload["title"]
            existing.tags = payload["tags"]
            existing.content = payload["content"]
            existing.sections = payload["sections"]
            existing.doc_metadata = payload["metadata"]
            existing.type = payload["type"]
            summary["updated"] += 1
        else:
            summary["skipped"] += 1

    # Embed newly inserted/updated documents
    try:
        from src.services.embeddings import embed_all_documents
        embed_summary = await embed_all_documents(session, force=False)
        log.info("embedding summary: %s", embed_summary)
    except Exception as e:
        log.warning("embedding after seed failed (Ollama running?): %s", e)

    return summary
