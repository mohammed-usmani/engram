from __future__ import annotations

import json
import logging

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models import Memory, MemoryCategory
from src.services.ollama_client import generate, embed

log = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """You are analyzing a conversation to extract factual information about the user ({name}).
Extract only concrete, reusable facts — not questions, not opinions, not conversation filler.

Categories:
- fact: biographical or career facts (e.g., "Has 2 years of freelance experience")
- preference: preferences or opinions (e.g., "Prefers NestJS over Express for production")
- experience: specific experiences or challenges (e.g., "Faced CORS issues when deploying CityFix")
- skill: skill-related observations (e.g., "Strong in PostgreSQL query optimization")

Return a JSON array: [{{"content": "...", "category": "..."}}]
Return an empty array [] if nothing worth extracting.

USER: {user_message}
ASSISTANT: {assistant_response}"""

_DEDUP_THRESHOLD = 0.92


async def extract_memories(
    user_message: str,
    assistant_response: str,
    session_id: int,
    db_session: AsyncSession,
) -> list[Memory]:
    prompt = _EXTRACTION_PROMPT.format(
        name=settings.user_name, user_message=user_message, assistant_response=assistant_response,
    )

    try:
        raw = await generate(prompt, system="You extract structured facts from conversations. Respond only with valid JSON.")
    except Exception as e:
        log.warning("memory extraction LLM call failed: %s", e)
        return []

    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        items = json.loads(text)
        if not isinstance(items, list):
            return []
    except (json.JSONDecodeError, ValueError):
        log.warning("memory extraction returned non-JSON: %s", raw[:200])
        return []

    created: list[Memory] = []
    for item in items:
        content = item.get("content", "").strip()
        cat_str = item.get("category", "fact").strip().lower()
        if not content:
            continue
        try:
            category = MemoryCategory(cat_str)
        except ValueError:
            category = MemoryCategory.FACT

        try:
            emb = await embed(content)
        except Exception as e:
            log.warning("memory embed failed: %s", e)
            continue

        existing = await _find_similar(emb, db_session, threshold=_DEDUP_THRESHOLD)
        if existing:
            existing.content = content
            existing.category = category
            existing.embedding = emb
            created.append(existing)
        else:
            mem = Memory(
                content=content,
                category=category,
                embedding=emb,
                source_session_id=session_id,
            )
            db_session.add(mem)
            created.append(mem)

    if created:
        await db_session.commit()
    return created


async def _find_similar(
    embedding: list[float], session: AsyncSession, threshold: float = 0.92
) -> Memory | None:
    stmt = (
        select(
            Memory,
            (1 - Memory.embedding.cosine_distance(embedding)).label("similarity"),
        )
        .where(Memory.embedding.isnot(None))
        .order_by(Memory.embedding.cosine_distance(embedding))
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    if row and float(row[1]) >= threshold:
        return row[0]
    return None


async def search_memories(
    query_embedding: list[float], session: AsyncSession, limit: int = 5
) -> list[Memory]:
    stmt = (
        select(Memory)
        .where(Memory.embedding.isnot(None))
        .order_by(Memory.embedding.l2_distance(query_embedding))
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def list_memories(
    session: AsyncSession, category: str | None = None
) -> list[Memory]:
    stmt = select(Memory).order_by(Memory.created_at.desc())
    if category:
        stmt = stmt.where(Memory.category == MemoryCategory(category))
    return list((await session.execute(stmt)).scalars().all())


async def delete_memory(memory_id: int, session: AsyncSession) -> None:
    await session.execute(delete(Memory).where(Memory.id == memory_id))
