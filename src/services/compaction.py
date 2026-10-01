from __future__ import annotations

import logging
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models import ChatSession, ChatMessage
from src.services.ollama_client import generate

log = logging.getLogger(__name__)

_SUMMARY_PROMPT = """Summarize the following conversation between a user and an AI assistant.
Preserve ALL facts, decisions, preferences, action items, and context discussed.
Be concise but thorough — this summary will replace the full conversation history.

CONVERSATION:
{conversation}"""

_KEEP_EXCHANGES = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


async def maybe_compact(
    session_id: int,
    db_session: AsyncSession,
    token_threshold: int | None = None,
) -> bool:
    threshold = token_threshold or settings.compaction_token_threshold

    msgs = (await db_session.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
    )).scalars().all()

    total_tokens = sum(_estimate_tokens(m.content) for m in msgs)
    if total_tokens < threshold:
        return False

    keep_count = _KEEP_EXCHANGES * 2
    if len(msgs) <= keep_count:
        return False

    to_compact = msgs[:-keep_count]

    conversation = "\n".join(f"{m.role.value}: {m.content}" for m in to_compact)

    prompt = _SUMMARY_PROMPT.format(conversation=conversation)
    try:
        summary = await generate(prompt, system="You summarize conversations accurately and concisely.")
    except Exception as e:
        log.warning("compaction LLM call failed: %s", e)
        return False

    chat_session = (await db_session.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )).scalar_one()
    chat_session.summary = summary

    compact_ids = [m.id for m in to_compact]
    await db_session.execute(
        delete(ChatMessage).where(ChatMessage.id.in_(compact_ids))
    )

    log.info("compacted session %d: removed %d messages, kept %d",
             session_id, len(compact_ids), keep_count)
    return True
