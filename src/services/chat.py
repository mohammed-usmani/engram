from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings as app_settings
from src.models import ChatSession, ChatMessage, MessageRole, ProviderSetting
from src.services.ollama_client import embed
from src.services.providers import get_provider
from src.services.retrieval import retrieve, RetrievedDoc
from src.services.memory import extract_memories, search_memories
from src.services.compaction import maybe_compact

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are {name}'s personal career assistant. You have access to their resume, projects, skills, work experience, and education through the context documents provided below.

CRITICAL RULES:
- ONLY state technologies and facts that are EXPLICITLY mentioned in the provided context documents
- NEVER guess, assume, or infer tech stacks — if a document doesn't list a technology, don't attribute it
- If a project uses SolidJS, do NOT call it React. If it uses MariaDB, do NOT call it MongoDB. Be precise.
- If asked about a specific tech stack (e.g., "MERN projects"), only include projects whose context documents explicitly list ALL components of that stack
- When unsure, say "Based on the available context, I don't have enough information" rather than guessing

When answering questions:
- Be specific and reference actual projects, technologies, and achievements FROM the provided documents
- Use STAR format (Situation, Task, Action, Result) when describing experiences
- Quote or closely paraphrase from the context — do not fabricate details

When optimizing for a job description:
- Identify which of {name}'s existing skills and projects match the JD
- Suggest how to reword existing experience to better align
- Flag gaps where {name} lacks direct experience
- Prioritize recent and impactful work"""

_MAX_HISTORY_MESSAGES = 20


@dataclass
class ChatResult:
    response: str
    session_id: int
    sources: list[dict[str, Any]] = field(default_factory=list)
    memories_extracted: int = 0


@dataclass
class ChatContext:
    """Pre-computed context for chat generation (retrieval done, prompt built)."""
    session_id: int
    chat_session: Any  # ChatSession ORM object
    system: str
    history: list[dict[str, str]]
    provider: Any  # LLMProvider instance
    retrieved: list[Any]
    message: str
    db_session: Any


async def prepare_chat(
    message: str,
    session_id: int | None,
    db_session: AsyncSession,
    provider_name: str | None = None,
    model_name: str | None = None,
) -> ChatContext:
    """Do everything up to generation: create session, save user msg, retrieve, build prompt."""
    from src.models import ProviderSetting
    from src.services.providers import get_provider

    # 1. Create or load session
    if session_id is None:
        chat_session = ChatSession(
            title=message[:60].strip() + ("..." if len(message) > 60 else ""),
            message_count=0,
        )
        db_session.add(chat_session)
        await db_session.commit()
        await db_session.refresh(chat_session)
        session_id = chat_session.id
    else:
        chat_session = (await db_session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )).scalar_one_or_none()
        if chat_session is None:
            raise ValueError(f"session not found: {session_id}")

    # Resolve provider
    p_name = provider_name or chat_session.provider or "ollama"
    p_model = model_name or chat_session.model

    ps = (await db_session.execute(
        select(ProviderSetting).where(ProviderSetting.provider == p_name)
    )).scalar_one_or_none()

    if ps and not p_model:
        p_model = ps.model

    provider = get_provider(
        p_name,
        api_key=ps.api_key if ps else None,
        model=p_model,
        base_url=ps.base_url if ps else None,
    )

    chat_session.provider = p_name
    chat_session.model = p_model or (provider.model if hasattr(provider, 'model') else None)

    # 2. Save user message
    db_session.add(ChatMessage(session_id=session_id, role=MessageRole.USER, content=message))
    await db_session.commit()

    # 3. Embed query
    try:
        query_embedding = await embed(message)
    except Exception as e:
        log.warning("query embed failed: %s", e)
        query_embedding = None

    # 4. Retrieve context docs
    try:
        retrieved = await retrieve(message, db_session, limit=12)
    except Exception as e:
        log.warning("retrieval failed: %s", e)
        retrieved = []

    # 5. Search memories
    memories = []
    if query_embedding:
        try:
            memories = await search_memories(query_embedding, db_session, limit=5)
        except Exception as e:
            log.warning("memory search failed: %s", e)

    # 6. Load recent history
    history_msgs = (await db_session.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(_MAX_HISTORY_MESSAGES + 1)
    )).scalars().all()
    history_msgs = list(reversed(history_msgs))
    if history_msgs and history_msgs[-1].role == MessageRole.USER:
        history_msgs = history_msgs[:-1]

    # 7. Build prompt
    name = app_settings.user_name
    system_parts = [_SYSTEM_PROMPT.format(name=name)]
    if memories:
        mem_text = "\n".join(f"- {m.content}" for m in memories)
        system_parts.append(f"\nContext about {name} from memory:\n{mem_text}")
    if retrieved:
        doc_text = "\n\n".join(
            f"[{r.type}: {r.title}]\n{r.content[:1500]}" for r in retrieved
        )
        system_parts.append(f"\nRelevant documents from {name}'s profile:\n{doc_text}")
    if chat_session.summary:
        system_parts.append(f"\nPrevious conversation context:\n{chat_session.summary}")

    system = "\n".join(system_parts)
    history = [{"role": m.role.value, "content": m.content} for m in history_msgs]

    return ChatContext(
        session_id=session_id,
        chat_session=chat_session,
        system=system,
        history=history,
        provider=provider,
        retrieved=retrieved,
        message=message,
        db_session=db_session,
    )


async def finalize_chat(ctx: ChatContext, response: str) -> ChatResult:
    """Save assistant message, extract memories, compact. Called after generation completes."""
    db_session = ctx.db_session

    db_session.add(ChatMessage(session_id=ctx.session_id, role=MessageRole.ASSISTANT, content=response))
    ctx.chat_session.message_count = (ctx.chat_session.message_count or 0) + 2
    await db_session.commit()

    sources = [{"slug": r.slug, "title": r.title, "type": r.type, "score": r.score} for r in ctx.retrieved]

    memories_extracted = 0
    try:
        new_mems = await extract_memories(ctx.message, response, ctx.session_id, db_session)
        memories_extracted = len(new_mems)
    except Exception as e:
        log.warning("memory extraction failed: %s", e)

    try:
        await maybe_compact(ctx.session_id, db_session)
        await db_session.commit()
    except Exception as e:
        log.warning("compaction failed: %s", e)

    return ChatResult(
        response=response,
        session_id=ctx.session_id,
        sources=sources,
        memories_extracted=memories_extracted,
    )


async def handle_chat(
    message: str,
    session_id: int | None,
    db_session: AsyncSession,
    provider_name: str | None = None,
    model_name: str | None = None,
) -> ChatResult:
    """Non-streaming chat: prepare → generate → finalize."""
    ctx = await prepare_chat(message, session_id, db_session, provider_name, model_name)

    try:
        response = await ctx.provider.generate(ctx.message, system=ctx.system, history=ctx.history)
    except Exception as e:
        log.exception("generation failed")
        response = f"Sorry, I couldn't generate a response: {e}"

    return await finalize_chat(ctx, response)
