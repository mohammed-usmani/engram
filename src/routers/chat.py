import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.models import ChatSession, ChatMessage
from src.schemas import ChatRequest, ChatResponse, SessionSummary, SessionDetail, MessageRead, SourceInfo
from src.services.chat import handle_chat, prepare_chat, finalize_chat

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, session: AsyncSession = Depends(get_session)):
    try:
        result = await handle_chat(
            body.message, body.session_id, session,
            provider_name=body.provider, model_name=body.model,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    return ChatResponse(
        response=result.response,
        session_id=result.session_id,
        sources=[SourceInfo(**s) for s in result.sources],
        memories_extracted=result.memories_extracted,
    )


@router.post("/stream")
async def chat_stream(body: ChatRequest, session: AsyncSession = Depends(get_session)):
    """SSE streaming chat endpoint. Sends token chunks as they arrive."""
    try:
        ctx = await prepare_chat(
            body.message, body.session_id, session,
            provider_name=body.provider, model_name=body.model,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))

    # Send session_id + sources first, then stream tokens, then finalize
    async def event_stream():
        # Initial metadata event
        sources = [{"slug": r.slug, "title": r.title, "type": r.type, "score": r.score} for r in ctx.retrieved]
        yield f"data: {json.dumps({'type': 'meta', 'session_id': ctx.session_id, 'sources': sources})}\n\n"

        # Stream tokens
        full_response = []
        try:
            async for token in ctx.provider.generate_stream(ctx.message, system=ctx.system, history=ctx.history):
                full_response.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            log.exception("streaming generation failed")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        # Finalize (save message, extract memories, compact)
        response_text = "".join(full_response)
        try:
            result = await finalize_chat(ctx, response_text)
            yield f"data: {json.dumps({'type': 'done', 'memories_extracted': result.memories_extracted})}\n\n"
        except Exception as e:
            log.warning("finalize failed: %s", e)
            yield f"data: {json.dumps({'type': 'done', 'memories_extracted': 0})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(
        select(ChatSession).order_by(ChatSession.updated_at.desc())
    )).scalars().all()
    return rows


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session_detail(session_id: int, session: AsyncSession = Depends(get_session)):
    chat_session = (await session.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )).scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(404, "session not found")

    msgs = (await session.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
    )).scalars().all()

    return SessionDetail(
        id=chat_session.id,
        title=chat_session.title,
        summary=chat_session.summary,
        messages=[MessageRead(role=m.role.value, content=m.content, created_at=m.created_at) for m in msgs],
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: int, session: AsyncSession = Depends(get_session)):
    chat_session = (await session.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )).scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(404, "session not found")
    await session.delete(chat_session)
    await session.commit()
