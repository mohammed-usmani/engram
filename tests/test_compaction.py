from unittest.mock import AsyncMock, patch
from sqlalchemy import select
from src.services.compaction import maybe_compact
from src.models import ChatSession, ChatMessage, MessageRole


async def test_no_compaction_when_short(session):
    s = ChatSession(title="Short chat")
    session.add(s)
    await session.commit()
    await session.refresh(s)

    session.add_all([
        ChatMessage(session_id=s.id, role=MessageRole.USER, content="Hi"),
        ChatMessage(session_id=s.id, role=MessageRole.ASSISTANT, content="Hello"),
    ])
    await session.commit()

    compacted = await maybe_compact(s.id, session)
    assert compacted is False


async def test_compaction_when_long(session):
    s = ChatSession(title="Long chat")
    session.add(s)
    await session.commit()
    await session.refresh(s)

    for i in range(30):
        session.add(ChatMessage(
            session_id=s.id, role=MessageRole.USER, content=f"Question {i} " + "x" * 400,
        ))
        session.add(ChatMessage(
            session_id=s.id, role=MessageRole.ASSISTANT, content=f"Answer {i} " + "y" * 400,
        ))
    await session.commit()

    with patch("src.services.compaction.generate", new_callable=AsyncMock,
               return_value="Summary of the conversation so far."):
        compacted = await maybe_compact(s.id, session, token_threshold=3000)

    assert compacted is True

    await session.refresh(s)
    assert s.summary is not None
    assert "Summary" in s.summary

    msgs = (await session.execute(
        select(ChatMessage).where(ChatMessage.session_id == s.id).order_by(ChatMessage.id)
    )).scalars().all()
    assert len(msgs) == 8
