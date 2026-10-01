from sqlalchemy import select
from src.models import ChatSession, ChatMessage, MessageRole, Memory, MemoryCategory


async def test_chat_session_and_messages(session):
    s = ChatSession(title="Test session")
    session.add(s)
    await session.commit()
    await session.refresh(s)

    session.add_all([
        ChatMessage(session_id=s.id, role=MessageRole.USER, content="Hello"),
        ChatMessage(session_id=s.id, role=MessageRole.ASSISTANT, content="Hi there"),
    ])
    await session.commit()

    msgs = (await session.execute(
        select(ChatMessage).where(ChatMessage.session_id == s.id).order_by(ChatMessage.id)
    )).scalars().all()
    assert len(msgs) == 2
    assert msgs[0].role == MessageRole.USER
    assert msgs[1].content == "Hi there"


async def test_memory_insert(session):
    m = Memory(
        content="Prefers NestJS over Express",
        category=MemoryCategory.PREFERENCE,
        embedding=[0.1] * 768,
    )
    session.add(m)
    await session.commit()
    await session.refresh(m)
    assert m.id is not None
    assert len(m.embedding) == 768


async def test_cascade_delete_session(session):
    s = ChatSession(title="To delete")
    session.add(s)
    await session.commit()
    await session.refresh(s)

    session.add(ChatMessage(session_id=s.id, role=MessageRole.USER, content="bye"))
    await session.commit()

    await session.delete(s)
    await session.commit()

    msgs = (await session.execute(
        select(ChatMessage).where(ChatMessage.session_id == s.id)
    )).scalars().all()
    assert len(msgs) == 0
