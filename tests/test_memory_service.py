from unittest.mock import AsyncMock, patch
from sqlalchemy import select
from src.services.memory import extract_memories, search_memories, list_memories, delete_memory
from src.models import Memory, MemoryCategory, ChatSession


async def test_extract_memories(session):
    s = ChatSession(title="test")
    session.add(s)
    await session.commit()
    await session.refresh(s)

    mock_gen_response = '[{"content": "Has 2 years of freelance experience", "category": "fact"}]'

    with patch("src.services.memory.generate", new_callable=AsyncMock, return_value=mock_gen_response):
        with patch("src.services.memory.embed", new_callable=AsyncMock, return_value=[0.1] * 768):
            memories = await extract_memories(
                "Tell me about your experience",
                "I have 2 years of freelance experience.",
                s.id, session,
            )

    assert len(memories) == 1
    assert "freelance" in memories[0].content
    assert memories[0].category == MemoryCategory.FACT

    db_mems = (await session.execute(select(Memory))).scalars().all()
    assert len(db_mems) == 1


async def test_extract_deduplicates(session):
    s = ChatSession(title="test")
    session.add(s)
    await session.commit()
    await session.refresh(s)

    session.add(Memory(
        content="Has freelance experience",
        category=MemoryCategory.FACT,
        embedding=[0.1] * 768,
        source_session_id=s.id,
    ))
    await session.commit()

    mock_gen = '[{"content": "Has 2 years freelance experience", "category": "fact"}]'
    with patch("src.services.memory.generate", new_callable=AsyncMock, return_value=mock_gen):
        with patch("src.services.memory.embed", new_callable=AsyncMock, return_value=[0.1] * 768):
            memories = await extract_memories("q", "a", s.id, session)

    db_mems = (await session.execute(select(Memory))).scalars().all()
    assert len(db_mems) == 1
    assert "2 years" in db_mems[0].content


async def test_search_memories(session):
    session.add_all([
        Memory(content="Likes NestJS", category=MemoryCategory.PREFERENCE, embedding=[0.9] * 768),
        Memory(content="Has Docker experience", category=MemoryCategory.SKILL, embedding=[0.1] * 768),
    ])
    await session.commit()

    results = await search_memories([0.85] * 768, session, limit=5)
    assert len(results) >= 1
    assert results[0].content == "Likes NestJS"


async def test_list_and_delete_memories(session):
    session.add(Memory(content="X", category=MemoryCategory.FACT, embedding=[0.1] * 768))
    await session.commit()

    mems = await list_memories(session)
    assert len(mems) == 1

    await delete_memory(mems[0].id, session)
    await session.commit()

    mems = await list_memories(session)
    assert len(mems) == 0
