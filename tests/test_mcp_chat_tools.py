import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.models import ChatSession, ChatMessage, MessageRole, Memory, MemoryCategory, ProviderSetting


@pytest_asyncio.fixture
async def mcp_chat_db(engine, monkeypatch):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    import src.mcp_server as srv
    monkeypatch.setattr(srv, "AsyncSessionLocal", Session)

    async with Session() as s:
        cs = ChatSession(title="Test session", message_count=2)
        s.add(cs)
        await s.commit()
        await s.refresh(cs)
        s.add_all([
            ChatMessage(session_id=cs.id, role=MessageRole.USER, content="Hi"),
            ChatMessage(session_id=cs.id, role=MessageRole.ASSISTANT, content="Hello!"),
        ])
        s.add(Memory(content="Knows Python", category=MemoryCategory.SKILL, embedding=[0.1] * 768))
        s.add(ProviderSetting(provider="ollama", is_active=True, model="mistral:latest"))
        await s.commit()
    yield


def _call(tool_obj):
    return getattr(tool_obj, "fn", tool_obj)


async def test_chat_tool(mcp_chat_db):
    from src.mcp_server import chat

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(return_value="Hello!")
    mock_provider.model = "mistral:latest"

    with patch("src.services.chat.retrieve", new_callable=AsyncMock, return_value=[]), \
         patch("src.services.chat.search_memories", new_callable=AsyncMock, return_value=[]), \
         patch("src.memory.llm.make", return_value=mock_provider), \
         patch("src.services.chat.extract_memories", new_callable=AsyncMock, return_value=[]), \
         patch("src.services.chat.maybe_compact", new_callable=AsyncMock, return_value=False), \
         patch("src.services.chat.embed", new_callable=AsyncMock, return_value=[0.5] * 768):
        result = await _call(chat)("Hi there")
    assert result["response"] == "Hello!"
    assert result["session_id"] is not None


async def test_list_chat_sessions_tool(mcp_chat_db):
    from src.mcp_server import list_chat_sessions
    result = await _call(list_chat_sessions)()
    assert len(result) >= 1
    assert result[0]["title"] == "Test session"


async def test_get_chat_session_tool(mcp_chat_db):
    from src.mcp_server import get_chat_session
    result = await _call(get_chat_session)(1)
    assert result["title"] == "Test session"
    assert len(result["messages"]) == 2


async def test_list_memories_tool(mcp_chat_db):
    from src.mcp_server import list_memories
    result = await _call(list_memories)()
    assert len(result) >= 1
    assert result[0]["content"] == "Knows Python"


async def test_delete_memory_tool(mcp_chat_db):
    from src.mcp_server import list_memories, delete_memory
    mems = await _call(list_memories)()
    mid = mems[0]["id"]
    result = await _call(delete_memory)(mid)
    assert result["deleted"] == mid
    mems2 = await _call(list_memories)()
    assert len(mems2) == 0
