import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import async_sessionmaker

from main import app
from src.database import get_session
from src.models import ChatSession, ChatMessage, MessageRole, Memory, MemoryCategory, ProviderSetting


@pytest_asyncio.fixture
async def chat_app(engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with Session() as s:
        s.add(ProviderSetting(provider="ollama", is_active=True, model="mistral:latest"))
        await s.commit()

    async def override():
        async with Session() as s:
            yield s

    app.dependency_overrides[get_session] = override
    yield
    app.dependency_overrides.clear()


async def test_chat_endpoint(chat_app):
    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(return_value="Hi Alex!")
    mock_provider.model = "mistral:latest"

    with patch("src.services.chat.retrieve", new_callable=AsyncMock, return_value=[]), \
         patch("src.services.chat.search_memories", new_callable=AsyncMock, return_value=[]), \
         patch("src.services.chat.get_provider", return_value=mock_provider), \
         patch("src.services.chat.extract_memories", new_callable=AsyncMock, return_value=[]), \
         patch("src.services.chat.maybe_compact", new_callable=AsyncMock, return_value=False), \
         patch("src.services.chat.embed", new_callable=AsyncMock, return_value=[0.5] * 768):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/api/chat", json={"message": "Hello"})

    assert r.status_code == 200
    data = r.json()
    assert data["response"] == "Hi Alex!"
    assert data["session_id"] is not None


async def test_list_sessions(chat_app, engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        s.add(ChatSession(title="Test session", message_count=2))
        await s.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/chat/sessions")

    assert r.status_code == 200
    assert len(r.json()) >= 1


async def test_get_session_detail(chat_app, engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        cs = ChatSession(title="Detail test", message_count=2)
        s.add(cs)
        await s.commit()
        await s.refresh(cs)
        s.add(ChatMessage(session_id=cs.id, role=MessageRole.USER, content="hi"))
        s.add(ChatMessage(session_id=cs.id, role=MessageRole.ASSISTANT, content="hello"))
        await s.commit()
        sid = cs.id

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(f"/api/chat/sessions/{sid}")

    assert r.status_code == 200
    assert len(r.json()["messages"]) == 2


async def test_delete_session(chat_app, engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        cs = ChatSession(title="To delete")
        s.add(cs)
        await s.commit()
        await s.refresh(cs)
        sid = cs.id

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.delete(f"/api/chat/sessions/{sid}")

    assert r.status_code == 204


async def test_memories_endpoint(chat_app, engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        s.add(Memory(content="Likes Python", category=MemoryCategory.PREFERENCE, embedding=[0.1] * 768))
        await s.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/memories")

    assert r.status_code == 200
    assert len(r.json()) >= 1
    assert r.json()[0]["content"] == "Likes Python"
