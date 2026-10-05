from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy import select
from src.services.chat import handle_chat, ChatResult
from src.models import ChatSession, ChatMessage, ContextDocument, DocType, Source, ProviderSetting


async def test_handle_chat_creates_session(session):
    session.add(ContextDocument(
        type=DocType.SKILL, slug="skill/python", title="Python",
        tags=["skill"], content="python backend fastapi",
        sections={}, source=Source.FILE, doc_metadata={},
        embedding=[0.5] * 768,
    ))
    session.add(ProviderSetting(provider="ollama", is_active=True, model="mistral:latest"))
    await session.commit()

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(return_value="Python is great!")
    mock_provider.model = "mistral:latest"

    with patch("src.services.chat.retrieve", new_callable=AsyncMock) as mock_ret, \
         patch("src.services.chat.search_memories", new_callable=AsyncMock, return_value=[]) as _, \
         patch("src.memory.llm.make", return_value=mock_provider), \
         patch("src.services.chat.extract_memories", new_callable=AsyncMock, return_value=[]) as _, \
         patch("src.services.chat.maybe_compact", new_callable=AsyncMock, return_value=False) as _, \
         patch("src.services.chat.embed", new_callable=AsyncMock, return_value=[0.5] * 768) as _:

        from src.services.retrieval import RetrievedDoc
        mock_ret.return_value = [RetrievedDoc(
            slug="skill/python", title="Python", type="skill",
            content="python backend fastapi", score=0.9, source_method="both",
        )]

        result = await handle_chat("Tell me about Python", None, session)

    assert isinstance(result, ChatResult)
    assert result.response == "Python is great!"
    assert result.session_id is not None
    assert len(result.sources) >= 1

    chat_session = (await session.execute(
        select(ChatSession).where(ChatSession.id == result.session_id)
    )).scalar_one()
    assert chat_session.title is not None
    assert chat_session.message_count == 2


async def test_handle_chat_continues_session(session):
    session.add(ProviderSetting(provider="ollama", is_active=True, model="mistral:latest"))
    s = ChatSession(title="Existing", message_count=2)
    session.add(s)
    await session.commit()
    await session.refresh(s)

    session.add_all([
        ChatMessage(session_id=s.id, role="user", content="Hi"),
        ChatMessage(session_id=s.id, role="assistant", content="Hello!"),
    ])
    await session.commit()

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(return_value="I'm good!")
    mock_provider.model = "mistral:latest"

    with patch("src.services.chat.retrieve", new_callable=AsyncMock, return_value=[]), \
         patch("src.services.chat.search_memories", new_callable=AsyncMock, return_value=[]), \
         patch("src.memory.llm.make", return_value=mock_provider), \
         patch("src.services.chat.extract_memories", new_callable=AsyncMock, return_value=[]), \
         patch("src.services.chat.maybe_compact", new_callable=AsyncMock, return_value=False), \
         patch("src.services.chat.embed", new_callable=AsyncMock, return_value=[0.5] * 768):

        result = await handle_chat("How are you?", s.id, session)

    assert result.session_id == s.id
    await session.refresh(s)
    assert s.message_count == 4
