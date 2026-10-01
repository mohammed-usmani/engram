from sqlalchemy import select
from src.models import ProviderSetting, ChatSession


async def test_provider_setting_crud(session):
    ps = ProviderSetting(provider="groq", api_key="gsk_test123", model="llama-3.3-70b-versatile", is_active=True)
    session.add(ps)
    await session.commit()
    await session.refresh(ps)

    assert ps.id is not None
    assert ps.provider == "groq"
    assert ps.api_key == "gsk_test123"
    assert ps.is_active is True


async def test_provider_unique(session):
    import pytest
    session.add(ProviderSetting(provider="groq", is_active=False))
    await session.commit()
    session.add(ProviderSetting(provider="groq", is_active=False))
    with pytest.raises(Exception):
        await session.commit()


async def test_chat_session_has_provider(session):
    cs = ChatSession(title="Test", provider="groq", model="llama-3.3-70b")
    session.add(cs)
    await session.commit()
    await session.refresh(cs)
    assert cs.provider == "groq"
    assert cs.model == "llama-3.3-70b"
