import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import async_sessionmaker

from main import app
from src.database import get_session
from src.models import ProviderSetting


@pytest_asyncio.fixture
async def settings_app(engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with Session() as s:
        s.add_all([
            ProviderSetting(provider="ollama", is_active=True),
            ProviderSetting(provider="groq", model="llama-3.3-70b-versatile",
                            base_url="https://api.groq.com/openai/v1", is_active=False),
        ])
        await s.commit()

    async def override():
        async with Session() as s:
            yield s

    app.dependency_overrides[get_session] = override
    yield
    app.dependency_overrides.clear()


async def test_list_providers(settings_app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/settings/providers")
    assert r.status_code == 200
    providers = r.json()
    assert len(providers) >= 2
    names = [p["provider"] for p in providers]
    assert "ollama" in names and "groq" in names
    # api_key should not be exposed
    groq = next(p for p in providers if p["provider"] == "groq")
    assert "api_key" not in groq
    assert groq["api_key_set"] is False


async def test_update_provider(settings_app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.put("/api/settings/providers/groq", json={
            "api_key": "gsk_test123", "model": "mixtral-8x7b-32768", "is_active": True
        })
    assert r.status_code == 200
    data = r.json()
    assert data["api_key_set"] is True
    assert data["model"] == "mixtral-8x7b-32768"
    assert data["is_active"] is True


async def test_get_models_cloud(settings_app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/settings/providers/groq/models")
    assert r.status_code == 200
    data = r.json()
    assert "llama-3.3-70b-versatile" in data["models"]
