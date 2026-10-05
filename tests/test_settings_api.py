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


LOCAL = {"host": "127.0.0.1:8001"}


async def test_update_provider(settings_app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.put("/api/settings/providers/groq", headers=LOCAL, json={
            "api_key": "gsk_test123", "model": "mixtral-8x7b-32768", "is_active": True
        })
        assert r.status_code == 200
        data = r.json()
        assert data["api_key_set"] is True and data["key_source"] == "saved"
        assert data["model"] == "mixtral-8x7b-32768"
        assert data["is_active"] is True
        assert "api_key" not in data
        # saved key is live for the memory chain right away (no restart)
        from src.memory import llm
        assert llm._key("groq") == "gsk_test123"
        await c.put("/api/settings/providers/groq", headers=LOCAL, json={"api_key": ""})
    assert llm.key_source("groq") in (None, "env")


async def test_keys_cannot_be_changed_remotely(settings_app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.put("/api/settings/providers/groq", headers={"host": "x.ts.net", "x-forwarded-for": "1.2.3.4"},
                        json={"api_key": "stolen"})
    assert r.status_code in (401, 403)


async def test_missing_providers_get_rows_and_together_lists_deepseek(settings_app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        names = [p["provider"] for p in (await c.get("/api/settings/providers", headers=LOCAL)).json()]
        models = (await c.get("/api/settings/providers/together/models", headers=LOCAL)).json()["models"]
    assert {"together", "mistral", "claude"} <= set(names)
    assert "deepseek-ai/DeepSeek-V4-Flash-0731" in models


async def test_model_assignments_drive_the_llm_chain(settings_app, monkeypatch):
    from src.memory import llm
    monkeypatch.setenv("TOGETHER_API_KEY", "t")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.put("/api/settings/models", headers=LOCAL, json={
            "extract": [{"provider": "together", "model": "openai/gpt-oss-20b"}, {"provider": "ollama", "model": None}],
            "chat": [{"provider": "ollama", "model": "qwen3:8b"}], "plan": []})
        assert r.status_code == 200, r.text
        assert llm.steps("extract") == [("together", "openai/gpt-oss-20b"), ("ollama", None)]
        assert llm.steps("reconcile") == llm.steps("extract")  # inherits
        assert llm.steps("chat") == [("ollama", "qwen3:8b")]
        bad = await c.put("/api/settings/models", headers=LOCAL, json={"extract": [{"provider": "groq"}]})
        assert bad.status_code == 422 and "no API key" in bad.json()["detail"]
        await c.put("/api/settings/models", headers=LOCAL, json={"extract": [], "chat": []})
    from src import settings_store
    settings_store._cache.clear()
    llm.configure({}, {})


async def test_get_models_cloud(settings_app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/settings/providers/groq/models")
    assert r.status_code == 200
    data = r.json()
    assert "llama-3.3-70b-versatile" in data["models"]
