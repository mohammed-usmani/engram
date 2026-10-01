import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import async_sessionmaker

from main import app
from src.database import get_session
from src.models import ContextDocument, DocType, Source


@pytest_asyncio.fixture
async def seeded(engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with Session() as s:
        s.add_all([
            ContextDocument(type=DocType.SKILL, slug="skill/python", title="Python",
                            tags=["skill", "languages"], content="python backend apis",
                            sections={"Summary": "hi"}, source=Source.FILE, doc_metadata={}),
            ContextDocument(type=DocType.PROJECT, slug="project/cityfix", title="CityFix",
                            tags=["project"], content="civic mobile app with geospatial queries",
                            sections={}, source=Source.FILE, doc_metadata={}),
        ])
        await s.commit()

    async def override():
        async with Session() as s:
            yield s

    app.dependency_overrides[get_session] = override
    yield
    app.dependency_overrides.clear()


async def test_list(seeded):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/documents")
    assert r.status_code == 200
    slugs = [d["slug"] for d in r.json()]
    assert "skill/python" in slugs and "project/cityfix" in slugs


async def test_filter_by_type(seeded):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/documents?type=skill")
    assert r.status_code == 200
    assert all(d["type"] == "skill" for d in r.json())


async def test_get_by_slug(seeded):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/documents/project/cityfix")
    assert r.status_code == 200
    assert r.json()["title"] == "CityFix"


async def test_missing_slug_404(seeded):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/documents/project/nope")
    assert r.status_code == 404


async def test_search(seeded):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/documents?q=geospatial")
    assert r.status_code == 200
    slugs = [d["slug"] for d in r.json()]
    assert "project/cityfix" in slugs
