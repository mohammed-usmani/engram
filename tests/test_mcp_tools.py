import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.models import ContextDocument, DocType, Source


@pytest_asyncio.fixture
async def mcp_db(engine, monkeypatch):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    import src.mcp_server as srv
    monkeypatch.setattr(srv, "AsyncSessionLocal", Session)
    async with Session() as s:
        s.add_all([
            ContextDocument(type=DocType.RESUME, slug="resume/master", title="Master Resume",
                            tags=["resume"], content="Alex Rivera backend engineer nestjs postgres redis",
                            sections={}, source=Source.FILE, doc_metadata={}),
            ContextDocument(type=DocType.PROJECT, slug="project/cityfix", title="CityFix",
                            tags=["project"], content="civic mobile app geospatial mongodb express",
                            sections={}, source=Source.FILE, doc_metadata={}),
            ContextDocument(type=DocType.SKILL, slug="skill/python", title="Python",
                            tags=["skill"], content="python fastapi asyncio data science",
                            sections={}, source=Source.FILE, doc_metadata={}),
            ContextDocument(type=DocType.EXPERIENCE, slug="experience/freelance",
                            title="Freelance Full-Stack Developer", tags=["experience"],
                            content="nestjs postgres redis websockets rate limiting freelance",
                            sections={}, source=Source.FILE, doc_metadata={}),
        ])
        await s.commit()
    yield


def _call(tool_obj):
    """Return the underlying async callable whether FastMCP exposes .fn or wraps it."""
    return getattr(tool_obj, "fn", tool_obj)


async def test_list_types(mcp_db):
    from src.mcp_server import list_document_types
    out = await _call(list_document_types)()
    assert "skill" in out and "resume" in out


async def test_list_documents(mcp_db):
    from src.mcp_server import list_documents
    out = await _call(list_documents)()
    slugs = {d["slug"] for d in out}
    assert {"resume/master", "project/cityfix", "skill/python"} <= slugs


async def test_get_document(mcp_db):
    from src.mcp_server import get_document
    out = await _call(get_document)("skill/python")
    assert out["title"] == "Python"
    assert "fastapi" in out["content"]


async def test_get_document_missing(mcp_db):
    from src.mcp_server import get_document
    out = await _call(get_document)("skill/nope")
    assert "error" in out


async def test_search_context(mcp_db):
    from src.mcp_server import search_context
    out = await _call(search_context)("geospatial mongodb", limit=5)
    assert any(d["slug"] == "project/cityfix" for d in out)


async def test_get_resume(mcp_db):
    from src.mcp_server import get_resume
    out = await _call(get_resume)()
    assert out["slug"] == "resume/master"


async def test_find_relevant_context(mcp_db):
    from src.mcp_server import find_relevant_context
    out = await _call(find_relevant_context)(
        "We need a backend engineer with NestJS, PostgreSQL, Redis, and WebSockets.", limit=9
    )
    assert out["resume"] is not None
    slugs = {e["slug"] for e in out["experiences"]}
    assert "experience/freelance" in slugs


async def test_save_edit_delete_document(mcp_db, monkeypatch):
    import src.routers.admin as admin
    from src.mcp_server import save_document, edit_document, delete_document, get_document

    async def fake_embed(text):
        return [0.25] * 768
    monkeypatch.setattr(admin, "embed", fake_embed)

    out = await _call(save_document)("project/voiceagent", "Claude-Code", "Stack: Python\n\nSummary:\nPhone agent.",
                                     title="voiceAgent", tags=["Voice"])
    assert out == {"slug": "project/voiceagent", "title": "voiceAgent", "type": "project", "tags": ["voice"]}
    doc = await _call(get_document)("project/voiceagent")
    assert doc["metadata"]["Stack"] == "Python" and doc["sections"]["Summary"] == "Phone agent."
    assert doc["updated_by"] == "claude-code"
    assert "error" in await _call(save_document)("project/voiceagent", "  ", "x")  # who is writing is required

    assert "error" in await _call(save_document)("project/new", "chatgpt", "x")  # new doc needs a title
    assert "error" in await _call(edit_document)("project/voiceagent", "nowhere", "y", "chatgpt")

    await _call(edit_document)("project/voiceagent", "Phone agent.", "Outbound phone agent.", "chatgpt")
    doc = await _call(get_document)("project/voiceagent")
    assert doc["updated_by"] == "chatgpt"
    assert doc["sections"]["Summary"] == "Outbound phone agent." and doc["title"] == "voiceAgent"

    await _call(save_document)("project/voiceagent", "gemini", title="voiceAgent v2")  # rename only
    doc = await _call(get_document)("project/voiceagent")
    assert doc["title"] == "voiceAgent v2" and doc["sections"]["Summary"] == "Outbound phone agent."

    assert await _call(delete_document)("project/voiceagent", "claude-code") == {"deleted": "project/voiceagent"}
    assert "error" in await _call(get_document)("project/voiceagent")


async def test_mcp_remember_requires_who_and_when(mcp_db):
    from src.mcp_server import remember
    assert "error" in await _call(remember)("Applied to Acme", "", "2026-10-03T10:00:00+05:30")
    assert "error" in await _call(remember)("Applied to Acme", "chatgpt", "yesterday")
