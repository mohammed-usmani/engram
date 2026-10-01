from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.models import ContextDocument, DocType, Source
from src.routers import admin


async def test_editing_on_site_rederives_sections_and_reembeds(client, engine, monkeypatch):
    async def fake_embed(text):
        return [0.25] * 768
    monkeypatch.setattr(admin, "embed", fake_embed)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        s.add(ContextDocument(type=DocType.SKILL, slug="skill/python", title="Python", tags=["skill"],
                              content="Summary:\nOld.\n", sections={"Summary": "Old."}, source=Source.MANUAL,
                              doc_metadata={}))
        await s.commit()

    r = await client.post("/api/admin/skill/python", data={
        "type": "skill", "title": "Python", "tags": "skill, languages",
        "content": "Category: Languages\n\nLevel: Advanced\n\nSummary:\nNew text.\n\nKey Concepts:\n- asyncio\n"},
        follow_redirects=False)
    assert r.status_code == 303

    async with Session() as s:
        doc = (await s.execute(select(ContextDocument).where(ContextDocument.slug == "skill/python"))).scalar_one()
    assert doc.sections == {"Summary": "New text.", "Key Concepts": "- asyncio"}
    assert doc.doc_metadata == {"Category": "Languages", "Level": "Advanced"}
    assert list(doc.embedding)[:2] == [0.25, 0.25]


async def test_new_document_from_site(client, engine, monkeypatch):
    async def fake_embed(text):
        return [0.5] * 768
    monkeypatch.setattr(admin, "embed", fake_embed)
    r = await client.post("/api/admin/new", data={"type": "project", "title": "TrailMap", "tags": "",
                                                    "content": "Summary:\nOffline hiking planner.\n"},
                          follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/api/admin/project/trailmap"
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        doc = (await s.execute(select(ContextDocument).where(ContextDocument.slug == "project/trailmap"))).scalar_one()
    assert doc.sections == {"Summary": "Offline hiking planner."} and doc.embedding is not None


async def test_import_button_adds_only_new_files(client):
    r = await client.post("/api/admin/import", follow_redirects=False)
    assert r.status_code == 303 and "import=" in r.headers["location"]
