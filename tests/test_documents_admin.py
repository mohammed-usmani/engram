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


def test_admin_pages_open_locally_but_not_through_a_tunnel(monkeypatch):
    from fastapi import HTTPException
    from starlette.requests import Request
    import src.routers.admin as admin
    monkeypatch.setattr(admin.settings, "admin_token", "secret")

    def req(headers):
        return Request({"type": "http", "path": "/api/admin", "headers": headers, "client": ("127.0.0.1", 5),
                        "query_string": b""})
    admin._check_token(req([(b"host", b"localhost:8001")]))  # local browser: allowed
    try:
        admin._check_token(req([(b"host", b"x.ts.net"), (b"x-forwarded-for", b"1.2.3.4")]))
        raise AssertionError("tunnelled request without a token was let in")
    except HTTPException as e:
        assert e.status_code == 401


async def test_browser_signs_in_through_a_tunnel(client, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "admin_token", "secret")
    tunnel = {"host": "x.ts.net", "x-forwarded-for": "1.2.3.4", "x-forwarded-proto": "https"}

    r = await client.get("/api/admin", headers={**tunnel, "accept": "text/html"})
    assert r.status_code == 303 and r.headers["location"].startswith("/login?next=")
    assert (await client.post("/login", data={"admin_token": "nope", "next": "/api/admin"}, headers=tunnel)).status_code == 401

    ok = await client.post("/login", data={"admin_token": "secret", "next": "/api/admin"}, headers=tunnel)
    assert ok.status_code == 303 and ok.headers["location"] == "/api/admin"
    assert "admin_token=secret" in ok.headers["set-cookie"] and "Secure" in ok.headers["set-cookie"]
    signed_in = await client.get("/api/admin", headers={**tunnel, "accept": "text/html", "cookie": "admin_token=secret"})
    assert signed_in.status_code == 200

    evil = await client.post("/login", data={"admin_token": "secret", "next": "//evil.example"}, headers=tunnel)
    assert evil.headers["location"] == "/admin"  # no open redirect
