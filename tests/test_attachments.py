"""Files attached to documents: stored in the DB, text extracted (PDF text, OCR), searchable, privacy inherited."""
import base64
import shutil

import pytest

from tests.test_mcp_tools import _call, mcp_db  # noqa: F401  (fixture)

import src.mcp_server as srv
from src import attachments


def _pdf(text: str) -> bytes:
    """A minimal one-page PDF; pdftotext rebuilds the (absent) xref table."""
    stream = f"BT /F1 18 Tf 40 700 Td ({text}) Tj ET".encode()
    return (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
            b"/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
            b"4 0 obj<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream endobj\n"
            b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


async def test_extract_text_plain_and_unknown():
    assert (await attachments.extract_text(b"Policy ABC123", "notes.txt"))[0] == "Policy ABC123"
    text, note = await attachments.extract_text(b"\x00\x01", "blob.bin")
    assert text == "" and "no text" in note


@pytest.mark.skipif(not shutil.which("pdftotext"), reason="poppler not installed")
async def test_extract_pdf_text():
    text, _ = await attachments.extract_text(_pdf("Policy number ZX9981"), "policy.pdf")
    assert "ZX9981" in text


async def test_attach_search_get_and_delete(mcp_db):
    out = await _call(srv.attach_file)(slug="project/cityfix", filename="design.txt",
                                       content_base64=_b64(b"uses the quadtree tiling scheme"), agent="claude-code")
    assert out["filename"] == "design.txt" and out["text_chars"] > 0
    hits = await _call(srv.search_context)(query="quadtree")
    assert [h["slug"] for h in hits] == ["project/cityfix"]
    doc = await _call(srv.get_document)(slug="project/cityfix")
    att = doc["attachments"][0]
    assert att["filename"] == "design.txt" and "quadtree" in att["text"]
    # same bytes again: no duplicate
    again = await _call(srv.attach_file)(slug="project/cityfix", filename="copy.txt",
                                         content_base64=_b64(b"uses the quadtree tiling scheme"), agent="claude-code")
    assert again["id"] == out["id"]
    gone = await _call(srv.delete_attachment)(slug="project/cityfix", attachment_id=out["id"], agent="claude-code")
    assert gone["deleted"] == out["id"]
    assert await _call(srv.search_context)(query="quadtree") == []


async def test_attach_errors_are_actionable(mcp_db):
    out = await _call(srv.attach_file)(slug="project/cityfix", filename="x.txt", content_base64="not base64!!",
                                       agent="claude-code")
    assert "base64" in out["error"] and "fix" in out
    out = await _call(srv.attach_file)(slug="projects/cityfix", filename="x.txt", content_base64=_b64(b"x"),
                                       agent="claude-code")
    assert "project/cityfix" in out["did_you_mean"]
    out = await _call(srv.attach_file)(slug="project/cityfix", filename="", content_base64=_b64(b"x"),
                                       agent="claude-code")
    assert "filename" in out["error"]
    out = await _call(srv.delete_attachment)(slug="project/cityfix", attachment_id=999, agent="claude-code")
    assert "999" in out["error"] and "get_document" in out["fix"]


async def test_private_doc_attachment_stays_out_of_search(mcp_db):
    await _call(srv.save_document)(slug="health/report", agent="claude-code", title="Report", content="Lab: x\n",
                                   privacy="private")
    await _call(srv.attach_file)(slug="health/report", filename="r.txt", content_base64=_b64(b"ferritin low"),
                                 agent="claude-code")
    assert await _call(srv.search_context)(query="ferritin") == []


async def test_admin_upload_and_download(mcp_db, client):
    h = {"host": "127.0.0.1:8001"}
    await _call(srv.save_document)(slug="note/fridge", agent="claude-code", title="Fridge", content="Brand: LG\n")
    r = await client.post("/api/admin/note/fridge/attachments", headers=h,
                          files={"file": ("manual.txt", b"defrost every month", "text/plain")})
    assert r.status_code in (200, 303), r.text
    doc = await _call(srv.get_document)(slug="note/fridge")
    att_id = doc["attachments"][0]["id"]
    r = await client.get(f"/api/admin/attachments/{att_id}", headers=h)
    assert r.status_code == 200 and r.content == b"defrost every month"
    assert "manual.txt" in r.headers["content-disposition"]
