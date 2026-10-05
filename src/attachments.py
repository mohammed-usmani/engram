"""Files attached to documents: PDF text via poppler (pdftotext), OCR via tesseract for images and
scanned PDFs. Their text is folded into the document's attachments_text, so search, recall and
embeddings cover it with no extra query path. Missing tools never fail an upload: the file is kept
and the reply says why there is no text.
"""
from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import shutil
import tempfile
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import ContextDocument, DocumentAttachment
from src.tool_errors import ToolInputError

MAX_BYTES = 20 * 1024 * 1024
_MAX_INDEXED_CHARS = 500_000  # ponytail: tsvector caps near 1MB; beyond this, attachments are stored but not indexed
_OCR_PAGES = 20
_TEXTUAL = ("application/json", "application/xml", "application/x-yaml", "application/csv")


def mime_of(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


async def _run(*args: str, data: bytes | None = None) -> str:
    p = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE if data is not None else None,
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await asyncio.wait_for(p.communicate(data), timeout=180)
    return out.decode("utf-8", "replace").strip()


async def _ocr_pdf(data: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.pdf"
        src.write_bytes(data)
        await _run("pdftoppm", "-r", "200", "-png", "-l", str(_OCR_PAGES), str(src), str(Path(tmp) / "page"))
        pages = sorted(Path(tmp).glob("page*.png"))
        return "\n\n".join([await _run("tesseract", str(p), "stdout") for p in pages]).strip()


async def extract_text(data: bytes, filename: str) -> tuple[str, str]:
    """(text, note) — note says how the text was obtained, or why there is none."""
    mime = mime_of(filename)
    if mime.startswith("text/") or mime in _TEXTUAL:
        return data.decode("utf-8", "replace").strip(), "read as text"
    if mime == "application/pdf":
        if not shutil.which("pdftotext"):
            return "", "no text extracted: install poppler (pdftotext) on the Engram machine"
        text = await _run("pdftotext", "-layout", "-", "-", data=data)
        if len(text) >= 20:
            return text, "PDF text layer"
        if shutil.which("tesseract") and shutil.which("pdftoppm"):
            return await _ocr_pdf(data), f"OCR of a scanned PDF (first {_OCR_PAGES} pages)"
        return text, "PDF has little or no text layer; install tesseract to OCR scans"
    if mime.startswith("image/"):
        if not shutil.which("tesseract"):
            return "", "no text extracted: install tesseract on the Engram machine for OCR"
        return await _run("tesseract", "stdin", "stdout", data=data), "OCR"
    return "", f"no text extracted: {mime} is stored and downloadable but not read"


def summary(a: DocumentAttachment, text_limit: int = 4000) -> dict:
    return {"id": a.id, "filename": a.filename, "mime": a.mime, "size": a.size,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "text": a.text[:text_limit], "text_truncated": len(a.text) > text_limit}


async def for_doc(session: AsyncSession, doc: ContextDocument) -> list[DocumentAttachment]:
    return list((await session.execute(select(DocumentAttachment).where(
        DocumentAttachment.document_id == doc.id).order_by(DocumentAttachment.id))).scalars())


async def _reindex(session: AsyncSession, doc: ContextDocument) -> None:
    from src.routers.admin import _refresh_derived
    doc.attachments_text = "\n\n".join(f"[{a.filename}]\n{a.text}" for a in await for_doc(session, doc)
                                       )[:_MAX_INDEXED_CHARS]
    await _refresh_derived(doc)


async def add(session: AsyncSession, doc: ContextDocument, filename: str, data: bytes,
              agent: str) -> tuple[DocumentAttachment, str]:
    filename = Path((filename or "").strip()).name
    if not filename:
        raise ToolInputError("`filename` is empty.", "Pass the file's name with its extension, e.g. 'policy.pdf' — "
                             "the extension decides how text is extracted.")
    if not data:
        raise ToolInputError("The file is empty (0 bytes).", "Send the file's full contents.")
    if len(data) > MAX_BYTES:
        raise ToolInputError(f"The file is {len(data) / 1e6:.1f} MB; the limit is {MAX_BYTES // 2**20} MB.",
                             "Compress or split it, or attach only the pages that matter.")
    sha = hashlib.sha256(data).hexdigest()
    existing = (await session.execute(select(DocumentAttachment).where(
        DocumentAttachment.document_id == doc.id, DocumentAttachment.sha256 == sha))).scalar_one_or_none()
    if existing:
        return existing, "already attached (same contents); nothing changed"
    text, note = await extract_text(data, filename)
    att = DocumentAttachment(document_id=doc.id, filename=filename, mime=mime_of(filename), size=len(data),
                             sha256=sha, text=text, data=data, created_by=agent)
    session.add(att)
    await session.flush()
    await _reindex(session, doc)
    await session.commit()
    return att, note


async def remove(session: AsyncSession, doc: ContextDocument, attachment_id: int) -> None:
    att = await session.get(DocumentAttachment, attachment_id)
    if att is None or att.document_id != doc.id:
        ids = [a.id for a in await for_doc(session, doc)]
        raise ToolInputError(f"Document '{doc.slug}' has no attachment {attachment_id}.",
                             f"Its attachment ids are {ids or 'none'}; get_document(slug) lists them.")
    await session.delete(att)
    await session.flush()
    await _reindex(session, doc)
    await session.commit()
