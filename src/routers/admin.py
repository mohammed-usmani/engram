from __future__ import annotations

import re
from pathlib import Path

from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from src.config import settings
from src.database import get_session
from src.doc_types import resolve_type, type_names
from src import attachments
from src.models import ContextDocument, DocumentAttachment, Source
from src.privacy import PRIVACY_LEVELS, is_remote, readable
from src.tool_errors import ToolInputError
import logging

from src.seed.loader import run_seed
from src.seed.parser import derive_fields
from src.memory.api import needs_token
from src.services.ollama_client import embed
from src.services.embeddings import embed_all_documents

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


async def _refresh_derived(doc: ContextDocument) -> None:
    """Sections/metadata come from the text, and search must see the edit immediately."""
    doc.sections, doc.doc_metadata = derive_fields(doc.content)
    try:
        doc.embedding = await embed(f"{doc.title}\n{doc.content}\n{doc.attachments_text or ''}"[:6000])
    except Exception as e:  # Ollama down: keep the edit, re-embed later via /api/admin/embed
        log.warning("re-embed failed for %s: %s", doc.slug, e)


def _check_token(request: Request) -> None:
    # Same rule as AuthMiddleware: your own browser on localhost needs no token.
    if settings.admin_token is None or not needs_token(
            request.url.path, dict(request.scope.get("headers") or []), request.scope.get("client")):
        return
    token = (
        request.query_params.get("token")
        or request.cookies.get("admin_token")
        or (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
    )
    if token != settings.admin_token:
        raise HTTPException(401, "admin token required")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80] or "untitled"


def _parse_tags(raw: str) -> list[str]:
    return [t.strip().lower() for t in (raw or "").split(",") if t.strip()]


# Registered on a separate router so these win the path match over /{slug:path}.
reseed_router = APIRouter(prefix="/api/admin", tags=["admin"])


@reseed_router.post("/embed")
async def embed_documents(
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    summary = await embed_all_documents(session, force=True)
    await session.commit()
    return summary


@reseed_router.get("/dump")
async def dump_all(
    format: str = "md",
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    """Dump every context document. format=md (human-readable markdown) or format=json."""
    from fastapi.responses import JSONResponse, PlainTextResponse

    rows = (
        await session.execute(
            select(ContextDocument).where(readable(ContextDocument.privacy))
            .order_by(ContextDocument.type, ContextDocument.title)
        )
    ).scalars().all()

    if format == "json":
        payload = [
            {
                "slug": r.slug,
                "title": r.title,
                "type": r.type,
                "privacy": r.privacy,
                "tags": r.tags,
                "source": r.source.value,
                "content": r.content,
                "sections": r.sections,
                "metadata": r.doc_metadata,
            }
            for r in rows
        ]
        return JSONResponse(
            content=payload,
            headers={"Content-Disposition": 'attachment; filename="context-dump.json"'},
        )

    # Markdown: one document per section
    parts: list[str] = [
        "# Personal Context Dump",
        f"\n_Total documents: {len(rows)}_\n",
    ]
    for r in rows:
        parts.append(f"\n---\n\n## [{r.type}] {r.title}")
        parts.append(f"\n**Slug:** `{r.slug}`  \n**Tags:** {', '.join(r.tags) or '—'}  \n**Source:** {r.source.value}")
        if r.doc_metadata:
            parts.append("\n**Metadata:**")
            for k, v in r.doc_metadata.items():
                parts.append(f"- {k}: {v}")
        parts.append("\n" + r.content.strip() + "\n")
    body = "\n".join(parts)
    return PlainTextResponse(
        content=body,
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="context-dump.md"'},
    )


@reseed_router.post("/import")
async def import_files(
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    """Add documents from DATA_DIR that aren't in the database yet. Never overwrites."""
    summary = await run_seed(session, settings.data_dir)
    await session.commit()
    return RedirectResponse(f"/api/admin?import={summary}", status_code=303)


TEMPLATES = {
    "passport": {"title": "Passport", "type": "travel", "privacy": "local-only",
                 "content": "Number: \nIssued: \nExpires: \nPlace of issue: \n\nNotes:\nRenew about 6 months before it expires.\n"},
    "insurance": {"title": "Insurance policy", "type": "finance", "privacy": "private",
                  "content": "Insurer: \nPolicy number: \nCovers: \nRenews: \nPremium: \n\nNotes:\n"},
    "person": {"title": "", "type": "person", "privacy": "normal",
               "content": "Relation: \nBirthday: \nPhone: \n\nNotes:\nThings they like, plans, what to remember.\n"},
    "health": {"title": "Health report", "type": "health", "privacy": "private",
               "content": "Lab: \nDate: \nNext check: \n\nResults:\nAttach the PDF below once saved.\n"},
}
EVERYDAY = ["note", "person", "health", "finance", "home", "travel", "learning", "reference"]


@router.get("", response_class=HTMLResponse)
async def admin_index(
    request: Request,
    type: str | None = None,
    q: str | None = None,
    privacy: str | None = None,
    has: str | None = None,
    sort: str = "updated",
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    from src import dates
    from src.doc_types import list_types
    from src.models import DocumentAttachment
    from src.ui.templating import render
    stmt = select(ContextDocument).where(readable(ContextDocument.privacy))
    if type:
        stmt = stmt.where(ContextDocument.type == type)
    if privacy in PRIVACY_LEVELS:
        stmt = stmt.where(ContextDocument.privacy == privacy)
    if has == "attachments":
        stmt = stmt.where(ContextDocument.id.in_(select(DocumentAttachment.document_id)))
    if q:
        tsq = func.plainto_tsquery("english", q)
        stmt = stmt.where(ContextDocument.search_vector.op("@@")(tsq))
        stmt = stmt.order_by(func.ts_rank(ContextDocument.search_vector, tsq).desc())
    else:
        stmt = stmt.order_by(*{"title": [func.lower(ContextDocument.title)],
                               "type": [ContextDocument.type, func.lower(ContextDocument.title)]}.get(
            sort, [ContextDocument.updated_at.desc()]))
    docs = (await session.execute(stmt.limit(500))).scalars().all()
    date_keys = {*dates.ONE_OFF, *dates.YEARLY}
    has_date = lambda d: any(k.strip().lower() in date_keys for k in (d.doc_metadata or {}))  # noqa: E731
    if has == "dates":
        docs = [d for d in docs if has_date(d)]
    all_docs = (await session.execute(select(ContextDocument).where(readable(ContextDocument.privacy)))).scalars().all()
    att_counts = dict((await session.execute(select(DocumentAttachment.document_id, func.count())
                                             .group_by(DocumentAttachment.document_id))).all())
    types = await list_types(session)
    return await render(
        request, "list.html", session, "documents", docs=docs, q=q or "", type=type or "", privacy=privacy or "",
        has=has or "", sort=sort, used_types=[t for t in types if t["count"]],
        empty_types=[t for t in types if not t["count"] and t["name"] in EVERYDAY],
        privacy_counts={lvl: sum(d.privacy == lvl for d in all_docs) for lvl in PRIVACY_LEVELS},
        n_attached=len(att_counts), n_dated=sum(has_date(d) for d in all_docs), total=len(all_docs),
        att_counts=att_counts, templates_=TEMPLATES, import_summary=request.query_params.get("import"),
    )


async def _editor(request: Request, session: AsyncSession, doc: ContextDocument | None, *, error: str | None = None,
                  prefill: dict | None = None, status_code: int = 200):
    """Editor page context: the text plus everything Engram derives from it."""
    from src import dates
    from src.memory.entities import match_text
    from src.ui.templating import render
    ctx = {"is_new": doc is None, "doc": doc, "privacy_levels": PRIVACY_LEVELS, "types": await type_names(session),
           "error": error, "prefill": prefill or {}}
    if doc is not None:
        items, unreadable = await dates.upcoming(session, 366)
        ctx.update(
            tracked=[i for i in items if i["slug"] == doc.slug],
            unreadable=[u for u in unreadable if u["slug"] == doc.slug],
            body_dates=dates.body_dates(doc.content, doc.doc_metadata),
            attachments=await attachments.for_doc(session, doc),
            topics=(await match_text(session, f"{doc.title}\n{doc.content}"))[:12],
            privacy_hint=_privacy_hint(doc) if doc.privacy == "normal" else None,
        )
    return await render(request, "edit.html", session, "documents", status_code=status_code, **ctx)


_SENSITIVE = [
    (re.compile(r"\b(lpa|ctc|salary|₹|rs\.?\s?\d|bank|account number|ifsc|upi)\b", re.I), "salary or bank details"),
    (re.compile(r"\b(passport|aadhaar|pan card|licen[cs]e number|ssn)\b", re.I), "ID numbers"),
    (re.compile(r"\b(diagnos|prescription|blood|medical|therapy|symptom)", re.I), "health details"),
    (re.compile(r"\b(password|pin|otp)\b", re.I), "credentials"),
]


def _privacy_hint(doc: ContextDocument) -> str | None:
    found = [label for rx, label in _SENSITIVE if rx.search(doc.content or "")]
    return (f"This mentions {', '.join(found)}. Private keeps it out of automatic briefs and search."
            if found else None)


@router.get("/new", response_class=HTMLResponse)
async def new_form(request: Request, template: str | None = None, session: AsyncSession = Depends(get_session),
                   _: None = Depends(_check_token)):
    return await _editor(request, session, None, prefill=TEMPLATES.get(template or "", {}))


@router.post("/new", response_class=HTMLResponse)
async def new_submit(
    request: Request,
    type: str = Form(...),
    slug: str = Form(""),
    title: str = Form(...),
    tags: str = Form(""),
    content: str = Form(...),
    type_description: str = Form(""),
    privacy: str = Form("normal"),
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    try:
        doc_type = await resolve_type(session, type.strip(), description=type_description)
        if privacy not in PRIVACY_LEVELS:
            raise ToolInputError(f"privacy must be one of {', '.join(PRIVACY_LEVELS)}", "")
    except ToolInputError as e:
        return await _editor(request, session, None, error=f"{e.payload['error']} {e.payload['fix']}", status_code=400,
                             prefill={"title": title, "type": type, "content": content, "privacy": privacy})

    chosen_slug = slug.strip() or f"{doc_type}/{_slugify(title)}"
    existing = (await session.execute(select(ContextDocument).where(ContextDocument.slug == chosen_slug))).scalar_one_or_none()
    if existing is not None:
        i = 2
        while (await session.execute(select(ContextDocument).where(ContextDocument.slug == f"{chosen_slug}-{i}"))).scalar_one_or_none() is not None:
            i += 1
        chosen_slug = f"{chosen_slug}-{i}"

    doc = ContextDocument(type=doc_type, slug=chosen_slug, title=title, tags=_parse_tags(tags),
                          content=content, sections={}, source=Source.MANUAL, doc_metadata={},
                          updated_by="admin", privacy=privacy)
    await _refresh_derived(doc)
    session.add(doc)
    await session.commit()
    return RedirectResponse(f"/api/admin/{chosen_slug}", status_code=303)


async def _readable_doc(session: AsyncSession, slug: str) -> ContextDocument:
    doc = (await session.execute(select(ContextDocument).where(
        ContextDocument.slug == slug, readable(ContextDocument.privacy)))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, "not found")
    return doc


# Attachment routes come before the /{slug:path} catch-alls below.
@router.get("/attachments/{attachment_id}")
async def download_attachment(attachment_id: int, session: AsyncSession = Depends(get_session),
                              _: None = Depends(_check_token)):
    a = await session.get(DocumentAttachment, attachment_id, options=[undefer(DocumentAttachment.data)])
    doc = a and await session.get(ContextDocument, a.document_id)
    if a is None or doc is None or not _is_readable(doc):
        raise HTTPException(404, "not found")
    return Response(a.data, media_type=a.mime,
                    headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(a.filename)}"})


@router.post("/attachments/{attachment_id}/delete")
async def delete_attachment(attachment_id: int, session: AsyncSession = Depends(get_session),
                            _: None = Depends(_check_token)):
    a = await session.get(DocumentAttachment, attachment_id)
    doc = a and await session.get(ContextDocument, a.document_id)
    if a is None or doc is None or not _is_readable(doc):
        raise HTTPException(404, "not found")
    await attachments.remove(session, doc, attachment_id)
    return RedirectResponse(f"/api/admin/{doc.slug}", status_code=303)


@router.post("/{slug:path}/attachments")
async def upload_attachment(slug: str, request: Request, file: UploadFile = File(...),
                            session: AsyncSession = Depends(get_session), _: None = Depends(_check_token)):
    """Browser form or REST multipart (`file` field). JSON reply unless the caller wants HTML."""
    doc = await _readable_doc(session, slug)
    try:
        a, note = await attachments.add(session, doc, file.filename or "", await file.read(), "admin")
    except ToolInputError as e:
        raise HTTPException(400, f"{e.payload['error']} {e.payload['fix']}") from None
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(f"/api/admin/{slug}", status_code=303)
    return {"id": a.id, "filename": a.filename, "size": a.size, "text_chars": len(a.text), "extraction": note}


def _is_readable(doc: ContextDocument) -> bool:
    return not (is_remote() and doc.privacy == "local-only")


@router.post("/{slug:path}/delete")
async def delete_doc(
    slug: str,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    doc = (await session.execute(select(ContextDocument).where(
        ContextDocument.slug == slug, readable(ContextDocument.privacy)))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, "not found")
    await session.delete(doc)
    await session.commit()
    return RedirectResponse("/api/admin", status_code=303)


@router.get("/{slug:path}", response_class=HTMLResponse)
async def edit_form(
    slug: str, request: Request,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    return await _editor(request, session, await _readable_doc(session, slug))


@router.post("/{slug:path}", response_class=HTMLResponse)
async def edit_submit(
    slug: str, request: Request,
    type: str = Form(...),
    title: str = Form(...),
    tags: str = Form(""),
    content: str = Form(...),
    type_description: str = Form(""),
    privacy: str = Form(""),
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    doc = (await session.execute(select(ContextDocument).where(
        ContextDocument.slug == slug, readable(ContextDocument.privacy)))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, "not found")
    try:
        doc.type = await resolve_type(session, type.strip(), description=type_description)
        if privacy:
            if privacy not in PRIVACY_LEVELS:
                raise ToolInputError(f"privacy must be one of {', '.join(PRIVACY_LEVELS)}", "")
            doc.privacy = privacy
        doc.title = title
        doc.tags = _parse_tags(tags)
        doc.content = content
        doc.source = Source.MANUAL
        doc.updated_by = "admin"
        await _refresh_derived(doc)
    except ToolInputError as e:
        await session.rollback()
        return await _editor(request, session, await _readable_doc(session, slug),
                             error=f"{e.payload['error']} {e.payload['fix']}", status_code=400,
                             prefill={"title": title, "type": type, "content": content, "privacy": privacy,
                                      "tags": tags, "type_description": type_description})
    await session.commit()
    return RedirectResponse(f"/api/admin/{slug}", status_code=303)
