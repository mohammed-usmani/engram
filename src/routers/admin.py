from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import get_session
from src.doc_types import resolve_type, type_names
from src.models import ContextDocument, Source
from src.privacy import PRIVACY_LEVELS, readable
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
        doc.embedding = await embed(f"{doc.title}\n{doc.content}"[:6000])
    except Exception as e:  # Ollama down: keep the edit, re-embed later via /api/admin/embed
        log.warning("re-embed failed for %s: %s", doc.slug, e)
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


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


@router.get("", response_class=HTMLResponse)
async def admin_index(
    request: Request,
    type: str | None = None,
    q: str | None = None,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    stmt = select(ContextDocument).where(readable(ContextDocument.privacy))
    if type:
        stmt = stmt.where(ContextDocument.type == type)
    if q:
        tsq = func.plainto_tsquery("english", q)
        stmt = stmt.where(ContextDocument.search_vector.op("@@")(tsq))
        stmt = stmt.order_by(func.ts_rank(ContextDocument.search_vector, tsq).desc())
    else:
        stmt = stmt.order_by(ContextDocument.type, ContextDocument.title)
    docs = (await session.execute(stmt.limit(500))).scalars().all()
    return templates.TemplateResponse("list.html", {
        "request": request, "docs": docs,
        "types": await type_names(session), "type": type, "q": q,
        "import_summary": request.query_params.get("import"),
        "active_page": "documents",
    })


@router.get("/new", response_class=HTMLResponse)
async def new_form(request: Request, session: AsyncSession = Depends(get_session),
                   _: None = Depends(_check_token)):
    return templates.TemplateResponse("edit.html", {
        "request": request, "is_new": True, "doc": None, "privacy_levels": PRIVACY_LEVELS,
        "types": await type_names(session), "error": None,
        "active_page": "documents",
    })


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
        return templates.TemplateResponse("edit.html", {
            "request": request, "is_new": True, "doc": None, "privacy_levels": PRIVACY_LEVELS,
            "types": await type_names(session), "error": f"{e.payload['error']} {e.payload['fix']}",
            "active_page": "documents",
        }, status_code=400)

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
    doc = (await session.execute(select(ContextDocument).where(
        ContextDocument.slug == slug, readable(ContextDocument.privacy)))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, "not found")
    return templates.TemplateResponse("edit.html", {
        "request": request, "is_new": False, "doc": doc, "privacy_levels": PRIVACY_LEVELS,
        "types": await type_names(session), "error": None,
        "active_page": "documents",
    })


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
        return templates.TemplateResponse("edit.html", {
            "request": request, "is_new": False, "doc": doc, "privacy_levels": PRIVACY_LEVELS,
            "types": await type_names(session), "error": f"{e.payload['error']} {e.payload['fix']}",
            "active_page": "documents",
        }, status_code=400)
    await session.commit()
    return RedirectResponse(f"/api/admin/{slug}", status_code=303)
