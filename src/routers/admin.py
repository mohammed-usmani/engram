from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import get_session
from src.models import ContextDocument, DocType, Source
from src.seed.loader import run_seed
from src.services.embeddings import embed_all_documents

router = APIRouter(prefix="/api/admin", tags=["admin"])
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


def _check_token(request: Request) -> None:
    if settings.admin_token is None:
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


def _parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    return json.loads(raw)


# Register /reseed on a separate router so it wins the path match over /{slug:path}.
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
            select(ContextDocument).order_by(ContextDocument.type, ContextDocument.title)
        )
    ).scalars().all()

    if format == "json":
        payload = [
            {
                "slug": r.slug,
                "title": r.title,
                "type": r.type.value,
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
        parts.append(f"\n---\n\n## [{r.type.value}] {r.title}")
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


@reseed_router.post("/reseed")
async def reseed(
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    summary = await run_seed(session, settings.data_dir)
    await session.commit()
    return RedirectResponse(f"/api/admin?reseed={summary}", status_code=303)


@router.get("", response_class=HTMLResponse)
async def admin_index(
    request: Request,
    type: str | None = None,
    q: str | None = None,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    stmt = select(ContextDocument)
    if type:
        stmt = stmt.where(ContextDocument.type == DocType(type))
    if q:
        tsq = func.plainto_tsquery("english", q)
        stmt = stmt.where(ContextDocument.search_vector.op("@@")(tsq))
        stmt = stmt.order_by(func.ts_rank(ContextDocument.search_vector, tsq).desc())
    else:
        stmt = stmt.order_by(ContextDocument.type, ContextDocument.title)
    docs = (await session.execute(stmt.limit(500))).scalars().all()
    return templates.TemplateResponse("list.html", {
        "request": request, "docs": docs,
        "types": [t.value for t in DocType], "type": type, "q": q,
        "reseed_summary": request.query_params.get("reseed"),
        "active_page": "documents",
    })


@router.get("/new", response_class=HTMLResponse)
async def new_form(request: Request, _: None = Depends(_check_token)):
    return templates.TemplateResponse("edit.html", {
        "request": request, "is_new": True, "doc": None,
        "types": [t.value for t in DocType], "error": None,
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
    sections: str = Form("{}"),
    metadata: str = Form("{}"),
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    try:
        doc_type = DocType(type)
        parsed_sections = _parse_json(sections)
        parsed_metadata = _parse_json(metadata)
    except Exception as e:
        return templates.TemplateResponse("edit.html", {
            "request": request, "is_new": True, "doc": None,
            "types": [t.value for t in DocType], "error": f"invalid input: {e}",
            "active_page": "documents",
        }, status_code=400)

    chosen_slug = slug.strip() or f"{doc_type.value}/{_slugify(title)}"
    existing = (await session.execute(select(ContextDocument).where(ContextDocument.slug == chosen_slug))).scalar_one_or_none()
    if existing is not None:
        i = 2
        while (await session.execute(select(ContextDocument).where(ContextDocument.slug == f"{chosen_slug}-{i}"))).scalar_one_or_none() is not None:
            i += 1
        chosen_slug = f"{chosen_slug}-{i}"

    session.add(ContextDocument(
        type=doc_type, slug=chosen_slug, title=title, tags=_parse_tags(tags),
        content=content, sections=parsed_sections, source=Source.MANUAL, doc_metadata=parsed_metadata,
    ))
    await session.commit()
    return RedirectResponse(f"/api/admin/{chosen_slug}", status_code=303)


@router.post("/{slug:path}/delete")
async def delete_doc(
    slug: str,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    doc = (await session.execute(select(ContextDocument).where(ContextDocument.slug == slug))).scalar_one_or_none()
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
    doc = (await session.execute(select(ContextDocument).where(ContextDocument.slug == slug))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, "not found")
    return templates.TemplateResponse("edit.html", {
        "request": request, "is_new": False, "doc": doc,
        "types": [t.value for t in DocType], "error": None,
        "active_page": "documents",
    })


@router.post("/{slug:path}", response_class=HTMLResponse)
async def edit_submit(
    slug: str, request: Request,
    type: str = Form(...),
    title: str = Form(...),
    tags: str = Form(""),
    content: str = Form(...),
    sections: str = Form("{}"),
    metadata: str = Form("{}"),
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_check_token),
):
    doc = (await session.execute(select(ContextDocument).where(ContextDocument.slug == slug))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, "not found")
    try:
        doc.type = DocType(type)
        doc.title = title
        doc.tags = _parse_tags(tags)
        doc.content = content
        doc.sections = _parse_json(sections)
        doc.doc_metadata = _parse_json(metadata)
    except Exception as e:
        return templates.TemplateResponse("edit.html", {
            "request": request, "is_new": False, "doc": doc,
            "types": [t.value for t in DocType], "error": f"invalid input: {e}",
            "active_page": "documents",
        }, status_code=400)
    await session.commit()
    return RedirectResponse(f"/api/admin/{slug}", status_code=303)
