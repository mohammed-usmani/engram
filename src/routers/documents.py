from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.models import ContextDocument
from src.privacy import automatic, readable
from src.schemas import DocumentRead, DocumentSummary

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("", response_model=list[DocumentSummary])
async def list_documents(
    type: str | None = None,
    tag: str | None = None,
    q: str | None = None,
    limit: int = Query(default=100, le=500),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(ContextDocument).where(readable(ContextDocument.privacy))
    if type is not None:
        stmt = stmt.where(ContextDocument.type == type)
    if tag is not None:
        stmt = stmt.where(ContextDocument.tags.contains([tag]))
    if q:
        tsq = func.plainto_tsquery("english", q)
        stmt = stmt.where(ContextDocument.search_vector.op("@@")(tsq), automatic(ContextDocument.privacy))
        stmt = stmt.order_by(func.ts_rank(ContextDocument.search_vector, tsq).desc())
    else:
        stmt = stmt.order_by(ContextDocument.type, ContextDocument.title)
    stmt = stmt.limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    return rows


@router.get("/{slug:path}", response_model=DocumentRead)
async def get_document(slug: str, session: AsyncSession = Depends(get_session)):
    row = (
        await session.execute(select(ContextDocument).where(
            ContextDocument.slug == slug, readable(ContextDocument.privacy)))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, f"document not found: {slug}")
    return row
