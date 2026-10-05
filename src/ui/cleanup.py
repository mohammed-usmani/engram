"""Cleanup page: suggestions to merge, dedupe and tidy topics; every fix is a confirmed JSON call."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src import settings_store as store
from src.database import get_session
from src.memory import cleanup
from src.memory.entities import CANON_KINDS
from src.ui.templating import render

router = APIRouter()


@router.get("/cleanup", response_class=HTMLResponse, include_in_schema=False)
async def cleanup_page(request: Request, m: int = 20, d: int = 20, o: int = 25,
                       session: AsyncSession = Depends(get_session)):
    data = await cleanup._load(session)
    merges = await cleanup.merge_groups(session, data)
    dupes = await cleanup.dupe_groups(session)
    kinds = await cleanup.kinds(session)
    once = await cleanup.once(session, data)
    return await render(
        request, "cleanup.html", session, "cleanup",
        merges=merges, dupes=dupes, kinds=kinds, once=once, limits={"m": m, "d": d, "o": o},
        canon=CANON_KINDS, me=await cleanup.self_suggestion(session), recall=store.get("recall") or {},
    )


class MergeBody(BaseModel):
    keep: str = Field(..., description="Slug of the topic that stays")
    merge: list[str] = Field(..., description="Slugs folded into it")


class DismissBody(BaseModel):
    key: str


class DedupeBody(BaseModel):
    keep: int
    remove: list[int]


class KindsBody(BaseModel):
    map: dict[str, str] = Field(..., description="kind -> one of company, person, project, topic, place, tool, org")


class FoldItem(BaseModel):
    slug: str
    into: str


class FoldBody(BaseModel):
    items: list[FoldItem]


@router.post("/api/cleanup/merge")
async def merge(body: MergeBody, session: AsyncSession = Depends(get_session)):
    return await cleanup.merge(session, body.keep, body.merge)


@router.post("/api/cleanup/dismiss")
async def dismiss(body: DismissBody, session: AsyncSession = Depends(get_session)):
    return await cleanup.dismiss(session, body.key)


@router.post("/api/cleanup/dedupe")
async def dedupe(body: DedupeBody, session: AsyncSession = Depends(get_session)):
    return await cleanup.dedupe(session, body.keep, body.remove)


@router.post("/api/cleanup/kinds")
async def kinds(body: KindsBody, session: AsyncSession = Depends(get_session)):
    return await cleanup.apply_kinds(session, body.map)


@router.post("/api/cleanup/fold")
async def fold(body: FoldBody, session: AsyncSession = Depends(get_session)):
    return await cleanup.fold(session, [i.model_dump() for i in body.items])
