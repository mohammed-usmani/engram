"""One Jinja environment for every admin page, plus the sidebar's live numbers."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select, text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from src.memory import USER_TZ, local_date
from src.models import ContextDocument
from src.privacy import is_remote, readable

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def _num(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def _ago(dt: datetime | None) -> str:
    if not dt:
        return "never"
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    if secs < 90:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    local = dt.astimezone(USER_TZ)
    today = datetime.now(USER_TZ).date()
    if local.date() == today:
        return "today, " + local.strftime("%H:%M")
    if (today - local.date()).days == 1:
        return "yesterday, " + local.strftime("%H:%M")
    return local.strftime("%b %-d")


def _day(dt: datetime | None) -> str:
    return dt.astimezone(USER_TZ).strftime("%b %-d") if dt else ""


def _time(dt: datetime | None) -> str:
    return dt.astimezone(USER_TZ).strftime("%H:%M") if dt else ""


templates.env.filters.update(local_date=local_date, num=_num, ago=_ago, day=_day, hm=_time,
                             local_time=lambda dt: dt.astimezone(USER_TZ).strftime("%Y-%m-%d %H:%M"),
                             longdate=lambda dt: dt.astimezone(USER_TZ).strftime("%A, %-d %B %Y"),
                             kindlabel=lambda k: (k or "other").replace("_", " "))


def last_backup() -> datetime | None:
    folder = Path(os.environ.get("ENGRAM_BACKUP_DIR", Path.home() / "dev/personal/engram-data/backups"))
    archives = sorted(folder.glob("engram-*.tar.age"), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
    return datetime.fromtimestamp(archives[-1].stat().st_mtime, timezone.utc) if archives else None


_cache: dict[str, tuple[float, dict]] = {}


async def nav_counts(session: AsyncSession) -> dict:
    """Sidebar numbers; cached briefly so every page load isn't five counts."""
    key = "remote" if is_remote() else "local"
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < 30:
        return hit[1]
    from src.memory import facts
    from src.memory.models import Entity, Episode, Reflection
    out = {
        "documents": await session.scalar(select(func.count()).select_from(ContextDocument)
                                          .where(readable(ContextDocument.privacy))),
        "topics": await session.scalar(select(func.count()).select_from(Entity)),
        "events": await session.scalar(select(func.count()).select_from(Episode)),
        "lessons": await session.scalar(select(func.count()).select_from(Reflection)),
        "facts": await session.scalar(sql(
            f"select count(*) from {facts.table()} where payload->>'superseded_by' is null")),
    }
    try:
        from src.memory import cleanup
        out["cleanup"] = await cleanup.badge(session)
    except Exception:  # suggestions are a nicety; the sidebar must always render
        out["cleanup"] = 0
    _cache[key] = (time.monotonic(), out)
    return out


def invalidate() -> None:
    _cache.clear()


async def render(request: Request, name: str, session: AsyncSession, active: str, status_code: int = 200, **ctx):
    ctx.update(request=request, active=active, nav=await nav_counts(session), remote=is_remote(),
               backup=last_backup(), now=datetime.now(timezone.utc))
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)
