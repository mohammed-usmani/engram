"""Recording traces. A failure to record is logged and swallowed: tracing must never break the
thing being traced, so each record goes in its own savepoint on the caller's session."""
from __future__ import annotations

import logging

from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)
KEEP_DAYS = 30


async def record(session: AsyncSession, kind: str, source: str | None, input: str, data: dict,
                 flags: list[str], duration_ms: int) -> int | None:
    from src.eval.models import Trace
    try:
        async with session.begin_nested():
            t = Trace(kind=kind, source=(source or "unknown")[:32], input=(input or "")[:20000], data=data,
                      flags=flags, duration_ms=duration_ms)
            session.add(t)
            await session.flush()
            tid = t.id
        await session.commit()
        return tid
    except Exception as e:
        log.warning("trace not recorded: %s", e)
        return None


async def prune(session: AsyncSession) -> int:
    n = (await session.execute(sql(f"delete from eval_traces where created_at < now() - interval '{KEEP_DAYS} days'"))).rowcount
    await session.commit()
    return n
