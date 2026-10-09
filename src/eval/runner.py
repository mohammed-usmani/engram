"""Runs an evaluation layer and keeps the result, so the Evaluation page can show scores over time.

    uv run python -m src.eval.runner health|recall|extract [provider model]
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.eval.models import EvalRun

log = logging.getLogger(__name__)
_running: set[str] = set()


async def run(session: AsyncSession, kind: str, trigger: str = "manual", provider: str | None = None,
              model: str | None = None) -> EvalRun:
    if kind in _running:
        raise RuntimeError(f"A {kind} evaluation is already running.")
    _running.add(kind)
    r = EvalRun(kind=kind, trigger=trigger, summary={}, details=[])
    session.add(r)
    await session.commit()
    try:
        if kind == "health":
            from src.eval import health
            score, details = await health.run(session)
            summary = {s: sum(c["status"] == s for c in details) for s in ("ok", "warn", "bad")}
        elif kind == "recall":
            from src.eval import recall_eval
            score, details, summary = await recall_eval.run(session)
        elif kind == "extract":
            from src.eval import extraction
            from src.memory import llm
            if not provider:
                provider, model = (llm.steps("extract") or [("ollama", None)])[0]
            score, details, summary = await extraction.run(provider, model, 1 if provider == "ollama" else 4)
        else:
            raise ValueError(f"unknown evaluation '{kind}'")
        r.score, r.details, r.summary = score, details, summary
    except Exception as e:
        log.exception("evaluation %s failed", kind)
        r.summary = {"error": f"{type(e).__name__}: {e}"[:500]}
    finally:
        _running.discard(kind)
        r.finished_at = datetime.now(timezone.utc)
        await session.commit()
    return r


async def close_interrupted(session: AsyncSession) -> int:
    """At startup: runs left unfinished by a crash or restart are marked interrupted, not left 'running' forever."""
    from sqlalchemy import update
    n = (await session.execute(update(EvalRun).where(EvalRun.finished_at.is_(None)).values(
        finished_at=datetime.now(timezone.utc), summary={"error": "interrupted (the server stopped mid-run)"}))).rowcount
    await session.commit()
    return n


async def latest(session: AsyncSession, kind: str, n: int = 30) -> list[EvalRun]:
    return list((await session.execute(select(EvalRun).where(EvalRun.kind == kind, EvalRun.finished_at.isnot(None))
                                       .order_by(EvalRun.id.desc()).limit(n))).scalars())


async def nightly(session: AsyncSession) -> None:
    """Health and recall every night (no paid calls); extraction only when asked, since it costs."""
    from src.eval.traces import prune
    for kind in ("health", "recall"):
        await run(session, kind, "nightly")
    await prune(session)


def main() -> None:
    from src.database import AsyncSessionLocal
    from src import settings_store

    async def go():
        async with AsyncSessionLocal() as s:
            await settings_store.load(s)
            r = await run(s, sys.argv[1], "cli", *(sys.argv[2:4]))
            print(f"{r.kind}: score {r.score} · {r.summary}")
    asyncio.run(go())


if __name__ == "__main__":
    main()
