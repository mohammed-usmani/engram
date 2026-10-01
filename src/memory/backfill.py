"""Import structured history that already exists on disk.

`uv run python -m src.memory.backfill [path/to/applications.md]` — rerunnable; status changes update the episode.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, time, timezone
from pathlib import Path

from sqlalchemy import select, text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from src.memory import USER_TZ as _TZ
from src.memory.entities import resolve
from src.memory.models import Episode
from src.services.ollama_client import embed

DEFAULT_APPS = Path(os.environ.get("ENGRAM_APPLICATIONS_MD", Path.home() / "applications.md"))


def _rows(md: str) -> list[dict]:
    lines = [l.strip() for l in md.splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = [h.strip().lower() for h in lines[0].strip("|").split("|")]
    out = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        row = dict(zip(header, cells))
        if row.get("date") and row.get("company"):
            out.append(row)
    return out


async def sync_applications(session: AsyncSession, path: Path = DEFAULT_APPS) -> dict:
    added = updated = 0
    for row in _rows(Path(path).read_text()):
        try:
            day = datetime.fromisoformat(row["date"]).date()
        except ValueError:
            continue
        key = f"{day.isoformat()}|{row['company'].lower()}"
        status = (row.get("status") or "applied").lower()
        existing = (await session.execute(
            select(Episode).where(sql("payload->>'applications_key' = :k").bindparams(k=key)))).scalar_one_or_none()
        if existing:
            if existing.outcome != status:
                existing.outcome = status
                updated += 1
            continue
        slugs = await resolve(session, [{"name": row["company"], "kind": "company"},
                                        {"name": "Job search", "kind": "topic"}])
        summary = f"Applied to {row['company']} for {row.get('role') or 'a role'}" + (
            f" via {row['via']}" if row.get("via") else "")
        session.add(Episode(
            occurred_at=datetime.combine(day, time(12), _TZ).astimezone(timezone.utc), kind="applied",
            summary=summary, outcome=status, importance=3, entities=sorted(slugs.values()),
            payload={"applications_key": key, "cv": row.get("cv"), "source": "applications.md"},
            source_agent="backfill", embedding=await embed(summary),
        ))
        added += 1
    if added or updated:
        await session.execute(sql("UPDATE entities SET dirty = true WHERE slug = 'topic:job-search'"))
    await session.commit()
    return {"added": added, "updated": updated}


async def _main() -> None:
    from src.database import AsyncSessionLocal
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_APPS
    async with AsyncSessionLocal() as s:
        print(await sync_applications(s, path))


if __name__ == "__main__":
    asyncio.run(_main())
