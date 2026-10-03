"""Background loop: drain the ingest queue; consolidate every few hours or when asked.

Runs inside the API process (main.py lifespan) or standalone: `python -m src.memory.worker`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from src.database import AsyncSessionLocal
from src.memory.consolidate import consolidate
from src.memory.ingest import CONCURRENCY, process_pending

log = logging.getLogger(__name__)

_soon = False


def request_consolidation() -> None:
    """Consolidate at the next idle moment (e.g. after a session ends)."""
    global _soon
    _soon = True


async def _slot(stop: asyncio.Event, idle_sleep: float, consolidates: bool) -> None:
    """One worker slot: claims the next due job as soon as it's free, so a stuck LLM call only
    holds its own slot (a fixed batch waited for its slowest job, one hang blocked four)."""
    global _soon
    every = float(os.environ.get("CONSOLIDATE_EVERY_H", "6")) * 3600
    last = time.monotonic()
    while not stop.is_set():
        try:
            async with AsyncSessionLocal() as session:
                n = await process_pending(session)
                if consolidates and n == 0 and (_soon or time.monotonic() - last >= every):
                    _soon = False
                    last = time.monotonic()
                    log.info("consolidation: %s", await consolidate(session))
        except Exception:
            log.exception("memory worker iteration failed")
            n = 0
        if n == 0:
            try:
                await asyncio.wait_for(stop.wait(), idle_sleep)
            except asyncio.TimeoutError:
                pass


async def run_worker(stop: asyncio.Event, idle_sleep: float = 2.0) -> None:
    await asyncio.gather(*(_slot(stop, idle_sleep, consolidates=i == 0) for i in range(max(1, CONCURRENCY))))


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    asyncio.run(run_worker(asyncio.Event()))
