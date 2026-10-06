"""Batch mode: send the whole extraction queue to Together's Batch API instead of live calls.

For a big backlog (an import, a day of long sessions) where waiting a few minutes to hours is
fine: batched requests cost about half, run in their own rate-limit pool, and a stuck request
can't hold up the queue. Switched on and off from the Memory page's Queue tab; the state lives
in memory_blocks under BATCH_SETTING.

While on, live workers stand aside (ingest.process_pending) and tick() — run by the worker about
once a minute — sends every pending job as one batch. Batches already sent are always collected,
even after switching off, and their results go through ingest.process_job like a live call, so
writes, duplicate checks and reconciliation are identical. Jobs a batch couldn't do go back to
the queue (and are counted as an attempt).

Together only accepts serverless models in batches; DeepSeek-V4-Flash is refused, so batches
use MEMORY_BATCH_MODEL (default gpt-oss-120b, 126/126 on scripts/eval_extraction.py).
"""
from __future__ import annotations

import json
import logging
import os

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.memory import ingest
from src.memory.extract import SYSTEM, extract_prompt, parse_extraction, redact
from src.memory.llm import parse_json
from src.memory.models import IngestJob, MemoryBlock

log = logging.getLogger(__name__)

BATCH_SETTING = "setting:batch_mode"
API = "https://api.together.xyz/v1"
DONE = {"COMPLETED", "FAILED", "EXPIRED", "CANCELLED"}
MAX_PER_BATCH = 2000


def model() -> str:
    return os.environ.get("MEMORY_BATCH_MODEL", "openai/gpt-oss-120b")  # 126/126 on eval_extraction


def available() -> bool:
    return bool(os.environ.get("TOGETHER_API_KEY"))


async def is_on(session: AsyncSession) -> bool:
    block = await session.get(MemoryBlock, BATCH_SETTING)
    return bool(block and block.content == "on")


async def set_mode(session: AsyncSession, on: bool) -> None:
    block = await session.get(MemoryBlock, BATCH_SETTING)
    if block is None:
        session.add(MemoryBlock(name=BATCH_SETTING, content="on" if on else "off"))
    else:
        block.content = "on" if on else "off"
    await session.commit()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=API, timeout=120,
                             headers={"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"})


async def submit(session: AsyncSession) -> str | None:
    """Send every pending job as one batch. Returns the batch id, or None if nothing was pending."""
    jobs = (await session.execute(
        select(IngestJob).where(IngestJob.status == "pending").order_by(IngestJob.id).limit(MAX_PER_BATCH)
    )).scalars().all()
    if not jobs:
        return None
    known = await ingest._known_entities(session)
    lines = "".join(json.dumps({"custom_id": str(j.id), "body": {
        "model": model(), "max_tokens": 8192,
        "messages": [{"role": "system", "content": SYSTEM + "\nRespond with valid JSON only."},
                     {"role": "user", "content": extract_prompt(redact(j.text), j.occurred_at, known)}],
    }}) + "\n" for j in jobs)
    async with _client() as c:
        up = await c.post("/files/upload", data={"purpose": "batch-api", "file_name": "engram-batch.jsonl",
                                                 "file_type": "jsonl"},
                          files={"file": ("engram-batch.jsonl", lines.encode(), "application/jsonl")})
        up.raise_for_status()
        made = await c.post("/batches", json={"input_file_id": up.json()["id"], "endpoint": "/v1/chat/completions",
                                              "completion_window": "24h"})
        made.raise_for_status()
    batch_id = (made.json().get("job") or made.json())["id"]
    for j in jobs:
        j.status, j.error = "batched", None
        j.result = {**(j.result or {}), "batch_id": batch_id}
    await session.commit()
    log.info("batch %s: sent %d jobs to %s", batch_id, len(jobs), model())
    return batch_id


def _content(line: dict) -> str | None:
    body = (line.get("response") or {}).get("body") or {}
    choices = body.get("choices") or []
    return choices[0]["message"]["content"] if choices else None


async def _requeue(session: AsyncSession, job: IngestJob, error: str) -> None:
    job.attempts += 1
    job.status = "failed" if job.attempts >= ingest.MAX_ATTEMPTS else "pending"
    job.error = f"batch: {error}"[:2000]
    await session.commit()


async def collect(session: AsyncSession) -> int:
    """Save the results of every finished batch. Returns how many jobs were saved."""
    jobs = (await session.execute(select(IngestJob).where(IngestJob.status == "batched"))).scalars().all()
    by_batch: dict[str, list[int]] = {}
    for j in jobs:
        by_batch.setdefault(j.result.get("batch_id", ""), []).append(j.id)
    saved = 0
    async with _client() as c:
        for batch_id, ids in by_batch.items():
            if not batch_id:
                continue
            b = (await c.get(f"/batches/{batch_id}")).json()
            b = b.get("job", b)
            if b.get("status") not in DONE:
                continue
            results: dict[str, dict] = {}
            for key in ("output_file_id", "error_file_id"):
                if b.get(key):
                    text = (await c.get(f"/files/{b[key]}/content")).text
                    for raw in text.splitlines():
                        if raw.strip():
                            line = json.loads(raw)
                            results[str(line.get("custom_id"))] = line
            for job_id in ids:
                job = await session.get(IngestJob, job_id)
                if job is None or job.status != "batched":  # retried by hand meanwhile
                    continue
                line = results.get(str(job_id))
                content = _content(line) if line else None
                if content is None:
                    err = (line or {}).get("error") or f"batch {b.get('status', '').lower()} without a result"
                    await _requeue(session, job, str(err.get("message") if isinstance(err, dict) else err))
                    continue
                try:
                    ex = parse_extraction(parse_json(content), job.occurred_at, redact(job.text))
                except Exception as e:
                    await _requeue(session, job, f"invalid JSON: {e}")
                    continue
                await ingest.process_job(session, job, extraction=ex)
                saved += 1
            log.info("batch %s %s: saved %d of %d", batch_id, b.get("status"), saved, len(ids))
    return saved


async def tick(session: AsyncSession) -> None:
    """Collect finished batches; in batch mode, send whatever is pending."""
    if not available():
        return
    await collect(session)
    if await is_on(session):
        await submit(session)


async def counts(session: AsyncSession) -> dict:
    from sqlalchemy import func
    rows = (await session.execute(select(IngestJob.status, func.count()).group_by(IngestJob.status))).all()
    return dict(rows)
