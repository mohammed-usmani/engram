"""Write path: queue → redact → extract → resolve → store → reconcile → mark dirty."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, text as sql
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

import httpx

from src.memory import USER_TZ, local_date
from src.memory import entities as ent
from src.memory import facts
from src.memory.extract import extract, redact
from src.memory.llm import LLMUnavailable, complete_json
from src.memory.models import Entity, Episode, IngestJob, SessionState
from src.services.ollama_client import embed

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
_EPISODE_DUP_SIM = 0.9      # same kind, ±1 day, this similar → same event, skip
_RECONCILE_MIN_SIM = 0.8    # cosine; the LLM makes the real call


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def enqueue(session: AsyncSession, text: str, agent: str | None = None, session_id: str | None = None,
                  occurred_at: datetime | None = None) -> tuple[int, bool]:
    """Queue text for extraction. Returns (job_id, created); retries of the same payload on the same day are no-ops."""
    when = occurred_at or _now()
    if when.tzinfo is None:  # agents send local wall-clock times
        when = when.replace(tzinfo=USER_TZ)
    text = redact(text)  # secrets never reach the DB, not even the queue
    digest = hashlib.sha256(f"{agent}|{session_id}|{local_date(when)}|{text}".encode()).hexdigest()
    row = (await session.execute(
        insert(IngestJob).values(content_hash=digest, text=text, agent=agent, session_id=session_id,
                                 occurred_at=when, result={})
        .on_conflict_do_nothing(index_elements=["content_hash"]).returning(IngestJob.id)
    )).scalar()
    await session.commit()
    if row is not None:
        return row, True
    return (await session.execute(select(IngestJob.id).where(IngestJob.content_hash == digest))).scalar_one(), False


async def note(session: AsyncSession, session_id: str, key: str, value: str, ttl_days: int = 7) -> None:
    value = redact(value)
    stmt = insert(SessionState).values(session_id=session_id, key=key, value=value,
                                       expires_at=_now() + timedelta(days=ttl_days))
    await session.execute(stmt.on_conflict_do_update(
        index_elements=["session_id", "key"], set_={"value": value, "expires_at": stmt.excluded.expires_at}))
    await session.commit()


async def teach(session: AsyncSession, name: str, steps: list[str], agent: str | None = None,
                entity_names: list[str] | None = None) -> str:
    mapping = await ent.resolve(session, [{"name": n, "kind": "topic"} for n in entity_names or []])
    await session.commit()
    body = redact(f"How to {name}: " + " → ".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
    return await facts.add_fact(body, kind="procedure", entities=list(mapping.values()), importance=4, agent=agent)


async def _is_duplicate_episode(session: AsyncSession, kind: str, when: datetime, vec: list[float],
                                entities: list[str]) -> bool:
    """Same event re-mentioned: same kind, ±1 day, same entities, near-identical wording."""
    rows = (await session.execute(
        select(Episode.entities, 1 - Episode.embedding.cosine_distance(vec))
        .where(Episode.kind == kind, Episode.embedding.isnot(None),
               Episode.occurred_at.between(when - timedelta(days=1), when + timedelta(days=1)))
        .order_by(Episode.embedding.cosine_distance(vec)).limit(5)
    )).all()
    # entities matter: "applied to Zomato" and "applied to Swiggy" read alike but are two applications
    return any(float(sim) >= _EPISODE_DUP_SIM and set(ents) == set(entities) for ents, sim in rows)


_RECONCILE_PROMPT = """For each numbered pair decide how the NEW statement about the user relates to the OLD one:
- "duplicate": same meaning, nothing new
- "supersedes": NEW replaces OLD (changed preference, updated fact, OLD no longer true)
- "compatible": both can be true
Return JSON: [{{"pair": <number>, "verdict": "duplicate|supersedes|compatible"}}]

{pairs}"""


async def _reconcile(new_facts: list[tuple[str, str]]) -> dict:
    """new_facts: (id, text). Supersede/dedupe against similar existing facts. Best-effort."""
    pairs = []
    for new_id, text in new_facts:
        for hit in await facts.similar(text, top_k=5):
            if hit["id"] != new_id and hit["id"] not in {n for n, _ in new_facts} and (hit["score"] or 0) >= _RECONCILE_MIN_SIM:
                pairs.append((new_id, hit["id"], text, hit["memory"]))
    if not pairs:
        return {"superseded": 0, "duplicates": 0}
    listing = "\n".join(f"{i}. OLD: {old_t}\n   NEW: {new_t}" for i, (_, _, new_t, old_t) in enumerate(pairs))
    try:
        verdicts = await complete_json(_RECONCILE_PROMPT.format(pairs=listing), task="reconcile")
    except Exception as e:
        log.warning("reconcile skipped: %s", e)
        return {"superseded": 0, "duplicates": 0}
    sup = dup = 0
    dropped: set[str] = set()
    for v in verdicts if isinstance(verdicts, list) else []:
        i = v.get("pair") if isinstance(v, dict) else None
        if not isinstance(i, int) or not 0 <= i < len(pairs):
            continue
        new_id, old_id, _, _ = pairs[i]
        if new_id in dropped:
            continue
        if v.get("verdict") == "duplicate":
            await facts.delete_fact(new_id)
            dropped.add(new_id)
            dup += 1
        elif v.get("verdict") == "supersedes":
            await facts.supersede(old_id, new_id)
            sup += 1
    return {"superseded": sup, "duplicates": dup}


async def _known_entities(session: AsyncSession) -> list[str]:
    return list((await session.execute(select(Entity.name).order_by(Entity.updated_at.desc()).limit(60))).scalars())


async def process_job(session: AsyncSession, job: IngestJob) -> dict:
    """Extract one job and write what it found. Several workers run this at once (one per
    MEMORY_WORKER_CONCURRENCY slot): the LLM call overlaps, the writes take turns under _WRITE_LOCK
    so duplicate checks and fact reconciliation never race each other."""
    job_id = job.id
    job.status, job.attempts = "processing", job.attempts + 1
    await session.commit()
    try:
        async with _WRITE_LOCK:
            await _undo_partial(session, job)
            known = await _known_entities(session)
        ex = await extract(redact(job.text), job.occurred_at, known)

        async with _WRITE_LOCK:
            names = {e["name"]: e["kind"] for e in ex.entities}
            for item in [*ex.episodes, *ex.facts, *ex.procedures]:
                for n in item["entities"]:
                    names.setdefault(n, "topic")
            mapping = await ent.resolve(session, [{"name": n, "kind": k} for n, k in names.items()])
            slugs = lambda ns: sorted({mapping[n] for n in ns if n in mapping})  # noqa: E731

            n_eps = 0
            for e in ex.episodes:
                vec = await embed(e["summary"])
                if await _is_duplicate_episode(session, e["kind"], e["occurred_at"], vec, slugs(e["entities"])):
                    continue
                session.add(Episode(
                    occurred_at=e["occurred_at"], kind=e["kind"], summary=e["summary"], outcome=e["outcome"],
                    sentiment=e["sentiment"], importance=e["importance"], entities=slugs(e["entities"]),
                    source_agent=job.agent, session_id=job.session_id, embedding=vec, payload={"job": job_id},
                ))
                n_eps += 1

            for n in ex.session_notes:
                if job.session_id:
                    await note(session, job.session_id, n["key"], n["value"])

            touched = set(mapping.values())
            if touched:
                await session.execute(sql("UPDATE entities SET dirty = true WHERE slug = ANY(:s)"), {"s": list(touched)})
            await session.commit()

            # mem0 writes last: they can't be rolled back with the DB transaction.
            # Each fact id is recorded as it is written so a retry can undo a partial run.
            new_facts = []
            pending = [(f["text"], dict(kind=f["kind"], entities=slugs(f["entities"]), importance=f["importance"]))
                       for f in ex.facts]
            pending += [(f"How to {p['name']}: " + " → ".join(f"{i}. {s}" for i, s in enumerate(p["steps"], 1)),
                         dict(kind="procedure", entities=slugs(p["entities"]), importance=4)) for p in ex.procedures]
            for body, kw in pending:
                fid = await facts.add_fact(body, agent=job.agent, **kw)
                new_facts.append((fid, body))
                job.result = {**job.result, "fact_ids": [f for f, _ in new_facts]}
                await session.commit()
            rec = await _reconcile(new_facts)

            job.status, job.error = "done", None
            job.result = {"episodes": n_eps, "facts": len(new_facts), "entities": sorted(touched),
                          "notes": len(ex.session_notes), "fact_ids": [f for f, _ in new_facts], **rec}
            await session.commit()
            return job.result
    except Exception as e:
        await session.rollback()
        job = await session.get(IngestJob, job_id)
        await session.refresh(job)
        job.error = str(e)[:2000]
        if isinstance(e, _OUTAGE):  # provider/Ollama unreachable: not the payload's fault, don't burn attempts
            job.attempts -= 1
            job.result = {**job.result, "_outages": int(job.result.get("_outages", 0)) + 1}
            job.status = "pending"
        else:
            job.status = "failed" if job.attempts >= MAX_ATTEMPTS else "pending"
        await session.commit()
        log.warning("ingest job %s attempt %s failed: %s", job.id, job.attempts, e)
        return {"error": job.error}


_OUTAGE = (LLMUnavailable, ConnectionError, httpx.ConnectError, httpx.TimeoutException)
STUCK_AFTER = "10 minutes"


async def _undo_partial(session: AsyncSession, job: IngestJob) -> None:
    """A retry starts clean: drop episodes and facts an earlier, interrupted run of this job wrote."""
    await session.execute(sql("DELETE FROM episodes WHERE payload->>'job' = :j"), {"j": str(job.id)})
    for fid in (job.result or {}).get("fact_ids", []):
        await facts.delete_fact(fid)
    if job.result and job.result.get("fact_ids"):
        job.result = {k: v for k, v in job.result.items() if k != "fact_ids"}
    await session.commit()


async def retry_job(session: AsyncSession, job_id: int) -> bool:
    job = await session.get(IngestJob, job_id)
    if job is None:
        return False
    job.status, job.attempts, job.error = "pending", 0, None
    await session.commit()
    return True


CONCURRENCY = int(os.environ.get("MEMORY_WORKER_CONCURRENCY", "4"))  # parallel worker slots
_WRITE_LOCK = asyncio.Lock()


async def process_pending(session: AsyncSession, limit: int = 1) -> int:
    """Process up to `limit` due jobs. Retries back off exponentially (2^n − 1 min, capped at 1h)."""
    # A crash/restart mid-job leaves it 'processing' forever unless we take it back.
    await session.execute(sql(f"UPDATE ingest_jobs SET status = 'pending' "
                              f"WHERE status = 'processing' AND updated_at < now() - interval '{STUCK_AFTER}'"))
    await session.commit()
    ids = (await session.execute(
        select(IngestJob.id)
        .where(IngestJob.status == "pending",
               IngestJob.updated_at <= sql(
                   "now() - make_interval(mins => least(60, power(2, ingest_jobs.attempts + "
                   "coalesce((ingest_jobs.result->>'_outages')::int, 0))::int - 1))"))
        .order_by(IngestJob.id).limit(limit).with_for_update(skip_locked=True)
    )).scalars().all()
    for job_id in ids:
        # Load each job fresh: a failed job rolls the session back, which expires anything loaded
        # earlier, and touching an expired job raised MissingGreenlet for the rest of the batch.
        await process_job(session, await session.get(IngestJob, job_id))
    return len(ids)
