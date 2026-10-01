"""Semantic / preference / procedural memory, stored in mem0 OSS on our pgvector.

We always write with infer=False: our own extractor has already structured the
data, so mem0 makes no LLM call. Supersession uses mem0's native
expiration_date (hidden from search, still returned by get) plus a
`superseded_by` pointer so history survives.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import AsyncMemory  # noqa: E402

from src.config import settings  # noqa: E402

USER = "me"
KINDS = ("fact", "preference", "procedure")
_mem: AsyncMemory | None = None


def _config() -> dict:
    conn = os.environ.get("MEM0_CONNECTION") or settings.database_url.replace("+asyncpg", "")
    base = settings.ollama_base_url
    return {
        "vector_store": {"provider": "pgvector", "config": {
            "connection_string": conn,
            "collection_name": os.environ.get("MEM0_COLLECTION", "mem0_memories"),
            "embedding_model_dims": 768,
        }},
        "embedder": {"provider": "ollama", "config": {"model": settings.ollama_embed_model, "ollama_base_url": base}},
        # Never called (infer=False everywhere) but mem0 requires one configured.
        "llm": {"provider": "ollama", "config": {"model": os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5-coder:7b"),
                                                 "ollama_base_url": base}},
        "history_db_path": str(Path(os.environ.get("MEM0_HISTORY_DB", Path.home() / ".mem0" / "history.db"))),
    }


async def store() -> AsyncMemory:
    global _mem
    if _mem is None:
        _mem = AsyncMemory.from_config(_config())
    return _mem


def _reset() -> None:
    global _mem
    _mem = None


def _filters(kinds: list[str] | None) -> dict:
    f: dict = {"user_id": USER}
    if kinds:
        f["kind"] = kinds[0] if len(kinds) == 1 else {"in": list(kinds)}
    return f


def _shape(r: dict) -> dict:
    return {"id": r["id"], "memory": r["memory"], "score": r.get("score"),
            "metadata": r.get("metadata") or {}, "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at")}


async def add_fact(text: str, kind: str, entities: list[str], importance: int = 3,
                   scope: str = "global", agent: str | None = None) -> str:
    if kind not in KINDS:
        kind = "fact"
    m = await store()
    res = await m.add(text, user_id=USER, infer=False, metadata={
        "kind": kind, "entities": entities, "importance": importance,
        "scope": scope, "agent": agent, "access_count": 0,
    })
    return res["results"][0]["id"]


async def search_facts(query: str, kinds: list[str] | None = None, top_k: int = 10) -> list[dict]:
    m = await store()
    res = await m.search(query, filters=_filters(kinds), top_k=top_k)
    return [_shape(r) for r in res["results"]]


async def all_facts(kinds: list[str] | None = None, limit: int = 500) -> list[dict]:
    m = await store()
    res = await m.get_all(filters=_filters(kinds), top_k=limit)
    return [_shape(r) for r in res["results"]]


async def get_fact(fact_id: str) -> dict | None:
    m = await store()
    r = await m.get(fact_id)
    return _shape(r) if r else None


async def supersede(old_id: str, new_id: str) -> None:
    m = await store()
    # expiration_date is only the "hide from search" switch; mem0 compares it to the
    # UTC date, so use a fixed past date and keep the real time in valid_to.
    await m.update(old_id, metadata={"superseded_by": new_id, "valid_to": datetime.now(timezone.utc).isoformat()},
                   expiration_date="2000-01-01")


async def touch(fact_id: str, access_count: int) -> None:
    m = await store()
    await m.update(fact_id, metadata={"access_count": access_count + 1})


async def delete_fact(fact_id: str) -> None:
    m = await store()
    try:
        await m.delete(fact_id)
    except ValueError:
        pass  # already gone


async def wipe() -> None:
    m = await store()
    await m.delete_all(user_id=USER)


async def similar(text: str, top_k: int = 5) -> list[dict]:
    """Pure cosine neighbours (mem0's search() score is a blend and not comparable across queries)."""
    import asyncio
    m = await store()
    vec = await asyncio.to_thread(m.embedding_model.embed, text, "search")
    rows = await asyncio.to_thread(m.vector_store.search, query=text, vectors=vec, top_k=top_k * 2,
                                   filters={"user_id": USER})
    out = [{"id": r.id, "memory": r.payload.get("data", ""), "score": r.score, "metadata": r.payload}
           for r in rows if not r.payload.get("superseded_by")]
    return out[:top_k]
