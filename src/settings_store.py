"""Settings changed from the admin pages, kept in memory_blocks as `setting:<name>` JSON.

Every value is cached in-process so hot paths (each LLM call, each recall) never touch the DB;
`apply()` pushes the LLM-related ones into src/memory/llm. The API process is the only writer,
so the cache is reloaded on every save and at startup.
"""
from __future__ import annotations

import json
import os
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.memory import llm
from src.memory.models import MemoryBlock
from src.models import ProviderSetting
from src.services.providers import list_provider_names

TASKS = {
    "chat": "Chat answers",
    "extract": "Memory extraction",
    "consolidate": "Topic summaries and lessons",
    "plan": "Recall planner",
    "reconcile": "Fact checks",
}
DEFAULTS: dict[str, Any] = {
    "models": {},                                           # task -> [{"provider", "model"}]
    "recall": {"self_entity": None, "pinned_cap": 0.5},     # pinned_cap: share of the budget, or None
    "worker": {"consolidate_every_h": 6},
    "flags": {"recovery_key_saved": False},
    "cleanup_dismissed": [],                                 # suggestion keys the user said "not the same" to
    "kind_map": {},                                          # extractor topic kind -> one of entities.CANON_KINDS
    "rejected_lessons": [],
    "health_verdicts": {},
    "eval_nightly": None,                                    # date the nightly evaluation last ran                                   # "factA|factB" -> LLM verdict on whether they contradict                                  # lesson texts marked wrong; consolidate skips look-alikes
}
_cache: dict[str, Any] = {}


def get(name: str) -> Any:
    v = _cache.get(name, DEFAULTS.get(name))
    return json.loads(json.dumps(v))  # callers may mutate their copy


async def load(session: AsyncSession) -> None:
    rows = (await session.execute(select(MemoryBlock).where(MemoryBlock.name.like("setting:%")))).scalars()
    _cache.clear()
    for r in rows:
        try:
            _cache[r.name.removeprefix("setting:")] = json.loads(r.content)
        except ValueError:
            pass  # e.g. setting:batch_mode is plain text, owned by src/memory/batch.py
    await ensure_providers(session)
    await apply(session)


async def put(session: AsyncSession, name: str, value: Any) -> Any:
    row = await session.get(MemoryBlock, f"setting:{name}")
    text = json.dumps(value)
    if row is None:
        session.add(MemoryBlock(name=f"setting:{name}", content=text))
    else:
        row.content = text
    await session.commit()
    _cache[name] = value
    await apply(session)
    return value


async def ensure_providers(session: AsyncSession) -> None:
    """One provider_settings row per provider the app knows (Together and Mistral were missing)."""
    have = set((await session.execute(select(ProviderSetting.provider))).scalars())
    for name in list_provider_names():
        if name not in have:
            session.add(ProviderSetting(provider=name, is_active=False))
    await session.commit()


async def apply(session: AsyncSession) -> None:
    rows = list((await session.execute(select(ProviderSetting))).scalars())
    assign = {task: [(s["provider"], s.get("model") or None) for s in steps]
              for task, steps in (get("models") or {}).items()}
    llm.configure({r.provider: r.api_key for r in rows}, assign,
                  {r.provider: r.model for r in rows}, {r.provider: r.base_url for r in rows})
    os.environ["CONSOLIDATE_EVERY_H"] = str(get("worker").get("consolidate_every_h", 6))
