"""Settings API: providers (keys, default models, test), which model does each job, recall and worker
options. Keys are write-only, and only a request from this computer can change them."""
import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import settings_store as store
from src.database import get_session
from src.memory import llm
from src.models import ProviderSetting
from src.privacy import is_remote
from src.schemas import ProviderUpdate, ProviderModels
from src.services.providers import list_provider_names, PROVIDER_MODELS

router = APIRouter(prefix="/api/settings", tags=["settings"])

LABELS = {"ollama": "Ollama", "dashscope": "DashScope (Alibaba)", "groq": "Groq", "cerebras": "Cerebras", "mistral": "Mistral", "together": "Together",
          "openai": "OpenAI", "gemini": "Gemini", "claude": "Claude"}


def _remote_guard(what: str) -> None:
    if is_remote():
        raise HTTPException(403, f"{what} can only be changed from the computer Engram runs on, "
                                 "not through the public link.")


def used_for(name: str) -> list[str]:
    out = []
    for task, label in store.TASKS.items():
        steps = llm.steps(task)
        if any(n == name for n, _ in steps):
            first = steps and steps[0][0] == name
            out.append(label if first or task == "chat" else f"{label} (fallback)")
    return out


def _provider_out(r: ProviderSetting) -> dict:
    src = llm.key_source(r.provider)
    return {"provider": r.provider, "label": LABELS.get(r.provider, r.provider),
            "api_key_set": bool(src), "key_source": src, "model": r.model, "effective_model": llm.default_model(r.provider),
            "base_url": r.base_url, "is_active": r.is_active, "used_for": used_for(r.provider)}


async def _rows(session: AsyncSession) -> list[ProviderSetting]:
    await store.ensure_providers(session)
    order = {n: i for i, n in enumerate(list_provider_names())}
    rows = list((await session.execute(select(ProviderSetting))).scalars())
    return sorted(rows, key=lambda r: order.get(r.provider, 99))


@router.get("/providers")
async def list_providers(session: AsyncSession = Depends(get_session)):
    return [_provider_out(r) for r in await _rows(session)]


@router.put("/providers/{provider_name}")
async def update_provider(provider_name: str, body: ProviderUpdate, session: AsyncSession = Depends(get_session)):
    if provider_name not in list_provider_names():
        raise HTTPException(404, f"unknown provider: {provider_name}")
    row = next(r for r in await _rows(session) if r.provider == provider_name)
    if body.api_key is not None:
        _remote_guard("API keys")
        row.api_key = body.api_key.strip() or None  # "" removes the saved key (.env, if any, applies again)
    if body.base_url is not None:
        _remote_guard("Provider URLs")
        row.base_url = body.base_url.strip() or None
    if body.model is not None:
        row.model = body.model.strip() or None
    if body.is_active is not None:
        row.is_active = body.is_active
    await session.commit()
    await store.apply(session)
    return _provider_out(row)


async def _models_for(name: str) -> list[str]:
    if name == "ollama":
        try:
            from src.services.providers.ollama_provider import OllamaProvider
            return await OllamaProvider().list_models()
        except Exception:
            return []
    return list(PROVIDER_MODELS.get(name, []))


@router.get("/providers/{provider_name}/models", response_model=ProviderModels)
async def get_models(provider_name: str):
    if provider_name not in list_provider_names():
        raise HTTPException(404, f"unknown provider: {provider_name}")
    models = await _models_for(provider_name)
    current = llm.default_model(provider_name)
    if current and current not in models:
        models.insert(0, current)
    return ProviderModels(provider=provider_name, models=models)


class TestIn(BaseModel):
    model: str | None = None


@router.post("/providers/{provider_name}/test")
async def test_provider(provider_name: str, body: TestIn | None = None):
    """One tiny real request ("reply OK"), so the user knows the key and model work."""
    if provider_name not in list_provider_names():
        raise HTTPException(404, f"unknown provider: {provider_name}")
    if not llm.key_source(provider_name):
        raise HTTPException(400, f"{LABELS[provider_name]} has no API key yet. Add one first.")
    p = llm.make(provider_name, (body.model if body else None) or None)
    t = time.monotonic()
    try:
        reply = await asyncio.wait_for(p.generate("Reply with the single word OK.", system="Be terse."),
                                       120 if provider_name == "ollama" else 45)  # a cold local model loads slowly
    except Exception as e:
        return {"ok": False, "model": getattr(p, "model", None), "error": (str(e) or type(e).__name__)[:300]}
    return {"ok": True, "model": getattr(p, "model", None), "ms": int((time.monotonic() - t) * 1000),
            "reply": (reply or "").strip()[:80]}


class Step(BaseModel):
    provider: str
    model: str | None = None


class ModelsIn(BaseModel):
    """Per task: a list of steps tried in order. Empty list = "same as extraction" (or the .env chain)."""
    chat: list[Step] | None = None
    extract: list[Step] | None = None
    consolidate: list[Step] | None = None
    plan: list[Step] | None = None
    reconcile: list[Step] | None = None


def _models_view() -> dict:
    saved = store.get("models")
    tasks = []
    for task, label in store.TASKS.items():
        eff = llm.steps(task)
        tasks.append({"task": task, "label": label, "saved": saved.get(task) or [],
                      "effective": [{"provider": n, "model": m or llm.default_model(n)} for n, m in eff],
                      "inherits": not saved.get(task) and task not in ("chat", "extract")})
    return {"tasks": tasks}


@router.get("/models")
async def get_assignments():
    return _models_view()


@router.put("/models")
async def set_assignments(body: ModelsIn, session: AsyncSession = Depends(get_session)):
    current = store.get("models")
    known = set(list_provider_names())
    for task in store.TASKS:
        steps = getattr(body, task)
        if steps is None:
            continue
        for s in steps:
            if s.provider not in known:
                raise HTTPException(422, f"{task}: unknown provider '{s.provider}'")
            if not llm.key_source(s.provider):
                raise HTTPException(422, f"{store.TASKS[task]}: {LABELS[s.provider]} has no API key. "
                                         "Add the key under Providers first.")
        if task == "chat" and len(steps) > 1:
            raise HTTPException(422, "Chat answers use one provider; pick a single model.")
        current[task] = [s.model_dump() for s in steps]
    await store.put(session, "models", current)
    return _models_view()


class RecallIn(BaseModel):
    self_entity: str | None = Field(None, description="Entity slug that is the user (never boosted or summarized)")
    pinned_cap: float | None = Field(None, ge=0.1, le=1.0, description="Max share of a brief for pinned sections")


@router.get("/recall")
async def get_recall():
    return store.get("recall")


@router.put("/recall")
async def set_recall(body: RecallIn, session: AsyncSession = Depends(get_session)):
    from src.memory.models import Entity
    if body.self_entity and not await session.scalar(select(Entity.id).where(Entity.slug == body.self_entity)):
        raise HTTPException(422, f"No topic '{body.self_entity}'.")
    return await store.put(session, "recall", {"self_entity": body.self_entity or None, "pinned_cap": body.pinned_cap})


class WorkerIn(BaseModel):
    consolidate_every_h: float = Field(..., ge=1, le=168)


@router.put("/worker")
async def set_worker(body: WorkerIn, session: AsyncSession = Depends(get_session)):
    return await store.put(session, "worker", body.model_dump())


class FlagsIn(BaseModel):
    recovery_key_saved: bool | None = None


@router.put("/flags")
async def set_flags(body: FlagsIn, session: AsyncSession = Depends(get_session)):
    flags = store.get("flags")
    flags.update({k: v for k, v in body.model_dump().items() if v is not None})
    return await store.put(session, "flags", flags)
