from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.models import ProviderSetting
from src.schemas import ProviderRead, ProviderUpdate, ProviderModels
from src.services.providers import get_provider, list_provider_names, PROVIDER_MODELS

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/providers", response_model=list[ProviderRead])
async def list_providers(session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(
        select(ProviderSetting).order_by(ProviderSetting.id)
    )).scalars().all()
    return [ProviderRead(
        provider=r.provider, api_key_set=r.api_key is not None and r.api_key != "",
        model=r.model, base_url=r.base_url, is_active=r.is_active,
    ) for r in rows]


@router.put("/providers/{provider_name}", response_model=ProviderRead)
async def update_provider(
    provider_name: str,
    body: ProviderUpdate,
    session: AsyncSession = Depends(get_session),
):
    row = (await session.execute(
        select(ProviderSetting).where(ProviderSetting.provider == provider_name)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, f"provider not found: {provider_name}")

    if body.api_key is not None:
        row.api_key = body.api_key
    if body.model is not None:
        row.model = body.model
    if body.base_url is not None:
        row.base_url = body.base_url
    if body.is_active is not None:
        row.is_active = body.is_active
    await session.commit()
    await session.refresh(row)

    return ProviderRead(
        provider=row.provider, api_key_set=row.api_key is not None and row.api_key != "",
        model=row.model, base_url=row.base_url, is_active=row.is_active,
    )


@router.get("/providers/{provider_name}/models", response_model=ProviderModels)
async def get_models(
    provider_name: str,
    session: AsyncSession = Depends(get_session),
):
    if provider_name not in list_provider_names():
        raise HTTPException(404, f"unknown provider: {provider_name}")

    if provider_name == "ollama":
        try:
            from src.services.providers.ollama_provider import OllamaProvider
            p = OllamaProvider()
            models = await p.list_models()
        except Exception:
            models = []
    else:
        models = PROVIDER_MODELS.get(provider_name, [])

    return ProviderModels(provider=provider_name, models=models)
