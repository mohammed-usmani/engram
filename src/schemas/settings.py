from __future__ import annotations
from pydantic import BaseModel, ConfigDict


class ProviderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    provider: str
    api_key_set: bool  # True if api_key is non-null, never expose actual key
    model: str | None
    base_url: str | None
    is_active: bool


class ProviderUpdate(BaseModel):
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    is_active: bool | None = None


class ProviderModels(BaseModel):
    provider: str
    models: list[str]
