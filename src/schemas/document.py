from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.models import DocType, Source


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    slug: str
    title: str
    type: DocType
    tags: list[str]
    source: Source


class DocumentRead(DocumentSummary):
    content: str
    sections: dict[str, str]
    doc_metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class DocumentCreate(BaseModel):
    type: DocType
    slug: str | None = None
    title: str
    tags: list[str] = []
    content: str
    sections: dict[str, str] = {}
    metadata: dict[str, Any] = {}


class DocumentUpdate(BaseModel):
    title: str | None = None
    tags: list[str] | None = None
    content: str | None = None
    sections: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    type: DocType | None = None
