from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict


class ChatRequest(BaseModel):
    message: str
    session_id: int | None = None
    provider: str | None = None
    model: str | None = None


class SourceInfo(BaseModel):
    slug: str
    title: str
    type: str
    score: float


class ChatResponse(BaseModel):
    response: str
    session_id: int
    sources: list[SourceInfo]
    memories_extracted: int


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    role: str
    content: str
    created_at: datetime


class SessionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str | None
    message_count: int
    updated_at: datetime


class SessionDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str | None
    summary: str | None
    messages: list[MessageRead]
