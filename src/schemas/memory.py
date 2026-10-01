from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class MemoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    content: str
    category: str
    source_session_id: int | None
    created_at: datetime
