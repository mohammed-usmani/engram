from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.schemas import MemoryRead
from src.services.memory import list_memories, delete_memory

router = APIRouter(prefix="/api/memories", tags=["memories"])


@router.get("", response_model=list[MemoryRead])
async def get_memories(category: str | None = None, session: AsyncSession = Depends(get_session)):
    mems = await list_memories(session, category=category)
    return [MemoryRead(
        id=m.id, content=m.content, category=m.category.value,
        source_session_id=m.source_session_id, created_at=m.created_at,
    ) for m in mems]


@router.delete("/{memory_id}", status_code=204)
async def remove_memory(memory_id: int, session: AsyncSession = Depends(get_session)):
    try:
        await delete_memory(memory_id, session)
        await session.commit()
    except Exception:
        raise HTTPException(404, "memory not found")
