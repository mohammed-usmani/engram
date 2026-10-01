import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings as app_settings
from src.database import AsyncSessionLocal, engine, get_session
from src.models import ContextDocument, DocType, MemoryCategory
from src.routers import documents, admin, chat, memories, settings as settings_router
from src.seed.loader import run_seed
from src.services.memory import list_memories as _list_all_memories
from src.memory.api import AuthMiddleware, router as memory_router
from src.memory.worker import run_worker
from src.mcp_server import mcp


log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=app_settings.log_level)
    async with AsyncSessionLocal() as session:
        count = (await session.execute(select(func.count()).select_from(ContextDocument))).scalar() or 0
        if count == 0:
            log.info("empty DB — bootstrapping from %s", app_settings.data_dir)
            summary = await run_seed(session, app_settings.data_dir)
            await session.commit()
            log.info("seed summary: %s", summary)
    stop = asyncio.Event()
    worker = asyncio.create_task(run_worker(stop)) if os.environ.get("MEMORY_WORKER", "1") != "0" else None
    async with mcp.session_manager.run():
        yield
    stop.set()
    if worker:
        await worker
    await engine.dispose()


app = FastAPI(title="Engram", lifespan=lifespan)
app.add_middleware(AuthMiddleware)
app.include_router(memory_router)
app.include_router(admin.reseed_router)
app.include_router(admin.router)
app.include_router(chat.router)
app.include_router(memories.router)
app.include_router(documents.router)
app.include_router(settings_router.router)


_templates = Jinja2Templates(directory=str(Path(__file__).parent / "src" / "templates"))


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return _templates.TemplateResponse("chat.html", {"request": request, "active_page": "chat"})


@app.get("/memories", response_class=HTMLResponse)
async def memories_page(
    request: Request,
    category: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    mems = await _list_all_memories(session, category=category)
    return _templates.TemplateResponse("memories.html", {
        "request": request,
        "memories": mems,
        "categories": [c.value for c in MemoryCategory],
        "category": category,
        "active_page": "memories",
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return _templates.TemplateResponse("settings.html", {"request": request, "active_page": "settings"})


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/types")
async def types() -> list[str]:
    return [t.value for t in DocType]


# MCP (streamable HTTP) at /mcp — mounted last so every FastAPI route above wins.
app.mount("/", mcp.streamable_http_app())
