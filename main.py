import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
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
from src.ui import pages as ui_pages, system_api, core_api, cleanup as ui_cleanup, memory_pages as ui_memory, ask as ui_ask
from src.ui.templating import templates
from fastapi.staticfiles import StaticFiles
from src import settings_store
from src.memory.worker import run_worker
from src.mcp_server import mcp
from src import oauth


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
        await settings_store.load(session)  # Settings-page choices (keys, models per job) before any work runs
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
app.include_router(ui_pages.router)
app.include_router(system_api.router)
app.include_router(core_api.router)
app.include_router(ui_cleanup.router)
app.include_router(ui_memory.router)
app.include_router(ui_ask.router)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "src" / "static")), name="static")
app.include_router(admin.reseed_router)
app.include_router(admin.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(memories.router)
app.include_router(settings_router.router)


@app.get("/admin", include_in_schema=False)
async def admin_shortcut(request: Request):
    q = request.url.query
    return RedirectResponse("/api/admin" + (f"?{q}" if q else ""))


def _safe_next(nxt: str) -> str:
    return nxt if nxt.startswith("/") and not nxt.startswith("//") else "/admin"


@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request, next: str = "/admin"):
    return templates.TemplateResponse(request, "login.html", {"next": _safe_next(next), "error": None})


@app.post("/login", include_in_schema=False)
async def login(request: Request, admin_token: str = Form(...), next: str = Form("/admin")):
    if not app_settings.admin_token or not hmac.compare_digest(admin_token.strip(), app_settings.admin_token):
        return templates.TemplateResponse(request, "login.html", {"next": _safe_next(next), "error": "Wrong token."},
                                          status_code=401)
    resp = RedirectResponse(_safe_next(next), status_code=303)
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp.set_cookie("admin_token", app_settings.admin_token, max_age=30 * 24 * 3600,
                    httponly=True, secure=secure, samesite="lax")
    return resp


@app.get("/logout", include_in_schema=False)
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("admin_token")
    return resp


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/types")
async def types(session: AsyncSession = Depends(get_session)) -> list[dict]:
    from src.doc_types import list_types
    return await list_types(session)


@app.get("/api/upcoming")
async def upcoming(days: int = 60, session: AsyncSession = Depends(get_session)) -> dict:
    from src.dates import DATE_RULE, upcoming as _upcoming
    if not 1 <= days <= 366:
        raise HTTPException(422, "days must be from 1 to 366")
    items, unreadable = await _upcoming(session, days)
    return {"items": items, "unreadable_dates": unreadable, "how_it_works": DATE_RULE}


# OAuth for MCP clients without a fixed header (only when PUBLIC_URL is set); before the mount.
app.router.routes.extend(oauth.routes)

# MCP (streamable HTTP) at /mcp — mounted last so every FastAPI route above wins.
app.mount("/", mcp.streamable_http_app())
