"""REST surface for assistants without MCP (and the OpenAPI spec for Gemini Gems / ChatGPT Actions)."""
from __future__ import annotations

import hmac
from datetime import datetime
from http.cookies import SimpleCookie
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import PlainTextResponse, RedirectResponse

from src.config import settings
from src.privacy import REQUEST_IS_REMOTE
from src.database import get_session
from src.memory import ingest, recall as rc
from src.memory.models import IngestJob
from src.memory.worker import request_consolidation
from src import oauth

router = APIRouter(prefix="/api/memory", tags=["memory"])


def bearer_ok(authorization: str | None) -> bool:
    return not settings.admin_token or authorization == f"Bearer {settings.admin_token}"


_FORWARDED = (b"x-forwarded-for", b"forwarded", b"x-real-ip", b"cf-connecting-ip", b"true-client-ip",
              b"x-forwarded-host")
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")
_OPEN_PATHS = ("/api/health", "/login", *oauth.OPEN_PATHS)


def is_remote_request(headers: dict, client) -> bool:
    """Anything except direct local use: loopback client, localhost Host, no proxy headers."""
    host = headers.get(b"host", b"").decode().rsplit(":", 1)[0] if b"host" in headers else ""
    local = bool(client) and client[0] in ("127.0.0.1", "::1") and host in _LOCAL_HOSTS
    return not local or any(h in headers for h in _FORWARDED)


def needs_token(path: str, headers: dict, client) -> bool:
    """Everything needs the token, except /api/health and direct local use.

    Direct local use = loopback client, localhost Host header (blocks DNS rebinding), and no proxy
    headers — tunnels (cloudflared, ngrok) connect from loopback too but always add forwarding headers.
    """
    if path in _OPEN_PATHS or path.startswith("/.well-known/"):
        return False
    return is_remote_request(headers, client)


def cookie_token(headers: dict) -> str | None:
    """The admin token a browser keeps after signing in on /login."""
    jar = SimpleCookie(headers.get(b"cookie", b"").decode())
    return jar["admin_token"].value if "admin_token" in jar else None


async def _authorized(authorization: str | None, cookie: str | None = None) -> bool:
    """The admin token (header or sign-in cookie), or an access token issued to an OAuth client."""
    if bearer_ok(authorization) or (cookie and hmac.compare_digest(cookie, settings.admin_token or "")):
        return True
    token = (authorization or "").removeprefix("Bearer ").strip()
    return bool(token and oauth.provider and await oauth.provider.load_access_token(token))


class AuthMiddleware:
    """When ADMIN_TOKEN is set, every non-local request needs `Authorization: Bearer <token>`."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # every request knows whether it came from this computer or through the public link / a proxy
            REQUEST_IS_REMOTE.set(is_remote_request(dict(scope.get("headers") or []), scope.get("client")))
        if scope["type"] == "http" and settings.admin_token:
            headers = dict(scope.get("headers") or [])
            if needs_token(scope["path"], headers, scope.get("client")) and \
                    not await _authorized(headers.get(b"authorization", b"").decode() or None,
                                          cookie_token(headers)):
                if scope["method"] == "GET" and b"text/html" in headers.get(b"accept", b""):
                    # A person in a browser: send them to sign in, then back here.
                    nxt = scope["path"] + ("?" + scope["query_string"].decode() if scope["query_string"] else "")
                    return await RedirectResponse(f"/login?{urlencode({'next': nxt})}", 303)(scope, receive, send)
                # Tells OAuth clients where to discover how to sign in.
                meta = oauth.resource_metadata_url()
                extra = {"WWW-Authenticate": f'Bearer resource_metadata="{meta}"'} if meta else None
                return await PlainTextResponse("Unauthorized\n", status_code=401, headers=extra)(scope, receive, send)
        await self.app(scope, receive, send)


class RememberIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=200_000,
                      description="What happened / what you learned about the user, in plain language")
    agent: str = Field(..., min_length=1, max_length=64,
                       description="Who is writing: claude-code, claude-web, chatgpt, gemini, codex, antigravity...")
    session_id: str | None = None
    occurred_at: datetime = Field(..., description="When it happened (not when it's being saved), ISO 8601")


class IngestIn(RememberIn):
    session_end: bool = False


class NoteIn(BaseModel):
    session_id: str
    key: str = Field(..., max_length=255)
    value: str
    ttl_days: int = Field(7, ge=1, le=90)


class TeachIn(BaseModel):
    name: str = Field(..., description="Task name, e.g. 'deploy cityfix'")
    steps: list[str] = Field(..., min_length=1)
    agent: str = Field(..., min_length=1, max_length=64, description="Who is writing, e.g. claude-code, chatgpt")
    entities: list[str] = Field(default_factory=list, description="Related topics/projects by name")


class RecallIn(BaseModel):
    situation: str = Field(..., min_length=1, description="What the user is doing or asking, in their words")
    budget_tokens: int = Field(1500, ge=100, le=8000)
    session_id: str | None = None
    fast: bool = Field(False, description="Skip the LLM planner (no network LLM call)")
    agent: str | None = Field(None, max_length=64, description="Who is asking, e.g. claude-code (shown in traces)")


@router.post("/remember", status_code=202)
async def remember(body: RememberIn, session: AsyncSession = Depends(get_session)):
    job_id, created = await ingest.enqueue(session, body.text, body.agent, body.session_id, body.occurred_at)
    return {"job_id": job_id, "queued": created}


class BatchModeIn(BaseModel):
    on: bool


@router.post("/batch-mode")
async def batch_mode(body: BatchModeIn, session: AsyncSession = Depends(get_session)):
    """Send the extraction queue to the provider's batch API (cheaper, slower) instead of live calls."""
    from src.memory import batch
    if body.on and not batch.available():
        raise HTTPException(400, "batch mode needs TOGETHER_API_KEY")
    await batch.set_mode(session, body.on)
    return {"on": body.on, "model": batch.model()}


@router.post("/ingest", status_code=202)
async def ingest_hook(body: IngestIn, session: AsyncSession = Depends(get_session)):
    """For session-end hooks: queue the transcript and consolidate once the queue drains."""
    job_id, created = await ingest.enqueue(session, body.text, body.agent, body.session_id, body.occurred_at)
    if body.session_end:
        request_consolidation()
    return {"job_id": job_id, "queued": created}


@router.post("/note")
async def note(body: NoteIn, session: AsyncSession = Depends(get_session)):
    await ingest.note(session, body.session_id, body.key, body.value, body.ttl_days)
    return {"ok": True}


@router.post("/teach")
async def teach(body: TeachIn, session: AsyncSession = Depends(get_session)):
    return {"id": await ingest.teach(session, body.name, body.steps, body.agent, body.entities)}


@router.post("/recall")
async def recall(body: RecallIn, session: AsyncSession = Depends(get_session)):
    who = (body.agent or "").strip().lower()[:24]
    return await rc.recall(session, body.situation, body.budget_tokens, body.session_id, body.fast,
                           source=f"rest:{who}" if who else "rest")


@router.get("/item/{item_id}")
async def expand(item_id: str, session: AsyncSession = Depends(get_session)):
    item = await rc.expand(session, item_id)
    if item is None:
        raise HTTPException(404, "not found")
    return item


@router.delete("/item/{item_id}")
async def forget(item_id: str, session: AsyncSession = Depends(get_session)):
    return {"deleted": await rc.forget(session, item_id)}


@router.get("/timeline/{entity}")
async def timeline(entity: str, since: datetime | None = None, limit: int = 100,
                   session: AsyncSession = Depends(get_session)):
    return await rc.timeline(session, entity, since, limit)


@router.post("/consolidate", status_code=202)
async def consolidate_soon():
    request_consolidation()
    return {"requested": True}


@router.post("/jobs/{job_id}/retry")
async def retry(job_id: int, session: AsyncSession = Depends(get_session)):
    if not await ingest.retry_job(session, job_id):
        raise HTTPException(404, "job not found")
    return {"requeued": True}


@router.get("/jobs")
async def jobs(status: str | None = None, limit: int = 50, session: AsyncSession = Depends(get_session)):
    q = select(IngestJob).order_by(IngestJob.id.desc()).limit(limit)
    if status:
        q = q.where(IngestJob.status == status)
    return [{"id": j.id, "status": j.status, "attempts": j.attempts, "agent": j.agent, "error": j.error,
             "result": j.result, "created_at": j.created_at, "preview": j.text[:160]}
            for j in (await session.execute(q)).scalars()]
