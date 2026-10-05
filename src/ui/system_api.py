"""System controls behind the Settings page: backups, the public link, connected apps, the admin token.
Anything that changes access or secrets works only from this computer."""
from __future__ import annotations

import asyncio
import json
import re
import secrets
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src import oauth
from src.config import settings
from src.privacy import is_remote

router = APIRouter(prefix="/api/system", tags=["system"])
REPO = Path(__file__).resolve().parent.parent.parent
TOKEN_FILE = Path.home() / ".config/engram/admin_token"
_backup_lock = asyncio.Lock()


def _local_only(what: str) -> None:
    if is_remote():
        raise HTTPException(403, f"{what} only works from the computer Engram runs on.")


async def _run(*args: str, timeout: float = 300) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await asyncio.wait_for(p.communicate(), timeout)
    return p.returncode, out.decode("utf-8", "replace").strip()


@router.post("/backup")
async def backup_now():
    """Runs scripts/backup.sh (encrypted dump, pushed to the private repo) and reports its last line."""
    if _backup_lock.locked():
        raise HTTPException(409, "A backup is already running.")
    async with _backup_lock:
        code, out = await _run("sh", str(REPO / "scripts/backup.sh"), timeout=600)
    if code != 0:
        raise HTTPException(500, f"Backup failed: {out[-400:]}")
    from src.ui.templating import invalidate
    invalidate()
    return {"ok": True, "summary": out.splitlines()[-1] if out else ""}


def _tailscale() -> str | None:
    return shutil.which("tailscale") or next((p for p in ("/usr/local/bin/tailscale",
                                                          "/Applications/Tailscale.app/Contents/MacOS/Tailscale")
                                              if Path(p).exists()), None)


@router.get("/public-link")
async def public_link():
    ts = _tailscale()
    if not ts:
        return {"available": False, "on": False, "url": settings.public_url}
    code, out = await _run(ts, "funnel", "status", "--json", timeout=20)
    try:
        status = json.loads(out) if code == 0 else {}
    except ValueError:
        status = {}
    on = any(v for v in (status.get("AllowFunnel") or {}).values())
    return {"available": True, "on": on, "url": settings.public_url}


class OnIn(BaseModel):
    on: bool


@router.post("/public-link")
async def set_public_link(body: OnIn):
    """Turns Tailscale Funnel to this server on or off. Off cuts ChatGPT and other web assistants."""
    _local_only("Turning the public link on or off")
    ts = _tailscale()
    if not ts:
        raise HTTPException(400, "Tailscale isn't installed on this computer.")
    on = body.on
    args = [ts, "funnel", "--bg", "8001"] if on else [ts, "funnel", "--https=443", "off"]
    code, out = await _run(*args, timeout=30)
    if code != 0:
        raise HTTPException(500, f"tailscale: {out[-300:]}")
    return await public_link()


@router.get("/apps")
async def connected_apps():
    """OAuth clients (ChatGPT, Gemini...) and whether they currently hold a live token."""
    p = oauth.provider
    if p is None:
        return []
    data = p._load()
    live = {t["client_id"] for kind in ("access", "refresh") for t in data[kind].values()}
    return [{"client_id": cid, "name": info.get("client_name") or cid, "issued_at": info.get("client_id_issued_at"),
             "connected": cid in live} for cid, info in data["clients"].items()]


@router.delete("/apps/{client_id}")
async def disconnect_app(client_id: str):
    _local_only("Disconnecting apps")
    p = oauth.provider
    if p is None:
        raise HTTPException(404, "OAuth is not enabled.")
    data = p._load()
    if client_id not in data["clients"]:
        raise HTTPException(404, "No such app.")
    data["clients"].pop(client_id)
    for kind in ("access", "refresh"):
        data[kind] = {h: t for h, t in data[kind].items() if t["client_id"] != client_id}
    p._save(data)
    return {"disconnected": client_id}


@router.get("/admin-token")
async def reveal_token():
    _local_only("Showing the admin token")
    return {"token": settings.admin_token}


@router.post("/admin-token/rotate")
async def rotate_token():
    """New token in .env, in ~/.config/engram/admin_token (read by the sync script) and in this process.
    Browsers signed in through the public link must sign in again; connected OAuth apps keep working."""
    _local_only("Replacing the admin token")
    new = secrets.token_urlsafe(32)
    env = REPO / ".env"
    text = env.read_text() if env.exists() else ""
    if re.search(r"^ADMIN_TOKEN=.*$", text, flags=re.M):
        text = re.sub(r"^ADMIN_TOKEN=.*$", f"ADMIN_TOKEN={new}", text, flags=re.M)
    else:
        text += f"\nADMIN_TOKEN={new}\n"
    env.write_text(text)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(new + "\n")
    TOKEN_FILE.chmod(0o600)
    settings.admin_token = new
    return {"token": new}


@router.get("/health")
async def health():
    """Embeddings (Ollama) and the fact store, checked live."""
    out = {}
    try:
        from src.services.ollama_client import embed
        vec = await asyncio.wait_for(embed("health check"), 15)
        out["embeddings"] = {"ok": bool(vec), "detail": f"{settings.ollama_embed_model}, {len(vec)} dimensions"}
    except Exception as e:
        out["embeddings"] = {"ok": False, "detail": (str(e) or type(e).__name__)[:200]}
    return out
