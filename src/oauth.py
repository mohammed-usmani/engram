"""OAuth 2.1 for MCP clients that can't send a fixed bearer header (ChatGPT, Gemini).

The MCP SDK serves discovery, /register, /authorize, /token and /revoke; this module is the
provider behind them. Approving a client means typing ADMIN_TOKEN on /oauth/consent, so OAuth
grants nothing the admin token doesn't already. Clients and tokens persist in a JSON file
(only token hashes are stored), so a restart doesn't sign ChatGPT out.

Enabled when both PUBLIC_URL (the HTTPS address clients reach, e.g. a Tailscale Funnel URL)
and ADMIN_TOKEN are set.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.routing import Route

from mcp.server.auth.provider import (
    AccessToken, AuthorizationCode, AuthorizationParams, RefreshToken, TokenError, construct_redirect_uri,
)
from mcp.server.auth.routes import create_auth_routes, create_protected_resource_routes
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from src.config import settings

ACCESS_TTL = 3600                 # ChatGPT refreshes silently
REFRESH_TTL = 180 * 24 * 3600     # re-approve twice a year
CODE_TTL = 300
PENDING_TTL = 600
CONSENT_PATH = "/oauth/consent"
OPEN_PATHS = ("/authorize", "/token", "/register", "/revoke", CONSENT_PATH)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class EngramOAuthProvider:
    def __init__(self, store: Path, public_url: str):
        self.store, self.public_url = store, public_url.rstrip("/")
        self.pending: dict[str, tuple[str, AuthorizationParams, float]] = {}  # consent waiting
        self.codes: dict[str, AuthorizationCode] = {}

    # -- persistence: {"clients": {id: info}, "access": {hash: tok}, "refresh": {hash: tok}}
    def _load(self) -> dict:
        try:
            return json.loads(self.store.read_text())
        except (OSError, ValueError):
            return {"clients": {}, "access": {}, "refresh": {}}

    def _save(self, data: dict) -> None:
        now = time.time()
        for kind in ("access", "refresh"):
            data[kind] = {h: t for h, t in data[kind].items() if not t.get("expires_at") or t["expires_at"] > now}
        self.store.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.chmod(0o600)
        tmp.replace(self.store)

    # -- clients
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        info = self._load()["clients"].get(client_id)
        return OAuthClientInformationFull.model_validate(info) if info else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        data = self._load()
        data["clients"][client_info.client_id] = client_info.model_dump(mode="json")
        self._save(data)

    # -- authorization: park the request, send the browser to the consent page
    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        now = time.time()
        self.pending = {k: v for k, v in self.pending.items() if v[2] > now}
        req = secrets.token_urlsafe(24)
        self.pending[req] = (client.client_id, params, now + PENDING_TTL)
        return f"{self.public_url}{CONSENT_PATH}?{urlencode({'req': req})}"

    def approve(self, req: str) -> str | None:
        """Consent given: mint a code, return the client's redirect URL."""
        entry = self.pending.pop(req, None)
        if entry is None or entry[2] < time.time():
            return None
        client_id, params, _ = entry
        code = AuthorizationCode(
            code=secrets.token_urlsafe(32), scopes=params.scopes or [], expires_at=time.time() + CODE_TTL,
            client_id=client_id, code_challenge=params.code_challenge, redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly, resource=params.resource,
        )
        self.codes[code.code] = code
        return construct_redirect_uri(str(params.redirect_uri), code=code.code, state=params.state)

    def deny(self, req: str) -> str | None:
        entry = self.pending.pop(req, None)
        if entry is None:
            return None
        params = entry[1]
        return construct_redirect_uri(str(params.redirect_uri), error="access_denied", state=params.state)

    async def load_authorization_code(self, client, authorization_code: str) -> AuthorizationCode | None:
        code = self.codes.get(authorization_code)
        if code is None or code.client_id != client.client_id or code.expires_at < time.time():
            return None
        return code

    async def exchange_authorization_code(self, client, authorization_code: AuthorizationCode) -> OAuthToken:
        if self.codes.pop(authorization_code.code, None) is None:
            raise TokenError("invalid_grant", "authorization code already used")
        return self._issue(client.client_id, authorization_code.scopes, authorization_code.resource)

    # -- tokens
    def _issue(self, client_id: str, scopes: list[str], resource: str | None = None) -> OAuthToken:
        now = int(time.time())
        access, refresh = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        data = self._load()
        data["access"][_hash(access)] = AccessToken(
            token="", client_id=client_id, scopes=scopes, expires_at=now + ACCESS_TTL, resource=resource,
        ).model_dump()
        data["refresh"][_hash(refresh)] = RefreshToken(
            token="", client_id=client_id, scopes=scopes, expires_at=now + REFRESH_TTL,
        ).model_dump()
        self._save(data)
        return OAuthToken(access_token=access, token_type="Bearer", expires_in=ACCESS_TTL,
                          refresh_token=refresh, scope=" ".join(scopes) or None)

    async def load_refresh_token(self, client, refresh_token: str) -> RefreshToken | None:
        t = self._load()["refresh"].get(_hash(refresh_token))
        if not t or t["client_id"] != client.client_id or (t["expires_at"] or 0) < time.time():
            return None
        return RefreshToken(**{**t, "token": refresh_token})

    async def exchange_refresh_token(self, client, refresh_token: RefreshToken, scopes: list[str]) -> OAuthToken:
        data = self._load()
        data["refresh"].pop(_hash(refresh_token.token), None)  # rotate: each refresh token works once
        self._save(data)
        return self._issue(client.client_id, scopes or refresh_token.scopes)

    async def load_access_token(self, token: str) -> AccessToken | None:
        t = self._load()["access"].get(_hash(token))
        if not t or (t["expires_at"] or 0) < time.time():
            return None
        return AccessToken(**{**t, "token": token})

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        data = self._load()
        h = _hash(token.token)
        data["access"].pop(h, None)
        data["refresh"].pop(h, None)
        # revoking either half ends the client's session
        for kind in ("access", "refresh"):
            data[kind] = {k: v for k, v in data[kind].items() if v["client_id"] != token.client_id}
        self._save(data)


_CONSENT_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Connect to Engram</title>
<style>body{{font:16px system-ui;max-width:420px;margin:12vh auto;padding:0 16px}}
input,button{{font:inherit;padding:10px;width:100%;box-sizing:border-box;margin-top:8px}}
.err{{color:#b00}}</style></head><body>
<h2>Connect {client} to Engram?</h2>
<p>It will be able to read and change your memories and documents.</p>
{error}<form method="post">
<input type="hidden" name="req" value="{req}">
<input type="password" name="admin_token" placeholder="Admin token" autofocus>
<button name="action" value="approve">Approve</button>
<button name="action" value="deny">Deny</button>
</form></body></html>"""


def _routes(provider: EngramOAuthProvider) -> list[Route]:
    async def consent(request: Request):
        if request.method == "GET":
            req = request.query_params.get("req", "")
            error = ""
        else:
            form = await request.form()
            req = str(form.get("req", ""))
            if form.get("action") == "deny":
                url = provider.deny(req)
                return RedirectResponse(url, 302) if url else HTMLResponse("Request expired.", 400)
            if hmac.compare_digest(str(form.get("admin_token", "")).strip(), settings.admin_token or ""):
                url = provider.approve(req)
                return RedirectResponse(url, 302) if url else HTMLResponse("Request expired. Connect again.", 400)
            error = '<p class="err">Wrong token.</p>'
        entry = provider.pending.get(req)
        if entry is None:
            return HTMLResponse("Request expired. Connect again from your assistant.", 400)
        client = await provider.get_client(entry[0])
        name = (client.client_name if client else None) or "this app"
        return HTMLResponse(_CONSENT_HTML.format(client=html.escape(name), req=html.escape(req), error=error))

    base = AnyHttpUrl(provider.public_url)
    return [
        *create_auth_routes(provider, issuer_url=base,
                            client_registration_options=ClientRegistrationOptions(enabled=True),
                            revocation_options=RevocationOptions(enabled=True)),
        *create_protected_resource_routes(AnyHttpUrl(f"{provider.public_url}/mcp"), [base], resource_name="Engram"),
        Route(CONSENT_PATH, consent, methods=["GET", "POST"]),
    ]


provider: EngramOAuthProvider | None = None
routes: list[Route] = []
if settings.public_url and settings.admin_token:
    provider = EngramOAuthProvider(settings.oauth_store, settings.public_url)
    routes = _routes(provider)


def resource_metadata_url() -> str | None:
    return f"{provider.public_url}/.well-known/oauth-protected-resource/mcp" if provider else None
