import base64
import hashlib
from urllib.parse import parse_qs, urlparse

from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette

from src import oauth
from src.memory import api

PUBLIC = "https://engram.example"


def _pkce():
    verifier = "v" * 64
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


async def test_chatgpt_style_oauth_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(oauth.settings, "admin_token", "admin-secret")
    provider = oauth.EngramOAuthProvider(tmp_path / "oauth.json", PUBLIC)
    app = Starlette(routes=oauth._routes(provider))
    async with AsyncClient(transport=ASGITransport(app=app), base_url=PUBLIC) as c:
        meta = (await c.get("/.well-known/oauth-protected-resource/mcp")).json()
        assert meta["authorization_servers"] == [PUBLIC + "/"]
        assert (await c.get("/.well-known/oauth-authorization-server")).json()["registration_endpoint"]

        client = (await c.post("/register", json={
            "client_name": "ChatGPT", "redirect_uris": ["https://chatgpt.com/cb"],
            "grant_types": ["authorization_code", "refresh_token"], "token_endpoint_auth_method": "none",
        })).json()
        verifier, challenge = _pkce()
        r = await c.get("/authorize", params={
            "response_type": "code", "client_id": client["client_id"], "redirect_uri": "https://chatgpt.com/cb",
            "code_challenge": challenge, "code_challenge_method": "S256", "state": "xyz",
        })
        consent = urlparse(r.headers["location"])
        assert consent.path == oauth.CONSENT_PATH
        req = parse_qs(consent.query)["req"][0]
        assert "ChatGPT" in (await c.get(oauth.CONSENT_PATH, params={"req": req})).text

        wrong = await c.post(oauth.CONSENT_PATH, data={"req": req, "admin_token": "nope", "action": "approve"})
        assert "Wrong token" in wrong.text
        ok = await c.post(oauth.CONSENT_PATH, data={"req": req, "admin_token": "admin-secret", "action": "approve"})
        back = parse_qs(urlparse(ok.headers["location"]).query)
        assert back["state"] == ["xyz"]

        tok = (await c.post("/token", data={
            "grant_type": "authorization_code", "code": back["code"][0], "client_id": client["client_id"],
            "redirect_uri": "https://chatgpt.com/cb", "code_verifier": verifier,
        })).json()
        assert await provider.load_access_token(tok["access_token"])
        assert tok["access_token"] not in (tmp_path / "oauth.json").read_text()  # only hashes on disk

        again = await c.post("/token", data={
            "grant_type": "authorization_code", "code": back["code"][0], "client_id": client["client_id"],
            "redirect_uri": "https://chatgpt.com/cb", "code_verifier": verifier,
        })
        assert again.status_code == 400  # codes are single-use

        new = (await c.post("/token", data={
            "grant_type": "refresh_token", "refresh_token": tok["refresh_token"], "client_id": client["client_id"],
        })).json()
        assert await provider.load_access_token(new["access_token"])

    # a restart keeps the session: a fresh provider reads the same file
    assert await oauth.EngramOAuthProvider(tmp_path / "oauth.json", PUBLIC).load_access_token(new["access_token"])

    # the middleware accepts the OAuth token from a remote client, and rejects others
    monkeypatch.setattr(oauth, "provider", provider)
    assert await api._authorized(f"Bearer {new['access_token']}")
    assert await api._authorized("Bearer admin-secret")
    assert not await api._authorized("Bearer forged")
