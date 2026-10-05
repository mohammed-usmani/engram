"""Per-document privacy and the request origin it is enforced against.

normal      everywhere: recall briefs, search, listings, every assistant
private     never in automatic recall or search; listed by title and returned when asked for by slug
local-only  as private, and invisible to remote requests (anything through the public link or a proxy)
"""
from contextvars import ContextVar

from sqlalchemy import ColumnElement

PRIVACY_LEVELS = ("normal", "private", "local-only")
PRIVACY_RULE = ("privacy: 'normal' (default — used everywhere), 'private' (never in automatic recall or search; "
                "listed by title and returned when you ask for it by slug), 'local-only' (like private, and "
                "invisible to requests that come through the public link, e.g. web assistants).")

# Set per HTTP request by the auth middleware; stdio MCP and in-process callers are local.
REQUEST_IS_REMOTE: ContextVar[bool] = ContextVar("engram_request_is_remote", default=False)


def is_remote() -> bool:
    return REQUEST_IS_REMOTE.get()


def readable(privacy_col) -> ColumnElement[bool] | bool:
    """Documents this request may see when it names or lists them."""
    return privacy_col != "local-only" if is_remote() else True


def automatic(privacy_col) -> ColumnElement[bool]:
    """Documents allowed into automatic recall briefs and search results."""
    return privacy_col == "normal"
