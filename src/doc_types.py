"""Document type registry: built-in types plus any created with a one-line description."""
from __future__ import annotations

import difflib
import re

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import BUILTIN_TYPES, ContextDocument, DocumentType
from src.privacy import readable
from src.tool_errors import ToolInputError

TYPE_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,39}$")
TYPE_RULE = ("A document's type comes from the `type` argument. If you omit `type` when creating a NEW "
             "document, it is taken from the slug prefix (the part before '/'), so 'project/x' becomes a "
             "project. When you pass `type`, the slug prefix does not have to match it. Existing documents keep "
             "their type unless you pass `type`. Any registered type works (see valid_types / "
             "list_document_types); to create a NEW type, also pass `type_description` (one line).")


async def _ensure_builtins(session: AsyncSession) -> None:
    have = set((await session.execute(select(DocumentType.name))).scalars())
    missing = [n for n in BUILTIN_TYPES if n not in have]
    for name in missing:
        session.add(DocumentType(name=name, description=BUILTIN_TYPES[name], builtin=True))
    if missing:
        await session.flush()


async def type_names(session: AsyncSession) -> list[str]:
    await _ensure_builtins(session)
    return sorted((await session.execute(select(DocumentType.name))).scalars())


async def list_types(session: AsyncSession) -> list[dict]:
    await _ensure_builtins(session)
    counts = dict((await session.execute(
        select(ContextDocument.type, func.count()).where(readable(ContextDocument.privacy))
        .group_by(ContextDocument.type))).all())
    rows = (await session.execute(select(DocumentType).order_by(DocumentType.builtin.desc(), DocumentType.name))).scalars()
    return [{"name": t.name, "description": t.description, "count": counts.get(t.name, 0)} for t in rows]


async def resolve_type(session: AsyncSession, value: str, *, from_slug: str | None = None,
                       description: str | None = None) -> str:
    """Return a valid registered type name, registering a new one when a description is given."""
    names = await type_names(session)
    where = f"taken from the slug prefix of '{from_slug}'" if from_slug else "passed as `type`"
    value = (value or "").strip()
    if value in names:
        return value
    if not TYPE_NAME_RE.match(value):
        raise ToolInputError(
            f"'{value}' is not a valid document type name ({where}).",
            "Type names are lowercase letters, digits and hyphens, 2-40 chars, starting with a letter "
            "(e.g. 'recipe', 'car-service'). Use one of valid_types or a new well-formed name.",
            valid_types=names, how_it_works=TYPE_RULE)
    close = difflib.get_close_matches(value, names, n=3, cutoff=0.8) + \
        [n for n in names if value in (n + "s", n + "es") or n in (value + "s", value + "es")]
    close = list(dict.fromkeys(close))
    if close:
        raise ToolInputError(
            f"'{value}' ({where}) looks like the existing type {close[0]!r}; not creating a near-duplicate.",
            f"Use type={close[0]!r}" + (" (pass `type` explicitly; the slug can keep its prefix)" if from_slug else "")
            + ". If you really need a separate type, choose a clearly different name.",
            did_you_mean=close, valid_types=names, how_it_works=TYPE_RULE)
    if not (description or "").strip():
        raise ToolInputError(
            f"'{value}' ({where}) is not an existing document type.",
            f"To create it, call again with type_description='<one line: what goes in {value} documents>'. "
            "Or use one of valid_types" + (" by passing `type` explicitly." if from_slug else "."),
            valid_types=names, how_it_works=TYPE_RULE)
    session.add(DocumentType(name=value, description=description.strip()[:300], builtin=False))
    await session.flush()
    return value
