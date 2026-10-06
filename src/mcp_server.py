from __future__ import annotations

import base64
import difflib
import logging
import re
from typing import Any

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import AsyncSessionLocal
from src import attachments as _att, dates as _dates
from src.doc_types import TYPE_RULE, list_types, resolve_type, type_names
from src.models import ContextDocument, DocType, Source, ChatSession, ChatMessage
from src.privacy import PRIVACY_LEVELS, PRIVACY_RULE, automatic, is_remote, readable
from src.tool_errors import ToolInputError
from src.services.chat import handle_chat
from src.services.memory import list_memories as _list_memories, delete_memory as _delete_memory

from mcp.server.transport_security import TransportSecuritySettings
from src.memory import ingest as _ingest, recall as _recall_mod
from src.memory.recall import recall as _recall

log = logging.getLogger(__name__)
mcp = FastMCP(
    "engram",
    instructions=(
        "Engram is the user's personal memory. Call recall() first for anything personal. "
        "Facts and events: remember(). Documents (resume, projects, experience, skills, "
        "education, achievements, certifications, and profile/* with the exact current text of "
        "each public profile) are the source of truth for their career: "
        "read with list_documents/get_document (for how-it-works or why-it-was-built-that-way "
        "questions about a project, call search_context with the topic, then get_document on "
        "the hits; project docs hold diagrams, parameters and decisions), change with edit_document (small edits) or "
        "save_document (create or replace), remove with delete_document. Every write needs "
        "`agent` (who you are: claude-code, claude-web, chatgpt, gemini, codex...) and remember() "
        "also needs `occurred_at`, the real date and time the event happened."
    ),
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    # Stateless: each request stands alone. ChatGPT's connector sometimes drops the
    # mcp-session-id header between calls ("Missing session ID" -> 400 -> 'connection failed').
    stateless_http=True,
)

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "as", "is", "are", "be", "we", "you", "our", "your", "it",
    "this", "that", "will", "have", "has", "had", "not", "no", "but", "can",
    "must", "should", "would", "may", "etc", "using", "use", "used", "role",
    "team", "work", "working", "build", "built",
}


def _t(d: ContextDocument) -> str:
    return getattr(d.type, "value", d.type)


def _doc_summary(d: ContextDocument) -> dict[str, Any]:
    return {"slug": d.slug, "title": d.title, "type": _t(d), "tags": d.tags, "privacy": d.privacy or "normal"}


def _doc_full(d: ContextDocument) -> dict[str, Any]:
    return {
        "slug": d.slug, "title": d.title, "type": _t(d), "tags": d.tags, "privacy": d.privacy or "normal",
        "content": d.content, "sections": d.sections, "metadata": d.doc_metadata,
        "updated_by": d.updated_by, "updated_at": d.updated_at.isoformat() if d.updated_at else None,
    }


SLUG_RULE = "Slugs look like '<type>/<name>', lowercase with hyphens, e.g. 'project/voice-agent', 'skill/python'."
ITEM_ID_RULE = ("Item ids come from recall(): 'e:<number>' event, 'r:<number>' lesson, "
                "'f:<uuid>' fact/preference/procedure, 'd:<slug>' document. Copy them exactly.")


async def _with_session(fn):
    try:
        async with AsyncSessionLocal() as session:
            return await fn(session)
    except ToolInputError as e:
        return e.payload
    except Exception as e:
        log.exception("tool failure")
        return {"error": f"internal error in Engram ({type(e).__name__}: {e})",
                "fix": "This is a server-side bug, not your input. Retrying with the same arguments will fail "
                       "the same way; tell the user Engram needs fixing (details are in its server log)."}


async def _find_doc(session: AsyncSession, slug: str) -> ContextDocument:
    doc = (await session.execute(select(ContextDocument).where(
        ContextDocument.slug == slug, readable(ContextDocument.privacy)))).scalar_one_or_none()
    if doc is None:
        slugs = list((await session.execute(select(ContextDocument.slug).where(readable(ContextDocument.privacy)))).scalars())
        close = difflib.get_close_matches(slug, slugs, n=5, cutoff=0.5)
        raise ToolInputError(f"No document with slug '{slug}'.",
                             "Use an exact slug from did_you_mean, list_documents or search_context. " + SLUG_RULE,
                             did_you_mean=close)
    return doc


@mcp.tool()
async def list_document_types() -> list[dict] | dict:
    """List the document types in use, each with a description and how many documents have it.
    Built-in types cover career (resume, project, skill, experience, education, achievement,
    certification, profile) and everyday life (interview, note, person, health, finance, home,
    travel, learning, reference). Any of these is valid for `type`; to create a new type, pass
    `type_description` to save_document. Then call list_documents(type=...) / get_document(slug)."""
    async def inner(session: AsyncSession):
        return await list_types(session)
    return await _with_session(inner)


@mcp.tool()
async def list_documents(type: str | None = None, tag: str | None = None) -> list[dict] | dict:
    """List all documents about the user (slim summaries — slug, title, type, tags, privacy).
    Private documents are listed by title only in the sense that their content is never pushed into
    recall/search; fetch one with get_document(slug) when the user asks for it.

    IMPORTANT: This returns only summaries. To get the FULL content of any document
    (including detailed project write-ups with tech stack, architecture, challenges,
    features, and metrics), call get_document(slug) with the slug from this list.

    Filter examples:
    - list_documents(type='project') → all project documents
    - list_documents(type='skill') → all skill documents
    - list_documents(type='experience') → all work experiences
    - list_documents(tag='backend') → filter by tag

    Project documents are long and detailed; call get_document to retrieve one."""
    async def inner(session: AsyncSession):
        stmt = select(ContextDocument).where(readable(ContextDocument.privacy))
        if type:
            names = await type_names(session)
            if type not in names:
                raise ToolInputError(f"'{type}' is not a document type.", "Filter with one of valid_types.",
                                     valid_types=names, how_it_works=TYPE_RULE)
            stmt = stmt.where(ContextDocument.type == type)
        if tag:
            stmt = stmt.where(ContextDocument.tags.contains([tag.lower()]))
        stmt = stmt.order_by(ContextDocument.type, ContextDocument.title)
        rows = (await session.execute(stmt)).scalars().all()
        return [_doc_summary(r) for r in rows]
    return await _with_session(inner)


@mcp.tool()
async def get_document(slug: str) -> dict:
    """Get the FULL content of one document by slug — this is how you access
    deep per-item detail (e.g., full project write-ups with tech stack, architecture,
    features, challenges, metrics; detailed skill breakdowns; full work experience
    descriptions). Use this AFTER list_documents to drill into specific items.

    Example slugs: 'project/trailmap', 'skill/python', 'skill/postgresql',
    'experience/example-labs', 'education/bsc', 'resume/master_resume'.

    Returns: {slug, title, type, tags, content (full text), sections (parsed sections),
    metadata (key-value attributes), attachments (files with their extracted text, first 4000 chars)}.
    An unknown slug returns `error`, `fix` and `did_you_mean`."""
    async def inner(session: AsyncSession):
        doc = await _find_doc(session, slug)
        return {**_doc_full(doc), "attachments": [_att.summary(a) for a in await _att.for_doc(session, doc)]}
    return await _with_session(inner)


@mcp.tool()
async def attach_file(slug: str, filename: str, content_base64: str, agent: str) -> dict:
    """Attach a file (PDF, image, text...) to an existing document, e.g. a scanned passport to
    'travel/passport' or a policy PDF to 'finance/car-insurance'. `content_base64` is the whole file,
    base64-encoded; `filename` needs its extension (it decides extraction: PDF text layer, OCR for images
    and scans, plain text). Max 20 MB. The extracted text becomes searchable with the document, and the file
    inherits the document's privacy. Re-sending identical bytes is a no-op. Create the document first with
    save_document if it does not exist. `agent` (required): who you are."""
    async def inner(session: AsyncSession):
        who, _ = _agent(agent)
        doc = await _find_doc(session, slug)
        try:
            data = base64.b64decode(content_base64 or "", validate=True)
        except (ValueError, TypeError):
            raise ToolInputError("`content_base64` is not valid base64.",
                                 "Send the file's raw bytes base64-encoded (standard alphabet, no data: URL prefix, "
                                 "no line breaks).") from None
        a, note = await _att.add(session, doc, filename, data, who)
        return {"id": a.id, "slug": doc.slug, "filename": a.filename, "mime": a.mime, "size": a.size,
                "text_chars": len(a.text), "extraction": note, "privacy": doc.privacy}
    return await _with_session(inner)


@mcp.tool()
async def delete_attachment(slug: str, attachment_id: int, agent: str) -> dict:
    """Remove one attached file from a document (ids are in get_document's `attachments`). `agent` required."""
    async def inner(session: AsyncSession):
        _agent(agent)
        doc = await _find_doc(session, slug)
        await _att.remove(session, doc, attachment_id)
        return {"deleted": attachment_id, "slug": doc.slug}
    return await _with_session(inner)


def _agent(agent: str | None) -> tuple[str, dict | None]:
    """Every write says who made it, so memory can be traced back to the assistant that wrote it."""
    who = (agent or "").strip().lower()
    if not who:
        raise ToolInputError("`agent` is required on every write and was empty.",
                             "Pass who you are, e.g. 'chatgpt', 'claude-code', 'claude-web', 'gemini', 'codex'. "
                             "It is stored as updated_by so the user can trace who wrote what.")
    return who[:64], None


async def _save(session: AsyncSession, doc: ContextDocument, agent: str) -> dict:
    # Same path as an edit in /admin: sections and metadata re-derived, re-embedded.
    from src.routers.admin import _refresh_derived
    doc.source = Source.MANUAL
    doc.updated_by = agent
    await _refresh_derived(doc)
    await session.commit()
    return _doc_summary(doc)


@mcp.tool()
async def save_document(slug: str, agent: str, content: str | None = None, title: str | None = None,
                        type: str | None = None, tags: list[str] | None = None,
                        type_description: str | None = None, privacy: str | None = None) -> dict:
    """Create or fully replace a document about the user (resume, project, experience...).

    `content` is plain text and replaces the whole document: `Key: value` lines at the top
    become metadata, a line like `Summary:` starts a section. Call get_document first and
    send back the full edited text. For a small change prefer edit_document. To only
    rename or retag an existing document, omit `content`.

    New document: slug is '<type>/<name>' (e.g. 'project/voiceagent'); `title` and `content` are
    required. HOW THE TYPE IS CHOSEN: `type` if you pass it; otherwise the slug prefix (before '/').
    Valid types: resume, project, skill, experience, education, achievement, certification, profile.
    If your slug prefix is not one of these (e.g. 'interview/...'), pass `type` explicitly — the slug
    can keep its prefix. Built-in types also cover everyday life: interview, note, person, health,
    finance, home, travel, learning, reference. A brand-new type needs `type_description` (one line).
    `privacy`: 'normal' (default), 'private' (kept out of automatic recall/search; returned by slug) or
    'local-only' (private, and invisible to web assistants that reach Engram through its public link).
    Use private/local-only for health, money, ID numbers, family matters.
    Existing document: title, type, tags and privacy are kept unless you pass them.
    On any mistake the reply has `error` (what was wrong) and `fix` (what to send instead).
    `agent` (required): who you are, e.g. claude-code, chatgpt, gemini; stored as updated_by."""
    async def inner(session: AsyncSession):
        who, err = _agent(agent)
        if err:
            return err
        if privacy is not None and privacy not in PRIVACY_LEVELS:
            raise ToolInputError(f"`privacy` '{privacy}' is not a privacy level.",
                                 "Use 'normal', 'private' or 'local-only'.", how_it_works=PRIVACY_RULE)
        if is_remote() and privacy == "local-only":
            raise ToolInputError("A remote request can't set privacy 'local-only': it would lock the document "
                                 "away from you immediately.", "Use 'private', or set local-only from this "
                                 "computer (admin page or a local assistant).", how_it_works=PRIVACY_RULE)
        doc = (await session.execute(select(ContextDocument).where(ContextDocument.slug == slug))).scalar_one_or_none()
        if doc is not None and doc.privacy == "local-only" and is_remote():
            raise ToolInputError(f"No document with slug '{slug}' is available to this connection.",
                                 "Choose a different slug for a new document.")
        if doc is None:
            if "/" not in (slug or "") or slug.startswith("/") or slug.endswith("/"):
                raise ToolInputError(f"Slug '{slug}' is not in '<type>/<name>' form.", SLUG_RULE,
                                     how_it_works=TYPE_RULE)
            missing = [n for n, v in (("title", title), ("content", content)) if not v]
            if missing:
                raise ToolInputError(
                    f"No document '{slug}' exists, so this call creates one, and that needs: {', '.join(missing)}.",
                    "Pass the missing field(s). If you meant to change an existing document, use its exact slug "
                    "(see list_documents) — then title/type/tags are kept unless you pass them.")
            doc_type = (await resolve_type(session, type, description=type_description) if type else
                        await resolve_type(session, slug.split("/")[0], from_slug=slug, description=type_description))
            doc = ContextDocument(slug=slug, title=title, type=doc_type, privacy=privacy or "normal",
                                  tags=[], content="", sections={}, doc_metadata={})
            session.add(doc)
        elif privacy is not None:
            doc.privacy = privacy
        if title:
            doc.title = title
        if type:
            doc.type = await resolve_type(session, type, description=type_description)
        if tags is not None:
            doc.tags = [t.strip().lower() for t in tags if t.strip()]
        if content is not None:
            doc.content = content
        return await _save(session, doc, who)
    return await _with_session(inner)


@mcp.tool()
async def edit_document(slug: str, old_text: str, new_text: str, agent: str) -> dict:
    """Change part of a document: replaces `old_text` with `new_text` in its content.
    `old_text` must appear exactly once (copy it from get_document, including line
    breaks); otherwise nothing changes and an error says why. Use '' as new_text to delete.
    `agent` (required): who you are, e.g. claude-code, chatgpt, gemini; stored as updated_by."""
    async def inner(session: AsyncSession):
        who, err = _agent(agent)
        if err:
            return err
        doc = await _find_doc(session, slug)
        if not old_text:
            raise ToolInputError("`old_text` is empty.", "Pass the exact text to replace, copied from get_document.")
        count = doc.content.count(old_text)
        if count == 0:
            raise ToolInputError(f"`old_text` was found 0 times in '{slug}'; it must match exactly once.",
                                 "Call get_document and copy old_text exactly (same spaces, line breaks, "
                                 "punctuation). For a large rewrite use save_document with the full text.")
        if count > 1:
            raise ToolInputError(f"`old_text` was found {count} times in '{slug}'; it must match exactly once.",
                                 "Include more surrounding text in old_text so it matches only one place.")
        doc.content = doc.content.replace(old_text, new_text)
        return await _save(session, doc, who)
    return await _with_session(inner)


@mcp.tool()
async def delete_document(slug: str, agent: str) -> dict:
    """Permanently delete a document. Only when the user asks, or it is wrong/duplicated.
    `agent` (required): who you are."""
    async def inner(session: AsyncSession):
        who, err = _agent(agent)
        if err:
            return err
        doc = await _find_doc(session, slug)
        await session.delete(doc)
        await session.commit()
        log.info("document %s deleted by %s", slug, who)
        return {"deleted": slug}
    return await _with_session(inner)


@mcp.tool()
async def search_context(query: str, limit: int = 10) -> list[dict] | dict:
    """Full-text search across all documents (resume, projects, skills, experiences,
    education, achievements, certifications). Returns ranked slim hits with snippets. Use this to find relevant
    documents by keyword (tech names, project names, concepts), then call
    get_document(slug) to retrieve full content of interesting results.

    Examples: search_context('PostgreSQL') → docs mentioning PostgreSQL
              search_context('voice AI') → voice-related projects/skills
              search_context('microservices')"""
    async def inner(session: AsyncSession):
        if not (query or "").strip():
            raise ToolInputError("`query` is empty.", "Pass keywords to search for, e.g. 'Redfox interview' or 'PostgreSQL'.")
        tsq = func.plainto_tsquery("english", query)
        stmt = (
            select(
                ContextDocument,
                func.ts_rank(ContextDocument.search_vector, tsq).label("rank"),
                func.ts_headline(
                    "english", ContextDocument.content + "\n" + ContextDocument.attachments_text, tsq,
                    "MaxFragments=2, MaxWords=25, MinWords=8",
                ).label("snippet"),
            )
            .where(ContextDocument.search_vector.op("@@")(tsq), automatic(ContextDocument.privacy))
            .order_by(func.ts_rank(ContextDocument.search_vector, tsq).desc())
            .limit(limit)
        )
        rows = (await session.execute(stmt)).all()
        return [
            {**_doc_summary(r[0]), "rank": float(r[1]), "snippet": r[2]}
            for r in rows
        ]
    return await _with_session(inner)


@mcp.tool()
async def get_resume() -> dict:
    """Return the master resume, plus `other_versions`: every other resume document (e.g. the LaTeX
    source) with its slug, so you can fetch the exact one you need with get_document(slug).

    If the user wants LaTeX, a PDF-ready file or "my usual template", look in other_versions for the
    LaTeX source and get_document it; never rebuild a template from memory.

    NOTE: This is a high-level resume only. For detailed, deep information about
    any specific project, skill, or work experience, use list_documents + get_document
    instead. Each project has a full write-up with architecture, tech stack, features,
    challenges, and metrics that goes far beyond the resume's one-line bullets."""
    async def inner(session: AsyncSession):
        rows = (await session.execute(
            select(ContextDocument).where(ContextDocument.type == DocType.RESUME.value, readable(ContextDocument.privacy))
            .order_by((ContextDocument.slug == "resume/master_resume").desc(), ContextDocument.updated_at.desc())
        )).scalars().all()
        if not rows:
            raise ToolInputError("There is no resume document yet.",
                                 "Create one with save_document(slug='resume/master_resume', type='resume', ...).")
        main = next((r for r in rows if "latex" not in r.slug.lower()), rows[0])
        return {**_doc_full(main), "other_versions": [
            {"slug": r.slug, "title": r.title, "privacy": r.privacy or "normal",
             "how_to_get": f"get_document('{r.slug}')"} for r in rows if r is not main]}
    return await _with_session(inner)


def _extract_terms(text: str, limit: int = 30) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{1,30}", text.lower())
    seen: list[str] = []
    for w in words:
        if w in _STOPWORDS or len(w) < 3:
            continue
        if w not in seen:
            seen.append(w)
        if len(seen) >= limit:
            break
    return seen


@mcp.tool()
async def find_relevant_context(job_description: str, limit: int = 20) -> dict:
    """Given a pasted job description, return a bundle with resume + top-matching
    projects, skills, and experiences. Optimized for resume tailoring workflows.
    """
    async def inner(session: AsyncSession):
        terms = _extract_terms(job_description)
        if not terms:
            return {"resume": None, "projects": [], "skills": [], "experiences": []}
        tsquery_str = " | ".join(terms)
        tsq = func.to_tsquery("english", tsquery_str)

        resume_row = (await session.execute(
            select(ContextDocument).where(ContextDocument.type == DocType.RESUME.value, readable(ContextDocument.privacy)).limit(1)
        )).scalar_one_or_none()

        async def top(doc_type: DocType, k: int):
            stmt = (
                select(ContextDocument, func.ts_rank(ContextDocument.search_vector, tsq).label("rank"))
                .where(ContextDocument.type == getattr(doc_type, 'value', doc_type), automatic(ContextDocument.privacy))
                .where(ContextDocument.search_vector.op("@@")(tsq))
                .order_by(func.ts_rank(ContextDocument.search_vector, tsq).desc())
                .limit(k)
            )
            rows = (await session.execute(stmt)).all()
            return [{**_doc_summary(r[0]), "rank": float(r[1]), "content": r[0].content} for r in rows]

        per_type = max(3, limit // 3)
        return {
            "job_terms": terms,
            "resume": _doc_full(resume_row) if resume_row else None,
            "projects": await top(DocType.PROJECT, per_type),
            "skills": await top(DocType.SKILL, per_type),
            "experiences": await top(DocType.EXPERIENCE, per_type),
        }
    return await _with_session(inner)


@mcp.tool()
async def chat(message: str, session_id: int | None = None,
               provider: str = "ollama", model: str | None = None) -> dict:
    """Chat with the user's personal context. Creates or continues a session.
    Specify provider (ollama/groq/cerebras/openai/gemini/claude) and model."""
    async def inner(session):
        result = await handle_chat(message, session_id, session,
                                   provider_name=provider, model_name=model)
        return {
            "response": result.response,
            "session_id": result.session_id,
            "sources": result.sources,
            "memories_extracted": result.memories_extracted,
        }
    return await _with_session(inner)


@mcp.tool()
async def list_providers() -> list[dict] | dict:
    """List available LLM providers and their configured status."""
    async def inner(session):
        from src.models import ProviderSetting
        rows = (await session.execute(
            select(ProviderSetting).order_by(ProviderSetting.id)
        )).scalars().all()
        return [{"provider": r.provider, "model": r.model,
                 "is_active": r.is_active, "has_key": bool(r.api_key)} for r in rows]
    return await _with_session(inner)


@mcp.tool()
async def list_chat_sessions() -> list[dict] | dict:
    """List all chat sessions with titles, message counts, and timestamps."""
    async def inner(session):
        rows = (await session.execute(
            select(ChatSession).order_by(ChatSession.updated_at.desc())
        )).scalars().all()
        return [{"id": r.id, "title": r.title, "message_count": r.message_count,
                 "updated_at": r.updated_at.isoformat() if r.updated_at else None} for r in rows]
    return await _with_session(inner)


@mcp.tool()
async def get_chat_session(session_id: int) -> dict:
    """Get full message history for a chat session."""
    async def inner(session):
        cs = (await session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )).scalar_one_or_none()
        if cs is None:
            raise ToolInputError(f"No chat session {session_id}.",
                                 "Use an id from list_chat_sessions. Note: these are Engram's built-in chat only; "
                                 "conversations from other assistants are in recall()/documents, not here.")
        msgs = (await session.execute(
            select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
        )).scalars().all()
        return {
            "id": cs.id, "title": cs.title, "summary": cs.summary,
            "messages": [{"role": m.role.value, "content": m.content,
                          "created_at": m.created_at.isoformat()} for m in msgs],
        }
    return await _with_session(inner)


@mcp.tool()
async def list_memories(category: str | None = None) -> list[dict] | dict:
    """List all stored memories about the user, optionally filtered by category."""
    async def inner(session):
        mems = await _list_memories(session, category=category)
        return [{"id": m.id, "content": m.content, "category": m.category.value,
                 "source_session_id": m.source_session_id,
                 "created_at": m.created_at.isoformat()} for m in mems]
    return await _with_session(inner)


@mcp.tool()
async def delete_memory(memory_id: int) -> dict:
    """Delete a specific memory by ID."""
    async def inner(session):
        await _delete_memory(memory_id, session)
        await session.commit()
        return {"deleted": memory_id}
    return await _with_session(inner)


# ------------------------------------------------------------------ agentic memory

def _check_item_id(item_id: str) -> None:
    prefix, _, key = (item_id or "").partition(":")
    if prefix not in ("e", "r", "f", "d") or not key or (prefix in ("e", "r") and not key.isdigit()):
        raise ToolInputError(f"'{item_id}' is not a memory item id.", ITEM_ID_RULE)

@mcp.tool()
async def recall(situation: str, session_id: str | None = None, budget_tokens: int = 1500, fast: bool = False) -> dict:
    """Get what you should know about the user for the current situation. CALL THIS FIRST whenever
    the user starts a task, mentions something personal (jobs, health, projects, people, plans), or
    you need their preferences. Pass the user's words as `situation`. Returns a ranked brief that
    blends their profile, the current task's notes, counts/timelines of past events (e.g. how many
    job applications, since when, outcomes), lessons learned, preferences, how they do things, and
    relevant documents. Item ids like [e:12] / [f:...] / [r:3] can be passed to expand()."""
    async def inner(session):
        if not (situation or "").strip():
            raise ToolInputError("`situation` is empty.", "Pass what the user said or is doing, in their words.")
        return await _recall(session, situation, budget_tokens, session_id, fast)
    return await _with_session(inner)


@mcp.tool()
async def remember(text: str, agent: str, occurred_at: str, session_id: str | None = None) -> dict:
    """Save something worth remembering about the user, in plain language. Call it when the user
    shares a fact, preference, decision, outcome or event ("I applied to Acme", "the interview went
    badly", "I prefer PDFs", "moved to Bangalore"), or at the end of meaningful work.

    Required:
    - `agent`: who you are, e.g. claude-code, claude-web, chatgpt, gemini, codex, antigravity, cursor.
    - `occurred_at`: when it HAPPENED (not when you're saving it), ISO 8601 with time and offset,
      e.g. '2026-09-29T18:00:00+05:30'. Use the current time only for things happening right now.
    One event per call; name companies and people exactly. Extraction runs in the background."""
    from datetime import datetime
    async def inner(session):
        who, err = _agent(agent)
        if err:
            return err
        if not (text or "").strip():
            raise ToolInputError("`text` is empty.", "Pass what happened or what you learned, in plain language.")
        try:
            when = datetime.fromisoformat(occurred_at)
        except (TypeError, ValueError):
            raise ToolInputError(
                f"`occurred_at` '{occurred_at}' is not an ISO 8601 date/time.",
                "Pass when the event HAPPENED (not when you save it), e.g. '2026-10-05T14:30:00+05:30'. "
                "Resolve words like 'yesterday' to a real date yourself.",
                how_it_works="A time without an offset is read in the user's timezone; a date alone means that day.") from None
        job_id, created = await _ingest.enqueue(session, text, who, session_id, when)
        return {"job_id": job_id, "queued": created}
    return await _with_session(inner)


@mcp.tool()
async def note(session_id: str, key: str, value: str) -> dict:
    """Short-term memory for the CURRENT task only (expires in 7 days): constraints like
    deadline=Friday, format=PDF, audience=CTO. recall(session_id=...) returns them."""
    async def inner(session):
        empty = [n for n, v in (("session_id", session_id), ("key", key), ("value", value)) if not (v or "").strip()]
        if empty:
            raise ToolInputError(f"Empty: {', '.join(empty)}.",
                                 "note needs your current session id, a short key (e.g. 'deadline') and its value.")
        await _ingest.note(session, session_id, key, value)
        return {"ok": True}
    return await _with_session(inner)


@mcp.tool()
async def teach(name: str, steps: list[str], agent: str, entities: list[str] | None = None) -> dict:
    """Save HOW the user does something, step by step (a procedure/workflow), e.g.
    name='deploy cityfix', steps=['run tests','build image','deploy','verify'].
    `agent` (required): who you are, e.g. claude-code, chatgpt, gemini."""
    async def inner(session):
        who, err = _agent(agent)
        if err:
            return err
        if not (name or "").strip() or not [x for x in (steps or []) if str(x).strip()]:
            raise ToolInputError("teach needs a `name` and at least one non-empty step in `steps`.",
                                 "e.g. name='deploy cityfix', steps=['run tests', 'build image', 'deploy', 'verify'].")
        return {"id": await _ingest.teach(session, name, steps, who, entities)}
    return await _with_session(inner)


@mcp.tool()
async def expand(item_id: str) -> dict:
    """Full record for an item id from recall(): e:<n> episode, r:<n> lesson (with its evidence),
    f:<uuid> fact/preference/procedure, d:<slug> document. Copy the id exactly as recall() printed it,
    including the prefix (e.g. 'e:860', not '860')."""
    async def inner(session):
        _check_item_id(item_id)
        item = await _recall_mod.expand(session, item_id)
        if not item:
            raise ToolInputError(f"No item '{item_id}'.",
                                 "Call recall() or timeline() to get current ids (it may have been forgotten). "
                                 + ITEM_ID_RULE)
        return item
    return await _with_session(inner)


@mcp.tool()
async def timeline(entity: str, since: str | None = None, limit: int = 100) -> list[dict] | dict:
    """Chronological events for one entity slug, e.g. 'topic:job-search', 'company:acme'
    (slugs appear in recall() plans). `since` is an ISO date."""
    from datetime import datetime
    async def inner(session):
        from src.memory.models import Entity
        if ":" not in (entity or ""):
            names = list((await session.execute(select(Entity.slug).where(Entity.slug.ilike(f"%{(entity or '').strip().lower()}%")).limit(8))).scalars())
            raise ToolInputError(f"`entity` '{entity}' is not an entity slug in kind:name form.",
                                 "Use kind:name, e.g. 'company:redfox', 'topic:job-search', 'project:voiceagent'. "
                                 "recall() plans list the slugs it matched.", did_you_mean=names)
        try:
            since_dt = datetime.fromisoformat(since) if since else None
        except ValueError:
            raise ToolInputError(f"`since` '{since}' is not an ISO date.",
                                 "Pass a date like '2026-10-01' (or a full ISO date/time), or omit it.") from None
        rows = await _recall_mod.timeline(session, entity, since_dt, limit)
        return [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()} for r in rows]
    return await _with_session(inner)


@mcp.tool()
async def upcoming(days: int = 60) -> dict:
    """Dates coming up in the user's documents within `days` (1-366), plus one-off dates overdue by up to
    30 days: expiries, renewals, due dates, appointments, birthdays, anniversaries. Each item has slug,
    title, label, date, days_left (negative = overdue) and `turns` for birthdays with a known year.
    Private documents appear as these reminder fields only; get_document(slug) for the rest.
    To track a new date, add a `Key: value` line at the top of a document (see how_it_works)."""
    async def inner(session):
        if not isinstance(days, int) or not 1 <= days <= 366:
            raise ToolInputError(f"`days` must be a whole number from 1 to 366, got {days!r}.",
                                 "Pass e.g. days=30 for the next month, or omit it for 60.")
        items, unreadable = await _dates.upcoming(session, days)
        out = {"items": items, "how_it_works": _dates.DATE_RULE}
        if unreadable:
            out["unreadable_dates"] = unreadable
            out["fix_unreadable"] = ("These date fields could not be parsed, so they are not tracked. "
                                     "Rewrite the value with edit_document as e.g. 2027-03-14.")
        return out
    return await _with_session(inner)


@mcp.tool()
async def forget(item_id: str) -> dict:
    """Permanently delete one memory item (e:/r:/f: id) — only when the user asks you to forget it
    or it is wrong. Documents (d:...) are deleted with delete_document instead."""
    async def inner(session):
        _check_item_id(item_id)
        if item_id.startswith("d:"):
            raise ToolInputError("forget() removes memory items, not documents.",
                                 "To delete a document use delete_document(slug=...). " + ITEM_ID_RULE)
        if not await _recall_mod.forget(session, item_id):
            raise ToolInputError(f"No item '{item_id}' to forget.", "Get current ids from recall(). " + ITEM_ID_RULE)
        return {"deleted": item_id}
    return await _with_session(inner)


@mcp.resource("context://documents")
async def resource_index() -> str:
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(ContextDocument).where(readable(ContextDocument.privacy)).order_by(ContextDocument.type, ContextDocument.title)
        )).scalars().all()
        return "\n".join(f"context://documents/{r.slug}\t{r.title}" for r in rows)


@mcp.resource("context://documents/{slug}")
async def resource_doc(slug: str) -> str:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(ContextDocument).where(
            ContextDocument.slug == slug, readable(ContextDocument.privacy)))).scalar_one_or_none()
        if row is None:
            return f"NOT FOUND: {slug}"
        return row.content


@mcp.prompt()
def tailor_resume(job_description: str) -> str:
    """Instruct the model to use the context store to tailor a resume to the given JD."""
    return (
        f"You are helping {settings.user_name} tailor their resume to the job description below.\n"
        "Step 1: call `find_relevant_context` with the job_description to pull relevant resume, projects, skills, experiences.\n"
        "Step 2: rewrite resume bullets so each one emphasizes skills/tech the JD asks for.\n"
        "Step 3: keep factual content truthful — do not invent experience.\n\n"
        f"JOB DESCRIPTION:\n{job_description}\n"
    )


@mcp.prompt()
def enhance_project_description(slug: str, target_role: str) -> str:
    """Rewrite a project description for a specific target role."""
    return (
        f"Call `get_document` with slug={slug!r}. Then rewrite the project description "
        f"to highlight aspects most relevant to a {target_role} role. Keep it honest, "
        f"concise, and metric-forward where numbers exist.\n"
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Personal Context MCP server")
    parser.add_argument("--http", action="store_true", help="Run as HTTP server instead of stdio")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port (default 8001)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host (default 0.0.0.0)")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level)

    if not args.http:
        mcp.run()  # stdio (default)
        return

    # HTTP transport
    import uvicorn
    app = mcp.streamable_http_app()

    if settings.admin_token:
        from src.memory.api import AuthMiddleware
        app = AuthMiddleware(app)
        log.info("HTTP MCP server starting on %s:%d with bearer auth", args.host, args.port)
    else:
        log.warning("HTTP MCP server starting on %s:%d with NO auth (set ADMIN_TOKEN to secure)",
                    args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
