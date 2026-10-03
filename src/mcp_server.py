from __future__ import annotations

import logging
import re
from typing import Any

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import AsyncSessionLocal
from src.models import ContextDocument, DocType, Source, ChatSession, ChatMessage
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
        "education, achievements, certifications) are the source of truth for their career: "
        "read with list_documents/get_document, change with edit_document (small edits) or "
        "save_document (create or replace), remove with delete_document."
    ),
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "as", "is", "are", "be", "we", "you", "our", "your", "it",
    "this", "that", "will", "have", "has", "had", "not", "no", "but", "can",
    "must", "should", "would", "may", "etc", "using", "use", "used", "role",
    "team", "work", "working", "build", "built",
}


def _doc_summary(d: ContextDocument) -> dict[str, Any]:
    return {"slug": d.slug, "title": d.title, "type": d.type.value, "tags": d.tags}


def _doc_full(d: ContextDocument) -> dict[str, Any]:
    return {
        "slug": d.slug, "title": d.title, "type": d.type.value, "tags": d.tags,
        "content": d.content, "sections": d.sections, "metadata": d.doc_metadata,
    }


async def _with_session(fn):
    try:
        async with AsyncSessionLocal() as session:
            return await fn(session)
    except Exception as e:
        log.exception("tool failure")
        return {"error": str(e)}


@mcp.tool()
async def list_document_types() -> list[str]:
    """List the document types stored about the user: resume, project (detailed
    write-ups), skill, experience, education, achievement, certification. Call list_documents
    with a type filter to see titles, then get_document(slug) for full content."""
    return [t.value for t in DocType]


@mcp.tool()
async def list_documents(type: str | None = None, tag: str | None = None) -> list[dict] | dict:
    """List all documents about the user (slim summaries — slug, title, type, tags).

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
        stmt = select(ContextDocument)
        if type:
            stmt = stmt.where(ContextDocument.type == DocType(type))
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
    metadata (key-value attributes)}."""
    async def inner(session: AsyncSession):
        row = (await session.execute(select(ContextDocument).where(ContextDocument.slug == slug))).scalar_one_or_none()
        if row is None:
            return {"error": f"document not found: {slug}"}
        return _doc_full(row)
    return await _with_session(inner)


async def _save(session: AsyncSession, doc: ContextDocument) -> dict:
    # Same path as an edit in /admin: sections and metadata re-derived, re-embedded.
    from src.routers.admin import _refresh_derived
    doc.source = Source.MANUAL
    await _refresh_derived(doc)
    await session.commit()
    return _doc_summary(doc)


@mcp.tool()
async def save_document(slug: str, content: str | None = None, title: str | None = None,
                        type: str | None = None, tags: list[str] | None = None) -> dict:
    """Create or fully replace a document about the user (resume, project, experience...).

    `content` is plain text and replaces the whole document: `Key: value` lines at the top
    become metadata, a line like `Summary:` starts a section. Call get_document first and
    send back the full edited text. For a small change prefer edit_document. To only
    rename or retag an existing document, omit `content`.

    New document: slug is '<type>/<name>' (e.g. 'project/voiceagent'); title is required,
    type defaults to the slug's prefix. Existing document: title, type and tags are kept
    unless you pass them. Types: see list_document_types."""
    async def inner(session: AsyncSession):
        doc = (await session.execute(select(ContextDocument).where(ContextDocument.slug == slug))).scalar_one_or_none()
        if doc is None:
            if not title or content is None:
                return {"error": "title and content are required for a new document"}
            doc = ContextDocument(slug=slug, title=title, type=DocType(type or slug.split("/")[0]),
                                  tags=[], content="", sections={}, doc_metadata={})
            session.add(doc)
        if title:
            doc.title = title
        if type:
            doc.type = DocType(type)
        if tags is not None:
            doc.tags = [t.strip().lower() for t in tags if t.strip()]
        if content is not None:
            doc.content = content
        return await _save(session, doc)
    return await _with_session(inner)


@mcp.tool()
async def edit_document(slug: str, old_text: str, new_text: str) -> dict:
    """Change part of a document: replaces `old_text` with `new_text` in its content.
    `old_text` must appear exactly once (copy it from get_document, including line
    breaks); otherwise nothing changes and an error says why. Use '' as new_text to delete."""
    async def inner(session: AsyncSession):
        doc = (await session.execute(select(ContextDocument).where(ContextDocument.slug == slug))).scalar_one_or_none()
        if doc is None:
            return {"error": f"document not found: {slug}"}
        count = doc.content.count(old_text) if old_text else 0
        if count != 1:
            return {"error": f"old_text found {count} times in {slug}; it must match exactly once"}
        doc.content = doc.content.replace(old_text, new_text)
        return await _save(session, doc)
    return await _with_session(inner)


@mcp.tool()
async def delete_document(slug: str) -> dict:
    """Permanently delete a document. Only when the user asks, or it is wrong/duplicated."""
    async def inner(session: AsyncSession):
        doc = (await session.execute(select(ContextDocument).where(ContextDocument.slug == slug))).scalar_one_or_none()
        if doc is None:
            return {"error": f"document not found: {slug}"}
        await session.delete(doc)
        await session.commit()
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
        tsq = func.plainto_tsquery("english", query)
        stmt = (
            select(
                ContextDocument,
                func.ts_rank(ContextDocument.search_vector, tsq).label("rank"),
                func.ts_headline(
                    "english", ContextDocument.content, tsq,
                    "MaxFragments=2, MaxWords=25, MinWords=8",
                ).label("snippet"),
            )
            .where(ContextDocument.search_vector.op("@@")(tsq))
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
    """Return the master resume document (summary-level).

    NOTE: This is a high-level resume only. For detailed, deep information about
    any specific project, skill, or work experience, use list_documents + get_document
    instead. Each project has a full write-up with architecture, tech stack, features,
    challenges, and metrics that goes far beyond the resume's one-line bullets."""
    async def inner(session: AsyncSession):
        row = (
            await session.execute(
                select(ContextDocument).where(ContextDocument.type == DocType.RESUME).limit(1)
            )
        ).scalar_one_or_none()
        if row is None:
            return {"error": "resume not found"}
        return _doc_full(row)
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
            select(ContextDocument).where(ContextDocument.type == DocType.RESUME).limit(1)
        )).scalar_one_or_none()

        async def top(doc_type: DocType, k: int):
            stmt = (
                select(ContextDocument, func.ts_rank(ContextDocument.search_vector, tsq).label("rank"))
                .where(ContextDocument.type == doc_type)
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
            return {"error": f"session not found: {session_id}"}
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

@mcp.tool()
async def recall(situation: str, session_id: str | None = None, budget_tokens: int = 1500, fast: bool = False) -> dict:
    """Get what you should know about the user for the current situation. CALL THIS FIRST whenever
    the user starts a task, mentions something personal (jobs, health, projects, people, plans), or
    you need their preferences. Pass the user's words as `situation`. Returns a ranked brief that
    blends their profile, the current task's notes, counts/timelines of past events (e.g. how many
    job applications, since when, outcomes), lessons learned, preferences, how they do things, and
    relevant documents. Item ids like [e:12] / [f:...] / [r:3] can be passed to expand()."""
    async def inner(session):
        return await _recall(session, situation, budget_tokens, session_id, fast)
    return await _with_session(inner)


@mcp.tool()
async def remember(text: str, agent: str | None = None, session_id: str | None = None,
                   occurred_at: str | None = None) -> dict:
    """Save something worth remembering about the user, in plain language. Call it when the user
    shares a fact, preference, decision, outcome or event ("I applied to Acme", "the interview went
    badly", "I prefer PDFs", "moved to Bangalore"), or at the end of meaningful work. Include dates
    and outcomes when known. `agent` = your name (claude-code, chatgpt, claude-web...). Extraction
    runs in the background; returns a job id immediately. `occurred_at` ISO time if not now."""
    from datetime import datetime
    async def inner(session):
        when = datetime.fromisoformat(occurred_at) if occurred_at else None
        job_id, created = await _ingest.enqueue(session, text, agent, session_id, when)
        return {"job_id": job_id, "queued": created}
    return await _with_session(inner)


@mcp.tool()
async def note(session_id: str, key: str, value: str) -> dict:
    """Short-term memory for the CURRENT task only (expires in 7 days): constraints like
    deadline=Friday, format=PDF, audience=CTO. recall(session_id=...) returns them."""
    async def inner(session):
        await _ingest.note(session, session_id, key, value)
        return {"ok": True}
    return await _with_session(inner)


@mcp.tool()
async def teach(name: str, steps: list[str], agent: str | None = None, entities: list[str] | None = None) -> dict:
    """Save HOW the user does something, step by step (a procedure/workflow), e.g.
    name='deploy cityfix', steps=['run tests','build image','deploy','verify']."""
    async def inner(session):
        return {"id": await _ingest.teach(session, name, steps, agent, entities)}
    return await _with_session(inner)


@mcp.tool()
async def expand(item_id: str) -> dict:
    """Full record for an item id from recall(): e:<n> episode, r:<n> lesson (with its evidence),
    f:<uuid> fact/preference/procedure, d:<slug> document."""
    async def inner(session):
        return await _recall_mod.expand(session, item_id) or {"error": f"not found: {item_id}"}
    return await _with_session(inner)


@mcp.tool()
async def timeline(entity: str, since: str | None = None, limit: int = 100) -> list[dict] | dict:
    """Chronological events for one entity slug, e.g. 'topic:job-search', 'company:acme'
    (slugs appear in recall() plans). `since` is an ISO date."""
    from datetime import datetime
    async def inner(session):
        rows = await _recall_mod.timeline(session, entity, datetime.fromisoformat(since) if since else None, limit)
        return [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()} for r in rows]
    return await _with_session(inner)


@mcp.tool()
async def forget(item_id: str) -> dict:
    """Permanently delete one memory item (e:/r:/f: id) — only when the user asks you to forget it
    or it is wrong."""
    async def inner(session):
        return {"deleted": await _recall_mod.forget(session, item_id)}
    return await _with_session(inner)


@mcp.resource("context://documents")
async def resource_index() -> str:
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(ContextDocument).order_by(ContextDocument.type, ContextDocument.title)
        )).scalars().all()
        return "\n".join(f"context://documents/{r.slug}\t{r.title}" for r in rows)


@mcp.resource("context://documents/{slug}")
async def resource_doc(slug: str) -> str:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(ContextDocument).where(ContextDocument.slug == slug))).scalar_one_or_none()
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
