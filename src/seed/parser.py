from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.models import DocType

log = logging.getLogger(__name__)

_DIR_TO_TYPE = {
    "resume": DocType.RESUME,
    "projects": DocType.PROJECT,
    "skills": DocType.SKILL,
    "work": DocType.EXPERIENCE,
    "education": DocType.EDUCATION,
    "achievements": DocType.ACHIEVEMENT,
    "certifications": DocType.CERTIFICATION,
}

_SECTION_HEADER_RE = re.compile(r"^([A-Z][A-Za-z0-9 &/'\-]{1,60}):\s*$")
_KV_LINE_RE = re.compile(r"^([A-Z][A-Za-z0-9 &/'\-]{1,60}):\s+(.+?)\s*$")

_TITLE_KEYS = ("Title", "Skill Name", "Name")


def parse_text(raw: str) -> tuple[dict[str, str], dict[str, str]]:
    """Split document text into `Key: value` header fields and `Section:` blocks."""
    lines = raw.splitlines()

    headers: dict[str, str] = {}
    sections: dict[str, str] = {}
    current_section: str | None = None
    current_buf: list[str] = []

    def _flush() -> None:
        nonlocal current_section, current_buf
        if current_section is not None:
            sections[current_section] = "\n".join(current_buf).strip()
        current_section = None
        current_buf = []

    in_header_zone = True

    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            if current_section is not None:
                current_buf.append("")
            continue

        m_section = _SECTION_HEADER_RE.match(stripped)
        if m_section:
            _flush()
            current_section = m_section.group(1).strip()
            in_header_zone = False
            continue

        if in_header_zone:
            m_kv = _KV_LINE_RE.match(stripped)
            if m_kv:
                headers[m_kv.group(1).strip()] = m_kv.group(2).strip()
                continue
            in_header_zone = False

        if current_section is not None:
            current_buf.append(line)

    _flush()
    return headers, sections


def derive_fields(raw: str) -> tuple[dict[str, str], dict[str, str]]:
    """(sections, metadata) for a document's text — used whenever a document is saved."""
    headers, sections = parse_text(raw)
    return sections, {k: v for k, v in headers.items() if k not in _TITLE_KEYS}


def parse_file(path: Path) -> dict[str, Any]:
    parent = path.parent.name
    if parent not in _DIR_TO_TYPE:
        raise ValueError(f"Unknown context directory: {parent}")
    doc_type = _DIR_TO_TYPE[parent]

    raw = path.read_text(encoding="utf-8")
    headers, sections = parse_text(raw)

    title: str | None = None
    for k in _TITLE_KEYS:
        if k in headers:
            title = headers[k]
            break
    if not title:
        title = path.stem.replace("_", " ").replace("-", " ").strip().title()

    slug = f"{doc_type.value}/{path.stem.lower()}"

    tag_candidates: list[str] = [doc_type.value]
    for k in ("Category", "Level", "Type"):
        v = headers.get(k)
        if v:
            tag_candidates.append(v)
    tags = sorted({t.strip().lower() for t in tag_candidates if t.strip()})

    metadata = {k: v for k, v in headers.items() if k not in _TITLE_KEYS}

    return {
        "type": doc_type,
        "slug": slug,
        "title": title,
        "tags": tags,
        "content": raw,
        "sections": sections,
        "metadata": metadata,
    }
