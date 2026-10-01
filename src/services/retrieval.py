from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import ContextDocument
from src.services.ollama_client import embed

log = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "as", "is", "are", "be", "we", "you", "our", "your", "it",
    "this", "that", "will", "have", "has", "had", "not", "no", "but", "can",
    "what", "which", "who", "how", "when", "where", "do", "does", "did",
    "my", "me", "his", "her", "i", "he", "she", "they", "them",
    "was", "were", "been", "being", "am",
    "tell", "about", "built", "made", "used", "using",
}


@dataclass
class RetrievedDoc:
    slug: str
    title: str
    type: str
    content: str
    score: float
    source_method: str  # "vector" | "fts" | "both"


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float64)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return [1.0] * len(scores)
    return ((arr - mn) / (mx - mn)).tolist()


async def retrieve(
    query: str,
    session: AsyncSession,
    limit: int = 8,
    vector_weight: float = 0.5,
    fts_weight: float = 0.5,
) -> list[RetrievedDoc]:
    query_embedding = await embed(query)

    # Vector search
    vec_stmt = (
        select(
            ContextDocument,
            (1 - ContextDocument.embedding.cosine_distance(query_embedding)).label("vec_score"),
        )
        .where(ContextDocument.embedding.isnot(None))
        .order_by(ContextDocument.embedding.cosine_distance(query_embedding))
        .limit(limit * 3)
    )
    vec_rows = (await session.execute(vec_stmt)).all()

    # FTS search — extract meaningful terms, use OR for broad recall
    terms = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{1,30}", query.lower()) if w not in _STOPWORDS and len(w) >= 2]
    fts_rows = []
    if terms:
        tsq_str = " | ".join(terms)
        tsq = func.to_tsquery("english", tsq_str)
        fts_stmt = (
            select(
                ContextDocument,
                func.ts_rank(ContextDocument.search_vector, tsq).label("fts_score"),
            )
            .where(ContextDocument.search_vector.op("@@")(tsq))
            .order_by(func.ts_rank(ContextDocument.search_vector, tsq).desc())
            .limit(limit * 3)
        )
        fts_rows = (await session.execute(fts_stmt)).all()

    # Merge
    slug_to_vec: dict[str, tuple[ContextDocument, float]] = {}
    slug_to_fts: dict[str, tuple[ContextDocument, float]] = {}

    for doc, score in vec_rows:
        slug_to_vec[doc.slug] = (doc, float(score))
    for doc, score in fts_rows:
        slug_to_fts[doc.slug] = (doc, float(score))

    all_slugs = set(slug_to_vec.keys()) | set(slug_to_fts.keys())
    if not all_slugs:
        return []

    vec_norm = dict(zip(slug_to_vec.keys(), _normalize([v[1] for v in slug_to_vec.values()]))) if slug_to_vec else {}
    fts_norm = dict(zip(slug_to_fts.keys(), _normalize([v[1] for v in slug_to_fts.values()]))) if slug_to_fts else {}

    results: list[RetrievedDoc] = []
    for slug in all_slugs:
        doc = (slug_to_vec.get(slug) or slug_to_fts.get(slug))[0]
        v = vec_norm.get(slug, 0.0) * vector_weight
        f = fts_norm.get(slug, 0.0) * fts_weight
        combined = v + f

        if slug in slug_to_vec and slug in slug_to_fts:
            method = "both"
        elif slug in slug_to_vec:
            method = "vector"
        else:
            method = "fts"

        results.append(RetrievedDoc(
            slug=doc.slug, title=doc.title, type=doc.type.value,
            content=doc.content, score=combined, source_method=method,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
