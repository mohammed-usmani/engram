from unittest.mock import AsyncMock, patch
from src.services.retrieval import retrieve, RetrievedDoc
from src.models import ContextDocument, DocType, Source


async def test_hybrid_retrieval(session):
    session.add_all([
        ContextDocument(
            type=DocType.SKILL, slug="skill/python", title="Python",
            tags=["skill"], content="python backend fastapi asyncio",
            sections={}, source=Source.FILE, doc_metadata={},
            embedding=[0.9] * 768,
        ),
        ContextDocument(
            type=DocType.PROJECT, slug="project/cityfix", title="CityFix",
            tags=["project"], content="civic mobile app geospatial mongodb",
            sections={}, source=Source.FILE, doc_metadata={},
            embedding=[0.1] * 768,
        ),
    ])
    await session.commit()

    with patch("src.services.retrieval.embed", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.85] * 768
        results = await retrieve("python backend", session, limit=5)

    assert len(results) >= 1
    assert all(isinstance(r, RetrievedDoc) for r in results)
    slugs = [r.slug for r in results]
    assert "skill/python" in slugs


async def test_retrieval_empty_db(session):
    with patch("src.services.retrieval.embed", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.5] * 768
        results = await retrieve("anything", session, limit=5)

    assert results == []
