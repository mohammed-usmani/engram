from unittest.mock import AsyncMock, patch
from sqlalchemy import select
from src.services.embeddings import embed_all_documents
from src.models import ContextDocument, DocType, Source


async def test_embed_all_documents(session):
    session.add_all([
        ContextDocument(type=DocType.SKILL, slug="skill/a", title="A",
                        tags=[], content="python backend", sections={},
                        source=Source.FILE, doc_metadata={}),
        ContextDocument(type=DocType.SKILL, slug="skill/b", title="B",
                        tags=[], content="javascript frontend", sections={},
                        source=Source.FILE, doc_metadata={}),
    ])
    await session.commit()

    with patch("src.services.embeddings.embed", new_callable=AsyncMock, return_value=[0.1] * 768) as mock:
        summary = await embed_all_documents(session)

    assert summary["embedded"] == 2
    assert summary["errors"] == 0
    assert mock.call_count == 2

    rows = (await session.execute(select(ContextDocument))).scalars().all()
    for r in rows:
        assert r.embedding is not None
        assert len(r.embedding) == 768


async def test_embed_only_missing(session):
    session.add_all([
        ContextDocument(type=DocType.SKILL, slug="skill/a", title="A",
                        tags=[], content="x", sections={},
                        source=Source.FILE, doc_metadata={}, embedding=[0.5] * 768),
        ContextDocument(type=DocType.SKILL, slug="skill/b", title="B",
                        tags=[], content="y", sections={},
                        source=Source.FILE, doc_metadata={}),
    ])
    await session.commit()

    with patch("src.services.embeddings.embed", new_callable=AsyncMock, return_value=[0.2] * 768) as mock:
        summary = await embed_all_documents(session, force=False)

    assert summary["embedded"] == 1
    mock.assert_called_once()
