import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import text
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

load_dotenv()

from src.database import get_session  # noqa: E402

# Use TEST_DATABASE_URL if set; otherwise derive from CONNECTION_STRING by swapping the db name.
_prod_url = os.getenv("CONNECTION_STRING", "")
_default_test = (
    _prod_url + "_test" if _prod_url else ""
)
TEST_DB_URL = os.getenv("TEST_DATABASE_URL") or _default_test
if not TEST_DB_URL:
    raise RuntimeError("Set TEST_DATABASE_URL or CONNECTION_STRING")

# Point the app at an empty data dir during tests so lifespan bootstrap doesn't seed real files.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="ctx_test_data_"))


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine(TEST_DB_URL, future=True)
    from src.models import Base
    import src.memory.models  # noqa: F401  registers memory tables

    # CREATE EXTENSION requires superuser and cannot run inside a transaction block;
    # use a raw autocommit connection so a permission error is isolated.
    async with eng.connect() as raw_conn:
        await raw_conn.execution_options(isolation_level="AUTOCOMMIT")
        try:
            await raw_conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            pass  # already exists or insufficient privilege — vector must be pre-installed

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("""
            CREATE OR REPLACE FUNCTION context_documents_tsv_update() RETURNS trigger AS $$
            BEGIN
              NEW.search_vector :=
                setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(NEW.content, '')), 'B');
              RETURN NEW;
            END
            $$ LANGUAGE plpgsql;
        """))
        await conn.execute(text("""
            CREATE TRIGGER context_documents_tsv_trg
            BEFORE INSERT OR UPDATE ON context_documents
            FOR EACH ROW EXECUTE FUNCTION context_documents_tsv_update();
        """))
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as s:
        yield s


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent.parent / "data" / "contexts"


@pytest_asyncio.fixture
async def mem0_store(monkeypatch, tmp_path):
    """mem0 pointed at the test DB in a throwaway collection (and a throwaway history file)."""
    from src.memory import facts
    monkeypatch.setenv("MEM0_HISTORY_DB", str(tmp_path / "mem0_history.db"))
    monkeypatch.setenv("MEM0_CONNECTION", TEST_DB_URL.replace("+asyncpg", ""))
    monkeypatch.setenv("MEM0_COLLECTION", "mem0_test")
    facts._reset()
    m = await facts.store()
    await facts.wipe()
    yield m
    await facts.wipe()
    facts._reset()


@pytest_asyncio.fixture
async def client(engine, mem0_store, monkeypatch):
    from main import app
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def _session():
        async with Session() as s:
            yield s
    app.dependency_overrides[get_session] = _session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _no_local_auth(monkeypatch):
    """Tests don't inherit the machine's ADMIN_TOKEN / PUBLIC_URL from .env."""
    from src.config import settings
    monkeypatch.setattr(settings, "admin_token", None)
    monkeypatch.setattr(settings, "public_url", None)
