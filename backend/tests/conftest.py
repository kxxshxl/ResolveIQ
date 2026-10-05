"""Test harness. DB-backed tests use a dedicated `resolveiq_test` database on the docker-compose Postgres
(`docker compose up -d db redis`); they are skipped, not failed, when it is unreachable. The LLM is the
deterministic MockProvider unless a test injects its own."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402,F401  (Windows event-loop policy)
from app.core.config import Settings, get_settings  # noqa: E402

TEST_DB = "resolveiq_test"


def _swap_db(url: str, db: str) -> str:
    p = urlparse(url)
    return urlunparse(p._replace(path=f"/{db}"))


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    base = get_settings()
    return Settings(
        database_url=_swap_db(base.database_url, TEST_DB), database_admin_url=_swap_db(base.database_admin_url, TEST_DB),
        cache_namespace=f"test-{uuid.uuid4().hex[:8]}", llm_providers="mock", api_keys="", rate_limit_per_minute=0, classifier_llm_fallback=False, reranker_enabled=True,
        affect_enabled=False)  # the 1.7 GB NLI model has its own tests (tests/test_affect.py)


@pytest.fixture(scope="session")
def test_database(test_settings):
    admin_root = _swap_db(test_settings.database_admin_url, "postgres")
    try:
        with psycopg.connect(admin_root, autocommit=True, connect_timeout=3) as c:
            c.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
            c.execute(f"CREATE DATABASE {TEST_DB}")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable ({type(exc).__name__}); run `docker compose up -d db redis`")
    from app.db.migrate import migrate

    migrate(test_settings.database_admin_url, test_settings.database_url)
    return TEST_DB


@pytest.fixture(scope="session")
def client(test_settings, test_database):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.scripts_seed_bridge import seed_corpus

    application = create_app(test_settings)
    with TestClient(application) as c:
        c.portal.call(seed_corpus, application.state.services)
        yield c


@pytest.fixture(scope="session")
def svc(client):
    return client.app.state.services


@pytest.fixture(scope="session")
def run(client):
    """Run a coroutine function on the app's event loop (where the DB pool lives)."""
    return lambda fn, *a, **kw: client.portal.call(lambda: fn(*a, **kw))


@pytest.fixture(scope="session")
def embedder(test_settings):
    from app.services.embedding import EmbeddingService

    e = EmbeddingService(test_settings)
    e.warmup()
    return e
