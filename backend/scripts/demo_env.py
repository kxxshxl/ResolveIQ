"""A scratch database and API for demonstrations, so that demo data never lands in your development database.

    python scripts/demo_env.py setup   # copy the dev database to `resolveiq_demo`, migrate it, empty the request log, add synthetic drift traffic
    python scripts/demo_env.py api     # run the API on :8100 against that copy (auth off, jobs inline, real LLM if Ollama is running)
    python scripts/demo_env.py drop    # delete the scratch database

Then, in other terminals: `python scripts/seed_demo_session.py --n 30 --novel 10 --force` (real resolutions, SIMULATED feedback) and `npm run dev` in frontend/.
The development database must be seeded first (python scripts/seed.py) because the copy is made from it.
"""
from __future__ import annotations

import asyncio
import os
import runpy
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import psycopg  # noqa: E402

import app  # noqa: E402,F401  (Windows event-loop policy)
from app.core.config import get_settings  # noqa: E402

NAME = "resolveiq_demo"


def swap(url: str, db: str) -> str:
    return urlunparse(urlparse(url)._replace(path=f"/{db}"))


def setup(s) -> None:
    template = urlparse(s.database_admin_url).path.lstrip("/")
    with psycopg.connect(swap(s.database_admin_url, "postgres"), autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {NAME} WITH (FORCE)")
        c.execute(f"CREATE DATABASE {NAME} TEMPLATE {template}")      # fails if something is connected to the dev database: stop the API first
    os.environ["DATABASE_URL"], os.environ["DATABASE_ADMIN_URL"] = swap(s.database_url, NAME), swap(s.database_admin_url, NAME)
    get_settings.cache_clear()
    from app.db.migrate import migrate

    print("migrations applied:", migrate())
    with psycopg.connect(os.environ["DATABASE_ADMIN_URL"], autocommit=True) as c:
        c.execute("DELETE FROM feedback; DELETE FROM resolution_requests; DELETE FROM taxonomy_proposals; DELETE FROM drift_snapshots; DELETE FROM jobs")
    from app.db.repository import Repository
    from app.evaluation.drift_traffic import seed_traffic

    async def go():
        repo = Repository(get_settings())
        await repo.open()
        try:
            await seed_traffic(repo)
        finally:
            await repo.close()

    asyncio.run(go())
    print(f"scratch database {NAME} ready; start the API with: python scripts/demo_env.py api")


def api(s) -> None:
    os.environ.update(DATABASE_URL=swap(s.database_url, NAME), DATABASE_ADMIN_URL=swap(s.database_admin_url, NAME), API_KEYS="", RATE_LIMIT_PER_MINUTE="0",
                      JOB_EXECUTION="inline", PORT="8100", CACHE_NAMESPACE="demo", LLM_PROVIDERS=os.environ.get("LLM_PROVIDERS", "ollama"))
    get_settings.cache_clear()
    sys.argv = ["dev_server.py"]
    runpy.run_path(str(HERE / "dev_server.py"), run_name="__main__")


def drop(s) -> None:
    with psycopg.connect(swap(s.database_admin_url, "postgres"), autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {NAME} WITH (FORCE)")
    print("dropped", NAME)


if __name__ == "__main__":
    modes = {"setup": setup, "api": api, "drop": drop}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        sys.exit(__doc__)
    modes[sys.argv[1]](get_settings())
