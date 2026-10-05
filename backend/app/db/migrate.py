"""Minimal forward-only migration runner. Run with the OWNER role: python -m app.db.migrate

Applies app/db/migrations/*.sql in order, records them in schema_migrations, then grants the
least-privilege runtime role (DML only, no DDL) on every table.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg

from app.core.config import get_settings
from app.core.logging import configure_logging

log = logging.getLogger("migrate")
MIGRATIONS = Path(__file__).parent / "migrations"


def _app_role(url: str) -> str:
    return urlparse(url).username or ""


def migrate(admin_url: str | None = None, app_url: str | None = None) -> list[str]:
    s = get_settings()
    admin_url = admin_url or s.database_admin_url
    app_role = _app_role(app_url or s.database_url)
    applied: list[str] = []
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY, applied_at timestamptz DEFAULT now())")
        done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(MIGRATIONS.glob("*.sql")):
            if path.name in done:
                continue
            sql = path.read_text(encoding="utf-8").replace("{{EMBEDDING_DIM}}", str(s.embedding_dim))
            with conn.transaction():
                conn.execute(sql)
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,))
            applied.append(path.name)
            log.info("applied migration", extra={"migration": path.name})
        if app_role and re.fullmatch(r"[A-Za-z0-9_]+", app_role):
            exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (app_role,)).fetchone()
            if exists:
                conn.execute(f"GRANT USAGE ON SCHEMA public TO {app_role}")
                conn.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {app_role}")
                conn.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {app_role}")
                conn.execute(f"REVOKE ALL ON schema_migrations FROM {app_role}")
    return applied


if __name__ == "__main__":
    configure_logging()
    done = migrate()
    print(f"migrations applied: {done or 'none (up to date)'}")
    sys.exit(0)
