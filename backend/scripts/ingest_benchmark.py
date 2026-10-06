"""Database-layer comparison of ticket ingestion: one transaction per ticket (upsert_ticket, the old batch path) vs one batched transaction (bulk_upsert_tickets).

    python scripts/ingest_benchmark.py [--n 2000]

Uses a throw-away copy of the application database (so the schema, extensions and grants are the real ones), random unit embeddings (the embedding model is NOT
part of this comparison: it measures what the database layer costs) and the same rows for both paths. Prints a JSON summary; writes nothing else.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402,F401  (Windows event-loop policy)
from app.core.config import get_settings  # noqa: E402
from app.db.repository import Repository  # noqa: E402

DB = "resolveiq_ingest_bench"


def swap(url: str, db: str) -> str:
    return urlunparse(urlparse(url)._replace(path=f"/{db}"))


def rows(n: int, tag: str, rng) -> tuple[list[dict], np.ndarray]:
    out = [{"ticket_id": f"BENCH-{tag}-{i}", "complaint_text": f"Benchmark complaint {tag} number {i} about the broadband dropping in the evening {uuid.uuid4().hex[:8]}",
            "intent": "broadband_disconnection", "product": "broadband", "severity": "medium", "sentiment": "neutral",
            "resolution_steps": ["Check the line statistics.", "Apply the stable line profile.", "Run a 72 hour monitored test."], "resolution_summary": f"Profile applied {i}",
            "resolved_at": "2026-01-01T00:00:00+00:00", "metadata": {"bench": tag}, "taxonomy_version": 1, "source": "bench"} for i in range(n)]
    X = rng.normal(size=(n, 384)).astype(np.float32)
    return out, X / np.linalg.norm(X, axis=1, keepdims=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    n = ap.parse_args().n
    s = get_settings()
    with psycopg.connect(swap(s.database_admin_url, "postgres"), autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
        c.execute(f"CREATE DATABASE {DB} TEMPLATE {urlparse(s.database_admin_url).path.lstrip('/')}")
    try:
        s.database_url = swap(s.database_url, DB)
        repo = Repository(s)
        await repo.open()
        rng = np.random.default_rng(1)
        a, Xa = rows(n, "loop", rng)
        b, Xb = rows(n, "bulk", rng)
        t = time.perf_counter()
        for r, v in zip(a, Xa):
            await repo.upsert_ticket({**r, "steps": None} | {"resolution_steps": r["resolution_steps"]}, v, "bench-model")
        loop_s = time.perf_counter() - t
        t = time.perf_counter()
        created, updated = await repo.bulk_upsert_tickets(b, Xb, "bench-model")
        bulk_s = time.perf_counter() - t
        t = time.perf_counter()
        c2, u2 = await repo.bulk_upsert_tickets(b, Xb, "bench-model")        # idempotent re-run: everything is an update
        again_s = time.perf_counter() - t
        n_rows = (await repo._fetch("SELECT count(*) AS n FROM tickets WHERE ticket_id LIKE 'BENCH-%%'"))[0]["n"]
        n_emb = (await repo._fetch("SELECT count(*) AS n FROM ticket_embeddings e JOIN tickets d ON d.id = e.ticket_pk WHERE d.ticket_id LIKE 'BENCH-%%'"))[0]["n"]
        await repo.close()
        print(json.dumps({"tickets_per_path": n, "per_ticket_transactions": {"seconds": round(loop_s, 2), "tickets_per_second": round(n / loop_s)},
                          "one_batched_transaction": {"seconds": round(bulk_s, 2), "tickets_per_second": round(n / bulk_s), "created": created, "updated": updated},
                          "speedup": round(loop_s / bulk_s, 1), "idempotent_rerun": {"seconds": round(again_s, 2), "created": c2, "updated": u2},
                          "rows_written": n_rows, "embeddings_written": n_emb, "note": "database layer only (random embeddings); the real ingestion also pays for embedding the text"}, indent=2))
    finally:
        with psycopg.connect(swap(s.database_admin_url, "postgres"), autocommit=True) as c:
            c.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")


if __name__ == "__main__":
    asyncio.run(main())
