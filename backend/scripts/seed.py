"""Load taxonomy + processed corpus into Postgres through the SAME ingestion pipeline the API uses.

  python scripts/seed.py              # (re)seed; idempotent upserts
  python scripts/seed.py --if-empty   # no-op when the corpus already has tickets (used by docker compose)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.classification.taxonomy import TaxonomyService  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.models.schemas import ArticleIn, TicketIn  # noqa: E402
from app.services.container import Services  # noqa: E402

log = logging.getLogger("seed")


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


async def seed(include_evolving: bool = False, if_empty: bool = False, svc: Services | None = None) -> dict:
    s = get_settings()
    own = svc is None
    if own:
        svc = await Services.build(s)
    try:
        if if_empty and (await svc.repo.counts())["tickets"] > 0:
            log.info("corpus already present; skipping seed")
            return {"skipped": True}
        t0 = time.perf_counter()
        if await svc.repo.taxonomy_version() == 0:
            await svc.repo.add_taxonomy_labels(TaxonomyService.load_seed(s.data_dir / "taxonomy.json"), "initial taxonomy v1")
        await svc.taxonomy.refresh()
        proc = s.data_dir / "processed"
        tickets = [TicketIn(**r) for r in read(proc / "tickets.jsonl")]
        res = await svc.ingestion.ingest_tickets_bulk(tickets, source="historical")
        n_art = 0
        for a in read(proc / "articles.jsonl"):
            status = a.pop("status", "active")
            await svc.ingestion.ingest_article(ArticleIn(**a))
            if status == "deprecated":
                await svc.repo.deprecate_article(a["article_id"], "superseded by newer article")
            n_art += 1
        await svc.repo.bump_corpus_version()
        await svc.ingestion.refresh_gauges()
        out = {**res, "articles": n_art, "seconds": round(time.perf_counter() - t0, 1)}
        log.info("seed complete", extra={"result": out})
        return out
    finally:
        if own:
            await svc.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-empty", action="store_true")
    args = ap.parse_args()
    configure_logging()
    print(asyncio.run(seed(if_empty=args.if_empty)))
