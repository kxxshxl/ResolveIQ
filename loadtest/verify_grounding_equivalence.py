"""Check that the evidence-embedding cache and de-duplication in citation validation do not change any grounding result.

Runs every evaluation query through the real pipeline twice, once with the new `embed_cached` and once with the original computation (one `encode_sync` of
every step and every repeated evidence unit), and compares each step's grounding score and flag and the validation verdict. Read-only: nothing is persisted.

    python loadtest/verify_grounding_equivalence.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
import app  # noqa: E402,F401
import numpy as np  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.models.schemas import ResolveRequest  # noqa: E402
from app.services.container import Services  # noqa: E402


async def main() -> None:
    svc = await Services.build(get_settings())
    try:
        queries = [json.loads(l)["text"] for l in (REPO / "data" / "eval" / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
        embedder = svc.embedder
        new_embed = embedder.embed_cached

        async def old_embed(texts):  # the original computation
            return await asyncio.to_thread(embedder.encode_sync, list(texts))

        steps = flips = verdict_changes = 0
        worst = 0.0
        for text in queries:
            embedder._units.clear()
            embedder.embed_cached = old_embed
            old = await svc.resolution.resolve(ResolveRequest(complaint=text), generate=False, persist=False, use_cache=False)
            embedder.embed_cached = new_embed
            new = await svc.resolution.resolve(ResolveRequest(complaint=text), generate=False, persist=False, use_cache=False)
            again = await svc.resolution.resolve(ResolveRequest(complaint=text), generate=False, persist=False, use_cache=False)  # served from the LRU
            for a in (new, again):
                for so, sn in zip(old.resolution.steps, a.resolution.steps):
                    steps += 1
                    flips += so.grounded != sn.grounded
                    worst = max(worst, abs((so.grounding_score or 0) - (sn.grounding_score or 0)))
                verdict_changes += (old.validation.valid != a.validation.valid) or (old.status != a.status)
        print(f"{len(queries)} queries, {steps} step comparisons (fresh and cache-served): grounded-flag changes = {flips}, "
              f"validation/status changes = {verdict_changes}, max |grounding score difference| = {worst:.4f}")
        sys.exit(1 if flips or verdict_changes else 0)
    finally:
        await svc.close()


if __name__ == "__main__":
    asyncio.run(main())
