"""Seeds the test database using the production seed script (same ingestion path)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from seed import seed  # noqa: E402


async def seed_corpus(svc) -> dict:
    return await seed(svc=svc)
