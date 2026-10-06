"""Put synthetic traffic into the database so the Drift monitoring tab has something to show.

    python scripts/seed_drift_traffic.py            # baseline (days 2-12), recent day, plus a synthetic smart-home-hub incident
    python scripts/seed_drift_traffic.py --clear    # remove exactly those rows again

The rows are marked (trace_id = 'synthetic-drift-demo') and the script refuses to add them to a database that already holds other logged
requests unless --force. Use a scratch database: this is for demonstration, never for production.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402,F401  (Windows event-loop policy)
from app.core.config import get_settings  # noqa: E402
from app.db.repository import Repository  # noqa: E402
from app.evaluation.drift_traffic import MARKER, clear_traffic, seed_traffic  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true", help="delete the synthetic rows")
    ap.add_argument("--force", action="store_true", help="add the rows even though other logged requests exist")
    a = ap.parse_args()
    repo = Repository(get_settings())
    await repo.open()
    try:
        if a.clear:
            print(f"removed {await clear_traffic(repo)} synthetic requests")
            return 0
        others = (await repo._fetch("SELECT count(*) AS n FROM resolution_requests WHERE trace_id IS DISTINCT FROM %s", (MARKER,)))[0]["n"]
        if others and not a.force:
            print(f"refusing: {others} real logged requests exist (they would sit in the same windows). Use a scratch database, or --force.")
            return 2
        await clear_traffic(repo)
        ids = await seed_traffic(repo)
        print(f"seeded 170 baseline + 70 recent + {len(ids['incident'])} incident requests. Run an analysis from the Drift monitoring tab "
              "(or POST /api/v1/monitoring/drift/run); remove them with --clear.")
        return 0
    finally:
        await repo.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
