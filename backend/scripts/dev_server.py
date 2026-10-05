"""Run the API locally (works on Windows, where uvicorn's default loop is incompatible with psycopg async)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import uvicorn  # noqa: E402

import app  # noqa: E402,F401  (sets the Windows event-loop policy)

if __name__ == "__main__":
    # loop="none": uvicorn >= 0.36 otherwise forces Windows' ProactorEventLoop, which psycopg's async mode cannot use;
    # with "none" asyncio.run uses the selector policy installed by `import app`.
    uvicorn.run("app.main:app", host="127.0.0.1", port=int(os.getenv("PORT", "8100")), loop="none")
