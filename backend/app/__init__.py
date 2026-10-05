import asyncio
import sys

# psycopg's async driver cannot run on Windows' default ProactorEventLoop.
if sys.platform == "win32":  # pragma: no cover
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
