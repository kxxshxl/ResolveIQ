"""A deliberately broken LLM endpoint, for testing how ResolveIQ behaves when the model hangs.

`hang`: accepts connections and never answers (the client only gets out by timing out), which is how a wedged Ollama or a saturated
GPU looks from the API. Usage: python loadtest/stub_llm.py --port 11999
"""
from __future__ import annotations

import argparse
import asyncio


async def _hang(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        await asyncio.sleep(3600)  # never reply; closed by the client when its timeout fires
    finally:
        writer.close()


async def main(port: int) -> None:
    server = await asyncio.start_server(_hang, "127.0.0.1", port)
    print(f"stub LLM (hang mode) listening on 127.0.0.1:{port}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11999)
    asyncio.run(main(ap.parse_args().port))
