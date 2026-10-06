"""Latency of /resolve through the frontend while the LLM is unreachable: the first request per replica pays the LLM time budget,
then the circuit breaker opens and the evidence-only fallback answers at pipeline speed.

    python k8s/local/degraded_latency.py [--n 6]
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_test import COMPLAINT, Ingress, call, secret  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    n = ap.parse_args().n
    key = {"X-API-Key": secret("API_KEYS").split(",")[0]}
    rows = []
    with Ingress() as fe:
        for i in range(n):
            s, body, _, ms = call(fe.url, "POST", "/api/v1/resolve", {"complaint": f"{COMPLAINT} (case ref {uuid.uuid4().hex[:8]})"}, key)
            rows.append((s, body.get("status"), body.get("generator"), ms))
            print(f"request {i + 1}: HTTP {s} status={body.get('status')} generator={body.get('generator')} in {ms / 1000:.1f} s")
    ok = all(r[0] == 200 and r[1] == "degraded" and r[2] == "extractive" for r in rows)
    print("all answers were evidence-only (degraded/extractive) with HTTP 200:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
