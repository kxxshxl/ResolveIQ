"""Availability during a rolling restart of the backend Deployment.

    python k8s/local/rollout_availability.py [--rps 1.5]

Sends a steady stream of GET /api/v1/search requests (a cheap, real, authenticated call that hits Postgres/pgvector and the embedder)
through the Ingress (-> frontend Service -> backend Service) while `kubectl rollout restart deployment/backend` replaces every pod, and reports how many requests failed.
The manifests promise maxUnavailable: 0, readiness-gated traffic and a 60 s termination grace period; this checks that on a real cluster.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_test import CTX, NS, Forward, Ingress, call, secret  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=float, default=1.5, help="keep below the per-key rate limit (RATE_LIMIT_PER_MINUTE=120 -> 2/s)")
    ap.add_argument("--port-forward", action="store_true", help="use kubectl port-forward to one frontend pod instead of the Ingress")
    a = ap.parse_args()
    key = {"X-API-Key": secret("API_KEYS").split(",")[0]}
    results: list[tuple[float, int, float]] = []
    stop = threading.Event()
    with (Forward("svc/frontend", 18080, 80) if a.port_forward else Ingress()) as fe:
        def load():
            while not stop.is_set():
                t0 = time.time()
                try:
                    s, _, _, ms = call(fe.url, "GET", "/api/v1/search?q=broadband+keeps+dropping+every+evening&strategy=dense&limit=3", headers=key, timeout=30)
                except Exception:  # noqa: BLE001 - a refused / reset connection is a failed request
                    s, ms = 0, (time.time() - t0) * 1000
                results.append((t0, s, ms))
                time.sleep(max(0.0, 1 / a.rps - (time.time() - t0)))

        th = threading.Thread(target=load, daemon=True)
        th.start()
        time.sleep(5)
        t_start = time.time()
        subprocess.run(["kubectl", "--context", CTX, "-n", NS, "rollout", "restart", "deployment/backend"], check=True, capture_output=True)
        subprocess.run(["kubectl", "--context", CTX, "-n", NS, "rollout", "status", "deployment/backend", "--timeout=600s"], check=True, capture_output=True)
        t_end = time.time()
        time.sleep(5)
        stop.set()
        th.join()
    during = [r for r in results if t_start <= r[0] <= t_end]
    bad = [r for r in results if r[1] != 200]
    lat = sorted(r[2] for r in results if r[1] == 200)
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else float("nan")  # noqa: E731
    print(f"rolling restart took {t_end - t_start:.0f} s; {len(results)} requests total, {len(during)} during the rollout")
    print(f"non-200 responses: {len(bad)} {sorted({r[1] for r in bad})}; latency of successful requests: P50 {p(0.5):.0f} ms, P95 {p(0.95):.0f} ms, max {lat[-1] if lat else 0:.0f} ms")
    for t0, s, ms in bad[:10]:
        print(f"  failed at +{t0 - t_start:.1f} s: HTTP {s} after {ms:.0f} ms")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
