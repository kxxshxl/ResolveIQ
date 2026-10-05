"""Worker service: executes background jobs from the Postgres queue.   python -m app.worker

Separate process/container from the API so that CPU-heavy work (batch embedding, evaluation, clustering) never competes
with interactive /resolve latency, and so each tier scales independently (N API replicas, M workers).

Queue semantics: claim with FOR UPDATE SKIP LOCKED, heartbeat while running, retry with exponential backoff up to
max_attempts, recover jobs whose worker died. Handlers are idempotent (upserts / replace-pending), so at-least-once is safe.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import time
from datetime import datetime, timezone

from prometheus_client import start_http_server

import app  # noqa: F401  (Windows event-loop policy)
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.jobs import HANDLERS, execute
from app.observability.drift import drift_report, publish_gauges
from app.observability.metrics import QUEUE_DEPTH
from app.services.container import Services

log = logging.getLogger("worker")
HEARTBEAT_SECONDS = 15
DRIFT_EVERY_SECONDS = 300
_last_drift = [float("-inf")]


async def _heartbeat(svc: Services, job_id: str) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        await svc.repo.heartbeat_job(job_id)


async def run_one(svc: Services, job: dict) -> str:
    """Execute one claimed job and record the outcome. Returns the final job status."""
    hb = asyncio.create_task(_heartbeat(svc, job["job_id"]))
    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(execute(svc, job["kind"], job["payload"] or {}, job["job_id"]), svc.settings.worker_job_timeout_seconds)
        await svc.repo.complete_job(job["job_id"], result)
        log.info("job succeeded", extra={"job_id": job["job_id"], "kind": job["kind"], "seconds": round(time.perf_counter() - t0, 2)})
        return "succeeded"
    except Exception as exc:  # noqa: BLE001 - a failing job must never take the worker down
        status = await svc.repo.fail_job(job["job_id"], f"{type(exc).__name__}: {exc}", svc.settings.worker_retry_backoff_seconds)
        log.error("job failed", extra={"job_id": job["job_id"], "kind": job["kind"], "attempt": job["attempts"], "next_status": status,
                                       "error": f"{type(exc).__name__}: {exc}"[:300]})
        return status
    finally:
        hb.cancel()


async def maintenance(svc: Services, s: Settings) -> None:
    recovered = await svc.repo.requeue_stale_jobs(s.worker_stale_seconds)
    if recovered:
        log.warning("recovered stale jobs", extra={"count": recovered})
    depth = await svc.repo.queue_depth()
    for status in ("queued", "running"):
        QUEUE_DEPTH.labels(status).set(depth.get(status, 0))
    if time.monotonic() - _last_drift[0] >= DRIFT_EVERY_SECONDS:
        _last_drift[0] = time.monotonic()
        report = await drift_report(svc.repo)
        publish_gauges(report)
        for a in report["alerts"]:
            log.warning("drift alert", extra=a)
    if s.discovery_schedule_hours > 0:
        last = await svc.repo.last_job_time("discover_classes")
        if last is None or (datetime.now(timezone.utc) - last).total_seconds() > s.discovery_schedule_hours * 3600:
            job_id = await svc.repo.create_job("discover_classes", {"scheduled": True})
            log.info("scheduled discovery job enqueued", extra={"job_id": job_id})


async def main() -> None:
    s = get_settings()
    configure_logging()
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    if s.worker_metrics_port:
        start_http_server(s.worker_metrics_port)
    svc = await Services.build(s)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows event loops
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
    log.info("worker started", extra={"worker_id": worker_id, "kinds": list(HANDLERS), "poll_s": s.worker_poll_seconds})
    next_maintenance = 0.0
    try:
        while not stop.is_set():
            if time.monotonic() >= next_maintenance:
                try:
                    await maintenance(svc, s)
                except Exception:  # noqa: BLE001
                    log.exception("maintenance failed")
                next_maintenance = time.monotonic() + 30
            try:
                job = await svc.repo.claim_job(worker_id, list(HANDLERS))
            except Exception:  # noqa: BLE001 - DB blip: back off and retry rather than crash-loop
                log.exception("claim failed")
                job = None
            if job:
                await run_one(svc, job)  # finish the current job before honouring a stop request
                continue
            try:
                await asyncio.wait_for(stop.wait(), s.worker_poll_seconds)
            except asyncio.TimeoutError:
                pass
    finally:
        log.info("worker stopping")
        await svc.close()


if __name__ == "__main__":
    asyncio.run(main())
