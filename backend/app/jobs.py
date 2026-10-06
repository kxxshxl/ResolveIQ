"""Background job handlers, shared by the worker service (queue mode) and the API process (inline mode).

One code path means the behaviour you test inline is the behaviour the worker runs in production.
"""
from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable

from app.models.schemas import TicketIn
from app.observability import tracing
from app.observability.metrics import EVAL_RUNS, JOB_DURATION, JOBS
from app.services.container import Services

log = logging.getLogger(__name__)
Handler = Callable[[Services, dict, str], Awaitable[dict]]


async def _ingest_batch(svc: Services, payload: dict, job_id: str) -> dict:
    return await svc.ingestion.ingest_tickets_bulk([TicketIn(**t) for t in payload["tickets"]])


async def _evaluate(svc: Services, payload: dict, job_id: str) -> dict:
    from app.evaluation.runner import run_suites

    suites = payload["suites"]
    results = await run_suites(svc, suites, payload.get("max_queries"))
    for s in suites:
        EVAL_RUNS.labels(s).inc()
    await svc.repo.save_eval_run(job_id, suites, results)
    return results


async def _discover(svc: Services, payload: dict, job_id: str) -> dict:
    return await svc.discovery.run(job_id=job_id, window_days=payload.get("window_days"))


async def _drift(svc: Services, payload: dict, job_id: str) -> dict:
    return await svc.drift.run(payload.get("window_hours"), payload.get("baseline_days"), job_id)


async def _reindex(svc: Services, payload: dict, job_id: str) -> dict:
    return await svc.ingestion.reindex()


TRACE_KEY = "_trace"  # W3C trace context of the enqueuing request, stored in the job payload
HANDLERS: dict[str, Handler] = {"ingest_batch": _ingest_batch, "evaluate": _evaluate, "discover_classes": _discover, "drift_analysis": _drift, "reindex": _reindex}


async def execute(svc: Services, kind: str, payload: dict, job_id: str) -> dict:
    t0 = time.perf_counter()
    links = tracing.links_from(payload.pop(TRACE_KEY, None))
    try:
        with tracing.span(f"job.{kind}", {"resolveiq.job_id": job_id, "resolveiq.job_kind": kind}, links=links):
            result = await HANDLERS[kind](svc, payload, job_id)
        JOBS.labels(kind, "succeeded").inc()
        return result
    except Exception:
        JOBS.labels(kind, "failed").inc()
        raise
    finally:
        JOB_DURATION.labels(kind).observe(time.perf_counter() - t0)


async def run_inline(svc: Services, job_id: str, kind: str, payload: dict) -> None:
    """Single-process mode (dev/tests): same handlers, no retries."""
    try:
        await svc.repo.update_job(job_id, "running")
        await svc.repo.complete_job(job_id, await execute(svc, kind, payload, job_id))
    except Exception as exc:  # noqa: BLE001
        log.exception("inline job failed", extra={"job_id": job_id, "kind": kind})
        await svc.repo.update_job(job_id, "failed", error=f"{type(exc).__name__}: {exc}"[:500])


async def submit(svc: Services, background, kind: str, payload: dict) -> str:
    """Enqueue a job. In queue mode a worker picks it up; in inline mode it runs in this process after the response."""
    ctx = tracing.inject_context()  # empty (and the payload unchanged) unless tracing is on
    job_id = await svc.repo.create_job(kind, {**payload, TRACE_KEY: ctx} if ctx else payload)
    if svc.settings.job_execution == "inline":
        background.add_task(run_inline, svc, job_id, kind, payload)
    return job_id
