"""REST endpoints. Routes stay thin: validation in, service call, serialisation out."""
from __future__ import annotations

import asyncio
import hmac
import time
import uuid
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app import jobs
from app.api.security import get_services, rate_limit
from app.core.errors import ConflictError, NotFoundError, RequestTimeout, ValidationFailure
from app.models.schemas import (ArticleIn, BatchIngestRequest, EvaluateRequest, FeedbackIn, IngestResult, ResolveRequest,
                                ResolveResponse, TaxonomyLabelIn, TicketIn)
from app.observability import tracing
from app.observability.drift import drift_report
from app.retrieval.service import STRATEGIES
from app.services.container import Services

ops = APIRouter(tags=["ops"])
api = APIRouter(prefix="/api/v1", dependencies=[Depends(rate_limit)])


# ------------------------------------------------------------------ ops
@ops.get("/health", summary="Liveness probe (process is up)")
async def health():
    return {"status": "ok"}


@ops.get("/health/ready", summary="Readiness probe (dependencies reachable, models loaded)")
async def ready(svc: Services = Depends(get_services)):
    checks: dict[str, object] = {}
    ok = True
    try:
        await asyncio.wait_for(svc.repo.ping(), 3)
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"], ok = f"unavailable: {type(exc).__name__}", False
    checks["models"] = "loaded" if svc.models_ready else "loading"
    ok = ok and svc.models_ready
    checks["redis"] = "ok" if await svc.cache.ping() else "unavailable (degraded: in-process cache)"
    try:
        c = await svc.repo.counts()
        checks["corpus"] = c
        if c["tickets"] + c["articles"] == 0:
            checks["corpus_warning"] = "empty corpus: run scripts/seed.py"
    except Exception:  # noqa: BLE001
        ok = False
    checks["affect_model"] = "loaded" if svc.affect.available else "unavailable (rule/kNN fallback for severity/sentiment)"
    try:
        checks["jobs"] = {"mode": svc.settings.job_execution, **await svc.repo.queue_depth()}
    except Exception:  # noqa: BLE001
        pass
    checks["llm"] = await svc.llm.health() if svc.llm.available else "none configured (evidence-only mode)"
    checks["tracing"] = tracing.status()
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503)


@ops.get("/metrics", include_in_schema=False)
async def metrics(request: Request, svc: Services = Depends(get_services)):
    token = svc.settings.metrics_token
    if token and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
        raise HTTPException(status_code=401, detail="metrics token required")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ------------------------------------------------------------------ resolve / search
@api.post("/resolve", response_model=ResolveResponse, summary="Parse a complaint, retrieve evidence, return a cited resolution")
async def resolve(req: ResolveRequest, svc: Services = Depends(get_services)):
    try:
        return await asyncio.wait_for(svc.resolution.resolve(req), svc.settings.request_timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise RequestTimeout(f"resolution exceeded {svc.settings.request_timeout_seconds:.0f}s timeout") from exc


@api.get("/search", summary="Search tickets / KB articles with a chosen retrieval strategy")
async def search(
    q: str = Query(..., min_length=3, max_length=1000),
    source: Literal["both", "ticket", "article"] = "both",
    strategy: str | None = Query(None, description=f"one of {STRATEGIES}"),
    limit: int = Query(5, ge=1, le=50),
    offset: int = Query(0, ge=0, le=200),
    intent: str | None = None,
    product: str | None = None,
    svc: Services = Depends(get_services),
):
    strategy = strategy or svc.settings.default_retrieval_strategy
    if strategy not in STRATEGIES:
        raise ValidationFailure(f"unknown strategy '{strategy}'; choose from {list(STRATEGIES)}")
    t0 = time.perf_counter()
    filters = {k: v for k, v in (("intent", intent), ("product", product)) if v}
    ctx = await svc.retrieval.make_context(q, filters or None)
    out: dict = {"query": q, "strategy": strategy, "limit": limit, "offset": offset}
    if source in ("both", "ticket"):
        out["tickets"] = await svc.retrieval.search(ctx, "ticket", strategy, limit, offset)
    if source in ("both", "article"):
        out["articles"] = await svc.retrieval.search(ctx, "article", strategy, limit, offset)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


# ------------------------------------------------------------------ ingestion
@api.post("/ingest/ticket", response_model=IngestResult, status_code=201, summary="Ingest one resolved ticket (searchable immediately)")
async def ingest_ticket(t: TicketIn, svc: Services = Depends(get_services)):
    return await svc.ingestion.ingest_ticket(t)


@api.post("/ingest/article", response_model=IngestResult, status_code=201, summary="Ingest / update one KB article")
async def ingest_article(a: ArticleIn, svc: Services = Depends(get_services)):
    return await svc.ingestion.ingest_article(a)


@api.post("/ingest/tickets/batch", status_code=202, summary="Asynchronous batch ingestion (returns a job id)")
async def ingest_batch(body: BatchIngestRequest, bg: BackgroundTasks, svc: Services = Depends(get_services)):
    job_id = await jobs.submit(svc, bg, "ingest_batch", {"tickets": [t.model_dump(mode="json") for t in body.tickets]})
    return {"job_id": job_id, "status": "queued", "poll": f"/api/v1/jobs/{job_id}"}


@api.get("/jobs/{job_id}", summary="Background job status")
async def get_job(job_id: str, svc: Services = Depends(get_services)):
    try:
        uuid.UUID(job_id)
    except ValueError as exc:
        raise ValidationFailure("job_id must be a UUID") from exc
    job = await svc.repo.get_job(job_id)
    if not job:
        raise NotFoundError("job not found")
    return job


class DeprecateBody(BaseModel):
    reason: str = Field(..., min_length=3, max_length=300)


@api.post("/articles/{article_id}/deprecate", summary="Mark a KB article stale; it stops being retrieved immediately")
async def deprecate_article(article_id: str, body: DeprecateBody, svc: Services = Depends(get_services)):
    if not await svc.repo.deprecate_article(article_id, body.reason):
        raise NotFoundError(f"article '{article_id}' not found")
    return {"article_id": article_id, "status": "deprecated", "corpus_version": await svc.repo.bump_corpus_version()}


# ------------------------------------------------------------------ browse (pagination)
@api.get("/tickets", summary="Paginated ticket listing")
async def list_tickets(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), intent: str | None = None,
                       svc: Services = Depends(get_services)):
    rows, total = await svc.repo.list_docs("ticket", limit, offset, intent)
    return {"total": total, "limit": limit, "offset": offset, "items": rows}


@api.get("/articles", summary="Paginated KB article listing")
async def list_articles(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), intent: str | None = None,
                        svc: Services = Depends(get_services)):
    rows, total = await svc.repo.list_docs("article", limit, offset, intent)
    return {"total": total, "limit": limit, "offset": offset, "items": rows}


# ------------------------------------------------------------------ taxonomy
@api.get("/taxonomy", summary="Current label taxonomy (versioned)")
async def get_taxonomy(svc: Services = Depends(get_services)):
    tax = svc.taxonomy.current
    return {"version": tax.version,
            "labels": {d: [{"id": l.label_id, "description": l.description, "team": l.team} for l in labs.values()]
                       for d, labs in tax.labels.items()}}


@api.post("/taxonomy/labels", status_code=201, summary="Add a new class (intent/product/...) without code changes")
async def add_label(body: TaxonomyLabelIn, svc: Services = Depends(get_services)):
    if svc.taxonomy.current.has(body.dimension, body.label_id):
        raise ConflictError(f"{body.dimension} '{body.label_id}' already exists")
    tax = await svc.taxonomy.add_label(body.dimension, body.label_id, body.description, body.keywords, body.examples, body.team)
    return {"dimension": body.dimension, "label_id": body.label_id, "taxonomy_version": tax.version}


# ------------------------------------------------------------------ monitoring
@api.get("/monitoring/drift", summary="Label-free production health: recent traffic vs baseline (abstention, evidence, label mix, feedback)")
async def monitoring_drift(window_hours: int = Query(24, ge=1, le=720), baseline_days: int = Query(14, ge=1, le=90),
                           svc: Services = Depends(get_services)):
    return await drift_report(svc.repo, window_hours, baseline_days)


# ------------------------------------------------------------------ emerging-class discovery
class DiscoverBody(BaseModel):
    window_days: int | None = Field(None, ge=1, le=365)


class AcceptBody(BaseModel):
    label_id: str | None = Field(None, pattern=r"^[a-z][a-z0-9_]{2,47}$", description="override the suggested intent id")
    description: str | None = Field(None, max_length=400)
    team: str | None = Field(None, max_length=80)
    merge_into: str | None = Field(None, description="extend this existing intent instead of creating a new one")
    note: str | None = Field(None, max_length=300)


class RejectBody(BaseModel):
    note: str | None = Field(None, max_length=300)


@api.post("/taxonomy/discover", status_code=202, summary="Cluster poorly-explained recent requests and propose new classes (async job)")
async def discover(body: DiscoverBody, bg: BackgroundTasks, svc: Services = Depends(get_services)):
    job_id = await jobs.submit(svc, bg, "discover_classes", body.model_dump(exclude_none=True))
    return {"job_id": job_id, "status": "queued", "poll": f"/api/v1/jobs/{job_id}"}


@api.get("/taxonomy/proposals", summary="Reviewable class proposals produced by discovery")
async def list_proposals(status: Literal["pending", "accepted", "rejected", "superseded", "all"] = "pending",
                         svc: Services = Depends(get_services)):
    return {"items": await svc.repo.list_proposals(None if status == "all" else status)}


@api.post("/taxonomy/proposals/{proposal_id}/accept", summary="Accept a proposal: create (or extend) an intent, bump taxonomy version")
async def accept_proposal(proposal_id: str, body: AcceptBody, svc: Services = Depends(get_services)):
    return await svc.discovery.accept(proposal_id, body.label_id, body.description, body.team, body.merge_into, body.note)


@api.post("/taxonomy/proposals/{proposal_id}/reject", summary="Reject a proposal (its requests will not be proposed again)")
async def reject_proposal(proposal_id: str, body: RejectBody, svc: Services = Depends(get_services)):
    return await svc.discovery.reject(proposal_id, body.note)


# ------------------------------------------------------------------ feedback / stats
@api.post("/feedback", status_code=201, summary="Agent feedback on a resolution")
async def feedback(body: FeedbackIn, svc: Services = Depends(get_services)):
    try:
        uuid.UUID(body.request_id)
    except ValueError as exc:
        raise ValidationFailure("request_id must be a UUID") from exc
    if not await svc.repo.request_exists(body.request_id):
        raise NotFoundError("request_id not found")
    if body.corrected_intent and not svc.taxonomy.current.has("intent", body.corrected_intent):
        raise ValidationFailure(f"unknown intent '{body.corrected_intent}'")
    return {"feedback_id": await svc.repo.save_feedback(body.request_id, body.rating, body.comment, body.corrected_intent)}


@api.get("/stats", summary="Corpus / taxonomy / provider overview")
async def stats(svc: Services = Depends(get_services)):
    return {"corpus": await svc.repo.counts(), "corpus_version": await svc.repo.corpus_version(),
            "taxonomy_version": svc.taxonomy.current.version, "embedding_model": svc.embedder.model_name,
            "reranker": svc.reranker.model_name if svc.reranker.enabled else None,
            "llm_providers": [f"{p.name}:{p.model}" for p in svc.llm.providers],
            "default_strategy": svc.settings.default_retrieval_strategy}


# ------------------------------------------------------------------ evaluation (async job)
@api.post("/evaluate", status_code=202, summary="Run evaluation suites in the background")
async def evaluate(body: EvaluateRequest, bg: BackgroundTasks, svc: Services = Depends(get_services)):
    if {"evolving", "discovery"} & set(body.suites) and not svc.settings.allow_mutating_eval:
        raise HTTPException(status_code=403, detail="the 'evolving' and 'discovery' suites temporarily write to the live corpus; set ALLOW_MUTATING_EVAL=true "
                                                    "or run it from the CLI against a non-production database")
    job_id = await jobs.submit(svc, bg, "evaluate", body.model_dump())
    return {"job_id": job_id, "status": "queued", "poll": f"/api/v1/evaluate/{job_id}"}


@api.get("/evaluate/{job_id}", summary="Evaluation job status/result")
async def evaluate_status(job_id: str, svc: Services = Depends(get_services)):
    return await get_job(job_id, svc)
