"""Endpoints behind the support console: cases and replay, retrieval lab, recurring clusters, feedback analytics, system health.

All of them live under /api/v1, so they share the API-key authentication and the rate limiter with the rest of the API.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.api.security import get_services, rate_limit
from app.core.errors import NotFoundError
from app.evaluation.summary import load_results
from app.models.schemas import Strategy
from app.services.container import Services
from app.system.health import database_health
from app.system.status import system_status

MAX_OFFSET = 1_000_000  # bounded: an unbounded offset reached Postgres as an out-of-range bigint (HTTP 500); page deeper with ?cursor=

console = APIRouter(prefix="/api/v1", dependencies=[Depends(rate_limit)], tags=["console"])


# ------------------------------------------------------------------ cases and replay
@console.get("/cases", summary="Recent resolutions as replayable cases (redacted complaint snippet, status, rating)")
async def list_cases(limit: int = Query(25, ge=1, le=100), offset: int = Query(0, ge=0, le=MAX_OFFSET),
                     status: Literal["resolved", "abstained", "degraded", "unreliable"] | None = None, intent: str | None = Query(None, max_length=64),
                     rated: Literal["none", "helpful", "not_helpful"] | None = None, svc: Services = Depends(get_services)):
    return await svc.cases.list(limit, offset, status, intent, rated)


@console.get("/cases/{request_id}", summary="One case: complaint, stage trace, provenance, lineage, retrieved sources, feedback")
async def get_case(request_id: str, svc: Services = Depends(get_services)):
    return await svc.cases.get(request_id)


class ReplayBody(BaseModel):
    strategy: Strategy | None = Field(None, description="retrieval strategy for the replay (default: the one the case used)")
    generate: bool = Field(True, description="false replays retrieval, evidence and validation without calling the LLM")
    deterministic: bool = Field(True, description="temperature 0 and a fixed seed, so a difference is not sampling noise")


@console.post("/cases/{request_id}/replay", summary="Re-run a stored case through the pipeline as configured now, and diff it against the original")
async def replay_case(request_id: str, body: ReplayBody, svc: Services = Depends(get_services)):
    return await svc.cases.replay(request_id, body.strategy, body.generate, body.deterministic)


class CompareBody(BaseModel):
    complaint: str = Field(..., min_length=5, max_length=4000)
    strategies: list[Strategy] | None = None
    k: int = Field(5, ge=1, le=10)
    scenario_id: str | None = Field(None, max_length=64, description="ground truth: results from this root-cause scenario count as relevant")


@console.post("/retrieval/compare", summary="Retrieval lab: one complaint through each strategy, side by side, with relevance when ground truth is known")
async def compare_retrieval(body: CompareBody, svc: Services = Depends(get_services)):
    return await svc.lab.compare(body.complaint, body.strategies, body.k, body.scenario_id)


class CaseCompareBody(BaseModel):
    strategies: list[Strategy] | None = None
    k: int = Field(5, ge=1, le=10)


@console.post("/cases/{request_id}/compare", summary="Compare retrieval strategies on a stored case's complaint")
async def compare_case(request_id: str, body: CaseCompareBody, svc: Services = Depends(get_services)):
    case = await svc.repo.get_case(request_id)
    if not case:
        raise NotFoundError("case not found")
    return await svc.lab.compare(case["complaint"], body.strategies, body.k)


@console.get("/retrieval/examples", summary="Labelled complaints to try in the retrieval lab (ground truth is known for these)")
async def retrieval_examples(limit: int = Query(40, ge=1, le=100), svc: Services = Depends(get_services)):
    return {"items": svc.lab.examples(limit)}


# ------------------------------------------------------------------ recurring complaint clusters
@console.get("/clusters/recurring", summary="Groups of semantically similar recent complaints (support-side clusters, not network incidents)")
async def recurring_clusters(days: int = Query(7, ge=1, le=60), min_size: int = Query(3, ge=2, le=50), distance: float = Query(0.55, ge=0.2, le=0.9),
                             svc: Services = Depends(get_services)):
    return await svc.clusters.recurring(days, min_size, distance)


# ------------------------------------------------------------------ feedback intelligence
@console.get("/quality/summary", summary="Helpful/rejection rates, problem intents, rejected sources, abstention reasons and failure patterns")
async def quality_summary(days: int = Query(30, ge=1, le=365), svc: Services = Depends(get_services)):
    return await svc.quality.summary(days)


@console.get("/quality/report", summary="Deterministic improvement report from feedback (advisory: changes no model, label or document)")
async def quality_report(days: int = Query(30, ge=1, le=365), svc: Services = Depends(get_services)):
    return await svc.quality.report(days)


# ------------------------------------------------------------------ recorded evaluation results
@console.get("/evaluation/results", summary="Recorded evaluation, drift-demo, scale-experiment and load-test results (files on disk, with their paths and write times)")
async def evaluation_results():
    return load_results()


# ------------------------------------------------------------------ system
@console.get("/system/status", summary="Versions, dependencies, LLM circuit state, queue, drift freshness and database findings")
async def status(svc: Services = Depends(get_services)):
    return await system_status(svc)


@console.get("/system/db", summary="Database health: sizes, index usage, HNSW settings, embedding coverage, integrity checks (no row contents)")
async def db_health(svc: Services = Depends(get_services)):
    return await database_health(svc.repo, svc.settings)
