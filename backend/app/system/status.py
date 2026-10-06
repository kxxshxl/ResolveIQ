"""Aggregated system status for the console: versions, dependencies, LLM circuit state, queue, drift freshness and database findings in one call."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.core.version import PIPELINE_VERSION
from app.observability import tracing
from app.rag.prompts import PROMPT_HASH, PROMPT_VERSION
from app.services.container import Services
from app.system.health import database_health


async def _safe(coro, default=None):
    try:
        return await asyncio.wait_for(coro, 4)
    except Exception as exc:  # noqa: BLE001 - one broken dependency must not blank the whole status page
        return {"error": type(exc).__name__} if default is None else default


async def system_status(svc: Services) -> dict:
    s = svc.settings
    checks: dict = {}
    checks["database"] = "ok" if await _safe(svc.repo.ping(), False) is True else "unavailable"
    checks["redis"] = "ok" if await _safe(svc.cache.ping(), False) is True else "unavailable (in-process cache fallback)"
    checks["models"] = "loaded" if svc.models_ready else "loading"
    checks["affect_model"] = "loaded" if svc.affect.available else "unavailable (rule/kNN fallback for severity and sentiment)"
    counts = await _safe(svc.repo.counts())
    queue = await _safe(svc.repo.queue_depth(), {})
    jobs = await _safe(svc.repo._fetch(
        "SELECT kind, max(created_at) AS last_created, count(*) FILTER (WHERE status='failed' AND created_at > now() - interval '24 hours') AS failed_24h, "
        "count(*) FILTER (WHERE status IN ('queued','running')) AS active FROM jobs GROUP BY kind ORDER BY kind"), [])
    snap = await _safe(svc.repo.latest_drift_snapshot(), None)
    drift = None
    if isinstance(snap, dict) and "report" in snap:
        age = (datetime.now(timezone.utc) - snap["created_at"]).total_seconds()
        drift = {"status": snap["report"].get("status"), "age_seconds": int(age), "alerts": len(snap["report"].get("alerts", [])), "stale": age > 2 * 3600 * max(s.drift_analysis_hours, 1) and s.drift_analysis_hours > 0}
    db = await _safe(database_health(svc.repo, s))
    llm_health = await _safe(svc.llm.health(), {}) if svc.llm.available else {}
    ready = checks["database"] == "ok" and svc.models_ready
    return {
        "status": "ready" if ready else "not_ready", "checks": checks, "corpus": counts,
        "versions": {"pipeline": PIPELINE_VERSION, "prompt": {"version": PROMPT_VERSION, "hash": PROMPT_HASH}, "taxonomy": svc.taxonomy.current.version,
                     "corpus": await _safe(svc.repo.corpus_version(), None), "embedding_model": svc.embedder.model_name, "reranker": svc.reranker.model_name if svc.reranker.enabled else None,
                     "affect_model": s.affect_model if svc.affect.available else None, "default_retrieval_strategy": s.default_retrieval_strategy},
        "llm": {"providers": svc.llm.status(), "reachable": llm_health, "budget_seconds": s.llm_total_budget_seconds, "max_concurrency": s.llm_max_concurrency,
                "deterministic_default": s.llm_deterministic},
        "queue": {"mode": s.job_execution, "depth": queue, "kinds": jobs}, "drift": drift,
        "tracing": tracing.status(), "security": {"authentication": bool(s.api_key_set), "rate_limit_per_minute": s.rate_limit_per_minute, "environment": s.app_env,
                                                 "metrics_protected": bool(s.metrics_token)},
        "database": {"status": db.get("status"), "findings": db.get("findings", []), "elapsed_ms": db.get("elapsed_ms")} if isinstance(db, dict) and "error" not in db else db}
