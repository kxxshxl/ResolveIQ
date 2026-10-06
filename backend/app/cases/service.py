"""Case service: list, inspect and replay stored resolutions.

A case is a `resolution_requests` row: the redacted complaint, the retrieved sources with their scores, the result (steps, citations, validation, lineage),
the stage trace and the provenance. Replay runs the stored (already redacted) complaint through the pipeline as configured NOW, never persisting the
replay, and reports what differs and why.
"""
from __future__ import annotations

import logging

from app.cases.diff import diff_cases
from app.core.errors import NotFoundError, ValidationFailure
from app.db.repository import Repository
from app.models.schemas import ResolveRequest
from app.observability import tracing
from app.observability.metrics import CASE_REPLAYS
from app.rag.pipeline import ResolutionService

log = logging.getLogger(__name__)


class CaseService:
    def __init__(self, repo: Repository, resolution: ResolutionService):
        self.repo, self.resolution = repo, resolution

    async def list(self, limit: int, offset: int, status: str | None, intent: str | None, rated: str | None) -> dict:
        rows, total = await self.repo.list_cases(limit, offset, status, intent, rated)
        return {"total": total, "limit": limit, "offset": offset, "items": rows}

    @tracing.traced("cases.get", attrs=lambda self, request_id: {"resolveiq.request_id": request_id})
    async def get(self, request_id: str) -> dict:
        case = await self.repo.get_case(request_id)
        if not case:
            raise NotFoundError("case not found")
        retrieved = case["retrieved"] or []
        # attach the current text of each retrieved source; a source deleted since is reported as such rather than silently dropped
        docs: dict[str, dict] = {}
        for kind in ("ticket", "article"):
            ids = [r["source_id"] for r in retrieved if r["source_type"] == kind]
            for d in await self.repo.get_docs(kind, ids) if ids else []:
                docs[d["source_id"]] = d
        items = []
        for r in retrieved:
            d = docs.get(r["source_id"])
            items.append({**r, "in_corpus": d is not None, "title": (d or {}).get("title"), "excerpt": ((d or {}).get("text") or "")[:220] or None})
        result = case["result"] or {}
        return {
            "request_id": case["request_id"], "trace_id": case["trace_id"], "created_at": case["created_at"], "complaint": case["complaint"],
            "status": case["status"], "confidence": case["confidence"], "latency_ms": case["latency_ms"], "classification": case["classification"],
            "retrieved": case["retrieved"], "sources": items, "result": result, "lineage": result.get("lineage"),
            "trace": case["trace"], "provenance": case["provenance"], "trace_available": case["trace"] is not None,
            "feedback": case["feedback"], "has_embedding": case["has_embedding"]}

    @tracing.traced("cases.replay", attrs=lambda self, request_id, strategy=None, generate=True, deterministic=True: {
        "resolveiq.request_id": request_id, "resolveiq.strategy": strategy, "resolveiq.generate": generate, "resolveiq.deterministic": deterministic},
        result=lambda r: {"resolveiq.replay.reproduced": r["diff"]["reproduced"], "resolveiq.replay.status": r["replay"]["status"]})
    async def replay(self, request_id: str, strategy: str | None = None, generate: bool = True, deterministic: bool = True) -> dict:
        case = await self.repo.get_case(request_id)
        if not case:
            raise NotFoundError("case not found")
        original_strategy = ((case["provenance"] or {}).get("retrieval") or {}).get("strategy")
        try:
            resp = await self.resolution.resolve(
                ResolveRequest(complaint=case["complaint"], strategy=strategy or original_strategy, deterministic=deterministic),
                generate=generate, persist=False, use_cache=False)
        except ValidationFailure:
            CASE_REPLAYS.labels("error").inc()
            raise
        replay = resp.model_dump(mode="json")
        diff = diff_cases(case, replay)
        CASE_REPLAYS.labels("identical" if diff["reproduced"] else "changed").inc()
        log.info("case replayed", extra={"request_id": request_id, "reproduced": diff["reproduced"], "strategy": strategy or original_strategy})
        return {"request_id": request_id, "replay": replay, "diff": diff, "persisted": False,
                "options": {"strategy": strategy or original_strategy, "generate": generate, "deterministic": deterministic}}
