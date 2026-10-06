"""Feedback service: validate and persist ratings with their context, and serve the analytics and improvement report."""
from __future__ import annotations

import uuid

from app.classification.taxonomy import TaxonomyService
from app.core.errors import NotFoundError, ValidationFailure
from app.core.pii import redact
from app.core.text import normalize_text
from app.db.repository import Repository
from app.models.schemas import FeedbackIn
from app.observability import tracing
from app.observability.metrics import FEEDBACK
from app.quality.analysis import analyze, findings, to_markdown


def _clean(text: str, limit: int) -> str:
    """Free text typed by an agent can contain customer details: redact it exactly like a complaint before it is stored."""
    return redact(normalize_text(text)).text[:limit]


class QualityService:
    def __init__(self, repo: Repository, taxonomy: TaxonomyService):
        self.repo, self.taxonomy = repo, taxonomy

    @tracing.traced("feedback.save", attrs=lambda self, body: {"resolveiq.request_id": body.request_id, "resolveiq.feedback.rating": body.rating,
                                                               "resolveiq.feedback.edited": body.edited_steps is not None, "resolveiq.feedback.reasons": len(body.reasons)})
    async def save(self, body: FeedbackIn) -> dict:
        try:
            uuid.UUID(body.request_id)
        except ValueError as exc:
            raise ValidationFailure("request_id must be a UUID") from exc
        ctx = await self.repo.get_request_context(body.request_id)
        if not ctx:
            raise NotFoundError("request_id not found")
        if body.corrected_intent and not self.taxonomy.current.has("intent", body.corrected_intent):
            raise ValidationFailure(f"unknown intent '{body.corrected_intent}'")
        known = {r["source_id"] for r in (ctx["retrieved"] or [])}
        stray = [s for s in body.rejected_sources if s not in known]
        if stray:
            raise ValidationFailure(f"rejected_sources must be sources retrieved for this request; not retrieved: {stray}")
        edited = [_clean(s, 1000) for s in body.edited_steps if s.strip()] if body.edited_steps is not None else None
        fid = await self.repo.save_feedback(body.request_id, body.rating, _clean(body.comment, 2000) if body.comment else None, body.corrected_intent, body.reasons,
                                            body.rejected_sources, edited, ctx["provenance"])
        FEEDBACK.labels(body.rating).inc()
        return {"feedback_id": fid}

    @tracing.traced("feedback.analyze", attrs=lambda self, days=30: {"resolveiq.window_days": days},
                    result=lambda r: {"resolveiq.feedback.ratings": r["totals"]["feedback"], "resolveiq.feedback.requests": r["totals"]["requests"]})
    async def summary(self, days: int = 30) -> dict:
        fb = await self.repo.feedback_rows(days)
        req = await self.repo.request_rows_for_quality(days)
        return analyze(fb, req, days)

    async def report(self, days: int = 30) -> dict:
        s = await self.summary(days)
        items = findings(s)
        return {"window_days": days, "findings": items, "markdown": to_markdown(s, items), "advisory_only": True,
                "note": "Deterministic: the same ratings always produce the same report. It changes no model, label or document."}
