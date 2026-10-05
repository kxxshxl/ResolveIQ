"""End-to-end resolution pipeline:
complaint -> redact -> classify -> retrieve (tickets+KB) -> evidence gate -> grounded generation -> citation validation.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid

from app.classification.pipeline import ComplaintClassifier
from app.classification.taxonomy import TaxonomyService
from app.core.config import Settings
from app.core.errors import LLMUnavailable, ResolveIQError
from app.core.logging import trace_id_var
from app.core.pii import redact
from app.core.text import looks_like_injection, normalize_text
from app.db.repository import Repository
from app.models.schemas import (Classification, EvidenceAssessment, ResolveRequest, ResolveResponse, Resolution,
                                RetrievedItem, ValidationReport)
from app.observability import tracing
from app.observability.metrics import (ABSTENTIONS, CACHE, EVIDENCE_CONFIDENCE, INJECTION_FLAGS, PII_REDACTIONS,
                                       RESOLUTIONS, STAGE_LATENCY)
from app.rag.citations import validate_resolution
from app.rag.evidence import assess_evidence, select_evidence
from app.rag.generator import GenerationError, generate_extractive, generate_with_llm
from app.retrieval.service import RetrievalService
from app.services.cache import Cache
from app.services.embedding import EmbeddingService
from app.services.llm.base import ResilientLLM

log = logging.getLogger(__name__)
DEFAULT_TEAMS = {"broadband": "network specialist", "mobile": "mobile network specialist"}


def _resolve_attrs(r: ResolveResponse) -> dict:
    """Outcome of one resolution as span attributes (labels, scores and timings only; never complaint text)."""
    c = r.classification
    attrs = {
        "resolveiq.request_id": r.request_id, "resolveiq.status": r.status, "resolveiq.confidence": r.confidence,
        "resolveiq.generator": r.generator, "resolveiq.cached": r.cached, "resolveiq.escalate": r.resolution.escalate,
        "resolveiq.intent": c.intent, "resolveiq.product": c.product, "resolveiq.severity": c.severity, "resolveiq.sentiment": c.sentiment,
        "resolveiq.evidence.sufficient": r.evidence.sufficient, "resolveiq.evidence.confidence": r.evidence.confidence,
        "resolveiq.validation.valid": r.validation.valid, "resolveiq.validation.citation_coverage": r.validation.citation_coverage,
        "resolveiq.validation.grounded_ratio": r.validation.grounded_ratio,
        "resolveiq.tickets": len(r.tickets), "resolveiq.articles": len(r.articles), "resolveiq.steps": len(r.resolution.steps),
        "resolveiq.warnings": len(r.warnings), "resolveiq.pii_redactions": sum(r.pii_redactions.values()),
    }
    attrs.update({f"resolveiq.latency_ms.{k}": v for k, v in r.latency_ms.items()})
    return attrs


class ResolutionService:
    def __init__(self, repo: Repository, retrieval: RetrievalService, classifier: ComplaintClassifier,
                 taxonomy: TaxonomyService, llm: ResilientLLM, embedder: EmbeddingService, cache: Cache, settings: Settings):
        self.repo, self.retrieval, self.classifier, self.taxonomy = repo, retrieval, classifier, taxonomy
        self.llm, self.embedder, self.cache, self.s = llm, embedder, cache, settings

    def _team(self, cls: Classification) -> str:
        lab = self.taxonomy.current.labels["intent"].get(cls.intent)
        return (lab.team if lab and lab.team else DEFAULT_TEAMS.get(cls.product, "senior support specialist"))

    @tracing.traced(
        "resolve",
        attrs=lambda self, req, generate=True, persist=True, use_cache=True: {
            "resolveiq.strategy": req.strategy or self.s.default_retrieval_strategy, "resolveiq.metadata_filters": req.use_metadata_filters,
            "resolveiq.generate": generate, "resolveiq.complaint_chars": len(req.complaint)},
        result=_resolve_attrs)
    async def resolve(self, req: ResolveRequest, generate: bool = True, persist: bool = True,
                      use_cache: bool = True) -> ResolveResponse:
        t_start = time.perf_counter()
        lat: dict[str, float] = {}
        trace_id = trace_id_var.get()
        request_id = str(uuid.uuid4())
        strategy = req.strategy or self.s.default_retrieval_strategy

        def tick(stage: str, since: float) -> float:
            now = time.perf_counter()
            lat[stage] = round((now - since) * 1000, 1)
            STAGE_LATENCY.labels(stage).observe(now - since)
            return now

        # 1. sanitise + redact (everything downstream, including the LLM prompt and logs, sees redacted text only)
        t = time.perf_counter()
        warnings: list[str] = []
        with tracing.span("resolve.preprocess"):
            clean = normalize_text(req.complaint)
            red = redact(clean)
            for k, v in red.counts.items():
                PII_REDACTIONS.labels(k).inc(v)
            injection = looks_like_injection(red.text)
            if injection:
                INJECTION_FLAGS.inc()
                warnings.append("Complaint contains instruction-like text; it was treated strictly as data.")
            tracing.annotate(resolveiq__pii_redactions=sum(red.counts.values()), resolveiq__injection_flagged=injection)
        t = tick("preprocess", t)

        cache_key = None
        if use_cache:
            with tracing.span("resolve.cache"):
                version = f"{await self.repo.corpus_version()}.{self.taxonomy.current.version}"
                digest = hashlib.sha256(f"{strategy}|{req.use_metadata_filters}|{red.text}".encode()).hexdigest()[:32]
                cache_key = f"resolve:{version}:{digest}"
                hit = await self.cache.get(cache_key)
                tracing.annotate(resolveiq__cache_hit=bool(hit))
            if hit:
                CACHE.labels("resolve", "hit").inc()
                hit.update(request_id=request_id, trace_id=trace_id, cached=True,
                           latency_ms={"total": round((time.perf_counter() - t_start) * 1000, 1)})
                cached_resp = ResolveResponse(**hit)
                if persist:
                    await self._persist(cached_resp, red.text)
                return cached_resp
            CACHE.labels("resolve", "miss").inc()

        # 2. embed + classify
        ctx = await self.retrieval.make_context(red.text)
        t = tick("embed", t)

        async def stage(name: str, coro):
            t0 = time.perf_counter()
            res = await coro
            lat[name] = round((time.perf_counter() - t0) * 1000, 1)
            STAGE_LATENCY.labels(name).observe(time.perf_counter() - t0)
            return res

        async def retrieve():
            return (await self.retrieval.search(ctx, "ticket", strategy, self.s.ticket_top_k),
                    await self.retrieval.search(ctx, "article", strategy, self.s.article_top_k))

        # 3. classify + retrieve. Retrieval does not need the classification (unless metadata filters are requested), so the two
        # overlap: the severity/sentiment NLI model (~40 ms GPU, ~1.2 s CPU) then hides behind retrieval and the evidence gate.
        if req.use_metadata_filters:
            cls = await stage("classify", self.classifier.classify(ctx, allow_llm=False))
            f = {}
            if cls.confidence.product >= 0.6:
                f["product"] = cls.product
            ctx.filters = f or None
            tickets, articles = await stage("retrieve", retrieve())
        else:
            cls, (tickets, articles) = await asyncio.gather(
                stage("classify", self.classifier.classify(ctx, allow_llm=False)), stage("retrieve", retrieve()))
        t = time.perf_counter()

        # 4. evidence gate (abstention is a feature)
        with tracing.span("resolve.evidence"):
            top_t = await self.retrieval.dense_top_similarity(ctx, "ticket")
            top_a = await self.retrieval.dense_top_similarity(ctx, "article")
            assessment = assess_evidence(tickets, articles, top_t, top_a, self.s)
            EVIDENCE_CONFIDENCE.observe(assessment.confidence)
            evidence = select_evidence(tickets, articles, self.s)
            tracing.annotate(resolveiq__evidence__sufficient=assessment.sufficient, resolveiq__evidence__confidence=assessment.confidence,
                             resolveiq__evidence__top_ticket_similarity=top_t, resolveiq__evidence__top_article_similarity=top_a,
                             resolveiq__evidence__selected=len(evidence))

        status, generator = "abstained", "none"
        resolution: Resolution
        citations: list = []
        validation = ValidationReport(valid=True)
        team = self._team(cls)

        if not assessment.sufficient or not evidence:
            ABSTENTIONS.labels("weak_evidence").inc()
            tracing.annotate(resolveiq__abstain_reason="weak_evidence")
            resolution = Resolution(
                issue_summary=f"Customer reports a {cls.product.replace('_', ' ')} problem ({cls.intent.replace('_', ' ')}).",
                steps=[], escalate=True,
                escalation_reason=f"Evidence is insufficient to recommend a grounded resolution; escalate to a {team}. ({assessment.reason})",
                uncertainty="No sufficiently similar resolved ticket or knowledge-base article was found.")
            tick("generate", t)
        else:
            # the LLM classification fallback overlaps with generation, so it adds no latency on the critical path
            (resolution, generator, status, citations, validation, extra), cls = await asyncio.gather(
                self._generate(red.text, cls, evidence, generate), self.classifier.refine(ctx, cls))
            warnings += extra
            t = tick("generate", t)

        # deterministic escalation policy on top of the model's own judgement
        if cls.severity == "critical" and status != "abstained" and not resolution.escalate:
            resolution.escalate = True
            resolution.escalation_reason = f"Critical severity: involve a {team} while applying these steps."
        if status == "unreliable":
            tracing.mark_error("citation/grounding validation failed")
            resolution.escalate = True
            resolution.escalation_reason = (resolution.escalation_reason or "") + \
                f" Automatic citation/grounding checks failed; verify every step with a {team} before acting."
            resolution.escalation_reason = resolution.escalation_reason.strip()

        confidence = assessment.confidence
        if status in ("resolved", "degraded"):
            confidence = round(confidence * (0.5 + 0.5 * validation.grounded_ratio), 3)
        elif status == "unreliable":
            confidence = round(min(confidence, 0.4) * 0.5, 3)
        elif status == "abstained":
            confidence = round(min(confidence, self.s.abstain_threshold - 0.01), 3)

        lat["total"] = round((time.perf_counter() - t_start) * 1000, 1)
        resp = ResolveResponse(
            request_id=request_id, trace_id=trace_id, status=status, classification=cls, tickets=tickets, articles=articles,
            resolution=resolution, citations=citations, validation=validation, evidence=assessment, confidence=confidence,
            generator=generator, pii_redactions=red.counts, warnings=warnings, latency_ms=lat)
        RESOLUTIONS.labels(status, generator.split(":")[0]).inc()

        if persist:
            await self._persist(resp, red.text)
        if cache_key and status in ("resolved", "abstained"):
            await self.cache.set(cache_key, resp.model_dump(mode="json"), ttl=self.s.cache_ttl_seconds)
        log.info("resolved", extra={"status": status, "intent": cls.intent, "confidence": confidence,
                                    "strategy": strategy, "latency_ms": lat["total"]})
        return resp

    @tracing.traced("resolve.persist")
    async def _persist(self, resp: ResolveResponse, complaint: str) -> None:
        """Audit trail: redacted complaint + classification + retrieved ids + result (best effort)."""
        try:
            await self.repo.save_request(
                resp.request_id, resp.trace_id, complaint, resp.classification.model_dump(),
                [{"source_type": i.source_type, "source_id": i.source_id, "score": i.score, "rank": i.rank}
                 for i in (*resp.tickets, *resp.articles)],
                resp.model_dump(mode="json", exclude={"tickets", "articles"}), resp.status, resp.confidence,
                int(resp.latency_ms["total"]))
        except Exception as exc:  # noqa: BLE001 - audit write must not fail the response
            log.warning("failed to persist resolution request", extra={"error": str(exc)[:200]})

    @tracing.traced("resolve.generate", attrs=lambda self, complaint, cls, evidence, generate: {"resolveiq.generate": generate, "resolveiq.evidence_items": len(evidence)},
                    result=lambda r: {"resolveiq.generator": r[1], "resolveiq.generation_status": r[2]})
    async def _generate(self, complaint: str, cls: Classification, evidence: list[RetrievedItem], generate: bool):
        warnings: list[str] = []
        resolution = generator = None
        degraded = False
        if generate and self.llm.available:
            try:
                resolution, res = await generate_with_llm(self.llm, complaint, cls, evidence)
                generator = f"{res.provider}:{res.model}"
            except (LLMUnavailable, GenerationError) as exc:
                warnings.append(f"LLM unavailable or invalid output ({exc.message[:160]}); returned evidence-only resolution.")
                degraded = True
        else:
            degraded = True
            if generate:
                warnings.append("No LLM provider configured; returned evidence-only resolution.")
        if resolution is None:
            resolution, generator = generate_extractive(cls, evidence), "extractive"

        resolution, citations, validation = await validate_resolution(resolution, evidence, self.embedder, self.s)
        if not resolution.steps and not resolution.escalate:
            resolution.escalate, resolution.escalation_reason = True, "The model produced no actionable steps."
        status = "resolved"
        if degraded:
            status = "degraded"
        if not validation.valid:
            status = "unreliable"
        if not resolution.steps:  # the model itself declined to answer from the evidence
            status = "abstained"
            ABSTENTIONS.labels("model_declined").inc()
            tracing.annotate(resolveiq__abstain_reason="model_declined")
        warnings += validation.warnings
        return resolution, generator, status, citations, validation, warnings
