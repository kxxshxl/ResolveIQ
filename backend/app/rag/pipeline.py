"""End-to-end resolution pipeline:
complaint -> redact -> classify -> retrieve (tickets+KB) -> evidence gate -> grounded generation -> citation validation.

Every resolution carries its own provenance (model, prompt, taxonomy, corpus, retrieval configuration), a stage-by-stage trace and its lineage
(the evidence graph), so a case can be audited, replayed and compared long after it ran.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field

from app.classification.pipeline import ComplaintClassifier
from app.classification.taxonomy import TaxonomyService
from app.core.config import Settings
from app.core.errors import LLMOverloaded, LLMUnavailable
from app.core.logging import trace_id_var
from app.core.pii import redact
from app.core.text import looks_like_injection, normalize_text
from app.core.version import PIPELINE_VERSION
from app.db.repository import Repository
from app.models.schemas import (Classification, Provenance, ResolveRequest, ResolveResponse, Resolution, RetrievedItem,
                                StageRecord, ValidationReport)
from app.observability import tracing
from app.observability.metrics import (ABSTENTIONS, CACHE, EVIDENCE_CONFIDENCE, INJECTION_FLAGS, PII_REDACTIONS,
                                       RESOLUTIONS, STAGE_LATENCY)
from app.rag.citations import validate_resolution
from app.rag.evidence import assess_evidence, select_evidence
from app.rag.generator import GenerationError, generate_extractive, generate_with_llm
from app.rag.lineage import build_lineage
from app.rag.prompts import PROMPT_HASH, PROMPT_VERSION, build_prompt
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
    if r.provenance:
        attrs.update({"resolveiq.prompt_version": r.provenance.prompt_version, "resolveiq.taxonomy_version": r.provenance.taxonomy_version,
                      "resolveiq.corpus_version": r.provenance.corpus_version})
    attrs.update({f"resolveiq.latency_ms.{k}": v for k, v in r.latency_ms.items()})
    return attrs


@dataclass
class Generated:
    resolution: Resolution
    generator: str
    status: str
    citations: list
    validation: ValidationReport
    warnings: list[str]
    in_prompt: list[RetrievedItem]
    meta: dict = field(default_factory=dict)         # what generation looked like: prompt build, tokens, temperature, seed
    stages: list[StageRecord] = field(default_factory=list)


class ResolutionService:
    def __init__(self, repo: Repository, retrieval: RetrievalService, classifier: ComplaintClassifier,
                 taxonomy: TaxonomyService, llm: ResilientLLM, embedder: EmbeddingService, cache: Cache, settings: Settings):
        self.repo, self.retrieval, self.classifier, self.taxonomy = repo, retrieval, classifier, taxonomy
        self.llm, self.embedder, self.cache, self.s = llm, embedder, cache, settings

    def _team(self, cls: Classification) -> str:
        lab = self.taxonomy.current.labels["intent"].get(cls.intent)
        return (lab.team if lab and lab.team else DEFAULT_TEAMS.get(cls.product, "senior support specialist"))

    def _primary_model(self) -> str:
        p = self.llm.providers[0] if self.llm.providers else None
        return f"{p.name}:{p.model}" if p else "extractive"

    @tracing.traced(
        "resolve",
        attrs=lambda self, req, generate=True, persist=True, use_cache=True: {
            "resolveiq.strategy": req.strategy or self.s.default_retrieval_strategy, "resolveiq.metadata_filters": req.use_metadata_filters,
            "resolveiq.generate": generate, "resolveiq.complaint_chars": len(req.complaint), "resolveiq.deterministic": req.deterministic},
        result=_resolve_attrs)
    async def resolve(self, req: ResolveRequest, generate: bool = True, persist: bool = True,
                      use_cache: bool = True) -> ResolveResponse:
        t_start = time.perf_counter()
        lat: dict[str, float] = {}
        trace: list[StageRecord] = []
        trace_id = trace_id_var.get()
        request_id = str(uuid.uuid4())
        strategy = req.strategy or self.s.default_retrieval_strategy
        deterministic = req.deterministic or self.s.llm_deterministic
        temperature = 0.0 if deterministic else self.s.llm_temperature
        seed = self.s.llm_seed if deterministic else None

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
        trace.append(StageRecord(name="preprocess", latency_ms=lat["preprocess"], detail={
            "chars_in": len(req.complaint), "chars_out": len(red.text), "pii_redactions": red.counts, "injection_flagged": injection}))

        corpus_version = await self.repo.corpus_version()
        cache_key = None
        cache_ms = None
        if use_cache:
            t_cache = time.perf_counter()
            with tracing.span("resolve.cache"):
                version = f"{corpus_version}.{self.taxonomy.current.version}"
                digest = hashlib.sha256(f"{strategy}|{req.use_metadata_filters}|{deterministic}|{PROMPT_HASH}|{self._primary_model()}|{red.text}".encode()).hexdigest()[:32]
                cache_key = f"resolve:{version}:{digest}"
                hit = await self.cache.get(cache_key)
                tracing.annotate(resolveiq__cache_hit=bool(hit))
            cache_ms = round((time.perf_counter() - t_cache) * 1000, 1)
            if hit:
                CACHE.labels("resolve", "hit").inc()
                served = round((time.perf_counter() - t_start) * 1000, 1)
                original = [StageRecord(**s) for s in hit.get("trace", [])]
                hit.update(request_id=request_id, trace_id=trace_id, cached=True, latency_ms={"total": served},
                           trace=[StageRecord(name="cache", latency_ms=served, detail={"hit": True, "note": "served from the response cache; the stages below are from the original run"}),
                                  *original])
                cached_resp = ResolveResponse(**hit)
                if persist:
                    await self._persist(cached_resp, red.text, None)
                return cached_resp
            CACHE.labels("resolve", "miss").inc()
        trace.append(StageRecord(name="cache", status="skipped" if not use_cache else "ok", latency_ms=cache_ms, detail={"enabled": use_cache, "hit": False}))

        # 2. embed + classify
        ctx = await self.retrieval.make_context(red.text)
        t = tick("embed", t)
        trace.append(StageRecord(name="embed", latency_ms=lat["embed"], detail={"model": self.embedder.model_name, "dimensions": int(len(ctx.embedding))}))

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
        trace.append(StageRecord(name="classify", latency_ms=lat["classify"], detail={
            "intent": cls.intent, "product": cls.product, "severity": cls.severity, "sentiment": cls.sentiment, "strategy": cls.strategy,
            "taxonomy_version": cls.taxonomy_version, "confidence": cls.confidence.model_dump()}))
        top = lambda items: ({"id": items[0].source_id, "score": items[0].score} if items else None)  # noqa: E731
        trace.append(StageRecord(name="retrieve", latency_ms=lat["retrieve"], detail={
            "strategy": strategy, "filters": sorted(ctx.filters or {}), "tickets": len(tickets), "articles": len(articles),
            "top_ticket": top(tickets), "top_article": top(articles), "timings_ms": {k: round(v, 1) for k, v in ctx.timings.items()},
            "adaptive": ctx.decisions or None}))
        uses_rerank = any("rerank_prob" in i.scores for i in (*tickets, *articles))
        trace.append(StageRecord(name="rerank", status="ok" if uses_rerank else "skipped", latency_ms=ctx.timings.get("rerank_ms") if uses_rerank else None, detail={
            "model": self.retrieval.reranker.model_name if uses_rerank else None,
            "top_probability": max((i.scores["rerank_prob"] for i in (*tickets, *articles) if "rerank_prob" in i.scores), default=None),
            **({} if uses_rerank else {"reason": "the strategy did not call for reranking"})}))

        # 4. evidence gate (abstention is a feature)
        t_ev = time.perf_counter()
        with tracing.span("resolve.evidence"):
            top_t = await self.retrieval.dense_top_similarity(ctx, "ticket")
            top_a = await self.retrieval.dense_top_similarity(ctx, "article")
            assessment = assess_evidence(tickets, articles, top_t, top_a, self.s)
            EVIDENCE_CONFIDENCE.observe(assessment.confidence)
            evidence = select_evidence(tickets, articles, self.s)
            tracing.annotate(resolveiq__evidence__sufficient=assessment.sufficient, resolveiq__evidence__confidence=assessment.confidence,
                             resolveiq__evidence__top_ticket_similarity=top_t, resolveiq__evidence__top_article_similarity=top_a,
                             resolveiq__evidence__selected=len(evidence))
        trace.append(StageRecord(name="evidence", latency_ms=round((time.perf_counter() - t_ev) * 1000, 1), detail={
            **assessment.model_dump(), "selected": [e.source_id for e in evidence], "abstain_threshold": self.s.abstain_threshold}))

        status, generator = "abstained", "none"
        resolution: Resolution
        citations: list = []
        validation = ValidationReport(valid=True)
        team = self._team(cls)
        in_prompt: list[RetrievedItem] = []
        gen_meta: dict = {}
        abstain_reason: str | None = None

        if not assessment.sufficient or not evidence:
            ABSTENTIONS.labels("weak_evidence").inc()
            tracing.annotate(resolveiq__abstain_reason="weak_evidence")
            abstain_reason = f"weak_evidence: {assessment.reason}"
            resolution = Resolution(
                issue_summary=f"Customer reports a {cls.product.replace('_', ' ')} problem ({cls.intent.replace('_', ' ')}).",
                steps=[], escalate=True,
                escalation_reason=f"Evidence is insufficient to recommend a grounded resolution; escalate to a {team}. ({assessment.reason})",
                uncertainty="No sufficiently similar resolved ticket or knowledge-base article was found.")
            tick("generate", t)
            trace += [StageRecord(name="generate", status="skipped", detail={"reason": "the evidence gate abstained, so the model was not called"}),
                      StageRecord(name="validate", status="skipped", detail={"reason": "nothing was generated"})]
        else:
            # the LLM classification fallback overlaps with generation, so it adds no latency on the critical path
            g, cls = await asyncio.gather(self._generate(red.text, cls, evidence, generate, temperature, seed), self.classifier.refine(ctx, cls))
            resolution, generator, status, citations, validation = g.resolution, g.generator, g.status, g.citations, g.validation
            warnings += g.warnings
            in_prompt, gen_meta = g.in_prompt, g.meta
            trace += g.stages
            if status == "abstained":
                abstain_reason = "model_declined: the model found no step the evidence supports"
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
        provenance = Provenance(
            pipeline_version=PIPELINE_VERSION, generator=generator, model=self._primary_model() if generator not in ("none", "extractive") else None,
            prompt_version=PROMPT_VERSION if gen_meta else None, prompt_hash=PROMPT_HASH if gen_meta else None,
            taxonomy_version=cls.taxonomy_version, corpus_version=corpus_version, embedding_model=self.embedder.model_name,
            reranker=self.retrieval.reranker.model_name if self.retrieval.reranker.enabled else None,
            retrieval=self.retrieval.config_snapshot(strategy, ctx), thresholds={
                "abstain": self.s.abstain_threshold, "grounding": self.s.grounding_threshold, "min_grounded_ratio": self.s.min_grounded_ratio},
            generation=gen_meta or {"note": "no model call", "deterministic": deterministic})
        resp = ResolveResponse(
            request_id=request_id, trace_id=trace_id, status=status, classification=cls, tickets=tickets, articles=articles,
            resolution=resolution, citations=citations, validation=validation, evidence=assessment, confidence=confidence,
            generator=generator, pii_redactions=red.counts, warnings=warnings, latency_ms=lat, provenance=provenance, trace=trace)
        try:
            resp.lineage = build_lineage(
                complaint=red.text, pii=red.counts, injection=injection, cls=cls, tickets=tickets, articles=articles, selected=evidence,
                in_prompt=[e.source_id for e in in_prompt] or ([e.source_id for e in evidence] if status != "abstained" else []), resolution=resolution,
                citations=citations, validation=validation, assessment=assessment, status=status, abstain_reason=abstain_reason)
        except Exception:  # noqa: BLE001 - the audit view must never take the answer down
            log.exception("failed to build lineage")
        RESOLUTIONS.labels(status, generator.split(":")[0]).inc()

        if persist:
            await self._persist(resp, red.text, ctx.embedding)
        if cache_key and (status in ("resolved", "abstained") or (status == "degraded" and self.s.cache_degraded_ttl_seconds > 0)):
            ttl = self.s.cache_ttl_seconds if status != "degraded" else min(self.s.cache_ttl_seconds, self.s.cache_degraded_ttl_seconds)
            await self.cache.set(cache_key, resp.model_dump(mode="json"), ttl=ttl)
        log.info("resolved", extra={"status": status, "intent": cls.intent, "confidence": confidence,
                                    "strategy": strategy, "latency_ms": lat["total"]})
        return resp

    @tracing.traced("resolve.persist")
    async def _persist(self, resp: ResolveResponse, complaint: str, embedding) -> None:
        """Audit trail: redacted complaint + classification + what was retrieved + result + stage trace + provenance (best effort)."""
        try:
            await self.repo.save_request(
                resp.request_id, resp.trace_id, complaint, resp.classification.model_dump(),
                [{"source_type": i.source_type, "source_id": i.source_id, "score": i.score, "rank": i.rank, "method": i.retrieval_method,
                  "scores": i.scores, "intent": i.metadata.get("intent")} for i in (*resp.tickets, *resp.articles)],
                resp.model_dump(mode="json", exclude={"tickets", "articles", "trace", "provenance"}), resp.status, resp.confidence,
                int(resp.latency_ms["total"]), embedding=embedding, embedding_model=self.embedder.model_name if embedding is not None else None,
                trace=[s.model_dump(mode="json") for s in resp.trace], provenance=resp.provenance.model_dump(mode="json") if resp.provenance else None)
        except Exception as exc:  # noqa: BLE001 - audit write must not fail the response
            log.warning("failed to persist resolution request", extra={"error": str(exc)[:200]})

    @tracing.traced("resolve.generate", attrs=lambda self, complaint, cls, evidence, generate, temperature=0.1, seed=None: {
                        "resolveiq.generate": generate, "resolveiq.evidence_items": len(evidence), "resolveiq.prompt_version": PROMPT_VERSION},
                    result=lambda g: {"resolveiq.generator": g.generator, "resolveiq.generation_status": g.status})
    async def _generate(self, complaint: str, cls: Classification, evidence: list[RetrievedItem], generate: bool,
                        temperature: float = 0.1, seed: int | None = None) -> Generated:
        warnings: list[str] = []
        stages: list[StageRecord] = []
        resolution = generator = None
        degraded = False
        meta: dict = {}
        in_prompt = evidence
        t_gen = time.perf_counter()
        llm_note = None
        if generate and self.llm.available:
            window = self.s.llm_num_ctx - self.s.llm_max_tokens - self.s.llm_context_reserve_tokens
            build = build_prompt(complaint, cls, evidence, context_budget_tokens=window)
            in_prompt = build.evidence
            meta = {**build.meta, "temperature": temperature, "seed": seed, "deterministic": seed is not None, "max_tokens": self.s.llm_max_tokens,
                    "num_ctx": self.s.llm_num_ctx}
            try:
                resolution, res = await generate_with_llm(self.llm, build, temperature=temperature, seed=seed)
                generator = f"{res.provider}:{res.model}"
                meta.update(prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens, llm_latency_ms=round(res.latency_s * 1000, 1))
            except LLMOverloaded:
                warnings.append("LLM is busy with other requests; returned evidence-only resolution.")
                degraded, llm_note = True, "all generation slots busy"
            except (LLMUnavailable, GenerationError) as exc:
                warnings.append(f"LLM unavailable or invalid output ({exc.message[:160]}); returned evidence-only resolution.")
                degraded, llm_note = True, f"{type(exc).__name__}"
        else:
            degraded = True
            llm_note = "no LLM provider configured" if generate else "evidence-only requested"
            if generate:
                warnings.append("No LLM provider configured; returned evidence-only resolution.")
        gen_ms = round((time.perf_counter() - t_gen) * 1000, 1)
        if resolution is None:
            resolution, generator = generate_extractive(cls, evidence), "extractive"
            in_prompt = evidence
        stages.append(StageRecord(name="generate", status="degraded" if degraded else "ok", latency_ms=gen_ms, detail={
            "generator": generator, **({"fallback_reason": llm_note} if degraded else {}),
            **{k: meta[k] for k in ("prompt_version", "prompt_hash", "temperature", "seed", "prompt_tokens", "completion_tokens", "est_tokens", "budget_tokens",
                                    "evidence_in_prompt", "collapsed_duplicates", "dropped_for_budget", "injection_sanitised") if k in meta}}))

        t_val = time.perf_counter()
        resolution, citations, validation = await validate_resolution(resolution, in_prompt, self.embedder, self.s)
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
        stages.append(StageRecord(name="validate", status="ok" if validation.valid else "error", latency_ms=round((time.perf_counter() - t_val) * 1000, 1), detail={
            "valid": validation.valid, "citation_coverage": validation.citation_coverage, "grounded_ratio": validation.grounded_ratio,
            "invalid_citations": validation.invalid_citations, "uncited_steps": validation.uncited_steps, "unsupported_steps": validation.unsupported_steps,
            "citations_emitted": validation.citations_emitted, "steps": len(resolution.steps)}))
        return Generated(resolution, generator, status, citations, validation, warnings, in_prompt, meta, stages)
