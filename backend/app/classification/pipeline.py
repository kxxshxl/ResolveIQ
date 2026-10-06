"""ComplaintClassifier: combines strategies into one structured Classification."""
from __future__ import annotations

import asyncio
import logging

from app.classification.affect import AffectModel
from app.classification.strategies import (Classifier, Dist, EmbeddingClassifier, LLMZeroShotClassifier,
                                           RuleClassifier)
from app.classification.taxonomy import DIMENSIONS, TaxonomyService
from app.core.config import Settings
from app.core.errors import LLMOverloaded
from app.observability import tracing
from app.models.schemas import Classification, Confidence
from app.observability.metrics import CLASSIFICATION_CONFIDENCE
from app.retrieval.service import QueryContext, RetrievalService
from app.services.llm.base import ResilientLLM

log = logging.getLogger(__name__)

# (rules_weight, embedding_weight) per dimension. Rules are sharp on explicit vocabulary (severity cues,
# sentiment words) while kNN generalises over paraphrase for intent/product. Values come from the grid
# search in app/evaluation/tuning.py run on the validation split only (see docs/evaluation.md).
ENSEMBLE_WEIGHTS: dict[str, tuple[float, float]] = {
    "intent": (0.35, 0.65), "product": (0.0, 1.0), "severity": (0.5, 0.5), "sentiment": (0.5, 0.5),
}
DEFAULTS = {"severity": "medium", "sentiment": "neutral"}


async def _no_affect() -> Dist:
    return {d: {} for d in DIMENSIONS}


class ComplaintClassifier:
    def __init__(self, taxonomy: TaxonomyService, retrieval: RetrievalService, llm: ResilientLLM, settings: Settings,
                 affect: AffectModel | None = None):
        self.taxonomy, self.s, self.llm, self.affect = taxonomy, settings, llm, affect
        self.rules = RuleClassifier()
        self.embedding = EmbeddingClassifier(retrieval, k=settings.knn_k)
        self.zero_shot = LLMZeroShotClassifier(llm)

    @tracing.traced(
        "classify",
        attrs=lambda self, ctx, strategy=None, allow_llm=True: {"resolveiq.classify.strategy": strategy or self.s.classifier_strategy,
                                                                "resolveiq.classify.allow_llm": allow_llm},
        result=lambda c: {"resolveiq.classify.resolved_strategy": c.strategy, "resolveiq.intent": c.intent, "resolveiq.product": c.product,
                          "resolveiq.severity": c.severity, "resolveiq.sentiment": c.sentiment,
                          "resolveiq.confidence.intent": c.confidence.intent, "resolveiq.confidence.product": c.confidence.product,
                          "resolveiq.confidence.severity": c.confidence.severity, "resolveiq.confidence.sentiment": c.confidence.sentiment,
                          "resolveiq.taxonomy_version": c.taxonomy_version})
    async def classify(self, ctx: QueryContext, strategy: str | None = None, allow_llm: bool = True) -> Classification:
        """Rules + kNN ensemble. With `allow_llm=False` the LLM fallback is skipped so the caller can run
        `refine()` concurrently with other work (the resolve pipeline overlaps it with generation)."""
        strategy = strategy or self.s.classifier_strategy
        tax = self.taxonomy.current
        if strategy == "rules":
            dist = await self.rules.predict(ctx, tax)
        elif strategy == "embedding":
            dist = await self.embedding.predict(ctx, tax)
        elif strategy == "llm":
            dist = await self.zero_shot.predict(ctx, tax)
        else:  # "ensemble" (LLM fallback per settings), "ensemble_llm" (fallback forced on), "ensemble_legacy" (no affect model)
            use_affect = self.affect is not None and self.affect.available and strategy != "ensemble_legacy"
            r, e, a = await asyncio.gather(
                self.rules.predict(ctx, tax), self.embedding.predict(ctx, tax),
                self.affect.predict(ctx, tax) if use_affect else _no_affect())
            dist = {}
            for dim in DIMENSIONS:
                if a[dim]:  # severity/sentiment come from the affect model: topic-neighbour voting cannot recover them
                    dist[dim] = a[dim]
                    continue
                wr, we = ENSEMBLE_WEIGHTS[dim]
                labels = set(r[dim]) | set(e[dim])
                dist[dim] = {l: wr * r[dim].get(l, 0.0) + we * e[dim].get(l, 0.0) for l in labels}
        result = self._to_result(dist, strategy, tax.version)
        if allow_llm and (strategy == "ensemble_llm" or (strategy == "ensemble" and self.s.classifier_llm_fallback)):
            result = await self.refine(ctx, result)
        return result

    async def refine(self, ctx: QueryContext, cls: Classification) -> Classification:
        """LLM zero-shot fallback for the dimensions whose confidence is below the threshold. Never raises."""
        affect_ok = self.affect is not None and self.affect.available and "legacy" not in cls.strategy
        weak = [d for d in DIMENSIONS if getattr(cls.confidence, d) < self.s.classifier_llm_threshold
                and not (affect_ok and d in ("severity", "sentiment"))]  # a 4B zero-shot LLM is weaker than the affect model here
        if not weak or not self.llm.available:
            return cls
        try:
            with tracing.span("classify.llm_fallback", {"resolveiq.classify.weak_dimensions": weak}):
                z = await self.zero_shot.predict(ctx, self.taxonomy.current)
        except LLMOverloaded:  # the model is busy with answers: keep the rule/kNN labels instead of competing with them
            return cls
        except Exception as exc:  # noqa: BLE001 - classification must never fail the request
            log.warning("llm classification fallback failed", extra={"error": str(exc)[:200]})
            return cls
        labels, conf = cls.model_dump(include=set(DIMENSIONS)), cls.confidence.model_dump()
        for dim in weak:
            if z[dim]:
                labels[dim], conf[dim] = next(iter(z[dim])), next(iter(z[dim].values()))
        return Classification(**labels, confidence=Confidence(**conf), strategy=f"{cls.strategy}+llm", taxonomy_version=cls.taxonomy_version)

    @staticmethod
    def _to_result(dist: Dist, strategy: str, version: int) -> Classification:
        picked, conf = {}, {}
        for dim in DIMENSIONS:
            d = dist.get(dim) or {}
            if d:
                label, p = max(d.items(), key=lambda kv: kv[1])
                picked[dim], conf[dim] = label, round(min(1.0, p), 3)
            else:
                picked[dim], conf[dim] = DEFAULTS.get(dim, "unknown"), 0.0
            CLASSIFICATION_CONFIDENCE.labels(dim).observe(conf[dim])
        return Classification(**picked, confidence=Confidence(**conf), strategy=strategy, taxonomy_version=version)
