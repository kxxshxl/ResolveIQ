"""Service container built once per process (stateless API: all state lives in Postgres/Redis)."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.classification.affect import AffectModel
from app.classification.pipeline import ComplaintClassifier
from app.classification.taxonomy import TaxonomyService
from app.core.config import Settings
from app.db.repository import Repository
from app.discovery.service import DiscoveryService
from app.ingestion.pipeline import IngestionService
from app.rag.pipeline import ResolutionService
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.service import RetrievalService
from app.services.cache import Cache
from app.services.embedding import EmbeddingService
from app.services.llm.base import ResilientLLM
from app.services.llm.providers import build_llm

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    repo: Repository
    cache: Cache
    embedder: EmbeddingService
    reranker: CrossEncoderReranker
    taxonomy: TaxonomyService
    retrieval: RetrievalService
    llm: ResilientLLM
    classifier: ComplaintClassifier
    ingestion: IngestionService
    resolution: ResolutionService
    discovery: DiscoveryService
    affect: AffectModel
    models_ready: bool = False

    @classmethod
    async def build(cls, settings: Settings, llm: ResilientLLM | None = None) -> "Services":
        repo = Repository(settings)
        await repo.open()
        cache = Cache(settings.redis_url, settings.cache_ttl_seconds, settings.cache_namespace)
        await cache.open()
        embedder = EmbeddingService(settings, cache)
        reranker = CrossEncoderReranker(settings)
        llm = llm or build_llm(settings)
        taxonomy = TaxonomyService(repo, embedder)
        retrieval = RetrievalService(repo, embedder, reranker, settings)
        affect = AffectModel(settings)
        classifier = ComplaintClassifier(taxonomy, retrieval, llm, settings, affect)
        ingestion = IngestionService(repo, embedder, taxonomy, classifier, settings)
        resolution = ResolutionService(repo, retrieval, classifier, taxonomy, llm, embedder, cache, settings)
        discovery = DiscoveryService(repo, retrieval, embedder, taxonomy, settings)
        svc = cls(settings, repo, cache, embedder, reranker, taxonomy, retrieval, llm, classifier, ingestion, resolution, discovery, affect)
        # load models off the event loop so startup does not block health probes' event loop
        await asyncio.to_thread(embedder.warmup)
        await asyncio.to_thread(reranker.warmup)
        await asyncio.to_thread(affect.load)
        svc.models_ready = True
        await taxonomy.refresh()
        await ingestion.refresh_gauges()
        return svc

    async def close(self) -> None:
        await self.cache.close()
        await self.repo.close()
