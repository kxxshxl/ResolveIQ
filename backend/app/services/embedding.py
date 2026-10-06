"""Embedding service: lazy model load, batching, query-embedding cache, async-safe wrappers."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Sequence

import numpy as np

from app.core.config import Settings
from app.observability import tracing
from app.observability.metrics import EMBEDDING_CACHE, EMBEDDING_LATENCY
from app.services.cache import Cache

log = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings, cache: Cache | None = None):
        self.s = settings
        self.cache = cache
        self.model_name = settings.embedding_model
        self._model = None
        self._lock = asyncio.Lock()
        self._units: OrderedDict[str, np.ndarray] = OrderedDict()  # LRU: evidence text -> embedding (touched only from the event loop)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            t0 = time.perf_counter()
            self._model = SentenceTransformer(self.model_name)
            log.info("embedding model loaded", extra={"model": self.model_name, "seconds": round(time.perf_counter() - t0, 2)})
        return self._model

    def warmup(self) -> None:
        self._load().encode(["warmup"], normalize_embeddings=True)

    @tracing.traced("embed.encode", attrs=lambda self, texts: {"resolveiq.embed.model": self.model_name, "resolveiq.batch_size": len(texts)})
    def encode_sync(self, texts: Sequence[str]) -> np.ndarray:
        """Normalised embeddings (cosine == dot product). Batched for throughput."""
        with_timer = time.perf_counter()
        vecs = self._load().encode(
            list(texts), batch_size=self.s.embedding_batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        EMBEDDING_LATENCY.labels("batch" if len(texts) > 1 else "single").observe(time.perf_counter() - with_timer)
        return np.asarray(vecs, dtype=np.float32)

    async def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        return await asyncio.to_thread(self.encode_sync, texts)

    async def embed_cached(self, texts: Sequence[str]) -> np.ndarray:
        """Embeddings for many short texts, row-aligned with `texts`. Each distinct text is embedded once per call, and recently seen texts are
        served from a bounded in-process LRU: citation validation re-embeds the same corpus sentences for every request that cites them."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        unique = list(dict.fromkeys(texts))
        size = self.s.embedding_unit_cache_size
        found: dict[str, np.ndarray] = {}
        for t in unique:
            vec = self._units.get(t) if size else None
            if vec is not None:
                self._units.move_to_end(t)
                found[t] = vec
        missing = [t for t in unique if t not in found]
        EMBEDDING_CACHE.labels("unit_hit").inc(len(found))
        EMBEDDING_CACHE.labels("unit_miss").inc(len(missing))
        tracing.annotate(resolveiq__embed__unique=len(unique), resolveiq__embed__cache_hits=len(found), resolveiq__embed__duplicates=len(texts) - len(unique))
        if missing:
            for t, vec in zip(missing, await self.embed_batch(missing)):
                found[t] = vec
                if size:
                    self._units[t] = vec
            while size and len(self._units) > size:
                self._units.popitem(last=False)
        return np.stack([found[t] for t in texts])

    @tracing.traced("embed.query", attrs=lambda self, text: {"resolveiq.embed.model": self.model_name})
    async def embed_query(self, text: str) -> np.ndarray:
        key = f"emb:{self.model_name}:{hashlib.sha256(text.encode()).hexdigest()[:32]}"
        if self.cache:
            hit = await self.cache.get(key)
            tracing.annotate(resolveiq__embed__cache_hit=hit is not None)
            if hit is not None:
                EMBEDDING_CACHE.labels("hit").inc()
                return np.asarray(hit, dtype=np.float32)
            EMBEDDING_CACHE.labels("miss").inc()
        vec = (await self.embed_batch([text]))[0]
        if self.cache:
            await self.cache.set(key, vec.tolist(), ttl=3600)
        return vec
