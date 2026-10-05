"""Cross-encoder reranker (query, passage) -> relevance probability. Loaded lazily; runs in a thread."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Sequence

from app.core.config import Settings
from app.observability.metrics import RERANK_LATENCY

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, settings: Settings):
        self.model_name = settings.reranker_model
        self.enabled = settings.reranker_enabled
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            t0 = time.perf_counter()
            self._model = CrossEncoder(self.model_name)
            log.info("reranker loaded", extra={"model": self.model_name, "seconds": round(time.perf_counter() - t0, 2)})
        return self._model

    def warmup(self) -> None:
        if self.enabled:
            self._load().predict([("warmup", "warmup")])

    def score_sync(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        t0 = time.perf_counter()
        logits = self._load().predict([(query, p) for p in passages], batch_size=32, show_progress_bar=False)
        RERANK_LATENCY.observe(time.perf_counter() - t0)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return await asyncio.to_thread(self.score_sync, query, passages)
