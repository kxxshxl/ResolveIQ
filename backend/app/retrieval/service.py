"""Retrieval orchestration: lexical | bm25 | dense | hybrid (RRF) | hybrid_reranked.

Every strategy returns the same RetrievedItem shape so strategies can be benchmarked independently
and swapped via config / request parameter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.errors import RetrievalError, ValidationFailure
from app.db.repository import Repository
from app.models.schemas import RetrievedItem
from app.observability.metrics import RETRIEVAL_FAILURES, RETRIEVAL_LATENCY
from app.retrieval.bm25 import Bm25Index
from app.retrieval.fusion import rrf_fuse
from app.retrieval.reranker import CrossEncoderReranker
from app.services.embedding import EmbeddingService

STRATEGIES = ("lexical", "bm25", "dense", "hybrid", "hybrid_reranked")


@dataclass
class QueryContext:
    """Per-request state: the query, its embedding, and memoised candidate lists so classification
    and retrieval share one embedding call and one dense search."""

    text: str
    embedding: np.ndarray
    filters: dict[str, str] | None = None
    memo: dict[tuple, list[dict]] = field(default_factory=dict)


def _rerank_passage(kind: str, doc: dict) -> str:
    return doc["text"] if kind == "ticket" else f"{doc['title']}. {doc['text'][:700]}"


class RetrievalService:
    def __init__(self, repo: Repository, embedder: EmbeddingService, reranker: CrossEncoderReranker, settings: Settings):
        self.repo, self.embedder, self.reranker, self.s = repo, embedder, reranker, settings
        self._bm25_cache: dict[str, tuple[int, Bm25Index]] = {}
        self._df_cache: dict[str, tuple[int, int, dict[str, int]]] = {}

    async def make_context(self, text: str, filters: dict | None = None) -> QueryContext:
        return QueryContext(text=text, embedding=await self.embedder.embed_query(text), filters=filters or None)

    # ------------------------------------------------------------- candidate generators (memoised)
    async def _dense(self, ctx: QueryContext, kind: str, k: int) -> list[dict]:
        key = ("dense", kind, k, tuple(sorted((ctx.filters or {}).items())))
        if key not in ctx.memo:
            ctx.memo[key] = await self.repo.dense_search(kind, ctx.embedding, k, self.embedder.model_name, ctx.filters)
        return ctx.memo[key]

    async def _discriminative_terms(self, text: str, kind: str) -> list[str]:
        """Postgres FTS ranking has no IDF, so a long complaint ORed term-by-term is dominated by common words.
        We prune the query to its rarest terms (document frequency from ts_stat, cached per corpus version)."""
        version = await self.repo.corpus_version()
        cached = self._df_cache.get(kind)
        if not cached or cached[0] != version:
            n, df = await self.repo.doc_frequencies(kind)
            self._df_cache[kind] = cached = (version, n, df)
        _, n, df = cached
        terms = list(dict.fromkeys(await self.repo.query_lexemes(text)))
        known = sorted((t for t in terms if t in df), key=lambda t: df[t])
        kept = [t for t in known if df[t] / max(n, 1) <= self.s.lexical_max_df][: self.s.lexical_max_terms]
        return kept or known[:3]

    async def _lexical(self, ctx: QueryContext, kind: str, k: int) -> list[dict]:
        key = ("lex", kind, k, tuple(sorted((ctx.filters or {}).items())))
        if key not in ctx.memo:
            terms = await self._discriminative_terms(ctx.text, kind)
            ctx.memo[key] = await self.repo.lexical_search(kind, terms, k, ctx.filters)
        return ctx.memo[key]

    async def _bm25(self, ctx: QueryContext, kind: str, k: int) -> list[dict]:
        version = await self.repo.corpus_version()
        cached = self._bm25_cache.get(kind)
        if not cached or cached[0] != version:
            self._bm25_cache[kind] = (version, Bm25Index(await self.repo.all_docs(kind)))
        hits = self._bm25_cache[kind][1].search(ctx.text, k * 3 if ctx.filters else k)
        if ctx.filters:
            hits = [h for h in hits if all(h.get(f) == v for f, v in ctx.filters.items())][:k]
        return hits

    async def dense_candidates(self, ctx: QueryContext, kind: str, k: int) -> list[dict]:
        return await self._dense(ctx, kind, k)

    async def dense_top_similarity(self, ctx: QueryContext, kind: str) -> float:
        rows = await self._dense(ctx, kind, 5)
        return float(rows[0]["score"]) if rows else 0.0

    # ------------------------------------------------------------- public API
    async def search(self, ctx: QueryContext, kind: str, strategy: str, k: int, offset: int = 0) -> list[RetrievedItem]:
        if strategy not in STRATEGIES:
            raise ValidationFailure(f"unknown retrieval strategy '{strategy}'; choose from {STRATEGIES}")
        if strategy == "hybrid_reranked" and not self.reranker.enabled:
            strategy = "hybrid"
        t0 = time.perf_counter()
        try:
            items = await self._run(ctx, kind, strategy, k + offset)
        except RetrievalError:
            RETRIEVAL_FAILURES.labels(strategy).inc()
            raise
        except Exception as exc:  # noqa: BLE001
            RETRIEVAL_FAILURES.labels(strategy).inc()
            raise RetrievalError(f"retrieval failed ({strategy}/{kind}): {type(exc).__name__}: {exc}") from exc
        finally:
            RETRIEVAL_LATENCY.labels(strategy, kind).observe(time.perf_counter() - t0)
        return items[offset:]

    async def _run(self, ctx: QueryContext, kind: str, strategy: str, k: int) -> list[RetrievedItem]:
        cand = self.s.candidate_k
        if strategy in ("lexical", "bm25", "dense"):
            rows = await {"lexical": self._lexical, "bm25": self._bm25, "dense": self._dense}[strategy](ctx, kind, max(k, 1))
            stage = {"lexical": "lexical", "bm25": "bm25", "dense": "dense_cosine"}[strategy]
            return [self._item(kind, r, i, strategy, r["score"], {stage: float(r["score"])}) for i, r in enumerate(rows[:k], 1)]

        dense = await self._dense(ctx, kind, cand)
        lex = await self._lexical(ctx, kind, cand)
        by_id: dict[str, dict] = {}
        for r in (*lex, *dense):
            by_id.setdefault(r["source_id"], r)
        fused = rrf_fuse({"dense": [r["source_id"] for r in dense], "lexical": [r["source_id"] for r in lex]}, k=self.s.rrf_k,
                         weights={"dense": self.s.rrf_weight_dense, "lexical": self.s.rrf_weight_lexical})
        dense_sim = {r["source_id"]: float(r["score"]) for r in dense}
        lex_score = {r["source_id"]: float(r["score"]) for r in lex}

        def stages(sid: str, rrf: float, ranks: dict[str, int]) -> dict[str, float]:
            out = {"rrf": rrf}
            if sid in dense_sim:
                out["dense_cosine"] = dense_sim[sid]
                out["dense_rank"] = float(ranks["dense"])
            if sid in lex_score:
                out["lexical"] = lex_score[sid]
                out["lexical_rank"] = float(ranks["lexical"])
            return out

        if strategy == "hybrid":
            return [self._item(kind, by_id[sid], i, "hybrid", rrf, stages(sid, rrf, ranks))
                    for i, (sid, rrf, ranks) in enumerate(fused[:k], 1)]

        top = fused[: self.s.rerank_top_n]
        probs = await self.reranker.score(ctx.text, [_rerank_passage(kind, by_id[sid]) for sid, _, _ in top])
        a = self.s.rerank_blend
        top_rrf = max((r for _, r, _ in top), default=1.0) or 1.0
        blended = [(a * p + (1 - a) * (rrf / top_rrf), p, sid, rrf, ranks) for (sid, rrf, ranks), p in zip(top, probs)]
        blended.sort(key=lambda x: (-x[0], -x[3]))
        items = []
        for i, (final, p, sid, rrf, ranks) in enumerate(blended[:k], 1):
            st = stages(sid, rrf, ranks)
            st["rerank_prob"] = p
            items.append(self._item(kind, by_id[sid], i, "hybrid_reranked", final, st))
        return items

    @staticmethod
    def _item(kind: str, row: dict[str, Any], rank: int, method: str, score: float, stages: dict[str, float]) -> RetrievedItem:
        meta = {"intent": row.get("intent"), "product": row.get("product")}
        if kind == "ticket":
            meta.update(severity=row.get("severity"), sentiment=row.get("sentiment"), resolved_at=str(row.get("resolved_at")))
        else:
            meta.update(tags=row.get("tags") or [], updated_at=str(row.get("updated_at")))
        steps = row.get("steps") or []
        return RetrievedItem(
            source_type=kind, source_id=row["source_id"], title=row["title"], text=row["text"], score=round(float(score), 5),
            rank=rank, retrieval_method=method, metadata=meta, steps=[str(s) for s in steps],
            resolution_summary=row.get("resolution_summary"), scores={k: round(v, 5) for k, v in stages.items()},
        )
