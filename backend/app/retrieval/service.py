"""Retrieval orchestration: lexical | bm25 | dense | hybrid (RRF) | hybrid_reranked | adaptive.

Every strategy returns the same RetrievedItem shape so strategies can be benchmarked independently
and swapped via config / request parameter.

`adaptive` is a bounded escalation ladder rather than a fixed pipeline: it starts with the cheapest retriever (dense) and only spends more
(a lexical leg with a reformulated query, then the cross-encoder) when the evidence it just found looks weak. Every decision is recorded on the
query context so a resolution can show why it did or did not escalate.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.errors import RetrievalError, ValidationFailure
from app.db.repository import Repository
from app.models.schemas import RetrievedItem
from app.observability import tracing
from app.observability.metrics import ADAPTIVE_STAGES, RETRIEVAL_FAILURES, RETRIEVAL_LATENCY
from app.retrieval.bm25 import Bm25Index
from app.retrieval.fusion import mmr_order, rrf_fuse
from app.retrieval.reranker import CrossEncoderReranker
from app.services.embedding import EmbeddingService

STRATEGIES = ("lexical", "bm25", "dense", "hybrid", "hybrid_reranked", "adaptive")


@dataclass
class QueryContext:
    """Per-request state: the query, its embedding, and memoised candidate lists so classification
    and retrieval share one embedding call and one dense search."""

    text: str
    embedding: np.ndarray
    filters: dict[str, str] | None = None
    memo: dict[tuple, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)     # accumulated milliseconds per retrieval step (dense, lexical, rerank, ...)
    decisions: list[dict] = field(default_factory=list)         # adaptive retrieval: one entry per stage taken


def _rerank_passage(kind: str, doc: dict) -> str:
    return doc["text"] if kind == "ticket" else f"{doc['title']}. {doc['text'][:700]}"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


@contextmanager
def _timed(ctx: QueryContext, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ctx.timings[key] = ctx.timings.get(key, 0.0) + (time.perf_counter() - t0) * 1000


class RetrievalService:
    def __init__(self, repo: Repository, embedder: EmbeddingService, reranker: CrossEncoderReranker, settings: Settings):
        self.repo, self.embedder, self.reranker, self.s = repo, embedder, reranker, settings
        self._bm25_cache: dict[str, tuple[int, Bm25Index]] = {}
        self._df_cache: dict[str, tuple[int, int, dict[str, int]]] = {}

    async def make_context(self, text: str, filters: dict | None = None) -> QueryContext:
        return QueryContext(text=text, embedding=await self.embedder.embed_query(text), filters=filters or None)

    def config_snapshot(self, strategy: str, ctx: QueryContext | None = None) -> dict:
        """The retrieval configuration that produced a result (recorded in provenance)."""
        s = self.s
        snap: dict[str, Any] = {
            "strategy": strategy, "ticket_top_k": s.ticket_top_k, "article_top_k": s.article_top_k, "evidence_tickets": s.evidence_tickets,
            "evidence_articles": s.evidence_articles, "candidate_k": s.candidate_k, "filters": sorted((ctx.filters or {}) if ctx else {}),
            "hnsw_ef_search": s.hnsw_ef_search}
        if strategy in ("hybrid", "hybrid_reranked", "adaptive"):
            snap.update(rrf_k=s.rrf_k, rrf_weights={"dense": s.rrf_weight_dense, "lexical": s.rrf_weight_lexical})
        if strategy in ("hybrid_reranked", "adaptive"):
            snap.update(reranker=self.reranker.model_name if self.reranker.enabled else None, rerank_blend=s.rerank_blend,
                        rerank_top_n=s.rerank_top_n if strategy == "hybrid_reranked" else s.adaptive_rerank_top_n)
        if strategy == "adaptive":
            snap["adaptive"] = {"margin_ticket": s.adaptive_margin_ticket, "margin_article": s.adaptive_margin_article, "rerank_gap": s.adaptive_rerank_gap,
                                "expand_kinds": s.adaptive_expand_kinds, "expansion_terms": s.adaptive_expansion_terms, "mmr": s.adaptive_mmr,
                                "decisions": list(ctx.decisions) if ctx else []}
        return snap

    # ------------------------------------------------------------- candidate generators (memoised)
    async def _dense(self, ctx: QueryContext, kind: str, k: int) -> list[dict]:
        flt = tuple(sorted((ctx.filters or {}).items()))
        key = ("dense", kind, k, flt)
        if key not in ctx.memo:
            for mk, rows in ctx.memo.items():     # a longer list for the same query already answers a shorter request
                if mk[0] == "dense" and mk[1] == kind and mk[3] == flt and mk[2] >= k:
                    return rows[:k]
            with _timed(ctx, "dense_ms"):
                ctx.memo[key] = await self.repo.dense_search(kind, ctx.embedding, k, self.embedder.model_name, ctx.filters)
        return ctx.memo[key]

    async def _df(self, kind: str) -> tuple[int, dict[str, int]]:
        version = await self.repo.corpus_version()
        cached = self._df_cache.get(kind)
        if not cached or cached[0] != version:
            n, df = await self.repo.doc_frequencies(kind)
            self._df_cache[kind] = cached = (version, n, df)
        return cached[1], cached[2]

    async def _discriminative_terms(self, text: str, kind: str) -> list[str]:
        """Postgres FTS ranking has no IDF, so a long complaint ORed term-by-term is dominated by common words.
        We prune the query to its rarest terms (document frequency from ts_stat, cached per corpus version)."""
        n, df = await self._df(kind)
        terms = list(dict.fromkeys(await self.repo.query_lexemes(text)))
        known = sorted((t for t in terms if t in df), key=lambda t: df[t])
        kept = [t for t in known if df[t] / max(n, 1) <= self.s.lexical_max_df][: self.s.lexical_max_terms]
        return kept or known[:3]

    async def _lexical(self, ctx: QueryContext, kind: str, k: int, extra_terms: tuple[str, ...] = ()) -> list[dict]:
        key = ("lex", kind, k, tuple(sorted((ctx.filters or {}).items())), extra_terms)
        if key not in ctx.memo:
            with _timed(ctx, "lexical_ms"):
                terms = list(dict.fromkeys([*await self._discriminative_terms(ctx.text, kind), *extra_terms]))
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
    @tracing.traced(
        "retrieval.search",
        attrs=lambda self, ctx, kind, strategy, k, offset=0: {"resolveiq.retrieval.kind": kind, "resolveiq.retrieval.strategy": strategy,
                                                              "resolveiq.retrieval.k": k, "resolveiq.retrieval.offset": offset,
                                                              "resolveiq.retrieval.filters": sorted(ctx.filters or {})},
        result=lambda items: {"resolveiq.retrieval.returned": len(items), "resolveiq.retrieval.top_score": items[0].score if items else None})
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

    async def _fuse(self, ctx: QueryContext, kind: str, cand: int, extra_terms: tuple[str, ...] = (),
                    extra_dense: list[dict] | None = None):
        """Dense + lexical candidates fused with weighted RRF. Returns (by_id, fused, stage-score helper)."""
        dense = await self._dense(ctx, kind, cand)
        lex = await self._lexical(ctx, kind, cand, extra_terms)
        rankings = {"dense": [r["source_id"] for r in dense], "lexical": [r["source_id"] for r in lex]}
        weights = {"dense": self.s.rrf_weight_dense, "lexical": self.s.rrf_weight_lexical}
        by_id: dict[str, dict] = {}
        for r in (*lex, *dense, *(extra_dense or [])):
            by_id.setdefault(r["source_id"], r)
        if extra_dense:
            rankings["dense_reformulated"] = [r["source_id"] for r in extra_dense]
            weights["dense_reformulated"] = self.s.rrf_weight_dense
        fused = rrf_fuse(rankings, k=self.s.rrf_k, weights=weights)
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

        return by_id, fused, stages

    async def _rerank(self, ctx: QueryContext, kind: str, fused: list, by_id: dict, stages, n: int, method: str, k: int) -> list[RetrievedItem]:
        top = fused[:n]
        with tracing.span("retrieval.rerank", {"resolveiq.rerank.model": getattr(self.reranker, "model_name", None),
                                               "resolveiq.rerank.candidates": len(top), "resolveiq.retrieval.kind": kind}) as sp:
            with _timed(ctx, "rerank_ms"):
                probs = await self.reranker.score(ctx.text, [_rerank_passage(kind, by_id[sid]) for sid, _, _ in top])
            if len(probs) and sp.is_recording():
                sp.set_attribute("resolveiq.rerank.top_probability", float(max(probs)))
        a = self.s.rerank_blend
        top_rrf = max((r for _, r, _ in top), default=1.0) or 1.0
        blended = [(a * p + (1 - a) * (rrf / top_rrf), p, sid, rrf, ranks) for (sid, rrf, ranks), p in zip(top, probs)]
        blended.sort(key=lambda x: (-x[0], -x[3]))
        items = []
        for i, (final, p, sid, rrf, ranks) in enumerate(blended[:k], 1):
            st = stages(sid, rrf, ranks)
            st["rerank_prob"] = p
            items.append(self._item(kind, by_id[sid], i, method, final, st))
        return items

    async def _run(self, ctx: QueryContext, kind: str, strategy: str, k: int) -> list[RetrievedItem]:
        cand = self.s.candidate_k
        if strategy == "adaptive":
            return await self._adaptive(ctx, kind, k)
        if strategy in ("lexical", "bm25", "dense"):
            rows = await {"lexical": self._lexical, "bm25": self._bm25, "dense": self._dense}[strategy](ctx, kind, max(k, 1))
            stage = {"lexical": "lexical", "bm25": "bm25", "dense": "dense_cosine"}[strategy]
            return [self._item(kind, r, i, strategy, r["score"], {stage: float(r["score"])}) for i, r in enumerate(rows[:k], 1)]

        by_id, fused, stages = await self._fuse(ctx, kind, cand)
        if strategy == "hybrid":
            return [self._item(kind, by_id[sid], i, "hybrid", rrf, stages(sid, rrf, ranks))
                    for i, (sid, rrf, ranks) in enumerate(fused[:k], 1)]
        return await self._rerank(ctx, kind, fused, by_id, stages, self.s.rerank_top_n, "hybrid_reranked", k)

    # ------------------------------------------------------------- adaptive
    @staticmethod
    def ambiguity(items: list[RetrievedItem]) -> float:
        """Gap between the best and second-best dense cosine (smaller = more ambiguous).

        Why this signal: on the validation split the best similarity says almost nothing about whether the top result is right (AUC 0.50 for tickets), but the gap
        to the runner-up does (AUC 0.62 for tickets, 0.87 for articles): when two sources look equally close, the ranking is a coin flip and a second opinion pays.
        """
        sims = [i.scores["dense_cosine"] for i in items[:2] if "dense_cosine" in i.scores]
        return float(sims[0] - sims[1]) if len(sims) == 2 else float("inf")

    async def _expansion_terms(self, ctx: QueryContext, kind: str, dense_rows: list[dict]) -> list[str]:
        """Pseudo-relevance feedback: rare terms that at least two of the three best dense candidates share and the query does not already contain."""
        top = dense_rows[:3]
        if len(top) < 2 or self.s.adaptive_expansion_terms <= 0:
            return []
        lexemes = await self.repo.query_lexemes_many([r["text"] + " " + (r.get("title") or "") for r in top] + [ctx.text])
        query_terms, doc_terms = set(lexemes[-1]), [set(x) for x in lexemes[:-1]]
        n, df = await self._df(kind)
        counts: dict[str, int] = {}
        for ts in doc_terms:
            for t in ts:
                counts[t] = counts.get(t, 0) + 1
        shared = [t for t, c in counts.items() if c >= 2 and t not in query_terms and t in df and df[t] / max(n, 1) <= self.s.lexical_max_df]
        return sorted(shared, key=lambda t: (df[t], t))[: self.s.adaptive_expansion_terms]

    def _decide(self, ctx: QueryContext, kind: str, stage: str, signal: str, value: float, threshold: float | None, action: str, reason: str, **extra) -> None:
        ctx.decisions.append({"kind": kind, "stage": stage, "signal": signal, "value": None if value == float("inf") else round(value, 4), "threshold": threshold,
                              "action": action, "reason": reason, "elapsed_ms": round(sum(ctx.timings.values()), 1), **extra})
        ADAPTIVE_STAGES.labels(kind, stage, action).inc()
        tracing.event("retrieval.adaptive.decision", resolveiq__retrieval__kind=kind, resolveiq__adaptive__stage=stage, resolveiq__adaptive__signal=signal,
                      resolveiq__adaptive__value=None if value == float("inf") else round(value, 4), resolveiq__adaptive__action=action)

    async def _adaptive(self, ctx: QueryContext, kind: str, k: int) -> list[RetrievedItem]:
        """Escalation ladder, each rung taken only when the previous result is ambiguous:
        1. dense (always)  ->  2. hybrid: dense + a lexical leg reformulated with terms the best candidates share  ->  3. cross-encoder rerank (optional, off by default)."""
        s, cand = self.s, self.s.candidate_k
        dense = await self._dense(ctx, kind, cand)
        first = [self._item(kind, r, i, "adaptive", r["score"], {"dense_cosine": float(r["score"]), "adaptive_stage": 0.0}) for i, r in enumerate(dense[: max(k, 5)], 1)]
        margin = self.ambiguity(first)
        tau = s.adaptive_margin_ticket if kind == "ticket" else s.adaptive_margin_article
        if not dense or margin >= tau:
            self._decide(ctx, kind, "dense", "margin", margin, tau, "stop", "rank 1 clearly ahead of rank 2" if dense else "nothing retrieved")
            return first[:k]
        self._decide(ctx, kind, "dense", "margin", margin, tau, "escalate", "rank 1 and rank 2 are nearly tied: add a lexical leg")

        extra = tuple(await self._expansion_terms(ctx, kind, dense)) if kind in s.adaptive_expand_kinds.split(",") else ()
        by_id, fused, stages = await self._fuse(ctx, kind, cand, extra)
        hybrid = [self._item(kind, by_id[sid], i, "adaptive", rrf, {**stages(sid, rrf, ranks), "adaptive_stage": 1.0})
                  for i, (sid, rrf, ranks) in enumerate(fused[: max(k, 5)], 1)]
        gap = (fused[0][1] - fused[1][1]) / fused[0][1] if len(fused) > 1 and fused[0][1] else float("inf")
        if s.adaptive_rerank_gap <= 0 or gap >= s.adaptive_rerank_gap or not self.reranker.enabled:
            why = "reranking is not enabled for this ladder" if s.adaptive_rerank_gap <= 0 else ("fused rank 1 clearly ahead" if self.reranker.enabled else "no reranker available")
            self._decide(ctx, kind, "hybrid", "rrf_gap", gap, s.adaptive_rerank_gap, "stop", why, expansion_terms=list(extra))
            return await self._finalise(ctx, kind, hybrid[:k], k)
        self._decide(ctx, kind, "hybrid", "rrf_gap", gap, s.adaptive_rerank_gap, "escalate", "fused top results nearly tied: rerank them", expansion_terms=list(extra))
        items = await self._rerank(ctx, kind, fused, by_id, stages, s.adaptive_rerank_top_n, "adaptive", max(k, 5))
        for it in items:
            it.scores["adaptive_stage"] = 2.0
        self._decide(ctx, kind, "rerank", "rerank_prob", items[0].scores.get("rerank_prob", 0.0) if items else 0.0, None, "stop", "reranked; the ladder is finished")
        return await self._finalise(ctx, kind, items, k)

    async def _finalise(self, ctx: QueryContext, kind: str, items: list[RetrievedItem], k: int) -> list[RetrievedItem]:
        if not self.s.adaptive_mmr or len(items) <= 2:
            return items[:k]
        ids = [i.source_id for i in items]
        vecs = await self.repo.get_embeddings(kind, ids, self.embedder.model_name)
        keep = [i for i in items if i.source_id in vecs]
        if len(keep) <= 2:
            return items[:k]
        rel = np.array([_clip01(i.scores.get("rerank_prob", i.scores.get("dense_cosine", i.score))) for i in keep])
        order = mmr_order(rel, np.stack([vecs[i.source_id] for i in keep]), k, self.s.adaptive_mmr_lambda)
        out = []
        for new_rank, idx in enumerate(order, 1):
            it = keep[idx].model_copy(update={"rank": new_rank})
            it.scores["mmr"] = round(float(rel[idx]), 5)
            out.append(it)
        return out

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
