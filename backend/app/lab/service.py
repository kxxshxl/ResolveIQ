"""Retrieval lab service.

Runs one (redacted, never stored) complaint through each retrieval strategy and returns, per strategy, the ranked tickets and KB articles, their scores, the
latency of that strategy alone (the query embedding is computed once and reported separately) and, when ground truth is known, whether each result is relevant.

Ground truth exists for the project's labelled evaluation queries: a result is relevant when it belongs to the same root-cause scenario as the query.
For any other complaint relevance is unknown and is reported as such, never guessed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import ValidationFailure
from app.core.pii import redact
from app.core.text import normalize_text
from app.db.repository import Repository
from app.evaluation.datasets import Corpus, build_corpus, load_eval_sets
from app.evaluation.metrics import retrieval_metrics
from app.observability import tracing
from app.retrieval.service import STRATEGIES, QueryContext, RetrievalService

LABELS = {
    "lexical": ("Keyword", "Postgres full-text search on the rarest terms of the complaint (what agents get today)"),
    "bm25": ("BM25", "in-memory BM25 keyword baseline"),
    "dense": ("Dense (semantic)", "pgvector cosine similarity of sentence embeddings"),
    "hybrid": ("Hybrid", "dense + keyword results fused with weighted reciprocal rank fusion"),
    "hybrid_reranked": ("Hybrid + reranker", "hybrid results re-scored by a cross-encoder"),
    "adaptive": ("Adaptive", "dense first; adds a keyword leg only when the top two results are nearly tied"),
}
DEFAULT_STRATEGIES = ("lexical", "dense", "hybrid", "hybrid_reranked", "adaptive")


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


@dataclass
class _Truth:
    scenario_id: str
    source: str     # which labelled set the query came from, or "supplied"
    intent: str | None = None


class LabService:
    def __init__(self, repo: Repository, retrieval: RetrievalService, settings: Settings):
        self.repo, self.retrieval, self.s = repo, retrieval, settings
        self._corpus: tuple[int, Corpus] | None = None
        self._sets = None

    async def _corpus_for_truth(self) -> Corpus:
        v = await self.repo.corpus_version()
        if not self._corpus or self._corpus[0] != v:
            self._corpus = (v, await build_corpus(self.repo))
        return self._corpus[1]

    def _eval_sets(self):
        if self._sets is None:
            try:
                self._sets = load_eval_sets(self.s.data_dir)
            except Exception:  # noqa: BLE001 - a deployment without the evaluation data still gets the lab, without ground truth
                self._sets = False
        return self._sets or None

    def examples(self, limit: int = 60) -> list[dict]:
        """Labelled complaints to try (spread over scenarios), with the hand-written ones first because they are the realistic test."""
        sets = self._eval_sets()
        if not sets:
            return []
        out, seen = [], set()
        for split, rows in (("gold", sets.gold), ("blind", sets.blind), ("test", sets.test)):
            for r in rows:
                if len(out) >= limit:
                    return out
                key = (split, r["scenario_id"])
                if key in seen:
                    continue
                seen.add(key)
                out.append({"qid": r["qid"], "split": split, "text": r["text"], "scenario_id": r["scenario_id"], "intent": r.get("intent")})
        return out

    def _truth(self, text: str, scenario_id: str | None) -> _Truth | None:
        if scenario_id:
            return _Truth(scenario_id, "supplied")
        sets = self._eval_sets()
        if not sets:
            return None
        n = _norm(text)
        for split, rows in (("gold", sets.gold), ("blind", sets.blind), ("test", sets.test), ("val", sets.val)):
            for r in rows:
                if _norm(r["text"]) == n:
                    return _Truth(r["scenario_id"], split, r.get("intent"))
        return None

    @tracing.traced("lab.compare", attrs=lambda self, complaint, strategies=None, k=5, scenario_id=None: {"resolveiq.complaint_chars": len(complaint), "resolveiq.retrieval.k": k},
                    result=lambda r: {"resolveiq.lab.ground_truth": r["ground_truth"]["available"], "resolveiq.lab.strategies": len(r["strategies"])})
    async def compare(self, complaint: str, strategies: list[str] | None = None, k: int = 5, scenario_id: str | None = None) -> dict:
        names = list(strategies or DEFAULT_STRATEGIES)
        bad = [n for n in names if n not in STRATEGIES]
        if bad:
            raise ValidationFailure(f"unknown retrieval strategies {bad}; choose from {STRATEGIES}")
        text = redact(normalize_text(complaint)).text
        truth = self._truth(complaint, scenario_id) or self._truth(text, scenario_id)
        corpus = await self._corpus_for_truth() if truth else None
        t0 = time.perf_counter()
        base = await self.retrieval.make_context(text)
        embed_ms = round((time.perf_counter() - t0) * 1000, 1)
        out_strategies = []
        ranks: dict[str, dict[str, int]] = {}
        for name in names:
            entry: dict = {"name": name, "label": LABELS[name][0], "description": LABELS[name][1], "kinds": {}}
            total = 0.0
            for kind in ("ticket", "article"):
                ctx = QueryContext(text=text, embedding=base.embedding)    # fresh memo: each strategy pays for its own work
                t = time.perf_counter()
                items = await self.retrieval.search(ctx, kind, name, k)
                ms = (time.perf_counter() - t) * 1000
                total += ms
                relevant = corpus.relevant(kind, truth.scenario_id) if truth and corpus else None
                rows = []
                for it in items:
                    rows.append({"id": it.source_id, "rank": it.rank, "score": it.score, "title": it.resolution_summary or it.title if kind == "ticket" else it.title,
                                 "excerpt": " ".join(it.text.split())[:170], "intent": it.metadata.get("intent"), "scores": it.scores,
                                 "relevant": (it.source_id in relevant) if relevant is not None else None})
                    ranks.setdefault(it.source_id, {})[name] = it.rank
                summary = None
                if relevant is not None:
                    m = retrieval_metrics([i.source_id for i in items], relevant, (1, k))
                    first = next((i.rank for i in items if i.source_id in relevant), None)
                    summary = {"first_relevant_rank": first, "hit@1": m["hit@1"], f"precision@{k}": round(m[f"p@{k}"], 3), "mrr": round(m["mrr"], 3)}
                entry["kinds"][kind] = {"results": rows, "latency_ms": round(ms, 2), "summary": summary,
                                        **({"adaptive": ctx.decisions} if name == "adaptive" else {})}
            entry["latency_ms"] = round(total, 2)
            out_strategies.append(entry)
        return {"complaint": text, "k": k, "embedding_ms": embed_ms, "strategies": out_strategies, "rank_movement": ranks,
                "ground_truth": {"available": truth is not None, "scenario_id": truth.scenario_id if truth else None, "source": truth.source if truth else None,
                                 "note": ("relevant = same root-cause scenario as this labelled query" if truth else
                                          "no ground truth for a free-text complaint: relevance is not guessed. Pick a labelled example to see it.")}}
