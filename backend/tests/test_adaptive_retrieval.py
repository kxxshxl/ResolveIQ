"""Adaptive retrieval: a bounded escalation ladder (dense -> hybrid -> optional rerank) driven by how ambiguous the first result is."""
import numpy as np
import pytest

from app.models.schemas import RetrievedItem
from app.retrieval.fusion import mmr_order
from app.retrieval.service import QueryContext, RetrievalService

Q = "My broadband keeps dropping around 8 PM each day and I've restarted the router twice already."


def _item(sim):
    return RetrievedItem(source_type="ticket", source_id="x", title="t", text="t", score=sim, rank=1, retrieval_method="dense", scores={"dense_cosine": sim})


# ---------------------------------------------------------------- pure pieces
def test_ambiguity_is_the_gap_between_the_top_two_dense_scores():
    assert RetrievalService.ambiguity([_item(0.71), _item(0.70)]) == pytest.approx(0.01)
    assert RetrievalService.ambiguity([_item(0.80), _item(0.50)]) == pytest.approx(0.30)
    assert RetrievalService.ambiguity([_item(0.8)]) == float("inf")


def test_mmr_trades_relevance_for_diversity_and_is_deterministic():
    v = np.array([[1, 0], [0.999, 0.045], [0, 1]], dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rel = [0.90, 0.89, 0.60]
    assert mmr_order(rel, v, 3, lam=1.0) == [0, 1, 2]            # lam = 1: pure relevance
    assert mmr_order(rel, v, 3, lam=0.5) == [0, 2, 1]            # the near-duplicate of the first pick is pushed down
    assert mmr_order(rel, v, 2, lam=0.5) == mmr_order(rel, v, 2, lam=0.5) and mmr_order([], v[:0], 3) == []


# ---------------------------------------------------------------- the ladder against the real corpus
def _search(run, svc, kind="ticket", strategy="adaptive", text=Q):
    async def go():
        ctx = await svc.retrieval.make_context(text)
        return ctx, await svc.retrieval.search(ctx, kind, strategy, 5)
    return run(go)


def test_a_clear_result_stops_at_dense_and_spends_nothing_more(svc, run, monkeypatch):
    monkeypatch.setattr(svc.settings, "adaptive_margin_ticket", 0.0)
    ctx, items = _search(run, svc)
    assert [d["action"] for d in ctx.decisions] == ["stop"] and ctx.decisions[0]["stage"] == "dense" and ctx.decisions[0]["signal"] == "margin"
    assert "lexical_ms" not in ctx.timings and "rerank_ms" not in ctx.timings            # no lexical query, no cross-encoder
    assert all(i.retrieval_method == "adaptive" and i.scores["adaptive_stage"] == 0.0 for i in items)
    dense = _search(run, svc, strategy="dense")[1]
    assert [i.source_id for i in items] == [i.source_id for i in dense]                    # identical to plain dense when it does not escalate


def test_an_ambiguous_result_escalates_to_hybrid_and_says_why(svc, run, monkeypatch):
    monkeypatch.setattr(svc.settings, "adaptive_margin_ticket", 9.0)                       # everything counts as ambiguous
    ctx, items = _search(run, svc)
    assert [(d["stage"], d["action"]) for d in ctx.decisions] == [("dense", "escalate"), ("hybrid", "stop")]
    assert "lexical_ms" in ctx.timings and "rerank_ms" not in ctx.timings and ctx.decisions[1]["reason"]
    assert all(i.scores["adaptive_stage"] == 1.0 and "rrf" in i.scores for i in items) and len({i.source_id for i in items}) == len(items)
    assert ctx.decisions[0]["threshold"] == 9.0 and ctx.decisions[0]["value"] is not None


def test_the_reranker_rung_is_bounded_and_only_taken_when_configured(svc, run, monkeypatch):
    monkeypatch.setattr(svc.settings, "adaptive_margin_ticket", 9.0)
    monkeypatch.setattr(svc.settings, "adaptive_rerank_gap", 9.0)                          # fused top-2 always "nearly tied"
    monkeypatch.setattr(svc.settings, "adaptive_rerank_top_n", 6)
    ctx, items = _search(run, svc)
    assert [d["stage"] for d in ctx.decisions] == ["dense", "hybrid", "rerank"] and "rerank_ms" in ctx.timings
    assert all(i.scores["adaptive_stage"] == 2.0 and "rerank_prob" in i.scores for i in items) and len(items) <= 6
    assert svc.retrieval.config_snapshot("adaptive", ctx)["rerank_top_n"] == 6


def test_query_expansion_adds_shared_rare_terms_to_the_lexical_leg(svc, run, monkeypatch):
    monkeypatch.setattr(svc.settings, "adaptive_margin_ticket", 9.0)
    monkeypatch.setattr(svc.settings, "adaptive_expand_kinds", "ticket")
    ctx, _ = _search(run, svc, text="the connection is poor in the evenings")
    hybrid = next(d for d in ctx.decisions if d["stage"] == "hybrid")
    assert hybrid["expansion_terms"] and len(hybrid["expansion_terms"]) <= svc.settings.adaptive_expansion_terms
    monkeypatch.setattr(svc.settings, "adaptive_expand_kinds", "")
    ctx2, _ = _search(run, svc, text="the connection is poor in the evenings")
    assert next(d for d in ctx2.decisions if d["stage"] == "hybrid")["expansion_terms"] == []


def test_mmr_option_returns_the_same_candidates_in_a_diversified_order(svc, run, monkeypatch):
    monkeypatch.setattr(svc.settings, "adaptive_margin_ticket", 9.0)
    plain = _search(run, svc)[1]
    monkeypatch.setattr(svc.settings, "adaptive_mmr", True)
    diverse = _search(run, svc)[1]
    assert len({i.source_id for i in diverse}) == len(diverse) == len(plain) and all("mmr" in i.scores for i in diverse)
    assert [i.rank for i in diverse] == list(range(1, len(diverse) + 1))


def test_articles_use_their_own_threshold_and_a_dense_memo_is_shared(svc, run, monkeypatch):
    monkeypatch.setattr(svc.settings, "adaptive_margin_article", 9.0)
    ctx, items = _search(run, svc, kind="article")
    assert ctx.decisions[0]["kind"] == "article" and ctx.decisions[0]["threshold"] == 9.0 and items
    # a longer dense list for the same query answers a shorter request without a second round trip
    n_before = len([k for k in ctx.memo if k[0] == "dense"])
    run(svc.retrieval.dense_top_similarity, ctx, "article")
    assert len([k for k in ctx.memo if k[0] == "dense"]) == n_before


# ---------------------------------------------------------------- through the API
def test_adaptive_is_selectable_per_request_and_its_decisions_are_part_of_the_provenance(client, svc, monkeypatch):
    monkeypatch.setattr(svc.settings, "adaptive_margin_ticket", 9.0)
    r = client.post("/api/v1/resolve", json={"complaint": Q, "strategy": "adaptive"}).json()
    ad = r["provenance"]["retrieval"]["adaptive"]
    assert r["provenance"]["retrieval"]["strategy"] == "adaptive" and ad["decisions"] and ad["margin_ticket"] == 9.0
    assert any(d["action"] == "escalate" for d in ad["decisions"])
    retrieve = next(s for s in r["trace"] if s["name"] == "retrieve")
    assert retrieve["detail"]["adaptive"] == ad["decisions"] and r["tickets"][0]["retrieval_method"] == "adaptive"
    s = client.get("/api/v1/search", params={"q": Q, "strategy": "adaptive", "limit": 3})
    assert s.status_code == 200 and len(s.json()["tickets"]) == 3
    assert client.post("/api/v1/resolve", json={"complaint": Q, "strategy": "nonsense"}).status_code == 422
