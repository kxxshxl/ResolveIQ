"""Resolution lineage (the evidence graph) as data.

complaint -> extracted attributes -> retrieved sources (with scores and reasons) -> evidence selected for the prompt -> resolution steps
-> citations (with per-edge support) -> grounding validation.

Only auditable signals are exposed: scores, ranks, ids, counts and short excerpts of corpus text (tickets and articles are already PII-redacted).
Nothing here comes from model reasoning; the single model-written field shown is the `uncertainty` statement the model returns as part of its answer.
"""
from __future__ import annotations

from collections import Counter

from app.models.schemas import (Citation, Classification, EvidenceAssessment, Lineage, LineageEdge, LineageSource, LineageStep, Resolution,
                                RetrievedItem, ValidationReport)

SCORE_LABELS = {"dense_cosine": "semantic similarity", "lexical": "keyword match score", "rrf": "fused rank score", "rerank_prob": "reranker relevance",
                "bm25": "BM25 score", "mmr": "diversity-adjusted score"}


def _excerpt(it: RetrievedItem) -> str:
    text = it.text if it.source_type == "ticket" else (it.title + ": " + it.text)
    text = " ".join(text.split())
    return text if len(text) <= 180 else text[:179] + "…"


def _why(it: RetrievedItem, cls: Classification, intent_counts: Counter, n_tickets: int, selected: bool, in_prompt: bool) -> list[str]:
    out = [f"ranked #{it.rank} by {it.retrieval_method.replace('_', ' ')}"]
    for key in ("dense_cosine", "lexical", "rerank_prob"):
        if key in it.scores:
            rank = it.scores.get("dense_rank") if key == "dense_cosine" else it.scores.get("lexical_rank") if key == "lexical" else None
            out.append(f"{SCORE_LABELS[key]} {it.scores[key]:.2f}" + (f" (rank {int(rank)} for that signal)" if rank else ""))
    intent = it.metadata.get("intent")
    if intent == cls.intent:
        out.append(f"same intent as the complaint ({cls.intent.replace('_', ' ')})")
    elif intent:
        out.append(f"different intent ({str(intent).replace('_', ' ')}) from the detected {cls.intent.replace('_', ' ')}")
    if it.source_type == "ticket" and n_tickets > 1 and intent:
        out.append(f"{intent_counts[str(intent)]} of the {n_tickets} retrieved tickets share this intent")
    if selected and not in_prompt:
        out.append("selected as evidence but left out of the prompt (context budget)")
    elif selected:
        out.append("selected as evidence for the answer")
    else:
        out.append("retrieved but not selected as evidence (lower rank)")
    return out


def build_lineage(*, complaint: str, pii: dict[str, int], injection: bool, cls: Classification, tickets: list[RetrievedItem],
                  articles: list[RetrievedItem], selected: list[RetrievedItem], in_prompt: list[str], resolution: Resolution,
                  citations: list[Citation], validation: ValidationReport, assessment: EvidenceAssessment, status: str,
                  abstain_reason: str | None) -> Lineage:
    sel_ids = {e.source_id for e in selected}
    prompt_ids = set(in_prompt)
    cited_by: dict[str, list[int]] = {c.source_id: c.cited_in_steps for c in citations}
    intent_counts = Counter(str(t.metadata.get("intent")) for t in tickets)
    sources = [
        LineageSource(
            id=it.source_id, type=it.source_type, title=it.title if it.source_type == "article" else (it.resolution_summary or it.title),
            excerpt=_excerpt(it), rank=it.rank, method=it.retrieval_method, score=it.score, stage_scores=it.scores, intent=it.metadata.get("intent"),
            matches_intent=it.metadata.get("intent") == cls.intent, selected=it.source_id in sel_ids, in_prompt=it.source_id in prompt_ids,
            cited_by_steps=cited_by.get(it.source_id, []), why=_why(it, cls, intent_counts, len(tickets), it.source_id in sel_ids, it.source_id in prompt_ids))
        for it in (*tickets, *articles)]
    steps = [LineageStep(index=i + 1, text=s.text, citations=s.citations, grounded=s.grounded, grounding_score=s.grounding_score, support=s.support)
             for i, s in enumerate(resolution.steps)]
    attrs = [{"name": n, "value": getattr(cls, n), "confidence": getattr(cls.confidence, n)} for n in ("intent", "product", "severity", "sentiment")]

    edges = [LineageEdge(kind="extracted", source="complaint", target=f"attr:{a['name']}", weight=a["confidence"]) for a in attrs]
    edges += [LineageEdge(kind="retrieved", source="complaint", target=f"src:{s.id}", weight=s.score) for s in sources]
    edges += [LineageEdge(kind="selected", source=f"src:{s.id}", target="evidence") for s in sources if s.selected]
    for st in steps:
        for cid in st.citations:
            edges.append(LineageEdge(kind="cited", source=f"src:{cid}", target=f"step:{st.index}", weight=st.support.get(cid)))

    retrieved_ids = {s.id for s in sources}
    cited_ids = {cid for st in steps for cid in st.citations}
    step_scores = [st.grounding_score for st in steps if st.grounding_score is not None]
    signals = {
        "evidence_strength": assessment.confidence, "source_agreement": assessment.consensus,
        "similarity": max(assessment.top_ticket_similarity, assessment.top_article_similarity), "reranker_signal": assessment.rerank_signal,
        "citation_coverage": validation.citation_coverage, "grounded_ratio": validation.grounded_ratio,
        "mean_step_support": round(sum(step_scores) / len(step_scores), 3) if step_scores else None,
        "uncertainty": resolution.uncertainty, "abstention_reason": abstain_reason, "escalate": resolution.escalate,
        "escalation_reason": resolution.escalation_reason, "status": status,
        "sources_retrieved": len(sources), "sources_selected": len(sel_ids), "sources_cited": len(cited_ids),
    }
    checks = {
        "every_citation_maps_to_a_retrieved_source": cited_ids <= retrieved_ids,
        "every_citation_was_in_the_prompt": cited_ids <= prompt_ids or not cited_ids,
        "no_invalid_citation_survived": not (set(validation.invalid_citations) & cited_ids),
        "every_edge_endpoint_exists": all(
            (e.target.startswith("step:") and int(e.target.split(":")[1]) <= len(steps) or e.target in ("evidence",) or e.target.startswith(("attr:", "src:")))
            and (e.source == "complaint" or e.source.removeprefix("src:") in retrieved_ids) for e in edges),
    }
    return Lineage(complaint={"text": complaint, "chars": len(complaint), "pii_redactions": pii, "injection_flagged": injection},
                   attributes=attrs, sources=sources, steps=steps, edges=edges, signals=signals, checks=checks)
