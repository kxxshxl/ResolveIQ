"""Evidence selection and sufficiency assessment (the abstention gate)."""
from __future__ import annotations

from collections import defaultdict

from app.core.config import Settings
from app.models.schemas import EvidenceAssessment, RetrievedItem


def select_evidence(tickets: list[RetrievedItem], articles: list[RetrievedItem], s: Settings) -> list[RetrievedItem]:
    return [*tickets[: s.evidence_tickets], *articles[: s.evidence_articles]]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def assess_evidence(tickets: list[RetrievedItem], articles: list[RetrievedItem], top_ticket_sim: float,
                    top_article_sim: float, s: Settings) -> EvidenceAssessment:
    """Combine three independent signals into one calibrated-ish score in [0,1]:

    * similarity  - best dense cosine of any ticket/article (is there anything close at all?)
    * rerank      - best cross-encoder probability among retrieved items (is it actually on-topic?)
    * consensus   - score-weighted share of the top tickets that agree on one intent (is the history coherent?)
    """
    best_sim = max(top_ticket_sim, top_article_sim)
    sim_component = _clip01((best_sim - 0.25) / 0.45)  # cosine 0.25 -> 0, 0.70 -> 1 (MiniLM scale)

    rerank_vals = [i.scores["rerank_prob"] for i in (*tickets, *articles) if "rerank_prob" in i.scores]
    rerank_signal = max(rerank_vals) if rerank_vals else None
    rerank_component = rerank_signal if rerank_signal is not None else sim_component

    weights: dict[str, float] = defaultdict(float)
    for t in tickets[:5]:
        weights[str(t.metadata.get("intent"))] += max(t.score, 1e-6)
    total = sum(weights.values())
    consensus = (max(weights.values()) / total) if total else 0.0

    confidence = 0.45 * sim_component + 0.30 * rerank_component + 0.25 * consensus
    if not tickets and not articles:
        confidence = 0.0
    sufficient = confidence >= s.abstain_threshold
    if not tickets and not articles:
        reason = "no relevant tickets or articles were retrieved"
    elif not sufficient:
        reason = (f"retrieved evidence is weak (confidence {confidence:.2f} < {s.abstain_threshold:.2f}; "
                  f"best similarity {best_sim:.2f}, consensus {consensus:.2f})")
    else:
        reason = "evidence is sufficient"
    return EvidenceAssessment(
        confidence=round(confidence, 3), top_ticket_similarity=round(top_ticket_sim, 3),
        top_article_similarity=round(top_article_sim, 3),
        rerank_signal=round(rerank_signal, 3) if rerank_signal is not None else None,
        consensus=round(consensus, 3), sufficient=sufficient, reason=reason)
