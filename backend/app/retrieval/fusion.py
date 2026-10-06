"""Reciprocal Rank Fusion - parameter-light, score-scale-free, and easy to explain/audit - and Maximal Marginal Relevance."""
from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np


def rrf_fuse(rankings: dict[str, Sequence[Hashable]], k: int = 60,
             weights: dict[str, float] | None = None) -> list[tuple[Hashable, float, dict[str, int]]]:
    """Fuse several ranked id lists.

    score(d) = sum_r  w_r / (k + rank_r(d))      (rank is 1-based)

    Returns [(id, fused_score, {retriever: rank})] sorted by fused score desc; ties are broken by
    the best individual rank, then by id for determinism.
    """
    scores: dict[Hashable, float] = {}
    ranks: dict[Hashable, dict[str, int]] = {}
    for name, ids in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for pos, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + pos)
            ranks.setdefault(doc_id, {})[name] = pos
    return sorted(
        ((d, s, ranks[d]) for d, s in scores.items()),
        key=lambda x: (-x[1], min(x[2].values()), str(x[0])),
    )


def mmr_order(relevance: Sequence[float], vectors: np.ndarray, k: int, lam: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance: repeatedly take the candidate with the best  lam * relevance - (1 - lam) * (max similarity to those already chosen).

    `relevance` should be on a 0..1 scale; `vectors` are unit-length, one row per candidate. Returns candidate indices in selection order.
    Deterministic: ties go to the earlier (better-ranked) candidate.
    """
    n = len(relevance)
    if n == 0:
        return []
    rel = np.asarray(relevance, dtype=float)
    chosen: list[int] = [int(np.argmax(rel))]
    sim = vectors @ vectors.T
    while len(chosen) < min(k, n):
        rest = [i for i in range(n) if i not in chosen]
        score = [lam * rel[i] - (1 - lam) * max(sim[i, j] for j in chosen) for i in rest]
        chosen.append(rest[int(np.argmax(score))])
    return chosen
