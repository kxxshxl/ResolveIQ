"""Reciprocal Rank Fusion - parameter-light, score-scale-free, and easy to explain/audit."""
from __future__ import annotations

from typing import Hashable, Sequence


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
