"""Emerging-class discovery: cluster complaints that existing classes explain poorly and propose new taxonomy labels.

Why a batch job rather than a per-request check: a single request cannot tell "novel topic" from "oddly worded known
topic" (the best single novelty signal reaches AUC ~0.8, see docs/evaluation.md). A *recurring* novel topic, however,
forms a tight cluster among low-evidence requests, while stray false positives scatter across many existing classes.
Everything here is pure (numpy/sklearn only) so it is unit-testable and shared by the service and the evaluation suite.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class DiscoveryParams:
    distance_threshold: float = 0.55   # average-linkage cosine distance at which clusters stop merging
    min_cluster_size: int = 4
    covered_similarity: float = 0.60   # members this close to existing tickets (mean top-1 cosine) are already covered ...
    extend_agreement: float = 0.60     # ... then: >= this share share one nearest intent -> "extend existing", else dropped as noise
    max_keywords: int = 6
    n_examples: int = 3


@dataclass
class Neighbors:
    """What the existing corpus says about one complaint (top-k similar resolved tickets)."""
    intents: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    top_similarity: float = 0.0


def cluster_embeddings(X: np.ndarray, p: DiscoveryParams) -> list[list[int]]:
    """Average-linkage agglomerative clustering on cosine distance; clusters smaller than min size are dropped as noise."""
    if len(X) < p.min_cluster_size:
        return []
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=p.distance_threshold).fit_predict(X)
    groups: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        groups.setdefault(int(l), []).append(i)
    return sorted((g for g in groups.values() if len(g) >= p.min_cluster_size), key=len, reverse=True)


def _slug(words: list[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "_", "_".join(words).lower()).strip("_")[:48] or "emerging_issue"


def keywords_per_cluster(cluster_texts: list[list[str]], background: list[str], k: int) -> list[list[str]]:
    """c-TF-IDF style: one document per cluster plus one background document of known tickets, so terms that are common
    in the existing corpus ('router', 'internet') are down-weighted and cluster-specific terms ('esim', 'qr code') rise."""
    docs = [" ".join(t) for t in cluster_texts] + [" ".join(background)]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    try:
        M = vec.fit_transform(docs)
    except ValueError:  # every document is stop words / too short: no keywords, but the proposal is still reviewable
        return [[] for _ in cluster_texts]
    terms = np.array(vec.get_feature_names_out())
    out = []
    for i in range(len(cluster_texts)):
        row = M[i].toarray().ravel()
        chosen: list[str] = []
        for j in np.argsort(-row):
            if row[j] <= 0 or len(chosen) >= k:
                break
            t = str(terms[j])
            if not any(t in c or c in t for c in chosen):  # drop 'sim' when 'esim' / 'sim card' already chosen
                chosen.append(t)
        out.append(chosen)
    return out


def build_proposals(texts: list[str], X: np.ndarray, neighbors: list[Neighbors], evidence: list[float], background: list[str],
                    p: DiscoveryParams, existing_labels: dict[str, str | None], request_ids: list[str] | None = None) -> list[dict]:
    """texts/X/neighbors/evidence are aligned per complaint. `existing_labels` maps intent id -> escalation team."""
    clusters = cluster_embeddings(X, p)
    if not clusters:
        return []
    kws = keywords_per_cluster([[texts[i] for i in c] for c in clusters], background, p.max_keywords)
    used = set(existing_labels)
    proposals = []
    for c, kw in zip(clusters, kws):
        centroid = X[c].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) or 1.0)
        sims = X[c] @ centroid
        order = [c[i] for i in np.argsort(-sims)]
        votes = Counter(neighbors[m].intents[0] for m in c if neighbors[m].intents)  # intent of each member's closest ticket
        nearest, count = (votes.most_common(1)[0] if votes else (None, 0))
        agreement = count / len(c) if c else 0.0
        top_sim = float(np.mean([neighbors[m].top_similarity for m in c]))
        covered = top_sim >= p.covered_similarity
        if covered and not (nearest and agreement >= p.extend_agreement):
            continue  # close to existing tickets but with no single owner class: a tone/format artefact, not a topic
        extend = covered
        products = Counter(pr for m in c for pr in neighbors[m].products[:1])
        label = _slug(kw[:2])
        base, n = label, 2
        while label in used:
            label, n = f"{base}_{n}", n + 1
        used.add(label)
        proposals.append({
            "recommendation": "extend_existing" if extend else "new_class",
            "label_id": nearest if extend else label,
            "description": f"Emerging issue seen in {len(c)} recent complaints, e.g. \"{texts[order[0]][:150]}\"",
            "keywords": kw, "examples": [texts[i][:200] for i in order[: p.n_examples]],
            "product": products.most_common(1)[0][0] if products else None,
            "team": existing_labels.get(nearest) if nearest else None,
            "size": len(c), "cohesion": round(float(np.mean(sims)), 4), "mean_evidence": round(float(np.mean([evidence[m] for m in c])), 4),
            "nearest_intent": nearest, "neighbor_agreement": round(agreement, 4), "mean_top_similarity": round(top_sim, 4),
            "member_indices": c, "member_request_ids": [request_ids[m] for m in c] if request_ids else [],
        })
    return proposals
