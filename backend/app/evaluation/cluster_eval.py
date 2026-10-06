"""Clustering evaluation: does the recurring-complaint clustering recover the groups we KNOW exist?

Ground truth: every labelled evaluation complaint belongs to one root-cause scenario (25 of them) and one intent (9). The routine under test is exactly the one
the Recurring clusters view runs (app.clusters.engine.cluster_requests). Deterministic: fixed model, fixed data, no sampling.
Also reported: the same data plus 30 complaints from 3 classes the system has never seen, to check that unseen topics come out as their own clusters.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from app.clusters.engine import cluster_requests
from app.evaluation.datasets import EvalSets, load_jsonl
from app.services.container import Services


def _score(labels_true: list[str], groups: list[list[int]], n: int) -> dict:
    """Purity (share of clustered complaints that belong to their cluster's majority label), plus ARI / NMI with unclustered complaints as singletons."""
    assigned = np.full(n, -1)
    for gi, g in enumerate(groups):
        assigned[g] = gi
    nxt = len(groups)
    for i in range(n):
        if assigned[i] < 0:
            assigned[i], nxt = nxt, nxt + 1
    clustered = [i for g in groups for i in g]
    purity = sum(Counter(labels_true[i] for i in g).most_common(1)[0][1] for g in groups) / max(1, len(clustered))
    return {"purity": round(purity, 4), "ari": round(float(adjusted_rand_score(labels_true, assigned)), 4), "nmi": round(float(normalized_mutual_info_score(labels_true, assigned)), 4)}


def _recovered(truth: list[str], groups: list[list[int]], wanted: set[str] | None = None) -> dict:
    """A label is recovered when one cluster is at least 70% that label and holds at least half of that label's complaints."""
    total = Counter(truth)
    got = {}
    for g in groups:
        lab, n = Counter(truth[i] for i in g).most_common(1)[0]
        if n / len(g) >= 0.7 and n >= 0.5 * total[lab] and (wanted is None or lab in wanted):
            got[lab] = max(got.get(lab, 0), n)
    return {"recovered": len(got), "of": len([k for k in total if wanted is None or k in wanted]), "labels": sorted(got)}


async def clustering_suite(svc: Services, sets: EvalSets) -> dict:
    rows_src = [("test", r) for r in sets.test] + [("gold", r) for r in sets.gold] + [("blind", r) for r in sets.blind] + [("val", r) for r in sets.val]
    seen, items = set(), []
    for split, r in rows_src:
        k = " ".join(r["text"].lower().split())
        if k not in seen:
            seen.add(k)
            items.append(r)
    novel = load_jsonl(svc.settings.data_dir / "eval" / "novel_stream.jsonl")
    now = datetime.now(timezone.utc)

    def build(rs: list[dict]):
        return [{"request_id": f"e{i}", "complaint": r["text"], "created_at": now - timedelta(hours=i % 120), "intent": r.get("intent"), "product": r.get("product"), "severity": r.get("severity"),
                 "status": "resolved", "evidence": None, "occurrences": 1} for i, r in enumerate(rs)]

    out: dict = {"complaints": len(items), "scenarios": len({r["scenario_id"] for r in items}), "intents": len({r["intent"] for r in items}), "sweep": []}
    X = await svc.embedder.embed_batch([r["text"] for r in items])
    for dist in (0.45, 0.55, 0.65):
        rows = build(items)
        clusters = cluster_requests(rows, X, min_size=3, distance=dist, days=7, now=now, max_clusters=200)
        idx = {r["request_id"]: i for i, r in enumerate(rows)}
        groups = [[idx[m] for m in c["member_request_ids"]] for c in clusters]
        scen = [r["scenario_id"] for r in items]
        entry = {"distance": dist, "clusters": len(groups), "coverage": round(sum(len(g) for g in groups) / len(items), 4), "mean_cluster_size": round(float(np.mean([len(g) for g in groups])), 2) if groups else 0,
                 "vs_scenario": {**_score(scen, groups, len(items)), **_recovered(scen, groups)}, "vs_intent": _score([r["intent"] for r in items], groups, len(items))}
        out["sweep"].append(entry)
    out["default"] = next(e for e in out["sweep"] if e["distance"] == 0.55)

    # unseen topics mixed into the same traffic
    mixed = items + [{**r, "scenario_id": r["intent"]} for r in novel]
    Xn = await svc.embedder.embed_batch([r["text"] for r in novel])
    Xm = np.vstack([X, Xn])
    rows = build(mixed)
    clusters = cluster_requests(rows, Xm, min_size=3, distance=0.55, days=7, now=now, max_clusters=200)
    idx = {r["request_id"]: i for i, r in enumerate(rows)}
    groups = [[idx[m] for m in c["member_request_ids"]] for c in clusters]
    truth = [r["scenario_id"] for r in mixed]
    out["with_unseen_topics"] = {"novel_complaints": len(novel), "novel_classes": sorted({r["intent"] for r in novel}), **_recovered(truth, groups, {r["intent"] for r in novel}),
                                 "note": "a class counts as recovered when one cluster is >= 70% that class and holds >= half of its complaints"}
    return out
