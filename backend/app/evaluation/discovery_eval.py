"""Emerging-class discovery evaluation.

Stream = hand-written known complaints (test + gold + blind) mixed with complaints from 3 genuinely new classes that the
taxonomy does not contain (number porting, voicemail, TV streaming app). The 3 classes were NOT used to tune the discovery
parameters (those were tuned on eSIM + roaming + validation queries). For every new class the first 7 complaints form the
stream and the last 3 are held out to test routing after a proposal is accepted.

The accept step temporarily adds taxonomy labels (always removed afterwards), like the `evolving` suite.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from app.discovery.clustering import build_proposals
from app.evaluation.datasets import EvalSets, load_jsonl
from app.retrieval.service import QueryContext
from app.services.container import Services

log = logging.getLogger(__name__)
STREAM_PER_CLASS = 7


async def discovery_suite(svc: Services, sets: EvalSets, mutate: bool = True) -> dict:
    novel_rows = load_jsonl(svc.settings.data_dir / "eval" / "novel_stream.jsonl")
    classes = sorted({r["intent"] for r in novel_rows})
    stream_novel = [r for c in classes for r in [x for x in novel_rows if x["intent"] == c][:STREAM_PER_CLASS]]
    holdout = [r for c in classes for r in [x for x in novel_rows if x["intent"] == c][STREAM_PER_CLASS:]]
    known = sets.test + sets.gold + sets.blind
    items = [(r["text"], r["intent"], True) for r in stream_novel] + [(r["text"], f"known:{r['scenario_id']}", False) for r in known]
    texts = [t for t, _, _ in items]

    X, neighbors, evidence = await svc.discovery.profile(texts)
    ev = np.array(evidence)
    thr = svc.settings.discovery_novelty_threshold
    flagged = [i for i in range(len(items)) if ev[i] < thr]
    is_novel = np.array([n for _, _, n in items])
    n_nov = int(is_novel.sum())
    out: dict = {
        "stream": {"novel": n_nov, "known": len(items) - n_nov, "novel_classes": classes},
        "candidate_selection": {
            "threshold": thr, "novel_flagged": int(sum(is_novel[i] for i in flagged)), "known_flagged": int(sum(not is_novel[i] for i in flagged)),
            "novel_recall": round(float(sum(is_novel[i] for i in flagged) / n_nov), 3),
            "known_flag_rate": round(float(sum(not is_novel[i] for i in flagged) / (len(items) - n_nov)), 3),
            "abstention_only_novel_recall": round(float(np.mean([ev[i] < svc.settings.abstain_threshold for i in range(len(items)) if is_novel[i]])), 3)}}

    labels = {k: v.team for k, v in svc.taxonomy.current.labels["intent"].items()}
    background = [d["text"] for d in await svc.repo.all_docs("ticket")]
    props = build_proposals([texts[i] for i in flagged], X[flagged], [neighbors[i] for i in flagged], [evidence[i] for i in flagged],
                            background, svc.discovery.params(), labels)
    truth = [items[i][1] for i in flagged]

    per_class, matched = {}, set()
    for c in classes:
        total = sum(1 for _, t, n in items if t == c)
        best = None
        for pi, p in enumerate(props):
            comp = Counter(truth[m] for m in p["member_indices"])
            got = comp.get(c, 0)
            if got and (best is None or got > best[1]):
                best = (pi, got, comp)
        if best:
            pi, got, comp = best
            p = props[pi]
            purity = got / p["size"]
            ok = purity >= 0.8 and p["recommendation"] == "new_class"
            if ok:
                matched.add(pi)
            per_class[c] = {"recovered": ok, "coverage": round(got / total, 3), "purity": round(purity, 3), "cluster_size": p["size"],
                            "recommendation": p["recommendation"], "keywords": p["keywords"][:5]}
        else:
            per_class[c] = {"recovered": False, "coverage": 0.0, "purity": 0.0}
    out["proposals"] = {
        "n": len(props), "new_class": sum(p["recommendation"] == "new_class" for p in props),
        "recovered_classes": sum(v["recovered"] for v in per_class.values()), "of": len(classes),
        "precision": round(len(matched) / max(1, sum(p["recommendation"] == "new_class" for p in props)), 3),
        "mean_cohesion": round(float(np.mean([p["cohesion"] for p in props])), 3) if props else None, "per_class": per_class,
        "spurious": [{"size": p["size"], "keywords": p["keywords"][:4], "mean_top_similarity": p["mean_top_similarity"],
                      "neighbor_agreement": p["neighbor_agreement"],
                      "composition": dict(Counter(truth[m] for m in p["member_indices"]).most_common(4))}
                     for pi, p in enumerate(props) if pi not in matched]}

    out["after_acceptance"] = await _after_acceptance(svc, props, truth, classes, holdout) if mutate and props else {"skipped": True}
    return out


async def _after_acceptance(svc: Services, props: list[dict], truth: list[str], classes: list[str], holdout: list[dict]) -> dict:
    """Accept every proposal that recovered a class (as a reviewer would) and measure routing of the 3 held-out complaints per class."""
    created: list[str] = []
    mapping: dict[str, str] = {}
    try:
        before = await _route(svc, holdout, {c: c for c in classes})
        for i, p in enumerate(props):
            if p["recommendation"] != "new_class":
                continue
            major = Counter(truth[m] for m in p["member_indices"]).most_common(1)[0][0]
            if major not in classes or major in mapping:
                continue
            lid = f"zz_eval_{i}"
            await svc.taxonomy.add_label("intent", lid, p["description"], p["keywords"], p["examples"], p["team"])
            created.append(lid)
            mapping[major] = lid
        after = await _route(svc, holdout, mapping)
        return {"accepted": len(created), "holdout": len(holdout), "routed_correctly_before": before, "routed_correctly_after": after,
                "note": "zero resolved tickets exist for the new classes; routing relies on proposal keywords/examples (zero-shot prototypes)"}
    finally:
        if created:
            await svc.repo.delete_taxonomy_labels("intent", created)
            await svc.taxonomy.refresh()


async def _route(svc: Services, holdout: list[dict], mapping: dict[str, str]) -> float:
    ok = 0
    for r in holdout:
        ctx = QueryContext(text=r["text"], embedding=await svc.embedder.embed_query(r["text"]))
        cls = await svc.classifier.classify(ctx)
        ok += cls.intent == mapping.get(r["intent"], "__none__")
    return round(ok / len(holdout), 3)
