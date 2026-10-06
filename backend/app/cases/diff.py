"""Compare a stored case with a replay of the same complaint through the current pipeline. Pure functions: no I/O, no model calls."""
from __future__ import annotations

import re

PROVENANCE_FIELDS = ("pipeline_version", "generator", "model", "prompt_version", "prompt_hash", "taxonomy_version", "corpus_version", "embedding_model", "reranker")


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _retrieved_ids(stored: list[dict] | None) -> list[str]:
    return [r["source_id"] for r in (stored or [])]


def diff_cases(old: dict, new: dict) -> dict:
    """`old` is the stored case as returned by CaseService.get; `new` is the replayed ResolveResponse as a dict.

    The result says what changed and, where it can tell, why: a different corpus version explains a different retrieved set; a different prompt or
    model explains different steps. `reproduced` is True only when the retrieved sources, their order, the status and the steps all match.
    """
    old_p, new_p = old.get("provenance") or {}, new.get("provenance") or {}
    prov = {f: {"old": old_p.get(f), "new": new_p.get(f)} for f in PROVENANCE_FIELDS if old_p.get(f) != new_p.get(f)}
    old_strategy = (old_p.get("retrieval") or {}).get("strategy")
    new_strategy = (new_p.get("retrieval") or {}).get("strategy")
    if old_strategy != new_strategy:
        prov["retrieval_strategy"] = {"old": old_strategy, "new": new_strategy}

    old_ids = _retrieved_ids(old.get("retrieved"))
    new_ids = [i["source_id"] for i in (*new["tickets"], *new["articles"])]
    old_rank = {sid: i + 1 for i, sid in enumerate(old_ids)}
    new_rank = {sid: i + 1 for i, sid in enumerate(new_ids)}
    retrieval = {
        "same_set": set(old_ids) == set(new_ids), "same_order": old_ids == new_ids,
        "added": [s for s in new_ids if s not in old_rank], "removed": [s for s in old_ids if s not in new_rank],
        "moved": [{"id": s, "old_rank": old_rank[s], "new_rank": new_rank[s]} for s in new_ids if s in old_rank and old_rank[s] != new_rank[s]][:12]}

    old_res = old.get("result") or {}
    old_steps = [s["text"] for s in (old_res.get("resolution") or {}).get("steps", [])]
    new_steps = [s["text"] for s in new["resolution"]["steps"]]
    old_n, new_n = {_norm(s) for s in old_steps}, {_norm(s) for s in new_steps}
    steps = {"same": old_n == new_n, "added": [s for s in new_steps if _norm(s) not in old_n], "removed": [s for s in old_steps if _norm(s) not in new_n],
             "old_count": len(old_steps), "new_count": len(new_steps)}

    old_cls, new_cls = old.get("classification") or {}, new["classification"]
    classification = {k: {"old": old_cls.get(k), "new": new_cls.get(k)} for k in ("intent", "product", "severity", "sentiment") if old_cls.get(k) != new_cls.get(k)}
    old_ev = (old_res.get("evidence") or {}).get("confidence")
    status = {"old": old.get("status"), "new": new["status"], "same": old.get("status") == new["status"]}
    evidence = {"old": old_ev, "new": new["evidence"]["confidence"], "delta": round(new["evidence"]["confidence"] - old_ev, 3) if old_ev is not None else None}
    old_cites = sorted({c for s in (old_res.get("resolution") or {}).get("steps", []) for c in s.get("citations", [])})
    new_cites = sorted({c for s in new["resolution"]["steps"] for c in s["citations"]})

    why: list[str] = []
    if "corpus_version" in prov:
        why.append(f"the corpus changed (version {prov['corpus_version']['old']} to {prov['corpus_version']['new']}), which can change what is retrieved")
    if "taxonomy_version" in prov or classification:
        why.append("the taxonomy or classifier changed, which can change the detected attributes")
    if "prompt_version" in prov or "prompt_hash" in prov:
        why.append("the prompt changed, which can change the generated steps")
    if "model" in prov or "generator" in prov:
        why.append(f"the generator changed ({prov.get('generator', prov.get('model'))['old']} to {prov.get('generator', prov.get('model'))['new']})")
    if "embedding_model" in prov:
        why.append("the embedding model changed, which changes every similarity score")
    if "retrieval_strategy" in prov:
        why.append(f"a different retrieval strategy was used ({old_strategy} to {new_strategy})")
    if not why and not (retrieval["same_order"] and steps["same"]):
        why.append("nothing in the recorded provenance changed: the difference comes from model sampling (use deterministic replay) or from data not versioned here")
    reproduced = retrieval["same_order"] and status["same"] and steps["same"]
    return {"reproduced": reproduced, "status": status, "classification": classification, "retrieval": retrieval, "steps": steps, "evidence": evidence,
            "citations": {"old": old_cites, "new": new_cites, "same": old_cites == new_cites}, "provenance_changes": prov, "likely_reasons": why}
