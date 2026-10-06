"""Suite orchestration shared by the API (/api/v1/evaluate) and the CLI (python -m app.evaluation.run)."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Sequence

from app.evaluation.datasets import load_eval_sets
from app.evaluation.cluster_eval import clustering_suite
from app.evaluation.discovery_eval import discovery_suite
from app.evaluation.robustness import robustness_suite
from app.evaluation.suites import adaptive_suite, classification_suite, evolving_suite, rag_e2e_suite, retrieval_suite
from app.services.container import Services
from app.services.llm.base import ResilientLLM
from app.services.llm.providers import OpenAICompatProvider


def alt_openai_variant(svc: Services, model: str) -> tuple[str, ResilientLLM]:
    """The OpenAI-compatible provider pointed at Ollama's /v1 endpoint (no paid API needed)."""
    prov = OpenAICompatProvider(svc.settings.ollama_base_url.rstrip("/") + "/v1", "", model)
    return f"openai_compat:{model}", ResilientLLM([prov], svc.settings)


async def run_suites(svc: Services, suites: Sequence[str], max_queries: int | None = None, *, include_llm_classifier: bool = False,
                     judge_n: int = 0, alt_openai_model: str | None = None) -> dict:
    sets = load_eval_sets(svc.settings.data_dir)
    t0 = time.perf_counter()
    counts = await svc.repo.counts()
    results: dict = {"meta": {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "embedding_model": svc.embedder.model_name,
        "reranker": svc.reranker.model_name if svc.reranker.enabled else None,
        "llm": [f"{p.name}:{p.model}" for p in svc.llm.providers], "corpus": counts,
        "eval_sets": {"val": len(sets.val), "test": len(sets.test), "gold": len(sets.gold), "blind": len(sets.blind), "ood": len(sets.ood), "evolving": len(sets.evolving)},
        "thresholds": {"abstain": svc.settings.abstain_threshold, "grounding": svc.settings.grounding_threshold,
                       "min_grounded_ratio": svc.settings.min_grounded_ratio}}}
    want = set(suites)
    if "classification" in want:
        results["classification"] = await classification_suite(svc, sets, max_queries, include_llm=include_llm_classifier)
    if "retrieval" in want:
        results["retrieval"] = await retrieval_suite(svc, sets, max_queries)
    if "robustness" in want:
        results["robustness"] = await robustness_suite(svc, sets, max_queries)
    if want & {"rag", "e2e"}:
        variants: list[tuple[str, ResilientLLM | None]] = [("extractive_no_llm", None)]
        if alt_openai_model:
            variants.append(alt_openai_variant(svc, alt_openai_model))
        results.update(await rag_e2e_suite(svc, sets, max_queries, want, judge_n=judge_n, variants=variants))
    if "adaptive" in want:
        results["adaptive"] = await adaptive_suite(svc, sets, max_queries)
    if "clustering" in want:
        results["clustering"] = await clustering_suite(svc, sets)
    if "evolving" in want:
        results["evolving"] = await evolving_suite(svc, sets, svc.settings.data_dir)
    if "discovery" in want:
        results["discovery"] = await discovery_suite(svc, sets)
    results["meta"]["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return results
