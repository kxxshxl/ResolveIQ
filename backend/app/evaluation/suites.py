"""Evaluation suites. Each returns a JSON-serialisable dict; nothing here is hard-coded or simulated."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from typing import Sequence

import numpy as np

from app.classification.taxonomy import DIMENSIONS
from app.evaluation.datasets import Corpus, EvalSets, build_corpus
from app.evaluation.metrics import classification_metrics, latency_stats, mean_dicts, retrieval_metrics
from app.models.schemas import ArticleIn, ResolveRequest, ResolveResponse, TicketIn
from app.rag.citations import containment, evidence_units
from app.rag.evidence import select_evidence
from app.rag.pipeline import ResolutionService
from app.retrieval.service import STRATEGIES, QueryContext
from app.services.container import Services
from app.services.llm.base import ResilientLLM
from app.services.llm.providers import OllamaProvider

log = logging.getLogger(__name__)
KS = (1, 3, 5, 10)


def _sample(rows: list[dict], n: int | None) -> list[dict]:
    if not n or n >= len(rows):
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


async def _contexts(svc: Services, rows: list[dict]) -> list[tuple[dict, np.ndarray, float]]:
    out = []
    for r in rows:
        t = time.perf_counter()
        ctx = await svc.retrieval.make_context(r["text"])
        out.append((r, ctx.embedding, (time.perf_counter() - t) * 1000))
    return out


# ============================================================ classification
async def classification_suite(svc: Services, sets: EvalSets, max_queries: int | None, include_llm: bool = False,
                               llm_sample: int = 40) -> dict:
    strategies = ["rules", "embedding", "ensemble_legacy", "ensemble"] + (["llm", "ensemble_llm"] if include_llm else [])
    result: dict = {"strategies": {}, "calibration": {}}
    orig_fallback = svc.settings.classifier_llm_fallback
    svc.settings.classifier_llm_fallback = False  # "ensemble" row = rules+kNN only; "ensemble_llm" row adds the fallback
    try:
        await _classification_loop(svc, sets, max_queries, strategies, llm_sample, result)
    finally:
        svc.settings.classifier_llm_fallback = orig_fallback
    return result


async def _classification_loop(svc, sets, max_queries, strategies, llm_sample, result) -> None:
    splits = [("test", _sample(sets.test, max_queries)), ("gold", sets.gold)] + ([("blind", sets.blind)] if sets.blind else [])
    for split_name, rows in splits:
        embs = await _contexts(svc, rows)
        for strat in strategies:
            use = embs if strat not in ("llm", "ensemble_llm") else embs[:: max(1, len(embs) // llm_sample)][:llm_sample]
            preds, lat = [], []
            for r, emb, _ in use:
                ctx = QueryContext(text=r["text"], embedding=emb)
                t = time.perf_counter()
                preds.append(await svc.classifier.classify(ctx, strategy=strat))
                lat.append((time.perf_counter() - t) * 1000)
            truth = [r for r, _, _ in use]
            per_dim = {d: classification_metrics([t[d] for t in truth], [getattr(p, d) for p in preds]) for d in DIMENSIONS}
            per_dim["latency"] = latency_stats(lat)
            result["strategies"].setdefault(strat, {})[split_name] = per_dim
            if strat == "ensemble" and split_name == "test":
                for d in DIMENSIONS:
                    ok = [getattr(p.confidence, d) for t, p in zip(truth, preds) if t[d] == getattr(p, d)]
                    bad = [getattr(p.confidence, d) for t, p in zip(truth, preds) if t[d] != getattr(p, d)]
                    result["calibration"][d] = {"mean_conf_correct": round(float(np.mean(ok)), 3) if ok else None,
                                                "mean_conf_wrong": round(float(np.mean(bad)), 3) if bad else None}


# ============================================================ retrieval
async def _rank_run(svc: Services, corpus: Corpus, items: list[tuple[dict, np.ndarray, float]], strategy: str, kind: str,
                    filter_fn=None) -> dict:
    per_q, lat = [], []
    for r, emb, _ in items:
        ctx = QueryContext(text=r["text"], embedding=emb, filters=filter_fn(r) if filter_fn else None)
        t = time.perf_counter()
        res = await svc.retrieval.search(ctx, kind, strategy, 10)
        lat.append((time.perf_counter() - t) * 1000)
        ranked = [x.source_id for x in res]
        per_q.append(retrieval_metrics(ranked, corpus.relevant(kind, r["scenario_id"]), KS))
    return {**mean_dicts(per_q), "latency": latency_stats(lat), "n": len(items)}


async def retrieval_suite(svc: Services, sets: EvalSets, max_queries: int | None) -> dict:
    corpus = await build_corpus(svc.repo)
    out: dict = {"tickets": {}, "articles": {}, "experiments": {}}
    splits = {"test": _sample(sets.test, max_queries), "gold": sets.gold, **({"blind": sets.blind} if sets.blind else {})}
    embedded = {name: await _contexts(svc, rows) for name, rows in splits.items()}
    out["embedding_latency"] = latency_stats([ms for items in embedded.values() for _, _, ms in items])
    for kind, key in (("ticket", "tickets"), ("article", "articles")):
        for split, items in embedded.items():
            for strat in STRATEGIES:
                out[key].setdefault(split, {})[strat] = await _rank_run(svc, corpus, items, strat, kind)

    # --- experiment: metadata filtering (hybrid_reranked, tickets) ------------------------------------------
    items = embedded["test"]
    preds = []
    for r, emb, _ in items:
        preds.append(await svc.classifier.classify(QueryContext(text=r["text"], embedding=emb)))
    pred_by_q = {r["qid"]: p for (r, _, _), p in zip(items, preds)}
    exp = {"no_filter": out["tickets"]["test"]["hybrid_reranked"]}
    exp["predicted_product_filter"] = await _rank_run(
        svc, corpus, items, "hybrid_reranked", "ticket",
        lambda r: {"product": pred_by_q[r["qid"]].product} if pred_by_q[r["qid"]].confidence.product >= 0.6 else None)
    exp["oracle_product_filter"] = await _rank_run(svc, corpus, items, "hybrid_reranked", "ticket", lambda r: {"product": r["product"]})
    out["experiments"]["metadata_filtering"] = exp
    out["experiments"]["product_filter_accuracy"] = round(float(np.mean([pred_by_q[r["qid"]].product == r["product"] for r, _, _ in items])), 4)

    # --- experiment: reranker candidate pool size ---------------------------------------------------------------
    pool = {}
    original = svc.settings.rerank_top_n
    try:
        for n in (5, 10, 20, 30):
            svc.settings.rerank_top_n = n
            pool[str(n)] = await _rank_run(svc, corpus, items, "hybrid_reranked", "ticket")
    finally:
        svc.settings.rerank_top_n = original
    out["experiments"]["rerank_pool_size"] = pool

    # --- stale-document hygiene: deprecated articles must never be returned -------------------------------------------
    leaks = 0
    for r, emb, _ in items + embedded["gold"]:
        for strat in STRATEGIES:
            res = await svc.retrieval.search(QueryContext(text=r["text"], embedding=emb), "article", strat, 10)
            leaks += sum(1 for x in res if x.source_id in corpus.deprecated)
    out["stale_article_leaks"] = {"deprecated_articles": sorted(corpus.deprecated), "returned_count": leaks}
    return out


# ============================================================ RAG + end-to-end
async def _run_resolutions(res_svc: ResolutionService, rows: list[dict], generate: bool) -> list[dict]:
    runs = []
    for r in rows:
        t = time.perf_counter()
        resp, err = None, None
        try:
            resp = await asyncio.wait_for(res_svc.resolve(ResolveRequest(complaint=r["text"]), generate=generate,
                                                          persist=False, use_cache=False), 120)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        runs.append({"row": r, "resp": resp, "error": err, "wall_ms": (time.perf_counter() - t) * 1000})
    return runs


def _step_texts(resp: ResolveResponse) -> list[str]:
    return [s.text for s in resp.resolution.steps]


async def _rag_metrics(svc: Services, corpus: Corpus, runs: list[dict]) -> dict:
    cited_total = invalid_total = rel_cites = tot_cites = 0
    step_total = halluc_steps = halluc_resp = answered = 0
    gold_recall, rel_scores, grounded, coverage = [], [], [], []
    for run in runs:
        resp: ResolveResponse | None = run["resp"]
        if not resp or not resp.resolution.steps:
            continue
        answered += 1
        scenario = run["row"].get("scenario_id")
        evid = select_evidence(resp.tickets, resp.articles, svc.settings)
        units = [u for e in evid for u in evidence_units(e)]
        steps = _step_texts(resp)
        bad = [s for s in steps if containment(s, units) < 0.5]  # lexical faithfulness, independent of the validator's embedding check
        step_total += len(steps)
        halluc_steps += len(bad)
        halluc_resp += int(bool(bad) or bool(resp.validation.invalid_citations))
        cited_total += resp.validation.citations_emitted
        invalid_total += len(resp.validation.invalid_citations)
        for c in resp.citations:
            tot_cites += 1
            rel_cites += int(corpus.scenario_of.get(c.source_id) == scenario)
        grounded.append(resp.validation.grounded_ratio)
        coverage.append(resp.validation.citation_coverage)
        gold = corpus.article_steps.get(scenario or "", [])
        if gold:
            mat = await asyncio.to_thread(svc.embedder.encode_sync, steps + gold)
            sims = mat[: len(steps)] @ mat[len(steps):].T
            gold_recall.append(float(np.mean(sims.max(axis=0) >= 0.6)))
            rel_scores.append(float(np.mean(sims.max(axis=1))))
    return {
        "answered": answered,
        "step_faithfulness": round(1 - halluc_steps / step_total, 4) if step_total else None,
        "step_hallucination_rate": round(halluc_steps / step_total, 4) if step_total else None,
        "response_hallucination_rate": round(halluc_resp / answered, 4) if answered else None,
        "citation_validity": round(1 - invalid_total / cited_total, 4) if cited_total else None,
        "citation_precision_vs_gold": round(rel_cites / tot_cites, 4) if tot_cites else None,
        "citation_coverage": round(float(np.mean(coverage)), 4) if coverage else None,
        "validator_grounded_ratio": round(float(np.mean(grounded)), 4) if grounded else None,
        "gold_step_recall": round(float(np.mean(gold_recall)), 4) if gold_recall else None,
        "answer_relevance_cosine": round(float(np.mean(rel_scores)), 4) if rel_scores else None,
    }


async def _judge(svc: Services, runs: list[dict], n: int) -> dict:
    """Optional LLM judge (provider/model configurable via JUDGE_*). A 4B self-judge is weak evidence, so it is
    reported separately and never used for gating."""
    s = svc.settings
    judge = OllamaProvider(s.ollama_base_url, s.judge_model) if s.judge_provider == "ollama" else None
    if judge is None:
        return {"skipped": f"unsupported JUDGE_PROVIDER={s.judge_provider}"}
    scores = []
    sample = [r for r in runs if r["resp"] and r["resp"].resolution.steps][:n]
    for run in sample:
        resp = run["resp"]
        evid = "\n".join(f"[{e.source_id}] {e.text[:300]} | {'; '.join(e.steps[:5])}" for e in select_evidence(resp.tickets, resp.articles, s))
        steps = "\n".join(f"{i}. {st.text}" for i, st in enumerate(resp.resolution.steps, 1))
        prompt = (f"COMPLAINT: {run['row']['text']}\n\nEVIDENCE:\n{evid}\n\nANSWER STEPS:\n{steps}\n\n"
                  'Rate 1-5 each: "faithfulness" (every step supported by the evidence) and "relevance" (answers the complaint). '
                  'Reply JSON only: {"faithfulness": n, "relevance": n}')
        try:
            r = await judge.generate("You are a strict evaluator.", prompt, json_mode=True, max_tokens=60, temperature=0.0, timeout=60)
            d = json.loads(r.text)
            scores.append((float(d["faithfulness"]), float(d["relevance"])))
        except Exception as exc:  # noqa: BLE001
            log.warning("judge call failed", extra={"error": str(exc)[:120]})
    if not scores:
        return {"n": 0}
    a = np.asarray(scores)
    return {"n": len(scores), "judge_model": s.judge_model, "faithfulness_1to5": round(float(a[:, 0].mean()), 2),
            "relevance_1to5": round(float(a[:, 1].mean()), 2)}


def _e2e_metrics(runs: list[dict], corpus: Corpus) -> dict:
    groups: dict[str, list[dict]] = {"in_domain": [], "ood": []}
    for r in runs:
        groups["ood" if r["row"].get("expected") == "abstain" else "in_domain"].append(r)
    ind, ood = groups["in_domain"], groups["ood"]

    def status_counts(rs):
        return dict(Counter((r["resp"].status if r["resp"] else "error") for r in rs))

    def correct(r):
        resp = r["resp"]
        return bool(resp and resp.status == "resolved" and any(
            corpus.scenario_of.get(c.source_id) == r["row"]["scenario_id"] for c in resp.citations))

    lat = [r["resp"].latency_ms["total"] for r in runs if r["resp"]]
    llm_lat = [r["resp"].latency_ms["generate"] for r in runs if r["resp"] and r["resp"].generator not in ("none",)]
    return {
        "in_domain": {"n": len(ind), "status": status_counts(ind),
                      "successful_resolution_rate": round(sum(1 for r in ind if r["resp"] and r["resp"].status == "resolved") / max(1, len(ind)), 4),
                      "correct_resolution_rate": round(sum(correct(r) for r in ind) / max(1, len(ind)), 4),
                      "false_abstention_rate": round(sum(1 for r in ind if r["resp"] and r["resp"].status == "abstained") / max(1, len(ind)), 4),
                      "unreliable_rate": round(sum(1 for r in ind if r["resp"] and r["resp"].status == "unreliable") / max(1, len(ind)), 4)},
        "out_of_domain": {"n": len(ood), "status": status_counts(ood),
                          "correct_abstention_rate": round(sum(1 for r in ood if r["resp"] and r["resp"].status == "abstained") / max(1, len(ood)), 4)},
        "overall_abstention_rate": round(sum(1 for r in runs if r["resp"] and r["resp"].status == "abstained") / max(1, len(runs)), 4),
        "latency_total": latency_stats(lat), "latency_generation_stage": latency_stats(llm_lat),
        "failures": {"pipeline_errors": sum(1 for r in runs if r["error"]),
                     "retrieval_failures": sum(1 for r in runs if r["error"] and "retrieval" in r["error"].lower()),
                     "generation_degraded": sum(1 for r in runs if r["resp"] and r["resp"].status == "degraded"),
                     "citation_validation_failures": sum(1 for r in runs if r["resp"] and r["resp"].status == "unreliable")},
        "tokens": None,
    }


async def rag_e2e_suite(svc: Services, sets: EvalSets, max_queries: int | None, want: set[str], judge_n: int = 0,
                        variants: Sequence[tuple[str, ResilientLLM | None]] = ()) -> dict:
    corpus = await build_corpus(svc.repo)
    in_rows = _sample(sets.test, max_queries or 40)
    gold = _sample(sets.gold, max_queries) if max_queries else sets.gold
    ood = _sample(sets.ood, max_queries) if max_queries else sets.ood
    rows = in_rows + gold + ood
    out: dict = {"sample": {"in_domain_test": len(in_rows), "gold": len(gold), "ood": len(ood)}}
    primary = f"{svc.llm.providers[0].name}:{svc.llm.providers[0].model}" if svc.llm.providers else "extractive"
    runs = await _run_resolutions(svc.resolution, rows, generate=True)
    out["primary_generator"] = primary
    if "rag" in want:
        out["rag"] = {"primary": {"generator": primary, **await _rag_metrics(svc, corpus, [r for r in runs if r["row"].get("expected") != "abstain"])}}
        if judge_n:
            out["rag"]["llm_judge"] = await _judge(svc, runs, judge_n)
        # design alternative: evidence-only (no LLM) and any extra providers, on the same in-domain sample
        for name, llm in variants:
            alt = svc.resolution if llm is None else ResolutionService(svc.repo, svc.retrieval, svc.classifier, svc.taxonomy, llm,
                                                                       svc.embedder, svc.cache, svc.settings)
            t = time.perf_counter()
            alt_runs = await _run_resolutions(alt, in_rows, generate=llm is not None)
            m = await _rag_metrics(svc, corpus, alt_runs)
            m["generator"] = name
            m["latency_total"] = latency_stats([r["resp"].latency_ms["total"] for r in alt_runs if r["resp"]])
            m["status"] = dict(Counter((r["resp"].status if r["resp"] else "error") for r in alt_runs))
            m["wall_seconds"] = round(time.perf_counter() - t, 1)
            out["rag"][name] = m
    if "e2e" in want:
        out["e2e"] = _e2e_metrics(runs, corpus)
    return out


# ============================================================ evolving data
async def evolving_suite(svc: Services, sets: EvalSets, data_dir) -> dict:
    """New classes + new documents become usable immediately, with no restart, rebuild or redeploy."""
    from app.evaluation.datasets import load_jsonl

    raw = data_dir / "raw"
    proc = data_dir / "processed"
    tax = json.loads((raw / "evolving_taxonomy.json").read_text(encoding="utf-8"))
    new_tickets = load_jsonl(proc / "evolving_tickets.jsonl")
    new_articles = load_jsonl(proc / "evolving_articles.jsonl")
    queries = sets.evolving

    async def measure(label: str) -> dict:
        corpus = await build_corpus(svc.repo)
        rr, intents_ok, statuses = [], [], []
        for q in queries:
            ctx = await svc.retrieval.make_context(q["text"])
            res = await svc.retrieval.search(ctx, "ticket", "hybrid_reranked", 10)
            rr.append(retrieval_metrics([x.source_id for x in res], corpus.relevant("ticket", q["scenario_id"]), (1, 5)))
            cls = await svc.classifier.classify(ctx)
            intents_ok.append(cls.intent == q["intent"])
            resp = await svc.resolution.resolve(ResolveRequest(complaint=q["text"]), generate=False, persist=False, use_cache=True)
            statuses.append(resp.status)
        return {"stage": label, **mean_dicts(rr), "intent_accuracy": round(float(np.mean(intents_ok)), 4),
                "status": dict(Counter(statuses)), "n": len(queries)}

    created_labels: list[str] = []
    ingested_t: list[str] = []
    ingested_a: list[str] = []
    out: dict = {}
    try:
        out["before_ingestion"] = await measure("before")
        v0 = await svc.repo.corpus_version()
        t0 = time.perf_counter()
        for lab in tax["intent"]:
            if not svc.taxonomy.current.has("intent", lab["id"]):
                await svc.taxonomy.add_label("intent", lab["id"], lab["description"], lab["keywords"], lab["examples"], lab.get("team"))
                created_labels.append(lab["id"])
        t_tax = time.perf_counter() - t0
        t1 = time.perf_counter()
        for a in new_articles:
            a = dict(a)
            a.pop("status", None)
            ingested_a.append((await svc.ingestion.ingest_article(ArticleIn(**a))).source_id)
        first = await svc.ingestion.ingest_ticket(TicketIn(**new_tickets[0]), source="eval_evolving")
        ingested_t.append(first.source_id)
        bulk = await svc.ingestion.ingest_tickets_bulk([TicketIn(**t) for t in new_tickets[1:]], source="eval_evolving")
        ingested_t += [t["ticket_id"] for t in new_tickets[1:]]
        t_ing = time.perf_counter() - t1
        out["after_ingestion"] = await measure("after")
        out["corpus_version"] = {"before": v0, "after": await svc.repo.corpus_version()}
        out["timings"] = {"add_taxonomy_labels_s": round(t_tax, 2), "ingest_2_articles_and_12_tickets_s": round(t_ing, 2),
                          "single_ticket_ingest_ms": first.embedding_ms}
        out["bulk_result"] = {k: bulk[k] for k in ("created", "updated", "failed")}
        out["restart_required"] = False
        out["added_intents"] = created_labels
    finally:  # leave the corpus exactly as we found it
        if ingested_t:
            await svc.repo.delete_by_ids("ticket", ingested_t)
        if ingested_a:
            await svc.repo.delete_by_ids("article", ingested_a)
        if created_labels:
            await svc.repo.delete_taxonomy_labels("intent", created_labels)
            await svc.taxonomy.refresh()
        await svc.repo.bump_corpus_version()
        await svc.ingestion.refresh_gauges()
    return out
