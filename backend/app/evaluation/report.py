"""Render results JSON -> markdown tables (so every number in the docs comes from a recorded run)."""
from __future__ import annotations

from typing import Any

DIMS = ("intent", "product", "severity", "sentiment")


def _f(v: Any, nd: int = 3) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(_f(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(r: dict) -> str:
    m = r.get("meta", {})
    L = ["# Evaluation results (auto-generated)", "",
         f"Generated {m.get('timestamp')} by `python -m app.evaluation.run`. Embedding `{m.get('embedding_model')}`, "
         f"reranker `{m.get('reranker')}`, LLM `{m.get('llm')}`. Corpus: {m.get('corpus')}. Eval sets: {m.get('eval_sets')}. "
         f"Thresholds: {m.get('thresholds')}.", ""]

    if "classification" in r:
        c = r["classification"]
        L += ["## 1. Classification", "", "Held-out paraphrase queries (`test`, templated: severity/sentiment rules share vocabulary with the generator, so "
              "those numbers are circular), the hand-written `gold` set (101) and the `blind` set (50, written after the affect model was frozen; "
              "never used for any selection). `ensemble_legacy` = rules + kNN for every dimension (before the affect model); `ensemble` = shipped.", ""]
        for split in ("test", "gold", "blind"):
            rows = []
            for strat, d in c["strategies"].items():
                if split not in d:
                    continue
                for dim in DIMS:
                    x = d[split][dim]
                    rows.append([strat, dim, x["accuracy"], x["macro_precision"], x["macro_recall"], x["macro_f1"], x["n"]])
            L += [f"**split = {split}**", "", table(["strategy", "dimension", "accuracy", "macro-P", "macro-R", "macro-F1", "n"], rows), ""]
        L += ["Ensemble confidence calibration (test): mean confidence when correct vs wrong", "",
              table(["dimension", "mean conf (correct)", "mean conf (wrong)"],
                    [[d, v["mean_conf_correct"], v["mean_conf_wrong"]] for d, v in c["calibration"].items()]), ""]
        lat = c["strategies"]["ensemble"]["test"]["latency"]
        L += [f"Ensemble classification latency (embedding excluded): p50 {lat['p50_ms']} ms, p95 {lat['p95_ms']} ms.", ""]

    if "retrieval" in r:
        rt = r["retrieval"]
        L += ["## 2. Retrieval", "", f"Query-embedding latency: {rt['embedding_latency']}. A document is relevant iff it belongs to the same root-cause scenario as the query "
              "(about 10 relevant tickets and 1 relevant KB article per query). Retrieval latency excludes query embedding.", ""]
        for key in ("tickets", "articles"):
            for split, strategies in rt[key].items():
                rows = [[s, x["p@5"], x["recall@10"], x["hit@1"], x["hit@5"], x["mrr"], x["ndcg@10"], x["latency"]["p50_ms"], x["latency"]["p95_ms"]]
                        for s, x in strategies.items()]
                L += [f"**{key}, split = {split}** (n = {next(iter(strategies.values()))['n']})", "",
                      table(["strategy", "P@5", "Recall@10", "Hit@1", "Hit@5", "MRR", "nDCG@10", "p50 ms", "p95 ms"], rows), ""]
        L += ["### Top-K sweep (tickets, test)", ""]
        rows = []
        for s in ("lexical", "bm25", "dense", "hybrid", "hybrid_reranked"):
            x = rt["tickets"]["test"][s]
            rows.append([s] + [x[f"hit@{k}"] for k in (1, 3, 5, 10)] + [x[f"ndcg@{k}"] for k in (1, 3, 5, 10)] + [x[f"p@{k}"] for k in (1, 3, 5, 10)])
        L += [table(["strategy"] + [f"Hit@{k}" for k in (1, 3, 5, 10)] + [f"nDCG@{k}" for k in (1, 3, 5, 10)] + [f"P@{k}" for k in (1, 3, 5, 10)], rows), ""]
        e = rt["experiments"]
        L += ["### Metadata filtering (hybrid_reranked, tickets, test)", "",
              table(["variant", "P@5", "Hit@1", "MRR", "nDCG@10", "p50 ms"],
                    [[k, v["p@5"], v["hit@1"], v["mrr"], v["ndcg@10"], v["latency"]["p50_ms"]] for k, v in e["metadata_filtering"].items()]), "",
              f"Predicted-product accuracy on these queries: {e['product_filter_accuracy']}.", "",
              "### Reranker candidate pool size (tickets, test)", "",
              table(["rerank_top_n", "P@5", "Hit@1", "MRR", "nDCG@10", "p50 ms"],
                    [[k, v["p@5"], v["hit@1"], v["mrr"], v["ndcg@10"], v["latency"]["p50_ms"]] for k, v in e["rerank_pool_size"].items()]), "",
              f"Stale-document hygiene: deprecated articles {rt['stale_article_leaks']['deprecated_articles']} returned "
              f"{rt['stale_article_leaks']['returned_count']} times across all strategies and queries.", ""]

    if "robustness" in r:
        rb = r["robustness"]
        cols = ["intent", "product", "severity", "sentiment", "ticket_hit@1", "ticket_hit@5", "article_hit@3"]
        L += ["## 2b. Robustness to messy input", "", f"Hand-written gold + blind queries (n = {rb['n']}) under deterministic perturbations; "
              "ensemble classifier and dense retrieval.", "",
              table(["perturbation"] + cols, [[p] + [v[c] for c in cols] for p, v in rb["perturbations"].items()]), "",
              f"Largest drop versus clean per metric: {rb['max_drop_vs_clean']}", ""]

    if "rag" in r:
        L += ["## 3. Generation / RAG", "", f"Sample sizes: {r['sample']}. Primary generator: `{r['primary_generator']}`.", ""]
        rows = []
        for name, x in r["rag"].items():
            if name == "llm_judge":
                continue
            rows.append([name, x.get("generator", name), x["answered"], x["step_faithfulness"], x["step_hallucination_rate"],
                         x["response_hallucination_rate"], x["citation_validity"], x["citation_precision_vs_gold"], x["citation_coverage"],
                         x["gold_step_recall"], x["answer_relevance_cosine"]])
        L += [table(["run", "generator", "answered", "step faithfulness", "step halluc.", "response halluc.", "citation validity",
                     "citation precision (gold)", "citation coverage", "gold-step recall", "answer relevance (cos)"], rows), ""]
        for name, x in r["rag"].items():
            if name not in ("primary", "llm_judge") and "latency_total" in x:
                L.append(f"- `{name}`: status {x['status']}, latency {x['latency_total']}")
        if "llm_judge" in r["rag"]:
            L += ["", f"Optional LLM judge: {r['rag']['llm_judge']}"]
        L.append("")

    if "e2e" in r:
        e = r["e2e"]
        L += ["## 4. End-to-end", "", table(["metric", "value"], [
            ["in-domain n", e["in_domain"]["n"]], ["successful resolution rate (status=resolved)", e["in_domain"]["successful_resolution_rate"]],
            ["correct resolution rate (resolved and cites gold-relevant source)", e["in_domain"]["correct_resolution_rate"]],
            ["false abstention rate (in-domain)", e["in_domain"]["false_abstention_rate"]], ["unreliable rate", e["in_domain"]["unreliable_rate"]],
            ["out-of-domain n", e["out_of_domain"]["n"]], ["correct abstention rate (out-of-domain)", e["out_of_domain"]["correct_abstention_rate"]],
            ["overall abstention rate", e["overall_abstention_rate"]], ["latency total (ms)", e["latency_total"]],
            ["latency generation stage (ms)", e["latency_generation_stage"]], ["failures", e["failures"]]]), "",
            f"In-domain status counts: {e['in_domain']['status']}; OOD status counts: {e['out_of_domain']['status']}.", ""]

    if "evolving" in r:
        v = r["evolving"]
        rows = [[k, x["hit@1"], x["hit@5"], x["mrr"], x["intent_accuracy"], x["status"]] for k, x in (("before ingestion", v["before_ingestion"]),
                                                                                                    ("after ingestion", v["after_ingestion"]))]
        L += ["## 5. Evolving data", "", f"Two new intents ({v.get('added_intents')}) with 2 KB articles and 12 tickets were added at runtime through the "
              "normal services - no restart, no index rebuild.", "",
              table(["stage", "Hit@1", "Hit@5", "MRR", "intent accuracy", "resolve status"], rows), "",
              f"Timings: {v['timings']}; corpus version {v['corpus_version']}; restart required: {v['restart_required']}.", ""]
    return "\n".join(L)
