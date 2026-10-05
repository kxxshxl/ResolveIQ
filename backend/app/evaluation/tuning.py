"""Hyper-parameter selection on the VALIDATION split only (never on `test` or `gold`).

  python -m app.evaluation.tuning

Sweeps (1) weighted-RRF / reranker-blend, (2) per-dimension rules-vs-kNN ensemble weights,
(3) the evidence-abstention threshold using in-domain val queries vs out-of-domain queries.
Prints the grids and the winners; the chosen values are then written into config / pipeline defaults by hand
and recorded in docs/evaluation.md so the provenance is explicit.
"""
from __future__ import annotations

import asyncio
import itertools
import logging

import numpy as np

from app.classification.pipeline import ENSEMBLE_WEIGHTS
from app.classification.taxonomy import DIMENSIONS
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.evaluation.datasets import build_corpus, load_eval_sets
from app.evaluation.metrics import mean_dicts, retrieval_metrics
from app.models.schemas import ResolveRequest
from app.retrieval.service import QueryContext
from app.services.container import Services


async def tune_retrieval(svc: Services, val: list[dict], corpus) -> tuple[float, float, float]:
    embs = [(r, (await svc.retrieval.make_context(r["text"])).embedding) for r in val]
    best, best_score = (1.0, 1.0, 0.2), -1.0
    print("\n== retrieval: lexical IDF cut-off, weighted RRF, rerank blend (val; objective = mean of ticket MRR, ticket nDCG@10, article MRR) ==")
    print("max_df w_lex  blend  t_hit@1  t_mrr  t_ndcg10  a_mrr  objective")
    for max_df, w_lex, blend in itertools.product((0.1, 0.2, 0.4), (0.0, 0.25, 0.5, 1.0), (0.0, 0.5, 1.0)):
        svc.settings.lexical_max_df, svc.settings.rrf_weight_lexical, svc.settings.rerank_blend = max_df, w_lex, blend
        t_rows, a_rows = [], []
        for r, emb in embs:
            for kind, rows in (("ticket", t_rows), ("article", a_rows)):
                res = await svc.retrieval.search(QueryContext(text=r["text"], embedding=emb), kind, "hybrid_reranked", 10)
                rows.append(retrieval_metrics([x.source_id for x in res], corpus.relevant(kind, r["scenario_id"]), (1, 10)))
        t, a = mean_dicts(t_rows), mean_dicts(a_rows)
        obj = (t["mrr"] + t["ndcg@10"] + a["mrr"]) / 3
        print(f"{max_df:<6} {w_lex:<6} {blend:<6} {t['hit@1']:<8} {t['mrr']:<6} {t['ndcg@10']:<9} {a['mrr']:<6} {obj:.4f}")
        if obj > best_score:
            best, best_score = (w_lex, blend, max_df), obj
    print(f"-> best: rrf_weight_lexical={best[0]} rerank_blend={best[1]} lexical_max_df={best[2]}  (objective {best_score:.4f})")
    return best


async def tune_ensemble(svc: Services, val: list[dict]) -> dict:
    embs = [(r, QueryContext(text=r["text"], embedding=(await svc.retrieval.make_context(r["text"])).embedding)) for r in val]
    tax = svc.taxonomy.current
    rules = [await svc.classifier.rules.predict(c, tax) for _, c in embs]
    knn = [await svc.classifier.embedding.predict(c, tax) for _, c in embs]
    print("\n== ensemble weights per dimension (val accuracy; weight = rules share, kNN = 1 - w) ==")
    chosen = {}
    for dim in DIMENSIONS:
        scores = {}
        for w in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
            ok = 0
            for (r, _), rd, kd in zip(embs, rules, knn):
                labels = set(rd[dim]) | set(kd[dim])
                comb = {l: w * rd[dim].get(l, 0.0) + (1 - w) * kd[dim].get(l, 0.0) for l in labels}
                pred = max(comb, key=comb.get) if comb else None
                ok += pred == r[dim]
            scores[w] = ok / len(embs)
        best_w = max(scores, key=lambda w: (scores[w], -abs(w - 0.5)))
        chosen[dim] = (round(best_w, 2), round(1 - best_w, 2))
        print(f"{dim:<10}", {k: round(v, 3) for k, v in scores.items()}, "->", chosen[dim])
    return chosen


async def tune_abstention(svc: Services, val: list[dict], ood: list[dict]) -> float:
    conf_in, conf_out = [], []
    for r in val:
        resp = await svc.resolution.resolve(ResolveRequest(complaint=r["text"]), generate=False, persist=False, use_cache=False)
        conf_in.append(resp.evidence.confidence)
    for r in ood:
        resp = await svc.resolution.resolve(ResolveRequest(complaint=r["text"]), generate=False, persist=False, use_cache=False)
        conf_out.append(resp.evidence.confidence)
    print("\n== abstention threshold (val in-domain vs OOD) ==")
    print(f"in-domain confidence: min {min(conf_in):.3f} p10 {np.percentile(conf_in, 10):.3f} median {np.median(conf_in):.3f}")
    print(f"OOD confidence:       max {max(conf_out):.3f} p90 {np.percentile(conf_out, 90):.3f} median {np.median(conf_out):.3f}")
    best_t, best_bal = 0.5, -1.0
    print("threshold  answered_in_domain  abstained_ood  balanced")
    for t in np.arange(0.30, 0.86, 0.05):
        tpr = float(np.mean([c >= t for c in conf_in]))
        tnr = float(np.mean([c < t for c in conf_out]))
        bal = (tpr + tnr) / 2
        print(f"{t:.2f}       {tpr:.3f}               {tnr:.3f}          {bal:.3f}")
        if bal > best_bal + 1e-9:
            best_t, best_bal = float(t), bal
    print(f"-> best balanced threshold = {best_t:.2f} (balanced accuracy {best_bal:.3f})")
    return best_t


async def main() -> None:
    configure_logging("ERROR")
    logging.getLogger("app").setLevel(logging.ERROR)
    svc = await Services.build(get_settings())
    try:
        sets = load_eval_sets(svc.settings.data_dir)
        corpus = await build_corpus(svc.repo)
        print(f"validation queries: {len(sets.val)} (test/gold untouched), OOD: {len(sets.ood)}")
        w_lex, blend, max_df = await tune_retrieval(svc, sets.val, corpus)
        svc.settings.rrf_weight_lexical, svc.settings.rerank_blend, svc.settings.lexical_max_df = w_lex, blend, max_df  # abstention is tuned on top of the chosen retrieval
        print("current ENSEMBLE_WEIGHTS:", ENSEMBLE_WEIGHTS)
        await tune_ensemble(svc, sets.val)
        await tune_abstention(svc, sets.val, sets.ood)
    finally:
        await svc.close()


if __name__ == "__main__":
    asyncio.run(main())
