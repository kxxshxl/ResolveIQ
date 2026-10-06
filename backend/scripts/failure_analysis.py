"""Where does the system get hand-written complaints wrong, and why? (no LLM; reads the live corpus, writes data/eval/results/failure_analysis.json)

    python scripts/failure_analysis.py

For the hand-written `gold` and `blind` complaints (151) it records every dense-retrieval miss (the top-1 ticket is from another root cause), grouped into confusions between
root causes, with the rank at which the right ticket did appear; every intent, severity and sentiment error of the shipped classifier, grouped into confusions; and the queries on
which the adaptive ladder and dense retrieval disagree. The end-to-end failures (the resolutions that did not cite the right scenario) are recorded by the e2e suite itself.
The evaluation queries are synthetic or hand-written: they contain no customer data.
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402,F401  (Windows event-loop policy)
from app.core.config import REPO_ROOT, get_settings  # noqa: E402
from app.evaluation.datasets import build_corpus, load_eval_sets  # noqa: E402
from app.retrieval.service import QueryContext  # noqa: E402
from app.services.container import Services  # noqa: E402

OUT = REPO_ROOT / "data" / "eval" / "results" / "failure_analysis.json"


def top(counter: Counter, n: int = 6, **extra) -> list[dict]:
    return [{"true": a, "predicted": b, "count": c, **extra} for (a, b), c in counter.most_common(n)]


async def main() -> None:
    svc = await Services.build(get_settings())
    try:
        sets = load_eval_sets(REPO_ROOT / "data")
        corpus = await build_corpus(svc.repo)
        intent_of_scenario: dict[str, str] = {}
        for r in await svc.repo.all_docs("ticket"):
            sid = (r["metadata"] or {}).get("scenario_id")
            if sid:
                intent_of_scenario[sid] = r["intent"]
        rows = [(split, r) for split, rs in (("gold", sets.gold), ("blind", sets.blind)) for r in rs]
        miss, conf_scen, examples = Counter(), Counter(), []
        cls_err = {d: Counter() for d in ("intent", "product", "severity", "sentiment")}
        cls_n = Counter()
        disagree = {"adaptive_better": [], "adaptive_worse": []}
        same_intent_miss = 0
        for split, r in rows:
            ctx = await svc.retrieval.make_context(r["text"])
            dense = await svc.retrieval.search(ctx, "ticket", "dense", 10)
            ids = [x.source_id for x in dense]
            rel = corpus.relevant("ticket", r["scenario_id"])
            if ids and ids[0] not in rel:
                got = corpus.scenario_of.get(ids[0], "?")
                miss[split] += 1
                conf_scen[(r["scenario_id"], got)] += 1
                same_intent_miss += intent_of_scenario.get(got) == r.get("intent")
                rank = next((i for i, s in enumerate(ids, 1) if s in rel), None)
                examples.append({"split": split, "qid": r.get("qid"), "text": r["text"], "true_scenario": r["scenario_id"], "retrieved_scenario": got, "rank_of_first_relevant": rank})
            ad = await svc.retrieval.search(QueryContext(text=r["text"], embedding=ctx.embedding), "ticket", "adaptive", 10)
            a_ok = bool(ad) and ad[0].source_id in rel
            d_ok = bool(ids) and ids[0] in rel
            if a_ok != d_ok:
                disagree["adaptive_better" if a_ok else "adaptive_worse"].append(r.get("qid"))
            cls = await svc.classifier.classify(await svc.retrieval.make_context(r["text"]))
            cls_n[split] += 1
            for dim in cls_err:
                truth, pred = r.get(dim), getattr(cls, dim)
                if truth is not None and truth != pred:
                    cls_err[dim][(truth, pred)] += 1
        out = {
            "note": "hand-written gold and blind complaints (151); dense retrieval and the shipped classifier; no LLM. Evaluation queries are synthetic or hand-written, no customer data.",
            "queries": {"gold": cls_n["gold"], "blind": cls_n["blind"]},
            "retrieval": {"top1_wrong": dict(miss), "top1_wrong_but_same_intent": same_intent_miss,
                          "confusions_true_to_retrieved_scenario": top(conf_scen, 8),
                          "examples": examples[:10]},
            "classification": {d: {"errors": sum(c.values()), "confusions": top(c, 5)} for d, c in cls_err.items()},
            "adaptive_vs_dense_top1": {k: len(v) for k, v in disagree.items()},
        }
        OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {OUT}")
    finally:
        await svc.close()


if __name__ == "__main__":
    asyncio.run(main())
