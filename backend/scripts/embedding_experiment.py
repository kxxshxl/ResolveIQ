"""Compare candidate embedding models for dense retrieval, in memory (no DB needed).

  python scripts/embedding_experiment.py [model ...]

Selection is made on the `val` split; `test` is reported for the chosen model so the headline numbers stay clean.
Only 384-dim models are compared because the schema stores vector(384); other dims need a new embeddings table
(see docs/production.md, "embedding model migration").
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation.metrics import mean_dicts, retrieval_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "thenlper/gte-small",
           "sentence-transformers/paraphrase-MiniLM-L6-v2", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"]


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(models: list[str]) -> None:
    from sentence_transformers import SentenceTransformer

    tickets = [t for t in read(ROOT / "data/processed/tickets.jsonl")]
    articles = [a for a in read(ROOT / "data/processed/articles.jsonl") if a["status"] == "active"]
    queries = read(ROOT / "data/eval/queries.jsonl")
    gold = read(ROOT / "data/eval/gold.jsonl")
    rel_t = {}
    for t in tickets:
        rel_t.setdefault(t["metadata"]["scenario_id"], set()).add(t["ticket_id"])
    rel_a = {a["metadata"]["scenario_id"]: {a["article_id"]} for a in articles}
    print(f"{'model':<52} {'split':<5} {'tHit@1':>7} {'tMRR':>6} {'tnDCG10':>8} {'aHit@1':>7} {'aMRR':>6}  enc_ms/q")
    for name in models:
        try:
            m = SentenceTransformer(name)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<52} FAILED to load: {str(exc)[:60]}")
            continue
        enc = lambda xs: m.encode(xs, normalize_embeddings=True, batch_size=64, show_progress_bar=False)  # noqa: E731
        T = enc([t["complaint_text"] for t in tickets])
        A = enc([a["title"] + ". " + a["content"] for a in articles])
        for split, rows in (("val", [q for q in queries if q["split"] == "val"]), ("test", [q for q in queries if q["split"] == "test"]),
                            ("gold", gold)):
            t0 = time.perf_counter()
            Q = enc([q["text"] for q in rows])
            ms = (time.perf_counter() - t0) * 1000 / len(rows)
            tm, am = [], []
            for q, v in zip(rows, Q):
                rt = [tickets[i]["ticket_id"] for i in np.argsort(-(T @ v))[:10]]
                ra = [articles[i]["article_id"] for i in np.argsort(-(A @ v))[:10]]
                tm.append(retrieval_metrics(rt, rel_t[q["scenario_id"]], (1, 10)))
                am.append(retrieval_metrics(ra, rel_a[q["scenario_id"]], (1, 10)))
            t, a = mean_dicts(tm), mean_dicts(am)
            print(f"{name:<52} {split:<5} {t['hit@1']:>7} {t['mrr']:>6} {t['ndcg@10']:>8} {a['hit@1']:>7} {a['mrr']:>6}  {ms:.1f}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
