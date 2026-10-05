"""In-memory BM25 - an EVALUATION-ONLY stronger keyword baseline.

Postgres FTS ranking (ts_rank_cd) has no IDF term, so a reviewer could argue it is a strawman
lexical baseline. We therefore also benchmark classic BM25 (Okapi) to make the semantic-vs-keyword
comparison fair. It is rebuilt whenever the corpus version changes and is not used in production
(it would not scale beyond memory); production lexical search is Postgres FTS.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "ly", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class Bm25Index:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.index = BM25Okapi([tokenize(d["title"] + " " + d["text"] if d.get("resolution_summary") is None else d["text"]) for d in docs]) if docs else None

    def search(self, query: str, k: int) -> list[dict]:
        if not self.index:
            return []
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{**self.docs[i], "score": float(scores[i])} for i in order if scores[i] > 0]
