"""Robustness suite: how much do classification and retrieval degrade when complaints are messy?

Real agents paste raw text: typos, no punctuation, signatures and unrelated asides, half-finished sentences, ALL CAPS.
Every perturbation is deterministic (seeded by the query id) so runs are comparable and the regression gate is stable.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
import string
from collections import defaultdict

import numpy as np

from app.classification.taxonomy import DIMENSIONS
from app.evaluation.datasets import EvalSets, build_corpus
from app.evaluation.metrics import retrieval_metrics
from app.services.container import Services

DISTRACTORS = (
    "Also, please update my delivery address when you get a chance.",
    "By the way, my neighbour mentioned your new offers.",
    "I will be away next weekend so a weekday call would be best.",
)
SIGNATURES = ("Thanks, Sam. Account ref 88213. Sent from my iPhone", "Regards, J. Patel (customer since 2019)", "thx!! - Dee")


def _rng(qid: str, tag: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{tag}:{qid}".encode()).hexdigest()[:12], 16))


def typos(text: str, qid: str, rate: float = 0.12) -> str:
    rng = _rng(qid, "typos")
    out = []
    for w in text.split(" "):
        core = re.sub(r"\W", "", w)
        if len(core) > 3 and rng.random() < rate:
            i = rng.randrange(1, len(w) - 1)
            op = rng.choice(("swap", "drop", "dup"))
            w = (w[:i] + w[i + 1] + w[i] + w[i + 2:]) if op == "swap" else (w[:i] + w[i + 1:] if op == "drop" else w[:i] + w[i] + w[i:])
        out.append(w)
    return " ".join(out)


def lowercase_nopunct(text: str, qid: str) -> str:
    return re.sub(r"\s+", " ", text.lower().translate(str.maketrans("", "", string.punctuation))).strip()


def noisy_wrapper(text: str, qid: str) -> str:
    rng = _rng(qid, "noise")
    return f"Hi team, {text} {rng.choice(DISTRACTORS)} {rng.choice(SIGNATURES)}"


def truncated(text: str, qid: str, keep: float = 0.6) -> str:
    words = text.split()
    return " ".join(words[: max(4, math.ceil(len(words) * keep))])


def shouting(text: str, qid: str) -> str:
    return text.upper()


PERTURBATIONS = {"clean": lambda t, q: t, "typos": typos, "no_punctuation_lowercase": lowercase_nopunct,
                 "noise_and_signature": noisy_wrapper, "truncated_60pct": truncated, "all_caps": shouting}


async def robustness_suite(svc: Services, sets: EvalSets, max_queries: int | None = None) -> dict:
    rows = sets.gold + sets.blind
    if max_queries:
        rows = rows[:max_queries]
    corpus = await build_corpus(svc.repo)
    out: dict = {"n": len(rows), "perturbations": {}}
    for name, fn in PERTURBATIONS.items():
        acc: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            text = fn(r["text"], r["qid"])
            ctx = await svc.retrieval.make_context(text)
            cls = await svc.classifier.classify(ctx)
            for d in DIMENSIONS:
                acc[d].append(float(getattr(cls, d) == r[d]))
            ranked = [x.source_id for x in await svc.retrieval.search(ctx, "ticket", "dense", 10)]
            m = retrieval_metrics(ranked, corpus.relevant("ticket", r["scenario_id"]), (1, 5))
            acc["ticket_hit@1"].append(m["hit@1"])
            acc["ticket_hit@5"].append(m["hit@5"])
            art = [x.source_id for x in await svc.retrieval.search(ctx, "article", "dense", 3)]
            acc["article_hit@3"].append(float(bool(set(art) & corpus.relevant("article", r["scenario_id"]))))
        out["perturbations"][name] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    clean = out["perturbations"]["clean"]
    out["max_drop_vs_clean"] = {k: round(max(clean[k] - p[k] for p in out["perturbations"].values()), 4) for k in clean}
    return out
