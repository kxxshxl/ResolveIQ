"""Pure metric functions (no I/O) so they are unit-testable and deterministic."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return (sum(1 for d in ranked[:k] if d in relevant) / len(relevant)) if relevant else 0.0


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(d in relevant for d in ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(ranked: Sequence[str], relevant: set[str], ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    out = {"mrr": reciprocal_rank(ranked, relevant)}
    for k in ks:
        out[f"p@{k}"] = precision_at_k(ranked, relevant, k)
        out[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        out[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
    return out


def mean_dicts(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            acc[k].append(v)
    return {k: round(float(np.mean(v)), 4) for k, v in acc.items()}


def latency_stats(ms: Sequence[float]) -> dict[str, float]:
    if not ms:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "n": 0}
    a = np.asarray(ms, dtype=float)
    return {"avg_ms": round(float(a.mean()), 1), "p50_ms": round(float(np.percentile(a, 50)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1), "n": int(a.size)}


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """Accuracy + macro-averaged precision/recall/F1 over the labels present in y_true."""
    labels = sorted(set(y_true))
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0
    ps, rs, fs = [], [], []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        ps.append(pr)
        rs.append(rc)
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"accuracy": round(acc, 4), "macro_precision": round(float(np.mean(ps)), 4),
            "macro_recall": round(float(np.mean(rs)), 4), "macro_f1": round(float(np.mean(fs)), 4), "n": len(y_true)}
