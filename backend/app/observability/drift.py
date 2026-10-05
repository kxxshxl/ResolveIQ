"""Production health signals that need no labels: is live traffic drifting away from what the system was built for?

Compares a recent window against a baseline window of logged /resolve requests. Rising abstention, falling evidence
confidence, or a shifting intent mix are the earliest, cheapest indicators of new ticket classes, a stale knowledge
base, or an embedding/model regression (labelled evals only run in CI, so this is the live counterpart).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.observability.metrics import DRIFT_JS, RECENT_ABSTENTION, RECENT_EVIDENCE, RECENT_NEGATIVE_FEEDBACK


@dataclass
class DriftThresholds:
    min_requests: int = 30            # below this the comparison is too noisy to alert on
    abstention_increase: float = 0.10
    evidence_drop: float = 0.05
    js_divergence: float = 0.10       # base-2 Jensen-Shannon, 0 (same) .. 1 (disjoint)
    negative_feedback_rate: float = 0.35


def distribution(values: list[str]) -> dict[str, float]:
    c = Counter(values)
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    a = np.array([p.get(k, 0.0) for k in keys])
    b = np.array([q.get(k, 0.0) for k in keys])
    m = (a + b) / 2

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return round(max(0.0, 0.5 * kl(a, m) + 0.5 * kl(b, m)), 4)


def summarize(rows: list[dict]) -> dict:
    """rows: {status, intent, severity, sentiment, evidence, latency_ms}"""
    n = len(rows)
    ev = [r["evidence"] for r in rows if r.get("evidence") is not None]
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    return {
        "n": n,
        "abstention_rate": round(sum(r["status"] == "abstained" for r in rows) / n, 4) if n else None,
        "mean_evidence": round(float(np.mean(ev)), 4) if ev else None,
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 1) if lat else None,
        "status": distribution([r["status"] for r in rows]),
        "intent": distribution([r["intent"] for r in rows if r.get("intent")]),
        "severity": distribution([r["severity"] for r in rows if r.get("severity")]),
        "sentiment": distribution([r["sentiment"] for r in rows if r.get("sentiment")]),
    }


def compare(recent: dict, baseline: dict, negative_feedback_rate: float | None, t: DriftThresholds | None = None) -> dict:
    t = t or DriftThresholds()
    out: dict = {"recent": recent, "baseline": baseline, "drift": {}, "alerts": []}
    for dim in ("intent", "severity", "sentiment"):
        out["drift"][f"js_{dim}"] = js_divergence(recent[dim], baseline[dim])
    out["drift"]["negative_feedback_rate"] = negative_feedback_rate
    if recent["n"] < t.min_requests or baseline["n"] < t.min_requests:
        out["status"] = "insufficient_data"
        return out
    if recent["abstention_rate"] - baseline["abstention_rate"] > t.abstention_increase:
        out["alerts"].append({"signal": "abstention_rate", "message": f"abstention rose from {baseline['abstention_rate']:.2f} to {recent['abstention_rate']:.2f}: "
                              "possible new ticket class or stale knowledge base (run class discovery)"})
    if recent["mean_evidence"] is not None and baseline["mean_evidence"] is not None and baseline["mean_evidence"] - recent["mean_evidence"] > t.evidence_drop:
        out["alerts"].append({"signal": "mean_evidence", "message": f"mean evidence confidence fell from {baseline['mean_evidence']:.2f} to {recent['mean_evidence']:.2f}: "
                              "retrieval quality regression or vocabulary shift"})
    for dim in ("intent", "severity", "sentiment"):
        if out["drift"][f"js_{dim}"] > t.js_divergence:
            out["alerts"].append({"signal": f"{dim}_distribution", "message": f"{dim} mix shifted (JS divergence {out['drift'][f'js_{dim}']:.3f} > {t.js_divergence})"})
    if negative_feedback_rate is not None and negative_feedback_rate > t.negative_feedback_rate:
        out["alerts"].append({"signal": "negative_feedback", "message": f"{negative_feedback_rate:.0%} of recent feedback is 'not helpful'"})
    out["status"] = "alert" if out["alerts"] else "ok"
    return out


async def drift_report(repo, window_hours: int = 24, baseline_days: int = 14, t: DriftThresholds | None = None) -> dict:
    recent = summarize(await repo.requests_window(0, window_hours))
    baseline = summarize(await repo.requests_window(window_hours, window_hours + baseline_days * 24))
    report = compare(recent, baseline, await repo.negative_feedback_rate(window_hours), t)
    report["window_hours"], report["baseline_days"] = window_hours, baseline_days
    return report


def publish_gauges(report: dict) -> None:
    for k, v in report["drift"].items():
        if k.startswith("js_"):
            DRIFT_JS.labels(k[3:]).set(v)
    r = report["recent"]
    if r["abstention_rate"] is not None:
        RECENT_ABSTENTION.set(r["abstention_rate"])
    if r["mean_evidence"] is not None:
        RECENT_EVIDENCE.set(r["mean_evidence"])
    if report["drift"].get("negative_feedback_rate") is not None:
        RECENT_NEGATIVE_FEEDBACK.set(report["drift"]["negative_feedback_rate"])
