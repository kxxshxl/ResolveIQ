"""Pure clustering of recent complaints into recurring groups, with the statistics a support lead needs. No I/O: shared by the service and the evaluation."""
from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np

from app.discovery.clustering import DiscoveryParams, cluster_embeddings, keywords_per_cluster

DISCLAIMER = "Support-side recurring complaint cluster: customers describing a similar problem. Not a confirmed network incident."


def _share(c: Counter) -> tuple[str | None, float]:
    if not c:
        return None, 0.0
    name, n = c.most_common(1)[0]
    return name, round(n / sum(c.values()), 3)


def time_pattern(times: list[datetime], now: datetime, days: int) -> dict:
    """When the complaints arrived: per-day counts over the window and a plain label for the trend."""
    times = sorted(times)
    start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    daily = [0] * days
    for t in times:
        i = (t.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - start).days
        if 0 <= i < days:
            daily[i] += 1
    last_24h = sum(1 for t in times if now - t <= timedelta(hours=24))
    prior = [d for d in daily[:-1]]
    prior_mean = float(np.mean(prior)) if prior else 0.0
    first, last = times[0], times[-1]
    if now - first <= timedelta(hours=24):
        label = "new: first seen in the last 24 hours"
    elif last_24h >= 3 and last_24h > 2 * max(prior_mean, 0.5):
        label = "rising: more than twice the earlier daily rate"
    elif now - last > timedelta(days=3):
        label = "fading: nothing in the last 3 days"
    else:
        label = "steady"
    return {"first_seen": first.isoformat(), "last_seen": last.isoformat(), "daily_counts": daily, "last_24h": last_24h, "earlier_daily_mean": round(prior_mean, 2), "label": label}


def cluster_requests(rows: list[dict], X: np.ndarray, *, min_size: int = 3, distance: float = 0.55, days: int = 7, now: datetime | None = None,
                     background: list[str] | None = None, max_clusters: int = 20) -> list[dict]:
    """rows[i] describes complaint i (request_id, complaint, created_at, intent, product, severity, status, evidence, occurrences); X[i] is its unit embedding."""
    now = now or datetime.now(timezone.utc)
    groups = cluster_embeddings(X, DiscoveryParams(distance_threshold=distance, min_cluster_size=min_size))[:max_clusters]
    if not groups:
        return []
    kws = keywords_per_cluster([[rows[i]["complaint"] for i in g] for g in groups], background or [r["complaint"] for r in rows], 5)
    out = []
    for g, kw in zip(groups, kws):
        members = [rows[i] for i in g]
        V = X[g]
        centroid = V.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) or 1.0)
        sims = V @ centroid
        order = np.argsort(-sims)
        intent, intent_share = _share(Counter(m["intent"] for m in members if m.get("intent")))
        product, product_share = _share(Counter(m["product"] for m in members if m.get("product")))
        cohesion = float(sims.mean())
        cohesion_norm = max(0.0, min(1.0, (cohesion - 0.4) / 0.5))
        ev = [m["evidence"] for m in members if m.get("evidence") is not None]
        rid = hashlib.sha1("|".join(sorted(m["request_id"] for m in members)).encode()).hexdigest()[:10]
        out.append({
            "cluster_id": rid, "size": len(members), "requests": sum(m.get("occurrences", 1) for m in members), "kind": "recurring_complaint_cluster",
            "keywords": kw, "representative": {"request_id": members[order[0]]["request_id"], "complaint": members[order[0]]["complaint"]},
            "intent": intent, "intent_share": intent_share, "product": product, "product_share": product_share,
            "severity_mix": dict(Counter(m.get("severity") or "unknown" for m in members).most_common(4)),
            "time_pattern": time_pattern([m["created_at"] for m in members], now, days),
            "confidence": round(0.5 * cohesion_norm + 0.5 * intent_share, 3), "confidence_parts": {"cohesion": round(cohesion, 3), "cohesion_score": round(cohesion_norm, 3), "intent_agreement": intent_share},
            "mean_evidence": round(float(np.mean(ev)), 3) if ev else None, "abstained_share": round(sum(m.get("status") == "abstained" for m in members) / len(members), 3),
            "examples": [{"request_id": members[j]["request_id"], "complaint": members[j]["complaint"][:200], "created_at": members[j]["created_at"].isoformat()} for j in order[:3]],
            "member_request_ids": [m["request_id"] for m in members], "disclaimer": DISCLAIMER})
    out.sort(key=lambda c: (-c["requests"], -c["size"], c["cluster_id"]))
    return out
