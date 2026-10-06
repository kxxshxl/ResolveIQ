"""Recurring-cluster service: recent requests (with the embeddings they were logged with) -> clusters -> linked to discovery proposals and drift."""
from __future__ import annotations

import logging
import time

import numpy as np

from app.clusters.engine import DISCLAIMER, cluster_requests
from app.core.config import Settings
from app.db.repository import Repository
from app.drift.detect import normalise_text
from app.observability import tracing
from app.observability.metrics import CLUSTER_RUN_SECONDS
from app.services.embedding import EmbeddingService

log = logging.getLogger(__name__)
MAX_ROWS = 1500
MAX_BACKFILL = 800
CACHE_SECONDS = 60


class ClusterService:
    def __init__(self, repo: Repository, embedder: EmbeddingService, settings: Settings):
        self.repo, self.embedder, self.s = repo, embedder, settings
        self._cache: dict[tuple, tuple[float, dict]] = {}

    async def _load(self, days: int) -> tuple[list[dict], np.ndarray, dict]:
        """Distinct recent complaints with embeddings. Rows logged without one (before migration 004, or served from the response cache) are embedded now and
        the vector is written back, so the next call is a plain read."""
        rows = await self.repo.recent_requests_with_embeddings(days, MAX_ROWS)
        seen: dict[str, dict] = {}
        for r in rows:                                   # newest first; identical complaints count once but keep their multiplicity
            key = normalise_text(r["complaint"])
            if key in seen:
                seen[key]["occurrences"] += 1
            else:
                seen[key] = {**r, "occurrences": 1}
        distinct = list(seen.values())
        missing = [r for r in distinct if r["embedding"] is None or r["embedding_model"] != self.embedder.model_name][:MAX_BACKFILL]
        stats = {"requests": len(rows), "distinct": len(distinct), "embedded_now": len(missing)}
        if missing:
            vecs = await self.embedder.embed_batch([r["complaint"] for r in missing])
            for r, v in zip(missing, vecs):
                r["embedding"], r["embedding_model"] = list(map(float, v)), self.embedder.model_name
            try:
                await self.repo.put_request_embeddings([(r["request_id"], r["embedding"]) for r in missing], self.embedder.model_name)
            except Exception as exc:  # noqa: BLE001 - the write-back is an optimisation
                log.warning("could not store request embeddings", extra={"error": str(exc)[:160]})
        usable = [r for r in distinct if r["embedding"] is not None]
        X = np.asarray([r["embedding"] for r in usable], dtype=np.float32) if usable else np.zeros((0, self.s.embedding_dim), dtype=np.float32)
        if len(X):
            X /= np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-9)
        return usable, X, stats

    @tracing.traced("clusters.compute", attrs=lambda self, days=7, min_size=3, distance=0.55: {"resolveiq.window_days": days, "resolveiq.cluster.min_size": min_size, "resolveiq.cluster.distance": distance},
                    result=lambda r: {"resolveiq.cluster.count": len(r["clusters"]), "resolveiq.cluster.distinct": r["stats"]["distinct"]})
    async def recurring(self, days: int = 7, min_size: int = 3, distance: float = 0.55) -> dict:
        t0 = time.perf_counter()
        latest = await self.repo.fetch_one("SELECT count(*) AS n, max(created_at) AS t FROM resolution_requests WHERE created_at > now() - make_interval(days => %s)", (days,))
        key = (days, min_size, round(distance, 3), latest["n"], str(latest["t"]))
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
            return {**hit[1], "cached": True}
        rows, X, stats = await self._load(days)
        clusters = cluster_requests(rows, X, min_size=min_size, distance=distance, days=days) if len(rows) >= min_size else []
        snap = await self.repo.latest_drift_snapshot()
        drift = [(c["cluster_id"], set(c.get("request_ids", []))) for c in ((snap or {}).get("report") or {}).get("emerging_clusters", []) if c.get("significant")]
        for c in clusters:
            members = set(c["member_request_ids"])
            props = await self.repo.proposals_overlapping(c["member_request_ids"])
            c["proposals"] = [{"proposal_id": p["proposal_id"], "status": p["status"], "recommendation": p["recommendation"], "label_id": p["label_id"],
                               "overlap_share": round(int(p["overlap"]) / max(1, len(members)), 3)} for p in props]
            hit_d = [(cid, len(members & ids) / max(1, len(members))) for cid, ids in drift if members & ids]
            c["drift_cluster"] = max(hit_d, key=lambda x: x[1])[0] if hit_d else None
            c["discovery"] = ("proposal " + c["proposals"][0]["status"]) if c["proposals"] else "no proposal yet"
            del c["member_request_ids"]
        elapsed = time.perf_counter() - t0
        CLUSTER_RUN_SECONDS.observe(elapsed)
        out = {"window_days": days, "min_size": min_size, "distance": distance, "clusters": clusters, "stats": {**stats, "clustered_complaints": sum(c["size"] for c in clusters),
               "elapsed_ms": round(elapsed * 1000, 1)}, "disclaimer": DISCLAIMER, "cached": False}
        self._cache[key] = (time.monotonic(), out)
        if len(self._cache) > 16:
            self._cache.pop(next(iter(self._cache)))
        return out
