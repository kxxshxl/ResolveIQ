"""Drift service: load the two windows, embed a bounded sample, run the detector, link new-topic clusters to discovery proposals,
persist the result, publish gauges. Runs in the worker (job `drift_analysis`) so embedding never competes with /resolve."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone

from app.core.config import Settings
from app.db.repository import Repository
from app.discovery.clustering import DiscoveryParams
from app.drift.detect import DriftConfig, Window, analyze, sample_for_embedding
from app.observability import tracing
from app.observability.metrics import DRIFT_ALERT, DRIFT_ANALYSES, DRIFT_EMERGING, DRIFT_LAST_RUN, DRIFT_UNSEEN
from app.services.embedding import EmbeddingService

log = logging.getLogger(__name__)
_fired: set[str] = set()   # signals currently exported as 1, so a signal that clears is reset to 0
OVERLAP_COVERED = 0.5      # a proposal containing at least this share of a cluster's complaints counts as covering it


class DriftService:
    def __init__(self, repo: Repository, embedder: EmbeddingService, settings: Settings):
        self.repo, self.embedder, self.s = repo, embedder, settings

    def config(self) -> DriftConfig:
        s = self.s
        return DriftConfig(
            min_requests=s.drift_min_requests, alpha=s.drift_alpha, psi_threshold=s.drift_psi_threshold, ks_threshold=s.drift_ks_threshold,
            abstention_increase=s.drift_abstention_increase, embedding_shift=s.drift_embedding_shift, unseen_quantile=s.drift_unseen_quantile,
            permutations=s.drift_permutations, novelty_threshold=s.discovery_novelty_threshold,
            # the clustering routine discovery uses (same minimum size); the distance is looser because here in-domain complaints are clustered too
            cluster=DiscoveryParams(distance_threshold=s.drift_cluster_distance, min_cluster_size=s.discovery_min_cluster_size))

    async def _window(self, newer_h: float, older_h: float, cap: int) -> Window:
        rows = await self.repo.requests_window_detail(newer_h, older_h)
        return Window(rows=rows, emb_rows=sample_for_embedding(rows, cap))

    @tracing.traced("drift.analyze", attrs=lambda self, window_hours=None, baseline_days=None, job_id=None: {"resolveiq.job_id": job_id},
                    result=lambda r: {"resolveiq.drift.alerts": r.get("alerts", 0), "resolveiq.drift.emerging_clusters": r.get("emerging_clusters", 0)})
    async def run(self, window_hours: int | None = None, baseline_days: int | None = None, job_id: str | None = None) -> dict:
        """Analyse, persist and return a compact summary (the full report is in drift_snapshots and served by the API)."""
        t0 = time.perf_counter()
        wh, bd = window_hours or self.s.drift_window_hours, baseline_days or self.s.drift_baseline_days
        recent = await self._window(0, wh, self.s.drift_max_embed_recent)
        baseline = await self._window(wh, wh + bd * 24, self.s.drift_max_embed_baseline)
        cfg = self.config()
        if len(recent.emb_rows) >= cfg.min_embedded and len(baseline.emb_rows) >= cfg.min_embedded:
            X = await self.embedder.embed_batch([r["complaint"] for r in recent.emb_rows] + [r["complaint"] for r in baseline.emb_rows])
            recent.X, baseline.X = X[: len(recent.emb_rows)], X[len(recent.emb_rows):]
        report = await asyncio.to_thread(analyze, recent, baseline, cfg)   # CPU-bound (permutation + resampling tests)
        report.update({"window_hours": wh, "baseline_days": bd, "generated_at": datetime.now(timezone.utc).isoformat(), "embedding_model": self.embedder.model_name})
        await self._link_proposals(report)
        report["discovery"] = await self._discovery_decision(report)
        snapshot_id = str(uuid.uuid4())
        report["snapshot_id"] = snapshot_id
        await self.repo.save_drift_snapshot(snapshot_id, wh, bd, report, job_id)
        publish(report)
        summary = {"snapshot_id": snapshot_id, "status": report["status"], "alerts": len(report["alerts"]),
                   "emerging_clusters": sum(c["significant"] for c in report["emerging_clusters"]), "n_recent": report["n_recent"],
                   "n_baseline": report["n_baseline"], "discovery": report["discovery"], "elapsed_s": round(time.perf_counter() - t0, 2)}
        log.info("drift analysis finished", extra={k: v for k, v in summary.items() if k != "discovery"})
        return summary

    async def _link_proposals(self, report: dict) -> None:
        """Attach, to every new-topic cluster, the discovery proposals that contain some of its complaints (live review state)."""
        for c in report["emerging_clusters"]:
            ids = c.get("request_ids", [])
            c["related_proposals"] = [{**p, "overlap_share": round(int(p["overlap"]) / max(1, len(ids)), 4)} for p in await self.repo.proposals_overlapping(ids)]

    def _needs_proposal(self, c: dict) -> bool:
        reviewed = any(p["overlap_share"] >= OVERLAP_COVERED for p in c.get("related_proposals", []))
        return bool(c["significant"] and not c["covered_by_corpus"] and not reviewed)

    async def _discovery_decision(self, report: dict) -> dict:
        todo = [c["cluster_id"] for c in report["emerging_clusters"] if self._needs_proposal(c)]
        if not todo:
            return {"recommended": False, "triggered": False, "reason": "no significant new-topic cluster lacks a discovery proposal"}
        out = {"recommended": True, "triggered": False, "clusters": todo}
        if not self.s.drift_trigger_discovery:
            return out | {"reason": "automatic discovery is off (DRIFT_TRIGGER_DISCOVERY=false); run it from the Class discovery tab"}
        if self.s.job_execution != "queue":
            return out | {"reason": "no worker in this deployment (JOB_EXECUTION=inline); run discovery from the Class discovery tab"}
        if await self.repo.active_job_exists("discover_classes"):
            return out | {"reason": "a discovery run is already queued or running"}
        last = await self.repo.last_job_time("discover_classes")
        if last is not None and (datetime.now(timezone.utc) - last).total_seconds() < self.s.drift_discovery_cooldown_hours * 3600:
            return out | {"reason": f"discovery ran less than {self.s.drift_discovery_cooldown_hours:g} h ago"}
        job_id = await self.repo.create_job("discover_classes", {"triggered_by": "drift"})
        log.info("drift enqueued a discovery run", extra={"job_id": job_id, "clusters": todo})
        return out | {"triggered": True, "job_id": job_id, "reason": "an unexplained new-topic cluster has no proposal yet; its proposal will need human review"}

    async def latest(self) -> dict | None:
        """The newest stored analysis, with each cluster's related proposals re-read so their review state is current."""
        snap = await self.repo.latest_drift_snapshot()
        if not snap:
            return None
        report = snap["report"]
        await self._link_proposals(report)
        for c in report["emerging_clusters"]:
            c.pop("request_ids", None)
        report["snapshot_id"], report["created_at"] = snap["snapshot_id"], snap["created_at"]
        age = (datetime.now(timezone.utc) - snap["created_at"]).total_seconds()
        report["age_seconds"] = int(age)
        return report


def publish(report: dict) -> None:
    now = {a["signal"] for a in report["alerts"]}
    for sig in now | _fired:
        DRIFT_ALERT.labels(sig).set(1 if sig in now else 0)
    _fired.clear()
    _fired.update(now)
    DRIFT_EMERGING.set(sum(c["significant"] for c in report["emerging_clusters"]))
    if report["embedding"].get("status") == "ok":
        DRIFT_UNSEEN.set(report["embedding"]["unseen_rate"])
    DRIFT_ANALYSES.labels(report["status"]).inc()
    DRIFT_LAST_RUN.set(time.time())
