"""Discovery service: select poorly-explained requests, cluster them, store reviewable taxonomy proposals."""
from __future__ import annotations

import hashlib
import logging
import time
import uuid

import numpy as np

from app.observability import tracing
from app.classification.taxonomy import TaxonomyService
from app.core.config import Settings
from app.core.errors import ConflictError, NotFoundError
from app.db.repository import Repository
from app.discovery.clustering import DiscoveryParams, Neighbors, build_proposals
from app.observability.metrics import DISCOVERY_PROPOSALS, DISCOVERY_RUNS
from app.rag.evidence import assess_evidence
from app.retrieval.service import QueryContext, RetrievalService
from app.services.embedding import EmbeddingService

log = logging.getLogger(__name__)


class DiscoveryService:
    def __init__(self, repo: Repository, retrieval: RetrievalService, embedder: EmbeddingService, taxonomy: TaxonomyService,
                 settings: Settings):
        self.repo, self.retrieval, self.embedder, self.taxonomy, self.s = repo, retrieval, embedder, taxonomy, settings

    def params(self) -> DiscoveryParams:
        s = self.s
        return DiscoveryParams(distance_threshold=s.discovery_distance_threshold, min_cluster_size=s.discovery_min_cluster_size,
                               extend_agreement=s.discovery_extend_agreement, covered_similarity=s.discovery_covered_similarity)

    async def profile(self, texts: list[str]) -> tuple[np.ndarray, list[Neighbors], list[float]]:
        """Embedding, nearest resolved tickets and evidence confidence for each complaint (same signals the resolve path uses)."""
        X = await self.embedder.embed_batch(texts)
        neighbors, evidence = [], []
        for text, vec in zip(texts, X):
            ctx = QueryContext(text=text, embedding=vec)
            tickets = await self.retrieval.search(ctx, "ticket", "dense", 5)
            articles = await self.retrieval.search(ctx, "article", "dense", 3)
            top_t = tickets[0].score if tickets else 0.0
            top_a = articles[0].score if articles else 0.0
            evidence.append(assess_evidence(tickets, articles, top_t, top_a, self.s).confidence)
            neighbors.append(Neighbors([str(t.metadata.get("intent")) for t in tickets], [str(t.metadata.get("product")) for t in tickets], top_t))
        return X, neighbors, evidence

    @tracing.traced("discovery.run", attrs=lambda self, job_id=None, window_days=None: {"resolveiq.job_id": job_id, "resolveiq.discovery.window_days": window_days},
                    result=lambda r: {f"resolveiq.discovery.{k}": v for k, v in r.items() if isinstance(v, (int, float))})
    async def run(self, job_id: str | None = None, window_days: int | None = None) -> dict:
        t0 = time.perf_counter()
        window = window_days or self.s.discovery_window_days
        rows = await self.repo.discovery_candidates(window, self.s.discovery_novelty_threshold, self.s.discovery_max_candidates)
        seen: dict[str, dict] = {}
        for r in rows:  # identical complaints (retries, copy-paste) must not look like a recurring topic
            seen.setdefault(hashlib.sha256(r["complaint"].strip().lower().encode()).hexdigest(), r)
        cands = list(seen.values())
        result: dict = {"window_days": window, "candidates": len(rows), "unique_candidates": len(cands), "proposals": 0}
        if len(cands) >= self.s.discovery_min_cluster_size:
            texts = [c["complaint"] for c in cands]
            X, neighbors, evidence = await self.profile(texts)
            background = [d["text"] for d in await self.repo.all_docs("ticket")]
            labels = {k: v.team for k, v in self.taxonomy.current.labels["intent"].items()}
            props = build_proposals(texts, X, neighbors, evidence, background, self.params(), labels,
                                    request_ids=[str(c["request_id"]) for c in cands])
            await self.repo.replace_pending_proposals(props, job_id)
            for p in props:
                DISCOVERY_PROPOSALS.labels(p["recommendation"]).inc()
            result["proposals"] = len(props)
            result["by_recommendation"] = {k: sum(p["recommendation"] == k for p in props) for k in ("new_class", "extend_existing")}
        DISCOVERY_RUNS.inc()
        result["elapsed_s"] = round(time.perf_counter() - t0, 2)
        log.info("discovery finished", extra=result)
        return result

    # ------------------------------------------------------------------ review actions
    async def accept(self, proposal_id: str, label_id: str | None = None, description: str | None = None,
                     team: str | None = None, merge_into: str | None = None, note: str | None = None) -> dict:
        prop = await self.repo.get_proposal(proposal_id)
        if not prop:
            raise NotFoundError("proposal not found")
        if prop["status"] != "pending":
            raise ConflictError(f"proposal is already {prop['status']}")
        tax = self.taxonomy.current
        target = merge_into or (prop["nearest_intent"] if prop["recommendation"] == "extend_existing" and not label_id else None)
        if target:
            if not tax.has("intent", target):
                raise NotFoundError(f"intent '{target}' not found")
            await self.repo.extend_taxonomy_label("intent", target, prop["keywords"], prop["examples"], note=f"extend {target} from proposal {proposal_id}")
            action = {"action": "extended", "label_id": target}
        else:
            new_id = label_id or prop["label_id"]
            if tax.has("intent", new_id):
                raise ConflictError(f"intent '{new_id}' already exists; pass merge_into to extend it")
            await self.taxonomy.add_label("intent", new_id, description or prop["description"], prop["keywords"], prop["examples"],
                                          team or prop["team"])
            action = {"action": "created", "label_id": new_id}
        await self.taxonomy.refresh()
        outcome = f"{action['action']} intent {action['label_id']}"  # audit trail: what accepting actually did
        await self.repo.decide_proposal(proposal_id, "accepted", f"{outcome} - {note}" if note else outcome)
        await self.repo.bump_corpus_version()  # classification changed -> invalidate cached responses
        return {**action, "taxonomy_version": self.taxonomy.current.version, "members": len(prop["member_request_ids"])}

    async def reject(self, proposal_id: str, note: str | None = None) -> dict:
        prop = await self.repo.get_proposal(proposal_id)
        if not prop:
            raise NotFoundError("proposal not found")
        if prop["status"] != "pending":
            raise ConflictError(f"proposal is already {prop['status']}")
        await self.repo.decide_proposal(proposal_id, "rejected", note)
        return {"proposal_id": proposal_id, "status": "rejected"}


def new_id() -> str:
    return str(uuid.uuid4())
