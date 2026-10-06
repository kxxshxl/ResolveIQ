"""Ingestion: clean -> PII redact -> label -> embed -> upsert. One path for seed data, API calls and batch jobs.

New resolved tickets are searchable the moment the transaction commits (no index rebuild, no restart):
HNSW and GIN indexes are maintained incrementally by Postgres.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from app.observability import tracing
from app.classification.pipeline import ComplaintClassifier
from app.classification.taxonomy import TaxonomyService
from app.core.config import Settings
from app.core.errors import ValidationFailure
from app.core.pii import redact
from app.core.text import looks_like_injection, normalize_text
from app.db.repository import Repository
from app.models.schemas import ArticleIn, IngestResult, TicketIn
from app.observability.metrics import CORPUS_SIZE, INGESTED, PII_REDACTIONS
from app.retrieval.service import QueryContext
from app.services.embedding import EmbeddingService

log = logging.getLogger(__name__)


def _clean_pii(text: str, counts: dict[str, int]) -> str:
    r = redact(normalize_text(text))
    for k, v in r.counts.items():
        counts[k] = counts.get(k, 0) + v
        PII_REDACTIONS.labels(k).inc(v)
    return r.text


def _security_warnings(metadata: dict) -> list[str]:
    if "instruction_like" in (metadata or {}).get("security_flags", []):
        return ["the document contains instruction-like text; it is stored, but sentences like that are removed before any model sees the document as evidence"]
    return []


class IngestionService:
    def __init__(self, repo: Repository, embedder: EmbeddingService, taxonomy: TaxonomyService,
                 classifier: ComplaintClassifier, settings: Settings):
        self.repo, self.embedder, self.taxonomy, self.classifier, self.s = repo, embedder, taxonomy, classifier, settings

    async def refresh_gauges(self) -> None:
        c = await self.repo.counts()
        CORPUS_SIZE.labels("ticket").set(c["tickets"])
        CORPUS_SIZE.labels("article").set(c["articles"])

    # ------------------------------------------------------------------ normalisation
    def _check_label(self, dim: str, value: str) -> None:
        if not self.taxonomy.current.has(dim, value):
            raise ValidationFailure(
                f"unknown {dim} '{value}' (taxonomy v{self.taxonomy.current.version}). "
                f"Register it first with POST /api/v1/taxonomy/labels.")

    async def prepare_ticket(self, t: TicketIn, embedding=None, source: str = "api") -> tuple[dict, dict[str, int], object]:
        counts: dict[str, int] = {}
        text = _clean_pii(t.complaint_text, counts)
        vec = embedding if embedding is not None else (await self.embedder.embed_batch([text]))[0]
        labels = {k: getattr(t, k) for k in ("intent", "product", "severity", "sentiment")}
        if not all(labels.values()):  # self-label from the existing corpus rather than demanding every field
            ctx = QueryContext(text=text, embedding=vec)
            pred = await self.classifier.classify(ctx)
            for k, v in labels.items():
                labels[k] = v or getattr(pred, k)
        for dim, v in labels.items():
            self._check_label(dim, v)
        flagged = looks_like_injection(" ".join([text, t.resolution_summary, *t.resolution_steps]))
        row = {
            "ticket_id": t.ticket_id or f"TKT-{uuid.uuid4().hex[:8].upper()}",
            "complaint_text": text, **labels,
            "resolution_steps": [_clean_pii(s, counts) for s in t.resolution_steps],
            "resolution_summary": _clean_pii(t.resolution_summary, counts),
            "resolved_at": t.resolved_at or datetime.now(timezone.utc),
            "metadata": {**t.metadata, **({"security_flags": ["instruction_like"]} if flagged else {})}, "taxonomy_version": self.taxonomy.current.version, "source": source,
        }
        return row, counts, vec

    # ------------------------------------------------------------------ single documents
    @tracing.traced("ingest.ticket", attrs=lambda self, t, source="api": {"resolveiq.ingest.source": source},
                    result=lambda r: {"resolveiq.ingest.created": r.created, "resolveiq.ingest.pii_redactions": sum(r.pii_redactions.values()),
                                      "resolveiq.corpus_version": r.corpus_version, "resolveiq.embedding_ms": r.embedding_ms})
    async def ingest_ticket(self, t: TicketIn, source: str = "api") -> IngestResult:
        t0 = time.perf_counter()
        row, counts, vec = await self.prepare_ticket(t, source=source)
        _, created = await self.repo.upsert_ticket(row, vec, self.embedder.model_name)
        version = await self.repo.bump_corpus_version()
        INGESTED.labels("ticket", "created" if created else "updated").inc()
        await self.refresh_gauges()
        return IngestResult(source_type="ticket", source_id=row["ticket_id"], created=created, pii_redactions=counts,
                            corpus_version=version, embedding_ms=round((time.perf_counter() - t0) * 1000, 1), warnings=_security_warnings(row["metadata"]))

    @tracing.traced("ingest.article", result=lambda r: {"resolveiq.ingest.created": r.created, "resolveiq.corpus_version": r.corpus_version,
                                                         "resolveiq.embedding_ms": r.embedding_ms})
    async def ingest_article(self, a: ArticleIn) -> IngestResult:
        t0 = time.perf_counter()
        self._check_label("intent", a.category)
        self._check_label("product", a.product)
        counts: dict[str, int] = {}
        content = _clean_pii(a.content, counts)
        title = normalize_text(a.title)
        vec = (await self.embedder.embed_batch([f"{title}. {content}"]))[0]
        steps = [_clean_pii(s, counts) for s in a.steps]
        flagged = looks_like_injection(" ".join([title, content, *steps]))
        row = {"article_id": a.article_id or f"KB-{uuid.uuid4().hex[:6].upper()}", "title": title, "content": content,
               "steps": steps, "category": a.category, "product": a.product,
               "tags": a.tags, "metadata": {**a.metadata, **({"security_flags": ["instruction_like"]} if flagged else {})}}
        _, created = await self.repo.upsert_article(row, vec, self.embedder.model_name)
        version = await self.repo.bump_corpus_version()
        INGESTED.labels("article", "created" if created else "updated").inc()
        await self.refresh_gauges()
        return IngestResult(source_type="article", source_id=row["article_id"], created=created, pii_redactions=counts,
                            corpus_version=version, embedding_ms=round((time.perf_counter() - t0) * 1000, 1), warnings=_security_warnings(row["metadata"]))

    # ------------------------------------------------------------------ bulk
    @tracing.traced("ingest.batch", attrs=lambda self, tickets, source="batch", job_id=None: {"resolveiq.ingest.source": source,
                                                                                         "resolveiq.batch_size": len(tickets)},
                    result=lambda r: {"resolveiq.ingest.created": r["created"], "resolveiq.ingest.updated": r["updated"],
                                      "resolveiq.ingest.failed": r["failed"], "resolveiq.corpus_version": r["corpus_version"]})
    async def ingest_tickets_bulk(self, tickets: list[TicketIn], source: str = "batch", job_id: str | None = None) -> dict:
        """Batch path: ONE embedding call for the whole batch (GPU/CPU friendly), then per-row upserts."""
        texts = [normalize_text(redact(normalize_text(t.complaint_text)).text) for t in tickets]
        vecs = await self.embedder.embed_batch(texts)
        created = updated = failed = 0
        errors: list[str] = []
        prepared: list[tuple[dict, object]] = []
        for t, v in zip(tickets, vecs):
            try:
                row, _, vec = await self.prepare_ticket(t, embedding=v, source=source)
                prepared.append((row, vec))
            except Exception as exc:  # noqa: BLE001 - one bad row must not abort the batch
                failed += 1
                errors.append(f"{t.ticket_id or t.complaint_text[:30]}: {exc}")
        # the fast path writes the whole batch in one transaction; if anything in it is rejected, fall back to row by row so one bad row costs only itself
        by_id = {r["ticket_id"]: (r, v) for r, v in prepared}      # duplicate ids inside one batch: the last one wins, as sequential upserts would
        unique = list(by_id.values())
        try:
            c, u = await self.repo.bulk_upsert_tickets([r for r, _ in unique], [v for _, v in unique], self.embedder.model_name)
            created, updated = c, u + (len(prepared) - len(unique))
        except Exception as exc:  # noqa: BLE001
            log.warning("batched upsert failed, retrying row by row", extra={"error": str(exc)[:200]})
            for row, vec in prepared:
                try:
                    _, was_new = await self.repo.upsert_ticket(row, vec, self.embedder.model_name)
                    created += was_new
                    updated += not was_new
                except Exception as exc2:  # noqa: BLE001
                    failed += 1
                    errors.append(f"{row['ticket_id']}: {exc2}")
        version = await self.repo.bump_corpus_version()
        INGESTED.labels("ticket", "created").inc(created)
        INGESTED.labels("ticket", "updated").inc(updated)
        INGESTED.labels("ticket", "failed").inc(failed)
        await self.refresh_gauges()
        result = {"created": created, "updated": updated, "failed": failed, "errors": errors[:20], "corpus_version": version}
        if job_id:
            await self.repo.update_job(job_id, "succeeded", result=result)
        return result

    @tracing.traced("ingest.reindex", result=lambda r: {f"resolveiq.reindex.{k}": v for k, v in r.items()})
    async def reindex(self, batch_size: int = 256) -> dict[str, int]:
        """Backfill embeddings for the CONFIGURED model (e.g. after switching EMBEDDING_MODEL). Idempotent,
        resumable (only rows lacking an embedding for this model are touched) and safe to run online:
        queries filter on model name, so the old model keeps serving until the switch."""
        done = {}
        for kind in ("ticket", "article"):
            rows = await self.repo.ids_missing_embedding(kind, self.embedder.model_name)
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i + batch_size]
                vecs = await self.embedder.embed_batch([r["text"] for r in chunk])
                await self.repo.put_embeddings(kind, [(r["id"], v) for r, v in zip(chunk, vecs)], self.embedder.model_name)
            done[kind] = len(rows)
        if any(done.values()):
            await self.repo.bump_corpus_version()
        return done
