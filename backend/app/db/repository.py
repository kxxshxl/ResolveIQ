"""PostgreSQL access: pooled async connections, vector/lexical search, CRUD.

All SQL lives here so the retrieval/ingestion layers stay storage-agnostic.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.core.config import Settings
from app.observability import tracing

log = logging.getLogger(__name__)

# source_type -> (doc table, embedding table, fk column, id column, text expression)
KINDS: dict[str, dict[str, str]] = {
    "ticket": {"table": "tickets", "emb": "ticket_embeddings", "fk": "ticket_pk", "id": "ticket_id",
               "status_ok": "d.status = 'active'", "intent": "intent", "title": "resolution_summary"},
    "article": {"table": "knowledge_articles", "emb": "kb_embeddings", "fk": "article_pk", "id": "article_id",
                "status_ok": "d.status = 'active'", "intent": "category", "title": "title"},
}


def vec_literal(v: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


class Repository:
    def __init__(self, settings: Settings):
        self.s = settings
        self.pool: AsyncConnectionPool | None = None

    # ------------------------------------------------------------- lifecycle
    async def open(self) -> None:
        async def configure(conn):
            await conn.set_autocommit(True)
            try:
                await conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")  # pgvector >= 0.8
            except Exception:  # noqa: BLE001 - older pgvector
                pass

        self.pool = AsyncConnectionPool(
            self.s.database_url,
            min_size=self.s.db_pool_min,
            max_size=self.s.db_pool_max,
            kwargs={"row_factory": dict_row, "options": f"-c statement_timeout={self.s.db_statement_timeout_ms}"},
            configure=configure,
            open=False,
            timeout=5,
        )
        await self.pool.open(wait=True, timeout=10)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def ping(self) -> bool:
        async with self.pool.connection() as conn:
            await conn.execute("SELECT 1")
        return True

    async def _fetch(self, sql: str, params: Any = None) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchall()

    async def _exec(self, sql: str, params: Any = None) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(sql, params)

    # ------------------------------------------------------------- meta
    async def corpus_version(self) -> int:
        rows = await self._fetch("SELECT value FROM meta WHERE key='corpus_version'")
        return int(rows[0]["value"]) if rows else 1

    async def bump_corpus_version(self) -> int:
        rows = await self._fetch(
            "UPDATE meta SET value = (value::int + 1)::text WHERE key='corpus_version' RETURNING value"
        )
        return int(rows[0]["value"])

    async def counts(self) -> dict[str, int]:
        rows = await self._fetch(
            """SELECT (SELECT count(*) FROM tickets WHERE status='active') AS tickets,
                      (SELECT count(*) FROM knowledge_articles WHERE status='active') AS articles,
                      (SELECT count(*) FROM knowledge_articles WHERE status='deprecated') AS deprecated_articles,
                      (SELECT count(*) FROM ticket_embeddings WHERE model=%s) AS ticket_embeddings,
                      (SELECT count(*) FROM kb_embeddings WHERE model=%s) AS kb_embeddings""",
            (self.s.embedding_model, self.s.embedding_model),
        )
        return {k: int(v) for k, v in rows[0].items()}

    # ------------------------------------------------------------- taxonomy
    async def taxonomy_version(self) -> int:
        rows = await self._fetch("SELECT coalesce(max(version),0) AS v FROM taxonomy_versions")
        return int(rows[0]["v"])

    async def load_taxonomy(self) -> list[dict]:
        return await self._fetch(
            "SELECT dimension,label_id,description,keywords,examples,rank,team,status,introduced_in "
            "FROM taxonomy_labels WHERE status='active' ORDER BY dimension, rank NULLS LAST, label_id"
        )

    async def add_taxonomy_labels(self, labels: list[dict], note: str) -> int:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute("SELECT coalesce(max(version),0)+1 AS v FROM taxonomy_versions")
                version = (await cur.fetchone())["v"]
                await conn.execute("INSERT INTO taxonomy_versions (version, note) VALUES (%s,%s)", (version, note))
                for lab in labels:
                    await conn.execute(
                        """INSERT INTO taxonomy_labels (dimension,label_id,description,keywords,examples,rank,team,introduced_in)
                           VALUES (%(dimension)s,%(label_id)s,%(description)s,%(keywords)s,%(examples)s,%(rank)s,%(team)s,%(v)s)
                           ON CONFLICT (dimension,label_id) DO UPDATE SET description=EXCLUDED.description,
                             keywords=EXCLUDED.keywords, examples=EXCLUDED.examples, team=EXCLUDED.team, status='active'""",
                        {**lab, "rank": lab.get("rank"), "team": lab.get("team"), "v": version},
                    )
        return version

    async def delete_taxonomy_labels(self, dimension: str, label_ids: Sequence[str]) -> None:
        await self._exec("DELETE FROM taxonomy_labels WHERE dimension=%s AND label_id = ANY(%s)", (dimension, list(label_ids)))

    # ------------------------------------------------------------- upserts
    @tracing.traced("db.upsert_ticket")
    async def upsert_ticket(self, t: dict, embedding: Sequence[float], model: str) -> tuple[int, bool]:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    """INSERT INTO tickets (ticket_id,complaint_text,intent,product,severity,sentiment,resolution_steps,
                         resolution_summary,resolved_at,metadata,taxonomy_version,source)
                       VALUES (%(ticket_id)s,%(complaint_text)s,%(intent)s,%(product)s,%(severity)s,%(sentiment)s,
                         %(steps)s,%(resolution_summary)s,%(resolved_at)s,%(metadata)s,%(taxonomy_version)s,%(source)s)
                       ON CONFLICT (ticket_id) DO UPDATE SET complaint_text=EXCLUDED.complaint_text, intent=EXCLUDED.intent,
                         product=EXCLUDED.product, severity=EXCLUDED.severity, sentiment=EXCLUDED.sentiment,
                         resolution_steps=EXCLUDED.resolution_steps, resolution_summary=EXCLUDED.resolution_summary,
                         resolved_at=EXCLUDED.resolved_at, metadata=EXCLUDED.metadata, status='active'
                       RETURNING id, (xmax = 0) AS created""",
                    {**t, "steps": Jsonb(t["resolution_steps"]), "metadata": Jsonb(t.get("metadata", {}))},
                )
                row = await cur.fetchone()
                await conn.execute(
                    """INSERT INTO ticket_embeddings (ticket_pk, model, embedding) VALUES (%s,%s,%s::vector)
                       ON CONFLICT (ticket_pk, model) DO UPDATE SET embedding = EXCLUDED.embedding""",
                    (row["id"], model, vec_literal(embedding)),
                )
        return row["id"], row["created"]

    @tracing.traced("db.upsert_article")
    async def upsert_article(self, a: dict, embedding: Sequence[float], model: str) -> tuple[int, bool]:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    """INSERT INTO knowledge_articles (article_id,title,content,steps,category,product,tags,metadata,updated_at)
                       VALUES (%(article_id)s,%(title)s,%(content)s,%(steps)s,%(category)s,%(product)s,%(tags)s,%(metadata)s,now())
                       ON CONFLICT (article_id) DO UPDATE SET title=EXCLUDED.title, content=EXCLUDED.content,
                         steps=EXCLUDED.steps, category=EXCLUDED.category, product=EXCLUDED.product, tags=EXCLUDED.tags,
                         metadata=EXCLUDED.metadata, updated_at=now(), status='active', deprecated_reason=NULL
                       RETURNING id, (xmax = 0) AS created""",
                    {**a, "steps": Jsonb(a.get("steps", [])), "metadata": Jsonb(a.get("metadata", {}))},
                )
                row = await cur.fetchone()
                await conn.execute(
                    """INSERT INTO kb_embeddings (article_pk, model, embedding) VALUES (%s,%s,%s::vector)
                       ON CONFLICT (article_pk, model) DO UPDATE SET embedding = EXCLUDED.embedding""",
                    (row["id"], model, vec_literal(embedding)),
                )
        return row["id"], row["created"]

    async def bulk_upsert(self, kind: str, rows: list[dict], embeddings: Sequence[Sequence[float]], model: str) -> int:
        n = 0
        for r, e in zip(rows, embeddings):
            await (self.upsert_ticket if kind == "ticket" else self.upsert_article)(r, e, model)
            n += 1
        return n

    async def deprecate_article(self, article_id: str, reason: str) -> bool:
        rows = await self._fetch(
            "UPDATE knowledge_articles SET status='deprecated', deprecated_reason=%s, updated_at=now() "
            "WHERE article_id=%s RETURNING id", (reason, article_id))
        return bool(rows)

    async def delete_by_ids(self, kind: str, ids: Iterable[str]) -> int:
        k = KINDS[kind]
        rows = await self._fetch(f"DELETE FROM {k['table']} WHERE {k['id']} = ANY(%s) RETURNING 1", (list(ids),))
        return len(rows)

    # ------------------------------------------------------------- search
    def _filters(self, kind: str, filters: dict | None, params: dict) -> str:
        k = KINDS[kind]
        clauses = [k["status_ok"]]
        if filters:
            if filters.get("intent"):
                clauses.append(f"d.{k['intent']} = %(f_intent)s")
                params["f_intent"] = filters["intent"]
            if filters.get("product"):
                clauses.append("d.product = %(f_product)s")
                params["f_product"] = filters["product"]
        return " AND ".join(clauses)

    def _select_cols(self, kind: str) -> str:
        if kind == "ticket":
            return ("d.ticket_id AS source_id, d.resolution_summary AS title, d.complaint_text AS text, d.intent, d.product, "
                    "d.severity, d.sentiment, d.resolution_steps AS steps, d.resolution_summary, d.resolved_at, d.metadata")
        return ("d.article_id AS source_id, d.title AS title, d.content AS text, d.category AS intent, d.product, "
                "d.steps AS steps, NULL::text AS resolution_summary, d.updated_at, d.tags, d.metadata")

    @tracing.traced("db.dense_search", attrs=lambda self, kind, query_vec, k, model, filters=None: {"resolveiq.retrieval.kind": kind, "resolveiq.retrieval.k": k,
                                                                                                   "resolveiq.embed.model": model},
                    result=lambda rows: {"resolveiq.db.rows": len(rows)})
    async def dense_search(self, kind: str, query_vec: Sequence[float], k: int, model: str,
                           filters: dict | None = None) -> list[dict]:
        meta = KINDS[kind]
        params: dict[str, Any] = {"q": vec_literal(query_vec), "model": model, "k": k}
        where = self._filters(kind, filters, params)
        sql = f"""SELECT {self._select_cols(kind)}, 1 - (e.embedding <=> %(q)s::vector) AS score
                  FROM {meta['emb']} e JOIN {meta['table']} d ON d.id = e.{meta['fk']}
                  WHERE e.model = %(model)s AND {where}
                  ORDER BY e.embedding <=> %(q)s::vector LIMIT %(k)s"""
        return await self._fetch(sql, params)

    async def query_lexemes(self, text: str) -> list[str]:
        rows = await self._fetch("SELECT lexeme FROM unnest(to_tsvector('english', %s))", (text,))
        return [r["lexeme"] for r in rows]

    async def doc_frequencies(self, kind: str) -> tuple[int, dict[str, int]]:
        """(n_docs, lexeme -> number of docs containing it), from Postgres' own tsvector statistics."""
        meta = KINDS[kind]
        n = (await self._fetch(f"SELECT count(*) AS n FROM {meta['table']} d WHERE d.status IN ('active')"))[0]["n"]
        rows = await self._fetch(
            f"SELECT word, ndoc FROM ts_stat($$SELECT search_text FROM {meta['table']} WHERE status = 'active'$$)")
        return int(n), {r["word"]: int(r["ndoc"]) for r in rows}

    @tracing.traced("db.lexical_search", attrs=lambda self, kind, lexemes, k, filters=None: {"resolveiq.retrieval.kind": kind, "resolveiq.retrieval.k": k,
                                                                                          "resolveiq.lexical.terms": len(lexemes)},
                    result=lambda rows: {"resolveiq.db.rows": len(rows)})
    async def lexical_search(self, kind: str, lexemes: list[str], k: int, filters: dict | None = None) -> list[dict]:
        """OR-semantics full-text search over (already stemmed, IDF-pruned) lexemes, ranked by ts_rank_cd."""
        meta = KINDS[kind]
        if not lexemes:
            return []
        params: dict[str, Any] = {"q": " | ".join(f"'{l.replace(chr(39), '')}'" for l in lexemes), "k": k}
        where = self._filters(kind, filters, params)
        sql = f"""WITH tsq AS (SELECT to_tsquery('simple', %(q)s) AS t)
                  SELECT {self._select_cols(kind)}, ts_rank_cd(d.search_text, tsq.t, 1) AS score
                  FROM {meta['table']} d, tsq
                  WHERE d.search_text @@ tsq.t AND {where}
                  ORDER BY score DESC LIMIT %(k)s"""
        return await self._fetch(sql, params)

    async def all_docs(self, kind: str) -> list[dict]:
        meta = KINDS[kind]
        params: dict[str, Any] = {}
        where = self._filters(kind, None, params)
        return await self._fetch(f"SELECT {self._select_cols(kind)} FROM {meta['table']} d WHERE {where}")

    async def get_docs(self, kind: str, ids: Sequence[str]) -> list[dict]:
        meta = KINDS[kind]
        return await self._fetch(
            f"SELECT {self._select_cols(kind)} FROM {meta['table']} d WHERE d.{meta['id']} = ANY(%s)", (list(ids),))

    async def list_docs(self, kind: str, limit: int, offset: int, intent: str | None = None) -> tuple[list[dict], int]:
        meta = KINDS[kind]
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        clause = ""
        if intent:
            clause = f"WHERE d.{meta['intent']} = %(intent)s"
            params["intent"] = intent
        order = "d.resolved_at DESC" if kind == "ticket" else "d.updated_at DESC"
        rows = await self._fetch(
            f"SELECT {self._select_cols(kind)}, d.status FROM {meta['table']} d {clause} ORDER BY {order} LIMIT %(limit)s OFFSET %(offset)s",
            params)
        total = (await self._fetch(f"SELECT count(*) AS n FROM {meta['table']} d {clause}", params))[0]["n"]
        return rows, int(total)

    async def ids_missing_embedding(self, kind: str, model: str) -> list[dict]:
        meta = KINDS[kind]
        text = "d.complaint_text" if kind == "ticket" else "d.title || '. ' || d.content"
        return await self._fetch(
            f"""SELECT d.id, {text} AS text, d.{meta['id']} AS source_id FROM {meta['table']} d
                WHERE NOT EXISTS (SELECT 1 FROM {meta['emb']} e WHERE e.{meta['fk']} = d.id AND e.model = %s)""", (model,))

    async def put_embeddings(self, kind: str, pairs: list[tuple[int, Sequence[float]]], model: str) -> None:
        meta = KINDS[kind]
        async with self.pool.connection() as conn:
            async with conn.transaction():
                for pk, emb in pairs:
                    await conn.execute(
                        f"INSERT INTO {meta['emb']} ({meta['fk']}, model, embedding) VALUES (%s,%s,%s::vector) "
                        f"ON CONFLICT ({meta['fk']}, model) DO UPDATE SET embedding = EXCLUDED.embedding",
                        (pk, model, vec_literal(emb)))

    # ------------------------------------------------------------- requests / feedback / jobs / evals
    @tracing.traced("db.save_request")
    async def save_request(self, request_id: str, trace_id: str, complaint: str, classification: dict,
                           retrieved: list[dict], result: dict, status: str, confidence: float, latency_ms: int) -> None:
        await self._exec(
            """INSERT INTO resolution_requests (request_id,trace_id,complaint,classification,retrieved,result,status,confidence,latency_ms)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (request_id, trace_id, complaint, Jsonb(classification), Jsonb(retrieved), Jsonb(result), status,
             confidence, latency_ms))

    async def request_exists(self, request_id: str) -> bool:
        return bool(await self._fetch("SELECT 1 FROM resolution_requests WHERE request_id=%s", (request_id,)))

    async def save_feedback(self, request_id: str, rating: str, comment: str | None, corrected_intent: str | None) -> int:
        rows = await self._fetch(
            "INSERT INTO feedback (request_id,rating,comment,corrected_intent) VALUES (%s,%s,%s,%s) RETURNING feedback_id",
            (request_id, rating, comment, corrected_intent))
        return rows[0]["feedback_id"]

    async def create_job(self, kind: str, payload: dict | None = None) -> str:
        job_id = str(uuid.uuid4())
        await self._exec("INSERT INTO jobs (job_id,kind,status,payload) VALUES (%s,%s,'queued',%s)",
                         (job_id, kind, Jsonb(payload or {})))
        return job_id

    async def update_job(self, job_id: str, status: str, result: dict | None = None, error: str | None = None) -> None:
        finished = datetime.now(timezone.utc) if status in ("succeeded", "failed") else None
        await self._exec("UPDATE jobs SET status=%s, result=%s, error=%s, finished_at=%s WHERE job_id=%s",
                         (status, Jsonb(result) if result is not None else None, error, finished, job_id))

    async def get_job(self, job_id: str) -> dict | None:
        rows = await self._fetch("SELECT job_id::text, kind, status, attempts, max_attempts, result, error, created_at, started_at, finished_at "
                                 "FROM jobs WHERE job_id=%s",
                                 (job_id,))
        return rows[0] if rows else None

    async def save_eval_run(self, run_id: str, suites: list[str], results: dict) -> None:
        await self._exec("INSERT INTO evaluation_runs (run_id,suites,results) VALUES (%s,%s,%s)",
                         (run_id, suites, Jsonb(json.loads(json.dumps(results, default=str)))))

    # ------------------------------------------------------------- emerging-class discovery
    async def discovery_candidates(self, window_days: int, novelty_threshold: float, limit: int) -> list[dict]:
        """Recent requests the corpus explains poorly (abstained or low evidence), not already reviewed in a proposal."""
        return await self._fetch(
            """SELECT r.request_id::text AS request_id, r.complaint, r.status, (r.result->'evidence'->>'confidence')::float AS evidence
               FROM resolution_requests r
               WHERE r.created_at > now() - make_interval(days => %s)
                 AND (r.status = 'abstained' OR (r.result->'evidence'->>'confidence')::float < %s)
                 AND NOT EXISTS (SELECT 1 FROM taxonomy_proposals p
                                 WHERE p.status IN ('accepted','rejected') AND r.request_id = ANY(p.member_request_ids))
               ORDER BY r.created_at DESC LIMIT %s""", (window_days, novelty_threshold, limit))

    async def replace_pending_proposals(self, proposals: list[dict], job_id: str | None) -> list[str]:
        ids = []
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("UPDATE taxonomy_proposals SET status='superseded', decided_at=now() WHERE status='pending'")
                for p in proposals:
                    pid = str(uuid.uuid4())
                    ids.append(pid)
                    await conn.execute(
                        """INSERT INTO taxonomy_proposals (proposal_id,recommendation,label_id,description,keywords,examples,product,team,
                               size,cohesion,mean_evidence,nearest_intent,neighbor_agreement,member_request_ids,job_id)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::uuid[],%s)""",
                        (pid, p["recommendation"], p["label_id"], p["description"], p["keywords"], p["examples"], p["product"], p["team"],
                         p["size"], p["cohesion"], p["mean_evidence"], p["nearest_intent"], p["neighbor_agreement"],
                         p["member_request_ids"], job_id))
        return ids

    async def list_proposals(self, status: str | None = "pending") -> list[dict]:
        cols = ("proposal_id::text AS proposal_id, status, recommendation, label_id, description, keywords, examples, product, team, size, "
                "cohesion, mean_evidence, nearest_intent, neighbor_agreement, cardinality(member_request_ids) AS members, created_at, decided_at, decided_note")
        if status:
            return await self._fetch(f"SELECT {cols} FROM taxonomy_proposals WHERE status=%s ORDER BY size DESC, created_at DESC", (status,))
        return await self._fetch(f"SELECT {cols} FROM taxonomy_proposals ORDER BY created_at DESC LIMIT 200")

    async def get_proposal(self, proposal_id: str) -> dict | None:
        try:
            uuid.UUID(proposal_id)
        except ValueError:
            return None
        rows = await self._fetch("SELECT *, proposal_id::text AS proposal_id, member_request_ids::text[] AS member_request_ids "
                                 "FROM taxonomy_proposals WHERE proposal_id=%s", (proposal_id,))
        return rows[0] if rows else None

    async def decide_proposal(self, proposal_id: str, status: str, note: str | None) -> None:
        await self._exec("UPDATE taxonomy_proposals SET status=%s, decided_note=%s, decided_at=now() WHERE proposal_id=%s",
                         (status, note, proposal_id))

    async def extend_taxonomy_label(self, dimension: str, label_id: str, keywords: list[str], examples: list[str], note: str) -> int:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute("SELECT coalesce(max(version),0)+1 AS v FROM taxonomy_versions")
                version = (await cur.fetchone())["v"]
                await conn.execute("INSERT INTO taxonomy_versions (version, note) VALUES (%s,%s)", (version, note))
                await conn.execute(
                    """UPDATE taxonomy_labels
                       SET keywords = ARRAY(SELECT DISTINCT unnest(keywords || %s::text[])),
                           examples = (examples || %s::text[])[1:30]
                       WHERE dimension=%s AND label_id=%s""", (keywords, examples, dimension, label_id))
        return version

    # ------------------------------------------------------------- job queue (worker service)
    async def claim_job(self, worker_id: str, kinds: Sequence[str]) -> dict | None:
        """Atomically take the oldest runnable job. SKIP LOCKED lets any number of workers poll without blocking each other."""
        rows = await self._fetch(
            """UPDATE jobs SET status='running', locked_by=%s, locked_at=now(), started_at=coalesce(started_at, now()), attempts=attempts+1
               WHERE job_id = (SELECT job_id FROM jobs WHERE status='queued' AND run_after <= now() AND kind = ANY(%s)
                               ORDER BY run_after, created_at FOR UPDATE SKIP LOCKED LIMIT 1)
               RETURNING job_id::text AS job_id, kind, payload, attempts, max_attempts""", (worker_id, list(kinds)))
        return rows[0] if rows else None

    async def heartbeat_job(self, job_id: str) -> None:
        await self._exec("UPDATE jobs SET locked_at=now() WHERE job_id=%s AND status='running'", (job_id,))

    async def complete_job(self, job_id: str, result: dict | None) -> None:
        await self._exec("UPDATE jobs SET status='succeeded', result=%s, error=NULL, finished_at=now(), locked_by=NULL WHERE job_id=%s",
                         (Jsonb(json.loads(json.dumps(result, default=str))) if result is not None else None, job_id))

    async def fail_job(self, job_id: str, error: str, backoff_seconds: float = 30.0) -> str:
        """Retry with exponential backoff until max_attempts, then mark failed. Returns the new status."""
        rows = await self._fetch(
            """UPDATE jobs SET error=%s, locked_by=NULL,
                      status = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'queued' END,
                      finished_at = CASE WHEN attempts >= max_attempts THEN now() ELSE NULL END,
                      run_after = now() + make_interval(secs => %s * power(2, greatest(attempts - 1, 0)))
               WHERE job_id=%s RETURNING status""", (error[:500], backoff_seconds, job_id))
        return rows[0]["status"] if rows else "failed"

    async def requeue_stale_jobs(self, timeout_seconds: float) -> int:
        """A worker that died mid-job stops heart-beating; hand its jobs back (or fail them once attempts are exhausted)."""
        rows = await self._fetch(
            """UPDATE jobs SET locked_by=NULL, error='worker lost (heartbeat timeout)',
                      status = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'queued' END,
                      finished_at = CASE WHEN attempts >= max_attempts THEN now() ELSE NULL END
               WHERE status='running' AND locked_at < now() - make_interval(secs => %s) RETURNING job_id""", (timeout_seconds,))
        return len(rows)

    async def queue_depth(self) -> dict[str, int]:
        rows = await self._fetch("SELECT status, count(*) AS n FROM jobs WHERE status IN ('queued','running') GROUP BY status")
        return {r["status"]: int(r["n"]) for r in rows}

    async def requests_window(self, newer_than_hours: float, older_than_hours: float, limit: int = 20000) -> list[dict]:
        """Logged /resolve requests between `older_than_hours` and `newer_than_hours` ago (label-free production signals)."""
        return await self._fetch(
            """SELECT status, classification->>'intent' AS intent, classification->>'severity' AS severity,
                      classification->>'sentiment' AS sentiment, (result->'evidence'->>'confidence')::float AS evidence, latency_ms
               FROM resolution_requests
               WHERE created_at <= now() - make_interval(hours => %s) AND created_at > now() - make_interval(hours => %s)
               ORDER BY created_at DESC LIMIT %s""", (newer_than_hours, older_than_hours, limit))

    async def negative_feedback_rate(self, window_hours: float) -> float | None:
        rows = await self._fetch(
            "SELECT count(*) FILTER (WHERE rating='not_helpful') AS bad, count(*) AS n FROM feedback "
            "WHERE created_at > now() - make_interval(hours => %s)", (window_hours,))
        n = int(rows[0]["n"])
        return round(int(rows[0]["bad"]) / n, 4) if n >= 5 else None

    # ------------------------------------------------------------- drift monitoring
    async def requests_window_detail(self, newer_than_hours: float, older_than_hours: float, limit: int = 20000) -> list[dict]:
        """Like requests_window, plus the complaint text and id (needed to embed, cluster and link requests to discovery proposals)."""
        return await self._fetch(
            """SELECT request_id::text AS request_id, complaint, status, classification->>'intent' AS intent, classification->>'product' AS product,
                      classification->>'severity' AS severity, classification->>'sentiment' AS sentiment,
                      (result->'evidence'->>'confidence')::float AS evidence, created_at
               FROM resolution_requests
               WHERE created_at <= now() - make_interval(hours => %s) AND created_at > now() - make_interval(hours => %s)
               ORDER BY created_at DESC LIMIT %s""", (newer_than_hours, older_than_hours, limit))

    async def drift_timeline(self, days: int, bucket: str) -> dict:
        """Requests per time bucket: volume, abstention, mean evidence and the intent mix. `bucket` is 'hour' or 'day' (validated by the caller)."""
        if bucket not in ("hour", "day"):
            raise ValueError("bucket must be 'hour' or 'day'")
        totals = await self._fetch(
            """SELECT date_trunc(%s, created_at) AS bucket, count(*) AS n, count(*) FILTER (WHERE status='abstained') AS abstained,
                      avg((result->'evidence'->>'confidence')::float) AS mean_evidence
               FROM resolution_requests WHERE created_at > now() - make_interval(days => %s) GROUP BY 1 ORDER BY 1""", (bucket, days))
        intents = await self._fetch(
            """SELECT date_trunc(%s, created_at) AS bucket, coalesce(classification->>'intent', 'unknown') AS intent, count(*) AS n
               FROM resolution_requests WHERE created_at > now() - make_interval(days => %s) GROUP BY 1, 2""", (bucket, days))
        by_bucket: dict = {}
        for r in intents:
            by_bucket.setdefault(r["bucket"], {})[r["intent"]] = int(r["n"])
        return {"bucket": bucket, "days": days, "points": [
            {"bucket": r["bucket"], "n": int(r["n"]), "abstention_rate": round(int(r["abstained"]) / int(r["n"]), 4),
             "mean_evidence": round(r["mean_evidence"], 4) if r["mean_evidence"] is not None else None, "intents": by_bucket.get(r["bucket"], {})}
            for r in totals]}

    async def save_drift_snapshot(self, snapshot_id: str, window_hours: int, baseline_days: int, report: dict, job_id: str | None) -> None:
        await self._exec(
            """INSERT INTO drift_snapshots (snapshot_id,window_hours,baseline_days,status,n_recent,n_baseline,alert_count,emerging_clusters,job_id,report)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (snapshot_id, window_hours, baseline_days, report["status"], report["n_recent"], report["n_baseline"], len(report["alerts"]),
             sum(1 for c in report["emerging_clusters"] if c["significant"]), job_id, Jsonb(json.loads(json.dumps(report, default=str)))))

    async def latest_drift_snapshot(self) -> dict | None:
        rows = await self._fetch("SELECT snapshot_id::text AS snapshot_id, created_at, window_hours, baseline_days, report "
                                 "FROM drift_snapshots ORDER BY created_at DESC LIMIT 1")
        return rows[0] if rows else None

    async def list_drift_snapshots(self, limit: int = 30) -> list[dict]:
        return await self._fetch(
            "SELECT snapshot_id::text AS snapshot_id, created_at, window_hours, baseline_days, status, n_recent, n_baseline, alert_count, emerging_clusters "
            "FROM drift_snapshots ORDER BY created_at DESC LIMIT %s", (limit,))

    async def proposals_overlapping(self, request_ids: list[str]) -> list[dict]:
        """Taxonomy proposals (any review state except superseded) that contain at least one of these requests, with how many."""
        if not request_ids:
            return []
        return await self._fetch(
            """SELECT proposal_id::text AS proposal_id, status, recommendation, label_id, size, nearest_intent, created_at,
                      cardinality(ARRAY(SELECT unnest(member_request_ids) INTERSECT SELECT unnest(%s::uuid[]))) AS overlap
               FROM taxonomy_proposals WHERE status <> 'superseded' AND member_request_ids && %s::uuid[]
               ORDER BY overlap DESC, created_at DESC""", (request_ids, request_ids))

    async def active_job_exists(self, kind: str) -> bool:
        rows = await self._fetch("SELECT 1 FROM jobs WHERE kind=%s AND status IN ('queued','running') LIMIT 1", (kind,))
        return bool(rows)

    async def last_job_time(self, kind: str) -> datetime | None:
        rows = await self._fetch("SELECT max(created_at) AS t FROM jobs WHERE kind=%s", (kind,))
        return rows[0]["t"] if rows else None
