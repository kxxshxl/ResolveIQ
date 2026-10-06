# Database engineering

One PostgreSQL 17 database with pgvector 0.8 holds documents, vectors, the full-text index, the taxonomy, the audit trail (cases and feedback), the drift snapshots and
the job queue. This page records what was reviewed, what was changed because of it, what was measured, and what was not.

Everything below is reproducible with scripts in `backend/scripts/` against the Compose Postgres; result files are in `data/eval/results/`.

## Schema and indexes

| Table | Purpose | Indexes (besides the primary key) |
|---|---|---|
| `tickets`, `knowledge_articles` | resolved tickets and KB articles (redacted text, labels, `metadata`), generated `search_text tsvector`, generated `content_hash` | GIN on `search_text`; btree on `intent`, `product`, `resolved_at`; `content_hash` (duplicate detection) |
| `ticket_embeddings`, `kb_embeddings` | one vector row per document **and embedding model** | HNSW `vector_cosine_ops` (m=16, ef_construction=64); primary key `(document, model)` |
| `resolution_requests` | one row per resolution (a *case*): redacted complaint, classification, retrieved sources, result, trace, provenance, query embedding | `created_at DESC`; `(status, created_at DESC)`; `(classification->>'intent', created_at DESC)`; partial index on rows still without an embedding |
| `feedback` | agent ratings with reasons, rejected sources, corrected intent, edited steps, provenance copy | `request_id` (the foreign key); `created_at DESC` |
| `jobs` | the queue (`FOR UPDATE SKIP LOCKED`, heartbeat, retry, recovery) | partial `(run_after) WHERE status = 'queued'` (the claim query); `(kind, created_at DESC)` |
| `taxonomy_labels`, `taxonomy_versions`, `taxonomy_proposals`, `drift_snapshots` | versioned taxonomy, discovery proposals, drift results | by status / created time |

Migrations are forward-only SQL files applied by `python -m app.db.migrate` as the owner role; the API connects as a role with DML privileges only. Migration 004 is the one that
came out of the review below.

## The review: query plans, found and fixed

`scripts/db_review.py` loads 20 000 requests and 50 000 feedback rows (synthetic, spread over 30 days) into a throw-away copy of the database and runs `EXPLAIN (ANALYZE, BUFFERS)`
on every query the console and the analysis jobs issue. Result file: `data/eval/results/db_query_plans.json`. Findings:

| Finding | Evidence | Change |
|---|---|---|
| `feedback.request_id` is a foreign key (`ON DELETE SET NULL`) with no index. Every delete of a request scans the whole feedback table, and the "latest rating per case" subquery in the case list scans it per row. | deleting 300 requests: 0.404 s without the index, 0.009 s with it (43.9x); latest-rating query: 26.9 ms without, 0.16 ms with | `feedback_request_idx` (migration 004) |
| Aggregations by intent over a time range (quality view, timeline) read the JSON column for every row in the window. | plan used a bitmap scan on `created_at` plus a filter | expression index `(classification->>'intent', created_at DESC)` |
| Back-filling query embeddings for rows logged before they were stored would scan the whole table. | plan before: sequential scan | partial index `WHERE embedding IS NULL` |
| The case list, drift window and timeline queries | all use an index scan on `created_at` or `status`; 0.01 to 5 ms on 20 000 rows | none needed |
| Ticket dense search shows a sequential scan | correct for 250 vectors: the planner is right not to use HNSW at this size (see the scale experiment for the 100 000-row plan) | none |

The 30-day feedback join (48 041 rows, 16.7 ms) is a hash join over everything in the window by design: it is the analysis job, not a request-path query.

## HNSW configuration and the 100 000-vector experiment

`scripts/pgvector_scale_benchmark.py` creates a **synthetic** table of 100 000 vectors with 384 dimensions (a Gaussian mixture of 400 topics, so that nearest neighbours exist, the
same dimension as the real embedding model), builds the same HNSW index the application uses, and measures build time, storage, latency percentiles and recall against exact search for
500 queries. It is labelled synthetic in its output: vectors without text say nothing about retrieval quality, only about what the index costs. Result file: `data/eval/results/pgvector_scale.json`.

| Measurement (one Postgres container, laptop, client and server on the same machine) | Value |
|---|---|
| Load | 100 000 rows in 11.6 s (8 626 rows/s) |
| HNSW build (m=16, ef_construction=64) | 22.4 s, serial (the container's 64 MB `/dev/shm` cannot back a parallel build, so this is an upper bound on build time) |
| Storage | table 156.5 MB, HNSW index 195.3 MB, 355 MB total (raw vector data 146.5 MB) |
| Plan | the planner uses the HNSW index (checked with `EXPLAIN`) |

Search, top-10, 500 queries, by `hnsw.ef_search`:

| ef_search | p50 ms | p95 ms | p99 ms | recall@10 vs exact | top-1 agreement |
|---|---|---|---|---|---|
| 10 | 1.19 | 1.59 | 1.96 | 0.781 | 0.788 |
| 20 | 1.21 | 1.55 | 1.87 | 0.896 | 0.902 |
| 40 (pgvector's default is 40) | 1.30 | 1.78 | 2.13 | 0.968 | 0.970 |
| **100 (the application's setting)** | 1.42 | 1.98 | 2.56 | 0.997 | 0.998 |
| 200 | 1.61 | 2.22 | 2.79 | 0.999 | 1.000 |

Sanity checks in the same run: a vector queried against the index returns itself at rank 1 (197 of 200 queries; the other three returned another vector first, which an approximate index may do), results are sorted by
similarity (100/100) and the reported similarity equals the exact dot product (100/100). Eight client threads sustained 2 944 queries/s at ef_search 40 with p99 4.7 ms.

**Decision: `HNSW_EF_SEARCH=100`.** At 100 000 vectors the default of 40 loses 3% of true neighbours; going to 100 costs about 0.1 ms at p50 and buys 0.997 recall. The setting is applied per connection
(`app/db/repository.py`) and is configurable. m and ef_construction stay at pgvector's defaults: nothing in the measurements argued for a larger graph at this size.

**Filtered search.** The retrieval path filters by product and intent. With a filter, an HNSW scan stops after `ef_search` candidates and may return fewer than `k` rows. The same run
measured this: with a filter matching 5% of rows and `hnsw.iterative_scan=off`, recall@10 was 0.188 and every query returned fewer than 10 rows; with `iterative_scan=relaxed_order`, recall was
0.982 and none came back short, at 2.2 ms instead of 1.5 ms p50. The application sets `relaxed_order` on every connection. A filter matching 0.5% of rows is planned as a btree scan (exact) either way.

What this does **not** show: real embeddings and real queries at that scale, memory pressure (the index is 195 MB; whether it stays cached is a property of the host), concurrent writes during
search, or a managed-cloud database. The corpus in this repository is 276 documents, so none of this is exercised by the demo; it is evidence that the chosen index and settings keep working
when the corpus is 400 times larger.

## Ingestion

Ingestion was one transaction per ticket, even in the batch path. `Repository.bulk_upsert_tickets` now writes a whole batch in one transaction with pipelined `executemany` statements (chunks of 500), and the batch
endpoint uses it. `scripts/ingest_benchmark.py` compares the two database paths on the same rows with random embeddings (the embedding model is excluded on purpose, so the number is what the
database layer costs). Result file: `data/eval/results/ingest_benchmark.json`.

| 2 000 tickets with embeddings (4 000 rows written per path) | tickets per second |
|---|---|
| one transaction per ticket (`upsert_ticket`, the old batch path) | 166 (12.0 s) |
| one batched transaction (`bulk_upsert_tickets`) | 717 (2.8 s), 4.3x |
| the same batch again (all updates, idempotent) | 1.3 s, 0 created, 2 000 updated |

The batch is all-or-nothing in the database; if it fails, the caller falls back to per-row writes so that one bad row costs only itself (tests: `test_one_bad_row_costs_only_itself`,
`test_if_the_batched_write_fails_the_batch_falls_back_to_row_by_row`). The real ingestion also pays for embedding the text, which this comparison leaves out on purpose, so the end-to-end gain is smaller than 4.3x.

Re-ingesting the same id updates the row (upsert), so ids cannot be duplicated. The same complaint and resolution under two different ids is detected, not prevented (a deliberate re-ingest must stay
possible): `content_hash` is a generated column with an index, and the health view reports groups of identical content.

## Pagination

Listing endpoints (`/tickets`, `/articles`) accept `?cursor=start` and then the `next_cursor` of the previous page: keyset pagination on `(sort key, id)`, which costs the same on page 1 and on
page 10 000, unlike `OFFSET`. `offset` still works for existing callers. The case list uses `created_at DESC` with the index above.

## Connection handling

`psycopg_pool.AsyncConnectionPool`, `DB_POOL_MIN=2`, `DB_POOL_MAX=10` per process, a 5 s `statement_timeout` on every connection (`DB_STATEMENT_TIMEOUT_MS`), request and job timeouts above that, and the two per-connection settings above
applied in the pool's `configure` hook so that every pooled connection has them. With N API replicas and M workers the server needs `max_connections` above `10 * (N + M)`; PgBouncer in transaction mode
is the next step and is compatible because no session state other than those settings is used (they would move into the role or database defaults).

## The database health view

`GET /api/v1/system/db` (and System health in the console) reports, without reading any row contents: server and pgvector versions, table and index sizes, estimated and exact row counts, dead tuples
and last autovacuum, the HNSW parameters of each vector index (and whether they are the defaults), index use counts (indexes never scanned are listed as candidates for removal, with the caveat that
a fresh database has no history), **embedding coverage** (documents without an embedding for the configured model, embeddings for other models), and **integrity** (orphaned embeddings, feedback
without a request, pending proposals whose requests are gone, tickets or articles with an intent that is not in the taxonomy, citations in recent resolutions that point at documents that no longer
exist, duplicate content). Each finding has a level and a sentence saying what to do. Tests: `tests/test_database_layer.py` and `tests/test_console_api.py` (the health report is run against a database
into which orphans and duplicates are inserted on purpose).
