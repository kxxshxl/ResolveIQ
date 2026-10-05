# Production scale, security and operations

Every row says **what exists in this repository now** versus **what a real production deployment should add**.
Docker Compose here is a local/dev topology, not a production deployment.

## 1. Scale and performance

| Concern | Implemented now | Production recommendation |
|---|---|---|
| **DB connection pooling** | `psycopg_pool.AsyncConnectionPool` (min 2 / max 10, env-tunable), per-connection `statement_timeout` (5 s) | Put PgBouncer (transaction mode) in front; size pools as `replicas × pool_max < max_connections`; separate read-replica pool for search |
| **pgvector indexing** | HNSW (`vector_cosine_ops`) on both embedding tables; `hnsw.iterative_scan=relaxed_order` so filtered queries still return `k` rows; GIN on `tsvector`; btree on `intent` / `product` / `resolved_at` | Tune `m` / `ef_construction` / `hnsw.ef_search` against recall targets; partial indexes per embedding model; partition tickets by `resolved_at` once > ~10 M rows; consider `halfvec` / binary quantisation to cut memory 2-32× |
| **Batch embedding** | `EmbeddingService.encode_sync` batches (64); bulk ingestion embeds a whole batch in one call; embeddings run in a worker thread so the event loop is never blocked | Dedicated embedding service (or GPU pool, e.g. Infinity / TEI) with request batching and autoscaling; the API then only does `POST /embed` |
| **Incremental ingestion** | Upsert by `ticket_id` / `article_id`; indexes are maintained by Postgres, so new rows are searchable immediately; `corpus_version` bump invalidates cached resolutions | CDC / queue-fed ingestion (Kafka, SQS) with idempotent consumers and a dead-letter queue |
| **Async / background jobs** | Separate **worker service** (`python -m app.worker`, own container / k8s Deployment). The API only enqueues (`202` + job id); the worker claims with `FOR UPDATE SKIP LOCKED`, heart-beats, retries with exponential backoff up to `max_attempts`, recovers jobs from dead workers, finishes the current job on SIGTERM, exposes `:9100/metrics` (queue depth, job duration, failures). Job kinds: `ingest_batch`, `evaluate`, `discover_classes`, `reindex`. Tested for exclusivity under concurrent claim, retry, permanent failure, backoff and stale recovery | Autoscale workers on `resolveiq_job_queue_depth` (KEDA); move to SQS / Redis Streams / Temporal when job volume or fan-out outgrows a Postgres queue (only `Repository.claim_job/complete_job/fail_job` change); dead-letter review for `failed` jobs |
| **Caching** | Redis: query-embedding cache (1 h) and full-response cache (5 min, key includes corpus + taxonomy version so ingestion invalidates it); in-process fallback when Redis is down; key namespace per deployment | Redis Cluster / managed Redis with TTL jitter; cache only non-PII keys (keys are SHA-256 of redacted text) |
| **Pagination** | `limit` / `offset` on `/search`, `/tickets`, `/articles` (bounded) | Keyset pagination on large listings |
| **Rate limiting** | Per-API-key (or per-IP) fixed window, shared across replicas through Redis (`429` + `Retry-After`) | Edge rate limiting at the gateway (envoy / API gateway) with token buckets and per-tenant quotas |
| **Timeouts** | Request timeout (`REQUEST_TIMEOUT_SECONDS` → `504`), LLM timeout, DB statement timeout, nginx `proxy_read_timeout` | Propagate deadlines end-to-end; budget per stage |
| **Retries / circuit breaking / fallback** | `ResilientLLM`: exponential-backoff retries, per-provider circuit breaker (opens after N failures, half-opens after cooldown), ordered provider chain (Ollama → OpenAI-compatible), final deterministic evidence-only fallback | Hedged requests, per-provider bulkheads, metrics-driven provider routing |
| **Model / provider abstraction** | `LLMProvider` protocol; `ollama`, `openai_compat` (OpenAI, vLLM, LM Studio, Ollama `/v1`), `mock`; selected by `LLM_PROVIDERS`. No paid API is required anywhere | Serve the LLM with vLLM / TGI on GPU nodes behind the same OpenAI-compatible interface, with autoscaling on queue depth |
| **Stateless API / horizontal scaling** | No in-process state that matters: documents, vectors, taxonomy, jobs, audit trail in Postgres; cache + rate limits in Redis; models loaded per process. Any number of replicas can run behind a load balancer | Kubernetes `Deployment` + HPA on CPU / p95 latency; readiness gates traffic until models and DB are up; PodDisruptionBudget; `startupProbe` for model load |
| **DB scaling** | Single Postgres | Managed Postgres with a read replica for retrieval; vertical scale first (pgvector is memory-bound: keep the HNSW index in RAM); shard by tenant / region only when one index no longer fits |
| **Embedding / index migration** | Embeddings are stored per `(row, model)`. Switching `EMBEDDING_MODEL` + running `ingestion.reindex()` backfills only rows missing that model's vectors; queries filter on the active model, so the old model keeps serving until the switch | Blue/green: backfill in the background, evaluate old vs new on the benchmark (`app/evaluation`), flip the config, delete old vectors. A model with a different dimension needs a new embeddings table (or `vector` column) - the schema notes this explicitly |
| **Observability** | Prometheus metrics for both services (request / stage / retrieval / embedding / rerank / affect / LLM latency, failures, tokens, circuit state, classification and evidence confidence, abstentions, citation failures, ingestion, evaluation runs, PII/injection/rate-limit counters, corpus size, job queue depth / duration / failures, discovery runs and proposals); JSON logs with trace id; Grafana dashboard from `infra/grafana`; **9 Prometheus alert rules** (`infra/prometheus/alerts.yml`, validated with `promtool`): error rate, p95 latency, LLM circuit open, abstention surge, evidence-confidence drop, intent-mix drift, negative feedback, queue backlog, failing jobs | Distributed tracing (OpenTelemetry), SLO burn-rate alerts, log shipping |
| **Label-free drift monitoring** | `GET /api/v1/monitoring/drift` compares the last 24 h of logged requests with a 14-day baseline: abstention rate, mean evidence confidence, p95 latency, Jensen-Shannon divergence of intent / severity / sentiment mix, negative-feedback share; the worker publishes them as gauges every 5 min and logs alerts. Needs >= 30 requests in each window before alerting | Slice by tenant / channel; add embedding-centroid drift; page only on sustained breach |
| **Quality regression gate** | `python -m app.evaluation.gate` fails CI when any recorded metric falls below its floor in `data/eval/thresholds.json` (classification on hand-written and blind sets, dense retrieval, robustness to typos/caps/noise, discovery recovery); an unmeasured check also fails, so a silently skipped suite cannot pass | Run nightly on a fresh sample of labelled production tickets; track metrics over time |
| **Health / readiness** | `/health` (liveness), `/health/ready` (DB reachable, models loaded, corpus status, Redis and LLM reported; 503 when not ready) | Separate startup / liveness / readiness probes in Kubernetes |
| **Containerisation** | Multi-stage unprivileged-nginx frontend, backend image with baked model weights, non-root, read-only root fs, dropped capabilities, healthchecks, one-shot `migrate` job, `docker-compose.prod.yml` + `k8s/` + CI | Image scanning, SBOM, signed images, distroless runtime |

### Capacity notes (measured on a laptop, indicative only)
Query embedding ≈ 1 ms (GPU) / a few ms (CPU); rules + kNN classification ≈ 3 ms; **severity/sentiment NLI model ≈ 40 ms on GPU,
≈ 1.2 s on CPU (float32, 8 threads)**, overlapped with retrieval so only the excess shows up in latency; dense retrieval ≈ 1-2 ms at 250
tickets; reranking 20 candidates ≈ 15-20 ms; LLM generation (Qwen3-4B on an RTX 4070) ≈ 5-6 s and dominates end-to-end latency.
At scale, the LLM tier - not retrieval - is the cost and throughput bottleneck, which is why abstention (no LLM call), the
response cache and the evidence-only fallback matter. Memory: the large NLI model needs about 2.5 GB resident per process, so
API and worker containers are sized at 5 GB limits; for CPU-only fleets use `AFFECT_MODEL=...deberta-v3-base-zeroshot-v2.0` (0.45 s,
but lower accuracy - retrain the artifact and re-run the gate) or serve the NLI model on a shared GPU.
**Do not quantise it to int8**: dynamic int8 took blind severity/sentiment from 0.64 / 0.78 to 0.48 / 0.34.
**Also:** recent `transformers` versions load this fp16 checkpoint in fp16 even on CPU (about 10x slower); the loader forces float32 on CPU.

## 2. Security

| Concern | Implemented now | Production recommendation |
|---|---|---|
| **Input validation** | Pydantic schemas with length / pattern limits; complaint ≤ 4000 chars; strategy / label ids validated; unknown taxonomy labels rejected (422) | Request-size limits at the gateway, schema versioning |
| **PII** | Regex redaction (email, phone, Luhn-valid cards, labelled account / customer ids, national-id patterns, IPv4) applied **before** storage, embedding, caching, logging and LLM prompts; only redacted text is ever persisted; redaction counts are exposed as metrics | NER-based scrubbing (names, addresses) e.g. Presidio, DLP scanning, retention policy and right-to-erasure job, field-level encryption for any raw data that must be kept |
| **Secrets** | Everything via environment variables (`.env` is git-ignored; `.env.example` has placeholders only); no keys in code or images | Secret manager (Vault / cloud KMS), short-lived DB credentials, rotation |
| **Safe logging** | Logs contain trace id, route, status, timing, labels; complaint text and PII are not logged | Central log redaction filter as defence in depth |
| **Prompt injection** | Complaint passed as quoted JSON data with an explicit "untrusted" rule; injection-like phrases flagged (`resolveiq_prompt_injection_flagged_total`, response warning); output must match a JSON schema; every citation validated against retrieved ids so injected "sources" cannot survive; the model has no tools / side effects | Dedicated injection classifier, canary tokens, output moderation |
| **Authentication** | `X-API-Key` (constant-time compare) on `/api/v1/*`; `/health*` and `/metrics` stay open for probes. Disabled when `API_KEYS` is empty (logged as a warning) | OIDC / JWT with per-tenant scopes, mTLS between services, `/metrics` bound to an internal network |
| **Least privilege** | Two DB roles: the owner role runs migrations; the API connects as `resolveiq_app` with `SELECT/INSERT/UPDATE/DELETE` only (no DDL, no access to `schema_migrations`); containers run as a non-root user; DB / Redis ports bound to `127.0.0.1` | Network policies, separate roles for ingestion and read-only search, row-level security for multi-tenancy |
| **Transport** | Production compose: automatic HTTPS (Caddy), HSTS and security headers, 1 MB body limit, `/metrics` not public; Kubernetes: TLS ingress via cert-manager | mTLS between services / service mesh |

## 3. Evolving classes, taxonomy, embeddings and stale knowledge

* **New ticket classes** - `POST /api/v1/taxonomy/labels` (data only). Rules pick up its keywords, the prototype
  embedding (from description + examples) makes it work zero-shot, and kNN improves automatically as tickets of that
  class are ingested. Demonstrated by `tests/test_ingestion_and_evolution.py` and the `evolving` evaluation suite.
* **Discovering classes nobody told the system about** - the sufficiency gate only abstains on 24% of complaints from genuinely
  new classes (the rest are answered from the nearest existing class, which can be confidently wrong). `POST /api/v1/taxonomy/discover`
  (also scheduled daily by the worker) clusters recent poorly-explained requests and writes *reviewable proposals* with keywords,
  exemplars, suggested product/team and a recommendation (`new_class` vs `extend_existing`). `POST .../proposals/{id}/accept`
  creates the intent (or extends an existing one), bumps the taxonomy version and invalidates cached answers; `.../reject`
  prevents those requests from being proposed again. On three never-seen classes it recovered 3/3 at purity 1.0 (precision 0.6);
  see `docs/evaluation.md`. Accepted classes work zero-shot from the proposal's keywords and examples and improve as resolved tickets
  are ingested. Humans stay in the loop on purpose: wrongly created categories misroute customers.
* **Taxonomy versioning** - every change inserts into `taxonomy_versions`; labels record `introduced_in`; tickets record
  `taxonomy_version`. Deprecating a label (`status='deprecated'`) hides it from classification without breaking
  history. When a class is *split* or *merged*, re-label affected tickets with a one-off job and bump the version.
* **Embedding model change** - see "Embedding / index migration" above. Quality gate: run `python -m app.evaluation.run
  --suites retrieval` on the old and new model and compare before switching.
* **Index rebuild** - HNSW is incremental; rebuild (`REINDEX INDEX CONCURRENTLY`) only after bulk loads or parameter changes.
* **Stale KB documents** - `POST /api/v1/articles/{id}/deprecate` flips `status`; every retrieval strategy filters on
  `status='active'`, verified by the `stale_article_leaks` evaluation check (0 leaks). Recommended: a scheduled job that
  flags articles not updated for N months or contradicting newer ones.
* **Feedback-driven improvement** - `POST /api/v1/feedback` stores helpful / not-helpful and a corrected intent against the
  audited request. *Not automated yet.* Recommended loop: (1) review `not_helpful` + abstentions weekly, (2) promote
  confirmed good resolutions through the normal ingestion endpoint (they become new evidence immediately), (3) use
  corrected intents as extra labelled examples / evaluation data, (4) track abstention-rate and citation-failure drift.

## 4. Kubernetes (manifests shipped in `k8s/`, validated but not applied to a live cluster; see `DEPLOYMENT.md`)

`backend` Deployment (N replicas, HPA) → Service → Ingress (TLS, auth, rate limit) · `worker` Deployment (`k8s/worker.yaml`:
same image, `python -m app.worker`, own NetworkPolicy, 120 s termination grace for in-flight jobs) · managed Postgres (+ PgBouncer, read replica) · managed Redis · `llm` Deployment on GPU nodes (vLLM) ·
`embedder` Deployment (TEI) · Prometheus Operator + Grafana · migration `Job` per release.
