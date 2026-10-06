# Production scale, security and operations

Every row says **what exists in this repository now** versus **what a real production deployment should add**.
Docker Compose here is a local/dev topology, not a production deployment.

## 1. Scale and performance

| Concern | Implemented now | Production recommendation |
|---|---|---|
| **DB connection pooling** | `psycopg_pool.AsyncConnectionPool` (min 2 / max 10, env-tunable), per-connection `statement_timeout` (5 s) | Put PgBouncer (transaction mode) in front; size pools as `replicas × pool_max < max_connections`; separate read-replica pool for search |
| **pgvector indexing** | HNSW (`vector_cosine_ops`, m=16, ef_construction=64) on both embedding tables, `hnsw.ef_search=100` (measured on 100 000 synthetic vectors: recall@10 0.968 to 0.997 for +0.1 ms, [`database.md`](database.md)); `hnsw.iterative_scan=relaxed_order` so filtered queries still return `k` rows (recall 0.19 to 0.98 in the same experiment); GIN on `tsvector`; btree on `intent` / `product` / `resolved_at` | Tune `m` / `ef_construction` / `hnsw.ef_search` against recall targets; partial indexes per embedding model; partition tickets by `resolved_at` once > ~10 M rows; consider `halfvec` / binary quantisation to cut memory 2-32× |
| **Batch embedding** | `EmbeddingService.encode_sync` batches (64); bulk ingestion embeds a whole batch in one call; embeddings run in a worker thread so the event loop is never blocked | Dedicated embedding service (or GPU pool, e.g. Infinity / TEI) with request batching and autoscaling; the API then only does `POST /embed` |
| **Incremental ingestion** | Upsert by `ticket_id` / `article_id`; indexes are maintained by Postgres, so new rows are searchable immediately; `corpus_version` bump invalidates cached resolutions | CDC / queue-fed ingestion (Kafka, SQS) with idempotent consumers and a dead-letter queue |
| **Async / background jobs** | Separate **worker service** (`python -m app.worker`, own container / k8s Deployment). The API only enqueues (`202` + job id); the worker claims with `FOR UPDATE SKIP LOCKED`, heart-beats, retries with exponential backoff up to `max_attempts`, recovers jobs from dead workers, finishes the current job on SIGTERM, exposes `:9100/metrics` (queue depth, job duration, failures). Job kinds: `ingest_batch`, `evaluate`, `discover_classes`, `reindex`. Tested for exclusivity under concurrent claim, retry, permanent failure, backoff and stale recovery | Autoscale workers on `resolveiq_job_queue_depth` (KEDA); move to SQS / Redis Streams / Temporal when job volume or fan-out outgrows a Postgres queue (only `Repository.claim_job/complete_job/fail_job` change); dead-letter review for `failed` jobs |
| **Caching** | Redis: query-embedding cache (1 h) and full-response cache (5 min, key includes corpus + taxonomy version so ingestion invalidates it); in-process fallback when Redis is down; key namespace per deployment | Redis Cluster / managed Redis with TTL jitter; cache only non-PII keys (keys are SHA-256 of redacted text) |
| **Pagination** | `limit` / `offset` on `/search`, `/tickets`, `/articles` (bounded), and keyset pagination (`?cursor=start`, then `next_cursor`) on the document listings, so page 10 000 costs the same as page 1; the case list is index-ordered by time | Keyset on every large listing |
| **Rate limiting** | Per-API-key (or per-IP) fixed window, shared across replicas through Redis (`429` + `Retry-After`) | Edge rate limiting at the gateway (envoy / API gateway) with token buckets and per-tenant quotas |
| **Timeouts** | Request timeout (`REQUEST_TIMEOUT_SECONDS` → `504`), LLM timeout, DB statement timeout, nginx `proxy_read_timeout` | Propagate deadlines end-to-end; budget per stage |
| **Retries / circuit breaking / fallback** | `ResilientLLM`: exponential-backoff retries, per-provider circuit breaker (opens after N failures or one timeout, half-opens after the cooldown with a single probe), a per-provider concurrency limit with a short queue, and one total time budget per generation, ordered provider chain (Ollama → OpenAI-compatible), final deterministic evidence-only fallback | Hedged requests, per-provider bulkheads, metrics-driven provider routing |
| **Model / provider abstraction** | `LLMProvider` protocol; `ollama`, `openai_compat` (OpenAI, vLLM, LM Studio, Ollama `/v1`), `mock`; selected by `LLM_PROVIDERS`. No paid API is required anywhere | Serve the LLM with vLLM / TGI on GPU nodes behind the same OpenAI-compatible interface, with autoscaling on queue depth |
| **Stateless API / horizontal scaling** | No in-process state that matters: documents, vectors, taxonomy, jobs, audit trail in Postgres; cache + rate limits in Redis; models loaded per process. Any number of replicas can run behind a load balancer | Kubernetes `Deployment` + HPA on CPU / p95 latency; readiness gates traffic until models and DB are up; PodDisruptionBudget; `startupProbe` for model load |
| **DB scaling** | Single Postgres | Managed Postgres with a read replica for retrieval; vertical scale first (pgvector is memory-bound: keep the HNSW index in RAM); shard by tenant / region only when one index no longer fits |
| **Embedding / index migration** | Embeddings are stored per `(row, model)`. Switching `EMBEDDING_MODEL` + running `ingestion.reindex()` backfills only rows missing that model's vectors; queries filter on the active model, so the old model keeps serving until the switch | Blue/green: backfill in the background, evaluate old vs new on the benchmark (`app/evaluation`), flip the config, delete old vectors. A model with a different dimension needs a new embeddings table (or `vector` column) - the schema notes this explicitly |
| **Observability** | Prometheus metrics for both services, including adaptive-retrieval decisions, feedback, case replays and cluster timing (request / stage / retrieval / embedding / rerank / affect / LLM latency, failures, tokens, circuit state, classification and evidence confidence, abstentions, citation failures, ingestion, evaluation runs, PII/injection/rate-limit counters, corpus size, job queue depth / duration / failures, discovery runs and proposals); JSON logs with trace id; Grafana dashboard from `infra/grafana`; **12 Prometheus alert rules** (`infra/prometheus/alerts.yml`, validated with `promtool`): error rate, p95 latency, LLM circuit open, abstention surge, evidence-confidence drop, intent-mix drift, new-topic and sustained drift, stale drift analysis, negative feedback, queue backlog, failing jobs; **opt-in OpenTelemetry tracing** across API, pipeline stages, Postgres, Redis, LLM calls and worker jobs ([`observability.md`](observability.md)) | SLO burn-rate alerts, log shipping, a retained trace backend (Tempo or Jaeger with storage) |
| **Statistical drift monitoring** | Worker job `drift_analysis` (every 6 h or on demand) compares the last 24 h with the previous 14 days: chi-square and per-category tests with PSI on intent / product / severity / sentiment, KS on evidence confidence, abstention test, permutation test on the embedding centroid, and exact tests on new-topic clusters; Holm-corrected to a 1% false-alarm budget, each alert also needs a minimum effect size. Results are stored (`drift_snapshots`), served by `GET /api/v1/monitoring/drift/status` and the Drift monitoring tab, published as gauges, and linked to the discovery proposals that cover each cluster (a significant uncovered cluster enqueues a discovery run; proposals still need review). Measured on a deterministic injection demo: 1.0% false alarms, 16% to 40% intent surge found 30/30, 10-complaint new topic found 67%, a 6-complaint one 7% by the alarm and 43% as a discovery proposal for a person to review ([`drift.md`](drift.md)) | Calibrate thresholds on your own traffic history; seasonality-aware or frozen baselines; per-tenant / per-channel slices; a stronger embedding model for new-topic power |
| **Label-free drift monitoring** | `GET /api/v1/monitoring/drift` compares the last 24 h of logged requests with a 14-day baseline: abstention rate, mean evidence confidence, p95 latency, Jensen-Shannon divergence of intent / severity / sentiment mix, negative-feedback share; the worker publishes them as gauges every 5 min and logs alerts. Needs >= 30 requests in each window before alerting | Slice by tenant / channel; add embedding-centroid drift; page only on sustained breach |
| **Quality regression gate** | `python -m app.evaluation.gate` fails CI when any recorded metric falls below its floor in `data/eval/thresholds.json` (classification on hand-written and blind sets, dense and adaptive retrieval, robustness to typos/caps/noise, recurring clusters, discovery recovery, end-to-end correctness); an unmeasured check also fails, so a silently skipped suite cannot pass | Run nightly on a fresh sample of labelled production tickets; track metrics over time |
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

### Load testing (measured, reproducible)
`POST /api/v1/resolve` was driven with Locust at 1, 5, 10 and 25 concurrent users, always against a throw-away database so the evaluation data and
taxonomy are untouched. Run it with `python loadtest/run_loadtest.py --suite <name> --scenarios no_llm llm --levels 1 5 10 25 --duration 60`
and `python loadtest/summarize.py <name>`; the full method, the raw per-request data and the generated tables are in
[`loadtest/`](../loadtest/README.md) (`results/2026-10-06/SUMMARY.md`, `results/2026-10-06-after-fix/SUMMARY.md`).

**Configuration.** Intel(R) Core(TM) i9-14900HX (24 cores / 32 threads), 31.7 GB RAM, NVIDIA GeForce RTX 4070 Laptop GPU shared by the LLM
(Ollama `qwen3:4b-instruct`, 100% on GPU), the severity/sentiment NLI model and the embedding model. One API process (uvicorn, no replicas), default
retrieval (`dense`), rate limiter and tracing off, Postgres and Redis in Docker. Closed-loop load, no think time (N users = N requests in flight), a fresh
API process per run, one unique case reference per complaint so no cache can hit (except where stated). Mix: 275 in-domain complaints from the evaluation
sets plus 5% out-of-domain, fixed seed. 60 s per run, 120 s for the real-LLM runs, and for the hung-LLM runs 180 s on the baseline code and 120 s after the fix. Latency columns are for
HTTP-200 responses; errors and timeouts are counted separately. \* = under 100 completed requests, so the P99 is effectively the maximum.

**Throughput and latency.**

| scenario | users | OK req/s | P50 | P95 | P99 | errors | timeouts | answers: LLM / evidence-only / abstained % | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| evidence-only (no LLM) | 1 | 15.96 | 62 ms | 75 ms | 78 ms | 0% | 0% | 0 / 90 / 10 |  |
| evidence-only (no LLM) | 5 | 17.61 | 283 ms | 312 ms | 354 ms | 0% | 0% | 0 / 90 / 10 |  |
| evidence-only (no LLM) | 10 | 17.40 | 570 ms | 627 ms | 685 ms | 0% | 0% | 0 / 90 / 10 |  |
| evidence-only (no LLM) | 25 | 17.39 | 1,426 ms | 1,533 ms | 1,592 ms | 0% | 0% | 0 / 89 / 11 |  |
| evidence-only, affect model off | 1 | 33.17 | 29 ms | 40 ms | 42 ms | 0% | 0% | 0 / 91 / 9 | ablation |
| evidence-only, affect model off | 10 | 39.98 | 245 ms | 341 ms | 388 ms | 0% | 0% | 0 / 90 / 10 | ablation |
| evidence-only, affect model off | 25 | 38.73 | 641 ms | 837 ms | 914 ms | 0% | 0% | 0 / 90 / 10 | ablation |
| cache hits (mock LLM, repeated complaints) | 10 | 370.47 | 19 ms | 23 ms | 511 ms | 0% | 0% | 92 / 0 / 8 | 98% cache hits |
| cache hits (mock LLM, repeated complaints) | 25 | 292.71 | 54 ms | 70 ms | 1,386 ms | 0% | 0% | 92 / 0 / 8 | 98% cache hits |
| full pipeline, real LLM (before fix) | 1 | 0.21 | 5,237 ms | 6,663 ms | 6,913 ms* | 0% | 0% | 84 / 0 / 16 |  |
| full pipeline, real LLM (before fix) | 5 | 0.24 | 24.4 s | 28.2 s | 28.9 s* | 0% | 0% | 76 / 0 / 24 |  |
| full pipeline, real LLM (before fix) | 10 | 0.19 | 26.4 s | 44.1 s | 44.6 s* | 23% | 23% | 70 / 0 / 30 |  |
| full pipeline, real LLM (before fix) | 25 | 0.21 | 23.7 s | 45.2 s | 45.2 s* | 44% | 44% | 64 / 0 / 36 |  |

**Main bottleneck: LLM generation, then the severity/sentiment model.**
1. **The LLM caps the full pipeline at about 0.2 answers/s no matter how many users there are** (0.21, 0.24,
   0.19 and 0.21 req/s at 1, 5, 10 and 25 users): the single local model behaves as a one-server queue, so latency grows linearly with users
   (P50 5,237 ms at 1 user, 24.4 s at 5). At one user the `generate` stage is 4,520 of 4,572 ms
   (99%); the GPU is 86-90% busy. Traces agree: `llm.call` takes 5.2 s with one user and
   23.5 s with five (4.5x) while every other span moves by at most 1.7x
   (`classify` is 1.6x slower too, consistent with the GPU being shared with the LLM).
2. **Without the LLM the NLI affect model is the limit** (evidence-only mode, 17.4 req/s from 5 users up, though a single user already reaches 16.0). Its throughput equals 1 / the
   one-user service time (56 ms), and the mean time of the `classify` stage grows from 36 to 535 ms at 10 users. In the traces
   `classify.affect` goes from 33 to 584 ms (17.5x) and accounts for almost all of the extra latency, while every database
   span stays within 1.2x of its unloaded time. Cause (code reading, not separately profiled): `AffectModel.cue_scores` holds a lock around each
   forward pass and every request runs its own 14-hypothesis pass, so concurrent requests queue. **Ablation:** with `AFFECT_ENABLED=false` the same load reaches 40.0 req/s (2.3x).
3. **After that, embedding and shared GPU/Python time** (about 40 req/s; with the affect model off the mean single-query embedding goes from
   4.6 ms at 1 user to 87 ms at 10, and the API process uses 1.8 cores on average;
   this second limit was not traced or profiled further).
4. **Not bottlenecks:** Postgres (retrieval 1.9 ms), Redis, and the API layer: repeated complaints answered from the response cache
   run at 370 req/s with a 19 ms P50 (cache hit ~7 ms). The load generator used at most 0.7 core.
   When this baseline was measured, evidence-only answers were deliberately **not** cached, so an outage paid the full pipeline cost per request; they are now cached for 30 s
   (`CACHE_DEGRADED_TTL_SECONDS`, see [`performance.md`](performance.md)).

**Rough sizing (arithmetic on the measurements, with an assumed workload).** If an agent triggers one resolution every 30-60 s, one local GPU running both
models serves about 6-13 agents with LLM answers, and the evidence-only path on the order of 500-1000. Production would use a dedicated LLM tier (or hosted model)
and several API replicas; none of that was measured here.

**When the LLM is unavailable or too slow** the evidence-only fallback must keep answering. Result: it does for a refused connection, but with the shipped defaults it
**did not** for a hung or overloaded LLM, which the load test exposed as two defects (fixed in this change):

| scenario | users | OK req/s | P50 | P95 | P99 | errors | timeouts | answers: LLM / evidence-only / abstained % | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| LLM down (connection refused) | 10 | 14.85 | 573 ms | 668 ms | 5,074 ms | 0% | 0% | 0 / 90 / 10 | before |
| LLM down (connection refused) | 10 | 15.50 | 591 ms | 677 ms | 4,669 ms | 0% | 0% | 0 / 90 / 10 | after fix |
| LLM down (connection refused) | 25 | 15.77 | 1,476 ms | 1,641 ms | 5,561 ms | 0% | 0% | 0 / 89 / 11 | after fix |
| LLM hangs, defaults | 1 | 0.00 | - | - | - | 100% | 100% | 0 / 0 / 0 | before: every LLM-needing request is a 504 |
| LLM hangs, defaults | 5 | 0.02 | 139 ms | 187 ms | 191 ms* | 71% | 71% | 0 / 0 / 100 | before: every LLM-needing request is a 504 |
| LLM hangs, defaults | 10 | 0.04 | 385 ms | 781 ms | 818 ms* | 74% | 74% | 0 / 0 / 100 | before: every LLM-needing request is a 504 |
| LLM hangs, `LLM_TIMEOUT_SECONDS=15` | 10 | 8.67 | 569 ms | 622 ms | 31.1 s | 0% | 0% | 0 / 90 / 10 | config-only workaround: breaker stampede |
| LLM hangs, defaults | 1 | 3.90 | 62 ms | 76 ms | 80 ms | 0% | 0% | 0 / 88 / 12 | after fix |
| LLM hangs, defaults | 5 | 13.02 | 230 ms | 300 ms | 320 ms | 0% | 0% | 0 / 90 / 10 | after fix |
| LLM hangs, defaults | 10 | 13.02 | 561 ms | 626 ms | 681 ms | 0% | 0% | 0 / 90 / 10 | after fix |
| LLM overloaded (real, 1 GPU) | 10 | 0.19 | 26.4 s | 44.1 s | 44.6 s* | 23% | 23% | 70 / 0 / 30 | before |
| LLM overloaded (real, 1 GPU) | 10 | 3.29 | 570 ms | 30.1 s | 30.9 s | 0% | 0% | 2 / 85 / 12 | after fix |
| LLM overloaded (real, 1 GPU) | 25 | 0.21 | 23.7 s | 45.2 s | 45.2 s* | 44% | 44% | 64 / 0 / 36 | before |
| LLM overloaded (real, 1 GPU) | 25 | 7.02 | 1,698 ms | 30.4 s | 32.5 s | 0% | 0% | 1 / 88 / 11 | after fix |

* **Timeouts were inconsistent.** One LLM attempt could wait 45 s, plus one retry, against a 60 s request timeout. A hung LLM therefore produced a 504 for every request that needed
  the model (74% at 10 users; the rest were abstentions that never call it) and the circuit breaker never opened, because the cancelled request never reported its failure. Under real overload
  (23% / 44% errors at 10 / 25 users) a fallback answer that takes ~100 ms was available but never returned. **Fix:** `LLM_TOTAL_BUDGET_SECONDS` (default 30 s) bounds all attempts of one
  generation, each attempt gets only the remaining time, a timeout counts as a breaker failure, and startup refuses (production) a budget not below `REQUEST_TIMEOUT_SECONDS`.
* **Half-open stampede.** After the cooldown every in-flight request probed the dead LLM at once (the `LLM_TIMEOUT_SECONDS=15` run: 8.7 req/s at 10 users, slower than 13.0 at 5, P99 31.1 s). **Fix:** half-open admits exactly one probe.
* **After the fix** a hung LLM at 10 users gives 0% errors, 13.0 req/s and a 681 ms P99 (before: 74% errors, 0.04 req/s); only the requests in flight before the breaker opens wait out the 30 s
  budget. For a refused connection the P99 stays at about 5 s because every request already in flight when the breaker opens pays two failed connects.
* **Trade-off to be aware of:** under sustained overload the breaker sheds almost everything to evidence-only answers (85% at 10 users, 88% at 25), i.e. a single local LLM
  can only back a handful of concurrent agents. Instant cited answers beat 60 s failures, but the real remedy is capacity.
* For several providers set `LLM_TIMEOUT_SECONDS` to at most the budget divided by the number of providers, otherwise a hung first provider uses the whole budget.

**Follow-up.** These are the *baseline* measurements. The optimisation work they motivated (an LLM concurrency limit, evidence-embedding cache, evidence-only answer cache; and the ideas
that were measured and rejected, such as NLI batching and removing its lock) is in [`performance.md`](performance.md), with before/after numbers and the recommended production configuration.
Still open: moving the NLI model to a shared service, API replicas behind the proxy (the Kubernetes manifests and `docker-compose.prod.yml` already run two), and a dedicated GPU for the LLM.

**Caveats.** One laptop, one API process, 25 users at most, closed loop, and runs of 60-120 s; the baseline real-LLM rows have only 25-45 requests each, so their percentiles are coarse. Tracing was off for the headline runs
and costs about 7-9% of throughput in this pipeline (14.9 vs 16.0 req/s at 1 user, 15.8 vs 17.4 at 10).

**Re-run on the final code (suite `2026-10-06-final`, `loadtest/results/2026-10-06-final/SUMMARY.md`; the tables in the README are generated from it).** After the console, adaptive-retrieval,
prompt and database changes, a shorter run of the same harness (45 s per evidence-only level, 90 s for the real-LLM levels, one level for the unreachable LLM) gave: evidence-only mode 18.4 req/s at 1 user
(p50 54 ms; the earlier suite measured 16.0 req/s and 62 ms) and about 21 req/s from 5 to 25 users, i.e. it saturates at the CPU limit of one process and latency then grows with the number of users (p50 1.2 s at 25),
with no errors at any level. With the real LLM, one user gets full answers at 0.24 req/s (p50 4.9 s); at 5 and 10 users most answers (64% and 77%) were served from evidence only because the single LLM slot was busy, with no
errors and a p95 of at most 10.4 s, which is the concurrency limit and shedding doing what they were built to do. With the LLM unreachable, 5 users got 17.6 req/s at p50 235 ms and no errors. This run was shorter than
the baseline (fewer levels, no hung-LLM scenarios), measured one process on one machine, and its commit hash is the last one before the documentation and evaluation changes (the request path is unchanged since).

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
* **Feedback-driven improvement** - `POST /api/v1/feedback` stores the rating, reasons, rejected sources, a corrected intent and edited steps against the
  audited request and the pipeline versions it rated. `GET /api/v1/quality/summary` and `/quality/report` (`app/quality/`) analyse it deterministically:
  problem intents ranked by a Wilson lower bound, most-rejected sources, weak KB articles, abstention reasons, failure patterns, and an improvement report.
  **Applying the report is deliberately manual**: a person edits or deprecates articles (ingestion bumps `corpus_version`), reviews taxonomy proposals,
  or changes a threshold behind the evaluation gate. Nothing is retrained or edited automatically, because a wrong automatic change misroutes customers silently.

## 4. Kubernetes (manifests in `k8s/base/`, applied and tested on a local kind cluster; full report in [`kubernetes.md`](kubernetes.md))

`backend` Deployment (N replicas, HPA) → Service → Ingress (TLS, auth, rate limit) · `worker` Deployment (`k8s/base/worker.yaml`:
same image, `python -m app.worker`, own NetworkPolicy, 120 s termination grace for in-flight jobs) · managed Postgres (+ PgBouncer, read replica) · managed Redis · `llm` Deployment on GPU nodes (vLLM) ·
`embedder` Deployment (TEI) · Prometheus Operator + Grafana · migration `Job` per release.

**What was actually run (2026-10-06, kind v0.30 / Kubernetes 1.34, one node on a laptop, `k8s/local/`):** the base manifests plus an overlay
with in-cluster Postgres/pgvector, Redis and Jaeger; 2 API replicas, 1 worker, 2 frontends, Ingress with ingress-nginx. A real complaint went
Ingress → frontend → API → Ollama and returned a validated, cited answer; synchronous and worker-executed ingestion worked; `/health`,
`/health/ready` and the token-protected `/metrics` behaved as designed; traces from the API and the worker reached Jaeger; with the LLM
unreachable the API kept answering from evidence only (`degraded` / `extractive`, readiness still 200) and recovered afterwards; NetworkPolicies
were checked by connection attempts; a rolling restart under traffic dropped 0 of 130 / 48 requests after a `preStop` delay was added (2 of
127 before). 22 automated checks pass (`k8s/local/e2e_test.py`), on the first deployment and again on the current code (2026-10-07, `k8s/local/results/2026-10-07/`).

**What that does not show:** multi-node behaviour, node failure or drains, a managed database, a GPU/in-cluster LLM, cert-manager and real TLS,
Prometheus/Grafana in the cluster, autoscaling under load, or backup/restore on Kubernetes. kind's network-policy agent was observed to lag and
once to fail open under memory pressure, so policy enforcement must be re-verified on a production CNI. Do not read this as a production-grade
Kubernetes deployment; it is a verified starting point.

**Defects the deployment found and that are fixed** (details and numbers in `kubernetes.md` §5): the seed data was never mounted; start-up
calls to the Hugging Face Hub hung under default-deny egress (now `HF_HUB_OFFLINE`); the migrate Job was OOM-killed at 2 Gi; FastAPI ≥ 0.142's
built-in OpenTelemetry switches itself on from `OTEL_EXPORTER_OTLP_ENDPOINT` and would have duplicated spans and exported exception text (now
disabled); **a taxonomy change on one API replica was invisible to the others and the worker until restart** (now synced every 10 s); an
unreachable (packet-dropping) LLM cost the full 30 s budget per probe (now `LLM_CONNECT_TIMEOUT_SECONDS`=5: 33 s → 13 s); 502s during rolling
restarts (now a `preStop` delay); and a wrong release order in the manifests' own comments.

**Recommended cloud configuration** (not tested): managed Postgres + Redis with TLS; secrets from a secret manager (External Secrets / Vault);
cert-manager + a real host; ≥ 3 nodes across zones; a CNI that enforces NetworkPolicy; a separate GPU pool for the LLM; HPA/KEDA on in-flight
requests and `resolveiq_job_queue_depth` rather than CPU alone; ServiceMonitors with the metrics bearer token; an OpenTelemetry Collector with
`OTEL_SAMPLE_RATIO` < 1. Note that `LLM_MAX_CONCURRENCY` is per replica, so N replicas can put N generations on one model server.

## 5. Sizing a real deployment (arithmetic on the measurements, not a measurement)

Nothing below was run. It turns the service times measured on the laptop into a first-cut deployment for an assumed contact centre, so the
order of the bottlenecks and the decisions they force are explicit. Every assumption is stated; change one and redo the multiplication.

**Assumed workload.** 2,000 agents online at the peak hour, one resolution per handled ticket plus 30% re-runs, 6 minutes average handle time:
2,000 x 1.3 / 360 s = **about 7 resolutions per second on average; size for 3x bursts, about 22 per second.** About 10% of requests abstain
(measured end to end: `overall_abstention_rate`), so about 20 per second reach the LLM before cache hits.

| tier | measured here | what 22 requests per second needs | decision it forces |
|---|---|---|---|
| **LLM** | one RTX 4070 serving `qwen3:4b-instruct` one request at a time: about 0.2 answers per second, about 5 s each (p50 4.9 s end to end); a generated answer averages 1,491 prompt and 313 completion tokens (p95 1,637 / 390, prompt budget 3,246; `e2e.tokens` in `latest.json`) | about 20 generations per second, i.e. about 30,000 prompt and 6,300 completion tokens per second. Serving them one at a time would take about 100 such GPUs, so the only viable shape is a batching server (vLLM or TGI) or a hosted model behind `openai_compat` | **Measure first:** answers per second per GPU at the p95 latency target with this prompt length on the chosen server. This number was not measured here and decides the cost of the system. Set `LLM_MAX_CONCURRENCY` per replica to the server's real parallelism divided by the number of API replicas, and keep the evidence-only fallback as the overload path (it served 64 to 77% of answers at 5 to 10 users on one GPU without errors) |
| **API (classification incl. NLI, retrieval, validation)** | one process saturates at about 21 requests per second with the NLI model on a GPU (`loadtest/results/2026-10-06-final`); the NLI model costs about 33 ms of GPU time or 1.2 s of CPU per request | 2 busy processes on GPU nodes; on CPU-only nodes about 22 x 1.2 = 26 cores busy with NLI alone | Run at least 4 API replicas (2 for load, 2 for a rolling update and a lost zone). Put the NLI model on a shared GPU inference service, or accept the base model (0.45 s on CPU, lower accuracy, re-run the gate). This is the most consequential non-LLM decision |
| **Postgres + pgvector** | dense search 1 to 2 ms at 250 tickets; 100,000 synthetic vectors: p99 2.6 ms at `ef_search` 100, 2,944 queries per second from 8 client threads at `ef_search` 40; HNSW index about 2 KB per vector | about 44 vector searches per second (tickets and articles): two orders of magnitude below the measured rate | Memory, not CPU: 1 million tickets is about 1.5 GB of float32 vectors plus about 2 GB of HNSW index, which a 16 GB instance keeps in RAM (`halfvec` halves it). Connections: replicas x `DB_POOL_MAX` (10) plus workers stays under 100 at this size; add PgBouncer before it grows. A read replica only when search competes with ingestion |
| **Redis** | query-embedding and response cache, shared rate limiter | a few hundred operations per second | managed Redis, single primary; it is not a system of record |
| **Workers** | 717 tickets per second through batched ingestion at the database layer (the embedding model excluded); drift analysis bounded at about 1,500 embeddings per run | nightly or streaming ingestion of the day's tickets, scheduled drift and discovery | 1 to 2 workers; scale on `resolveiq_job_queue_depth` (KEDA). The embedding model, not Postgres, bounds re-indexing |

**What fails first as load grows**, in order: the LLM (shed to evidence-only answers, visible as `resolveiq_llm_shed_total` and the share of `degraded`
answers), then NLI classification CPU or GPU time (latency grows with users, measured), then nothing in this design until well past this size. The
database and Redis are not near their limits at this load. **Failure isolation:** a dead or slow LLM degrades answers to evidence-only within one
breaker cool-down and keeps readiness green (measured on Kubernetes and in the load test); a dead Redis falls back to in-process caches; a
dead database fails readiness and takes the replica out of rotation; a stuck job is recovered by another worker after its heartbeat expires.
