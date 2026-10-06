# Observability: logs, metrics and traces

| Signal | Tool | Always on? | Answers |
|---|---|---|---|
| Logs | JSON on stdout, each line carries `trace_id` | yes | what happened, in order, for one request |
| Metrics | Prometheus (`/metrics`, worker `:9100`) + Grafana | yes | how is the system doing over time |
| **Traces** | **OpenTelemetry (OTLP)** | **opt-in** | **where did the time go in this one request** |

Metrics say that p95 got worse; a trace says whether the extra 400 ms was the NLI model, the vector search, a retry against the LLM
or a cache miss. Tracing is off by default and costs nothing when off (see [Overhead](#overhead)).

## Viewing traces in two minutes

```bash
# 1. a trace viewer (optional, one container, in-memory: traces vanish on restart)
docker compose --profile tracing up -d jaeger          # UI on http://127.0.0.1:16686

# 2. run the API (and the worker, if you use queue mode) with tracing on
cd backend
export OTEL_TRACES_EXPORTER=otlp OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
python scripts/dev_server.py                           # service "resolveiq-api"
JOB_EXECUTION=queue python -m app.worker               # service "resolveiq-worker"

# 3. send a request, with an id you can search for later
curl -H 'X-Request-Id: demo-001' -H 'Content-Type: application/json' \
     -d '{"complaint":"My broadband drops every evening around 8."}' http://127.0.0.1:8100/api/v1/resolve
```

Open Jaeger, pick service `resolveiq-api`, and either click a recent trace or search the tag **`resolveiq.trace_id=demo-001`**.
With the full Docker stack the same switches go through `.env` (`OTEL_TRACES_EXPORTER=otlp`); the compose file already points the
containers at `http://jaeger:4318`.

No collector or UI at hand? `OTEL_TRACES_EXPORTER=console` prints one JSON line per span to stdout. Any OTLP/HTTP backend works
(Tempo, Honeycomb, Grafana Cloud, an OpenTelemetry Collector): set `OTEL_EXPORTER_OTLP_ENDPOINT` and, if the backend needs a key,
`OTEL_EXPORTER_OTLP_HEADERS=x-api-key=...`.

## What a resolve trace looks like

Captured from a real request against the local stack (Postgres, Redis, Ollama `qwen3:4b-instruct`), 37 spans, 5.7 s in total:

```
POST /api/v1/resolve                      5704 ms  http.status_code=200      <- FastAPI instrumentation
  INCRBY, EXPIRE                                                            <- rate limiter (Redis)
  resolve                                 5698 ms  status=resolved intent=broadband_disconnection
    resolve.preprocess                       0 ms  PII redaction, injection check
    resolve.cache                            2 ms  cache_hit=False           <- + Postgres SELECT, Redis GET
    embed.query                              7 ms                            <- + embed.encode, Redis GET/SET
    retrieval.search  (tickets)             11 ms  returned=5
      db.dense_search -> SELECT              10 ms
    classify                               111 ms  intent=...  confidences
      classify.affect                      105 ms  severity / sentiment NLI model
      db.dense_search ...                          kNN classifier
    retrieval.search  (articles)             2 ms  returned=3
    resolve.evidence                         2 ms  sufficient, confidence, top similarities
    resolve.generate                      5566 ms
      llm.generate                        5545 ms  gen_ai.system=ollama      <- the provider chain
        llm.call                          5545 ms  attempt=0
          POST                            5544 ms  http.status_code=200      <- httpx instrumentation (Ollama)
      citations.validate                     21 ms  valid, coverage, grounded_ratio
        embed.encode                         20 ms  batch_size=104
    resolve.persist                          8 ms                            <- audit-trail insert
```

Reading it: 97% of this request is the LLM call; classification (105 ms of it the affect model) and retrieval overlap and are
cheap. That is exactly the "where did the time go" answer metrics cannot give.

### Spans

Attribute names below omit their `resolveiq.` prefix (for example `latency_ms.total` is `resolveiq.latency_ms.total`); `gen_ai.*` and `db.*` follow the OpenTelemetry conventions.

| Span | From | Useful attributes |
|---|---|---|
| `POST /api/v1/...` | FastAPI instrumentation (health and metrics excluded) | HTTP route/status, **`resolveiq.trace_id`** |
| `resolve` | `rag/pipeline.py` | `request_id`, `status`, `confidence`, `generator`, `cached`, `escalate`, `intent/product/severity/sentiment`, `evidence.*`, `validation.*`, `latency_ms.<stage>` (the same timings the API returns) |
| `resolve.preprocess` | pipeline | `pii_redactions`, `injection_flagged` |
| `resolve.cache` | pipeline | `cache_hit` |
| `embed.query`, `embed.encode` | `services/embedding.py` | `embed.model`, `embed.cache_hit`, `batch_size` |
| `classify`, `classify.affect`, `classify.llm_fallback` | `classification/` | `classify.strategy`, `confidence.<dimension>`, `taxonomy_version`, weak dimensions |
| `retrieval.search`, `retrieval.rerank` | `retrieval/service.py` | `retrieval.kind/strategy/k`, `returned`, `top_score`, `rerank.candidates`, `rerank.top_probability` |
| `resolve.evidence` | pipeline | `evidence.sufficient`, `confidence`, top ticket/article similarity |
| `resolve.generate`, `generate.extractive` | pipeline / generator | `generator`, `generation_status`, `abstain_reason` |
| `llm.generate`, `llm.call` | `services/llm/base.py` | `gen_ai.*` (system, model, token usage, temperature), `llm.attempt`, `llm.provider_chain`; failed attempts are error spans with the exception, an open circuit is an event |
| `citations.validate` | `rag/citations.py` | `validation.valid`, `citation_coverage`, `grounded_ratio`, invalid/uncited/unsupported counts |
| `resolve.persist`, `db.*` | pipeline / repository | `db.dense_search`, `db.lexical_search`, `db.save_request`, `db.upsert_*` plus the SQL spans below |
| `SELECT`, `INSERT`, ... | psycopg instrumentation | `db.statement` (SQL text, never parameter values) |
| `GET`, `SET`, `INCRBY`, ... | Redis instrumentation | arguments are sanitised (`GET ?`) |
| `POST` (to the LLM) | httpx instrumentation, LLM client only | method, URL, status |
| `ingest.ticket`, `ingest.article`, `ingest.batch`, `ingest.reindex` | `ingestion/pipeline.py` | `ingest.created/updated/failed`, `batch_size`, `corpus_version`, `embedding_ms` |
| `worker.job`, `job.<kind>`, `discovery.run` | `worker.py`, `jobs.py`, `discovery/service.py` | `job_id`, `job_kind`, `job_attempt`, discovery counts |
| `drift.analyze` | `drift/service.py` | `job_id`, `drift.alerts`, `drift.emerging_clusters` |
| `retrieval.adaptive.decision` (span event on `retrieval.search`) | `retrieval/service.py` | `retrieval.kind`, `adaptive.stage` (dense, hybrid, rerank), `adaptive.signal`, `adaptive.value`, `adaptive.action` (stop or escalate) |
| `cases.get`, `cases.replay` | `cases/service.py` | `request_id`, `strategy`, `generate`, `deterministic`, `replay.reproduced`, `replay.status` |
| `lab.compare` | `lab/service.py` | `complaint_chars` (a length, never the text), `retrieval.k` |
| `clusters.compute` | `clusters/service.py` | `window_days`, `cluster.min_size`, `cluster.distance`, `cluster.count`, `cluster.distinct` |
| `feedback.save`, `feedback.analyze` | `quality/service.py` | `request_id`, `feedback.rating`, `window_days` |
| `db.health` | `system/health.py` | `db.findings`, `db.status` |

### Metrics added with the console and adaptive retrieval

| Metric | Labels | Meaning |
|---|---|---|
| `resolveiq_adaptive_retrieval_stages_total` | `kind`, `stage`, `action` | which rung of the adaptive ladder ran and whether it stopped or escalated; the ratio shows how often the cheap path was enough |
| `resolveiq_feedback_total` | `rating` | agent feedback received |
| `resolveiq_case_replays_total` | `outcome` | replays of stored cases (`identical`, `changed`, `error`) |
| `resolveiq_recurring_clusters_seconds` | | time to compute the recurring-complaint groups |
| `resolveiq_llm_in_flight`, `resolveiq_llm_shed_total`, `resolveiq_llm_queue_wait_seconds` | `provider` | LLM concurrency slots in use, calls refused because every slot was busy, time waiting for a slot |

Everything else keeps its name and meaning (`resolveiq_pipeline_stage_latency_seconds`, `resolveiq_retrieval_latency_seconds`, the drift gauges and alerts, ...). The complete list is `backend/app/observability/metrics.py`.
No label ever carries complaint text: labels are statuses, strategies, stages, intents and reasons.

## Ids: nothing about the existing ones changed

* `X-Request-Id` in, `X-Trace-Id` out, `trace_id` in error bodies and in `resolution_requests.trace_id`, and the `trace_id` field of
  every log line behave exactly as before.
* The same id is stored on the HTTP span as **`resolveiq.trace_id`**, so a trace is found from a log line, a support ticket or a row of
  `resolution_requests` by searching that tag. The response's `request_id` is stored on the `resolve` span as `resolveiq.request_id`.
* While tracing is on, log lines additionally carry **`otel_trace_id`** and **`otel_span_id`**, so a log line opens its trace and back.
* An incoming W3C `traceparent` header is honoured: a trace started by an upstream gateway continues through ResolveIQ.

## Jobs and the worker

`POST /ingest/tickets/batch`, `/taxonomy/discover` and `/evaluate` return 202 and run later, possibly in the worker process. The enqueuing
request's trace context is stored in the job row (`_trace`, stripped before the handler runs). The worker starts its **own** trace
(`worker.job` > `job.<kind>` > `ingest.batch` > ...) with a **span link** back to the request, rather than making the job a child, so a job that
waits for a minute does not stretch the request's trace. In Jaeger the link shows as "References". In inline mode (`JOB_EXECUTION=inline`)
the job simply nests under the request.

Background activity that is not part of any request or job (connection-pool health checks, the worker's queue polling and heartbeat, readiness
probes) is **dropped** by a sampler rule instead of producing thousands of one-span traces.

## Configuration

All from environment variables or `.env`; nothing is hard-coded.

| Variable | Default | Meaning |
|---|---|---|
| `OTEL_TRACES_EXPORTER` | `none` | `none` (off), `otlp`, or `console`. Anything else is rejected at startup |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | empty (SDK default `http://localhost:4318`) | OTLP/HTTP base URL; `/v1/traces` is appended |
| `OTEL_EXPORTER_OTLP_HEADERS` | empty | `k=v,k2=v2`, for hosted backends |
| `OTEL_SERVICE_NAME` | `resolveiq` | reported as `<name>-api` and `<name>-worker` |
| `OTEL_SAMPLE_RATIO` | `1.0` | head sampling for new traces; an upstream `traceparent` that is already sampled is honoured |

The standard `OTEL_EXPORTER_OTLP_*` and `OTEL_BSP_*` variables are also read by the SDK itself when set in the process environment.

## Privacy

Spans carry labels, counts, sizes, scores and timings. They never carry the complaint, the redacted complaint, prompts, model output or
retrieved text; Redis arguments are sanitised and SQL parameters are not recorded. `tests/test_tracing.py` asserts this by planting a marker
string in a complaint and searching every attribute and event of every span for it.

## Overhead

With tracing off (the default) the SDK is not imported, `span()` returns a null context and `@traced` calls the function directly (its attribute
extractors are not even evaluated). With tracing on, each request creates about 30 to 40 spans. Measured with the load test (evidence-only pipeline, 100% sampling, Jaeger on the same machine):
14.9 vs 16.0 req/s at 1 user (-7%) and 15.8 vs 17.4 req/s at 10 users (-9%); see [`production.md`](production.md#load-testing-measured-reproducible). Use `OTEL_SAMPLE_RATIO` to bound it. Spans are exported in batches off the request path (OTLP) and flushed at shutdown.

## Tests

`backend/tests/test_tracing.py` (17 tests) builds the app with an in-memory span exporter, so no collector is needed:

* the whole stage list appears in one trace and nests as documented; reranking, cache hits, ingestion and job links are covered;
* span attributes match the API response (status, intent, confidence, every stage latency);
* `X-Request-Id` / `X-Trace-Id` / error-body `trace_id` are unchanged, and an incoming `traceparent` is continued;
* health and metrics produce no spans; a failing LLM produces error spans while the API still answers;
* no complaint text in any span; logs carry the OTel ids only inside a span;
* off means off: no spans, no attribute evaluation, unknown exporter names rejected.

```bash
cd backend && python -m pytest tests/test_tracing.py -q      # DB tests need `docker compose up -d db redis`
```

## Limits

* Metrics stay in Prometheus; OpenTelemetry is used for traces only.
* The bundled Jaeger keeps traces in memory. For retention use a real backend (Tempo, Jaeger with storage) through the same OTLP endpoint.
* The frontend is not instrumented; a browser-originated trace would start at the API (or at the gateway, via `traceparent`).
* Kubernetes manifests do not yet set the `OTEL_*` variables.

## Kubernetes

The ConfigMap carries the same switches (`OTEL_TRACES_EXPORTER` = `none` by default, `OTEL_SERVICE_NAME`, `OTEL_SAMPLE_RATIO`,
`OTEL_EXPORTER_OTLP_ENDPOINT`); credentials for a hosted backend (`OTEL_EXPORTER_OTLP_HEADERS`) belong in the Secret. The local overlay
(`k8s/local/`) turns tracing on and sends it to an in-cluster Jaeger (`kubectl -n resolveiq port-forward svc/jaeger 16686:16686`); tested: the
API and the worker both appear as services, and one resolve is one tree of 34-38 spans with a single server span (find it by the
`resolveiq.trace_id` tag, which equals the `X-Trace-Id` header and the nginx request id).

Gotcha found there: FastAPI >= 0.142 switches on its **own** OpenTelemetry (traces, metrics, and logs including exception text) whenever
`OTEL_EXPORTER_OTLP_ENDPOINT` is set in the process environment, which is how Kubernetes sets it. The application disables that native
integration (`app/main.py`) so there is exactly one span tree, nothing is exported to the metrics/logs OTLP paths, and exception messages stay
in the pod. `/metrics` stays token-protected: scrape it with a ServiceMonitor that reads the bearer token from a Secret (annotation-based
scraping cannot send a token).
