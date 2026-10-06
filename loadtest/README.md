# Load testing ResolveIQ

Measures `POST /api/v1/resolve` under concurrent support-agent traffic with [Locust](https://locust.io), and records throughput, latency
percentiles, error and timeout rates, per-stage timings, CPU/GPU use and what the pipeline decided. Results and analysis:
[`results/2026-10-06/SUMMARY.md`](results/2026-10-06/SUMMARY.md) and the "Load testing" section of [`docs/production.md`](../docs/production.md).

## Run it

Prerequisites: the dev stack's Postgres and Redis (`docker compose up -d db redis`), a seeded-capable backend environment
(`pip install -r loadtest/requirements.txt`), and, for the `llm` scenarios, Ollama with `qwen3:4b-instruct` on `localhost:11434`.
Run from the repository root and keep the machine otherwise idle (the API, the GPU and the load generator share it).

```bash
# one scenario, four concurrency levels, 60 s each
python loadtest/run_loadtest.py --suite 2026-10-06 --scenarios no_llm --levels 1 5 10 25 --duration 60

# the full-pipeline run with the real LLM (longer, because one answer takes seconds)
python loadtest/run_loadtest.py --suite 2026-10-06 --scenarios llm --levels 1 5 10 25 --duration 120

# fallback behaviour when the LLM is down or hangs
python loadtest/run_loadtest.py --suite 2026-10-06 --scenarios llm_down --levels 1 5 10 25 --duration 60
python loadtest/run_loadtest.py --suite 2026-10-06 --scenarios llm_hang --levels 1 5 10 --duration 180

# a traced run (starts the optional Jaeger container) to see where the time goes inside the pipeline
python loadtest/run_loadtest.py --suite 2026-10-06 --scenarios no_llm llm --levels 1 10 --duration 60 --trace --tag traced

python loadtest/summarize.py 2026-10-06        # writes results/2026-10-06/SUMMARY.md from the raw files
```

Scenarios (`run_loadtest.py -h`): `llm`, `no_llm` (evidence-only, no LLM provider), `no_llm_no_affect` (ablation without the severity/sentiment
NLI model), `no_llm_warm` (repeated complaints, response cache can hit), `llm_down` (connection refused), `llm_hang` (the model accepts the
connection and never answers; `loadtest/stub_llm.py`), `llm_hang_tuned` (same, with `LLM_TIMEOUT_SECONDS=15`), `mock_llm_warm` (instant mock model, repeated
complaints), and the repeated-complaint workloads `llm_hot`, `no_llm_hot`, `llm_down_hot` (70% of requests repeat one of 20 popular complaints exactly; the
cache is warmed first). Any run can override the API's settings with `--env KEY=VALUE` (repeatable; use `--tag` to keep the folders apart), for example
`--env LLM_MAX_CONCURRENCY=2` or `--env CACHE_TTL_SECONDS=1`. `loadtest/profile_affect.py` profiles the NLI model in isolation. The optimisation
experiments built on this are in [`docs/performance.md`](../docs/performance.md).

## What the harness guarantees

* **No contamination.** Every invocation creates the throw-away database `resolveiq_loadtest`, migrates and seeds it from scratch (the same 250
  tickets / 26 articles as the evaluation corpus), uses Redis DB 2 with a per-run cache namespace, and drops/cleans both at the end. The
  development database, its taxonomy, and the evaluation data are never connected to. (The code refuses to operate on any other database name.)
* **Fresh server per run.** The API is restarted for every (scenario, concurrency) pair, so the LLM circuit breaker, caches and model state of
  one run cannot influence the next. A few warm-up requests per run are excluded from the measurement (none for the `llm_hang*` scenarios).
* **Rate limiter off, tracing off.** The default limiter (120 requests/min) would answer 429 above 2 req/s; tracing is off so that it cannot
  affect the numbers (`--trace` runs are kept in separate `-traced` folders).
* **Closed-loop load.** N users each send the next request as soon as the previous one answers (no think time), so N = requests in flight. This
  measures capacity and queueing, not an arrival schedule. Request mix: 275 in-domain complaints from `data/eval/{queries,gold_v2,gold_blind}.jsonl`
  plus 5% out-of-domain questions from `ood.jsonl`, shuffled with a fixed seed; in `cold` cache mode every complaint carries a unique case
  reference so no response/embedding cache can hit.
* **Nothing is estimated.** Percentiles are computed from the individual request timings (`requests.jsonl.gz`) with numpy; Locust's own CSVs are
  kept next to them. Server-side numbers come from the API's responses and a before/after diff of its Prometheus `/metrics`.

## What is in a run folder (`results/<suite>/<scenario>/c<N>/`)

| File | Content |
|---|---|
| `summary.json` | everything below, machine-readable: config, throughput, latency percentiles (successful and all), error/timeout counts by kind, outcome mix, stage timings, Prometheus deltas, resource use |
| `requests.jsonl.gz` | one line per request: client latency, HTTP status, outcome, generator, cached, server stage timings |
| `locust_stats*.csv`, `locust_failures.csv` | Locust's standard CSV output |
| `resources.csv` | per second: API CPU (cores), RSS, load-generator CPU, system CPU, GPU utilisation and memory |
| `server_metrics.prom.{before,after}.json.gz` | raw `/metrics` snapshots |
| `api.log` | last 60 lines of the API log (the full log is one JSON line per request) |

## Caveats

* One API process on one laptop (see `environment.json`), with the LLM and the severity/sentiment model sharing one 8 GB GPU. Production runs two
  or more API replicas and would normally put the LLM on its own hardware, so absolute numbers do not transfer; the shape of the curves and the
  ordering of the bottlenecks do.
* Fewer than 100 completed requests make a P99 meaningless (it is then the maximum); the summary marks those rows.
* Requests still in flight when a run ends are not counted, which slightly understates throughput for runs where one request takes tens of seconds.
