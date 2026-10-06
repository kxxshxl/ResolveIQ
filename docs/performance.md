# Performance experiments

Driven by the load-test findings in [`production.md`](production.md#load-testing-measured-reproducible): the local LLM is the dominant bottleneck (a
single-server queue, about 0.2 generations/s), the severity/sentiment NLI model is the next one, and Postgres, Redis and the API layer are not limits.
Each candidate change was measured before it was kept; changes that did not move a number were not made. Raw data: [`loadtest/results/`](../loadtest/results)
(`2026-10-06-after-fix` = the code before this work, `2026-10-06-optimized` = after, `2026-10-06-perf-experiments` = the experiments behind the decisions). Method, hardware and caveats are those of the
load test (Intel(R) Core(TM) i9-14900HX, 31.7 GB, NVIDIA GeForce RTX 4070 Laptop GPU shared by LLM, NLI and embedding models; one API process; closed-loop users).

## Summary

| change | motivated by | result |
|---|---|---|
| **LLM concurrency limit** (`LLM_MAX_CONCURRENCY=1`, `LLM_QUEUE_WAIT_SECONDS=5`), generation outranks the classification fallback | overload storm: every request waited on the one model | at 10 users P95 30.1 s -> 9.0 s, LLM answers/s 0.075 -> 0.184 (2.4x); median latency rises (570 ms -> 5.1 s) |
| **Evidence-text embedding cache + de-duplication** in citation validation | trace: `citations.validate` embedded 104 mostly repeated texts per request | evidence-only path **17.4 -> 21.1 req/s (+22%)** at 10 users, P99 685 ms -> 591 ms |
| **Cache evidence-only answers for 30 s** (`CACHE_DEGRADED_TTL_SECONDS`) | degraded answers were never cached, so an outage paid the full pipeline cost per repeat | with 70% repeated complaints: 23.2 -> 63.3 req/s (2.7x), P50 428 ms -> 8 ms |
| NLI model: lock removal, batching across requests, padding / `inference_mode` | NLI is the next bottleneck | **no gain, not adopted** (see 2.2) |
| LLM limits above 1 | "more concurrency" | **no gain** (same LLM output, longer calls and tails), default stays 1 |

## 1. Baseline

Code at commit `c291825` (recorded as `1063ce1` before the history was rewritten; no LLM concurrency limit; LLM and breaker fixes from the previous phase in place). No LLM, 10 users: 17.4 req/s,
P50 570 ms. Real LLM, 10 users: 570 ms P50 but 30.1 s P95, because the circuit breaker repeatedly opens and
sheds 85% of requests to evidence-only answers while the requests that do wait sit in the model's queue (LLM busy 235%, i.e. several calls overlapping, only
0.075 model-generated answers/s).

## 2. Changes and measured effect

### 2.1 LLM concurrency limit (priority 1)
`ResilientLLM` now holds a semaphore per provider. A generation waits at most `LLM_QUEUE_WAIT_SECONDS` for a slot (counted inside the existing total time budget) and
otherwise degrades to the evidence-only answer. The zero-shot classification fallback has *low* priority and never queues: if the model is busy it keeps the rule/kNN labels
(in the baseline single-user run 15% of the model's calls were this fallback, not answers). Being busy is not an outage: it does not count as a breaker failure and a half-open probe is claimed before queueing.
New metrics: `llm_in_flight`, `llm_shed_total{provider,priority}`, `llm_queue_wait_seconds`.

The default was chosen by measurement, not assumption (real LLM, 10 users, 120 s each):

| LLM setting (10 users, 120 s) | req/s | P50 | P95 | P99 | LLM answers/s | LLM busy | mean LLM call | LLM-generated / evidence-only |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| no limit (previous code) | 3.29 | 570 ms | 30.1 s | 30.9 s | 0.075 | 235% | 17.5 s | 2% / 85% |
| 1 slot, no queueing | 10.06 | 895 ms | 989 ms | 1.1 s | 0.075 | 99% | 13.2 s | 1% / 89% |
| 1 slot, wait up to 5 s | 2.02 | 5.1 s | 9.7 s | 11.0 s | 0.176 | 96% | 5.5 s | 9% / 77% |
| 1 slot, wait up to 15 s | 0.74 | 15.1 s | 19.0 s | 19.7 s | 0.184 | 96% | 5.2 s | 25% / 54% |
| 2 slots, wait up to 5 s | 1.80 | 5.1 s | 13.7 s | 16.9 s | 0.184 | 191% | 10.4 s | 10% / 75% |
| 4 slots, wait up to 5 s | 1.45 | 5.1 s | 24.7 s | 25.7 s | 0.184 | 371% | 19.3 s | 13% / 72% |

* **More slots do not add LLM throughput.** 1, 2 and 4 slots all deliver about 0.18 answers/s: the model is a serial server, so extra slots only make each call longer
  (5.5 s -> 10.4 s -> 19.3 s) and the tail worse (P95 9.7 s -> 13.7 s -> 24.7 s). The default is therefore **1**.
  (A backend that really batches, such as a server started with `OLLAMA_NUM_PARALLEL=N` or vLLM, would justify N; measure it the same way.)
* **Shedding instantly is not free.** With no queueing the model produced only 0.075 answers/s and each call took 13.2 s instead of 5.5 s: the flood of evidence-only requests
  that follow competes for the same GPU (NLI mean 824 ms vs 114 ms, GPU 98% busy) and slows the model's decoding.
* **The queue window trades latency for model answers, not throughput.** 15 s gives the same 0.18 answers/s as 5 s but P95 19.0 s vs 9.7 s. The default is **5 s**, about one service time.

### 2.2 NLI / affect inference (priority 2): investigated, nothing adopted
`loadtest/profile_affect.py` (`results/2026-10-06-perf-experiments/nli_profile.txt`):
* One request is 14 premise/hypothesis pairs: tokenising takes 1.0 ms, the forward pass 33.2 ms. It is **compute-bound on the GPU**, not lock-bound.
* **Batching across requests gains at most 1.02x** (cost per text is flat from 1 to 32 texts), so an inter-request micro-batcher would add complexity for nothing.
* **Removing the lock does not help:** 0.97x with 2 threads and 0.87x with 4 (and the lock is probably what protects the shared fast tokenizer), so it stays.
* Padding to a multiple of 8 gives 0.99x and `inference_mode` 1.02x: noise.
The honest conclusion is that the 14-hypothesis DeBERTa-large pass costs about 33 ms of GPU time per request. Reducing it means fewer hypotheses, a smaller model or a second GPU, each of which changes
accuracy or hardware and needs the evaluation gate; none was done here. What *did* reduce GPU time per request was elsewhere: the embeddings below.

### 2.3 Caching (priority 3)
* **Evidence-text embeddings (new).** `citations.validate` embedded every evidence sentence once per (step, cited source) pair, so repeated corpus text was recomputed for every request.
  It now de-duplicates within a request and keeps a bounded in-process LRU (`EMBEDDING_UNIT_CACHE_SIZE`, default 20,000 entries, ~30 MB) across requests. Results are unchanged: 150 queries, 1662 step comparisons (fresh and cache-served): grounded-flag changes = 0, validation/status changes = 0, max |grounding score difference| = 0.0000 (`loadtest/verify_grounding_equivalence.py`, plus a unit test that the vectors match to 1e-4). No LLM, 10 users: de-duplication alone 17.4 -> 19.4 req/s, with the cache 21.1; the
  `generate` stage (extractive answer + validation) fell from 18.6 ms to 3.3 ms.
* **Response cache with repeated complaints (existing, now measured).** Real LLM, 70% of requests repeating one of 20 popular complaints exactly, cache warmed first: **6.7 vs 1.8 req/s
  (3.7x) with the cache off**, hit rate 71%, P50 11 ms vs 5.1 s, P95 5.3 s vs 9.6 s.
  The LLM did the same work either way (22 vs 23 calls in 120 s): the cache removes repeat complaints from the queue, it does not speed the model up. It helps only for *exact* repeats
  (after normalisation and PII redaction); near-duplicates are not served. Any ingestion or taxonomy change invalidates the whole cache (by design).
* **Evidence-only answers (new, 30 s).** They were never cached. Same repeated workload with no LLM: **63.3 vs 23.2 req/s (2.7x)**,
  P50 8 ms vs 428 ms, hit rate 70%. Cost: a cached degraded answer can outlive an LLM recovery by up to the TTL (30 s, the same as the breaker cooldown);
  `CACHE_DEGRADED_TTL_SECONDS=0` turns it off. With unique complaints it changes nothing.
* **Query-embedding cache (existing).** Visible only as a small effect: the repeated workload without the degraded cache ran +10% faster than unique complaints
  (23.2 vs 21.1 req/s), consistent with ~7 ms saved on 70% of requests.

## 3. Benchmark: before vs after

**Evidence-only path (no LLM), 60 s per run.**

| users | req/s before | req/s after | P50 before | P50 after | P95 before | P95 after | P99 before | P99 after | errors | NLI mean before | NLI mean after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.0 | **18.6** (+17%) | 62 ms | 53 ms | 75 ms | 62 ms | 78 ms | 66 ms | 0% | 35 ms | 35 ms |
| 5 | 17.6 | **21.0** (+19%) | 283 ms | 237 ms | 312 ms | 258 ms | 354 ms | 302 ms | 0% | 245 ms | 215 ms |
| 10 | 17.4 | **21.1** (+22%) | 570 ms | 472 ms | 627 ms | 521 ms | 685 ms | 591 ms | 0% | 533 ms | 452 ms |
| 25 | 17.4 | **20.8** (+20%) | 1.4 s | 1.2 s | 1.5 s | 1.3 s | 1.6 s | 1.3 s | 0% | 1.4 s | 1.2 s |

**Full pipeline with the real LLM, 120 s per run** (`*` = fewer than 100 requests; "LLM busy" counts overlapping calls, so above 100% means several calls were in flight at once; "LLM answers/s" excludes cache hits).

| users | | req/s | P50 | P95 | P99 | errors | answers: LLM / evidence-only / abstained | LLM answers/s | LLM busy | NLI mean |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | before | 0.23 | 5.2 s | 6.1 s | 6.7 s* | 0% | 81% / 0% / 19% | 0.184 | 108% | 37 ms |
| 1 | **after** | 0.23 | 4.8 s | 6.4 s | 7.1 s* | 0% | 82% / 0% / 18% | 0.192 | 97% | 47 ms |
| 5 | before | 0.23 | 25.1 s | 29.2 s | 29.9 s* | 0% | 79% / 0% / 21% | 0.184 | 547% | 69 ms |
| 5 | **after** | 0.94 | 5.1 s | 10.2 s | 10.5 s | 0% | 20% / 66% / 13% | 0.192 | 98% | 87 ms |
| 10 | before | 3.29 | 570 ms | 30.1 s | 30.9 s | 0% | 2% / 85% / 12% | 0.075 | 235% | 382 ms |
| 10 | **after** | 2.01 | 5.1 s | 9.0 s | 11.2 s | 0% | 10% / 76% / 15% | 0.184 | 97% | 120 ms |
| 25 | before | 7.02 | 1.7 s | 30.4 s | 32.5 s | 0% | 1% / 88% / 11% | 0.050 | 123% | 1.5 s |
| 25 | **after** | 5.04 | 5.1 s | 6.2 s | 12.1 s | 0% | 3% / 86% / 11% | 0.134 | 99% | 172 ms |

**Failure modes (resilience from the previous phase must survive).**

| scenario | | req/s | P50 | P95 | P99 | errors |
|---|---|---:|---:|---:|---:|---:|
| LLM down, 10 users | before | 15.5 | 591 ms | 677 ms | 4.7 s | 0% |
| LLM down, 10 users | **after** | 19.9 | 462 ms | 505 ms | 582 ms | 0% |
| LLM down, 25 users | before | 15.8 | 1.5 s | 1.6 s | 5.6 s | 0% |
| LLM down, 25 users | **after** | 19.6 | 1.2 s | 1.3 s | 5.9 s | 0% |
| LLM hung, 10 users | before | 13.0 | 561 ms | 626 ms | 681 ms | 0% |
| LLM hung, 10 users | **after** | 16.6 | 434 ms | 494 ms | 5.1 s | 0% |

What the tables say, without rounding in our favour:
* The non-LLM path is **+22% faster at 10 users** and its latency is -17% at P50, everywhere from 1 to 25 users, with no errors.
* With the real LLM and a single user nothing changes (as intended). From 5 users up the **tail is bounded**: P95 29.2 s -> 10.2 s (5 users), 30.1 s -> 9.0 s (10), 30.4 s -> 6.2 s (25),
  the model runs at 97% utilisation with no overlapping calls instead of 235%, and model-generated answers per second rise at 10 and 25 users (0.075 -> 0.184, 0.050 -> 0.134).
* **It is not faster everywhere.** At 10 and 25 users the *median* latency is higher (570 ms -> 5.1 s, 1.7 s -> 5.1 s) and completed requests per second are lower
  (3.29 -> 2.01, 7.02 -> 5.04), because requests now wait up to 5 s for the model instead of being turned away at once by the breaker.
  At 5 users the share of model-generated answers drops (79% -> 20%) in exchange for a 5x lower median; the total model output is unchanged (0.184 vs 0.192 answers/s) because the model was already saturated.
* The model never gets faster: **about 0.19 answers/s is the ceiling** of this GPU (best run) however the traffic is shaped.
* **Two regressions in my own first version of the limiter were found by this benchmark and fixed**, in a hung-LLM run (10 users; baseline 13.0 req/s):
  (1) requests queued for a slot while the breaker's half-open probe held it: 6.8 req/s, P99 5.6 s, so the breaker is now consulted first;
  (2) with one slot the failures that open the breaker arrive one per time budget, so three of them took ~90 s while every request waited out the queue window: 8.4 req/s,
  P95 5.1 s, so a timeout now opens the breaker at once. Both runs are kept under `2026-10-06-perf-experiments/llm_hang/`. Final: **16.6 req/s, P95 494 ms**.
* **One number got worse:** the hung-LLM P99 is 5.1 s against 681 ms before. The few requests in flight when the model hangs now wait the 5 s queue window before degrading, where before the
  ten in-flight requests all waited out the 30 s budget (too few to reach P99, which is why that P99 looked good). The LLM-down P99 (about 5 s, the first wave of refused connections) is unchanged.

**How far to trust the numbers.** At the saturated levels (5, 10 and 25 users the same configuration is limited by the same resource) throughput varies by 1.3% before and 1.5% after, so the 17-22% gain on the
non-LLM path is well outside the noise. The baseline was measured earlier in the session than the optimized runs, and conditions drift: three consecutive runs taken right after
15 minutes of heavy LLM load ran at 8.4, 8.0 and 8.2 req/s with the GPU only 29% busy, and the same
`llm_down` configuration re-run on the idle machine gave 19.9 req/s. The cause was not established (the GPU's cumulative thermal and power-slowdown counters were non-zero, which fits but does not prove it).
Those runs are kept in `2026-10-06-perf-experiments/noisy-environment/` and excluded from the tables; the failure-mode rows above are from re-runs on the idle machine. Treat differences under ~10% as unproven, and note that the
real-LLM rows have 25-240 requests each. The limiter sweep in 2.1 ran before two later breaker refinements (breaker consulted first, timeouts open it at once), which do not affect a healthy-LLM scenario.

## 4. Trade-offs
* **Latency distribution vs model answers.** `LLM_QUEUE_WAIT_SECONDS` decides whether an agent waits for the model (more LLM answers, higher median) or gets the evidence-only answer at once (0 s: lowest median, but the model is starved by GPU contention).
* **Evidence-only answers cached 30 s** can be a little stale after an LLM recovery; they are never cached when the knob is 0.
* **In-process embedding cache** is per API replica (cold after a restart, ~30 MB at the default size).
* A limit of 1 assumes a serial backend. Raising it for a backend that batches is a configuration decision that must be measured, not assumed.

## 5. Recommended production configuration
| setting | value | why |
|---|---|---|
| `LLM_MAX_CONCURRENCY` | 1 per replica for a local single-GPU Ollama; the backend's real parallelism otherwise | measured: more slots lengthen calls without adding output |
| `LLM_QUEUE_WAIT_SECONDS` | 5 (about one service time); 0 only if you prefer instant evidence-only answers over model answers | latency vs model-answer share |
| `LLM_TOTAL_BUDGET_SECONDS` / `REQUEST_TIMEOUT_SECONDS` | 30 / 60 (budget must stay below the request timeout) | previous phase |
| `CACHE_TTL_SECONDS` / `CACHE_DEGRADED_TTL_SECONDS` | 300 / 30 | response cache; short for degraded answers so recovery shows quickly |
| `EMBEDDING_UNIT_CACHE_SIZE` | 20000 | evidence sentences recur across requests |
| replicas | at least 2 API replicas; the LLM on its own GPU or hosted | NLI and LLM share one GPU here and slow each other |
| monitor | `llm_shed_total`, `llm_queue_wait_seconds`, `llm_in_flight`, `llm_circuit_open`, share of `degraded` answers | alert when most answers are evidence-only |

## 6. Remaining bottleneck
1. **The model:** about 0.19 answers/s on this GPU, shared with the NLI and embedding models; busy 97% under load. Only more or separate hardware, a faster/smaller model, shorter answers
   (needs the RAG evaluation), or a batching server raises it. In the baseline single-user run 15% of the model's calls were the classification fallback; they are now skipped whenever it is busy.
2. **The NLI model**, about 33 ms of GPU time per request, caps the no-LLM path near 21 req/s here.
3. Not measured: more than one API replica, a dedicated LLM tier, or a larger corpus.

## 7. Reproduce
```bash
python loadtest/run_loadtest.py --suite mytest --scenarios no_llm --levels 1 5 10 25 --duration 60
python loadtest/run_loadtest.py --suite mytest --scenarios llm --levels 1 5 10 25 --duration 120
python loadtest/run_loadtest.py --suite mytest --scenarios llm --levels 10 --duration 120 --env LLM_MAX_CONCURRENCY=2 --tag n2-q5   # a sweep point
python loadtest/run_loadtest.py --suite mytest --scenarios llm_hot --levels 10 --duration 120 --env CACHE_TTL_SECONDS=1 --tag cache-off
python loadtest/profile_affect.py
```
