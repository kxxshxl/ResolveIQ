# LLM engineering

The language model is the least trustworthy and most expensive part of the pipeline, so the code around it is built to bound it, to record exactly what it was given and
what it did, and to keep answering when it is not there. Nothing here depends on a particular model: the default is `qwen3:4b-instruct` on Ollama, and any
OpenAI-compatible endpoint works through the same interface.

## What goes in: a versioned, bounded, sanitised prompt (`rag/prompts.py`)

* **Versioning.** `PROMPT_VERSION` (currently `grounded-v3`) is bumped by hand when the instructions change; `PROMPT_HASH` is a hash of the system prompt itself, so the identity of the
  instructions changes even if somebody forgets the version. Both are stored with every resolution, shown in the console, and part of the response-cache key.
* **Bounded context.** The budget is `LLM_NUM_CTX - LLM_MAX_TOKENS - LLM_CONTEXT_RESERVE_TOKENS` tokens (4 characters per token estimate; it is a budget guard, not a billing figure). Degradation order:
  collapse evidence blocks whose resolution steps duplicate another block's (always; the later block becomes a pointer, and stays citable), then clip long text in three levels, then drop the lowest-ranked block.
  At least one block is always kept. What was collapsed, clipped or dropped is recorded (`collapsed_duplicates`, `clip_level`, `dropped_for_budget`) and the validator checks citations against **what the model actually saw**, not
  against what was retrieved.
* **Sanitised evidence.** Retrieved text is untrusted: forged `[ID]` headers are neutralised, role tokens removed, instruction-like sentences replaced (details and tests in [`security.md`](security.md)). The complaint is
  quoted as JSON data. The model returns one JSON object (summary, steps with citations, escalate flag and reason, uncertainty); invalid JSON is a generation error that falls back, not a crash.

## What it did: structured generation metadata

Every resolution has a `provenance` object and a `generate` stage in its trace that record: provider and model, prompt version and hash, temperature, seed, whether the run was deterministic, `num_ctx`, `max_tokens`,
the estimated prompt tokens against the budget, the prompt and completion token counts the provider reported, the LLM latency, which evidence ids were in the prompt, and why the answer was a fallback when it was.
The same facts feed Prometheus (`resolveiq_llm_latency_seconds`, `resolveiq_llm_tokens_total`, failures by reason) and the OpenTelemetry `llm.generate` / `llm.call` spans (token usage, temperature, attempt, provider chain).
No prompt text or model output is ever put in a span or a metric label.

## Reproducibility

A resolution can be reproduced when these are fixed, and all of them are recorded: model, prompt version and hash, taxonomy version, corpus version, embedding model, retrieval strategy and its configuration, and the
generation settings. `LLM_DETERMINISTIC=true` (or `"deterministic": true` on a request, which the console's *Reproducible* switch sets) uses temperature 0 and a fixed seed (`LLM_SEED`). Case replay uses it by default so that a
difference in the diff is a change in the pipeline, not sampling noise. Two caveats stated plainly: the corpus and taxonomy versions must be the same for the evidence to be the same (the replay diff reports when they are not), and
a deterministic seed makes one server reproducible, not two different builds of the model runtime. The response cache key contains the prompt hash, the model and the mode, so changing any of them cannot serve an old answer.

## Keeping it available (unchanged by this phase, covered by tests and the load test)

* **One total time budget** per generation (`LLM_TOTAL_BUDGET_SECONDS`, 30 s by default, below the 60 s request timeout), so retries and a provider chain cannot add up to a gateway timeout.
  Every attempt is wrapped in `asyncio.wait_for`, so the budget is a hard bound and not only an HTTP timeout: httpx timeouts are per operation (connect, the gap
  between reads), and a server that trickles bytes, or a provider that ignores its timeout, used to outlive the budget (a 2 s budget held for 10 s in the regression test).
* **A concurrency limit** per provider (`LLM_MAX_CONCURRENCY`, 1 by default for a single local GPU) with a short queue; beyond it the call is shed and the answer degrades to evidence-only instead of queueing without bound.
* **A circuit breaker** per provider (opens after consecutive failures or a timeout, half-opens after a cooldown with a single probe), exposed as a gauge and in `GET /system/status`.
  The probe slot is released however the probing request ends, including when it is cancelled while queueing for a generation slot; before that fix such a
  cancellation left the breaker refusing every call until the process restarted.
* **Untrusted model output**: any JSON the model returns that does not match the schema (including a step that is a bare number) is a generation error that
  falls back to the evidence-only answer, never an HTTP 500.
* **A provider abstraction**: `ollama`, `openai_compat` (OpenAI, vLLM, LM Studio, Ollama `/v1`), `mock`; ordered chain; the last resort is deterministic evidence-only extraction (`degraded`), which is also what runs when no provider is configured.
* **Abstention before generation**: weak evidence never reaches the model.

## What was measured, and what was not

The effect of the model on answer quality (faithfulness, citation precision, step recall, correct resolution) is in the generated results in the [README](../README.md#9-evaluation) and [`evaluation.md`](evaluation.md), with the
evidence-only baseline next to it. Latency under load, with and without the model, is in [`performance.md`](performance.md) and [`production.md`](production.md). Not measured: any model other than Qwen3-4B (the OpenAI-compatible
path was exercised against the same model through Ollama's `/v1`), cost per request, and quality at other temperatures. The LLM judge is the same 4B model grading itself and is reported separately for that reason.
