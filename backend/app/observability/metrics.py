"""Prometheus metrics. One module so every metric the dashboards use is defined in one place."""
from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram

P = "resolveiq_"
_LAT = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)

HTTP_REQUESTS = Counter(P + "http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram(P + "http_request_latency_seconds", "HTTP latency", ["path"], buckets=_LAT)
STAGE_LATENCY = Histogram(P + "pipeline_stage_latency_seconds", "Resolve pipeline stage latency", ["stage"], buckets=_LAT)

RETRIEVAL_LATENCY = Histogram(P + "retrieval_latency_seconds", "Retrieval latency", ["strategy", "source_type"], buckets=_LAT)
RETRIEVAL_FAILURES = Counter(P + "retrieval_failures_total", "Retrieval failures", ["strategy"])
EMBEDDING_LATENCY = Histogram(P + "embedding_latency_seconds", "Embedding latency", ["kind"], buckets=_LAT)
EMBEDDING_CACHE = Counter(P + "embedding_cache_total", "Embedding cache lookups", ["result"])
RERANK_LATENCY = Histogram(P + "rerank_latency_seconds", "Cross-encoder rerank latency", buckets=_LAT)

LLM_LATENCY = Histogram(P + "llm_latency_seconds", "LLM call latency", ["provider"], buckets=_LAT)
LLM_FAILURES = Counter(P + "llm_failures_total", "LLM failures", ["provider", "reason"])
LLM_TOKENS = Counter(P + "llm_tokens_total", "LLM tokens (when the provider reports usage)", ["provider", "type"])
LLM_CIRCUIT_OPEN = Gauge(P + "llm_circuit_open", "1 when the provider circuit breaker is open", ["provider"])
LLM_IN_FLIGHT = Gauge(P + "llm_in_flight", "LLM generations currently holding a concurrency slot", ["provider"])
LLM_SHED = Counter(P + "llm_shed_total", "LLM calls not made because every concurrency slot was busy", ["provider", "priority"])
LLM_QUEUE_WAIT = Histogram(P + "llm_queue_wait_seconds", "Time spent waiting for an LLM concurrency slot", ["provider"], buckets=_LAT)

CLASSIFICATION_CONFIDENCE = Histogram(
    P + "classification_confidence", "Classifier confidence", ["dimension"], buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)
AFFECT_LATENCY = Histogram(P + "affect_latency_seconds", "Severity/sentiment model latency", buckets=_LAT)
RESOLUTIONS = Counter(P + "resolutions_total", "Resolve outcomes", ["status", "generator"])
ABSTENTIONS = Counter(P + "abstentions_total", "Abstentions / escalations due to weak evidence", ["reason"])
EVIDENCE_CONFIDENCE = Histogram(P + "evidence_confidence", "Evidence confidence score", buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
CITATION_FAILURES = Counter(P + "citation_validation_failures_total", "Citation validation problems", ["type"])
INGESTED = Counter(P + "ingestion_total", "Documents ingested", ["source_type", "result"])
EVAL_RUNS = Counter(P + "evaluation_runs_total", "Evaluation runs", ["suite"])
PII_REDACTIONS = Counter(P + "pii_redactions_total", "PII entities redacted", ["type"])
INJECTION_FLAGS = Counter(P + "prompt_injection_flagged_total", "Complaints flagged as possible prompt injection")
RATE_LIMITED = Counter(P + "rate_limited_total", "Requests rejected by the rate limiter")
CACHE = Counter(P + "cache_total", "Response cache lookups", ["cache", "result"])
CORPUS_SIZE = Gauge(P + "corpus_documents", "Active documents in the corpus", ["source_type"])
DISCOVERY_RUNS = Counter(P + "discovery_runs_total", "Emerging-class discovery runs")
DISCOVERY_PROPOSALS = Counter(P + "discovery_proposals_total", "Taxonomy proposals created", ["recommendation"])
DRIFT_JS = Gauge(P + "drift_js_divergence", "Jensen-Shannon divergence of recent vs baseline label mix", ["dimension"])
RECENT_ABSTENTION = Gauge(P + "recent_abstention_rate", "Abstention rate over the recent window (worker-published)")
RECENT_EVIDENCE = Gauge(P + "recent_mean_evidence", "Mean evidence confidence over the recent window (worker-published)")
RECENT_NEGATIVE_FEEDBACK = Gauge(P + "recent_negative_feedback_rate", "Share of 'not helpful' feedback in the recent window")
QUEUE_DEPTH = Gauge(P + "job_queue_depth", "Jobs waiting or running", ["status"])
JOBS = Counter(P + "jobs_total", "Background jobs finished by the worker", ["kind", "status"])
JOB_DURATION = Histogram(P + "job_duration_seconds", "Background job duration", ["kind"], buckets=(0.1, 0.5, 1, 5, 15, 60, 300, 900))


@contextmanager
def timed(histogram, *labels):
    start = time.perf_counter()
    try:
        yield
    finally:
        h = histogram.labels(*labels) if labels else histogram
        h.observe(time.perf_counter() - start)
