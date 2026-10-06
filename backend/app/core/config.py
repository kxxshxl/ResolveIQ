"""Environment-driven settings. Every tunable lives here; nothing is hard-coded elsewhere."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(REPO_ROOT / ".env"), ".env"), extra="ignore", case_sensitive=False
    )

    app_env: str = "dev"
    data_dir: Path = REPO_ROOT / "data"

    # --- storage ---
    database_url: str = "postgresql://resolveiq_app:change-me-app@127.0.0.1:5433/resolveiq"
    database_admin_url: str = "postgresql://resolveiq_owner:change-me-owner@127.0.0.1:5433/resolveiq"
    db_pool_min: int = 2
    db_pool_max: int = 10
    db_statement_timeout_ms: int = 5000
    redis_url: str = "redis://127.0.0.1:6380/0"
    cache_ttl_seconds: int = 300
    cache_degraded_ttl_seconds: int = 30  # also cache evidence-only (degraded) answers this long (0 = never; capped at cache_ttl_seconds). Matches the LLM
                                          # circuit cooldown: a cached degraded answer can outlive an LLM recovery by at most this long
    cache_namespace: str = "riq"   # key prefix: lets several deployments share one Redis safely

    # --- API hardening ---
    api_keys: str = ""  # comma separated; empty disables auth (dev only)
    rate_limit_per_minute: int = 120
    request_timeout_seconds: float = 60.0
    cors_origins: str = "*"
    expose_api_docs: bool = False   # /docs, /redoc and /openapi.json are always on in dev; in production they stay off unless this is true

    # --- embeddings / reranker ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_unit_cache_size: int = 20000   # in-process LRU of embeddings for evidence text used by citation validation (0 = off); ~30 MB at 20k
    taxonomy_sync_seconds: float = 10.0         # how often each process checks Postgres for a taxonomy change made by another replica (0 = off)
    embedding_batch_size: int = 64
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True
    default_retrieval_strategy: str = "dense"  # chosen on validation evidence, see docs/evaluation.md
    hnsw_ef_search: int = 100                  # pgvector recall/latency knob (pgvector default 40). Synthetic 100k-vector experiment: recall@10 0.968 -> 0.997 for +0.1 ms p50 (docs/database.md)

    # --- adaptive retrieval: start cheap (dense) and spend more only when the evidence looks weak; thresholds chosen on the validation split ---
    adaptive_margin_ticket: float = 0.0        # dense rank-1 vs rank-2 cosine gap below which a TICKET search counts as ambiguous and escalates. 0 = never: on the tuning queries
                                               # every ticket escalation lowered MRR, because dense is the best single leg for ticket-to-ticket similarity (docs/evaluation.md)
    adaptive_margin_article: float = 0.0181    # the same for KB articles: escalate to hybrid when the top two are this close (about the 25th percentile of the tuning queries)
    adaptive_rerank_gap: float = 0.0           # after the hybrid stage: rerank only if the fused rank-1 score leads rank-2 by less than this fraction. 0 = never: on validation
                                               # the extra rung cost ~9 ms and gained nothing (docs/evaluation.md)
    adaptive_expand_kinds: str = ""            # which searches reformulate the lexical leg with terms the best candidates share ("ticket,article" to enable). Off: no gain on the tuning queries
    adaptive_expansion_terms: int = 6
    adaptive_mmr: bool = False                 # diversify the final list with MMR (off unless the benchmark shows it helps)
    adaptive_mmr_lambda: float = 0.7
    adaptive_rerank_top_n: int = 12            # bounded: fewer cross-encoder passes than the fixed hybrid_reranked pipeline

    # --- retrieval ---
    candidate_k: int = 30          # per-retriever candidates before fusion
    lexical_max_df: float = 0.4    # drop query terms present in >20% of documents (IDF pruning for Postgres FTS)
    lexical_max_terms: int = 12
    rrf_k: int = 60                # RRF damping constant
    rrf_weight_dense: float = 1.0  # weighted RRF (tuned on the validation split, see docs/evaluation.md)
    rrf_weight_lexical: float = 0.25
    rerank_blend: float = 0.5      # final = a*rerank_prob + (1-a)*normalised_rrf ; 1.0 = pure cross-encoder order
    rerank_top_n: int = 20         # candidates passed to the cross-encoder
    ticket_top_k: int = 5
    article_top_k: int = 3
    evidence_tickets: int = 3
    evidence_articles: int = 2

    # --- grounding / abstention (calibrated in docs/evaluation.md) ---
    abstain_threshold: float = 0.55
    grounding_threshold: float = 0.45  # step-vs-evidence cosine to count a step as supported
    min_grounded_ratio: float = 0.6

    # --- classification ---
    classifier_strategy: str = "ensemble"  # rules | embedding | ensemble
    classifier_llm_fallback: bool = True   # LLM zero-shot when ensemble confidence is low
    classifier_llm_threshold: float = 0.45
    knn_k: int = 10

    # --- affect model (severity + sentiment): NLI cue extraction + learned combiner, see classification/affect.py ---
    affect_enabled: bool = True
    affect_model: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    affect_artifact: str = ""      # default: <data_dir>/models/affect_v1.json
    affect_device: str = "auto"    # auto | cpu | cuda
    affect_cpu_threads: int = 4
    affect_max_tokens: int = 256

    # --- LLM ---
    llm_providers: str = "ollama"  # ordered fallback chain: ollama,openai_compat,mock
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct"
    openai_compat_base_url: str = ""
    openai_compat_api_key: str = ""
    openai_compat_model: str = ""
    llm_timeout_seconds: float = 45.0       # one attempt
    llm_connect_timeout_seconds: float = 5.0  # reaching the model server; a host that drops packets then fails in seconds, not after the whole budget
    llm_total_budget_seconds: float = 30.0  # all attempts and providers for one generation; MUST stay below request_timeout_seconds,
                                            # otherwise a hung LLM turns every request into a 504 instead of the evidence-only fallback
    llm_max_retries: int = 1
    llm_max_tokens: int = 700
    llm_temperature: float = 0.1
    llm_deterministic: bool = False         # temperature 0 and a fixed seed: the same prompt gives the same answer (evaluation, replay, regression tests)
    llm_seed: int = 7                       # used when deterministic (a request may also ask for it)
    llm_num_ctx: int = 4096                 # the model's context window as configured on the server; the prompt budget is derived from it
    llm_context_reserve_tokens: int = 150   # safety margin below the window on top of the reserved answer length
    llm_max_concurrency: int = 1            # simultaneous generations per provider (0 = unlimited). A local model serves requests one at a time,
                                            # so more in flight only lengthens every answer; see docs/performance.md
    llm_queue_wait_seconds: float = 5.0     # how long an answer may wait for a free slot before the request degrades to evidence-only
    llm_circuit_failure_threshold: int = 3
    llm_circuit_cooldown_seconds: float = 30.0

    # --- emerging-class discovery (tuned on a held-out stream, see docs/evaluation.md) ---
    discovery_window_days: int = 30
    discovery_novelty_threshold: float = 0.80   # requests with evidence confidence below this (or abstained) are candidates
    discovery_max_candidates: int = 2000
    discovery_distance_threshold: float = 0.55
    discovery_min_cluster_size: int = 4
    discovery_extend_agreement: float = 0.60
    discovery_covered_similarity: float = 0.60

    # --- drift monitoring (explainable statistical tests on live traffic; see docs/drift.md) ---
    drift_window_hours: int = 24                # the "recent" window
    drift_baseline_days: int = 14               # the baseline: this many days immediately before the recent window
    drift_alpha: float = 0.01                   # family-wise false-alarm budget per report (Bonferroni across the tests that ran)
    drift_min_requests: int = 30                # windows smaller than this are reported but never alerted on
    drift_psi_threshold: float = 0.10           # effect-size gates: a significant but tiny difference is not an alert
    drift_ks_threshold: float = 0.15
    drift_abstention_increase: float = 0.10
    drift_embedding_shift: float = 0.02
    drift_unseen_quantile: float = 0.05         # descriptive: share of recent complaints farther from the baseline than this quantile of baseline-to-baseline distances
    drift_cluster_distance: float = 0.65        # average-linkage cosine distance for grouping recent+baseline complaints into candidate new-topic clusters (looser than discovery's 0.55: see docs/drift.md)
    drift_unexplained_weight: float = 0.75      # share of the new-topic test budget for clusters the corpus does not explain (weighted Bonferroni; 0 = equal split; docs/drift.md 3.3)
    drift_max_embed_recent: int = 500           # distinct complaints embedded per window (evenly thinned above this)
    drift_max_embed_baseline: int = 1000
    drift_permutations: int = 500
    drift_analysis_hours: float = 6.0           # >0: the worker enqueues an analysis this often (0 = only on demand)
    drift_trigger_discovery: bool = True        # an unexplained new-topic cluster with no proposal enqueues a discovery run (proposals still need human review)
    drift_discovery_cooldown_hours: float = 12.0

    # --- background jobs / worker service ---
    job_execution: str = "inline"           # inline: API process runs jobs after the response (dev/tests) | queue: a worker service does
    worker_poll_seconds: float = 2.0
    worker_job_timeout_seconds: float = 1800.0
    worker_stale_seconds: float = 90.0      # running job without a heartbeat for this long is handed back to the queue
    worker_retry_backoff_seconds: float = 30.0
    worker_metrics_port: int = 9100         # 0 disables the worker's Prometheus endpoint
    discovery_schedule_hours: float = 0.0   # >0: the worker enqueues a discovery run this often

    # --- evaluation ---
    allow_mutating_eval: bool = False  # the `evolving` suite temporarily writes to the live corpus (always cleaned up)

    judge_provider: str = "ollama"
    judge_model: str = "qwen3:4b-instruct"

    metrics_token: str = ""  # when set, /metrics requires "Authorization: Bearer <token>"

    # --- tracing (OpenTelemetry; off unless OTEL_TRACES_EXPORTER=otlp|console, see docs/observability.md) ---
    otel_traces_exporter: str = "none"          # none | otlp | console
    otel_service_name: str = "resolveiq"        # the API reports as <name>-api, the worker as <name>-worker
    otel_exporter_otlp_endpoint: str = ""       # OTLP/HTTP base URL, e.g. http://localhost:4318 (empty: SDK default / standard OTEL_* env)
    otel_exporter_otlp_headers: str = ""        # "k=v,k2=v2", for hosted backends that need an API key
    otel_sample_ratio: float = 1.0              # head sampling for new traces; a sampled parent is always honoured

    def production_problems(self) -> list[str]:
        """Insecure settings that must never reach a production deployment (checked at startup when APP_ENV=production)."""
        problems = []
        if not self.api_key_set:
            problems.append("API_KEYS is empty: authentication would be disabled")
        elif any(len(k) < 24 for k in self.api_key_set):
            problems.append("every API key must be at least 24 characters")
        if self.cors_origins.strip() == "*":
            problems.append("CORS_ORIGINS must list explicit origins (not '*')")
        for name in ("database_url", "database_admin_url", "redis_url"):
            if "change-me" in getattr(self, name):
                problems.append(f"{name.upper()} still contains a placeholder password")
        if self.database_url == self.database_admin_url:
            problems.append("the API must connect with the least-privilege role, not the owner role")
        if self.rate_limit_per_minute <= 0:
            problems.append("RATE_LIMIT_PER_MINUTE must be > 0")
        if self.allow_mutating_eval:
            problems.append("ALLOW_MUTATING_EVAL must be false")
        if self.llm_total_budget_seconds >= self.request_timeout_seconds:
            problems.append("LLM_TOTAL_BUDGET_SECONDS must be below REQUEST_TIMEOUT_SECONDS, or the evidence-only fallback can never answer")
        return problems

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def provider_chain(self) -> list[str]:
        return [p.strip() for p in self.llm_providers.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
