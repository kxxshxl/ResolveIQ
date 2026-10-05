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
    cache_namespace: str = "riq"   # key prefix: lets several deployments share one Redis safely

    # --- API hardening ---
    api_keys: str = ""  # comma separated; empty disables auth (dev only)
    rate_limit_per_minute: int = 120
    request_timeout_seconds: float = 60.0
    cors_origins: str = "*"

    # --- embeddings / reranker ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True
    default_retrieval_strategy: str = "dense"  # chosen on validation evidence, see docs/evaluation.md

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
    llm_total_budget_seconds: float = 30.0  # all attempts and providers for one generation; MUST stay below request_timeout_seconds,
                                            # otherwise a hung LLM turns every request into a 504 instead of the evidence-only fallback
    llm_max_retries: int = 1
    llm_max_tokens: int = 700
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
