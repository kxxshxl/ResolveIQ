import type { Classification, ResolveResponse } from "./types";

// ---------------------------------------------------------------- provenance, trace and lineage (the evidence graph)
export interface StageRecord { name: string; status: "ok" | "skipped" | "degraded" | "error"; latency_ms: number | null; detail: Record<string, unknown> }
export interface Provenance {
  pipeline_version: string; generator: string; model: string | null; prompt_version: string | null; prompt_hash: string | null; taxonomy_version: number | null; corpus_version: number | null;
  embedding_model: string | null; reranker: string | null; retrieval: Record<string, unknown>; thresholds: Record<string, number>; generation: Record<string, unknown>;
}
export interface LineageSource {
  id: string; type: "ticket" | "article"; title: string; excerpt: string; rank: number; method: string; score: number; stage_scores: Record<string, number>; why: string[];
  intent: string | null; matches_intent: boolean; selected: boolean; in_prompt: boolean; cited_by_steps: number[];
}
export interface LineageStep { index: number; text: string; citations: string[]; grounded: boolean | null; grounding_score: number | null; support: Record<string, number> }
export interface LineageEdge { kind: "extracted" | "retrieved" | "selected" | "cited"; source: string; target: string; weight: number | null }
export interface Lineage {
  complaint: { text: string; chars: number; pii_redactions: Record<string, number>; injection_flagged: boolean };
  attributes: { name: string; value: string; confidence: number }[]; sources: LineageSource[]; steps: LineageStep[]; edges: LineageEdge[];
  signals: Record<string, unknown>; checks: Record<string, boolean>;
}
export interface AdaptiveDecision { kind: string; stage: string; signal: string; value: number | null; threshold: number | null; action: string; reason: string; expansion_terms?: string[] }

/** The resolve response with the fields added for auditability. */
export type Resolved = ResolveResponse & { provenance?: Provenance | null; trace?: StageRecord[]; lineage?: Lineage | null };

// ---------------------------------------------------------------- cases and replay
export interface CaseListItem {
  request_id: string; created_at: string; complaint: string; status: ResolveResponse["status"]; confidence: number; latency_ms: number; intent: string | null; product: string | null;
  generator: string | null; prompt_version: string | null; has_trace: boolean; rating: "helpful" | "not_helpful" | null;
}
export interface CaseSource { source_type: "ticket" | "article"; source_id: string; score: number; rank: number; method?: string; scores?: Record<string, number>; intent?: string; in_corpus: boolean; title: string | null; excerpt: string | null }
export interface CaseFeedback { feedback_id: number; rating: string; comment: string | null; corrected_intent: string | null; reasons: string[]; rejected_sources: string[]; edited_steps: string[] | null; created_at: string }
export interface CaseDetail {
  request_id: string; trace_id: string; created_at: string; complaint: string; status: ResolveResponse["status"]; confidence: number; latency_ms: number; classification: Classification;
  sources: CaseSource[]; result: Partial<ResolveResponse>; lineage: Lineage | null; trace: StageRecord[] | null; provenance: Provenance | null; trace_available: boolean; feedback: CaseFeedback[]; has_embedding: boolean;
}
export interface CaseDiff {
  reproduced: boolean; status: { old: string; new: string; same: boolean }; classification: Record<string, { old: string; new: string }>;
  retrieval: { same_set: boolean; same_order: boolean; added: string[]; removed: string[]; moved: { id: string; old_rank: number; new_rank: number }[] };
  steps: { same: boolean; added: string[]; removed: string[]; old_count: number; new_count: number }; evidence: { old: number | null; new: number; delta: number | null };
  citations: { old: string[]; new: string[]; same: boolean }; provenance_changes: Record<string, { old: unknown; new: unknown }>; likely_reasons: string[];
}
export interface ReplayResult { request_id: string; replay: Resolved; diff: CaseDiff; persisted: boolean; options: { strategy: string | null; generate: boolean; deterministic: boolean } }

// ---------------------------------------------------------------- retrieval lab
export interface LabResultRow { id: string; rank: number; score: number; title: string; excerpt: string; intent: string | null; scores: Record<string, number>; relevant: boolean | null }
export interface LabKind { results: LabResultRow[]; latency_ms: number; summary: { first_relevant_rank: number | null; "hit@1": number; mrr: number } | null; adaptive?: AdaptiveDecision[] }
export interface LabStrategy { name: string; label: string; description: string; latency_ms: number; kinds: { ticket: LabKind; article: LabKind } }
export interface LabResult {
  complaint: string; k: number; embedding_ms: number; strategies: LabStrategy[]; rank_movement: Record<string, Record<string, number>>;
  ground_truth: { available: boolean; scenario_id: string | null; source: string | null; note: string };
}
export interface LabExample { qid: string; split: string; text: string; scenario_id: string; intent: string | null }

// ---------------------------------------------------------------- recurring clusters, quality, system
export interface RecurringCluster {
  cluster_id: string; size: number; requests: number; kind: string; keywords: string[]; representative: { request_id: string; complaint: string }; intent: string | null; intent_share: number;
  product: string | null; product_share: number; severity_mix: Record<string, number>;
  time_pattern: { first_seen: string; last_seen: string; daily_counts: number[]; last_24h: number; earlier_daily_mean: number; label: string };
  confidence: number; confidence_parts: { cohesion: number; cohesion_score: number; intent_agreement: number }; mean_evidence: number | null; abstained_share: number;
  examples: { request_id: string; complaint: string; created_at: string }[]; disclaimer: string;
  proposals: { proposal_id: string; status: string; recommendation: string; label_id: string; overlap_share: number }[]; drift_cluster: string | null; discovery: string;
}
export interface RecurringResult { window_days: number; min_size: number; distance: number; clusters: RecurringCluster[]; stats: Record<string, number>; disclaimer: string; cached: boolean }

export interface IntentQuality { intent: string; requests: number; feedback: number; helpful: number; not_helpful: number; abstained: number; rejection_rate: number | null; rejection_lower_bound: number; abstention_rate: number | null }
export interface SourceQuality { source_id: string; type: string; cited_in_rejected: number; cited_in_helpful: number; rejection_share: number | null; explicit_rejections: number; intents: string[] }
export interface QualitySummary {
  window_days: number;
  totals: { requests: number; feedback: number; requests_with_feedback: number; coverage: number | null; helpful: number; not_helpful: number; helpful_rate: number | null; rejection_rate: number | null; edited_by_agent: number; with_corrected_intent: number };
  by_intent: IntentQuality[]; problem_intents: IntentQuality[]; rejected_sources: SourceQuality[]; weak_articles: SourceQuality[];
  abstention: { count: number; rate: number | null; by_reason: Record<string, number>; by_intent: Record<string, number>; by_product: Record<string, number> };
  failure_patterns: { pattern: string; count: number; share: number | null; detail: string }[];
  feedback_reasons: Record<string, number>; intent_confusions: { predicted: string; corrected: string; count: number }[];
}
export interface QualityFinding { kind: string; severity: "high" | "medium" | "info"; title: string; evidence: Record<string, unknown>; suggested_action: string }
export interface QualityReport { window_days: number; findings: QualityFinding[]; markdown: string; advisory_only: boolean; note: string }

export interface DbFinding { level: string; what: string; detail: string }
export interface SystemStatus {
  status: string; checks: Record<string, string>; corpus: Record<string, number> | null;
  versions: { pipeline: string; prompt: { version: string; hash: string }; taxonomy: number; corpus: number | null; embedding_model: string; reranker: string | null; affect_model: string | null; default_retrieval_strategy: string };
  llm: { providers: { name: string; model: string; circuit: string; consecutive_failures: number; slots: { limit: number; in_use: number } }[]; reachable: Record<string, boolean>; budget_seconds: number; max_concurrency: number; deterministic_default: boolean };
  queue: { mode: string; depth: Record<string, number>; kinds: { kind: string; last_created: string | null; failed_24h: number; active: number }[] };
  drift: { status: string; age_seconds: number; alerts: number; stale: boolean } | null; tracing: string;
  security: { authentication: boolean; rate_limit_per_minute: number; environment: string; metrics_protected: boolean };
  database: { status: string; findings: DbFinding[]; elapsed_ms: number } | { error: string };
}
export interface DbHealth {
  status: string; findings: DbFinding[]; elapsed_ms: number;
  server: { postgres: string; pgvector: string | null; connections: number; max_connections: number; buffer_cache_hit_ratio: number | null; deadlocks: number; rollback_ratio: number };
  pool: { min: number; max: number; size: number | null; available: number | null; requests_waiting: number | null; statement_timeout_ms: number };
  vector_search: { embedding_model: string; dimensions: number; hnsw_ef_search: number; indexes: { name: string; table: string; bytes: number; m: number; ef_construction: number; defaults: boolean }[] };
  tables: { name: string; rows: number; total_bytes: number; table_bytes: number; index_bytes: number; dead_rows: number }[];
  indexes: { name: string; table: string; method: string | null; bytes: number; scans: number; is_unique: boolean }[];
  embedding_coverage: Record<string, number>; integrity: Record<string, number>; duplicates: Record<string, number>;
}
/** Recorded result files (evaluation, drift demo, scale experiment, load test), rendered defensively because they are JSON from disk. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type EvalResults = Record<string, any>;
