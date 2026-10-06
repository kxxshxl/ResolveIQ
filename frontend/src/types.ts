export interface Confidence { intent: number; product: number; severity: number; sentiment: number }
export interface Classification { intent: string; product: string; severity: string; sentiment: string; confidence: Confidence; strategy: string; taxonomy_version: number }
export interface Item {
  source_type: "ticket" | "article"; source_id: string; title: string; text: string; score: number; rank: number;
  retrieval_method: string; metadata: Record<string, unknown>; steps: string[]; resolution_summary?: string | null; scores: Record<string, number>;
}
export interface Step { text: string; citations: string[]; grounded: boolean | null; grounding_score: number | null; support?: Record<string, number> }
export interface Citation { source_type: string; source_id: string; title: string; score: number; cited_in_steps: number[] }
export interface Validation { valid: boolean; invalid_citations: string[]; uncited_steps: number[]; unsupported_steps: number[]; grounded_ratio: number; citation_coverage: number; warnings: string[] }
export interface Evidence { confidence: number; top_ticket_similarity: number; top_article_similarity: number; rerank_signal: number | null; consensus: number; sufficient: boolean; reason: string }
export interface Resolution { issue_summary: string; steps: Step[]; escalate: boolean; escalation_reason: string | null; uncertainty: string | null }
export interface ResolveResponse {
  request_id: string; trace_id: string; status: "resolved" | "abstained" | "degraded" | "unreliable";
  classification: Classification; tickets: Item[]; articles: Item[]; resolution: Resolution; citations: Citation[];
  validation: Validation; evidence: Evidence; confidence: number; generator: string; pii_redactions: Record<string, number>;
  warnings: string[]; latency_ms: Record<string, number>; cached: boolean;
}
export interface SearchResponse { query: string; strategy: string; tickets?: Item[]; articles?: Item[]; latency_ms: number }

export interface Proposal {
  proposal_id: string; status: "pending" | "accepted" | "rejected" | "superseded"; recommendation: "new_class" | "extend_existing";
  label_id: string; description: string; keywords: string[]; examples: string[]; product: string | null; team: string | null;
  size: number; cohesion: number; mean_evidence: number; nearest_intent: string | null; neighbor_agreement: number;
  members: number; created_at: string; decided_at: string | null; decided_note: string | null;
}
export interface TaxonomyLabel { id: string; description: string; team: string | null }
export interface Taxonomy { version: number; labels: Record<string, TaxonomyLabel[]> }
export interface Job { job_id: string; kind: string; status: "queued" | "running" | "succeeded" | "failed"; attempts: number; result: Record<string, unknown> | null; error: string | null }
export interface DiscoveryResult { window_days: number; candidates: number; unique_candidates: number; proposals: number; elapsed_s: number; by_recommendation?: Record<string, number> }
export interface AcceptResult { action: "created" | "extended"; label_id: string; taxonomy_version: number; members: number }
export interface AcceptBody { label_id?: string; description?: string; team?: string; merge_into?: string; note?: string }

export interface DriftCategory {
  label: string; baseline_n: number; recent_n: number; baseline_share: number; recent_share: number; delta: number; p_value: number; psi_share: number;
  flag: "new" | "vanished" | "up" | "down" | null;
}
export interface DriftDimension { psi: number; p_value: number; chi2_p_value: number; dof: number; pooled_categories: string[]; categories: DriftCategory[]; drifted: boolean }
export interface DriftAlert { signal: string; kind: string; p_value: number | null; effect: Record<string, unknown>; message: string; cluster_id?: string }
export interface RelatedProposal {
  proposal_id: string; status: "pending" | "accepted" | "rejected"; recommendation: "new_class" | "extend_existing"; label_id: string; size: number;
  nearest_intent: string | null; overlap: number; overlap_share: number;
}
export interface DriftCluster {
  cluster_id: string; size: number; share_of_recent: number; baseline_members: number; recent_share_of_cluster: number; enrichment: number; kind: "new_topic" | "surge";
  p_value: number; significant: boolean; keywords: string[]; examples: string[]; mean_evidence: number | null; abstained_share: number;
  intent_mix: Record<string, number>; covered_by_corpus: boolean; cohesion: number; related_proposals: RelatedProposal[];
}
export interface DriftSummaryStats { n: number; mean: number; p10: number; median: number; p90: number }
export interface DriftReport {
  status: "ok" | "alert" | "insufficient_data" | "no_analysis"; note?: string;
  snapshot_id?: string; created_at?: string; age_seconds?: number; window_hours?: number; baseline_days?: number; n_recent?: number; n_baseline?: number;
  alerts?: DriftAlert[]; distributions?: Record<string, DriftDimension>;
  quality?: { evidence: { baseline: DriftSummaryStats | null; recent: DriftSummaryStats | null; ks_d: number; p_value: number; drifted: boolean };
              abstention: { baseline_rate: number; recent_rate: number; p_value: number; drifted: boolean } };
  embedding?: { status: string; reason?: string; centroid_distance?: number; centroid_p_value?: number; centroid_drifted?: boolean; unseen_rate?: number; unseen_expected_rate?: number;
                n_recent?: number; n_baseline?: number };
  emerging_clusters?: DriftCluster[];
  tests?: { n: number; family_alpha: number; method: string };
  thresholds?: Record<string, number>;
  discovery?: { recommended: boolean; triggered: boolean; reason: string; job_id?: string; clusters?: string[] };
}
export interface DriftHistoryItem { snapshot_id: string; created_at: string; status: "ok" | "alert" | "insufficient_data"; n_recent: number; n_baseline: number; alert_count: number; emerging_clusters: number }
export interface DriftTimelinePoint { bucket: string; n: number; abstention_rate: number; mean_evidence: number | null; intents: Record<string, number> }
export interface DriftTimeline { bucket: string; days: number; points: DriftTimelinePoint[] }
