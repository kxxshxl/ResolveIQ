export interface Confidence { intent: number; product: number; severity: number; sentiment: number }
export interface Classification { intent: string; product: string; severity: string; sentiment: string; confidence: Confidence; strategy: string; taxonomy_version: number }
export interface Item {
  source_type: "ticket" | "article"; source_id: string; title: string; text: string; score: number; rank: number;
  retrieval_method: string; metadata: Record<string, unknown>; steps: string[]; resolution_summary?: string | null; scores: Record<string, number>;
}
export interface Step { text: string; citations: string[]; grounded: boolean | null; grounding_score: number | null }
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
