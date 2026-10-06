import type {
  CaseDetail, CaseListItem, DbHealth, EvalResults, LabExample, LabResult, QualityReport, QualitySummary, RecurringResult, ReplayResult, Resolved, SystemStatus,
} from "./consoleTypes";
import type { AcceptBody, AcceptResult, DriftHistoryItem, DriftReport, DriftTimeline, Job, Proposal, SearchResponse, Taxonomy } from "./types";

export const getKey = () => localStorage.getItem("riq_api_key") ?? "";
export const setKey = (k: string) => localStorage.setItem("riq_api_key", k);

export async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const key = getKey();
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(key ? { "X-API-Key": key } : {}), ...(init?.headers ?? {}) },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = body?.error?.message ?? body?.detail ?? res.statusText;
    const details = body?.error?.details?.map((d: { field: string; issue: string }) => `${d.field}: ${d.issue}`).join("; ");
    throw new Error(`${res.status} ${msg}${details ? ` (${details})` : ""}`);
  }
  return body as T;
}

export const resolve = (complaint: string, strategy: string, useFilters: boolean, deterministic = false) =>
  call<Resolved>("/api/v1/resolve", {
    method: "POST",
    body: JSON.stringify({ complaint, strategy: strategy || null, use_metadata_filters: useFilters, deterministic }),
  });

export const search = (q: string, strategy: string, limit = 3) =>
  call<SearchResponse>(`/api/v1/search?${new URLSearchParams({ q, strategy, source: "ticket", limit: String(limit) })}`);

export interface FeedbackBody {
  request_id: string; rating: "helpful" | "not_helpful"; comment?: string; corrected_intent?: string; reasons?: string[]; rejected_sources?: string[]; edited_steps?: string[];
}
export const feedback = (body: FeedbackBody) => call<{ feedback_id: number }>("/api/v1/feedback", { method: "POST", body: JSON.stringify(body) });

export const ready = () => call<{ status: string; checks: Record<string, unknown> }>("/health/ready").catch(() => null);

export const getTaxonomy = () => call<Taxonomy>("/api/v1/taxonomy");

export const listProposals = (status: Proposal["status"] | "all" = "all") =>
  call<{ items: Proposal[] }>(`/api/v1/taxonomy/proposals?status=${status}`).then((r) => r.items);

export const runDiscovery = (windowDays?: number) =>
  call<{ job_id: string }>("/api/v1/taxonomy/discover", { method: "POST", body: JSON.stringify(windowDays ? { window_days: windowDays } : {}) });

export const getJob = (id: string) => call<Job>(`/api/v1/jobs/${id}`);

export const acceptProposal = (id: string, body: AcceptBody) =>
  call<AcceptResult>(`/api/v1/taxonomy/proposals/${id}/accept`, { method: "POST", body: JSON.stringify(body) });

export const rejectProposal = (id: string, note?: string) =>
  call<{ proposal_id: string; status: string }>(`/api/v1/taxonomy/proposals/${id}/reject`, { method: "POST", body: JSON.stringify(note ? { note } : {}) });

export const getDriftStatus = () => call<DriftReport>("/api/v1/monitoring/drift/status");

export const runDriftAnalysis = () => call<{ job_id: string }>("/api/v1/monitoring/drift/run", { method: "POST", body: JSON.stringify({}) });

export const getDriftHistory = (limit = 30) => call<{ items: DriftHistoryItem[] }>(`/api/v1/monitoring/drift/history?limit=${limit}`).then((r) => r.items);

export const getDriftTimeline = (days = 14, bucket: "day" | "hour" = "day") => call<DriftTimeline>(`/api/v1/monitoring/drift/timeline?days=${days}&bucket=${bucket}`);

// ---- console endpoints: cases, replay, retrieval lab, clusters, quality, system, evaluation
export const listCases = (p: { limit?: number; offset?: number; status?: string; rated?: string } = {}) => {
  const q = new URLSearchParams({ limit: String(p.limit ?? 25), offset: String(p.offset ?? 0), ...(p.status ? { status: p.status } : {}), ...(p.rated ? { rated: p.rated } : {}) });
  return call<{ total: number; items: CaseListItem[] }>(`/api/v1/cases?${q}`);
};
export const getCase = (id: string) => call<CaseDetail>(`/api/v1/cases/${id}`);
export const replayCase = (id: string, body: { strategy?: string; generate: boolean; deterministic: boolean }) =>
  call<ReplayResult>(`/api/v1/cases/${id}/replay`, { method: "POST", body: JSON.stringify({ ...body, strategy: body.strategy || null }) });
export const compareCase = (id: string, strategies?: string[]) =>
  call<LabResult>(`/api/v1/cases/${id}/compare`, { method: "POST", body: JSON.stringify({ strategies: strategies ?? null, k: 5 }) });
export const compareRetrieval = (complaint: string, strategies: string[], scenario_id?: string) =>
  call<LabResult>("/api/v1/retrieval/compare", { method: "POST", body: JSON.stringify({ complaint, strategies, k: 5, scenario_id: scenario_id ?? null }) });
export const labExamples = () => call<{ items: LabExample[] }>("/api/v1/retrieval/examples?limit=60").then((r) => r.items);
export const recurringClusters = (days = 7, minSize = 3) => call<RecurringResult>(`/api/v1/clusters/recurring?days=${days}&min_size=${minSize}`);
export const qualitySummary = (days = 30) => call<QualitySummary>(`/api/v1/quality/summary?days=${days}`);
export const qualityReport = (days = 30) => call<QualityReport>(`/api/v1/quality/report?days=${days}`);
export const systemStatus = () => call<SystemStatus>("/api/v1/system/status");
export const dbHealth = () => call<DbHealth>("/api/v1/system/db");
export const evaluationResults = () => call<EvalResults>("/api/v1/evaluation/results");
export const runEvaluation = (suites: string[], maxQueries?: number) =>
  call<{ job_id: string }>("/api/v1/evaluate", { method: "POST", body: JSON.stringify({ suites, max_queries: maxQueries ?? null }) });
