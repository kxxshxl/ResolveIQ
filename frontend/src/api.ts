import type { AcceptBody, AcceptResult, Job, Proposal, ResolveResponse, SearchResponse, Taxonomy } from "./types";

export const getKey = () => localStorage.getItem("riq_api_key") ?? "";
export const setKey = (k: string) => localStorage.setItem("riq_api_key", k);

async function call<T>(path: string, init?: RequestInit): Promise<T> {
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

export const resolve = (complaint: string, strategy: string, useFilters: boolean) =>
  call<ResolveResponse>("/api/v1/resolve", {
    method: "POST",
    body: JSON.stringify({ complaint, strategy: strategy || null, use_metadata_filters: useFilters }),
  });

export const search = (q: string, strategy: string, limit = 3) =>
  call<SearchResponse>(`/api/v1/search?${new URLSearchParams({ q, strategy, source: "ticket", limit: String(limit) })}`);

export const feedback = (request_id: string, rating: "helpful" | "not_helpful") =>
  call<{ feedback_id: number }>("/api/v1/feedback", { method: "POST", body: JSON.stringify({ request_id, rating }) });

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
