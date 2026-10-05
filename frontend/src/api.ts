import type { ResolveResponse, SearchResponse } from "./types";

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
