import { useState } from "react";
import { dbHealth, systemStatus } from "../api";
import type { DbHealth, SystemStatus } from "../consoleTypes";
import { useAsync, useInterval } from "../hooks";
import { Bar, EmptyState, ErrorState, KV, Loading, Panel, Pill, Stat, ago, bytes, pct, type Tone } from "../ui";

const tone = (ok: boolean | undefined): Tone => (ok ? "ok" : "bad");

function Dependencies({ s }: { s: SystemStatus }) {
  return (
    <Panel title="Dependencies" subtitle={s.status === "ready" ? "Ready to serve" : "Not ready"}>
      <div className="grid3">
        {Object.entries(s.checks).map(([k, v]) => <div className="signal" key={k}><span className="name">{k.replace(/_/g, " ")}</span><span className="row"><span className={`dot ${/^(ok|loaded)$/.test(v) ? "ok" : "warn"}`} /><strong style={{ fontSize: 13 }}>{v}</strong></span></div>)}
      </div>
      {s.corpus && <div className="stats"><Stat name="Tickets" value={s.corpus.tickets} /><Stat name="Articles" value={s.corpus.articles} /><Stat name="Deprecated" value={s.corpus.deprecated_articles} /><Stat name="Ticket embeddings" value={s.corpus.ticket_embeddings} /></div>}
    </Panel>
  );
}

function Llm({ s }: { s: SystemStatus }) {
  return (
    <Panel title="Language model" subtitle={`Total budget ${s.llm.budget_seconds}s per answer · ${s.llm.max_concurrency} generation(s) at a time per replica${s.llm.deterministic_default ? " · deterministic by default" : ""}`}>
      {s.llm.providers.length === 0 && <EmptyState title="No provider configured">Answers are evidence-only (extractive).</EmptyState>}
      {s.llm.providers.map((p) => (
        <div className="row wrap" key={p.name} style={{ padding: "6px 0", borderTop: "1px solid var(--line)" }}>
          <strong>{p.name}</strong><code>{p.model}</code>
          <Pill tone={p.circuit === "closed" ? "ok" : p.circuit === "half-open" ? "warn" : "bad"} title="Circuit breaker: open means calls are being skipped after repeated failures">circuit {p.circuit}</Pill>
          <Pill tone={tone(s.llm.reachable[p.name])}>{s.llm.reachable[p.name] ? "reachable" : "unreachable"}</Pill>
          <span className="grow" /><span className="small muted">slots {p.slots.in_use}/{p.slots.limit} · failures in a row {p.consecutive_failures}</span>
        </div>
      ))}
    </Panel>
  );
}

function Database({ d }: { d: DbHealth }) {
  const max = Math.max(1, ...d.tables.map((t) => t.total_bytes));
  return (
    <>
      <Panel title="Database" subtitle={`PostgreSQL ${d.server.postgres.split(" ")[0]} · pgvector ${d.server.pgvector ?? "n/a"} · checked in ${d.elapsed_ms} ms`} actions={<Pill tone={d.status === "ok" ? "ok" : "warn"}>{d.status}</Pill>}>
        {d.findings.length === 0 ? <p className="small"><Pill tone="ok">no findings</Pill> embeddings cover every active document, no orphans, no duplicates.</p> : d.findings.map((f) => <div key={f.what} className={`finding ${f.level === "warn" ? "medium" : ""}`}><strong>{f.what}</strong><div className="small">{f.detail}</div></div>)}
        <div className="stats">
          <Stat name="Connections" value={`${d.server.connections}/${d.server.max_connections}`} /><Stat name="Cache hit ratio" value={pct(d.server.buffer_cache_hit_ratio)} hint="Share of block reads served from memory" />
          <Stat name="Pool" value={`${d.pool.size ?? "?"}/${d.pool.max}`} hint={`${d.pool.available ?? "?"} idle, ${d.pool.requests_waiting ?? 0} waiting`} /><Stat name="Deadlocks" value={d.server.deadlocks} /><Stat name="ef_search" value={d.vector_search.hnsw_ef_search} hint="HNSW recall/latency knob" />
        </div>
      </Panel>
      <div className="grid2">
        <Panel title="Tables" subtitle="Total size including indexes">
          {d.tables.map((t) => <div className="hbar" key={t.name}><span className="name">{t.name.replace(/_/g, " ")}</span><Bar value={t.total_bytes / max} /><span className="small" style={{ textAlign: "right" }}>{bytes(t.total_bytes)}</span></div>)}
        </Panel>
        <Panel title="Vector search">
          <KV rows={[["Embedding model", d.vector_search.embedding_model], ["Dimensions", d.vector_search.dimensions],
            ...d.vector_search.indexes.map((i): [string, React.ReactNode] => [i.table.replace(/_/g, " "), `HNSW m=${i.m}, ef_construction=${i.ef_construction}${i.defaults ? " (defaults)" : ""}, ${bytes(i.bytes)}`]),
            ["Missing embeddings", `${d.embedding_coverage.tickets_missing} tickets, ${d.embedding_coverage.articles_missing} articles`],
            ["Duplicate content groups", `${d.duplicates.ticket_duplicate_groups} tickets, ${d.duplicates.article_duplicate_groups} articles`]]} />
        </Panel>
      </div>
      <Panel title="Indexes" subtitle="Scans since statistics were last reset">
        <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Index</th><th>Table</th><th>Type</th><th className="num">Size</th><th className="num">Scans</th></tr></thead>
          <tbody>{d.indexes.map((i) => <tr key={i.name}><td><code>{i.name}</code></td><td>{i.table}</td><td>{i.method}{i.is_unique ? " (unique)" : ""}</td><td className="num">{bytes(i.bytes)}</td><td className="num">{i.scans}</td></tr>)}</tbody></table></div>
      </Panel>
    </>
  );
}

export default function SystemHealth() {
  const s = useAsync(() => systemStatus(), []);
  const db = useAsync(() => dbHealth(), []);
  const [auto, setAuto] = useState(true);
  useInterval(() => { s.reload(); db.reload(); }, 15000, auto);
  const v = s.data?.versions;
  return (
    <>
      <div className="page-head"><div className="grow"><h1>System health</h1><p>Dependencies, the model and its circuit breaker, queue and drift freshness, versions, and a database health check. Everything here is safe to expose: sizes, counts and states, never row contents or credentials.</p></div>
        <label className="field check"><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />Refresh every 15 s</label><button className="btn" onClick={() => { s.reload(); db.reload(); }}>Refresh now</button></div>
      {s.error && <ErrorState message={s.error} onRetry={s.reload} />}
      {s.loading && !s.data && <Panel><Loading what="system status" /></Panel>}
      {s.data && v && (
        <>
          <div className={`banner ${s.data.status === "ready" ? "ok" : "bad"}`} role="status"><strong>{s.data.status === "ready" ? "All systems ready" : "Not ready"}</strong><span>pipeline {v.pipeline}</span><span>prompt {v.prompt.version}</span><span>taxonomy v{v.taxonomy}</span><span>corpus v{v.corpus}</span></div>
          <div className="grid2">
            <Dependencies s={s.data} />
            <Llm s={s.data} />
          </div>
          <div className="grid2">
            <Panel title="Background work" subtitle={`Job mode: ${s.data.queue.mode}`}>
              <div className="stats"><Stat name="Queued" value={s.data.queue.depth.queued ?? 0} /><Stat name="Running" value={s.data.queue.depth.running ?? 0} /></div>
              {s.data.queue.kinds.length === 0 ? <p className="muted small">No jobs yet.</p> : <table className="tbl"><thead><tr><th>Job</th><th>Last</th><th className="num">Failed (24 h)</th></tr></thead><tbody>{s.data.queue.kinds.map((k) => <tr key={k.kind}><td>{k.kind.replace(/_/g, " ")}</td><td>{ago(k.last_created)}</td><td className="num">{k.failed_24h || "–"}</td></tr>)}</tbody></table>}
              <h3>Drift monitor</h3>
              {s.data.drift ? <p className="small"><Pill tone={s.data.drift.status === "alert" ? "bad" : s.data.drift.status === "ok" ? "ok" : "warn"}>{s.data.drift.status}</Pill> last analysis {ago(s.data.drift.age_seconds)}{s.data.drift.stale && <> · <Pill tone="warn">stale</Pill></>}{s.data.drift.alerts ? ` · ${s.data.drift.alerts} alert(s)` : ""}</p> : <p className="muted small">No drift analysis has run.</p>}
            </Panel>
            <Panel title="Versions and security">
              <KV rows={[["Embedding model", v.embedding_model], ["Reranker", v.reranker ?? "off"], ["Affect model", v.affect_model ?? "unavailable (rule fallback)"], ["Default retrieval", v.default_retrieval_strategy], ["Prompt hash", v.prompt.hash],
                ["Tracing", s.data.tracing], ["Authentication", s.data.security.authentication ? "API keys required" : "off (development only)"], ["Rate limit", `${s.data.security.rate_limit_per_minute}/min per key`], ["Environment", s.data.security.environment], ["Metrics endpoint", s.data.security.metrics_protected ? "token protected" : "open"]]} />
            </Panel>
          </div>
        </>
      )}
      {db.error && <ErrorState message={db.error} onRetry={db.reload} />}
      {db.loading && !db.data && <Panel><Loading what="database health" /></Panel>}
      {db.data && <Database d={db.data} />}
    </>
  );
}
