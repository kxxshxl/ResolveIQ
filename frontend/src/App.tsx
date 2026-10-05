import { useEffect, useMemo, useState } from "react";
import { feedback, getKey, ready, resolve, search, setKey } from "./api";
import type { Item, ResolveResponse, SearchResponse } from "./types";

const EXAMPLES = [
  "My broadband drops every evening around 8 and I've already restarted the router twice. I work from home and this is costing me.",
  "You took my monthly payment twice on the 3rd and I want the extra one refunded. This is unacceptable.",
  "There is no mobile signal anywhere in our village since the storm and elderly neighbours cannot call for help.",
  "My eSIM QR code keeps failing to activate on my new phone.",
  "What is the best pizza place near the city centre?",
];
const STRATEGIES = [
  ["", "Server default (dense)"],
  ["dense", "Dense (semantic)"],
  ["hybrid", "Hybrid (dense + lexical, RRF)"],
  ["hybrid_reranked", "Hybrid + cross-encoder rerank"],
  ["lexical", "Lexical only (keyword baseline)"],
];
const STAGES = ["preprocess", "embed", "classify", "retrieve", "generate"];
const pct = (x: number) => `${Math.round(x * 100)}%`;

function Bar({ value, tone }: { value: number; tone?: string }) {
  return (
    <div className="bar" role="meter" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className={`bar-fill ${tone ?? ""}`} style={{ width: pct(Math.max(0, Math.min(1, value))) }} />
    </div>
  );
}

function Detected({ r }: { r: ResolveResponse }) {
  const c = r.classification;
  const rows: [string, string, number][] = [
    ["Intent", c.intent, c.confidence.intent],
    ["Product", c.product, c.confidence.product],
    ["Severity", c.severity, c.confidence.severity],
    ["Sentiment", c.sentiment, c.confidence.sentiment],
  ];
  return (
    <section className="card">
      <h2>1 · Complaint understanding</h2>
      <div className="grid4">
        {rows.map(([k, v, conf]) => (
          <div key={k} className="detected">
            <span className="muted">{k}</span>
            <strong className={`pill sev-${k === "Severity" ? v : "x"}`}>{v.replace(/_/g, " ")}</strong>
            <Bar value={conf} tone={conf < 0.45 ? "warn" : ""} />
            <span className="muted small">confidence {pct(conf)}</span>
          </div>
        ))}
      </div>
      <p className="muted small">
        strategy <code>{c.strategy}</code> · taxonomy v{c.taxonomy_version}
        {Object.keys(r.pii_redactions).length > 0 && <> · PII redacted: {Object.entries(r.pii_redactions).map(([k, v]) => `${k}×${v}`).join(", ")}</>}
      </p>
    </section>
  );
}

function ItemCard({ it, cited, active, onHover }: { it: Item; cited: boolean; active: boolean; onHover: (id: string | null) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`item ${cited ? "cited" : ""} ${active ? "active" : ""}`} onMouseEnter={() => onHover(it.source_id)} onMouseLeave={() => onHover(null)}>
      <div className="row">
        <span className="rank">#{it.rank}</span>
        <code className="id">{it.source_id}</code>
        {cited && <span className="pill ok">cited</span>}
        <span className="grow" />
        <span className="muted small">{it.retrieval_method}</span>
        <strong>{it.score.toFixed(2)}</strong>
      </div>
      <Bar value={it.score} />
      <div className="title">{it.source_type === "ticket" ? it.text : it.title}</div>
      <div className="muted small">
        {String(it.metadata.intent).replace(/_/g, " ")} · {String(it.metadata.product)}
        {it.source_type === "ticket" && <> · {String(it.metadata.severity)}</>}
        {it.source_type === "ticket" && it.resolution_summary && <> · fix: {it.resolution_summary}</>}
      </div>
      <button className="link" onClick={() => setOpen(!open)}>{open ? "hide" : "show"} score breakdown</button>
      {open && (
        <div className="breakdown">
          {Object.entries(it.scores).map(([k, v]) => <span key={k}><span className="muted">{k}</span> {v.toFixed(3)}</span>)}
        </div>
      )}
    </div>
  );
}

function Evidence({ r, hover, setHover }: { r: ResolveResponse; hover: string | null; setHover: (id: string | null) => void }) {
  const cited = new Set(r.citations.map((c) => c.source_id));
  return (
    <section className="card">
      <h2>3 · Retrieved evidence <span className="muted small">(method: {r.tickets[0]?.retrieval_method ?? "-"})</span></h2>
      <div className="cols">
        <div>
          <h3>Resolved tickets</h3>
          {r.tickets.map((t) => <ItemCard key={t.source_id} it={t} cited={cited.has(t.source_id)} active={hover === t.source_id} onHover={setHover} />)}
        </div>
        <div>
          <h3>Knowledge-base articles</h3>
          {r.articles.map((a) => <ItemCard key={a.source_id} it={a} cited={cited.has(a.source_id)} active={hover === a.source_id} onHover={setHover} />)}
        </div>
      </div>
    </section>
  );
}

function Resolution({ r, hover, setHover }: { r: ResolveResponse; hover: string | null; setHover: (id: string | null) => void }) {
  const [sent, setSent] = useState<string | null>(null);
  const banner: Record<string, [string, string]> = {
    resolved: ["ok", "Grounded resolution"],
    degraded: ["warn", "Evidence-only resolution (LLM unavailable)"],
    unreliable: ["bad", "Unreliable - automatic citation/grounding checks failed"],
    abstained: ["bad", "Abstained - evidence is insufficient"],
  };
  const [tone, label] = banner[r.status];
  const res = r.resolution;
  return (
    <section className="card">
      <h2>2 · Recommended resolution</h2>
      <div className={`banner ${tone}`}><strong>{label}</strong><span className="grow" /><span>confidence {pct(r.confidence)}</span></div>
      <p>{res.issue_summary}</p>
      {res.steps.length > 0 ? (
        <ol className="steps">
          {res.steps.map((s, i) => (
            <li key={i} className={s.grounded === false ? "ungrounded" : ""}>
              <span>{s.text}</span>
              <span className="chips">
                {s.citations.map((c) => (
                  <code key={c} className={`chip ${hover === c ? "active" : ""}`} onMouseEnter={() => setHover(c)} onMouseLeave={() => setHover(null)}>{c}</code>
                ))}
                {s.grounded === false && <span className="pill bad">not supported by cited text</span>}
                {s.grounding_score != null && <span className="muted small">support {s.grounding_score.toFixed(2)}</span>}
              </span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="muted">No steps were generated because the retrieved evidence does not cover this problem.</p>
      )}
      {res.uncertainty && <p className="muted small">Uncertainty: {res.uncertainty}</p>}
      <div className={`escalation ${res.escalate ? "yes" : "no"}`}>
        <strong>Escalation: {res.escalate ? "recommended" : "not required"}</strong>
        {res.escalation_reason && <div>{res.escalation_reason}</div>}
      </div>
      <div className="row gap">
        <span className="muted small">Was this helpful?</span>
        {sent ? <span className="pill ok">thanks - {sent}</span> : (
          <>
            <button onClick={() => feedback(r.request_id, "helpful").then(() => setSent("helpful"))}>👍 Helpful</button>
            <button onClick={() => feedback(r.request_id, "not_helpful").then(() => setSent("not helpful"))}>👎 Not helpful</button>
          </>
        )}
      </div>
    </section>
  );
}

function Assurance({ r }: { r: ResolveResponse }) {
  const e = r.evidence, v = r.validation;
  return (
    <section className="card">
      <h2>4 · Grounding &amp; confidence</h2>
      <div className="grid2">
        <div>
          <h3>Evidence sufficiency <span className={`pill ${e.sufficient ? "ok" : "bad"}`}>{e.sufficient ? "sufficient" : "insufficient"}</span></h3>
          <table className="kv">
            <tbody>
              <tr><td>overall evidence confidence</td><td>{e.confidence.toFixed(2)}</td></tr>
              <tr><td>best ticket similarity (cosine)</td><td>{e.top_ticket_similarity.toFixed(2)}</td></tr>
              <tr><td>best article similarity (cosine)</td><td>{e.top_article_similarity.toFixed(2)}</td></tr>
              <tr><td>reranker signal</td><td>{e.rerank_signal != null ? e.rerank_signal.toFixed(2) : "n/a"}</td></tr>
              <tr><td>ticket consensus (same intent)</td><td>{pct(e.consensus)}</td></tr>
            </tbody>
          </table>
          <p className="muted small">{e.reason}</p>
        </div>
        <div>
          <h3>Citation validation <span className={`pill ${v.valid ? "ok" : "bad"}`}>{v.valid ? "passed" : "failed"}</span></h3>
          <table className="kv">
            <tbody>
              <tr><td>citation coverage</td><td>{pct(v.citation_coverage)}</td></tr>
              <tr><td>steps supported by cited text</td><td>{pct(v.grounded_ratio)}</td></tr>
              <tr><td>invalid (invented) ids removed</td><td>{v.invalid_citations.length ? v.invalid_citations.join(", ") : "none"}</td></tr>
              <tr><td>generator</td><td><code>{r.generator}</code></td></tr>
            </tbody>
          </table>
          {[...v.warnings, ...r.warnings.filter((w) => !v.warnings.includes(w))].map((w) => <p key={w} className="warn-text small">⚠ {w}</p>)}
        </div>
      </div>
      <div className="stages">
        {STAGES.map((s) => (
          <div key={s} className="stage"><span className="muted small">{s}</span><strong>{r.latency_ms[s] != null ? `${Math.round(r.latency_ms[s])} ms` : "-"}</strong></div>
        ))}
        <div className="stage total"><span className="muted small">total{r.cached ? " (cached)" : ""}</span><strong>{Math.round(r.latency_ms.total)} ms</strong></div>
      </div>
      <p className="muted small">trace <code>{r.trace_id}</code> · request <code>{r.request_id}</code></p>
    </section>
  );
}

function Compare({ text }: { text: string }) {
  const [res, setRes] = useState<Record<string, SearchResponse> | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    const out: Record<string, SearchResponse> = {};
    for (const s of ["lexical", "dense"]) out[s] = await search(text, s, 3);
    setRes(out);
    setBusy(false);
  };
  return (
    <section className="card">
      <h2>Why semantic search? <span className="muted small">keyword vs embedding retrieval for this complaint</span></h2>
      {!res && <button onClick={run} disabled={busy}>{busy ? "Searching…" : "Compare keyword search with semantic search"}</button>}
      {res && (
        <div className="cols">
          {Object.entries(res).map(([s, r]) => (
            <div key={s}>
              <h3>{s === "lexical" ? "Keyword (Postgres full-text)" : "Semantic (pgvector)"} <span className="muted small">{r.latency_ms} ms</span></h3>
              {(r.tickets ?? []).map((t) => (
                <div key={t.source_id} className="item">
                  <div className="row"><span className="rank">#{t.rank}</span><code className="id">{t.source_id}</code><span className="grow" /><span className="pill">{String(t.metadata.intent).replace(/_/g, " ")}</span></div>
                  <div className="title">{t.text}</div>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [strategy, setStrategy] = useState("");
  const [filters, setFilters] = useState(false);
  const [key, setKeyState] = useState(getKey());
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [r, setR] = useState<ResolveResponse | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [health, setHealth] = useState<string>("checking…");

  useEffect(() => { ready().then((h) => setHealth(h ? h.status : "backend unreachable")); }, []);
  const submit = async () => {
    setBusy(true); setErr(null); setKey(key);
    try { setR(await resolve(text, strategy, filters)); } catch (e) { setErr((e as Error).message); setR(null); } finally { setBusy(false); }
  };
  const canSubmit = useMemo(() => text.trim().length >= 5 && !busy, [text, busy]);

  return (
    <div className="app">
      <header>
        <div><h1>ResolveIQ</h1><span className="muted">Semantic ticket resolution assistant for telecom support agents</span></div>
        <span className={`pill ${health === "ready" ? "ok" : "bad"}`}>backend: {health}</span>
      </header>

      <section className="card input">
        <label htmlFor="complaint"><h2>Customer complaint</h2></label>
        <textarea id="complaint" rows={4} maxLength={4000} value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste the raw customer complaint…" />
        <div className="examples">
          {EXAMPLES.map((e, i) => <button key={i} className="ghost" onClick={() => setText(e)}>{e.slice(0, 44)}…</button>)}
        </div>
        <div className="row gap wrap">
          <label className="field">Retrieval strategy
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>{STRATEGIES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
          </label>
          <label className="field check"><input type="checkbox" checked={filters} onChange={(e) => setFilters(e.target.checked)} /> narrow by predicted product</label>
          <label className="field">API key (if enabled)
            <input type="password" value={key} onChange={(e) => setKeyState(e.target.value)} placeholder="X-API-Key" />
          </label>
          <span className="grow" />
          <button className="primary" disabled={!canSubmit} onClick={submit}>{busy ? "Resolving…" : "Resolve"}</button>
        </div>
        {err && <p className="error">⚠ {err}</p>}
      </section>

      {r && (
        <>
          <Detected r={r} />
          <Resolution r={r} hover={hover} setHover={setHover} />
          <Evidence r={r} hover={hover} setHover={setHover} />
          <Assurance r={r} />
          <Compare key={r.request_id} text={text} />
        </>
      )}
    </div>
  );
}
