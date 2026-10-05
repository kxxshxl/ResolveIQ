import { useEffect, useMemo, useRef, useState } from "react";
import { feedback, getKey, resolve, search, setKey } from "../api";
import type { Item, ResolveResponse, SearchResponse } from "../types";
import { Bar, CopyButton, Skeleton, label, pct } from "../ui";

const EXAMPLES = [
  ["Broadband drops", "My broadband drops every evening around 8 and I've already restarted the router twice. I work from home and this is costing me."],
  ["Double charge", "You took my monthly payment twice on the 3rd and I want the extra one refunded. This is unacceptable."],
  ["No signal (urgent)", "There is no mobile signal anywhere in our village since the storm and elderly neighbours cannot call for help."],
  ["eSIM activation", "My eSIM QR code keeps failing to activate on my new phone."],
  ["Out of scope", "What is the best pizza place near the city centre?"],
];
const STRATEGIES = [
  ["", "Server default (dense)"],
  ["dense", "Dense (semantic)"],
  ["hybrid", "Hybrid (dense + lexical, RRF)"],
  ["hybrid_reranked", "Hybrid + cross-encoder rerank"],
  ["lexical", "Lexical only (keyword baseline)"],
];
const STAGES = ["preprocess", "embed", "classify", "retrieve", "generate"];
const MAX_CHARS = 4000;

function Summary({ r }: { r: ResolveResponse }) {
  const tone = { resolved: "ok", degraded: "warn", unreliable: "bad", abstained: "bad" }[r.status];
  const text = { resolved: "Grounded resolution", degraded: "Evidence only (LLM unavailable)", unreliable: "Unreliable: grounding checks failed", abstained: "Abstained: not enough evidence" }[r.status];
  return (
    <div className={`summary ${tone}`} role="status">
      <strong>{text}</strong>
      <span>confidence {pct(r.confidence)}</span>
      <span>{Math.round(r.latency_ms.total)} ms{r.cached ? " (cached)" : ""}</span>
      <span>{r.resolution.escalate ? "escalate" : "no escalation"}</span>
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
            <strong className={`pill sev-${k === "Severity" ? v : "x"}`}>{label(v)}</strong>
            <Bar value={conf} tone={conf < 0.45 ? "warn" : ""} />
            <span className="muted small">confidence {pct(conf)}{conf < 0.45 && " (low)"}</span>
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
        {label(String(it.metadata.intent))} · {String(it.metadata.product)}
        {it.source_type === "ticket" && <> · {String(it.metadata.severity)}</>}
        {it.source_type === "ticket" && it.resolution_summary && <> · fix: {it.resolution_summary}</>}
      </div>
      <button className="link" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? "hide" : "show"} score breakdown</button>
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
          {r.tickets.length === 0 && <p className="muted small">No similar resolved tickets.</p>}
          {r.tickets.map((t) => <ItemCard key={t.source_id} it={t} cited={cited.has(t.source_id)} active={hover === t.source_id} onHover={setHover} />)}
        </div>
        <div>
          <h3>Knowledge-base articles</h3>
          {r.articles.length === 0 && <p className="muted small">No matching articles.</p>}
          {r.articles.map((a) => <ItemCard key={a.source_id} it={a} cited={cited.has(a.source_id)} active={hover === a.source_id} onHover={setHover} />)}
        </div>
      </div>
    </section>
  );
}

function asPlainText(r: ResolveResponse) {
  const res = r.resolution;
  const lines = [res.issue_summary, ""];
  res.steps.forEach((s, i) => lines.push(`${i + 1}. ${s.text}${s.citations.length ? ` [${s.citations.join(", ")}]` : ""}`));
  if (res.uncertainty) lines.push("", `Uncertainty: ${res.uncertainty}`);
  if (res.escalate) lines.push("", `Escalate: ${res.escalation_reason ?? "recommended"}`);
  return lines.join("\n");
}

function Resolution({ r, hover, setHover }: { r: ResolveResponse; hover: string | null; setHover: (id: string | null) => void }) {
  const [sent, setSent] = useState<string | null>(null);
  const [fbErr, setFbErr] = useState<string | null>(null);
  const res = r.resolution;
  const send = (rating: "helpful" | "not_helpful") =>
    feedback(r.request_id, rating).then(() => { setSent(rating.replace("_", " ")); setFbErr(null); }).catch((e: Error) => setFbErr(e.message));
  return (
    <section className="card">
      <div className="row"><h2 className="grow">2 · Recommended resolution</h2>{res.steps.length > 0 && <CopyButton text={asPlainText(r)}>Copy resolution</CopyButton>}</div>
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
        {sent ? <span className="pill ok">thanks: {sent}</span> : (
          <>
            <button onClick={() => send("helpful")}>👍 Helpful</button>
            <button onClick={() => send("not_helpful")}>👎 Not helpful</button>
          </>
        )}
        {fbErr && <span className="error small">⚠ {fbErr}</span>}
      </div>
    </section>
  );
}

function Assurance({ r }: { r: ResolveResponse }) {
  const e = r.evidence, v = r.validation;
  const total = Math.max(1, r.latency_ms.total);
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
      <h3>Latency by stage</h3>
      <div className="stages">
        {STAGES.map((s) => (
          <div key={s} className="stage">
            <span className="muted small">{s}</span>
            <strong>{r.latency_ms[s] != null ? `${Math.round(r.latency_ms[s])} ms` : "-"}</strong>
            <Bar value={(r.latency_ms[s] ?? 0) / total} />
          </div>
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
  const [err, setErr] = useState<string | null>(null);
  const run = async () => {
    setBusy(true); setErr(null);
    try {
      const out: Record<string, SearchResponse> = {};
      for (const s of ["lexical", "dense"]) out[s] = await search(text, s, 3);
      setRes(out);
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  return (
    <section className="card">
      <h2>Why semantic search? <span className="muted small">keyword vs embedding retrieval for this complaint</span></h2>
      {!res && <button onClick={run} disabled={busy}>{busy ? "Searching…" : "Compare keyword search with semantic search"}</button>}
      {err && <p className="error small">⚠ {err}</p>}
      {res && (
        <div className="cols">
          {Object.entries(res).map(([s, r]) => (
            <div key={s}>
              <h3>{s === "lexical" ? "Keyword (Postgres full-text)" : "Semantic (pgvector)"} <span className="muted small">{r.latency_ms} ms</span></h3>
              {(r.tickets ?? []).length === 0 && <p className="muted small">No matches.</p>}
              {(r.tickets ?? []).map((t) => (
                <div key={t.source_id} className="item">
                  <div className="row"><span className="rank">#{t.rank}</span><code className="id">{t.source_id}</code><span className="grow" /><span className="pill">{label(String(t.metadata.intent))}</span></div>
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

function Loading() {
  return (
    <div aria-busy="true" aria-live="polite">
      {["Complaint understanding", "Recommended resolution", "Retrieved evidence"].map((t) => (
        <section key={t} className="card"><h2 className="muted">{t}</h2><Skeleton lines={3} /></section>
      ))}
    </div>
  );
}

export default function Resolve() {
  const [text, setText] = useState(EXAMPLES[0][1]);
  const [strategy, setStrategy] = useState("");
  const [filters, setFilters] = useState(false);
  const [key, setKeyState] = useState(getKey());
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [r, setR] = useState<ResolveResponse | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const results = useRef<HTMLDivElement>(null);

  const canSubmit = useMemo(() => text.trim().length >= 5 && !busy, [text, busy]);
  const submit = async () => {
    if (!canSubmit) return;
    setBusy(true); setErr(null); setKey(key);
    try { setR(await resolve(text, strategy, filters)); } catch (e) { setErr((e as Error).message); setR(null); } finally { setBusy(false); }
  };
  useEffect(() => { if (r) results.current?.scrollIntoView({ behavior: "smooth", block: "start" }); }, [r?.request_id]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <section className="card input">
        <label htmlFor="complaint"><h2>Customer complaint</h2></label>
        <textarea
          id="complaint" rows={4} maxLength={MAX_CHARS} value={text} onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(); }}
          placeholder="Paste the raw customer complaint…"
        />
        <div className="row small muted"><span>Try an example:</span><span className="grow" /><span>{text.length}/{MAX_CHARS} · Ctrl+Enter to resolve</span></div>
        <div className="examples">
          {EXAMPLES.map(([name, body]) => <button key={name} className="ghost" onClick={() => setText(body)}>{name}</button>)}
        </div>
        <div className="row gap wrap">
          <label className="field">Retrieval strategy
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>{STRATEGIES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
          </label>
          <label className="field check"><input type="checkbox" checked={filters} onChange={(e) => setFilters(e.target.checked)} /> narrow by predicted product</label>
          <label className="field">API key (if enabled)
            <input type="password" value={key} onChange={(e) => setKeyState(e.target.value)} placeholder="X-API-Key" autoComplete="off" />
          </label>
          <span className="grow" />
          {(r || text) && <button className="ghost" onClick={() => { setText(""); setR(null); setErr(null); }}>Clear</button>}
          <button className="primary" disabled={!canSubmit} onClick={submit}>{busy ? "Resolving…" : "Resolve"}</button>
        </div>
        {err && <p className="error" role="alert">⚠ {err}</p>}
      </section>

      <div ref={results} className="results">
        {busy && <Loading />}
        {!busy && r && (
          <>
            <Summary r={r} />
            <Detected r={r} />
            <Resolution r={r} hover={hover} setHover={setHover} />
            <Evidence r={r} hover={hover} setHover={setHover} />
            <Assurance r={r} />
            <Compare key={r.request_id} text={text} />
          </>
        )}
      </div>
    </>
  );
}
