import { useMemo, useState } from "react";
import { compareRetrieval, labExamples } from "../api";
import type { LabKind, LabResult, LabStrategy } from "../consoleTypes";
import { useAsync } from "../hooks";
import { Bar, EmptyState, ErrorState, Kbd, Panel, Pill, Tabs, label, ms } from "../ui";

const ALL = ["lexical", "dense", "hybrid", "hybrid_reranked", "adaptive"];
const NAME: Record<string, string> = { lexical: "Keyword", dense: "Dense (semantic)", hybrid: "Hybrid", hybrid_reranked: "Hybrid + reranker", adaptive: "Adaptive" };

type KindKey = "ticket" | "article";

function Column({ s, kind, hover, setHover, best }: { s: LabStrategy; kind: KindKey; hover: string | null; setHover: (id: string | null) => void; best: boolean }) {
  const k: LabKind = s.kinds[kind];
  const sum = k.summary;
  return (
    <section className={`lab-col ${best ? "best" : ""}`} aria-label={s.label}>
      <header>
        <strong>{s.label}</strong>
        <span className="muted small">{s.description}</span>
        <span className="row gap-s wrap small">
          <span className="nowrap">{ms(k.latency_ms)}</span>
          {sum && <Pill tone={sum.first_relevant_rank === 1 ? "ok" : sum.first_relevant_rank ? "warn" : "bad"}>{sum.first_relevant_rank ? `first relevant: #${sum.first_relevant_rank}` : "no relevant result"}</Pill>}
        </span>
        {k.adaptive && k.adaptive.length > 0 && <span className="small muted">{k.adaptive.map((d) => `${d.stage}: ${d.action}`).join(" → ")}</span>}
      </header>
      <ol>
        {k.results.map((r) => (
          <li key={r.id} className={`res ${r.relevant ? "relevant" : ""} ${hover === r.id ? "hl" : ""}`} onMouseEnter={() => setHover(r.id)} onMouseLeave={() => setHover(null)}>
            <span className="row"><span className="rk">{r.rank}</span><code className="id">{r.id}</code><span className="grow" />
              {r.relevant === true && <span className="verdict ok">✓ relevant</span>}{r.relevant === false && <span className="verdict bad">✗ not relevant</span>}{r.relevant === null && <span className="verdict unknown">relevance unknown</span>}
            </span>
            <Bar value={Math.min(1, Math.max(0, r.score))} tone={r.relevant === false ? "warn" : r.relevant ? "ok" : ""} />
            <span className="sn">{r.excerpt}</span>
            <span className="muted small">{label(r.intent)} · score {r.score.toFixed(3)}{r.scores.rerank_prob != null ? ` · reranker ${r.scores.rerank_prob.toFixed(2)}` : ""}</span>
          </li>
        ))}
        {k.results.length === 0 && <li className="muted small">No results.</li>}
      </ol>
    </section>
  );
}

/** The result of one lab run: one column per strategy, identical sources highlight across columns on hover. */
export function LabColumns({ lab }: { lab: LabResult }) {
  const [kind, setKind] = useState<KindKey>("ticket");
  const [hover, setHover] = useState<string | null>(null);
  const best = useMemo(() => {
    if (!lab.ground_truth.available) return null;
    const scored = lab.strategies.map((s) => ({ n: s.name, v: s.kinds[kind].summary ? (s.kinds[kind].summary!.mrr) : -1 }));
    const top = Math.max(...scored.map((x) => x.v));
    return new Set(scored.filter((x) => x.v === top && top > 0).map((x) => x.n));
  }, [lab, kind]);
  return (
    <div>
      <div className="row wrap" style={{ marginBottom: 8 }}>
        <Tabs label="Result type" value={kind} onChange={setKind} tabs={[{ id: "ticket", name: "Resolved tickets" }, { id: "article", name: "KB articles" }]} />
        <span className="grow" />
        <span className="muted small">query embedded once in {ms(lab.embedding_ms)}; each column's latency is that strategy alone · hover a result to find it in the other columns</span>
      </div>
      {lab.ground_truth.available
        ? <p className="small"><Pill tone="ok">ground truth known</Pill> {lab.ground_truth.note} (<code>{lab.ground_truth.scenario_id}</code>)</p>
        : <p className="small"><Pill>relevance unknown</Pill> <span className="muted">{lab.ground_truth.note}</span></p>}
      <div className="lab-cols">{lab.strategies.map((s) => <Column key={s.name} s={s} kind={kind} hover={hover} setHover={setHover} best={!!best?.has(s.name)} />)}</div>
    </div>
  );
}

function Headline({ lab }: { lab: LabResult }) {
  if (!lab.ground_truth.available) return null;
  const rows = lab.strategies.map((s) => ({ s, t: s.kinds.ticket.summary }));
  const kw = rows.find((r) => r.s.name === "lexical")?.t, de = rows.find((r) => r.s.name === "dense")?.t;
  if (!kw || !de) return null;
  const f = (x: number | null) => (x ? `rank ${x}` : "not in the top 5");
  return (
    <div className={`banner ${(de.first_relevant_rank ?? 99) < (kw.first_relevant_rank ?? 99) ? "ok" : "info"}`}>
      <strong>Same complaint, different words</strong>
      <span>Keyword search puts the first relevant past ticket at <strong>{f(kw.first_relevant_rank)}</strong>; semantic search at <strong>{f(de.first_relevant_rank)}</strong>.</span>
    </div>
  );
}

export default function RetrievalLab() {
  const examples = useAsync(() => labExamples(), []);
  const [text, setText] = useState("");
  const [scenario, setScenario] = useState<string | undefined>();
  const [picked, setPicked] = useState("");
  const [chosen, setChosen] = useState<string[]>(ALL);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [lab, setLab] = useState<LabResult | null>(null);
  const run = async () => {
    if (text.trim().length < 5 || !chosen.length) return;
    setBusy(true); setErr(null);
    try { setLab(await compareRetrieval(text, ALL.filter((s) => chosen.includes(s)), scenario)); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  const byId = useMemo(() => new Map((examples.data ?? []).map((e) => [e.qid, e])), [examples.data]);
  return (
    <>
      <div className="page-head"><div className="grow"><h1>Retrieval lab</h1><p>One complaint through keyword, semantic, hybrid, reranked and adaptive search, side by side. Pick a labelled example to see which results are actually relevant; any other text is shown without guessing.</p></div></div>
      <Panel title="Complaint">
        <label htmlFor="lab-text" className="sr-only">Complaint</label>
        <textarea id="lab-text" rows={3} value={text} maxLength={4000} placeholder="Describe the problem in the customer's own words…" onChange={(e) => { setText(e.target.value); setScenario(undefined); setPicked(""); }}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); } }} />
        <div className="row gap wrap">
          <label className="field" style={{ minWidth: 280 }}>Labelled example (ground truth known)
            <select value={picked} onChange={(e) => { const x = byId.get(e.target.value); setPicked(e.target.value); if (x) { setText(x.text); setScenario(x.scenario_id); } }}>
              <option value="">{examples.loading ? "Loading…" : examples.error ? "Examples unavailable" : "Choose a labelled example…"}</option>
              {(examples.data ?? []).map((e) => <option key={e.qid} value={e.qid}>[{e.split}] {e.text.slice(0, 70)}…</option>)}
            </select>
          </label>
          <fieldset style={{ border: 0, margin: 0, padding: 0 }}><legend className="small muted">Strategies</legend>
            <div className="tag-grid">{ALL.map((s) => <label key={s} className="field check"><input type="checkbox" checked={chosen.includes(s)} onChange={() => setChosen(chosen.includes(s) ? chosen.filter((x) => x !== s) : [...chosen, s])} />{NAME[s]}</label>)}</div></fieldset>
          <span className="grow" />
          <button className="btn primary" disabled={busy || text.trim().length < 5 || !chosen.length} onClick={run}>{busy ? <><span className="spin" /> Searching…</> : <>Compare <Kbd>Ctrl</Kbd><Kbd>↵</Kbd></>}</button>
        </div>
        {err && <ErrorState message={err} />}
      </Panel>
      {!lab && !busy && <Panel><EmptyState title="No comparison yet">Choose a labelled example and press Compare. Keyword search struggles when the complaint uses different words from past tickets; that gap is the point of semantic search.</EmptyState></Panel>}
      {lab && <Panel title="Results" subtitle={lab.ground_truth.available ? "Green results belong to the same root-cause scenario as the query" : undefined}><Headline lab={lab} /><LabColumns lab={lab} /></Panel>}
    </>
  );
}
