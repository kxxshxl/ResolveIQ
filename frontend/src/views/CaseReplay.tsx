import { useEffect, useMemo, useState } from "react";
import { compareCase, getCase, listCases, replayCase } from "../api";
import type { CaseDetail, CaseDiff, LabResult, ReplayResult } from "../consoleTypes";
import { useAsync } from "../hooks";
import { EmptyState, ErrorState, KV, Loading, Panel, Pill, Stat, Tabs, ago, label, ms, pct, when } from "../ui";
import EvidenceGraph, { type Selection } from "./EvidenceGraph";
import { LabColumns } from "./RetrievalLab";
import { ProvenanceTable, StageTimeline, statusText, statusTone } from "./shared";

const STRATEGIES = ["", "dense", "adaptive", "hybrid", "hybrid_reranked", "lexical"];

function Diff({ d, replay }: { d: CaseDiff; replay: ReplayResult["replay"] }) {
  const changes = Object.entries(d.provenance_changes);
  return (
    <div>
      <div className={`banner ${d.reproduced ? "ok" : "warn"}`} role="status">
        <strong>{d.reproduced ? "Reproduced exactly" : "The replay differs from the original"}</strong>
        <span>status {d.status.same ? d.status.new : `${d.status.old} → ${d.status.new}`}</span>
        <span>evidence {d.evidence.old != null ? `${d.evidence.old.toFixed(2)} → ${d.evidence.new.toFixed(2)}` : d.evidence.new.toFixed(2)}</span>
        <span>{d.steps.old_count} → {d.steps.new_count} steps</span>
      </div>
      {d.likely_reasons.length > 0 && <><h3>Why it may differ</h3><ul className="diff-list">{d.likely_reasons.map((w) => <li key={w}>{w}</li>)}</ul></>}
      {changes.length > 0 && (
        <><h3>Pipeline changes since the original</h3>
          <div className="tbl-wrap"><table className="tbl"><thead><tr><th>What</th><th>Original</th><th>Replay</th></tr></thead><tbody>{changes.map(([k, v]) => <tr key={k}><td>{label(k)}</td><td><code>{String(v.old ?? "n/a")}</code></td><td><code>{String(v.new ?? "n/a")}</code></td></tr>)}</tbody></table></div></>
      )}
      {Object.keys(d.classification).length > 0 && <><h3>Understanding changed</h3><ul className="diff-list">{Object.entries(d.classification).map(([k, v]) => <li key={k}>{k}: <span className="diff-del">{label(v.old)}</span> → <span className="diff-add">{label(v.new)}</span></li>)}</ul></>}
      <h3>Retrieved sources</h3>
      <p className="small">{d.retrieval.same_order ? "Same sources in the same order." : d.retrieval.same_set ? "Same sources, different order." : "Different sources."}</p>
      <ul className="diff-list small">
        {d.retrieval.added.map((s) => <li key={`a${s}`} className="diff-add">+ {s} (new)</li>)}{d.retrieval.removed.map((s) => <li key={`r${s}`} className="diff-del">− {s}</li>)}
        {d.retrieval.moved.map((m) => <li key={`m${m.id}`}>{m.id}: rank {m.old_rank} → {m.new_rank}</li>)}
      </ul>
      <h3>Steps</h3>
      {d.steps.same ? <p className="small">Same steps.</p> : <ul className="diff-list small">{d.steps.removed.map((s) => <li key={`r${s}`} className="diff-del">{s}</li>)}{d.steps.added.map((s) => <li key={`a${s}`} className="diff-add">{s}</li>)}</ul>}
      <p className="muted small">Replay answered with <code>{replay.generator}</code> in {ms(replay.latency_ms.total)}. It was not stored and does not appear in the case list.</p>
    </div>
  );
}

function Detail({ id }: { id: string }) {
  const c = useAsync(() => getCase(id), [id]);
  const [tab, setTab] = useState<"trace" | "lineage" | "provenance" | "sources" | "feedback">("trace");
  const [sel, setSel] = useState<Selection>(null);
  const [strategy, setStrategy] = useState("");
  const [generate, setGenerate] = useState(true);
  const [det, setDet] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [replay, setReplay] = useState<ReplayResult | null>(null);
  const [lab, setLab] = useState<LabResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { setReplay(null); setLab(null); setErr(null); setSel(null); setTab("trace"); }, [id]);

  if (c.loading && !c.data) return <Panel><Loading what="case" /></Panel>;
  if (c.error && !c.data) return <Panel><ErrorState message={c.error} onRetry={c.reload} /></Panel>;
  const d = c.data as CaseDetail;
  const run = async () => { setBusy("replay"); setErr(null); try { setReplay(await replayCase(id, { strategy, generate, deterministic: det })); } catch (e) { setErr((e as Error).message); } finally { setBusy(null); } };
  const compare = async () => { setBusy("compare"); setErr(null); try { setLab(await compareCase(id)); } catch (e) { setErr((e as Error).message); } finally { setBusy(null); } };
  const res = d.result;
  return (
    <>
      <Panel title={<>Case <code>{d.request_id.slice(0, 8)}</code></>} subtitle={`${when(d.created_at)} · trace ${d.trace_id}`} actions={<Pill tone={statusTone(d.status)}>{statusText[d.status] ?? d.status}</Pill>}>
        <blockquote>{d.complaint}</blockquote>
        <div className="stats">
          <Stat name="Intent" value={label(d.classification.intent)} /><Stat name="Confidence" value={pct(d.confidence)} /><Stat name="Latency" value={ms(d.latency_ms)} />
          <Stat name="Steps" value={res.resolution?.steps.length ?? 0} /><Stat name="Rating" value={d.feedback.length ? d.feedback[d.feedback.length - 1].rating.replace("_", " ") : "none"} tone={d.feedback.at(-1)?.rating === "not_helpful" ? "bad" : d.feedback.at(-1)?.rating === "helpful" ? "ok" : undefined} />
        </div>
        {res.resolution && res.resolution.steps.length > 0 && <ol className="small" style={{ margin: "6px 0 0", paddingLeft: 20 }}>{res.resolution.steps.map((s, i) => <li key={i}>{s.text} <span className="muted">[{s.citations.join(", ")}]</span></li>)}</ol>}
      </Panel>

      <Panel title="Replay" subtitle="Run this complaint through the pipeline as it is configured now, and see what changed and why. Replays are never stored.">
        <div className="row gap wrap">
          <label className="field">Retrieval<select value={strategy} onChange={(e) => setStrategy(e.target.value)}>{STRATEGIES.map((s) => <option key={s} value={s}>{s ? s.replace("_", " + ") : "Same as the original"}</option>)}</select></label>
          <label className="field check"><input type="checkbox" checked={generate} onChange={(e) => setGenerate(e.target.checked)} />Call the model</label>
          <label className="field check" title="Temperature 0 and a fixed seed, so a difference is not sampling noise"><input type="checkbox" checked={det} onChange={(e) => setDet(e.target.checked)} />Reproducible</label>
          <span className="grow" />
          <button className="btn" disabled={busy !== null} onClick={compare}>{busy === "compare" ? <><span className="spin" /> Comparing…</> : "Compare retrieval strategies"}</button>
          <button className="btn primary" disabled={busy !== null} onClick={run}>{busy === "replay" ? <><span className="spin" /> Replaying…</> : "Replay case"}</button>
        </div>
        {err && <ErrorState message={err} />}
        {replay && <Diff d={replay.diff} replay={replay.replay} />}
        {lab && <><h3>The same complaint through each retrieval strategy</h3><LabColumns lab={lab} /></>}
      </Panel>

      <Panel>
        <Tabs variant="line" label="Case details" value={tab} onChange={setTab} tabs={[{ id: "trace", name: "Stage trace" }, { id: "lineage", name: "Lineage" }, { id: "provenance", name: "Provenance" }, { id: "sources", name: "Sources", count: d.sources.length }, { id: "feedback", name: "Feedback", count: d.feedback.length }]} />
        {tab === "trace" && (d.trace ? <StageTimeline trace={d.trace} /> : <EmptyState title="No stage trace for this case">It was logged before traces were recorded. Replay it to see one.</EmptyState>)}
        {tab === "lineage" && (d.lineage ? <EvidenceGraph lineage={d.lineage} selected={sel} onSelect={setSel} /> : <EmptyState title="No lineage for this case">It was logged before lineage was recorded.</EmptyState>)}
        {tab === "provenance" && (d.provenance ? <ProvenanceTable p={d.provenance} /> : <EmptyState title="No provenance recorded" />)}
        {tab === "sources" && (
          <div className="tbl-wrap"><table className="tbl"><thead><tr><th>#</th><th>Source</th><th>Score</th><th>Method</th><th>Still in corpus</th><th>Text</th></tr></thead>
            <tbody>{d.sources.map((s) => <tr key={s.source_id}><td>{s.rank}</td><td><code>{s.source_id}</code></td><td>{s.score.toFixed(2)}</td><td>{s.method ?? "n/a"}</td><td>{s.in_corpus ? <Pill tone="ok">yes</Pill> : <Pill tone="warn">deleted since</Pill>}</td><td className="small">{s.excerpt ?? "—"}</td></tr>)}</tbody></table></div>
        )}
        {tab === "feedback" && (d.feedback.length === 0 ? <EmptyState title="No feedback on this case" /> : d.feedback.map((f) => (
          <div className="finding" key={f.feedback_id}>
            <div className="row"><Pill tone={f.rating === "helpful" ? "ok" : "bad"}>{f.rating.replace("_", " ")}</Pill><span className="muted small">{ago(f.created_at)}</span></div>
            {f.reasons.length > 0 && <div className="chips">{f.reasons.map((r) => <code key={r} className="chip">{r}</code>)}</div>}
            {f.corrected_intent && <p className="small">Corrected intent: <strong>{label(f.corrected_intent)}</strong></p>}
            {f.rejected_sources.length > 0 && <p className="small">Rejected sources: {f.rejected_sources.join(", ")}</p>}
            {f.comment && <blockquote>{f.comment}</blockquote>}
            {f.edited_steps && <><p className="small muted">Agent version:</p><ol className="small">{f.edited_steps.map((s, i) => <li key={i}>{s}</li>)}</ol></>}
          </div>
        )))}
      </Panel>
    </>
  );
}

export default function CaseReplay({ caseId, onSelect }: { caseId: string | null; onSelect: (id: string) => void }) {
  const [status, setStatus] = useState("");
  const [rated, setRated] = useState("");
  const list = useAsync(() => listCases({ limit: 40, status, rated }), [status, rated]);
  const items = list.data?.items ?? [];
  const active = useMemo(() => caseId ?? null, [caseId]);
  return (
    <>
      <div className="page-head"><div className="grow"><h1>Case replay</h1><p>Every resolution is stored with the stages it went through, what was retrieved and scored, and which model, prompt, taxonomy and corpus produced it. Open a case to debug it, or replay it through today's pipeline.</p></div></div>
      <div className="master">
        <div>
          <Panel title="Recent cases" subtitle={list.data ? `${list.data.total} total` : undefined}>
            <div className="row gap-s wrap" style={{ marginBottom: 8 }}>
              <label className="field" style={{ minWidth: 0, flex: 1 }}>Status<select value={status} onChange={(e) => setStatus(e.target.value)}><option value="">All</option>{["resolved", "abstained", "degraded", "unreliable"].map((s) => <option key={s}>{s}</option>)}</select></label>
              <label className="field" style={{ minWidth: 0, flex: 1 }}>Rating<select value={rated} onChange={(e) => setRated(e.target.value)}><option value="">All</option><option value="not_helpful">Not helpful</option><option value="helpful">Helpful</option><option value="none">Unrated</option></select></label>
            </div>
            {list.loading && !list.data && <Loading what="cases" />}
            {list.error && <ErrorState message={list.error} onRetry={list.reload} />}
            {list.data && items.length === 0 && <EmptyState title="No cases yet">Resolve a complaint and it will appear here.</EmptyState>}
            <div className="case-list" role="listbox" aria-label="Cases">
              {items.map((c) => (
                <button key={c.request_id} className="case" role="option" aria-selected={active === c.request_id} onClick={() => onSelect(c.request_id)}>
                  <span className="row"><Pill tone={statusTone(c.status)}>{c.status}</Pill>{c.rating && <Pill tone={c.rating === "helpful" ? "ok" : "bad"}>{c.rating === "helpful" ? "👍" : "👎"}</Pill>}<span className="grow" /><span className="muted small">{ago(c.created_at)}</span></span>
                  <span className="snip">{c.complaint}</span>
                  <span className="muted small">{label(c.intent)} · {pct(c.confidence)} · {ms(c.latency_ms)}{c.has_trace ? "" : " · no trace"}</span>
                </button>
              ))}
            </div>
          </Panel>
        </div>
        <div>{active ? <Detail id={active} /> : <Panel><EmptyState title="Select a case">Choose one on the left to see its stage trace, lineage and provenance, and to replay it.</EmptyState></Panel>}</div>
      </div>
    </>
  );
}
