import { useState } from "react";
import { listCases, qualityReport, qualitySummary } from "../api";
import type { IntentQuality, SourceQuality } from "../consoleTypes";
import { useAsync } from "../hooks";
import { Bar, CopyButton, EmptyState, ErrorState, Loading, Panel, Pill, Stat, Tabs, ago, label, pct } from "../ui";

function IntentBars({ rows }: { rows: IntentQuality[] }) {
  if (!rows.length) return <EmptyState title="No ratings per intent yet" />;
  return (
    <div>
      <div className="hbar small muted" aria-hidden="true"><span>Intent</span><span>Share of ratings that were "not helpful"</span><span style={{ textAlign: "right" }}>n</span></div>
      {rows.map((i) => (
        <div className="hbar" key={i.intent} title={`${i.not_helpful} of ${i.feedback} ratings; conservative (95% lower bound) rate ${pct(i.rejection_lower_bound)}`}>
          <span className="name">{label(i.intent)}</span><Bar value={i.rejection_rate ?? 0} tone={(i.rejection_lower_bound ?? 0) >= 0.3 ? "bad" : (i.rejection_rate ?? 0) >= 0.3 ? "warn" : ""} /><span className="small" style={{ textAlign: "right" }}>{pct(i.rejection_rate)} · {i.feedback}</span>
        </div>
      ))}
    </div>
  );
}

function Sources({ rows, empty }: { rows: SourceQuality[]; empty: string }) {
  if (!rows.length) return <EmptyState title={empty} />;
  return (
    <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Source</th><th className="num">Cited in rejected</th><th className="num">Cited in helpful</th><th className="num">Explicitly rejected</th><th>Intents</th></tr></thead>
      <tbody>{rows.map((s) => <tr key={s.source_id}><td><code>{s.source_id}</code> <span className="muted small">{s.type}</span></td><td className="num">{s.cited_in_rejected}</td><td className="num">{s.cited_in_helpful}</td><td className="num">{s.explicit_rejections}</td><td className="small">{s.intents.map(label).join(", ")}</td></tr>)}</tbody></table></div>
  );
}

export default function Quality({ onOpenCase }: { onOpenCase: (id: string) => void }) {
  const [days, setDays] = useState(30);
  const s = useAsync(() => qualitySummary(days), [days]);
  const rep = useAsync(() => qualityReport(days), [days]);
  const bad = useAsync(() => listCases({ limit: 5, rated: "not_helpful" }), [days]);
  const [tab, setTab] = useState<"intents" | "sources" | "abstention" | "failures">("intents");
  const t = s.data?.totals;
  return (
    <>
      <div className="page-head">
        <div className="grow"><h1>Feedback &amp; quality</h1><p>What agents say about the resolutions, turned into a deterministic, evidence-backed improvement report. It is advice for people: nothing here retrains a model or changes a label or a document.</p></div>
        <label className="field narrow">Window<select value={days} onChange={(e) => setDays(Number(e.target.value))}>{[7, 30, 90].map((d) => <option key={d} value={d}>Last {d} days</option>)}</select></label>
      </div>
      {s.error && <ErrorState message={s.error} onRetry={s.reload} />}
      {s.loading && !s.data && <Panel><Loading what="feedback analytics" /></Panel>}
      {t && s.data && (
        <>
          <div className="kpis">
            <Stat name="Requests" value={t.requests} hint="Resolutions logged in the window" />
            <Stat name="Ratings" value={t.feedback} hint={`${t.requests_with_feedback} requests were rated`} />
            <Stat name="Helpful" value={pct(t.helpful_rate)} tone={t.helpful_rate != null && t.helpful_rate >= 0.7 ? "ok" : t.helpful_rate != null && t.helpful_rate < 0.5 ? "bad" : undefined} />
            <Stat name="Not helpful" value={pct(t.rejection_rate)} tone={t.rejection_rate != null && t.rejection_rate >= 0.3 ? "bad" : undefined} />
            <Stat name="Rated" value={pct(t.coverage)} hint="Share of requests that received a rating" />
            <Stat name="Abstained" value={pct(s.data.abstention.rate)} hint="Requests the system declined to answer" />
          </div>
          {t.feedback === 0 && <Panel><EmptyState title="No ratings in this window yet">Agents rate resolutions on the Resolve page (helpful or not, with optional reasons, rejected sources and an edited version). Until then the report can only describe abstentions and failures.</EmptyState></Panel>}

          <div className="split">
            <Panel title="Where it falls short">
              <Tabs label="Quality breakdown" value={tab} onChange={setTab} tabs={[{ id: "intents", name: "Problem intents" }, { id: "sources", name: "Rejected sources", count: s.data.rejected_sources.length }, { id: "abstention", name: "Abstentions" }, { id: "failures", name: "Failure patterns", count: s.data.failure_patterns.length }]} />
              {tab === "intents" && <><IntentBars rows={s.data.problem_intents.length ? s.data.problem_intents : s.data.by_intent.filter((i) => i.feedback > 0).slice(0, 8)} /><p className="muted small">Ranked by the conservative (lower-bound) rejection rate, so three bad ratings do not outrank thirty. Intents with fewer than 3 ratings are not ranked.</p></>}
              {tab === "sources" && <><h3>Cited in rejected resolutions</h3><Sources rows={s.data.rejected_sources} empty="No source stands out" /><h3>Weak knowledge-base articles</h3><Sources rows={s.data.weak_articles} empty="No weak article found" /></>}
              {tab === "abstention" && (
                <><div className="stats"><Stat name="Abstained requests" value={s.data.abstention.count} /><Stat name="Rate" value={pct(s.data.abstention.rate)} /></div>
                  <h3>Why</h3>{Object.keys(s.data.abstention.by_reason).length ? Object.entries(s.data.abstention.by_reason).map(([k, v]) => <div className="hbar" key={k}><span className="name">{label(k)}</span><Bar value={v / Math.max(1, s.data!.abstention.count)} /><span className="small" style={{ textAlign: "right" }}>{v}</span></div>) : <EmptyState title="No abstentions" />}
                  <h3>Where (by detected intent)</h3>{Object.entries(s.data.abstention.by_intent).map(([k, v]) => <div className="hbar" key={k}><span className="name">{label(k)}</span><Bar value={v / Math.max(1, s.data!.abstention.count)} /><span className="small" style={{ textAlign: "right" }}>{v}</span></div>)}</>
              )}
              {tab === "failures" && (s.data.failure_patterns.length ? s.data.failure_patterns.map((p) => (
                <div className="finding" key={p.pattern}><div className="row"><strong>{p.pattern}</strong><span className="grow" /><Pill>{p.count} · {pct(p.share)}</Pill></div><p className="muted small">{p.detail}</p></div>
              )) : <EmptyState title="No failure patterns in this window" />)}
            </Panel>

            <Panel title="Agent signals">
              <h3>Reasons given</h3>
              {Object.keys(s.data.feedback_reasons).length ? <div className="chips">{Object.entries(s.data.feedback_reasons).map(([k, v]) => <code key={k} className="chip">{label(k)} × {v}</code>)}</div> : <p className="muted small">No reasons recorded.</p>}
              <h3>Intent corrections</h3>
              {s.data.intent_confusions.length ? s.data.intent_confusions.map((c) => <p key={c.predicted + c.corrected} className="small"><strong>{label(c.predicted)}</strong> → <strong>{label(c.corrected)}</strong> × {c.count}</p>) : <p className="muted small">No corrections.</p>}
              <h3>Agent-edited resolutions</h3>
              <p className="small">{t.edited_by_agent} of {t.feedback} ratings include the agent's own version. Open the case to read it.</p>
              <h3>Latest cases rated not helpful</h3>
              {(bad.data?.items ?? []).length === 0 && <p className="muted small">None.</p>}
              {(bad.data?.items ?? []).map((c) => <button key={c.request_id} className="case" style={{ marginBottom: 6 }} onClick={() => onOpenCase(c.request_id)}><span className="snip">{c.complaint}</span><span className="muted small">{label(c.intent)} · {ago(c.created_at)}</span></button>)}
            </Panel>
          </div>

          <Panel title="Improvement report" subtitle={rep.data?.note} actions={rep.data ? <CopyButton text={rep.data.markdown}>Copy as Markdown</CopyButton> : undefined}>
            {rep.loading && !rep.data && <Loading what="report" />}
            {rep.error && <ErrorState message={rep.error} onRetry={rep.reload} />}
            {rep.data && rep.data.findings.length === 0 && <EmptyState title="No findings">Nothing crossed the minimum-evidence thresholds (at least 3 ratings per item).</EmptyState>}
            {rep.data?.findings.map((f, i) => (
              <div key={i} className={`finding ${f.severity}`}>
                <div className="row"><strong>{f.title}</strong><span className="grow" /><Pill tone={f.severity === "high" ? "bad" : f.severity === "medium" ? "warn" : "info"}>{f.severity}</Pill></div>
                <div className="ev">{Object.entries(f.evidence).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join("  ")}</div>
                <div className="small">{f.suggested_action}</div>
              </div>
            ))}
          </Panel>
        </>
      )}
    </>
  );
}
