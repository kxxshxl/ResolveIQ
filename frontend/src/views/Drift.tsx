import { useCallback, useEffect, useState } from "react";
import { getDriftHistory, getDriftStatus, getDriftTimeline, getJob, runDriftAnalysis } from "../api";
import type { DriftAlert, DriftCluster, DriftDimension, DriftHistoryItem, DriftReport, DriftTimeline } from "../types";
import { Skeleton, Tabs, ToastBar, label, pct, type Toast } from "../ui";
import Recurring from "./Recurring";

const DIMENSIONS: [string, string][] = [["intent", "Intent"], ["product", "Product"], ["severity", "Severity"], ["sentiment", "Sentiment"]];
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitForJob(id: string): Promise<{ status: string; error: string | null }> {
  for (let i = 0; i < 120; i++) {
    const j = await getJob(id);
    if (j.status === "succeeded" || j.status === "failed") return j;
    await sleep(2000);
  }
  throw new Error("The analysis is still running; refresh in a moment.");
}

const ago = (s?: number) => (s === undefined ? "" : s < 90 ? "just now" : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`);
const pv = (p: number | null | undefined) => (p == null ? "" : p < 0.001 ? `p=${p.toExponential(1)}` : `p=${p.toFixed(3)}`);
const WINDOW = (r: DriftReport) => (r.window_hours && r.window_hours % 24 === 0 ? `${r.window_hours / 24} d` : `${r.window_hours} h`);

function StatusBanner({ r }: { r: DriftReport }) {
  const tone = r.status === "alert" ? "bad" : r.status === "ok" ? "ok" : "warn";
  const text = r.status === "alert" ? `Drift detected: ${r.alerts?.length ?? 0} signal${r.alerts?.length === 1 ? "" : "s"}` : r.status === "ok" ? "No drift detected" : "Not enough traffic to judge";
  return (
    <div className={`summary ${tone}`} role="status">
      <strong>{text}</strong>
      <span>recent {r.n_recent} requests (last {WINDOW(r)}) vs baseline {r.n_baseline} (previous {r.baseline_days} d)</span>
      <span className="muted">analysed {ago(r.age_seconds)}</span>
    </div>
  );
}

function Alerts({ alerts }: { alerts: DriftAlert[] }) {
  if (!alerts.length) return <p className="muted small">Nothing crossed both the significance and the effect-size thresholds.</p>;
  return (
    <ul className="alerts">
      {alerts.map((a, i) => (
        <li key={i} className="alert-item">
          <span className={`pill ${a.kind === "emerging_cluster" ? "bad" : a.kind === "quality" ? "warn" : ""}`}>{a.kind.replace("_", " ")}</span>
          <span>{a.message}</span>
        </li>
      ))}
    </ul>
  );
}

function Compare({ dim, psiGate }: { dim: DriftDimension; psiGate: number }) {
  const rows = dim.categories.filter((c) => c.baseline_n + c.recent_n > 0).slice(0, 12);
  const max = Math.max(0.01, ...rows.flatMap((c) => [c.baseline_share, c.recent_share]));
  return (
    <>
      <p className="small">
        <span className={`pill ${dim.drifted ? "bad" : "ok"}`}>{dim.drifted ? "shifted" : "stable"}</span>{" "}
        <span className="muted">PSI {dim.psi.toFixed(2)} (an alert needs at least {psiGate.toFixed(2)} and significance) · {pv(dim.p_value)} after correcting for the number of categories</span>
      </p>
      <table className="cmp" aria-label="Baseline and recent share per category">
        <thead><tr><th>Category</th><th>Baseline</th><th>Recent</th><th>Change</th></tr></thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.label}>
              <td>{label(c.label)}{c.flag && <span className={`pill ${c.flag === "up" || c.flag === "new" ? "bad" : "warn"}`}>{c.flag}</span>}</td>
              <td><div className="bar" aria-hidden="true"><div className="bar-fill base" style={{ width: pct(c.baseline_share / max) }} /></div><span className="small muted">{pct(c.baseline_share)} · {c.baseline_n}</span></td>
              <td><div className="bar" aria-hidden="true"><div className={`bar-fill ${c.flag ? "warn" : ""}`} style={{ width: pct(c.recent_share / max) }} /></div><span className="small muted">{pct(c.recent_share)} · {c.recent_n}</span></td>
              <td className={c.delta > 0.03 ? "up" : c.delta < -0.03 ? "down" : "muted"}>{c.delta > 0 ? "+" : ""}{(c.delta * 100).toFixed(0)} pts</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function ClusterCard({ c, discovery }: { c: DriftCluster; discovery?: DriftReport["discovery"] }) {
  const linked = c.related_proposals.length > 0;
  return (
    <article className={`card proposal ${c.significant ? "" : "superseded"}`}>
      <div className="row wrap-start">
        <h3 className="proposal-title">{c.keywords.slice(0, 3).join(", ") || "unnamed group"}</h3>
        <span className={`pill ${c.significant ? "bad" : ""}`}>{c.significant ? (c.kind === "new_topic" ? "new topic" : "surging topic") : "watching"}</span>
        <span className={`pill ${c.covered_by_corpus ? "ok" : "warn"}`}>{c.covered_by_corpus ? "corpus explains it" : "corpus does not explain it"}</span>
        <span className="grow" />
        <span className="muted small">{c.size} recent complaints · {pv(c.p_value)}</span>
      </div>
      <div className="stats">
        <div className="stat" title="How many of this group's members are from the recent window, compared with what chance gives."><span className="muted small">Share recent</span><strong>{pct(c.recent_share_of_cluster)}</strong></div>
        <div className="stat" title="Mean evidence confidence of its complaints. Low means existing tickets and articles do not explain them."><span className="muted small">Mean evidence</span><strong>{c.mean_evidence == null ? "n/a" : pct(c.mean_evidence)}</strong></div>
        <div className="stat" title="Share of its complaints the system abstained on."><span className="muted small">Abstained</span><strong>{pct(c.abstained_share)}</strong></div>
        <div className="stat" title="Complaints from the baseline that fall in the same group."><span className="muted small">In baseline</span><strong>{c.baseline_members}</strong></div>
      </div>
      <div className="chips">{c.keywords.map((k) => <code key={k} className="chip">{k}</code>)}</div>
      <div className="examples-list">{c.examples.slice(0, 2).map((e, i) => <blockquote key={i}>{e}</blockquote>)}</div>
      <p className="small muted">Currently classified as: {Object.entries(c.intent_mix).map(([k, v]) => `${label(k)} (${v})`).join(", ") || "n/a"}</p>
      <div className="related">
        <strong className="small">Related discovery proposals</strong>
        {linked ? (
          <ul className="alerts">
            {c.related_proposals.map((p) => (
              <li key={p.proposal_id} className="alert-item">
                <span className={`pill ${p.status === "accepted" ? "ok" : p.status === "rejected" ? "bad" : "warn"}`}>{p.status}</span>
                <span>{p.recommendation === "new_class" ? "new intent" : "extends"} <strong>{label(p.label_id)}</strong> · covers {pct(p.overlap_share)} of this group ({p.overlap} complaints)</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="small muted">
            {c.covered_by_corpus ? "Not needed: existing tickets explain these complaints." : discovery?.triggered ? "A discovery run was queued for this group; its proposal will appear here." : "No proposal yet. Run discovery to turn this group into a reviewable proposal."}
          </p>
        )}
        {linked && <a className="small" href="#/discovery">Review in Class discovery →</a>}
        {!linked && !c.covered_by_corpus && <a className="small" href="#/discovery">Open Class discovery →</a>}
      </div>
    </article>
  );
}

function History({ items }: { items: DriftHistoryItem[] }) {
  if (!items.length) return <p className="muted small">No analyses yet.</p>;
  return (
    <div className="history" role="list" aria-label="Past analyses, oldest to newest">
      {[...items].reverse().map((h) => (
        <div key={h.snapshot_id} role="listitem" className={`hcell ${h.status}`} title={`${new Date(h.created_at).toLocaleString()} · ${h.status} · ${h.alert_count} alert(s), ${h.emerging_clusters} new-topic cluster(s) · ${h.n_recent} recent vs ${h.n_baseline} baseline`} />
      ))}
    </div>
  );
}

function Timeline({ tl }: { tl: DriftTimeline }) {
  const max = Math.max(1, ...tl.points.map((p) => p.n));
  if (!tl.points.length) return <p className="muted small">No requests logged in this period.</p>;
  return (
    <table className="cmp" aria-label="Requests per day">
      <thead><tr><th>{tl.bucket === "day" ? "Day" : "Hour"}</th><th>Requests</th><th>Abstained</th><th>Top intent</th></tr></thead>
      <tbody>
        {tl.points.map((p) => {
          const top = Object.entries(p.intents).sort((a, b) => b[1] - a[1])[0];
          return (
            <tr key={p.bucket}>
              <td>{new Date(p.bucket).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</td>
              <td><div className="bar" aria-hidden="true"><div className="bar-fill" style={{ width: pct(p.n / max) }} /></div><span className="small muted">{p.n}</span></td>
              <td className={p.abstention_rate > 0.2 ? "up" : "muted"}>{pct(p.abstention_rate)}</td>
              <td className="small">{top ? `${label(top[0])} (${pct(top[1] / p.n)})` : ""}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function Drift() {
  const [view, setView] = useState<"drift" | "recurring">("drift");
  return (
    <>
      <div className="page-head"><div className="grow"><h1>Drift monitoring</h1><p>Is live traffic moving away from what the system was built on, and which complaints keep coming back? Distribution drift is tested statistically; recurring groups are linked to the class-discovery proposals that cover them.</p></div></div>
      <Tabs variant="line" label="Drift views" value={view} onChange={setView} tabs={[{ id: "drift", name: "Distribution drift" }, { id: "recurring", name: "Recurring complaint clusters" }]} />
      {view === "drift" ? <DriftAnalysis /> : <Recurring />}
    </>
  );
}

function DriftAnalysis() {
  const [report, setReport] = useState<DriftReport | null>(null);
  const [history, setHistory] = useState<DriftHistoryItem[]>([]);
  const [timeline, setTimeline] = useState<DriftTimeline | null>(null);
  const [dim, setDim] = useState("intent");
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [toast, setToast] = useState<Toast | null>(null);
  const closeToast = useCallback(() => setToast(null), []);

  const reload = useCallback(async () => {
    try {
      const [r, h, t] = await Promise.all([getDriftStatus(), getDriftHistory(40), getDriftTimeline(14, "day")]);
      setReport(r); setHistory(h); setTimeline(t); setLoadErr(null);
    } catch (e) { setLoadErr((e as Error).message); }
  }, []);
  useEffect(() => { reload(); }, [reload]);

  const analyse = async () => {
    setRunning(true);
    try {
      const { job_id } = await runDriftAnalysis();
      const job = await waitForJob(job_id);
      if (job.status === "failed") throw new Error(job.error ?? "The analysis failed.");
      await reload();
      setToast({ tone: "ok", text: "Analysis finished." });
    } catch (e) { setToast({ tone: "bad", text: (e as Error).message }); } finally { setRunning(false); }
  };

  const none = report?.status === "no_analysis";
  const clusters = report?.emerging_clusters ?? [];
  return (
    <>
      <section className="card">
        <div className="row wrap-start">
          <div className="grow">
            <h2>Distribution drift tests</h2>
            <p className="muted small">
              Compares recent requests with a longer baseline just before them (DRIFT_WINDOW_HOURS and DRIFT_BASELINE_DAYS; the windows used are shown with each report). Every signal is a statistical test with a minimum effect size, so a quiet period does not alert and a small
              wobble in a large window does not either. Nothing here changes the taxonomy: new topics become proposals that a person reviews.
            </p>
          </div>
          <button className="primary" onClick={analyse} disabled={running}>{running ? "Analysing…" : "Run analysis now"}</button>
        </div>
        {running && <p className="muted small" role="status">The worker embeds a sample of recent complaints and runs the tests. This page checks every 2 seconds.</p>}
        {loadErr && <p className="error" role="alert">⚠ {loadErr} <button className="link" onClick={reload}>retry</button></p>}
        {!report && !loadErr && <Skeleton lines={3} />}
        {none && <p className="muted">{report?.note}</p>}
        {report && !none && <StatusBanner r={report} />}
        {report && !none && report.note && <p className="muted small">{report.note}</p>}
        {report && !none && <Alerts alerts={report.alerts ?? []} />}
      </section>

      {report && !none && report.distributions && (
        <section className="card">
          <h2>Current vs baseline distribution</h2>
          <div className="tabs" role="tablist" aria-label="Dimension">
            {DIMENSIONS.map(([k, name]) => (
              <button key={k} role="tab" aria-selected={dim === k} className={dim === k ? "tab active" : "tab"} onClick={() => setDim(k)}>
                {name}{report.distributions?.[k]?.drifted && <span className="count attn">!</span>}
              </button>
            ))}
          </div>
          {report.distributions[dim] && <Compare dim={report.distributions[dim]} psiGate={report.thresholds?.psi_threshold ?? 0.1} />}
          <div className="stats">
            {report.quality && (
              <>
                <div className="stat" title="Mean evidence confidence, baseline then recent."><span className="muted small">Evidence (mean)</span>
                  <strong>{report.quality.evidence.baseline ? pct(report.quality.evidence.baseline.mean) : "n/a"} → {report.quality.evidence.recent ? pct(report.quality.evidence.recent.mean) : "n/a"}</strong></div>
                <div className="stat" title="Share of requests the system declined to answer."><span className="muted small">Abstention</span>
                  <strong>{pct(report.quality.abstention.baseline_rate)} → {pct(report.quality.abstention.recent_rate)}</strong></div>
              </>
            )}
            {report.embedding?.status === "ok" && (
              <>
                <div className="stat" title="Cosine distance between the average complaint embedding of each window."><span className="muted small">Embedding shift</span>
                  <strong>{(report.embedding.centroid_distance ?? 0).toFixed(3)}</strong></div>
                <div className="stat" title={`Share of recent complaints farther from every baseline complaint than most baseline complaints are from each other (${pct(report.embedding.unseen_expected_rate ?? 0.05)} expected).`}><span className="muted small">Unlike baseline</span>
                  <strong>{pct(report.embedding.unseen_rate ?? 0)}</strong></div>
              </>
            )}
          </div>
          {report.embedding?.status === "skipped" && <p className="muted small">Embedding checks skipped: {report.embedding.reason}</p>}
        </section>
      )}

      {report && !none && (
        <>
          <h2 className="section-title">Emerging topics</h2>
          {clusters.length === 0 && (
            <section className="card empty"><strong>No emerging topics.</strong><p className="muted small">No group of recent complaints stands out from the baseline.</p></section>
          )}
          {clusters.map((c) => <ClusterCard key={c.cluster_id} c={c} discovery={report.discovery} />)}
          {report.discovery?.recommended && <p className="muted small">Discovery: {report.discovery.reason}</p>}
        </>
      )}

      <section className="card">
        <h2>Over time</h2>
        <p className="muted small">Each cell is one analysis (green stable, red drift, grey not enough data).</p>
        <History items={history} />
        {timeline && <Timeline tl={timeline} />}
      </section>
      <ToastBar toast={toast} onClose={closeToast} />
    </>
  );
}
