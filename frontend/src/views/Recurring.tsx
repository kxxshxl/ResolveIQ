import { useState } from "react";
import { recurringClusters } from "../api";
import type { RecurringCluster } from "../consoleTypes";
import { useAsync } from "../hooks";
import { Bar, EmptyState, ErrorState, Loading, Panel, Pill, Stat, ago, label, pct } from "../ui";

function Spark({ counts }: { counts: number[] }) {
  const max = Math.max(1, ...counts);
  return <div className="spark" role="img" aria-label={`Complaints per day, oldest to newest: ${counts.join(", ")}`}>{counts.map((c, i) => <i key={i} style={{ height: `${Math.max(8, (c / max) * 100)}%` }} title={`${c}`} />)}</div>;
}

function Card({ c }: { c: RecurringCluster }) {
  const tp = c.time_pattern;
  const tone = tp.label.startsWith("new") || tp.label.startsWith("rising") ? "warn" : tp.label.startsWith("fading") ? "neutral" : "info";
  return (
    <article className="panel proposal">
      <div className="row wrap-start">
        <h3 className="proposal-title">{c.keywords.slice(0, 3).join(", ") || "recurring complaint"}</h3>
        <Pill tone={tone}>{tp.label}</Pill>
        {c.proposals[0] ? <Pill tone={c.proposals[0].status === "accepted" ? "ok" : c.proposals[0].status === "rejected" ? "bad" : "warn"}>discovery proposal {c.proposals[0].status}</Pill> : <Pill>no discovery proposal yet</Pill>}
        {c.drift_cluster && <Pill tone="bad" title="This group is also flagged as a significant new topic by the drift analysis">also flagged by drift</Pill>}
        <span className="grow" /><span className="muted small">{c.size} distinct complaints{c.requests > c.size ? ` · ${c.requests} requests` : ""}</span>
      </div>
      <div className="stats">
        <Stat name="Common intent" value={`${label(c.intent)} ${pct(c.intent_share)}`} hint="The most common detected intent among these complaints, and its share" />
        <Stat name="Product" value={`${label(c.product)} ${pct(c.product_share)}`} />
        <Stat name="Confidence" value={pct(c.confidence)} hint={`Tightness of the group (${c.confidence_parts.cohesion.toFixed(2)} mean similarity to the centre) averaged with intent agreement (${pct(c.confidence_parts.intent_agreement)}). A heuristic, not a probability.`} />
        <Stat name="Last 24 h" value={`${tp.last_24h} (earlier ${tp.earlier_daily_mean}/day)`} />
        <Stat name="Mean evidence" value={c.mean_evidence == null ? "n/a" : pct(c.mean_evidence)} hint="How well existing tickets and articles explain these complaints" /><Stat name="Abstained" value={pct(c.abstained_share)} />
      </div>
      <div className="row"><span className="muted small">per day</span><Spark counts={tp.daily_counts} /><span className="muted small">first seen {ago(tp.first_seen)} · last {ago(tp.last_seen)}</span></div>
      <h3>Representative complaint</h3>
      <blockquote>{c.representative.complaint}</blockquote>
      {c.examples.length > 1 && <details><summary className="small muted" style={{ cursor: "pointer" }}>More examples</summary>{c.examples.slice(1).map((e) => <blockquote key={e.request_id}>{e.complaint}</blockquote>)}</details>}
      <div className="chips">{c.keywords.map((k) => <code key={k} className="chip">{k}</code>)}</div>
      <div className="related">
        <strong className="small">Taxonomy</strong>
        {c.proposals.length ? c.proposals.map((p) => <p className="small" key={p.proposal_id}>{p.recommendation === "new_class" ? "Proposes a new intent" : "Proposes extending"} <strong>{label(p.label_id)}</strong> ({p.status}), covering {pct(p.overlap_share)} of this group. <a href="#/discovery">Review in Discovery →</a></p>)
          : <p className="small muted">{c.mean_evidence != null && c.mean_evidence >= 0.8 ? "Existing tickets already explain these complaints: no new intent needed." : <>No proposal yet. <a href="#/discovery">Run discovery →</a></>}</p>}
      </div>
    </article>
  );
}

export default function Recurring() {
  const [days, setDays] = useState(7);
  const [min, setMin] = useState(3);
  const r = useAsync(() => recurringClusters(days, min), [days, min]);
  return (
    <>
      <div className="banner info" role="note">
        <strong>Support-side recurring complaint clusters.</strong>
        <span>Groups of customers describing a similar problem. They are not detected network incidents: nothing here looks at network telemetry.</span>
      </div>
      <div className="row gap wrap" style={{ marginTop: 0 }}>
        <label className="field narrow">Window<select value={days} onChange={(e) => setDays(Number(e.target.value))}>{[1, 3, 7, 14, 30].map((d) => <option key={d} value={d}>{d} day{d > 1 ? "s" : ""}</option>)}</select></label>
        <label className="field narrow">Minimum size<select value={min} onChange={(e) => setMin(Number(e.target.value))}>{[3, 4, 5, 8].map((d) => <option key={d} value={d}>{d} complaints</option>)}</select></label>
        {r.data && <span className="muted small">{r.data.stats.distinct} distinct complaints from {r.data.stats.requests} requests · {r.data.stats.clustered_complaints} in a cluster · {r.data.stats.elapsed_ms} ms{r.data.cached ? " (cached)" : ""}</span>}
      </div>
      {r.error && <ErrorState message={r.error} onRetry={r.reload} />}
      {r.loading && !r.data && <Panel><Loading what="clusters" /></Panel>}
      {r.data && r.data.clusters.length === 0 && <Panel><EmptyState title="No recurring groups in this window">Nothing was reported by at least {min} customers in a similar way. Widen the window or lower the minimum size.</EmptyState></Panel>}
      {r.data?.clusters.map((c) => <Card key={c.cluster_id} c={c} />)}
    </>
  );
}
