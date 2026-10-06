import { useState, type ReactNode } from "react";
import { evaluationResults, getJob, runEvaluation } from "../api";
import type { EvalResults } from "../consoleTypes";
import { useAsync } from "../hooks";
import { EmptyState, ErrorState, Loading, Panel, Pill, Stat, Tabs, ago, fixed, label, ms, pct } from "../ui";

type Rec = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any -- recorded result files

const STRATS: [string, string][] = [["lexical", "Keyword (Postgres FTS)"], ["bm25", "BM25"], ["dense", "Dense (semantic)"], ["hybrid", "Hybrid (RRF)"], ["hybrid_reranked", "Hybrid + reranker"], ["adaptive", "Adaptive"]];

function Source({ f }: { f?: { path: string; written: string } }) {
  return f ? <p className="muted small">Recorded in <code>{f.path}</code>, written {ago(f.written)}. This is a recording, not a live measurement.</p> : null;
}

function best(rows: Rec[], key: string, high = true) {
  const vals = rows.map((r) => r[key]).filter((v) => typeof v === "number") as number[];
  return vals.length ? (high ? Math.max(...vals) : Math.min(...vals)) : null;
}

function RetrievalTable({ data }: { data: Rec }) {
  const rows = STRATS.filter(([k]) => data?.[k]).map(([k, name]) => ({ k, name, ...data[k], p50: data[k].latency?.p50_ms }));
  const b = { "hit@1": best(rows, "hit@1"), "hit@5": best(rows, "hit@5"), mrr: best(rows, "mrr"), "ndcg@10": best(rows, "ndcg@10"), p50: best(rows, "p50", false) };
  return (
    <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Strategy</th><th className="num">Hit@1</th><th className="num">Hit@5</th><th className="num">MRR</th><th className="num">nDCG@10</th><th className="num">p50 latency</th></tr></thead>
      <tbody>{rows.map((r) => <tr key={r.k}><td>{r.name}</td>{(["hit@1", "hit@5", "mrr", "ndcg@10"] as const).map((c) => <td key={c} className={`num ${r[c] === b[c] ? "best" : ""}`}>{fixed(r[c], 3)}</td>)}<td className={`num ${r.p50 === b.p50 ? "best" : ""}`}>{ms(r.p50)}</td></tr>)}</tbody></table></div>
  );
}

function Retrieval({ r, f }: { r: Rec; f?: { path: string; written: string } }) {
  const splits = ["gold", "blind", "test"].filter((s) => r.retrieval?.tickets?.[s]);
  const [split, setSplit] = useState(splits[0] ?? "gold");
  const [kind, setKind] = useState<"tickets" | "articles">("tickets");
  const d = r.retrieval?.[kind]?.[split];
  const n = d ? Object.values<Rec>(d)[0]?.n : null;
  return (
    <Panel title="Retrieval quality" subtitle="Which strategy puts a past resolution from the same root cause first? Bold = best in column.">
      <div className="row wrap"><Tabs label="Query set" value={split} onChange={setSplit} tabs={splits.map((s) => ({ id: s, name: s === "gold" ? "Hand-written (gold)" : s === "blind" ? "Blind (hand-written, never tuned on)" : "Templated paraphrases" }))} />
        <Tabs label="Corpus" value={kind} onChange={setKind} tabs={[{ id: "tickets", name: "Resolved tickets" }, { id: "articles", name: "KB articles" }]} /></div>
      {d ? <RetrievalTable data={d} /> : <EmptyState title="No retrieval results recorded" />}
      <p className="muted small">{n ? `${n} queries. ` : ""}Hand-written sets are the honest numbers: the templated set shares vocabulary with the generator that produced the corpus.</p>
      <Source f={f} />
    </Panel>
  );
}

function Adaptive({ r, f }: { r: Rec; f?: { path: string; written: string } }) {
  const a = r.adaptive;
  if (!a) return <Panel title="Adaptive retrieval"><EmptyState title="Not recorded yet">Run <code>python -m app.evaluation.run --suites adaptive</code>.</EmptyState></Panel>;
  const held = a.config.held_out as string[];
  return (
    <Panel title="Adaptive retrieval" subtitle={`Dense first; adds a keyword leg only when rank 1 and rank 2 are nearly tied. Thresholds tuned on ${a.config.tuned_on}; judged on ${held.join(" and ")} only.`}>
      <div className="grid2">
        {held.flatMap((split) => (["ticket", "article"] as const).map((kind) => {
          const d = a.results[split][kind];
          const lad = a.ladder[split][kind];
          const diff = a.paired_differences[split][kind].adaptive_minus_dense.mrr;
          return (
            <div key={split + kind}>
              <h3>{split} · {kind === "ticket" ? "tickets" : "articles"} <span className="muted small">(n={d.dense.n})</span></h3>
              <table className="tbl"><thead><tr><th>Strategy</th><th className="num">Hit@1</th><th className="num">MRR</th><th className="num">mean latency</th><th className="num">reranker passages</th></tr></thead>
                <tbody>{["dense", "hybrid", "hybrid_reranked", "adaptive"].map((s) => <tr key={s}><td>{label(s)}</td><td className="num">{fixed(d[s]["hit@1"], 3)}</td><td className="num">{fixed(d[s].mrr, 3)}</td><td className="num">{ms(d[s].latency.avg_ms)}</td><td className="num">{d[s].cost.reranker_passages_per_query}</td></tr>)}</tbody></table>
              <p className="small muted">Adaptive stopped at dense for {pct(lad.stopped_at_dense)} of queries. MRR difference vs dense: {fixed(diff.mean_diff, 3)} (95% interval {fixed(diff.ci95[0], 3)} to {fixed(diff.ci95[1], 3)}){diff.ci95[0] <= 0 && diff.ci95[1] >= 0 ? ": not distinguishable from zero." : "."}</p>
            </div>
          );
        }))}
      </div>
      {a.ablations && (
        <><h3>Ablations on the held-out queries ({a.ablations.n_queries})</h3>
          <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Variant</th><th className="num">ticket MRR</th><th className="num">article MRR</th><th className="num">article Hit@1</th><th className="num">article mean ms</th></tr></thead>
            <tbody>{Object.entries<Rec>(a.ablations).filter(([k]) => k !== "n_queries").map(([k, v]) => <tr key={k}><td>{label(k)}</td><td className="num">{fixed(v.ticket.mrr, 3)}</td><td className="num">{fixed(v.article.mrr, 3)}</td><td className="num">{fixed(v.article["hit@1"], 3)}</td><td className="num">{v.article.mean_ms}</td></tr>)}</tbody></table></div></>
      )}
      <Source f={f} />
    </Panel>
  );
}

function Classification({ r, f }: { r: Rec; f?: { path: string; written: string } }) {
  const c = r.classification?.strategies;
  if (!c) return null;
  const dims = ["intent", "product", "severity", "sentiment"];
  const rows: [string, Rec | undefined][] = [["Rules + kNN only (before the affect model) · gold", c.ensemble_legacy?.gold], ["Shipped (NLI affect model) · gold", c.ensemble?.gold], ["Rules + kNN only · blind", c.ensemble_legacy?.blind], ["Shipped · blind", c.ensemble?.blind]];
  return (
    <Panel title="Classification" subtitle="Accuracy of intent, product, severity and sentiment on hand-written complaints">
      <div className="tbl-wrap"><table className="tbl"><thead><tr><th>System · set</th>{dims.map((d) => <th key={d} className="num">{d}</th>)}</tr></thead>
        <tbody>{rows.filter(([, v]) => v).map(([n, v]) => <tr key={n}><td>{n}</td>{dims.map((d) => <td key={d} className="num">{fixed(v![d].accuracy, 2)}</td>)}</tr>)}</tbody></table></div>
      <Source f={f} />
    </Panel>
  );
}

function Grounding({ r, f }: { r: Rec; f?: { path: string; written: string } }) {
  const g = r.rag?.primary, e = r.e2e;
  if (!g && !e) return null;
  return (
    <Panel title="Grounding, citations and abstention" subtitle={g ? `Generator ${g.generator}; ${g.answered} answered complaints` : undefined}>
      {g && <div className="kpis">
        <Stat name="Step faithfulness" value={fixed(g.step_faithfulness, 3)} hint="Share of steps whose words are found in the cited evidence (independent of the validator)" />
        <Stat name="Citation validity" value={fixed(g.citation_validity, 3)} hint="Share of emitted citation ids that were really retrieved" />
        <Stat name="Citation coverage" value={fixed(g.citation_coverage, 3)} /><Stat name="Citation precision" value={fixed(g.citation_precision_vs_gold, 3)} hint="Cited sources that belong to the right scenario" />
        <Stat name="Gold-step recall" value={fixed(g.gold_step_recall, 3)} hint="Share of the reference procedure's steps the answer covers" /></div>}
      {e && <div className="kpis">
        <Stat name="Resolved correctly" value={pct(e.in_domain.correct_resolution_rate)} hint={`${e.in_domain.n} in-domain complaints`} /><Stat name="False abstention" value={pct(e.in_domain.false_abstention_rate)} />
        <Stat name="Out-of-domain abstained" value={pct(e.out_of_domain.correct_abstention_rate)} hint={`${e.out_of_domain.n} unrelated requests`} /><Stat name="Pipeline errors" value={e.failures.pipeline_errors} />
        <Stat name="Latency p50 / p95" value={`${ms(e.latency_total.p50_ms)} / ${ms(e.latency_total.p95_ms)}`} hint="Dominated by local LLM generation" /></div>}
      <Source f={f} />
    </Panel>
  );
}

function Evolving({ r, f }: { r: Rec; f?: { path: string; written: string } }) {
  const d = r.discovery, c = r.clustering, ev = r.evolving;
  if (!d && !c && !ev) return null;
  return (
    <Panel title="Evolving data and intents" subtitle="New resolved tickets, new intents nobody named, and recurring complaint groups">
      <div className="kpis">
        {ev && <Stat name="New intents searchable" value={`${fixed(ev.before_ingestion?.["hit@5"], 2)} → ${fixed(ev.after_ingestion?.["hit@5"], 2)}`} hint="Hit@5 before and after ingesting 2 new intents, no restart" />}
        {d && <><Stat name="Unseen intents recovered" value={`${d.proposals.recovered_classes}/${d.proposals.of}`} hint="Clusters proposing a new intent" /><Stat name="Proposal precision" value={fixed(d.proposals.precision, 2)} /><Stat name="Candidate recall" value={pct(d.candidate_selection.novel_recall)} hint="Novel complaints that low evidence flags" /><Stat name="Abstention alone catches" value={pct(d.candidate_selection.abstention_only_novel_recall)} /></>}
        {c && <><Stat name="Clusters purity vs intent" value={fixed(c.default.vs_intent.purity, 2)} hint={`${c.default.clusters} clusters over ${c.complaints} complaints`} /><Stat name="Scenario recovery" value={`${c.default.vs_scenario.recovered}/${c.default.vs_scenario.of}`} /><Stat name="Unseen topics as own clusters" value={`${c.with_unseen_topics.recovered}/${c.with_unseen_topics.of}`} /></>}
      </div>
      <Source f={f} />
    </Panel>
  );
}

function Drift({ r, f }: { r: Rec; f?: { path: string; written: string } }) {
  const d = r.drift_demo;
  if (!d) return null;
  const c = d.control;
  return (
    <Panel title="Drift detection (injection demo)" subtitle="Known changes injected into otherwise unchanged traffic; the statistics are the production code, the traffic and evidence are stand-ins">
      <div className="kpis">{c.map((x: Rec) => <Stat key={x.n_baseline + String(x.distinct_complaints)} name={`False alarms, nothing changed${x.distinct_complaints ? " (distinct)" : ""}`} value={`${x.false_alarms}/${x.trials}`} hint={`95% interval ${pct(x.ci95[0])} to ${pct(x.ci95[1])}`} tone={x.false_alarm_rate <= 0.02 ? "ok" : "warn"} />)}</div>
      <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Injected change</th><th className="num">Detected</th><th>95% interval</th></tr></thead>
        <tbody>{Object.entries<Rec>(d.scenarios).map(([k, v]) => <tr key={k}><td>{v.description}</td><td className="num">{v.detected}/{v.trials}</td><td className="small">{pct(v.ci95[0])} to {pct(v.ci95[1])}</td></tr>)}</tbody></table></div>
      <Source f={f} />
    </Panel>
  );
}

function Scale({ r, f }: { r: Rec; f?: { path: string; written: string } }) {
  const s = r.pgvector_scale, p = r.db_query_plans, ib = r.ingest_benchmark;
  if (!s && !p) return null;
  return (
    <Panel title="Database engineering" subtitle="Synthetic experiments in throw-away databases: not production benchmarks">
      {s && (
        <>
          <h3>pgvector at {s.setup.vectors.toLocaleString()} synthetic {s.setup.dimensions}-d vectors</h3>
          <div className="stats"><Stat name="Index build" value={`${s.index_build.seconds} s`} hint={s.index_build.note} /><Stat name="Load" value={`${s.load.seconds} s`} /><Stat name="HNSW index" value={`${s.storage.hnsw_index_mb} MB`} /><Stat name="Table" value={`${s.storage.table_mb} MB`} /><Stat name="8 threads" value={`${s.concurrency.queries_per_second} qps`} hint={`p95 ${s.concurrency.p95_ms} ms`} /></div>
          <div className="tbl-wrap"><table className="tbl"><thead><tr><th>ef_search</th><th className="num">recall@10</th><th className="num">p50</th><th className="num">p95</th><th className="num">p99</th></tr></thead>
            <tbody>{Object.entries<Rec>(s.search).map(([k, v]) => <tr key={k}><td>{k.replace("ef_search=", "")}</td><td className="num">{fixed(v["recall@10"], 3)}</td><td className="num">{ms(v.p50_ms)}</td><td className="num">{ms(v.p95_ms)}</td><td className="num">{ms(v.p99_ms)}</td></tr>)}</tbody></table></div>
          <h3>Filtered search</h3>
          <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Filter</th><th className="num">recall (iterative scan off)</th><th className="num">recall (relaxed order)</th></tr></thead>
            <tbody>{Object.entries<Rec>(s.filtered_search).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{fixed(v["iterative_scan=off"]["recall@10"], 3)}{v["iterative_scan=off"].queries_returning_fewer_than_k ? ` (${v["iterative_scan=off"].queries_returning_fewer_than_k}% short)` : ""}</td><td className="num">{fixed(v["iterative_scan=relaxed_order"]["recall@10"], 3)}</td></tr>)}</tbody></table></div>
          <p className="small muted">Sanity: {Object.entries<string>(s.sanity).map(([k, v]) => `${label(k)} ${v}`).join(" · ")}</p>
        </>
      )}
      {p?.foreign_key_index && <p className="small"><Pill tone="ok">index review</Pill> the foreign-key index on feedback made deleting 300 requests {p.foreign_key_index.speedup}× faster ({p.foreign_key_index.delete_300_requests_without_index_s} s → {p.foreign_key_index.delete_300_requests_with_index_s} s) and the per-case rating lookup {Math.round(p.foreign_key_index.latest_rating_query_ms_without_index / p.foreign_key_index.latest_rating_query_ms_with_index)}× faster, at {p.foreign_key_index.feedback_rows.toLocaleString()} feedback rows.</p>}
      {ib && <p className="small"><Pill tone="ok">batched ingestion</Pill> {ib.speedup}× faster at the database layer ({ib.per_ticket_transactions.tickets_per_second} → {ib.one_batched_transaction.tickets_per_second} tickets/s).</p>}
      <Source f={f} />
    </Panel>
  );
}

function Load({ r, f }: { r: Rec; f?: { path: string; written: string } }) {
  const l = r.load_test;
  if (!l) return null;
  return (
    <Panel title="Load test" subtitle={l.description}>
      <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Scenario</th><th className="num">Users</th><th className="num">OK req/s</th><th className="num">P50</th><th className="num">P95</th><th className="num">Errors</th></tr></thead>
        <tbody>{(l.rows as Rec[]).map((x, i) => <tr key={i}><td>{x.scenario}</td><td className="num">{x.users}</td><td className="num">{x.rps}</td><td className="num">{ms(x.p50)}</td><td className="num">{ms(x.p95)}</td><td className="num">{x.errors}</td></tr>)}</tbody></table></div>
      <p className="muted small">{l.caveat}</p><Source f={f} />
    </Panel>
  );
}

export default function Evaluation() {
  const res = useAsync<EvalResults>(() => evaluationResults(), []);
  const [job, setJob] = useState<{ state: string; result?: Rec } | null>(null);
  const quick = async () => {
    setJob({ state: "queued" });
    try {
      const { job_id } = await runEvaluation(["retrieval", "adaptive"], 30);
      for (let i = 0; i < 120; i++) {
        const j = await getJob(job_id);
        setJob({ state: j.status, result: j.result as Rec | undefined });
        if (j.status === "succeeded" || j.status === "failed") { if (j.status === "failed") setJob({ state: `failed: ${j.error ?? ""}` }); return; }
        await new Promise((r) => setTimeout(r, 2000));
      }
    } catch (e) { setJob({ state: `failed: ${(e as Error).message}` }); }
  };
  const d = res.data as Rec | null;
  const files = (k: string) => d?.files?.[k];
  const meta = d?.latest?.meta;
  let body: ReactNode = null;
  if (res.loading && !d) body = <Panel><Loading what="evaluation results" /></Panel>;
  else if (res.error) body = <ErrorState message={res.error} onRetry={res.reload} />;
  else if (d && !d.latest) body = <Panel><EmptyState title="No evaluation recorded">Run <code>python -m app.evaluation.run</code> to produce <code>data/eval/results/latest.json</code>.</EmptyState></Panel>;
  else if (d) body = (
    <>
      <Retrieval r={d.latest} f={files("latest")} /><Adaptive r={d.latest} f={files("latest")} /><Classification r={d.latest} f={files("latest")} /><Grounding r={d.latest} f={files("latest")} /><Evolving r={d.latest} f={files("latest")} />
      <Drift r={d} f={files("drift_demo")} /><Scale r={d} f={files("pgvector_scale")} /><Load r={d} f={files("load_test")} />
    </>
  );
  return (
    <>
      <div className="page-head"><div className="grow"><h1>Evaluation</h1><p>Every number on this page was measured by a script in this repository and read from a result file; none is typed in. Held-out hand-written queries are the headline, templated ones are shown for comparison.</p></div>
        <button className="btn" disabled={job !== null && !/^(succeeded|failed)/.test(job.state)} onClick={quick}>{job && !/^(succeeded|failed)/.test(job.state) ? <><span className="spin" /> Running…</> : "Run a quick live check"}</button></div>
      {meta && <div className="stats"><Stat name="Recorded" value={ago(meta.timestamp)} /><Stat name="Embedding model" value={String(meta.embedding_model).split("/").pop()} /><Stat name="LLM" value={(meta.llm ?? []).join(", ") || "none"} /><Stat name="Corpus" value={`${meta.corpus.tickets} tickets, ${meta.corpus.articles} articles`} /></div>}
      {job?.result?.adaptive && (
        <Panel title="Live quick check" subtitle="30 queries per split on the running system; not saved"><div className="grid2">{(Object.entries<Rec>(job.result.adaptive.results).slice(0, 2)).map(([split, k]) => <div key={split}><h3>{split}</h3><RetrievalTable data={k.ticket} /></div>)}</div></Panel>
      )}
      {job && /^failed/.test(job.state) && <ErrorState message={job.state} />}
      {body}
    </>
  );
}
