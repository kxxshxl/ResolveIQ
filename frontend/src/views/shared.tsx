import type { Lineage, LineageSource, Provenance, StageRecord } from "../consoleTypes";
import { Bar, KV, Pill, label, ms, pct, type Tone } from "../ui";
import type { Selection } from "./EvidenceGraph";

const fmt = (v: unknown): string => (v == null ? "n/a" : typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : typeof v === "object" ? JSON.stringify(v) : String(v));

const STAGE_TEXT: Record<string, string> = {
  preprocess: "clean and redact", cache: "response cache", embed: "embed the complaint", classify: "understand", retrieve: "retrieve", rerank: "rerank", evidence: "evidence gate", generate: "generate", validate: "validate citations",
};

/** The stage-by-stage trace of one resolution: where the time went, what each stage decided, no customer text. */
export function StageTimeline({ trace }: { trace: StageRecord[] }) {
  const max = Math.max(1, ...trace.map((s) => s.latency_ms ?? 0));
  return (
    <div>
      <div className="timeline">
        {trace.map((s, i) => (
          <div key={`${i}-${s.name}`} className="stage-row" title={STAGE_TEXT[s.name]}>
            <span className="name">{s.name}</span>
            <div className={`stage-bar ${s.status}`}><i style={{ width: `${Math.max(2, ((s.latency_ms ?? 0) / max) * 100)}%` }} /></div>
            <span className="small nowrap">{s.status === "skipped" ? "skipped" : ms(s.latency_ms)}</span>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 10 }}>
        {trace.map((s, i) => (
          <details className="stage" key={`${i}-${s.name}`}>
            <summary><strong style={{ textTransform: "capitalize" }}>{s.name}</strong><Pill tone={s.status === "ok" ? "ok" : s.status === "error" ? "bad" : s.status === "degraded" ? "warn" : "neutral"}>{s.status}</Pill><span className="muted small">{STAGE_TEXT[s.name]}</span></summary>
            <pre>{Object.entries(s.detail).map(([k, v]) => `${k}: ${fmt(v)}`).join("\n") || "(no details)"}</pre>
          </details>
        ))}
      </div>
    </div>
  );
}

export function ProvenanceTable({ p }: { p: Provenance }) {
  const g = p.generation;
  const r = p.retrieval as Record<string, unknown>;
  const rows: [string, React.ReactNode][] = [
    ["Pipeline version", p.pipeline_version], ["Generator", <code key="g">{p.generator}</code>], ["Prompt", p.prompt_version ? `${p.prompt_version} · ${p.prompt_hash}` : "no model call"],
    ["Taxonomy version", fmt(p.taxonomy_version)], ["Corpus version", fmt(p.corpus_version)], ["Embedding model", p.embedding_model ?? "n/a"], ["Reranker", p.reranker ?? "off"],
    ["Retrieval strategy", fmt(r.strategy)], ["Abstain / grounding thresholds", `${p.thresholds.abstain} / ${p.thresholds.grounding}`],
  ];
  if (g.temperature != null) rows.push(["Temperature · seed", `${fmt(g.temperature)} · ${g.seed ?? "none"}${g.deterministic ? " (deterministic)" : ""}`]);
  if (g.est_tokens != null) rows.push(["Prompt size", `${g.est_tokens} of ${g.budget_tokens} tokens budgeted${Array.isArray(g.collapsed_duplicates) && g.collapsed_duplicates.length ? ` · ${g.collapsed_duplicates.length} duplicate resolution(s) collapsed` : ""}${Array.isArray(g.dropped_for_budget) && g.dropped_for_budget.length ? ` · ${g.dropped_for_budget.length} dropped` : ""}`]);
  if (g.prompt_tokens != null) rows.push(["Model tokens in / out", `${g.prompt_tokens} / ${g.completion_tokens ?? "n/a"}`]);
  return <KV rows={rows} />;
}

function signalTone(v: number | null | undefined, good = 0.7, ok = 0.5): "ok" | "warn" | "bad" | "" { return v == null ? "" : v >= good ? "ok" : v >= ok ? "warn" : "bad"; }

/** Safe, auditable quality signals (no model reasoning is shown or stored). */
export function Signals({ lineage }: { lineage: Lineage }) {
  const s = lineage.signals as Record<string, number | string | boolean | null>;
  const meters: [string, string, number | null | undefined, string][] = [
    ["Evidence strength", "Combined similarity, reranker and agreement score that the abstention gate uses", s.evidence_strength as number, ""],
    ["Source agreement", "Share of the top tickets that agree on one intent", s.source_agreement as number, ""],
    ["Best similarity", "Highest cosine similarity of any retrieved source", s.similarity as number, ""],
    ["Reranker signal", "Best cross-encoder probability (empty when no reranker ran)", s.reranker_signal as number | null, ""],
    ["Citation coverage", "Steps that carry at least one valid citation", s.citation_coverage as number, ""],
    ["Grounded steps", "Steps supported by the text they cite", s.grounded_ratio as number, ""],
    ["Mean step support", "Average support of a step by its cited evidence", s.mean_step_support as number | null, ""],
  ];
  return (
    <div>
      <div className="signals">
        {meters.map(([name, hint, v]) => (
          <div className="signal" key={name} title={hint}><span className="name">{name}</span><strong>{v == null ? "n/a" : v.toFixed(2)}</strong>{v != null && <Bar value={v} tone={signalTone(v)} />}</div>
        ))}
      </div>
      <KV rows={[
        ["Sources retrieved / selected / cited", `${s.sources_retrieved} / ${s.sources_selected} / ${s.sources_cited}`],
        ["Uncertainty stated by the model", (s.uncertainty as string) || "none"],
        ["Abstention reason", (s.abstention_reason as string) || "not abstained"],
        ["Escalation", s.escalate ? ((s.escalation_reason as string) || "recommended") : "not required"],
      ]} />
      <ul className="small" style={{ margin: "8px 0 0", paddingLeft: 18 }}>
        {Object.entries(lineage.checks).map(([k, v]) => <li key={k} className={v ? "" : "error"}>{v ? "✓" : "✗"} {k.replace(/_/g, " ")}</li>)}
      </ul>
    </div>
  );
}

/** Retrieved sources with the reasons they were retrieved. `selected` drives highlighting in both directions with the steps. */
export function SourceList({ sources, selected, onSelect, citedBy }: { sources: LineageSource[]; selected: Selection; onSelect: (s: Selection) => void; citedBy: (id: string) => number[] }) {
  const stepSel = selected?.kind === "step" ? selected.id : null;
  const group = (type: "ticket" | "article", title: string) => {
    const items = sources.filter((s) => s.type === type);
    return (
      <div>
        <h3>{title} <span className="muted small">({items.length})</span></h3>
        {items.length === 0 && <p className="muted small">None retrieved.</p>}
        {items.map((s) => {
          const cited = citedBy(s.id);
          const isSel = selected?.kind === "source" && selected.id === s.id;
          const hl = isSel || (stepSel != null && cited.includes(stepSel));
          const dim = selected != null && !hl;
          return (
            <button key={s.id} className={`src ${cited.length ? "cited" : ""} ${hl ? "hl" : ""} ${dim ? "dim" : ""}`} aria-pressed={isSel} onClick={() => onSelect(isSel ? null : { kind: "source", id: s.id })}>
              <span className="row wrap" style={{ alignItems: "center" }}>
                <span className="muted small">#{s.rank}</span><code className="id">{s.id}</code>
                {cited.length > 0 ? <Pill tone="ok">cited by step {cited.join(", ")}</Pill> : s.selected ? <Pill tone="info">in evidence, not cited</Pill> : <Pill>retrieved only</Pill>}
                <span className="grow" /><strong>{s.score.toFixed(2)}</strong>
              </span>
              <Bar value={s.score} />
              <span className="title">{s.type === "ticket" ? s.excerpt : s.title}</span>
              <span className="muted small">{label(s.intent)}{s.matches_intent ? " · same intent" : ""}</span>
              {(isSel || stepSel != null && cited.includes(stepSel)) && (
                <ul className="why">{s.why.map((w) => <li key={w}>{w}</li>)}</ul>
              )}
            </button>
          );
        })}
      </div>
    );
  };
  return <div>{group("ticket", "Resolved tickets")}{group("article", "Knowledge-base articles")}</div>;
}

export const statusTone = (s: string): Tone => (s === "resolved" ? "ok" : s === "degraded" ? "warn" : "bad");
export const statusText: Record<string, string> = { resolved: "Grounded resolution", degraded: "Evidence only (LLM unavailable or busy)", unreliable: "Unreliable: grounding checks failed", abstained: "Abstained: not enough evidence" };
export { pct };
