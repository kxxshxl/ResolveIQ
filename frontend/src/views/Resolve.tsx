import { useEffect, useMemo, useRef, useState } from "react";
import { feedback, getTaxonomy, resolve } from "../api";
import type { Resolved } from "../consoleTypes";
import { useAsync, useHotkeys } from "../hooks";
import { Bar, CopyButton, EmptyState, Kbd, Loading, Panel, Pill, Stat, Tabs, label, ms, pct } from "../ui";
import EvidenceGraph, { type Selection } from "./EvidenceGraph";
import { ProvenanceTable, Signals, SourceList, StageTimeline, statusText, statusTone } from "./shared";

const EXAMPLES: [string, string][] = [
  ["Broadband drops every evening", "My broadband drops every evening around 8 and I've already restarted the router twice. I work from home and this is costing me."],
  ["Charged twice", "You took my monthly payment twice on the 3rd and I want the extra one refunded. This is unacceptable."],
  ["No signal, urgent", "There is no mobile signal anywhere in our village since the storm and elderly neighbours cannot call for help."],
  ["eSIM will not activate", "My eSIM QR code keeps failing to activate on my new phone."],
  ["Out of scope", "What is the best pizza place near the city centre?"],
];
const STRATEGIES: [string, string][] = [
  ["", "Server default"], ["dense", "Dense (semantic)"], ["adaptive", "Adaptive (dense, escalates when unsure)"], ["hybrid", "Hybrid (dense + keyword)"],
  ["hybrid_reranked", "Hybrid + cross-encoder"], ["lexical", "Keyword only"],
];
const REASONS: [string, string][] = [
  ["wrong_intent", "Wrong problem type"], ["steps_incorrect", "A step is wrong"], ["steps_missing", "Steps missing"], ["irrelevant_source", "Irrelevant source"],
  ["outdated_source", "Outdated source"], ["too_vague", "Too vague"], ["unsafe", "Unsafe advice"], ["other", "Other"],
];
const MAX_CHARS = 4000;

function plain(r: Resolved) {
  const res = r.resolution;
  const lines = [res.issue_summary, ""];
  res.steps.forEach((s, i) => lines.push(`${i + 1}. ${s.text}${s.citations.length ? ` [${s.citations.join(", ")}]` : ""}`));
  if (res.uncertainty) lines.push("", `Uncertainty: ${res.uncertainty}`);
  if (res.escalate) lines.push("", `Escalate: ${res.escalation_reason ?? "recommended"}`);
  return lines.join("\n");
}

function FeedbackBox({ r }: { r: Resolved }) {
  const taxonomy = useAsync(() => getTaxonomy(), []);
  const [phase, setPhase] = useState<"idle" | "reject" | "sent">("idle");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [reasons, setReasons] = useState<string[]>([]);
  const [rejected, setRejected] = useState<string[]>([]);
  const [corrected, setCorrected] = useState("");
  const [steps, setSteps] = useState(r.resolution.steps.map((s) => s.text).join("\n"));
  const [comment, setComment] = useState("");
  useEffect(() => { setPhase("idle"); setReasons([]); setRejected([]); setCorrected(""); setComment(""); setSteps(r.resolution.steps.map((s) => s.text).join("\n")); setErr(null); }, [r.request_id]); // eslint-disable-line react-hooks/exhaustive-deps
  const toggle = (list: string[], set: (v: string[]) => void, v: string) => set(list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  const send = async (rating: "helpful" | "not_helpful") => {
    setBusy(true); setErr(null);
    try {
      const original = r.resolution.steps.map((s) => s.text).join("\n");
      await feedback({
        request_id: r.request_id, rating,
        ...(rating === "not_helpful" ? {
          reasons, rejected_sources: rejected, ...(corrected ? { corrected_intent: corrected } : {}), ...(comment.trim() ? { comment: comment.trim() } : {}),
          ...(steps.trim() && steps.trim() !== original.trim() ? { edited_steps: steps.split("\n").map((s) => s.trim()).filter(Boolean) } : {}),
        } : {}),
      });
      setPhase("sent");
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  if (phase === "sent") return <p className="small"><Pill tone="ok">Thanks, feedback recorded</Pill> <span className="muted">It appears in Feedback &amp; Quality and the case history; nothing is retrained automatically.</span></p>;
  return (
    <div>
      <div className="row gap-s wrap" style={{ alignItems: "center" }}>
        <span className="muted small">Was this resolution helpful?</span>
        <button className="btn sm" disabled={busy} onClick={() => send("helpful")}>👍 Helpful</button>
        <button className="btn sm" aria-expanded={phase === "reject"} onClick={() => setPhase(phase === "reject" ? "idle" : "reject")}>👎 Not helpful</button>
        {err && <span className="error small" role="alert">⚠ {err}</span>}
      </div>
      {phase === "reject" && (
        <form className="form" onSubmit={(e) => { e.preventDefault(); send("not_helpful"); }}>
          <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
            <legend className="small muted">What was wrong? (optional)</legend>
            <div className="tag-grid">{REASONS.map(([k, name]) => <label key={k} className="field check"><input type="checkbox" checked={reasons.includes(k)} onChange={() => toggle(reasons, setReasons, k)} />{name}</label>)}</div>
          </fieldset>
          {(r.lineage?.sources.length ?? 0) > 0 && (
            <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
              <legend className="small muted">Sources that should not have been used</legend>
              <div className="tag-grid">{r.lineage!.sources.filter((s) => s.selected).map((s) => (
                <button type="button" key={s.id} className={`chip ${rejected.includes(s.id) ? "on" : ""}`} aria-pressed={rejected.includes(s.id)} onClick={() => toggle(rejected, setRejected, s.id)}>{s.id}</button>
              ))}</div>
            </fieldset>
          )}
          <label className="field">The correct problem type, if the detected one is wrong
            <select value={corrected} onChange={(e) => setCorrected(e.target.value)}>
              <option value="">Detected: {label(r.classification.intent)}</option>
              {(taxonomy.data?.labels.intent ?? []).filter((i) => i.id !== r.classification.intent).map((i) => <option key={i.id} value={i.id}>{label(i.id)}</option>)}
            </select>
          </label>
          <label className="field">Your version of the steps, one per line (kept redacted, used only for the improvement report)
            <textarea rows={4} value={steps} onChange={(e) => setSteps(e.target.value)} />
          </label>
          <label className="field">Comment (optional)<input value={comment} maxLength={500} onChange={(e) => setComment(e.target.value)} /></label>
          <div className="row gap-s"><button className="btn primary" type="submit" disabled={busy}>{busy ? "Saving…" : "Send feedback"}</button><button type="button" className="btn" onClick={() => setPhase("idle")}>Cancel</button></div>
        </form>
      )}
    </div>
  );
}

export default function Resolve({ onOpenCase }: { onOpenCase: (id: string) => void }) {
  const [text, setText] = useState(EXAMPLES[0][1]);
  const [strategy, setStrategy] = useState("");
  const [filters, setFilters] = useState(false);
  const [det, setDet] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [r, setR] = useState<Resolved | null>(null);
  const [sel, setSel] = useState<Selection>(null);
  const [tab, setTab] = useState<"evidence" | "signals" | "trace">("evidence");
  const input = useRef<HTMLTextAreaElement>(null);
  const results = useRef<HTMLDivElement>(null);
  const canSubmit = useMemo(() => text.trim().length >= 5 && !busy, [text, busy]);

  const submit = async () => {
    if (!canSubmit) return;
    setBusy(true); setErr(null); setSel(null);
    try { setR(await resolve(text, strategy, filters, det)); } catch (e) { setErr((e as Error).message); setR(null); } finally { setBusy(false); }
  };
  useEffect(() => { if (r) results.current?.scrollIntoView({ behavior: "smooth", block: "start" }); }, [r?.request_id]); // eslint-disable-line react-hooks/exhaustive-deps
  useHotkeys((k) => {
    if (k === "/") { input.current?.focus(); return true; }
    if (k === "esc") { setSel(null); return false; }
    if (r && /^[1-9]$/.test(k) && Number(k) <= r.resolution.steps.length) { setSel({ kind: "step", id: Number(k) }); return true; }
    return false;
  });

  const lin = r?.lineage ?? null;
  const citedBy = (id: string) => lin?.sources.find((s) => s.id === id)?.cited_by_steps ?? [];
  const selStep = sel?.kind === "step" ? sel.id : null;
  const citedSet = selStep != null ? new Set(r?.resolution.steps[selStep - 1]?.citations ?? []) : null;
  const srcSel = sel?.kind === "source" ? sel.id : null;

  return (
    <>
      <div className="page-head"><div className="grow"><h1>Resolve a complaint</h1><p>Paste a customer complaint. The answer is built only from resolved tickets and knowledge-base articles, every step cites its source, and the system says so when it does not know.</p></div></div>

      <Panel className="composer" title="Customer complaint" actions={<span className="muted small">{text.length}/{MAX_CHARS} · <Kbd>Ctrl</Kbd>+<Kbd>Enter</Kbd> to resolve · <Kbd>/</Kbd> to focus</span>}>
        <label htmlFor="complaint" className="sr-only">Customer complaint</label>
        <textarea id="complaint" ref={input} rows={3} maxLength={MAX_CHARS} value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste the raw customer complaint…"
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submit(); } }} />
        <div className="row gap wrap">
          <label className="field">Example<select value="" onChange={(e) => { const x = EXAMPLES.find(([n]) => n === e.target.value); if (x) setText(x[1]); }}><option value="">Choose an example…</option>{EXAMPLES.map(([n]) => <option key={n}>{n}</option>)}</select></label>
          <label className="field">Retrieval<select value={strategy} onChange={(e) => setStrategy(e.target.value)}>{STRATEGIES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
          <label className="field check"><input type="checkbox" checked={filters} onChange={(e) => setFilters(e.target.checked)} />Narrow by predicted product</label>
          <label className="field check" title="Temperature 0 and a fixed seed: the same complaint gives the same answer"><input type="checkbox" checked={det} onChange={(e) => setDet(e.target.checked)} />Reproducible</label>
          <span className="grow" />
          {(r || text) && <button className="btn ghost" onClick={() => { setText(""); setR(null); setErr(null); setSel(null); input.current?.focus(); }}>Clear</button>}
          <button className="btn primary" disabled={!canSubmit} onClick={submit}>{busy ? <><span className="spin" /> Resolving…</> : "Resolve"}</button>
        </div>
        {err && <div className="errorbox" role="alert"><strong>Could not resolve</strong><span className="small">{err}</span></div>}
      </Panel>

      <div ref={results} style={{ scrollMarginTop: 12 }} aria-live="polite">
        {busy && <><Panel title="Understanding"><Loading what="complaint understanding" /></Panel><Panel title="Recommended resolution"><Loading what="resolution" /></Panel></>}
        {!busy && !r && !err && <Panel><EmptyState title="Nothing resolved yet">Pick an example or paste a complaint and press <strong>Resolve</strong>. You will see what was understood, the recommended steps with their sources, and how sure the system is.</EmptyState></Panel>}
        {!busy && r && lin && (
          <>
            <div className={`banner ${statusTone(r.status)}`} role="status">
              <strong>{statusText[r.status]}</strong>
              <span>confidence {pct(r.confidence)}</span>
              <span>{r.resolution.escalate ? "escalation recommended" : "no escalation needed"}</span>
              <span>{ms(r.latency_ms.total)}{r.cached ? " (cached)" : ""}</span>
              <span className="grow" />
              {r.provenance && <span className="small" title="Which model, prompt, taxonomy and corpus produced this answer">{r.provenance.generator} · prompt {r.provenance.prompt_version ?? "n/a"} · taxonomy v{r.provenance.taxonomy_version} · corpus v{r.provenance.corpus_version}</span>}
              <button className="btn sm" onClick={() => onOpenCase(r.request_id)}>Open as case</button>
            </div>

            <div className="split">
              <div>
                <Panel title="Understanding" subtitle={`Detected with ${r.classification.strategy}; taxonomy v${r.classification.taxonomy_version}${Object.keys(r.pii_redactions).length ? ` · PII redacted: ${Object.entries(r.pii_redactions).map(([k, v]) => `${k}×${v}`).join(", ")}` : ""}`}>
                  <div className="attr-row">
                    {lin.attributes.map((a) => (
                      <div key={a.name} className="attr"><span className="k">{a.name}</span><span className="v">{label(a.value)}</span><Bar value={a.confidence} tone={a.confidence < 0.45 ? "warn" : ""} /><span className="muted small">{pct(a.confidence)} confident{a.confidence < 0.45 ? " (low)" : ""}</span></div>
                    ))}
                  </div>
                </Panel>

                <Panel title="Recommended resolution" subtitle={r.resolution.issue_summary} actions={r.resolution.steps.length > 0 ? <CopyButton text={plain(r)}>Copy resolution</CopyButton> : undefined}>
                  {r.resolution.steps.length > 0 ? (
                    <ol className="steps" aria-label="Resolution steps. Select one to highlight its evidence; press the number key to jump to a step.">
                      {r.resolution.steps.map((s, i) => {
                        const n = i + 1;
                        const on = selStep === n;
                        const dim = (selStep != null && !on) || (srcSel != null && !s.citations.includes(srcSel));
                        return (
                          <li key={n}>
                            <button className={`step ${on ? "sel" : ""} ${dim ? "dim" : ""} ${s.grounded === false ? "ungrounded" : ""}`} aria-pressed={on} onClick={() => setSel(on ? null : { kind: "step", id: n })}>
                              <span>{s.text}</span>
                              <span className="meta">
                                {s.citations.map((c) => <code key={c} className={`chip ${srcSel === c || citedSet?.has(c) ? "active" : ""}`}>{c}{s.support?.[c] != null ? ` · ${s.support[c].toFixed(2)}` : ""}</code>)}
                                {s.citations.length === 0 && <Pill tone="bad">no citation</Pill>}
                                {s.grounded === false && <Pill tone="warn">not supported by the cited text</Pill>}
                              </span>
                            </button>
                          </li>
                        );
                      })}
                    </ol>
                  ) : <p className="muted">No steps were generated: the retrieved evidence does not cover this problem.</p>}
                  {r.resolution.uncertainty && <p className="muted small">Uncertainty: {r.resolution.uncertainty}</p>}
                  <div className={`escalation ${r.resolution.escalate ? "yes" : "no"}`}><strong>Escalation: {r.resolution.escalate ? "recommended" : "not required"}</strong>{r.resolution.escalation_reason && <div>{r.resolution.escalation_reason}</div>}</div>
                  {[...r.validation.warnings, ...r.warnings.filter((w) => !r.validation.warnings.includes(w))].map((w) => <p key={w} className="warn-text small">⚠ {w}</p>)}
                  <FeedbackBox r={r} />
                </Panel>
              </div>

              <div className="sticky">
                <Panel title="Evidence" subtitle="Where the answer comes from">
                  <Tabs label="Evidence views" value={tab} onChange={setTab} tabs={[{ id: "evidence", name: "Sources", count: lin.sources.length }, { id: "signals", name: "Confidence" }, { id: "trace", name: "Trace" }]} />
                  {tab === "evidence" && <SourceList sources={lin.sources} selected={sel} onSelect={setSel} citedBy={citedBy} />}
                  {tab === "signals" && (
                    <>
                      <div className="stats"><Stat name="Evidence" value={r.evidence.sufficient ? "sufficient" : "insufficient"} tone={r.evidence.sufficient ? "ok" : "bad"} /><Stat name="Citation check" value={r.validation.valid ? "passed" : "failed"} tone={r.validation.valid ? "ok" : "bad"} /></div>
                      <p className="muted small">{r.evidence.reason}</p>
                      <Signals lineage={lin} />
                      {r.provenance && <><h3>How this answer was produced</h3><ProvenanceTable p={r.provenance} /></>}
                    </>
                  )}
                  {tab === "trace" && (r.trace?.length ? <StageTimeline trace={r.trace} /> : <EmptyState title="No trace recorded" />)}
                </Panel>
              </div>
            </div>

            <Panel title="Resolution lineage" subtitle="complaint → what was understood → what was retrieved → what was cited by which step → what was checked. Nothing in this graph is model reasoning: only scores, ids and checks.">
              <EvidenceGraph lineage={lin} selected={sel} onSelect={setSel} />
            </Panel>
          </>
        )}
      </div>
    </>
  );
}
