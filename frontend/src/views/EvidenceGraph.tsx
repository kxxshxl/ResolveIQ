import type { Lineage } from "../consoleTypes";
import { label, pct } from "../ui";

export type Selection = { kind: "step"; id: number } | { kind: "source"; id: string } | null;

const clip = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s);
const W = 1000, NODE_H = 36, GAP = 10, TOP = 30;
const COL = { complaint: { x: 8, w: 128 }, attrs: { x: 172, w: 128 }, sources: { x: 336, w: 252 }, steps: { x: 640, w: 236 }, check: { x: 912, w: 84 } };

/** The resolution lineage as a picture: complaint -> attributes -> retrieved sources -> steps -> validation. Clicking a step or a source highlights what supports it. */
export default function EvidenceGraph({ lineage, selected, onSelect }: { lineage: Lineage; selected: Selection; onSelect: (s: Selection) => void }) {
  const { sources, steps, attributes } = lineage;
  const rows = Math.max(sources.length, steps.length, attributes.length, 1);
  const H = TOP + rows * (NODE_H + GAP) + 8;
  const place = (n: number, i: number) => TOP + (H - TOP - n * (NODE_H + GAP) + GAP) / 2 + i * (NODE_H + GAP);
  const cy = (n: number, i: number) => place(n, i) + NODE_H / 2;
  const srcY = new Map(sources.map((s, i) => [s.id, cy(sources.length, i)]));
  const stepY = new Map(steps.map((s, i) => [s.index, cy(steps.length, i)]));
  const mid = H / 2;

  const citedByStep = (i: number) => new Set(steps.find((s) => s.index === i)?.citations ?? []);
  const stepsOfSource = (id: string) => new Set(steps.filter((s) => s.citations.includes(id)).map((s) => s.index));
  const activeSources = selected?.kind === "step" ? citedByStep(selected.id) : selected?.kind === "source" ? new Set([selected.id]) : null;
  const activeSteps = selected?.kind === "source" ? stepsOfSource(selected.id) : selected?.kind === "step" ? new Set([selected.id]) : null;

  const curve = (x1: number, y1: number, x2: number, y2: number) => { const dx = (x2 - x1) * 0.5; return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`; };
  const key = (fn: () => void) => (e: React.KeyboardEvent) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fn(); } };
  const ok = Object.values(lineage.checks).every(Boolean);

  return (
    <div>
      <div className="graph-wrap">
        <svg className="graph" viewBox={`0 0 ${W} ${H}`} role="group" aria-label="Evidence graph: complaint, extracted attributes, retrieved sources, resolution steps and validation. Select a step or a source to see what supports it.">
          {([["Complaint", COL.complaint], ["Understood as", COL.attrs], ["Retrieved sources", COL.sources], ["Resolution steps", COL.steps], ["Checks", COL.check]] as const).map(([t, c]) => (
            <text key={t} className="g-col" x={c.x} y={16}>{t}</text>
          ))}

          {attributes.map((a, i) => <path key={a.name} className="g-edge extracted" strokeWidth={1} opacity={0.25 + 0.5 * a.confidence} d={curve(COL.complaint.x + COL.complaint.w, mid, COL.attrs.x, cy(attributes.length, i))} />)}
          {sources.map((s) => (
            <path key={`r-${s.id}`} className={`g-edge retrieved ${activeSources && !activeSources.has(s.id) ? "dim" : activeSources ? "hl" : ""}`} strokeWidth={1 + 2.2 * s.score} opacity={activeSources ? undefined : 0.12 + 0.5 * s.score}
              d={curve(COL.complaint.x + COL.complaint.w, mid, COL.sources.x, srcY.get(s.id)!)} />
          ))}
          {steps.flatMap((st) => st.citations.map((c) => {
            const hl = selected ? (selected.kind === "step" ? selected.id === st.index : selected.id === c) : false;
            return <path key={`c-${st.index}-${c}`} className={`g-edge cited ${selected ? (hl ? "hl" : "dim") : ""}`} strokeWidth={1.2 + 3 * (st.support[c] ?? 0.4)}
              d={curve(COL.sources.x + COL.sources.w, srcY.get(c) ?? mid, COL.steps.x, stepY.get(st.index)!)}><title>{`${c} supports step ${st.index}: ${pct(st.support[c])}`}</title></path>;
          }))}
          {steps.map((st) => <path key={`v-${st.index}`} className="g-edge extracted" strokeWidth={1} opacity={0.3} d={curve(COL.steps.x + COL.steps.w, stepY.get(st.index)!, COL.check.x, mid)} />)}

          <g className="g-node static"><rect x={COL.complaint.x} y={mid - 30} width={COL.complaint.w} height={60} rx={8} />
            <text x={COL.complaint.x + 10} y={mid - 8}>Customer complaint</text><text className="sub" x={COL.complaint.x + 10} y={mid + 8}>{lineage.complaint.chars} characters</text>
            <text className="sub" x={COL.complaint.x + 10} y={mid + 21}>{Object.keys(lineage.complaint.pii_redactions).length ? "PII redacted" : "no PII found"}{lineage.complaint.injection_flagged ? " · flagged" : ""}</text></g>

          {attributes.map((a, i) => (
            <g key={a.name} className="g-node static"><title>{`${a.name}: ${a.value} (confidence ${pct(a.confidence)})`}</title>
              <rect x={COL.attrs.x} y={place(attributes.length, i)} width={COL.attrs.w} height={NODE_H} rx={7} />
              <text x={COL.attrs.x + 9} y={place(attributes.length, i) + 15}>{clip(label(a.value), 17)}</text>
              <text className="sub" x={COL.attrs.x + 9} y={place(attributes.length, i) + 28}>{a.name} · {pct(a.confidence)}</text></g>
          ))}

          {sources.map((s, i) => {
            const y = place(sources.length, i);
            const sel = selected?.kind === "source" && selected.id === s.id;
            const dim = activeSources !== null && !activeSources.has(s.id);
            const cited = s.cited_by_steps.length > 0;
            const act = () => onSelect(sel ? null : { kind: "source", id: s.id });
            return (
              <g key={s.id} className={`g-node ${sel ? "sel" : ""} ${cited ? "cited" : ""} ${dim ? "dim" : ""}`} role="button" tabIndex={0} aria-pressed={sel} aria-label={`Source ${s.id}, rank ${s.rank}, score ${s.score.toFixed(2)}${cited ? `, cited by step ${s.cited_by_steps.join(", ")}` : ", not cited"}`} onClick={act} onKeyDown={key(act)}>
                <title>{`${s.id} (${s.type})\n${s.title}\n${s.why.join("\n")}`}</title>
                <rect x={COL.sources.x} y={y} width={COL.sources.w} height={NODE_H} rx={7} />
                <text x={COL.sources.x + 9} y={y + 15}>#{s.rank} {s.id}{s.selected ? "" : "  (not selected)"}</text>
                <text className="sub" x={COL.sources.x + 9} y={y + 28}>{clip(s.title, 34)}</text>
                <text className="sub" x={COL.sources.x + COL.sources.w - 8} y={y + 15} textAnchor="end">{s.score.toFixed(2)}</text>
              </g>
            );
          })}

          {steps.map((st, i) => {
            const y = place(steps.length, i);
            const sel = selected?.kind === "step" && selected.id === st.index;
            const dim = activeSteps !== null && !activeSteps.has(st.index);
            const act = () => onSelect(sel ? null : { kind: "step", id: st.index });
            return (
              <g key={st.index} className={`g-node ${sel ? "sel" : ""} ${dim ? "dim" : ""} ${st.grounded === false ? "warn" : ""}`} role="button" tabIndex={0} aria-pressed={sel}
                aria-label={`Step ${st.index}: ${st.text}. ${st.citations.length} citation(s). ${st.grounded === false ? "Not supported by the cited text." : "Supported."}`} onClick={act} onKeyDown={key(act)}>
                <title>{`${st.index}. ${st.text}`}</title>
                <rect x={COL.steps.x} y={y} width={COL.steps.w} height={NODE_H} rx={7} />
                <text x={COL.steps.x + 9} y={y + 15}>{st.index}. {clip(st.text, 34)}</text>
                <text className="sub" x={COL.steps.x + 9} y={y + 28}>{st.citations.join(", ") || "no citation"} · support {st.grounding_score != null ? st.grounding_score.toFixed(2) : "n/a"}</text>
              </g>
            );
          })}

          <g className={`g-node static ${ok ? "ok-fill" : "bad"}`}><title>{Object.entries(lineage.checks).map(([k, v]) => `${v ? "✓" : "✗"} ${k.replace(/_/g, " ")}`).join("\n")}</title>
            <rect x={COL.check.x} y={mid - 30} width={COL.check.w} height={60} rx={8} />
            <text x={COL.check.x + 9} y={mid - 8}>{ok ? "All checks" : "Check failed"}</text><text className="sub" x={COL.check.x + 9} y={mid + 8}>{Object.values(lineage.checks).filter(Boolean).length}/{Object.keys(lineage.checks).length} passed</text></g>
        </svg>
      </div>
      <div className="legend" style={{ marginTop: 8 }}>
        <span><i style={{ background: "var(--accent)" }} />retrieved (thicker = higher score)</span><span><i style={{ background: "var(--ok)" }} />cited (thicker = better supported)</span>
        <span>click a step or a source to follow its evidence</span>
      </div>
    </div>
  );
}
