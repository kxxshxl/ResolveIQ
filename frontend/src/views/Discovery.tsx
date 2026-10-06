import { useCallback, useEffect, useMemo, useState } from "react";
import { acceptProposal, getJob, getTaxonomy, listProposals, rejectProposal, runDiscovery } from "../api";
import type { DiscoveryResult, Job, Proposal, Taxonomy } from "../types";
import { Skeleton, ToastBar, label, pct, type Toast } from "../ui";

type Filter = Proposal["status"] | "all";
const FILTERS: [Filter, string][] = [["pending", "Pending"], ["accepted", "Accepted"], ["rejected", "Rejected"], ["all", "All"]];
const LABEL_RE = /^[a-z][a-z0-9_]{2,47}$/;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Polls the background job until it finishes; the worker may take a while to pick it up, so give up after ~3 minutes. */
async function waitForJob(id: string, onUpdate: (j: Job) => void): Promise<Job> {
  for (let i = 0; i < 90; i++) {
    const j = await getJob(id);
    onUpdate(j);
    if (j.status === "succeeded" || j.status === "failed") return j;
    await sleep(2000);
  }
  throw new Error("Discovery is still running; refresh in a moment to see the proposals.");
}

function Stat({ name, value, hint }: { name: string; value: string; hint: string }) {
  return (
    <div className="stat" title={hint}>
      <span className="muted small">{name}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ProposalCard({ p, taxonomy, onDone, onToast }: { p: Proposal; taxonomy: Taxonomy | null; onDone: () => void; onToast: (t: Toast) => void }) {
  const [mode, setMode] = useState<"view" | "accept" | "reject">("view");
  const [busy, setBusy] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const extend = p.recommendation === "extend_existing";
  const intents = taxonomy?.labels.intent ?? [];
  const [labelId, setLabelId] = useState(p.label_id);
  const [description, setDescription] = useState(p.description);
  const [team, setTeam] = useState(p.team ?? "");
  const [mergeInto, setMergeInto] = useState(extend ? (p.nearest_intent ?? "") : "");
  const [note, setNote] = useState("");
  const pending = p.status === "pending";
  const idError = !mergeInto && !LABEL_RE.test(labelId) ? "lowercase letters, digits and underscores; 3 to 48 characters, starting with a letter" : null;
  const idExists = !mergeInto && intents.some((i) => i.id === labelId);

  const act = async (fn: () => Promise<string>) => {
    setBusy(true);
    try { onToast({ tone: "ok", text: await fn() }); onDone(); } catch (e) { onToast({ tone: "bad", text: (e as Error).message }); } finally { setBusy(false); }
  };
  const accept = () => act(async () => {
    const res = await acceptProposal(p.proposal_id, {
      ...(mergeInto ? { merge_into: mergeInto } : { label_id: labelId, description: description || undefined, team: team || undefined }),
      note: note || undefined,
    });
    return `${res.action === "created" ? "Created" : "Extended"} intent "${res.label_id}" - taxonomy is now v${res.taxonomy_version} (${res.members} complaints will classify to it).`;
  });
  const reject = () => act(async () => { await rejectProposal(p.proposal_id, note || undefined); return `Rejected "${p.label_id}". Its complaints will not be proposed again.`; });

  const examples = showAll ? p.examples : p.examples.slice(0, 2);
  return (
    <article className={`card proposal ${p.status}`}>
      <div className="row wrap-start">
        <h3 className="proposal-title">{extend ? `Extend ${label(p.nearest_intent ?? p.label_id)}` : `New intent: ${p.keywords.slice(0, 3).join(", ")}`}</h3>
        <span className={`pill ${extend ? "warn" : "ok"}`}>{extend ? "extends an existing intent" : "new intent"}</span>
        {!pending && <span className={`pill ${p.status === "accepted" ? "ok" : "bad"}`}>{p.status}</span>}
        <span className="grow" />
        <span className="muted small">{p.members} complaints{p.product ? ` · ${p.product}` : ""}</span>
      </div>

      <div className="stats">
        <Stat name="Cohesion" value={pct(p.cohesion)} hint="How similar the complaints in this cluster are to each other." />
        <Stat name="Mean evidence" value={pct(p.mean_evidence)} hint="How well existing tickets and articles explain these complaints. Low means poorly covered." />
        <Stat name="Neighbour agreement" value={pct(p.neighbor_agreement)} hint="Share of complaints whose closest resolved ticket belongs to the nearest existing intent." />
      </div>

      <div className="chips">{p.keywords.map((k) => <code key={k} className="chip">{k}</code>)}</div>

      <div className="examples-list">
        {examples.map((e, i) => <blockquote key={i}>{e}</blockquote>)}
        {p.examples.length > 2 && <button className="link" onClick={() => setShowAll(!showAll)}>{showAll ? "show fewer" : `show all ${p.examples.length} examples`}</button>}
      </div>

      {!pending && (
        <p className="muted small">
          {p.status} {p.decided_at ? new Date(p.decided_at).toLocaleString() : ""}{p.decided_note ? ` · ${p.decided_note}` : ""}
        </p>
      )}

      {pending && mode === "view" && (
        <div className="row gap">
          <button className="primary" onClick={() => setMode("accept")}>{extend ? "Review & extend" : "Review & accept"}</button>
          <button onClick={() => setMode("reject")}>Reject</button>
        </div>
      )}

      {pending && mode === "accept" && (
        <form className="form" onSubmit={(e) => { e.preventDefault(); accept(); }}>
          <label className="field">Add to an existing intent instead (optional)
            <select value={mergeInto} onChange={(e) => setMergeInto(e.target.value)}>
              <option value="">Create a new intent</option>
              {intents.map((i) => <option key={i.id} value={i.id}>{label(i.id)}{i.id === p.nearest_intent ? " (nearest)" : ""}</option>)}
            </select>
          </label>
          {!mergeInto && (
            <>
              <label className="field">Intent id
                <input value={labelId} onChange={(e) => setLabelId(e.target.value)} aria-invalid={!!idError || idExists} />
                {idError && <span className="error small">{idError}</span>}
                {idExists && <span className="error small">An intent with this id already exists. Pick another id or add to it above.</span>}
              </label>
              <label className="field">Description
                <textarea rows={2} maxLength={400} value={description} onChange={(e) => setDescription(e.target.value)} />
              </label>
              <label className="field">Owning team
                <input value={team} maxLength={80} onChange={(e) => setTeam(e.target.value)} placeholder="e.g. Network Operations" />
              </label>
            </>
          )}
          <label className="field">Note (optional)
            <input value={note} maxLength={300} onChange={(e) => setNote(e.target.value)} placeholder="Why you are accepting this" />
          </label>
          <p className="muted small">
            This {mergeInto ? `adds this cluster's keywords and examples to "${label(mergeInto)}"` : "creates the intent"}, bumps the taxonomy version and invalidates cached responses.
          </p>
          <div className="row gap">
            <button type="submit" className="primary" disabled={busy || (!mergeInto && (!!idError || idExists))}>{busy ? "Saving…" : "Confirm"}</button>
            <button type="button" onClick={() => setMode("view")} disabled={busy}>Cancel</button>
          </div>
        </form>
      )}

      {pending && mode === "reject" && (
        <form className="form" onSubmit={(e) => { e.preventDefault(); reject(); }}>
          <label className="field">Reason (optional)
            <input value={note} maxLength={300} onChange={(e) => setNote(e.target.value)} placeholder="e.g. duplicate of billing_dispute" />
          </label>
          <div className="row gap">
            <button type="submit" disabled={busy}>{busy ? "Rejecting…" : "Confirm reject"}</button>
            <button type="button" onClick={() => setMode("view")} disabled={busy}>Cancel</button>
          </div>
        </form>
      )}
    </article>
  );
}

export default function Discovery({ onPendingChange }: { onPendingChange?: (n: number) => void }) {
  const [items, setItems] = useState<Proposal[] | null>(null);
  const [taxonomy, setTaxonomy] = useState<Taxonomy | null>(null);
  const [filter, setFilter] = useState<Filter>("pending");
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [windowDays, setWindowDays] = useState("");
  const [running, setRunning] = useState<string | null>(null);
  const [lastRun, setLastRun] = useState<DiscoveryResult | null>(null);
  const [toast, setToast] = useState<Toast | null>(null);
  const closeToast = useCallback(() => setToast(null), []);

  const reload = useCallback(async () => {
    try {
      const [p, t] = await Promise.all([listProposals("all"), getTaxonomy()]);
      setItems(p); setTaxonomy(t); setLoadErr(null);
      onPendingChange?.(p.filter((x) => x.status === "pending").length);
    } catch (e) { setLoadErr((e as Error).message); }
  }, [onPendingChange]);
  useEffect(() => { reload(); }, [reload]);

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: items?.length ?? 0 };
    for (const p of items ?? []) c[p.status] = (c[p.status] ?? 0) + 1;
    return c;
  }, [items]);
  const shown = useMemo(() => (items ?? []).filter((p) => filter === "all" || p.status === filter), [items, filter]);

  const discover = async () => {
    const days = windowDays ? Number(windowDays) : undefined;
    if (days !== undefined && (!Number.isInteger(days) || days < 1 || days > 365)) { setToast({ tone: "bad", text: "Window must be a whole number of days between 1 and 365." }); return; }
    setRunning("queued");
    try {
      const { job_id } = await runDiscovery(days);
      const job = await waitForJob(job_id, (j) => setRunning(j.status));
      if (job.status === "failed") throw new Error(job.error ?? "Discovery job failed.");
      const result = job.result as unknown as DiscoveryResult;
      setLastRun(result);
      setToast({ tone: "ok", text: result.proposals ? `Found ${result.proposals} proposal${result.proposals === 1 ? "" : "s"}.` : "No emerging classes found in this window." });
      setFilter("pending");
      await reload();
    } catch (e) { setToast({ tone: "bad", text: (e as Error).message }); } finally { setRunning(null); }
  };

  return (
    <>
      <section className="card">
        <div className="row wrap-start">
          <div className="grow">
            <h2>Emerging-class discovery</h2>
            <p className="muted small">
              Complaints that existing tickets and articles explain poorly are clustered. Each cluster becomes a proposal: a <strong>new intent</strong>, or an <strong>extension</strong> of
              the nearest existing intent. Nothing changes until a person accepts it.
              {taxonomy && <> Current taxonomy: <strong>v{taxonomy.version}</strong>, {taxonomy.labels.intent?.length ?? 0} intents.</>}
            </p>
          </div>
          <label className="field narrow">Look back (days)
            <input inputMode="numeric" value={windowDays} onChange={(e) => setWindowDays(e.target.value.replace(/\D/g, ""))} placeholder="server default" />
          </label>
          <button className="primary" onClick={discover} disabled={running !== null}>{running ? `Running (${running})…` : "Run discovery"}</button>
        </div>
        {lastRun && (
          <div className="stats">
            <Stat name="Candidates" value={String(lastRun.candidates)} hint="Recent requests the system explained poorly." />
            <Stat name="Unique" value={String(lastRun.unique_candidates)} hint="After removing identical complaints (retries, copy-paste)." />
            <Stat name="Proposals" value={String(lastRun.proposals)} hint="Clusters large and coherent enough to review." />
            <Stat name="Window" value={`${lastRun.window_days} d`} hint="How far back the run looked." />
            <Stat name="Took" value={`${lastRun.elapsed_s} s`} hint="Wall-clock time of the job." />
          </div>
        )}
        {running && <p className="muted small" role="status">The job runs in the background worker. This page checks every 2 seconds.</p>}
      </section>

      <div className="tabs" role="tablist" aria-label="Proposal status">
        {FILTERS.map(([f, name]) => (
          <button key={f} role="tab" aria-selected={filter === f} className={filter === f ? "tab active" : "tab"} onClick={() => setFilter(f)}>
            {name} <span className="count">{counts[f] ?? 0}</span>
          </button>
        ))}
      </div>

      {loadErr && <p className="error" role="alert">⚠ {loadErr} <button className="link" onClick={reload}>retry</button></p>}
      {!items && !loadErr && <section className="card"><Skeleton lines={4} /></section>}
      {items && shown.length === 0 && (
        <section className="card empty">
          <strong>{filter === "pending" ? "Nothing to review." : `No ${filter === "all" ? "" : filter + " "}proposals.`}</strong>
          <p className="muted small">
            {filter === "pending" ? "Run discovery to look for complaint clusters the current classes do not cover. It needs enough recent poorly-explained complaints to form a cluster." : "Proposals show up here once they have been reviewed."}
          </p>
        </section>
      )}
      {shown.map((p) => <ProposalCard key={p.proposal_id} p={p} taxonomy={taxonomy} onDone={reload} onToast={setToast} />)}
      <ToastBar toast={toast} onClose={closeToast} />
    </>
  );
}
