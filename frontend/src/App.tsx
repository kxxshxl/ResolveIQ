import { useEffect, useState } from "react";
import { getDriftStatus, listProposals, ready } from "./api";
import Discovery from "./views/Discovery";
import Drift from "./views/Drift";
import Resolve from "./views/Resolve";

type View = "resolve" | "discovery" | "drift";
const HASH: Record<View, string> = { resolve: "#/", discovery: "#/discovery", drift: "#/drift" };
const fromHash = (): View => (location.hash === "#/discovery" ? "discovery" : location.hash === "#/drift" ? "drift" : "resolve");

export default function App() {
  const [view, setView] = useState<View>(fromHash());
  const [health, setHealth] = useState("checking…");
  const [pending, setPending] = useState(0);
  const [driftAlerts, setDriftAlerts] = useState(0);

  useEffect(() => {
    const onHash = () => setView(fromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => { ready().then((h) => setHealth(h ? h.status : "unreachable")); }, []);
  useEffect(() => { listProposals("pending").then((p) => setPending(p.length)).catch(() => undefined); }, [view]);
  useEffect(() => { getDriftStatus().then((r) => setDriftAlerts(r.status === "alert" ? r.alerts?.length ?? 1 : 0)).catch(() => undefined); }, [view]);

  const go = (v: View) => { location.hash = HASH[v]; setView(v); };
  const tab = (v: View, name: string, badge?: number) => (
    <button role="tab" aria-selected={view === v} className={view === v ? "tab active" : "tab"} onClick={() => go(v)}>
      {name}{!!badge && <span className="count attn">{badge}</span>}
    </button>
  );

  return (
    <div className="app">
      <header>
        <div><h1>ResolveIQ</h1><span className="muted">Semantic ticket resolution assistant for telecom support agents</span></div>
        <span className={`pill ${health === "ready" ? "ok" : "bad"}`} role="status">backend: {health}</span>
      </header>
      <nav className="tabs main" role="tablist" aria-label="Sections">
        {tab("resolve", "Resolve a complaint")}
        {tab("discovery", "Class discovery", pending)}
        {tab("drift", "Drift monitoring", driftAlerts)}
      </nav>
      {view === "resolve" ? <Resolve /> : view === "discovery" ? <Discovery onPendingChange={setPending} /> : <Drift />}
    </div>
  );
}
