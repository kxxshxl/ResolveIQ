import { useEffect, useState } from "react";
import { listProposals, ready } from "./api";
import Discovery from "./views/Discovery";
import Resolve from "./views/Resolve";

type View = "resolve" | "discovery";
const fromHash = (): View => (location.hash === "#/discovery" ? "discovery" : "resolve");

export default function App() {
  const [view, setView] = useState<View>(fromHash());
  const [health, setHealth] = useState("checking…");
  const [pending, setPending] = useState(0);

  useEffect(() => {
    const onHash = () => setView(fromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => { ready().then((h) => setHealth(h ? h.status : "unreachable")); }, []);
  useEffect(() => { listProposals("pending").then((p) => setPending(p.length)).catch(() => undefined); }, [view]);

  const go = (v: View) => { location.hash = v === "resolve" ? "#/" : "#/discovery"; setView(v); };
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
      </nav>
      {view === "resolve" ? <Resolve /> : <Discovery onPendingChange={setPending} />}
    </div>
  );
}
