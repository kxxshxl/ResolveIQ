import { useCallback, useEffect, useState, type ReactNode } from "react";
import { getDriftStatus, getKey, listProposals, ready, setKey } from "./api";
import { useHotkeys, useInterval } from "./hooks";
import { Dialog, Kbd } from "./ui";
import CaseReplay from "./views/CaseReplay";
import Discovery from "./views/Discovery";
import Drift from "./views/Drift";
import Evaluation from "./views/Evaluation";
import Quality from "./views/Quality";
import Resolve from "./views/Resolve";
import RetrievalLab from "./views/RetrievalLab";
import SystemHealth from "./views/SystemHealth";

type View = "resolve" | "cases" | "lab" | "discovery" | "drift" | "quality" | "evaluation" | "health";
const ROUTES: Record<View, string> = { resolve: "#/", cases: "#/cases", lab: "#/lab", discovery: "#/discovery", drift: "#/drift", quality: "#/quality", evaluation: "#/evaluation", health: "#/health" };
const KEYS: Record<string, View> = { r: "resolve", c: "cases", l: "lab", d: "discovery", m: "drift", q: "quality", e: "evaluation", h: "health" };

const parse = (): { view: View; param: string | null } => {
  const [path, param] = location.hash.replace(/^#\/?/, "").split("/");
  const view = (Object.keys(ROUTES) as View[]).find((v) => ROUTES[v] === `#/${path}`) ?? "resolve";
  return { view, param: param ?? null };
};

const Icon = ({ d }: { d: string }) => <svg className="ico" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={d} /></svg>;
const ICONS: Record<View, string> = {
  resolve: "M2 3.5h12M2 8h8M2 12.5h5M12 10.5l2 2-2 2",
  cases: "M3 2.5h10v11H3zM5.5 6h5M5.5 8.5h5M5.5 11h3",
  lab: "M6 2v5L2.5 13a1 1 0 0 0 .9 1.5h9.2A1 1 0 0 0 13.5 13L10 7V2M5 2h6",
  discovery: "M8 2.5v2M8 11.5v2M2.5 8h2M11.5 8h2M4.2 4.2l1.4 1.4M10.4 10.4l1.4 1.4M4.2 11.8l1.4-1.4M10.4 5.6l1.4-1.4",
  drift: "M2 11l3-4 3 2 3-5 3 3M2 14h12",
  quality: "M3 13V8M8 13V3M13 13V6",
  evaluation: "M3 3h10v10H3zM3 6.5h10M7 6.5V13",
  health: "M2 8h3l1.5-4 3 8L11 8h3",
};
const NAV: { title: string; items: { v: View; name: string; key: string }[] }[] = [
  { title: "Work", items: [{ v: "resolve", name: "Resolve", key: "r" }, { v: "cases", name: "Case replay", key: "c" }, { v: "lab", name: "Retrieval lab", key: "l" }] },
  { title: "Improve", items: [{ v: "discovery", name: "Discovery", key: "d" }, { v: "drift", name: "Drift monitoring", key: "m" }, { v: "quality", name: "Feedback & quality", key: "q" }] },
  { title: "Operate", items: [{ v: "evaluation", name: "Evaluation", key: "e" }, { v: "health", name: "System health", key: "h" }] },
];

function Shortcuts({ onClose }: { onClose: () => void }) {
  return (
    <Dialog title="Keyboard shortcuts" onClose={onClose}>
      <div className="help">
        {NAV.flatMap((g) => g.items).map((i) => <span key={i.v} style={{ display: "contents" }}><span><Kbd>g</Kbd> <Kbd>{i.key}</Kbd></span><span>{i.name}</span></span>)}
        <span><Kbd>/</Kbd></span><span>Focus the complaint box (Resolve)</span>
        <span><Kbd>Ctrl</Kbd> <Kbd>Enter</Kbd></span><span>Resolve</span>
        <span><Kbd>1</Kbd>–<Kbd>9</Kbd></span><span>Select a resolution step and show its evidence</span>
        <span><Kbd>Esc</Kbd></span><span>Clear the selection, close this dialog</span>
        <span><Kbd>?</Kbd></span><span>This help</span>
      </div>
      <p className="muted small" style={{ marginTop: 12 }}>Plain keys are ignored while you are typing in a field.</p>
    </Dialog>
  );
}

export default function App() {
  const [route, setRoute] = useState(parse());
  const [health, setHealth] = useState<"checking" | "ready" | "unreachable" | "not_ready">("checking");
  const [pending, setPending] = useState(0);
  const [driftAlerts, setDriftAlerts] = useState(0);
  const [help, setHelp] = useState(false);
  const [menu, setMenu] = useState(false);
  const [key, setKeyState] = useState(getKey());

  useEffect(() => { const on = () => { setRoute(parse()); setMenu(false); window.scrollTo({ top: 0 }); }; window.addEventListener("hashchange", on); return () => window.removeEventListener("hashchange", on); }, []);
  const poll = useCallback(() => {
    ready().then((h) => setHealth(h ? (h.status === "ready" ? "ready" : "not_ready") : "unreachable"));
    listProposals("pending").then((p) => setPending(p.length)).catch(() => undefined);
    getDriftStatus().then((r) => setDriftAlerts(r.status === "alert" ? r.alerts?.length ?? 1 : 0)).catch(() => undefined);
  }, []);
  useEffect(poll, [poll, route.view]);
  useInterval(poll, 45000);

  const go = (v: View, param?: string) => { location.hash = ROUTES[v] + (param ? `/${param}` : ""); };
  useHotkeys((k) => {
    if (k === "?") { setHelp(true); return true; }
    if (k === "esc") { setHelp(false); setMenu(false); return false; }
    if (k.startsWith("g ") && KEYS[k[2]]) { go(KEYS[k[2]]); return true; }
    return false;
  });

  const badge = (v: View) => (v === "discovery" && pending ? { n: pending, bad: false } : v === "drift" && driftAlerts ? { n: driftAlerts, bad: true } : null);
  const nav: ReactNode = (
    <nav className="nav" aria-label="Sections">
      {NAV.map((g) => (
        <div className="nav-group" key={g.title}>
          <div className="nav-title">{g.title}</div>
          {g.items.map((i) => {
            const b = badge(i.v);
            return (
              <a key={i.v} href={ROUTES[i.v]} aria-current={route.view === i.v ? "page" : undefined} title={`g then ${i.key}`}>
                <Icon d={ICONS[i.v]} />{i.name}{b && <span className={`badge ${b.bad ? "bad" : ""}`} aria-label={`${b.n} need attention`}>{b.n}</span>}
              </a>
            );
          })}
        </div>
      ))}
    </nav>
  );

  const pill = health === "ready" ? <span className="pill ok"><span className="dot ok" />backend ready</span> : health === "checking" ? <span className="pill">checking…</span> : <span className="pill bad"><span className="dot bad" />backend {health.replace("_", " ")}</span>;
  return (
    <div className="shell">
      <a className="skip" href="#main">Skip to content</a>
      <aside className={`side ${menu ? "open" : ""}`}>
        <div className="brand"><div className="brand-mark" aria-hidden="true">R</div><div><b>ResolveIQ</b><span>Support console</span></div></div>
        {nav}
        <div className="side-foot">
          {pill}
          <label className="field" style={{ minWidth: 0 }}>API key (if enabled)<input type="password" value={key} placeholder="X-API-Key" autoComplete="off" onChange={(e) => { setKeyState(e.target.value); setKey(e.target.value); }} /></label>
          <button className="btn sm ghost" onClick={() => setHelp(true)}>Keyboard shortcuts <Kbd>?</Kbd></button>
        </div>
      </aside>
      <div>
        <div className="topbar"><button className="btn sm" aria-label="Open navigation" aria-expanded={menu} onClick={() => setMenu(!menu)}>☰ Menu</button><b>ResolveIQ</b><span className="grow" />{pill}</div>
        <main className="main" id="main" tabIndex={-1}>
          {route.view === "resolve" && <Resolve onOpenCase={(id) => go("cases", id)} />}
          {route.view === "cases" && <CaseReplay caseId={route.param} onSelect={(id) => go("cases", id)} />}
          {route.view === "lab" && <RetrievalLab />}
          {route.view === "discovery" && <Discovery onPendingChange={setPending} />}
          {route.view === "drift" && <Drift />}
          {route.view === "quality" && <Quality onOpenCase={(id) => go("cases", id)} />}
          {route.view === "evaluation" && <Evaluation />}
          {route.view === "health" && <SystemHealth />}
        </main>
      </div>
      {help && <Shortcuts onClose={() => setHelp(false)} />}
    </div>
  );
}
