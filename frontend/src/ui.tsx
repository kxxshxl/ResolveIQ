import { useEffect, useId, useRef, useState, type ReactNode } from "react";

// ---------------------------------------------------------------- formatting
export const pct = (x: number | null | undefined) => (x == null ? "n/a" : `${Math.round(x * 100)}%`);
export const label = (s: string | null | undefined) => (s ?? "").replace(/_/g, " ");
export const ms = (x: number | null | undefined) => (x == null ? "n/a" : x >= 1000 ? `${(x / 1000).toFixed(1)} s` : x >= 10 ? `${Math.round(x)} ms` : `${x.toFixed(1)} ms`);
export const fixed = (x: number | null | undefined, d = 2) => (x == null ? "n/a" : x.toFixed(d));
export const bytes = (n: number) => (n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(1)} GB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(1)} MB` : n >= 1024 ? `${Math.round(n / 1024)} KB` : `${n} B`);
export const ago = (iso: string | number | null | undefined) => {
  if (iso == null) return "never";
  const s = typeof iso === "number" ? iso : (Date.now() - new Date(iso).getTime()) / 1000;
  return s < 90 ? "just now" : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`;
};
export const when = (iso: string) => new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

// ---------------------------------------------------------------- primitives
export type Tone = "ok" | "warn" | "bad" | "info" | "neutral";

export function Pill({ tone = "neutral", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return <span className={`pill ${tone}`} title={title}>{children}</span>;
}

export function Bar({ value, tone }: { value: number; tone?: string }) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div className="bar" role="meter" aria-valuenow={Math.round(v * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className={`bar-fill ${tone ?? ""}`} style={{ width: pct(v) }} />
    </div>
  );
}

export function Stat({ name, value, hint, tone }: { name: string; value: ReactNode; hint?: string; tone?: Tone }) {
  return (
    <div className={`stat ${tone ?? ""}`} title={hint}>
      <span className="stat-name">{name}</span>
      <strong className="stat-value">{value}</strong>
    </div>
  );
}

export function Panel({ title, subtitle, actions, children, className = "", id }: { title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; id?: string }) {
  const uid = useId();
  return (
    <section className={`panel ${className}`} id={id} aria-labelledby={title ? uid : undefined}>
      {(title || actions) && (
        <header className="panel-head">
          <div className="grow">{title && <h2 id={uid}>{title}</h2>}{subtitle && <p className="muted small">{subtitle}</p>}</div>
          {actions && <div className="row gap-s">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty" role="status">
      <strong>{title}</strong>
      {children && <p className="muted small">{children}</p>}
      {action}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="errorbox" role="alert">
      <strong>Something went wrong</strong>
      <span className="small">{message}</span>
      {onRetry && <button className="btn sm" onClick={onRetry}>Retry</button>}
    </div>
  );
}

export function Skeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="skeleton" aria-hidden="true">
      {Array.from({ length: lines }, (_, i) => <div key={i} className="skel-line" style={{ width: `${92 - i * 14}%` }} />)}
    </div>
  );
}

export function Loading({ what }: { what: string }) {
  return <div aria-busy="true" aria-live="polite"><span className="sr-only">Loading {what}</span><Skeleton lines={4} /></div>;
}

export function KV({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <table className="kv"><tbody>{rows.map(([k, v]) => <tr key={k}><th scope="row">{k}</th><td>{v}</td></tr>)}</tbody></table>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="kbd">{children}</kbd>;
}

/** Accessible tab list (arrow keys move, Home/End jump). The caller renders the active panel. */
export function Tabs<T extends string>({ tabs, value, onChange, label: aria, variant = "pill" }: { tabs: { id: T; name: ReactNode; count?: number }[]; value: T; onChange: (v: T) => void; label: string; variant?: "pill" | "line" }) {
  const ref = useRef<HTMLDivElement>(null);
  const onKey = (e: React.KeyboardEvent) => {
    const i = tabs.findIndex((t) => t.id === value);
    const next = e.key === "ArrowRight" ? (i + 1) % tabs.length : e.key === "ArrowLeft" ? (i - 1 + tabs.length) % tabs.length : e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : -1;
    if (next >= 0) { e.preventDefault(); onChange(tabs[next].id); (ref.current?.querySelectorAll<HTMLButtonElement>("[role=tab]")[next])?.focus(); }
  };
  return (
    <div className={`tabs ${variant}`} role="tablist" aria-label={aria} ref={ref} onKeyDown={onKey}>
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} tabIndex={value === t.id ? 0 : -1} className={value === t.id ? "tab active" : "tab"} onClick={() => onChange(t.id)}>
          {t.name}{t.count != null && <span className="count">{t.count}</span>}
        </button>
      ))}
    </div>
  );
}

/** Copies text and flips the label to "Copied" for a moment; falls back silently where the clipboard API is blocked. */
export function CopyButton({ text, children = "Copy" }: { text: string; children?: string }) {
  const [done, setDone] = useState(false);
  useEffect(() => {
    if (!done) return;
    const t = setTimeout(() => setDone(false), 1500);
    return () => clearTimeout(t);
  }, [done]);
  return (
    <button className="btn sm ghost" onClick={() => navigator.clipboard?.writeText(text).then(() => setDone(true)).catch(() => undefined)}>
      {done ? "Copied ✓" : children}
    </button>
  );
}

export interface Toast { tone: "ok" | "bad"; text: string }

export function ToastBar({ toast, onClose }: { toast: Toast | null; onClose: () => void }) {
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(onClose, toast.tone === "ok" ? 6000 : 10000);
    return () => clearTimeout(t);
  }, [toast, onClose]);
  if (!toast) return null;
  return (
    <div className={`toast ${toast.tone}`} role={toast.tone === "bad" ? "alert" : "status"}>
      <span>{toast.text}</span>
      <button className="btn sm ghost" onClick={onClose} aria-label="Dismiss">✕</button>
    </div>
  );
}

/** Modal dialog: focus moves in, Escape and the backdrop close it, focus returns to the opener. */
export function Dialog({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  const id = useId();
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    box.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("keydown", onKey); opener?.focus?.(); };
  }, [onClose]);
  return (
    <div className="overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby={id} tabIndex={-1} ref={box}>
        <header className="panel-head"><h2 id={id} className="grow">{title}</h2><button className="btn sm ghost" onClick={onClose} aria-label="Close">✕</button></header>
        {children}
      </div>
    </div>
  );
}
