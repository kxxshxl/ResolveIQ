import { useEffect, useState } from "react";

export const pct = (x: number) => `${Math.round(x * 100)}%`;
export const label = (s: string) => s.replace(/_/g, " ");

export function Bar({ value, tone }: { value: number; tone?: string }) {
  return (
    <div className="bar" role="meter" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className={`bar-fill ${tone ?? ""}`} style={{ width: pct(Math.max(0, Math.min(1, value))) }} />
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
    <button className="ghost" onClick={() => navigator.clipboard?.writeText(text).then(() => setDone(true)).catch(() => undefined)}>
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
      <button className="ghost" onClick={onClose} aria-label="Dismiss">✕</button>
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
