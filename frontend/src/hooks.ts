import { useCallback, useEffect, useRef, useState } from "react";

export interface Async<T> { data: T | null; error: string | null; loading: boolean; reload: () => void }

/** Runs `fn` on mount and whenever `deps` change; keeps the previous data visible while reloading so the page does not flash empty. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []): Async<T> {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: true });
  const [tick, setTick] = useState(0);
  const latest = useRef(0);
  useEffect(() => {
    const id = ++latest.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    fn().then(
      (data) => { if (id === latest.current) setState({ data, error: null, loading: false }); },
      (e: Error) => { if (id === latest.current) setState((s) => ({ ...s, error: e.message, loading: false })); },
    );
  }, [...deps, tick]); // eslint-disable-line react-hooks/exhaustive-deps
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { ...state, reload };
}

/** Calls `fn` every `ms` while `on` is true (and the tab is visible). */
export function useInterval(fn: () => void, ms: number, on = true) {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    if (!on) return;
    const id = setInterval(() => { if (!document.hidden) saved.current(); }, ms);
    return () => clearInterval(id);
  }, [ms, on]);
}

export function useLocalState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try { const raw = localStorage.getItem(key); return raw === null ? initial : (JSON.parse(raw) as T); } catch { return initial; }
  });
  const set = useCallback((v: T) => { setValue(v); try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* storage unavailable: the setting just does not persist */ } }, [key]);
  return [value, set];
}

const isTyping = (t: EventTarget | null) => {
  const el = t as HTMLElement | null;
  return !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
};

/** Global shortcuts. Plain keys are ignored while typing; "g x" sequences work like in GitHub. */
export function useHotkeys(handler: (combo: string, e: KeyboardEvent) => boolean | void) {
  const saved = useRef(handler);
  saved.current = handler;
  useEffect(() => {
    let pendingG = 0;
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.altKey) return;
      const typing = isTyping(e.target);
      if (e.key === "Escape") { saved.current("esc", e); return; }
      if (e.ctrlKey) { if (e.key === "Enter") saved.current("ctrl+enter", e); return; }
      if (typing) return;
      if (pendingG && Date.now() - pendingG < 900) {
        pendingG = 0;
        if (saved.current(`g ${e.key.toLowerCase()}`, e)) e.preventDefault();
        return;
      }
      if (e.key.toLowerCase() === "g" && !e.shiftKey) { pendingG = Date.now(); return; }
      if (saved.current(e.key, e)) e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
