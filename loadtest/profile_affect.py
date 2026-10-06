"""Profile the severity/sentiment NLI model (DeBERTa-v3-large zero-shot) in isolation: where one request's time goes, how batching scales, whether
the lock around the forward pass is what limits throughput, and whether cheap forward-pass options help.

    python loadtest/profile_affect.py            # run on an idle GPU; prints a report (the recorded run is results/*/nli_profile.txt)

Findings that drove the optimisation work (see docs/performance.md): tokenisation is ~1 ms, the forward pass is ~35 ms and compute-bound, batching
gives <= 1.08x, dropping the lock gives <= ~1.1x, and padding/inference_mode options do not help, so none of them were adopted.
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
import app  # noqa: E402,F401  (event-loop policy)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from app.classification.affect import CUES, AffectModel  # noqa: E402
from app.core.config import Settings  # noqa: E402

rows = [json.loads(line)["text"] for line in (REPO / "data" / "eval" / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
m = AffectModel(Settings())
m.load()
hyps = list(CUES.values())
print(f"device={m._device} dtype={next(m._model.parameters()).dtype} attention={getattr(m._model.config, '_attn_implementation', None)} "
      f"cues={len(CUES)} max_tokens={m.s.affect_max_tokens}")


def sync() -> None:
    if m._device == "cuda":
        torch.cuda.synchronize()


def timeit(fn, n=30, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(ts), statistics.median(ts), max(ts)


# 1. where does one request's time go?
text = rows[0]
prem = [text for _ in hyps]


def tokenize():
    return m._tok(prem, hyps, return_tensors="pt", padding=True, truncation=True, max_length=m.s.affect_max_tokens)


enc = tokenize().to(m._device)
print(f"\n[1] one request = {len(hyps)} premise/hypothesis pairs, encoded shape {tuple(enc['input_ids'].shape)}")


def forward():
    with torch.no_grad():
        return m._model(**enc).logits


for name, fn in (("tokenize (CPU)", tokenize), ("forward (GPU)", forward), ("whole cue_scores([text])", lambda: m.cue_scores([text]))):
    print("    %-26s mean/median/max ms: %.1f / %.1f / %.1f" % ((name,) + timeit(fn)))

# 2. does batching across requests help?
print("\n[2] batching: texts per forward pass -> ms per text (speed-up vs one text per call)")
base = None
for n in (1, 2, 4, 8, 16, 32):
    texts = rows[:n]
    _, med, _ = timeit(lambda: m.cue_scores(texts, batch=n), n=15, warm=2)
    base = base or med / n
    print(f"    n={n:2d}  {med:7.1f} ms/call  {med / n:6.1f} ms/text  x{base / (med / n):.2f}")
single = np.vstack([m.cue_scores([t]) for t in rows[:64]])
batched = m.cue_scores(rows[:64], batch=16)
ps, pb = m.combine(single), m.combine(batched)
flips = sum(1 for a, b in zip(ps, pb) for d in ("severity", "sentiment") if max(a[d], key=a[d].get) != max(b[d], key=b[d].get))
print(f"    batched vs single, 64 texts: max |dp| = {np.abs(single - batched).max():.5f}, label flips {flips}/128")

# 3. is the lock what limits throughput?
errors: list[str] = []


def unlocked(texts):
    with torch.no_grad():
        p2 = [t for t in texts for _ in hyps]
        h2 = [h for _ in texts for h in hyps]
        e = m._tok(p2, h2, return_tensors="pt", padding=True, truncation=True, max_length=m.s.affect_max_tokens).to(m._device)
        return torch.softmax(m._model(**e).logits, -1)[:, m._entail_idx].float().cpu().numpy()


def worker(k, out):
    try:
        for i in range(10):
            out.append(unlocked([rows[(k * 10 + i) % len(rows)]]))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")


print("\n[3] the lock around the forward pass (calls per second)")
t0 = time.perf_counter()
for i in range(20):
    m.cue_scores([rows[i]])
locked = 20 / (time.perf_counter() - t0)
print(f"    locked, 1 thread:            {locked:5.1f} calls/s")
for threads in (2, 4):
    errors.clear()
    out: list = []
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(k, out)) for k in range(threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    rate = len(out) / (time.perf_counter() - t0)
    print(f"    unlocked, {threads} threads:         {rate:5.1f} calls/s  (x{rate / locked:.2f}, errors: {sorted(set(errors))[:1] or 'none'})")


# 4. cheap forward-pass options
def run(text, pad_multiple=None, inference=False):
    kw = {"pad_to_multiple_of": pad_multiple} if pad_multiple else {}
    ctx = torch.inference_mode if inference else torch.no_grad
    with ctx():
        e = m._tok([text for _ in hyps], hyps, return_tensors="pt", padding=True, truncation=True, max_length=256, **kw).to(m._device)
        p = torch.softmax(m._model(**e).logits, -1)[:, m._entail_idx]
        sync()
        return p.float().cpu().numpy()


def bench(**kw):
    for t in rows[:5]:
        run(t, **kw)
    ts = []
    for t in rows[:40]:
        t0 = time.perf_counter()
        run(t, **kw)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


print("\n[4] forward-pass options (median ms per request)")
b0 = bench()
print(f"    baseline                    {b0:5.1f}")
for label, kw in (("pad_to_multiple_of=8", {"pad_multiple": 8}), ("pad_to_multiple_of=16", {"pad_multiple": 16}), ("inference_mode", {"inference": True})):
    v = bench(**kw)
    print(f"    {label:27s} {v:5.1f}  x{b0 / v:.2f}")
