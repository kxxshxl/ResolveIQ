"""Locust load test for POST /api/v1/resolve.

Every simulated support agent sends the next complaint as soon as the previous answer arrives (no think time), so N users means N
requests in flight: this measures capacity and latency under saturation, not a calendar of arrivals.

Complaints come from the repository's own evaluation sets (`data/eval/{queries,gold_v2,gold_blind}.jsonl`, in-domain) plus a small share of
out-of-domain questions (`ood.jsonl`) so the abstain path is exercised. The pool is shuffled with a fixed seed, so a run is repeatable.

Cache mode (LT_CACHE_MODE):
  cold  (default) every request gets a unique case reference appended, so neither the response cache nor the embedding cache can hit:
        this is the real cost of the pipeline.
  warm  complaints repeat exactly, so after the first pass most answers come from the Redis response cache.

Per-request records and an exact-percentile summary are written to $LT_OUT (summary.json, requests.jsonl.gz); Locust's own CSVs are
written next to them by the runner (`--csv`). Configuration comes from environment variables so that run_loadtest.py can drive it:
LT_OUT, LT_META (json), LT_CACHE_MODE, LT_OOD_SHARE, LT_CLIENT_TIMEOUT, LT_SEED.
"""
from __future__ import annotations

import gzip
import itertools
import json
import os
import random
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from locust import HttpUser, constant, events, task

REPO = Path(__file__).resolve().parents[1]
CACHE_MODE = os.getenv("LT_CACHE_MODE", "cold")
OOD_SHARE = float(os.getenv("LT_OOD_SHARE", "0.05"))
CLIENT_TIMEOUT = float(os.getenv("LT_CLIENT_TIMEOUT", "90"))  # longer than the API's own 60 s request timeout, so 504s are observed, not masked
SEED = int(os.getenv("LT_SEED", "42"))
OUT = Path(os.getenv("LT_OUT", str(REPO / "loadtest" / "results" / "_adhoc")))
META = json.loads(os.getenv("LT_META", "{}"))
STAGES = ("preprocess", "embed", "classify", "retrieve", "generate", "total")


def _texts(name: str) -> list[str]:
    path = REPO / "data" / "eval" / f"{name}.jsonl"
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


IN_DOMAIN = _texts("queries") + _texts("gold_v2") + _texts("gold_blind")
OUT_OF_DOMAIN = _texts("ood")
_user_ids = itertools.count()
RECORDS: list[dict] = []
_window = {"start": None, "end": None}


class SupportAgent(HttpUser):
    wait_time = constant(0)

    def on_start(self) -> None:
        self.rng = random.Random(SEED * 1000 + next(_user_ids))

    def _complaint(self) -> str:
        pool = OUT_OF_DOMAIN if self.rng.random() < OOD_SHARE else IN_DOMAIN
        text = self.rng.choice(pool)
        return f"{text} Case ref {uuid.UUID(int=self.rng.getrandbits(128)).hex[:8]}." if CACHE_MODE == "cold" else text

    @task
    def resolve(self) -> None:
        body = {"complaint": self._complaint()}
        t0 = time.perf_counter()
        with self.client.post("/api/v1/resolve", json=body, name="POST /api/v1/resolve", timeout=CLIENT_TIMEOUT, catch_response=True) as r:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            rec = {"t_end": time.time(), "ms": round(elapsed_ms, 1), "http": r.status_code}
            if r.status_code == 200:
                data = r.json()
                rec.update(kind="ok", status=data["status"], generator=data["generator"].split(":")[0], cached=data["cached"],
                           server_ms=data["latency_ms"], escalate=data["resolution"]["escalate"])
            elif r.status_code == 0:
                rec["kind"] = "client_timeout" if "Timeout" in repr(getattr(r, "error", "")) else "connection_error"
                r.failure(rec["kind"])
            else:
                code = ""
                try:
                    code = r.json().get("error", {}).get("code", "")
                except Exception:  # noqa: BLE001
                    pass
                rec["kind"] = "server_timeout" if r.status_code == 504 else f"http_{r.status_code}"
                rec["error_code"] = code
                r.failure(f"{r.status_code} {code}")
            RECORDS.append(rec)


@events.test_start.add_listener
def _start(environment, **_):
    _window["start"] = time.time()


@events.test_stop.add_listener
def _stop(environment, **_):
    _window["end"] = time.time()


def _pct(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    p50, p90, p95, p99 = (float(x) for x in np.percentile(a, [50, 90, 95, 99]))
    return {"n": len(a), "mean": round(float(a.mean()), 1), "p50": round(p50, 1), "p90": round(p90, 1), "p95": round(p95, 1),
            "p99": round(p99, 1), "max": round(float(a.max()), 1)}


@events.quitting.add_listener
def _write(environment, **_):
    OUT.mkdir(parents=True, exist_ok=True)
    wall = max((_window["end"] or time.time()) - (_window["start"] or time.time()), 1e-9)
    n = len(RECORDS)
    ok = [r for r in RECORDS if r["kind"] == "ok"]
    kinds = Counter(r["kind"] for r in RECORDS)
    timeouts = kinds["server_timeout"] + kinds["client_timeout"]
    stage = {s: _pct([r["server_ms"][s] for r in ok if s in r["server_ms"]]) for s in STAGES}
    summary = {
        "meta": META, "users": environment.parsed_options.num_users if environment.parsed_options else None,
        "cache_mode": CACHE_MODE, "ood_share": OOD_SHARE, "seed": SEED, "client_timeout_s": CLIENT_TIMEOUT,
        "pool": {"in_domain": len(IN_DOMAIN), "out_of_domain": len(OUT_OF_DOMAIN)},
        "wall_seconds": round(wall, 1),
        "requests": {"completed": n, "ok": len(ok), "failed": n - len(ok), "by_kind": dict(kinds),
                     "error_rate": round((n - len(ok)) / n, 4) if n else None,
                     "timeout_rate": round(timeouts / n, 4) if n else None},
        "throughput": {"completed_per_s": round(n / wall, 3), "ok_per_s": round(len(ok) / wall, 3)},
        "outcome": dict(Counter(r["status"] for r in ok)), "generator": dict(Counter(r["generator"] for r in ok)),
        "cache_hits": sum(1 for r in ok if r["cached"]),
        "latency_ms_ok": _pct([r["ms"] for r in ok]),
        "latency_ms_all": _pct([r["ms"] for r in RECORDS]),
        "server_stage_ms_ok": stage,
        "notes": ["Percentiles are numpy linear-interpolated over the individual requests; P99 is not meaningful below ~100 requests.",
                  "Requests still in flight when the run ends are not counted."],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with gzip.open(OUT / "requests.jsonl.gz", "wt", encoding="utf-8") as f:  # one JSON line per request; compressed to keep the repository small
        for r in RECORDS:
            f.write(json.dumps(r) + "\n")
