"""Reproducible load-test runner for ResolveIQ.

For every (scenario, concurrency) it starts a fresh API process, warms it up, drives it with Locust, scrapes the Prometheus metrics
before and after, samples CPU/memory/GPU once a second, and writes everything under loadtest/results/<suite>/<scenario>/c<N>/.

Isolation (nothing here touches the development or evaluation data):
  * a throw-away Postgres database `resolveiq_loadtest` is created, migrated and seeded from scratch, and dropped at the end;
  * Redis DB index 2 with a unique cache namespace per run (cleaned up afterwards);
  * the API runs on its own port with the rate limiter disabled (otherwise it would answer 429 above 2 requests/s) and tracing off,
    unless --trace is given.
A fresh API process per run also resets the LLM circuit breaker and caches, so runs do not influence each other.

Examples (from the repository root; see loadtest/README.md):
  python loadtest/run_loadtest.py --suite 2026-10-06 --scenarios no_llm --levels 1 5 10 25 --duration 60
  python loadtest/run_loadtest.py --suite 2026-10-06 --scenarios llm --levels 1 5 10 25 --duration 120
  python loadtest/run_loadtest.py --suite 2026-10-06 --scenarios no_llm llm --levels 1 10 --duration 60 --trace --tag traced
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx
import psutil
import psycopg

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
sys.path.insert(0, str(BACKEND))
from app.core.config import get_settings  # noqa: E402

TEST_DB = "resolveiq_loadtest"
API_PORT, STUB_PORT, DEAD_PORT = 8110, 11999, 11998
REDIS_DB = 2
BASE = f"http://127.0.0.1:{API_PORT}"

SCENARIOS: dict[str, dict] = {
    "llm": {"desc": "full pipeline, local LLM (Ollama) generates the answer", "env": {}, "warmup": 3},
    "no_llm": {"desc": "evidence-only mode: no LLM provider, everything except generation", "env": {"LLM_PROVIDERS": ""}, "warmup": 3},
    "no_llm_no_affect": {"desc": "ablation: evidence-only and the severity/sentiment NLI model off (AFFECT_ENABLED=false, rule/kNN fallback)",
                         "env": {"LLM_PROVIDERS": "", "AFFECT_ENABLED": "false"}, "warmup": 3},
    "mock_llm_warm": {"desc": "built-in mock LLM (instant, deterministic) with repeated complaints: measures the response-cache path and the API layer without the model",
                      "env": {"LLM_PROVIDERS": "mock"}, "warmup": 3, "cache": "warm"},
    "llm_hot": {"desc": "real LLM, 70% of requests repeat one of 20 popular complaints exactly (response and embedding caches can serve them)",
                "env": {}, "warmup": 3, "hot": (20, 0.7)},
    "no_llm_hot": {"desc": "evidence-only, 70% of requests repeat one of 20 popular complaints exactly", "env": {"LLM_PROVIDERS": ""}, "warmup": 3,
                   "hot": (20, 0.7)},
    "llm_down_hot": {"desc": "LLM unreachable, 70% of requests repeat one of 20 popular complaints exactly",
                     "env": {"OLLAMA_BASE_URL": f"http://127.0.0.1:{DEAD_PORT}"}, "warmup": 2, "hot": (20, 0.7)},
    "no_llm_warm": {"desc": "evidence-only, repeated complaints (response cache can hit)", "env": {"LLM_PROVIDERS": ""}, "warmup": 3, "cache": "warm"},
    "llm_down": {"desc": "LLM unreachable (connection refused), production defaults",
                 "env": {"OLLAMA_BASE_URL": f"http://127.0.0.1:{DEAD_PORT}"}, "warmup": 2},
    "llm_hang": {"desc": "LLM accepts the connection and never answers, production defaults (45 s timeout, 1 retry)",
                 "env": {"OLLAMA_BASE_URL": f"http://127.0.0.1:{STUB_PORT}"}, "stub": True, "warmup": 0},
    "llm_hang_tuned": {"desc": "same hang, LLM_TIMEOUT_SECONDS=15 (worst case 31 s < the 60 s request timeout)",
                       "env": {"OLLAMA_BASE_URL": f"http://127.0.0.1:{STUB_PORT}", "LLM_TIMEOUT_SECONDS": "15"}, "stub": True, "warmup": 0},
}


# ------------------------------------------------------------------------------------------------ helpers
def swap_db(url: str, db: str) -> str:
    return urlunparse(urlparse(url)._replace(path=f"/{db}"))


def port_listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_tree(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        pass


def base_env(suite: str, run: str) -> dict[str, str]:
    s = get_settings()
    redis = urlparse(s.redis_url)
    env = dict(os.environ)
    env.update({
        "DATABASE_URL": swap_db(s.database_url, TEST_DB), "DATABASE_ADMIN_URL": swap_db(s.database_admin_url, TEST_DB),
        "REDIS_URL": urlunparse(redis._replace(path=f"/{REDIS_DB}")), "CACHE_NAMESPACE": f"lt-{suite}-{run}",
        "RATE_LIMIT_PER_MINUTE": "0", "API_KEYS": "", "JOB_EXECUTION": "inline", "APP_ENV": "dev", "PORT": str(API_PORT),
        "OTEL_TRACES_EXPORTER": "none", "PYTHONUNBUFFERED": "1",
    })
    return env


def setup_database(env: dict[str, str]) -> None:
    assert TEST_DB == "resolveiq_loadtest", "refusing to touch any other database"
    root = swap_db(get_settings().database_admin_url, "postgres")
    with psycopg.connect(root, autocommit=True, connect_timeout=5) as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        c.execute(f"CREATE DATABASE {TEST_DB}")
    for cmd in ([sys.executable, "-m", "app.db.migrate"], [sys.executable, "scripts/seed.py"]):
        r = subprocess.run(cmd, cwd=BACKEND, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    print(f"database {TEST_DB} created, migrated and seeded", flush=True)


def drop_database_and_cache(suite: str) -> None:
    root = swap_db(get_settings().database_admin_url, "postgres")
    try:
        with psycopg.connect(root, autocommit=True, connect_timeout=5) as c:
            c.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        print(f"database {TEST_DB} dropped", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"could not drop {TEST_DB}: {exc}", flush=True)
    try:
        import redis

        r = redis.Redis.from_url(urlunparse(urlparse(get_settings().redis_url)._replace(path=f"/{REDIS_DB}")))
        n = sum(1 for k in r.scan_iter("lt-*", count=1000) if r.delete(k))
        print(f"redis db {REDIS_DB}: removed {n} load-test keys", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"redis cleanup skipped: {exc}", flush=True)


def hot_set(n: int) -> list[str]:
    """The same popular-complaint set locustfile.py picks (same pool order and seed)."""
    import random

    pool: list[str] = []
    for name in ("queries", "gold_v2", "gold_blind"):
        pool += [json.loads(line)["text"] for line in (REPO / "data" / "eval" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return random.Random(int(os.getenv("LT_SEED", "42"))).sample(pool, n)


def wait_ready(timeout: float = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(f"{BASE}/health/ready", timeout=3).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise RuntimeError("API did not become ready")


# ------------------------------------------------------------------------------------------------ metrics
def scrape() -> dict[tuple, float]:
    from prometheus_client.parser import text_string_to_metric_families

    text = httpx.get(f"{BASE}/metrics", timeout=10).text
    out = {}
    for fam in text_string_to_metric_families(text):
        for sm in fam.samples:
            out[(sm.name, tuple(sorted(sm.labels.items())))] = sm.value
    return out


def digest_metrics(before: dict, after: dict) -> dict:
    """Server-side view of the same window: mean latency per stage and what the pipeline decided, from Prometheus deltas."""
    def delta(name: str, **labels) -> float:
        want = tuple(sorted(labels.items()))
        tot = 0.0
        for (n, lab), v in after.items():
            if n == name and all(item in lab for item in want):
                tot += v - before.get((n, lab), 0.0)
        return tot

    def hist(base: str, label: str | None = None, values: list[str] | None = None) -> dict:
        out = {}
        for v in values or [None]:
            kw = {label: v} if label else {}
            n, s = delta(f"{base}_count", **kw), delta(f"{base}_sum", **kw)
            if n:
                out[v or "all"] = {"count": int(n), "mean_ms": round(s / n * 1000, 1)}
        return out

    def counter(name: str) -> dict:
        agg: dict[str, float] = {}
        for (n, lab), v in after.items():
            if n == name:
                d = v - before.get((n, lab), 0.0)
                if d:
                    agg[",".join(f"{k}={x}" for k, x in lab)] = int(d)
        return agg

    stages = ["preprocess", "embed", "classify", "retrieve", "generate"]
    return {
        "stage": hist("resolveiq_pipeline_stage_latency_seconds", "stage", stages),
        "http_resolve": hist("resolveiq_http_request_latency_seconds", "path", ["/api/v1/resolve"]),
        "affect_model": hist("resolveiq_affect_latency_seconds"), "embedding": hist("resolveiq_embedding_latency_seconds", "kind", ["single", "batch"]),
        "retrieval": hist("resolveiq_retrieval_latency_seconds", "source_type", ["ticket", "article"]),
        "rerank": hist("resolveiq_rerank_latency_seconds"), "llm_call": hist("resolveiq_llm_latency_seconds", "provider", ["ollama"]),
        "resolutions": counter("resolveiq_resolutions_total"), "llm_failures": counter("resolveiq_llm_failures_total"),
        "cache": counter("resolveiq_cache_total"), "abstentions": counter("resolveiq_abstentions_total"),
        "rate_limited": counter("resolveiq_rate_limited_total"),
        "llm_shed": counter("resolveiq_llm_shed_total"), "llm_queue_wait": hist("resolveiq_llm_queue_wait_seconds", "provider", ["ollama"]),
        "llm_circuit_open_at_end": {dict(lab).get("provider", ""): v for (n, lab), v in after.items() if n == "resolveiq_llm_circuit_open"},
    }


# ------------------------------------------------------------------------------------------------ resource sampling
class Sampler(threading.Thread):
    """Once a second: CPU and memory of the API process tree and the load generator, whole-machine CPU, GPU utilisation."""

    def __init__(self, api_pid: int, out: Path):
        super().__init__(daemon=True)
        self.api, self.out, self.stop_flag, self.locust_pid = psutil.Process(api_pid), out, threading.Event(), None
        self.rows: list[dict] = []
        self._procs: dict[int, psutil.Process] = {}
        self.nvsmi = shutil.which("nvidia-smi")

    def _cores(self, root_pid: int | None) -> float:
        """CPU of a process tree in cores. psutil needs the *same* Process objects across calls to compute a delta."""
        if root_pid is None or not psutil.pid_exists(root_pid):
            return 0.0
        total = 0.0
        try:
            tree = [psutil.Process(root_pid), *psutil.Process(root_pid).children(recursive=True)]
        except psutil.NoSuchProcess:
            return 0.0
        for p in tree:
            proc = self._procs.setdefault(p.pid, p)
            try:
                total += proc.cpu_percent(None)
            except psutil.NoSuchProcess:
                pass
        return round(total / 100, 2)

    def _rss_mb(self) -> int:
        try:
            tree = [self.api, *self.api.children(recursive=True)]
            return round(sum(p.memory_info().rss for p in tree) / 2**20)
        except psutil.NoSuchProcess:
            return 0

    def run(self) -> None:
        psutil.cpu_percent(None)
        self._cores(self.api.pid)
        self._cores(self.locust_pid)
        t0 = time.time()
        while not self.stop_flag.wait(1.0):
            row = {"t": round(time.time() - t0, 1), "system_cpu_pct": psutil.cpu_percent(None), "api_cpu_cores": self._cores(self.api.pid),
                   "api_rss_mb": self._rss_mb(), "locust_cpu_cores": self._cores(self.locust_pid)}
            if self.nvsmi:
                try:
                    q = subprocess.run([self.nvsmi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5).stdout.split(",")
                    row["gpu_util_pct"], row["gpu_mem_mb"] = float(q[0]), float(q[1])
                except Exception:  # noqa: BLE001
                    pass
            self.rows.append(row)

    def finish(self) -> dict:
        self.stop_flag.set()
        self.join(timeout=5)
        if self.rows:
            with (self.out / "resources.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for r in self.rows for k in r}))
                w.writeheader()
                w.writerows(self.rows)

        def stat(key: str) -> dict | None:
            v = [r[key] for r in self.rows if key in r]
            return {"mean": round(sum(v) / len(v), 2), "max": round(max(v), 2)} if v else None
        return {k: stat(k) for k in ("api_cpu_cores", "api_rss_mb", "locust_cpu_cores", "system_cpu_pct", "gpu_util_pct", "gpu_mem_mb")}


# ------------------------------------------------------------------------------------------------ one run
def run_one(suite: str, scenario: str, users: int, duration: int, out_root: Path, trace: bool, tag: str, extra_env: dict | None = None) -> dict:
    cfg = SCENARIOS[scenario]
    run = f"{scenario}-c{users}{('-' + tag) if tag else ''}"
    out = out_root / scenario / f"c{users}{('-' + tag) if tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    env = base_env(suite, run)
    env.update(cfg["env"])
    env.update(extra_env or {})  # --env KEY=VALUE overrides, applied last
    if trace:
        env.update({"OTEL_TRACES_EXPORTER": "otlp", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318"})
    print(f"\n=== {run}: {cfg['desc']} | {users} users x {duration}s{' | tracing ON' if trace else ''}", flush=True)

    stub = None
    if cfg.get("stub"):
        stub = subprocess.Popen([sys.executable, str(REPO / "loadtest" / "stub_llm.py"), "--port", str(STUB_PORT)], stdout=subprocess.DEVNULL)
        time.sleep(1)
    elif "OLLAMA_BASE_URL" in cfg["env"] and port_listening(DEAD_PORT):
        raise RuntimeError(f"port {DEAD_PORT} must be closed for the llm_down scenario")
    log_path = out / "api.log"
    log = log_path.open("w", encoding="utf-8")
    api = subprocess.Popen([sys.executable, "scripts/dev_server.py"], cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT)
    started = time.time()
    try:
        wait_ready()
        startup_s = round(time.time() - started, 1)
        warm = [json.loads(line)["text"] for line in (REPO / "data" / "eval" / "queries.jsonl").read_text(encoding="utf-8").splitlines()[:cfg["warmup"]]]
        for text in warm:  # first-use costs (kernels, connection pool) stay out of the measurement
            try:
                httpx.post(f"{BASE}/api/v1/resolve", json={"complaint": text + f" warmup {time.time_ns()}"}, timeout=120)
            except httpx.HTTPError:
                pass
        if cfg.get("hot"):  # a popular-complaints workload measures the steady state, so answer each popular complaint once first
            for text in hot_set(cfg["hot"][0]):
                try:
                    httpx.post(f"{BASE}/api/v1/resolve", json={"complaint": text}, timeout=120)
                except httpx.HTTPError:
                    pass
        before = scrape()
        t_start_us = int(time.time() * 1e6)

        sampler = Sampler(api.pid, out)
        meta = {"scenario": scenario, "description": cfg["desc"], "users": users, "duration_s": duration, "api_startup_s": startup_s,
                "overrides": {**cfg["env"], **(extra_env or {})}, "tracing": trace}
        lenv = dict(os.environ, LT_OUT=str(out), LT_META=json.dumps(meta), LT_CACHE_MODE=cfg.get("cache", "cold"),
                    LT_HOT_SET=str(cfg.get("hot", (0, 0))[0]), LT_HOT_SHARE=str(cfg.get("hot", (0, 0))[1]))
        cmd = [sys.executable, "-m", "locust", "-f", str(REPO / "loadtest" / "locustfile.py"), "--headless", "-u", str(users), "-r", str(users),
               "-t", f"{duration}s", "--host", BASE, "--csv", str(out / "locust"), "--stop-timeout", "0", "--loglevel", "WARNING"]
        with (out / "locust.log").open("w", encoding="utf-8") as lf:
            lp = subprocess.Popen(cmd, cwd=REPO, env=lenv, stdout=lf, stderr=subprocess.STDOUT)
            sampler.locust_pid = lp.pid
            sampler.start()
            lp.wait()
        resources = sampler.finish()
        after = scrape()
        if trace:
            time.sleep(8)  # let the batch span exporter flush before the process is killed

        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        summary["server_metrics"] = digest_metrics(before, after)
        llm = (summary["server_metrics"].get("llm_call") or {}).get("ollama") or (summary["server_metrics"].get("llm_call") or {}).get("mock") or {}
        wall = max(summary["wall_seconds"], 1e-9)
        summary["llm"] = {"calls": llm.get("count", 0), "calls_per_s": round(llm.get("count", 0) / wall, 3),
                          "mean_call_ms": llm.get("mean_ms"), "busy_fraction": round(llm.get("count", 0) * (llm.get("mean_ms") or 0) / 1000 / wall, 3),
                          "answers_per_s": round(summary.get("llm_generated", 0) / wall, 3),  # model-generated answers, cache hits excluded
                          "shed": summary["server_metrics"].get("llm_shed")}
        summary["resources"] = resources
        for snap_name, snap in (("before", before), ("after", after)):
            with gzip.open(out / f"server_metrics.prom.{snap_name}.json.gz", "wt", encoding="utf-8") as zf:
                json.dump({f"{k[0]}{dict(k[1])}": v for k, v in snap.items()}, zf)
        if trace:
            summary["trace_spans"] = jaeger_span_stats(t_start_us, int(time.time() * 1e6))
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        r, lat = summary["requests"], summary["latency_ms_ok"]
        print(f"    completed={r['completed']} ok={r['ok']} errors={r['error_rate']} timeouts={r['timeout_rate']} "
              f"ok/s={summary['throughput']['ok_per_s']} p50={lat.get('p50')} p95={lat.get('p95')} p99={lat.get('p99')} ms", flush=True)
        return summary
    finally:
        kill_tree(api)
        log.close()
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        log_path.write_text("\n".join(tail) + "\n", encoding="utf-8")  # keep only the tail: the full log is one JSON line per request
        kill_tree(stub)
        time.sleep(3)


def jaeger_span_stats(start_us: int, end_us: int) -> dict:
    """Per-span-name duration statistics for the traces of this run, read back from Jaeger."""
    import numpy as np

    q = {"service": "resolveiq-api", "start": start_us, "end": end_us, "limit": 3000}
    traces = httpx.get("http://127.0.0.1:16686/api/traces", params=q, timeout=60).json().get("data", [])
    durations: dict[str, list[float]] = {}
    for t in traces:
        for sp in t["spans"]:
            durations.setdefault(sp["operationName"], []).append(sp["duration"] / 1000)
    stats = {name: {"count": len(v), "mean_ms": round(float(np.mean(v)), 1), "p95_ms": round(float(np.percentile(v, 95)), 1)}
             for name, v in durations.items() if len(v) >= 3}
    return {"traces": len(traces), "spans": dict(sorted(stats.items(), key=lambda kv: -kv[1]["mean_ms"]))}


def cpu_name() -> str:
    """Marketing name of the CPU (platform.processor() only says 'Intel64 Family 6 ...' on Windows)."""
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=15).stdout.strip()
            if out:
                return out
        elif platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor()


def environment_info() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    s = get_settings()
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": platform.platform(), "python": platform.python_version(),
        "cpu": platform.processor(), "cpu_name": cpu_name(), "cpu_cores_physical": psutil.cpu_count(logical=False), "cpu_threads": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "git_commit": sh(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(sh(["git", "-C", str(REPO), "status", "--porcelain"])),
        "gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "api": "1 uvicorn process (scripts/dev_server.py), no replicas, rate limiter off, tracing off (unless a run says otherwise)",
        "models": {"embedding": s.embedding_model, "reranker": s.reranker_model if s.reranker_enabled else None, "affect": s.affect_model,
                   "default_retrieval_strategy": s.default_retrieval_strategy, "llm": f"ollama:{s.ollama_model} at {s.ollama_base_url}"},
        "llm_settings": {"timeout_s": s.llm_timeout_seconds, "retries": s.llm_max_retries, "circuit_threshold": s.llm_circuit_failure_threshold,
                         "circuit_cooldown_s": s.llm_circuit_cooldown_seconds, "request_timeout_s": s.request_timeout_seconds},
        "ollama_ps": sh(["docker", "exec", "resolveiq-ollama-1", "ollama", "ps"]) or "n/a",
    }
    from importlib.metadata import PackageNotFoundError, version  # not `import locust`: that monkey-patches this process with gevent

    for pkg in ("locust", "torch", "sentence-transformers", "fastapi", "uvicorn", "psycopg"):
        try:
            info.setdefault("versions", {})[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, help="results folder name under loadtest/results/, e.g. 2026-10-06")
    ap.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), required=True)
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 5, 10, 25], help="concurrent users (= requests in flight)")
    ap.add_argument("--duration", type=int, default=60, help="seconds per run")
    ap.add_argument("--trace", action="store_true", help="export OpenTelemetry traces to a Jaeger container and add per-span statistics")
    ap.add_argument("--tag", default="", help="suffix for the run folder (use with --trace so headline runs stay separate)")
    ap.add_argument("--keep-db", action="store_true")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra environment for the API process, repeatable, e.g. --env LLM_MAX_CONCURRENCY=2 (use --tag to keep runs apart)")
    args = ap.parse_args()

    out_root = REPO / "loadtest" / "results" / args.suite
    out_root.mkdir(parents=True, exist_ok=True)
    env_file = out_root / "environment.json"
    if not env_file.exists():
        env_file.write_text(json.dumps(environment_info(), indent=2), encoding="utf-8")
        diff = subprocess.run(["git", "-C", str(REPO), "diff", "--", "backend/app"], capture_output=True, text=True).stdout
        if diff:  # the code under test differs from the recorded commit: keep the exact difference next to the results
            (out_root / "backend_code_changes.diff").write_text(diff, encoding="utf-8")
    jaeger = False
    try:
        setup_database(base_env(args.suite, "setup"))
        if args.trace:
            subprocess.run(["docker", "compose", "--profile", "tracing", "up", "-d", "jaeger"], cwd=REPO, check=True, capture_output=True)
            jaeger = True
            time.sleep(6)
        for scenario in args.scenarios:
            for users in args.levels:
                run_one(args.suite, scenario, users, args.duration, out_root, args.trace, args.tag, dict(kv.split("=", 1) for kv in args.env))
    finally:
        if jaeger:
            subprocess.run(["docker", "compose", "--profile", "tracing", "rm", "-sf", "jaeger"], cwd=REPO, capture_output=True)
        if not args.keep_db:
            drop_database_and_cache(args.suite)


if __name__ == "__main__":
    main()
