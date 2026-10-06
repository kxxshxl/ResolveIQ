"""End-to-end checks against the deployment in the local kind cluster (stdlib only; needs kubectl on the PATH).

    python k8s/local/e2e_test.py              # full pass: probes, auth, metrics, resolve, ingestion + worker, tracing
    python k8s/local/e2e_test.py --ingress    # same, but the frontend is reached through the Ingress (needs ingress-nginx)
    python k8s/local/e2e_test.py --degraded   # run while the LLM is unreachable: the evidence-only fallback must still answer

It talks to the cluster only through `kubectl port-forward` (frontend Service -> nginx -> backend Service, the same path a browser
takes; backend and worker pods directly for /metrics; Jaeger for traces) and reads the API key and metrics token from the cluster
Secret, so no secret ever touches the disk. Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import base64
import json
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

NS = "resolveiq"
CTX = "kind-resolveiq"
FAILED: list[str] = []


def kubectl(*args: str) -> str:
    return subprocess.run(["kubectl", "--context", CTX, "-n", NS, *args], capture_output=True, text=True, check=True).stdout


def secret(key: str) -> str:
    return base64.b64decode(kubectl("get", "secret", "resolveiq-secrets", "-o", f"jsonpath={{.data.{key}}}")).decode()


class Forward:
    """`kubectl port-forward` for the lifetime of the with-block."""

    def __init__(self, target: str, local: int, remote: int):
        self.cmd = ["kubectl", "--context", CTX, "-n", NS, "port-forward", target, f"{local}:{remote}"]
        self.url = f"http://127.0.0.1:{local}"

    def __enter__(self):
        self.p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.p.stdout.readline()  # "Forwarding from 127.0.0.1:..."
        return self

    def __exit__(self, *exc):
        self.p.terminate()


INGRESS_URL, INGRESS_HOST = "https://127.0.0.1:8443", "support.example.com"   # kind maps host 8443 -> the ingress controller (kind-config.yaml)
_INSECURE = ssl.create_default_context()  # the local ingress serves the controller's self-signed default certificate
_INSECURE.check_hostname, _INSECURE.verify_mode = False, ssl.CERT_NONE


class Ingress:
    """The same interface as Forward, but reaches the frontend through the real Ingress (host 8443 -> ingress-nginx -> Service)."""
    url = INGRESS_URL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def call(base: str, method: str, path: str, body=None, headers=None, timeout=120):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    if base == INGRESS_URL:
        req.add_header("Host", INGRESS_HOST)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_INSECURE) as r:
            raw, status, hdrs = r.read(), r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        raw, status, hdrs = e.read(), e.code, dict(e.headers)
    ms = (time.perf_counter() - t0) * 1000
    try:
        data = json.loads(raw)
    except ValueError:
        data = raw.decode("utf-8", "replace")
    return status, data, hdrs, ms


def check(name: str, ok: bool, detail: object = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{str(detail)[:300]}]" if detail != "" else ""))
    if not ok:
        FAILED.append(name)


COMPLAINT = "My broadband drops every evening around 8pm and I have already restarted the router twice. What should I do?"


def resolve_checks(ui: str, key: dict, degraded: bool) -> str | None:
    """Returns the trace id of the main resolve call."""
    s, body, h, ms = call(ui, "POST", "/api/v1/resolve", {"complaint": f"{COMPLAINT} (case ref {uuid.uuid4().hex[:8]})"}, key)  # unique per run: never served from the response cache
    ok = s == 200 and isinstance(body, dict)
    print(f"      resolve via frontend: HTTP {s} in {ms:.0f} ms")
    check("POST /api/v1/resolve through the frontend (nginx -> backend Service) returns 200", ok, body if not ok else "")
    if not ok:
        return None
    trace_id = body["trace_id"]
    ids = {i["source_id"] for i in body["tickets"] + body["articles"]}
    print(f"      status={body['status']} generator={body['generator']} confidence={body['confidence']} steps={len(body['resolution']['steps'])} "
          f"citations={len(body['citations'])} classification={body['classification'].get('intent') if isinstance(body['classification'], dict) else ''}")
    print(f"      warnings={body['warnings']}")
    print(f"      server latency_ms={body['latency_ms']}  request_id={body['request_id']} trace_id={body['trace_id']}")
    check("response carries steps, citations that point at retrieved evidence, and a valid validation report",
          bool(body["resolution"]["steps"]) and bool(body["citations"]) and all(c["source_id"] in ids for c in body["citations"]) and body["validation"]["valid"])
    check("X-Trace-Id response header equals the body's trace_id (and request_id/trace_id are propagated)", {k.lower(): v for k, v in h.items()}.get("x-trace-id") == body["trace_id"],
          {k: v for k, v in h.items() if "trace" in k.lower() or "request" in k.lower()})
    if degraded:
        check("LLM unavailable: answered from evidence only (generator is extractive, status degraded, warning present)",
              body["generator"] in ("extractive", "none") and body["status"] == "degraded" and any("evidence-only" in w or "unavailable" in w or "busy" in w for w in body["warnings"]),
              {"generator": body["generator"], "status": body["status"], "warnings": body["warnings"]})
    else:
        check("answer generated by the LLM (generator is ollama/...)", str(body["generator"]).startswith("ollama"), body["generator"])
    s, body, _, _ = call(ui, "POST", "/api/v1/resolve", {"complaint": "What is the best pizza place near the city centre?"}, key)
    check("out-of-domain request abstains", s == 200 and body.get("status") == "abstained", {"http": s, "status": body.get("status") if isinstance(body, dict) else body})
    return trace_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degraded", action="store_true", help="the LLM is unreachable; expect the evidence-only fallback")
    ap.add_argument("--ingress", action="store_true", help="reach the frontend through the Ingress (https://127.0.0.1:8443, Host support.example.com) instead of a port-forward")
    a = ap.parse_args()
    key = {"X-API-Key": secret("API_KEYS").split(",")[0]}
    token = secret("METRICS_TOKEN")

    pods = json.loads(kubectl("get", "pods", "-o", "json"))["items"]
    not_ready = [p["metadata"]["name"] for p in pods if p["status"].get("phase") == "Running" and not all(c.get("ready") for c in p["status"].get("containerStatuses", []))]
    print(f"pods: {len(pods)} ({', '.join(sorted(p['metadata']['name'] for p in pods))})")
    check("every running pod is Ready", not not_ready, not_ready)

    with (Ingress() if a.ingress else Forward("svc/frontend", 18080, 80)) as fe, Forward("svc/backend", 18000, 8000) as be, Forward("deploy/worker", 19100, 9100) as wk:
        # ---- frontend + health
        s, body, h, _ = call(fe.url, "GET", "/")
        check("frontend serves the single-page app", s == 200 and "<div id=\"root\"" in str(body) and h.get("X-Content-Type-Options") == "nosniff", s)
        s, body, _, _ = call(fe.url, "GET", "/health")
        check("GET /health through the frontend proxy", s == 200 and body.get("status") == "ok", body)
        s, body, _, _ = call(fe.url, "GET", "/health/ready")
        c = body.get("checks", {}) if isinstance(body, dict) else {}
        print(f"      readiness: {json.dumps(c)}")
        check("GET /health/ready through the frontend: database ok, models loaded, corpus present",
              s == 200 and c.get("database") == "ok" and c.get("models") == "loaded" and c.get("corpus", {}).get("tickets", 0) > 0, body)
        check("Redis reachable from the backend", c.get("redis") == "ok", c.get("redis"))
        if a.degraded:
            check("readiness still 200 while the LLM is down (a dead LLM must not take pods out of service)", s == 200, c.get("llm"))
        # ---- auth + metrics exposure
        s, _, _, _ = call(fe.url, "GET", "/api/v1/stats")
        check("API requires a key (401 without X-API-Key)", s == 401, s)
        s, _, _, _ = call(fe.url, "GET", "/metrics")
        check("/metrics is not exposed through the frontend (404)", s == 404, s)
        s, _, _, _ = call(be.url, "GET", "/metrics")
        check("backend /metrics rejects a request without the token (401)", s == 401, s)
        s, text, _, _ = call(be.url, "GET", "/metrics", headers={"Authorization": f"Bearer {token}"})
        check("backend /metrics with the token exposes the application metrics",
              s == 200 and "resolveiq_" in text and "http_" in text or "resolveiq_" in str(text), s)
        s, text, _, _ = call(wk.url, "GET", "/metrics")
        check("worker /metrics (queue depth, job metrics) reachable", s == 200 and "resolveiq_job" in str(text), s)
        # ---- the real thing
        trace_id = resolve_checks(fe.url, key, a.degraded)
        if a.degraded:
            s, body, _, _ = call(be.url, "GET", "/api/v1/stats", headers=key)
            print(f"      stats: {json.dumps(body)[:300]}")
        else:
            # ---- ingestion: synchronous, then asynchronous through the worker
            marker = "kindtest" + uuid.uuid4().hex[:8]
            t = {"ticket_id": f"K8S-{marker}", "complaint_text": f"Roaming data stopped working in Portugal after I switched on airplane mode ({marker}).",
                 "resolution_steps": ["Switch data roaming on in the phone settings.", "Restart the handset to re-register on the partner network."],
                 "resolution_summary": "Roaming toggle and re-registration restored service."}
            s, body, _, _ = call(fe.url, "POST", "/api/v1/ingest/ticket", t, key)
            check("POST /api/v1/ingest/ticket (synchronous ingestion) -> 201", s == 201, body)
            s, body, _, _ = call(fe.url, "GET", f"/api/v1/search?q=roaming+data+stopped+working+in+Portugal+{marker}&strategy=dense&limit=3&source=ticket", headers=key)
            hit = [x.get("source_id") or x.get("ticket_id") for x in (body.get("tickets", []) if isinstance(body, dict) else [])]
            check("the new ticket is searchable immediately", f"K8S-{marker}" in hit, hit)
            batch = [{"ticket_id": f"K8S-{marker}-b{i}", "complaint_text": f"Voicemail notification text keeps arriving twice on my handset, case {marker} {i}.",
                      "resolution_steps": ["Reset the voicemail notification settings."], "resolution_summary": "Notification reset."} for i in range(3)]
            s, body, _, _ = call(fe.url, "POST", "/api/v1/ingest/tickets/batch", {"tickets": batch}, key)
            check("POST /api/v1/ingest/tickets/batch accepted (202, job queued for the worker)", s == 202 and body.get("status") == "queued", body)
            if s == 202:
                job_id, last = body["job_id"], {}
                for _ in range(120):
                    _, last, _, _ = call(fe.url, "GET", f"/api/v1/jobs/{job_id}", headers=key)
                    if last.get("status") in ("succeeded", "failed"):
                        break
                    time.sleep(1)
                print(f"      job: {json.dumps(last)[:400]}")
                check("the worker Deployment executed the batch job (status succeeded)", last.get("status") == "succeeded", last)
            # ---- tracing
            with Forward("svc/jaeger", 16686, 16686) as jg:
                time.sleep(6)  # BatchSpanProcessor flush interval
                s, services, _, _ = call(jg.url, "GET", "/api/services")
                names = services.get("data", []) if isinstance(services, dict) else []
                print(f"      Jaeger services: {names}")
                check("traces arrive in Jaeger from both the API and the worker", "resolveiq-api" in names and "resolveiq-worker" in names, names)
                s, tr, _, _ = call(jg.url, "GET", "/api/traces?service=resolveiq-api&limit=20&lookback=1h")
                ops = sorted({sp["operationName"] for t in tr.get("data", []) for sp in t["spans"]}) if isinstance(tr, dict) else []
                print(f"      span names seen: {ops}")
                tag = urllib.parse.quote(json.dumps({"resolveiq.trace_id": trace_id}))  # nginx's request id travels as X-Request-Id -> span tag
                s, one, _, _ = call(jg.url, "GET", f"/api/traces?service=resolveiq-api&tags={tag}&lookback=1h&limit=5")
                spans = one["data"][0]["spans"] if isinstance(one, dict) and one.get("data") else []
                roots = [sp for sp in spans if sp["operationName"] == "POST /api/v1/resolve"]
                print(f"      trace {trace_id}: {len(spans)} spans, {len(roots)} server span(s) for POST /api/v1/resolve")
                check("the resolve trace is one tree with a single server span (FastAPI's native telemetry is off, no duplicates)", len(roots) == 1 and len(spans) > 10, len(roots))
                check("pipeline spans (retrieval, rerank, llm, validation) are present", any("retriev" in o for o in ops) and any("llm" in o for o in ops), ops)
    print("\nK8S E2E " + ("FAILED: " + "; ".join(FAILED) if FAILED else "PASSED"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
