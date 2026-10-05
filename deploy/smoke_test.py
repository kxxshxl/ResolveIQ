"""Post-deploy smoke test against a running deployment (stdlib only).

  python deploy/smoke_test.py https://support.example.com --api-key <key> [--insecure]
Exit code 0 = healthy. Checks: TLS/headers, liveness, readiness, auth enforcement, metrics not public, search, grounded
resolve with valid citations, out-of-domain abstention, input validation.
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--insecure", action="store_true", help="accept a self-signed / internal-CA certificate")
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    ctx = ssl.create_default_context()
    if a.insecure:
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    base = a.base.rstrip("/")
    failures: list[str] = []

    def call(method, path, body=None, key=True):
        req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None)
        req.add_header("Content-Type", "application/json")
        if key:
            req.add_header("X-API-Key", a.api_key)
        try:
            with urllib.request.urlopen(req, timeout=a.timeout, context=ctx) as r:
                return r.status, dict(r.headers), json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, dict(e.headers), json.loads(raw)
            except ValueError:
                return e.code, dict(e.headers), {}

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
        if not ok:
            failures.append(name)

    s, h, _ = call("GET", "/health", key=False)
    check("liveness", s == 200)
    check("security headers", h.get("X-Content-Type-Options", "").lower() == "nosniff" and "X-Frame-Options" in {k.title() if k.lower() != "x-frame-options" else "X-Frame-Options" for k in h})
    if base.startswith("https"):
        check("HSTS", any(k.lower() == "strict-transport-security" for k in h))
    s, _, body = call("GET", "/health/ready", key=False)
    check("readiness", s == 200 and body.get("status") == "ready", body)
    check("corpus loaded", (body.get("checks", {}).get("corpus", {}).get("tickets", 0)) > 0)
    s, _, _ = call("GET", "/api/v1/stats", key=False)
    check("auth enforced (no key -> 401)", s == 401, s)
    s, _, _ = call("GET", "/metrics", key=False)
    check("metrics not publicly exposed", s in (401, 404), s)
    s, _, body = call("GET", "/api/v1/search?q=my+broadband+keeps+dropping+every+evening&strategy=dense&limit=3")
    check("semantic search", s == 200 and len(body.get("tickets", [])) == 3, body)
    s, _, body = call("POST", "/api/v1/resolve", {"complaint": "My broadband drops every evening around 8 and I have restarted the router twice."})
    ids = {i["source_id"] for i in body.get("tickets", []) + body.get("articles", [])}
    ok = s == 200 and body.get("status") in ("resolved", "degraded") and body["resolution"]["steps"] and \
        all(c["source_id"] in ids for c in body["citations"]) and body["validation"]["valid"]
    check("resolve returns grounded, validated citations", ok, str(body)[:200])
    print(f"     status={body.get('status')} generator={body.get('generator')} confidence={body.get('confidence')}")
    s, _, body = call("POST", "/api/v1/resolve", {"complaint": "What is the best pizza place near the city centre?"})
    check("out-of-domain request abstains", s == 200 and body.get("status") == "abstained" and not body["resolution"]["steps"])
    s, _, _ = call("POST", "/api/v1/resolve", {"complaint": ""})
    check("invalid input -> 422", s == 422, s)
    print("\nSMOKE TEST " + ("FAILED: " + ", ".join(failures) if failures else "PASSED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
