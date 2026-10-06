"""OpenTelemetry tracing: span tree and attributes of the real request path, preserved trace ids, privacy and the off switch.

The DB-backed tests build a second app whose spans go to an in-memory exporter (no collector needed).
"""
from __future__ import annotations

import json
import logging
import re
import uuid

import pytest

from app.core.config import Settings
from app.core.logging import JsonFormatter
from app.observability import tracing

SECRET = "zqxjv-secret-marker"  # must never appear in any span
COMPLAINT = f"My broadband drops every evening and the router has been restarted twice {SECRET}."


# ---------------------------------------------------------------------------------------------- unit tests (no DB)
def test_tracing_is_a_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing, "_state", tracing._State())
    assert tracing.status() == "disabled"
    with tracing.span("anything", {"a": 1}) as sp:
        assert not sp.is_recording()

    calls = []

    @tracing.traced("never", attrs=lambda x: calls.append("attrs") or {}, result=lambda r: calls.append("result") or {})
    def double(x):
        return x * 2

    assert double(21) == 42 and calls == []  # attribute extractors are not even evaluated
    assert tracing.inject_context() == {} and tracing.links_from({"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}) == []
    tracing.annotate(resolveiq__x=1)  # must not raise


def test_default_settings_leave_tracing_off():
    assert Settings().otel_traces_exporter == "none"
    assert tracing.configure(Settings(otel_traces_exporter="none")) is tracing._state.enabled


def test_unknown_exporter_is_rejected_loudly():
    with pytest.raises(ValueError, match="OTEL_TRACES_EXPORTER"):
        tracing.configure(Settings(otel_traces_exporter="jaeger"))


def test_otlp_endpoint_and_headers_are_normalised():
    kw = tracing._otlp_kwargs(Settings(otel_exporter_otlp_endpoint="http://collector:4318/", otel_exporter_otlp_headers="x-api-key=abc, team=ml"))
    assert kw == {"endpoint": "http://collector:4318/v1/traces", "headers": {"x-api-key": "abc", "team": "ml"}}
    assert tracing._otlp_kwargs(Settings(otel_exporter_otlp_endpoint="http://c:4318/v1/traces"))["endpoint"] == "http://c:4318/v1/traces"
    assert tracing._otlp_kwargs(Settings(otel_exporter_otlp_endpoint="", otel_exporter_otlp_headers="")) == {}


def test_attribute_cleaning_never_fails_on_odd_values():
    out = tracing._clean({"a": None, "b": 1, "c": [1, 2], "d": {"x": 1}, "e": object, "f": ["x", 2]})
    assert "a" not in out and out["b"] == 1 and out["c"] == [1, 2] and out["d"] == "{'x': 1}" and isinstance(out["e"], str) and out["f"] == ["x", 2]


def test_fastapi_native_telemetry_is_switched_off(monkeypatch):
    """FastAPI >= 0.142 auto-enables its own OTLP traces/metrics/logs when a standard OTEL_EXPORTER_OTLP_ENDPOINT is in the
    process environment (how Kubernetes sets it). That would duplicate our spans and export exception text, so it must stay off."""
    from app.main import create_app

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.invalid:4318")
    app = create_app(Settings())
    cfg = getattr(app, "_telemetry", None)
    if cfg is None:
        pytest.skip("this FastAPI version has no native telemetry")
    assert not cfg["auto_configure"] and not any(cfg[k] for k in ("tracing", "metrics", "logs", "operation_spans"))


# ---------------------------------------------------------------------------------------------- with a real span pipeline
@pytest.fixture(scope="module")
def traced(test_settings, test_database):
    from fastapi.testclient import TestClient
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from app.main import create_app
    from tests.scripts_seed_bridge import seed_corpus

    exporter = InMemorySpanExporter()
    application = create_app(test_settings, span_exporter=exporter)
    with TestClient(application) as c:
        c.portal.call(seed_corpus, application.state.services)
        exporter.clear()
        yield c, exporter
    tracing.shutdown()


def _by_name(spans):
    out: dict[str, list] = {}
    for s in spans:
        out.setdefault(s.name, []).append(s)
    return out


def _resolve(traced, complaint=COMPLAINT, request_id=None, strategy=None):
    c, exporter = traced
    exporter.clear()
    headers = {"X-Request-Id": request_id} if request_id else {}
    body = {"complaint": complaint, **({"strategy": strategy} if strategy else {})}
    r = c.post("/api/v1/resolve", json=body, headers=headers)
    assert r.status_code == 200, r.text
    return r, list(exporter.get_finished_spans())


def test_resolve_produces_one_trace_covering_every_stage(traced):
    r, spans = _resolve(traced, f"{COMPLAINT} {uuid.uuid4().hex}", request_id="req-trace-123")
    names = _by_name(spans)
    server = next(s for s in spans if s.name.startswith("POST /api/v1/resolve"))
    assert len({s.context.trace_id for s in spans}) == 1, "everything belongs to the request's trace"

    for expected in ("resolve", "resolve.preprocess", "resolve.cache", "embed.query", "classify", "retrieval.search", "resolve.evidence",
                     "resolve.generate", "llm.generate", "llm.call", "citations.validate", "resolve.persist", "db.dense_search",
                     "db.save_request"):
        assert expected in names, f"missing span {expected}; got {sorted(names)}"
    assert len(names["retrieval.search"]) == 2  # tickets and articles
    assert {s.attributes["resolveiq.retrieval.kind"] for s in names["retrieval.search"]} == {"ticket", "article"}
    assert any("db.system" in s.attributes or "db.system.name" in s.attributes for s in spans), "psycopg spans from the standard instrumentation"

    # nesting: HTTP span -> resolve -> stages -> children
    resolve = names["resolve"][0]
    assert resolve.parent.span_id == server.context.span_id
    for stage in ("resolve.preprocess", "resolve.cache", "classify", "retrieval.search", "resolve.evidence", "resolve.generate"):
        assert names[stage][0].parent.span_id == resolve.context.span_id, stage
    generate = names["resolve.generate"][0]
    assert names["llm.generate"][0].parent.span_id == generate.context.span_id
    assert names["llm.call"][0].parent.span_id == names["llm.generate"][0].context.span_id
    assert names["citations.validate"][0].parent.span_id == generate.context.span_id


def test_span_attributes_expose_outcome_and_latency(traced):
    r, spans = _resolve(traced, f"{COMPLAINT} {uuid.uuid4().hex}")
    body = r.json()
    resolve = _by_name(spans)["resolve"][0].attributes
    assert resolve["resolveiq.request_id"] == body["request_id"]
    assert resolve["resolveiq.status"] == body["status"]
    assert resolve["resolveiq.intent"] == body["classification"]["intent"]
    assert resolve["resolveiq.confidence"] == pytest.approx(body["confidence"])
    assert resolve["resolveiq.cached"] is False
    for stage in ("embed", "classify", "retrieve", "generate", "total"):  # the same stage timings the API returns
        assert resolve[f"resolveiq.latency_ms.{stage}"] == pytest.approx(body["latency_ms"][stage])
    assert "resolveiq.validation.grounded_ratio" in resolve and "resolveiq.evidence.sufficient" in resolve

    names = _by_name(spans)
    assert names["classify"][0].attributes["resolveiq.confidence.intent"] >= 0
    top = [s.attributes for s in names["retrieval.search"]]
    assert all(a["resolveiq.retrieval.returned"] >= 1 and "resolveiq.retrieval.top_score" in a for a in top)
    llm = names["llm.generate"][0].attributes
    assert llm["gen_ai.operation.name"] == "chat" and llm["gen_ai.system"] == "mock" and "resolveiq.llm.prompt_chars" in llm
    assert names["citations.validate"][0].attributes["resolveiq.validation.valid"] is True


def test_reranked_strategy_adds_a_rerank_span(traced):
    _, spans = _resolve(traced, f"{COMPLAINT} {uuid.uuid4().hex}", strategy="hybrid_reranked")
    rerank = _by_name(spans)["retrieval.rerank"]
    assert rerank and rerank[0].attributes["resolveiq.rerank.candidates"] >= 1 and "resolveiq.rerank.top_probability" in rerank[0].attributes


def test_cache_hit_is_visible_and_skips_the_pipeline(traced):
    text = f"{COMPLAINT} cache-probe {uuid.uuid4().hex}"
    _resolve(traced, text)
    _, spans = _resolve(traced, text)
    names = _by_name(spans)
    assert names["resolve"][0].attributes["resolveiq.cached"] is True
    assert names["resolve.cache"][0].attributes["resolveiq.cache_hit"] is True
    assert "classify" not in names and "llm.generate" not in names


def test_existing_trace_id_behaviour_is_preserved(traced):
    c, _ = traced
    r, spans = _resolve(traced, f"{COMPLAINT} {uuid.uuid4().hex}", request_id="req-trace-abc")
    assert r.headers["X-Trace-Id"] == "req-trace-abc" and r.json()["trace_id"] == "req-trace-abc"
    server = next(s for s in spans if s.name.startswith("POST /api/v1/resolve"))
    assert server.attributes["resolveiq.trace_id"] == "req-trace-abc"  # searchable from a log line or support ticket

    generated = c.post("/api/v1/resolve", json={"complaint": f"{COMPLAINT} {uuid.uuid4().hex}"})
    assert re.fullmatch(r"[0-9a-f]{16}", generated.headers["X-Trace-Id"]) and generated.json()["trace_id"] == generated.headers["X-Trace-Id"]

    bad = c.post("/api/v1/resolve", json={"complaint": "x"}, headers={"X-Request-Id": "req-bad-1"})  # error envelopes keep carrying it
    assert bad.status_code == 422 and bad.json()["error"]["trace_id"] == "req-bad-1"
    unsafe = c.get("/api/v1/stats", headers={"X-Request-Id": "bad id with spaces!"})
    assert unsafe.headers["X-Trace-Id"] != "bad id with spaces!"


def test_incoming_w3c_traceparent_is_continued(traced):
    c, exporter = traced
    exporter.clear()
    tid = "4bf92f3577b34da6a3ce929d0e0e4736"
    c.get("/api/v1/stats", headers={"traceparent": f"00-{tid}-00f067aa0ba902b7-01"})
    server = next(s for s in exporter.get_finished_spans() if s.name.startswith("GET /api/v1/stats"))
    assert format(server.context.trace_id, "032x") == tid and server.parent.span_id == int("00f067aa0ba902b7", 16)


def test_health_and_metrics_are_not_traced(traced):
    c, exporter = traced
    exporter.clear()
    c.get("/health")
    c.get("/health/ready")
    c.get("/metrics")
    assert [s.name for s in exporter.get_finished_spans()] == [], "probes and scrapes (and their DB/Redis calls) must not create traces"


def test_no_complaint_text_or_prompt_ends_up_in_any_span(traced):
    _, spans = _resolve(traced, f"{COMPLAINT} {uuid.uuid4().hex}")
    dump = json.dumps([{"name": s.name, "attrs": dict(s.attributes), "events": [(e.name, dict(e.attributes)) for e in s.events]} for s in spans], default=str)
    assert SECRET not in dump
    assert "restarted" not in dump, "no part of the complaint may be recorded"
    redis_statements = [s.attributes.get("db.statement", "") for s in spans if s.attributes.get("db.system") == "redis"]
    assert all(len(stmt) < 120 and "{" not in stmt and "[" not in stmt for stmt in redis_statements), "cached values must not be recorded"


def test_ingestion_is_traced_with_counts_not_content(traced):
    c, exporter = traced
    tid = f"TKT-TR{uuid.uuid4().hex[:6]}"
    exporter.clear()
    body = {"ticket_id": tid, "complaint_text": f"Doorbell camera loses its link whenever the microwave runs {SECRET}.", "intent": "device_issue",
            "product": "wifi_router", "severity": "low", "sentiment": "neutral", "resolution_steps": ["Move the base station."],
            "resolution_summary": "Interference."}
    try:
        assert c.post("/api/v1/ingest/ticket", json=body).status_code == 201
        spans = list(exporter.get_finished_spans())
        names = _by_name(spans)
        ing = names["ingest.ticket"][0]
        assert ing.attributes["resolveiq.ingest.created"] is True and ing.attributes["resolveiq.ingest.source"] == "api"
        assert "embed.query" not in names and "embed.encode" in names and "db.upsert_ticket" in names
        assert SECRET not in json.dumps([dict(s.attributes) for s in spans], default=str)
    finally:
        c.portal.call(lambda: c.app.state.services.repo.delete_by_ids("ticket", [tid]))
        c.portal.call(c.app.state.services.repo.bump_corpus_version)


def test_logs_carry_otel_ids_inside_a_span_only(traced):
    fmt = JsonFormatter()
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "hello", None, None)
    assert "otel_trace_id" not in json.loads(fmt.format(rec))
    with tracing.span("log-correlation"):
        payload = json.loads(fmt.format(rec))
        assert re.fullmatch(r"[0-9a-f]{32}", payload["otel_trace_id"]) and re.fullmatch(r"[0-9a-f]{16}", payload["otel_span_id"])
        assert payload["trace_id"] == "-"  # the application trace id field is untouched


def test_job_keeps_a_link_to_the_enqueuing_request(traced):
    from app import jobs

    c, exporter = traced
    svc = c.app.state.services
    seen = {}

    async def noop(_svc, payload, job_id):
        seen["payload"] = dict(payload)
        return {"ok": True}

    jobs.HANDLERS["noop_trace"] = noop
    try:
        with tracing.span("enqueuer") as parent:
            carrier = tracing.inject_context()
        assert "traceparent" in carrier
        exporter.clear()
        c.portal.call(jobs.execute, svc, "noop_trace", {"x": 1, jobs.TRACE_KEY: carrier}, "job-1")
    finally:
        jobs.HANDLERS.pop("noop_trace", None)
    span = next(s for s in exporter.get_finished_spans() if s.name == "job.noop_trace")
    assert span.attributes["resolveiq.job_id"] == "job-1" and span.attributes["resolveiq.job_kind"] == "noop_trace"
    assert [l.context.trace_id for l in span.links] == [parent.get_span_context().trace_id]
    assert span.parent is None or span.parent.span_id != parent.get_span_context().span_id  # linked, not parented: it may run much later
    assert seen["payload"] == {"x": 1}, "the handler never sees the tracing key"


def test_unreachable_llm_is_recorded_as_error_without_breaking_the_response(traced):
    """A failing provider shows up as an error span, while the API still answers (evidence-only mode)."""
    from app.services.llm.base import ResilientLLM

    c, exporter = traced
    svc = c.app.state.services

    class Broken:
        name, model = "broken", "none"

        async def generate(self, *a, **k):
            raise ConnectionError("llm down")

        async def healthy(self):
            return False

    saved = svc.resolution.llm
    svc.resolution.llm = ResilientLLM([Broken()], svc.settings)
    try:
        r, spans = _resolve(traced, f"{COMPLAINT} {uuid.uuid4().hex}")
    finally:
        svc.resolution.llm = saved
    assert r.json()["status"] in ("degraded", "abstained")
    calls = [s for s in _by_name(spans)["llm.call"] if s.attributes["gen_ai.system"] == "broken"]  # the classifier's own fallback still uses the mock
    assert len(calls) == svc.settings.llm_max_retries + 1 and all(s.status.status_code.name == "ERROR" for s in calls)
    assert [s.attributes["resolveiq.llm.attempt"] for s in calls] == list(range(len(calls)))
    assert any(e.name == "exception" for e in calls[0].events), "the failure is recorded on the span"
    failed_chain = [s for s in _by_name(spans)["llm.generate"] if s.status.status_code.name == "ERROR"]
    assert len(failed_chain) == 1
