"""OpenTelemetry tracing: opt-in, vendor-neutral (OTLP), and free when switched off.

Enable with `OTEL_TRACES_EXPORTER=otlp` (+ `OTEL_EXPORTER_OTLP_ENDPOINT`) or `console`. Left at `none` (the default) nothing is
imported beyond the tiny OTel API, no SDK is created and `span()` / `@traced` reduce to a plain call.

Standard instrumentation covers HTTP (FastAPI), Postgres (psycopg), Redis and the LLM HTTP client (httpx). The pipeline
stages (classify, embed, retrieve, rerank, generate, validate, ingest, jobs) add their own spans through `span()` / `@traced`.

Privacy: spans carry labels, counts, sizes, scores and timings only. Complaint text, prompts and model output are never
attached. The application's own `trace_id` (X-Trace-Id / X-Request-Id, stored in `resolution_requests.trace_id`) is kept
as the span attribute `resolveiq.trace_id`, so a request id from a log line or a support ticket finds its trace.
"""
from __future__ import annotations

import functools
import inspect
import logging
from contextlib import nullcontext
from typing import Any, Callable, ContextManager

from opentelemetry import propagate, trace
from opentelemetry.trace import Link, SpanKind, Status, StatusCode

log = logging.getLogger(__name__)
SERVICE_VERSION = "1.0.0"
# health probes and metrics scrapes are never interesting traces
EXCLUDED_URLS = "/health,/metrics"


class _State:
    provider: Any = None  # SDK TracerProvider once configured
    tracer: Any = None
    exporter: str = "none"
    instrumentors: list = []
    instrumented_apps: list = []
    httpx_instrumentor: Any = None

    @property
    def enabled(self) -> bool:
        return self.provider is not None


_state = _State()


# ------------------------------------------------------------------------------------------------ configuration
def configure(settings, role: str = "api", span_exporter=None, app=None) -> bool:
    """Initialise tracing once per process. Returns True when tracing is active.

    `span_exporter` lets tests inject an in-memory exporter; `app` is instrumented for HTTP if given.
    """
    kind = (settings.otel_traces_exporter or "none").strip().lower()
    if kind not in ("none", "otlp", "console") and span_exporter is None:
        raise ValueError(f"OTEL_TRACES_EXPORTER must be one of none|otlp|console, got '{kind}'")
    if span_exporter is None and kind == "none":
        return _state.enabled
    if not _state.enabled:
        _build(settings, role, kind, span_exporter)
    if app is not None and _state.enabled and app not in _state.instrumented_apps:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, tracer_provider=_state.provider, excluded_urls=EXCLUDED_URLS,
                                            exclude_spans=["receive", "send"])  # drop the per-message ASGI spans: pure noise
        _state.instrumented_apps.append(app)
    return _state.enabled


def _build(settings, role: str, kind: str, span_exporter) -> None:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
    from opentelemetry.sdk.trace.sampling import Decision, ParentBased, Sampler, SamplingResult, TraceIdRatioBased

    class DropOrphanClientSpans(Sampler):
        """Database/Redis/HTTP client spans that are not part of a request or job (pool health checks, worker queue polling,
        readiness probes) would each become a one-span trace. Dropping them keeps the trace list to real work."""

        def __init__(self, delegate: Sampler):
            self._delegate = delegate

        def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None, links=None, trace_state=None):
            if kind == SpanKind.CLIENT and not trace.get_current_span(parent_context).get_span_context().is_valid:
                return SamplingResult(Decision.DROP)
            return self._delegate.should_sample(parent_context, trace_id, name, kind, attributes, links, trace_state)

        def get_description(self) -> str:
            return f"DropOrphanClientSpans({self._delegate.get_description()})"

    ratio = min(max(float(settings.otel_sample_ratio), 0.0), 1.0)
    resource = Resource.create({"service.name": f"{settings.otel_service_name}-{role}", "service.version": SERVICE_VERSION,
                                "deployment.environment": settings.app_env})
    provider = TracerProvider(resource=resource, sampler=ParentBased(root=DropOrphanClientSpans(TraceIdRatioBased(ratio))))
    if span_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        kind = "memory"
    elif kind == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(formatter=lambda s: s.to_json(indent=None) + "\n")))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(**_otlp_kwargs(settings))))

    _state.provider, _state.tracer, _state.exporter = provider, provider.get_tracer("resolveiq", SERVICE_VERSION), kind
    if isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):  # first one wins; never override someone else's
        trace.set_tracer_provider(provider)
    _instrument_libraries(provider)
    log.info("tracing enabled", extra={"exporter": kind, "service": f"{settings.otel_service_name}-{role}", "sample_ratio": ratio})


def _otlp_kwargs(settings) -> dict:
    """Standard OTEL_EXPORTER_OTLP_* variables work through the SDK; these settings additionally make them readable from `.env`."""
    kwargs: dict[str, Any] = {}
    endpoint = settings.otel_exporter_otlp_endpoint.strip()
    if endpoint:
        kwargs["endpoint"] = endpoint if endpoint.rstrip("/").endswith("/v1/traces") else endpoint.rstrip("/") + "/v1/traces"
    headers = {k.strip(): v.strip() for k, _, v in (p.partition("=") for p in settings.otel_exporter_otlp_headers.split(",") if "=" in p)}
    if headers:
        kwargs["headers"] = headers
    return kwargs


def _instrument_libraries(provider) -> None:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    psycopg, redis = PsycopgInstrumentor(), RedisInstrumentor()
    psycopg.instrument(tracer_provider=provider)
    redis.instrument(tracer_provider=provider)
    _state.instrumentors = [psycopg, redis]
    _state.httpx_instrumentor = HTTPXClientInstrumentor


def instrument_http_client(client) -> None:
    """Trace calls made through one httpx client (the LLM providers' client). Per-client so that unrelated httpx use,
    for example FastAPI's TestClient, is not traced."""
    if _state.enabled and _state.httpx_instrumentor is not None:
        _state.httpx_instrumentor.instrument_client(client, tracer_provider=_state.provider)


def shutdown() -> None:
    """Flush and tear down (process exit, and between tests)."""
    if not _state.enabled:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    for app in _state.instrumented_apps:
        FastAPIInstrumentor.uninstrument_app(app)
    for ins in _state.instrumentors:
        ins.uninstrument()
    provider = _state.provider
    _state.provider = _state.tracer = _state.httpx_instrumentor = None
    _state.instrumentors, _state.instrumented_apps, _state.exporter = [], [], "none"
    provider.shutdown()


def flush() -> None:
    if _state.enabled:
        _state.provider.force_flush(5000)


def status() -> str:
    return _state.exporter if _state.enabled else "disabled"


# ------------------------------------------------------------------------------------------------ span helpers
def _clean(attrs: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (attrs or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, bool, int, float)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, bool, int, float)) for x in v):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out


def span(name: str, attrs: dict[str, Any] | None = None, *, kind: SpanKind = SpanKind.INTERNAL,
         links: list[Link] | None = None) -> ContextManager:
    """`with span("retrieval.search", {"resolveiq.k": 5}) as sp:`  Exceptions are recorded and mark the span as an error."""
    if not _state.enabled:
        return nullcontext(trace.INVALID_SPAN)
    return _state.tracer.start_as_current_span(name, kind=kind, attributes=_clean(attrs), links=links)


def annotate(**attrs: Any) -> None:
    """Add attributes to the span that is current (no-op when tracing is off or no span is active)."""
    if _state.enabled:
        sp = trace.get_current_span()
        if sp.is_recording():
            sp.set_attributes(_clean({k.replace("__", "."): v for k, v in attrs.items()}))


def event(name: str, **attrs: Any) -> None:
    if _state.enabled:
        sp = trace.get_current_span()
        if sp.is_recording():
            sp.add_event(name, _clean({k.replace("__", "."): v for k, v in attrs.items()}))


def mark_error(description: str) -> None:
    if _state.enabled:
        sp = trace.get_current_span()
        if sp.is_recording():
            sp.set_status(Status(StatusCode.ERROR, description))


def traced(name: str, attrs: Callable[..., dict] | None = None, result: Callable[[Any], dict] | None = None,
           kind: SpanKind = SpanKind.INTERNAL):
    """Wrap a sync or async function in a span. `attrs` receives the call's arguments and `result` its return value; both are
    best-effort (an attribute extractor that raises is ignored: tracing must never change behaviour)."""

    def safe(fn: Callable | None, *a, **k) -> dict:
        try:
            return fn(*a, **k) if fn else {}
        except Exception:  # noqa: BLE001
            return {}

    def deco(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*args, **kwargs):
                if not _state.enabled:
                    return await fn(*args, **kwargs)
                with span(name, safe(attrs, *args, **kwargs), kind=kind) as sp:
                    out = await fn(*args, **kwargs)
                    if result and sp.is_recording():
                        sp.set_attributes(_clean(safe(result, out)))
                    return out
            return awrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _state.enabled:
                return fn(*args, **kwargs)
            with span(name, safe(attrs, *args, **kwargs), kind=kind) as sp:
                out = fn(*args, **kwargs)
                if result and sp.is_recording():
                    sp.set_attributes(_clean(safe(result, out)))
                return out
        return wrapper

    return deco


# ------------------------------------------------------------------------------------------------ context propagation
def inject_context() -> dict[str, str]:
    """W3C trace context of the current span, for storing with a queued job ({} when tracing is off)."""
    carrier: dict[str, str] = {}
    if _state.enabled:
        propagate.inject(carrier)
    return carrier


def links_from(carrier: dict[str, str] | None) -> list[Link]:
    """A job runs long after the request that enqueued it, so it starts its own trace *linked* to the request's."""
    if not _state.enabled or not carrier:
        return []
    sc = trace.get_current_span(propagate.extract(carrier)).get_span_context()
    return [Link(sc)] if sc.is_valid else []


def current_ids() -> tuple[str, str] | None:
    """(trace_id, span_id) of the active span as hex, for log correlation."""
    sc = trace.get_current_span().get_span_context()
    return (format(sc.trace_id, "032x"), format(sc.span_id, "016x")) if sc.is_valid else None
