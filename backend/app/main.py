"""FastAPI application factory."""
from __future__ import annotations

import inspect
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.console import console
from app.api.routes import api, ops
from app.core.config import Settings, get_settings
from app.core.errors import ResolveIQError
from app.core.logging import configure_logging, trace_id_var
from app.core.version import PIPELINE_VERSION
from app.observability import tracing
from app.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS
from app.services.container import Services
from app.services.llm.base import ResilientLLM

log = logging.getLogger("resolveiq")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# FastAPI >= 0.142 ships its own OpenTelemetry integration that switches itself on, for traces, metrics AND logs (exception messages
# and stack traces), as soon as a standard OTEL_EXPORTER_OTLP_ENDPOINT is present in the process environment, which is exactly how
# Kubernetes configures it. We trace through app.observability.tracing (opt-in, scrubbed of complaint text, one span tree), so
# the native one is switched off to avoid duplicate spans, an unwanted OTLP metrics/logs export and exception text leaving the pod.
_NO_NATIVE_TELEMETRY = (
    {"telemetry": {"tracing": False, "metrics": False, "logs": False, "operation_spans": False, "auto_configure": False}}
    if "telemetry" in inspect.signature(FastAPI.__init__).parameters else {})


def create_app(settings: Settings | None = None, llm: ResilientLLM | None = None, span_exporter=None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.app_env == "production":
            problems = settings.production_problems()
            if problems:
                raise RuntimeError("refusing to start with an insecure production configuration: " + "; ".join(problems))
        if not settings.api_key_set:
            log.warning("API authentication is DISABLED (API_KEYS empty) - acceptable for local dev only")
        app.state.services = await Services.build(settings, llm=llm)
        log.info("resolveiq ready", extra={"providers": settings.provider_chain, "env": settings.app_env})
        yield
        await app.state.services.close()
        tracing.flush()

    hide_docs = settings.app_env == "production" and not settings.expose_api_docs
    app = FastAPI(
        title="ResolveIQ - Telecom Ticket Resolution Assistant", version=PIPELINE_VERSION, lifespan=lifespan,
        description="Semantic retrieval + grounded, cited resolutions for telecom support agents.",
        **({"docs_url": None, "redoc_url": None, "openapi_url": None} if hide_docs else {}), **_NO_NATIVE_TELEMETRY)
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def trace_and_metrics(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        trace_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex[:16]
        token = trace_id_var.set(trace_id)
        tracing.annotate(resolveiq__trace_id=trace_id)  # the id in X-Trace-Id and logs, searchable on the HTTP span
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Trace-Id"] = trace_id
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")  # route template keeps label cardinality bounded
            elapsed = time.perf_counter() - start
            HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            HTTP_LATENCY.labels(path).observe(elapsed)
            if path not in ("/metrics", "/health"):
                log.info("request", extra={"method": request.method, "path": path, "status": status,
                                           "duration_ms": round(elapsed * 1000, 1)})
            trace_id_var.reset(token)

    def envelope(code: str, message: str, status: int, extra: dict | None = None, headers: dict | None = None):
        body = {"error": {"code": code, "message": message, "trace_id": trace_id_var.get(), **(extra or {})}}
        return JSONResponse(body, status_code=status, headers=headers)

    @app.exception_handler(ResolveIQError)
    async def _app_error(_: Request, exc: ResolveIQError):
        if exc.status_code >= 500:
            log.error("request failed", extra={"code": exc.code, "error": exc.message[:300]})
        return envelope(exc.code, exc.message, exc.status_code)

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException):
        return envelope("http_error", str(exc.detail), exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "issue": e["msg"]} for e in exc.errors()]
        return envelope("validation_error", "request validation failed", 422, {"details": details})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return envelope("internal_error", "unexpected server error (see logs with this trace_id)", 500)

    app.include_router(ops)
    app.include_router(api)
    app.include_router(console)
    tracing.configure(settings, role="api", span_exporter=span_exporter, app=app)  # no-op unless OTEL_TRACES_EXPORTER is set
    return app


app = create_app()
