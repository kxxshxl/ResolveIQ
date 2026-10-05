"""Structured JSON logging with a per-request trace id carried in a contextvar."""
from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar

from app.observability.tracing import current_ids

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

_STD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }
        otel = current_ids()
        if otel:  # lets a log line be opened as its trace (and back) when tracing is on
            payload["otel_trace_id"], payload["otel_span_id"] = otel
        for k, v in record.__dict__.items():
            if k not in _STD_ATTRS:
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3", "psycopg.pool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
