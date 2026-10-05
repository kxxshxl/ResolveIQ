from __future__ import annotations


class ResolveIQError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if status_code:
            self.status_code = status_code


class NotFoundError(ResolveIQError):
    status_code, code = 404, "not_found"


class ConflictError(ResolveIQError):
    status_code, code = 409, "conflict"


class ValidationFailure(ResolveIQError):
    status_code, code = 422, "validation_error"


class RetrievalError(ResolveIQError):
    status_code, code = 503, "retrieval_unavailable"


class LLMUnavailable(ResolveIQError):
    status_code, code = 503, "llm_unavailable"


class RequestTimeout(ResolveIQError):
    status_code, code = 504, "request_timeout"
