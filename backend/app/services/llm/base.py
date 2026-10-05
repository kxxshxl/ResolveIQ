"""LLM provider abstraction: pluggable providers behind retry + circuit breaker + ordered fallback."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol

from app.core.config import Settings
from app.core.errors import LLMUnavailable
from app.observability import tracing
from app.observability.metrics import LLM_CIRCUIT_OPEN, LLM_FAILURES, LLM_LATENCY, LLM_TOKENS

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0


class LLMProvider(Protocol):
    name: str
    model: str

    async def generate(self, system: str, user: str, *, json_mode: bool = True, max_tokens: int = 700,
                       temperature: float = 0.1, timeout: float = 45.0) -> LLMResult: ...

    async def healthy(self) -> bool: ...


class CircuitBreaker:
    """Opens after N consecutive failures; half-opens after a cooldown (one trial call)."""

    def __init__(self, threshold: int, cooldown_s: float, name: str):
        self.threshold, self.cooldown_s, self.name = threshold, cooldown_s, name
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.cooldown_s:
            return False  # half-open: allow a trial
        return True

    def record_success(self) -> None:
        self.failures, self.opened_at = 0, None
        LLM_CIRCUIT_OPEN.labels(self.name).set(0)

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()
            LLM_CIRCUIT_OPEN.labels(self.name).set(1)
            log.warning("llm circuit opened", extra={"provider": self.name, "failures": self.failures})


class ResilientLLM:
    """Tries providers in order; each is guarded by retries (exp. backoff) and a circuit breaker."""

    def __init__(self, providers: list[LLMProvider], settings: Settings):
        self.providers = providers
        self.s = settings
        self.breakers = {
            p.name: CircuitBreaker(settings.llm_circuit_failure_threshold, settings.llm_circuit_cooldown_seconds, p.name)
            for p in providers
        }

    @property
    def available(self) -> bool:
        return bool(self.providers)

    @tracing.traced(
        "llm.generate",
        attrs=lambda self, system, user, *, json_mode=True, max_tokens=None, temperature=0.1: {
            "gen_ai.operation.name": "chat", "gen_ai.request.temperature": temperature,
            "gen_ai.request.max_tokens": max_tokens or self.s.llm_max_tokens, "resolveiq.llm.json_mode": json_mode,
            "resolveiq.llm.provider_chain": [p.name for p in self.providers], "resolveiq.llm.prompt_chars": len(system) + len(user)},
        result=lambda r: {"gen_ai.system": r.provider, "gen_ai.response.model": r.model, "gen_ai.usage.input_tokens": r.prompt_tokens,
                          "gen_ai.usage.output_tokens": r.completion_tokens, "resolveiq.llm.response_chars": len(r.text)})
    async def generate(self, system: str, user: str, *, json_mode: bool = True, max_tokens: int | None = None,
                       temperature: float = 0.1) -> LLMResult:
        errors: list[str] = []
        for p in self.providers:
            br = self.breakers[p.name]
            if br.is_open:
                errors.append(f"{p.name}: circuit open")
                tracing.event("llm.circuit_open", gen_ai__system=p.name)
                continue
            for attempt in range(self.s.llm_max_retries + 1):
                t0 = time.perf_counter()
                try:
                    with tracing.span("llm.call", {"gen_ai.system": p.name, "gen_ai.request.model": p.model, "resolveiq.llm.attempt": attempt}) as sp:
                        res = await p.generate(system, user, json_mode=json_mode, max_tokens=max_tokens or self.s.llm_max_tokens,
                                               temperature=temperature, timeout=self.s.llm_timeout_seconds)
                        if sp.is_recording():
                            sp.set_attributes(tracing._clean({"gen_ai.usage.input_tokens": res.prompt_tokens,
                                                              "gen_ai.usage.output_tokens": res.completion_tokens}))
                    LLM_LATENCY.labels(p.name).observe(time.perf_counter() - t0)
                    if res.prompt_tokens:
                        LLM_TOKENS.labels(p.name, "prompt").inc(res.prompt_tokens)
                    if res.completion_tokens:
                        LLM_TOKENS.labels(p.name, "completion").inc(res.completion_tokens)
                    br.record_success()
                    return res
                except Exception as exc:  # noqa: BLE001
                    reason = "timeout" if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "Timeout" in type(exc).__name__ else "error"
                    LLM_FAILURES.labels(p.name, reason).inc()
                    errors.append(f"{p.name}[{attempt}]: {type(exc).__name__}: {exc}")
                    log.warning("llm call failed", extra={"provider": p.name, "attempt": attempt, "reason": reason, "error": str(exc)[:200]})
                    if attempt < self.s.llm_max_retries:
                        await asyncio.sleep(0.5 * (2 ** attempt))
            br.record_failure()
        raise LLMUnavailable("all LLM providers failed: " + "; ".join(errors[-4:]))

    async def health(self) -> dict[str, bool]:
        out = {}
        for p in self.providers:
            try:
                out[p.name] = await asyncio.wait_for(p.healthy(), 3)
            except Exception:  # noqa: BLE001
                out[p.name] = False
        return out
