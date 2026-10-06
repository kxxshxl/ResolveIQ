"""LLM provider abstraction: pluggable providers behind retry + circuit breaker + ordered fallback."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol

from app.core.config import Settings
from app.core.errors import LLMOverloaded, LLMUnavailable
from app.observability import tracing
from app.observability.metrics import (LLM_CIRCUIT_OPEN, LLM_FAILURES, LLM_IN_FLIGHT, LLM_LATENCY, LLM_QUEUE_WAIT, LLM_SHED,
                                       LLM_TOKENS)

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


MIN_ATTEMPT_SECONDS = 1.0  # an attempt with less time than this left in the budget is not worth starting


class CircuitBreaker:
    """Opens after N consecutive failures; half-opens after a cooldown (one trial call)."""

    def __init__(self, threshold: int, cooldown_s: float, name: str):
        self.threshold, self.cooldown_s, self.name = threshold, cooldown_s, name
        self.failures = 0
        self.opened_at: float | None = None
        self._probing = False

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.cooldown_s:
            return False  # half-open: allow a trial
        return True

    def allow(self) -> bool:
        """May a call go through? Closed: yes. Open: no. Half-open (cooldown elapsed): exactly ONE caller probes and the rest are refused
        until it reports, so a recovering or still-dead provider is not hit by every in-flight request at once."""
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at < self.cooldown_s or self._probing:
            return False
        self._probing = True
        return True

    def release(self) -> None:
        """Give up the half-open probe slot (idempotent), e.g. when the probing request was cancelled before it could report."""
        self._probing = False

    def record_success(self) -> None:
        self.failures, self.opened_at, self._probing = 0, None, False
        LLM_CIRCUIT_OPEN.labels(self.name).set(0)

    def record_failure(self, decisive: bool = False) -> None:
        """`decisive` (a timeout) trips the breaker at once: a call that burned the whole time budget on a model that normally answers in seconds
        is strong evidence it is wedged, and with a concurrency limit of 1 waiting for `threshold` of them would take threshold x budget."""
        self.failures = max(self.failures + 1, self.threshold) if decisive else self.failures + 1
        self._probing = False
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
        # concurrency limit per provider: more generations in flight than the backend can run at once only lengthens all of them
        self.slots = {p.name: asyncio.Semaphore(settings.llm_max_concurrency) for p in providers} if settings.llm_max_concurrency > 0 else {}

    async def _take_slot(self, name: str, priority: str, deadline: float) -> bool:
        """Wait (for `normal` priority, at most llm_queue_wait_seconds and never past the time budget) for a generation slot.
        `low` priority work, such as the classification fallback, never queues: if the model is busy it is simply skipped."""
        slot, t0 = self.slots[name], time.perf_counter()
        if not slot.locked():
            await slot.acquire()
        else:
            wait = 0.0 if priority == "low" else min(self.s.llm_queue_wait_seconds, deadline - time.monotonic() - MIN_ATTEMPT_SECONDS)
            got = False
            if wait > 0:
                try:
                    await asyncio.wait_for(slot.acquire(), wait)
                    got = True
                except asyncio.TimeoutError:
                    pass
            if not got:
                LLM_SHED.labels(name, priority).inc()
                tracing.event("llm.shed", gen_ai__system=name, resolveiq__llm__priority=priority)
                return False
        LLM_QUEUE_WAIT.labels(name).observe(time.perf_counter() - t0)
        LLM_IN_FLIGHT.labels(name).inc()
        return True

    def _release_slot(self, name: str) -> None:
        slot = self.slots.get(name)
        if slot is not None:
            slot.release()
            LLM_IN_FLIGHT.labels(name).dec()

    @property
    def available(self) -> bool:
        return bool(self.providers)

    @tracing.traced(
        "llm.generate",
        attrs=lambda self, system, user, *, json_mode=True, max_tokens=None, temperature=0.1, priority="normal": {
            "gen_ai.operation.name": "chat", "gen_ai.request.temperature": temperature,
            "gen_ai.request.max_tokens": max_tokens or self.s.llm_max_tokens, "resolveiq.llm.json_mode": json_mode,
            "resolveiq.llm.provider_chain": [p.name for p in self.providers], "resolveiq.llm.prompt_chars": len(system) + len(user),
            "resolveiq.llm.priority": priority},
        result=lambda r: {"gen_ai.system": r.provider, "gen_ai.response.model": r.model, "gen_ai.usage.input_tokens": r.prompt_tokens,
                          "gen_ai.usage.output_tokens": r.completion_tokens, "resolveiq.llm.response_chars": len(r.text)})
    async def generate(self, system: str, user: str, *, json_mode: bool = True, max_tokens: int | None = None,
                       temperature: float = 0.1, priority: str = "normal") -> LLMResult:
        errors: list[str] = []
        shed = attempted = False
        deadline = time.monotonic() + self.s.llm_total_budget_seconds  # one budget for every attempt of every provider
        for p in self.providers:
            br = self.breakers[p.name]
            probe = br.opened_at is not None  # a call that gets through an opened breaker is the half-open probe (no await before allow(): atomic)
            if not br.allow():  # open, or half-open with the probe already in flight: there is nothing to wait for, degrade at once
                errors.append(f"{p.name}: circuit open")
                tracing.event("llm.circuit_open", gen_ai__system=p.name)
                continue
            if p.name in self.slots and not await self._take_slot(p.name, priority, deadline):
                if probe:
                    br.release()  # we never made the probe call, so let another request try
                errors.append(f"{p.name}: all {self.s.llm_max_concurrency} generation slot(s) busy")
                shed = True
                continue
            tried = timed_out = False
            try:
                for attempt in range(self.s.llm_max_retries + 1):
                    remaining = deadline - time.monotonic()
                    if remaining < MIN_ATTEMPT_SECONDS:
                        errors.append(f"{p.name}: time budget of {self.s.llm_total_budget_seconds:.0f}s used up")
                        break
                    tried = attempted = True
                    t0 = time.perf_counter()
                    try:
                        with tracing.span("llm.call", {"gen_ai.system": p.name, "gen_ai.request.model": p.model, "resolveiq.llm.attempt": attempt}) as sp:
                            res = await p.generate(system, user, json_mode=json_mode, max_tokens=max_tokens or self.s.llm_max_tokens,
                                                   temperature=temperature, timeout=min(self.s.llm_timeout_seconds, remaining))
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
                        timed_out = reason == "timeout"
                        LLM_FAILURES.labels(p.name, reason).inc()
                        errors.append(f"{p.name}[{attempt}]: {type(exc).__name__}: {exc}")
                        log.warning("llm call failed", extra={"provider": p.name, "attempt": attempt, "reason": reason, "error": str(exc)[:200]})
                        backoff = 0.5 * (2 ** attempt)
                        if attempt < self.s.llm_max_retries:
                            if deadline - time.monotonic() > backoff + MIN_ATTEMPT_SECONDS:
                                await asyncio.sleep(backoff)
                            else:
                                errors.append(f"{p.name}: no time left to retry")
                                break
                if tried:
                    br.record_failure(decisive=timed_out)
            finally:
                if probe:
                    br.release()  # a half-open probe that was cancelled mid-flight must not wedge the breaker open (only its owner may release it)
                self._release_slot(p.name)
            if deadline - time.monotonic() < MIN_ATTEMPT_SECONDS:
                break
        if shed and not attempted:  # nothing was tried because the model was busy: overload, not an outage
            raise LLMOverloaded("LLM busy: " + "; ".join(errors[-4:]))
        raise LLMUnavailable("all LLM providers failed: " + "; ".join(errors[-4:]))

    async def health(self) -> dict[str, bool]:
        out = {}
        for p in self.providers:
            try:
                out[p.name] = await asyncio.wait_for(p.healthy(), 3)
            except Exception:  # noqa: BLE001
                out[p.name] = False
        return out
