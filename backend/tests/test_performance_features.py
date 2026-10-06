"""End-to-end behaviour of the load-test driven changes: the LLM concurrency limit under concurrent HTTP traffic, and the evidence-only answer cache."""
from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.services.llm.base import LLMResult, ResilientLLM

COMPLAINT = "My broadband drops every evening around eight and restarting the router twice did not help."


class SlowLLM:
    """An LLM that takes `delay` s per call and records how many calls overlapped. Its answer is a valid 'no actionable steps' resolution."""

    name, model = "slow", "m"

    def __init__(self, delay: float):
        self.delay, self.calls, self.in_flight, self.max_in_flight = delay, 0, 0, 0

    async def generate(self, system, user, **kw):
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return LLMResult(text='{"issue_summary": "s", "steps": [], "escalate": true, "escalation_reason": "test"}', provider=self.name, model="m")
        finally:
            self.in_flight -= 1

    async def healthy(self):
        return True


def _app(test_settings, test_database, llm=None, **overrides):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.scripts_seed_bridge import seed_corpus

    settings = test_settings.model_copy(update={"cache_namespace": f"perf-{uuid.uuid4().hex[:8]}", **overrides})
    application = create_app(settings, llm=llm(settings) if llm else None)
    client = TestClient(application)
    client.__enter__()
    client.portal.call(seed_corpus, application.state.services)
    return client


@pytest.fixture(scope="module")
def busy_llm(test_settings, test_database):
    provider = SlowLLM(delay=1.0)
    client = _app(test_settings, test_database, llm=lambda s: ResilientLLM([provider], s), llm_max_concurrency=1, llm_queue_wait_seconds=0.0,
                  classifier_llm_fallback=True)
    yield client, provider
    client.__exit__(None, None, None)


def test_concurrent_requests_never_exceed_the_llm_limit_and_the_overflow_degrades(busy_llm):
    client, provider = busy_llm

    def one(i):
        r = client.post("/api/v1/resolve", json={"complaint": f"{COMPLAINT} variant {i} {uuid.uuid4().hex}"})
        return r.status_code, r.json()

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(one, range(4)))

    assert [code for code, _ in results] == [200] * 4, "no request fails because the LLM is busy"
    assert provider.max_in_flight == 1, "the limit holds under real concurrent traffic"
    served = [b for _, b in results if b["generator"].startswith("slow")]
    shed = [b for _, b in results if b["status"] == "degraded"]
    assert len(served) >= 1 and len(shed) >= 1 and len(served) + len(shed) == 4
    assert all(any("busy" in w for w in b["warnings"]) for b in shed), "a degraded answer says why"
    assert all(b["resolution"]["steps"] for b in shed), "the evidence-only fallback still gives cited steps"
    llm = client.app.state.services.llm
    assert llm.breakers["slow"].failures == 0 and not llm.breakers["slow"].is_open, "overload must not trip the circuit breaker"


def test_evidence_only_answers_are_cached_only_when_enabled(test_settings, test_database):
    client = _app(test_settings, test_database, llm_providers="")  # no LLM at all: every answer is evidence-only
    try:
        settings = client.app.state.services.settings
        for ttl, expect_cached in ((0, False), (30, True)):
            settings.cache_degraded_ttl_seconds = ttl
            text = f"{COMPLAINT} ttl {ttl} {uuid.uuid4().hex}"
            first = client.post("/api/v1/resolve", json={"complaint": text}).json()
            second = client.post("/api/v1/resolve", json={"complaint": text}).json()
            assert first["status"] == second["status"] == "degraded" and first["cached"] is False
            assert second["cached"] is expect_cached, f"cache_degraded_ttl_seconds={ttl}"
            if expect_cached:
                assert second["resolution"] == first["resolution"] and second["latency_ms"]["total"] < first["latency_ms"]["total"]
    finally:
        client.__exit__(None, None, None)
