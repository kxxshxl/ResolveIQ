"""Concrete providers: Ollama (default, local), OpenAI-compatible (fallback), Mock (tests/offline)."""
from __future__ import annotations

import json
import re
import time

import httpx

from app.observability import tracing
from app.core.config import Settings
from app.services.llm.base import LLMResult, ResilientLLM


def _timeout(total: float, connect: float) -> httpx.Timeout:
    """`total` bounds reading the answer (a model may legitimately think for many seconds); `connect` bounds reaching the server. A host that
    silently drops packets (a network policy, a partition, a dead node) would otherwise cost the whole generation budget on every probe."""
    return httpx.Timeout(total, connect=min(total, connect))


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, connect_timeout: float = 5.0, num_ctx: int = 4096):
        self.base_url, self.model, self.connect_timeout, self.num_ctx = base_url.rstrip("/"), model, connect_timeout, num_ctx
        self._client = httpx.AsyncClient()
        tracing.instrument_http_client(self._client)

    async def generate(self, system, user, *, json_mode=True, max_tokens=700, temperature=0.1, timeout=45.0, seed=None) -> LLMResult:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.num_ctx, **({"seed": seed} if seed is not None else {})},
        }
        if json_mode:
            body["format"] = "json"
        t0 = time.perf_counter()
        r = await self._client.post(f"{self.base_url}/api/chat", json=body, timeout=_timeout(timeout, self.connect_timeout))
        r.raise_for_status()
        d = r.json()
        return LLMResult(
            text=d["message"]["content"], provider=f"{self.name}", model=self.model,
            prompt_tokens=d.get("prompt_eval_count"), completion_tokens=d.get("eval_count"),
            latency_s=time.perf_counter() - t0, temperature=temperature, seed=seed,
        )

    async def healthy(self) -> bool:
        r = await self._client.get(f"{self.base_url}/api/tags", timeout=3)
        return r.status_code == 200 and any(m["name"] == self.model for m in r.json().get("models", []))


class OpenAICompatProvider:
    """Any /v1/chat/completions server: OpenAI, vLLM, LM Studio, or Ollama's /v1 endpoint."""

    name = "openai_compat"

    def __init__(self, base_url: str, api_key: str, model: str, connect_timeout: float = 5.0):
        self.base_url, self.api_key, self.model, self.connect_timeout = base_url.rstrip("/"), api_key, model, connect_timeout
        self._client = httpx.AsyncClient()
        tracing.instrument_http_client(self._client)

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def generate(self, system, user, *, json_mode=True, max_tokens=700, temperature=0.1, timeout=45.0, seed=None) -> LLMResult:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens, "stream": False,
            **({"seed": seed} if seed is not None else {}),
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        r = await self._client.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers(), timeout=_timeout(timeout, self.connect_timeout))
        r.raise_for_status()
        d = r.json()
        text = re.sub(r"<think>.*?</think>", "", d["choices"][0]["message"]["content"] or "", flags=re.S).strip()
        u = d.get("usage") or {}
        return LLMResult(text=text, provider=self.name, model=self.model, prompt_tokens=u.get("prompt_tokens"),
                         completion_tokens=u.get("completion_tokens"), latency_s=time.perf_counter() - t0, temperature=temperature, seed=seed)

    async def healthy(self) -> bool:
        r = await self._client.get(f"{self.base_url}/models", headers=self._headers(), timeout=3)
        return r.status_code == 200


class MockProvider:
    """Deterministic offline provider. Reads the evidence blocks out of the prompt and echoes the
    first block's resolution steps with correct citations. Used by tests and `LLM_PROVIDERS=mock`."""

    name = "mock"
    model = "mock-extractive"

    async def generate(self, system, user, *, json_mode=True, max_tokens=700, temperature=0.1, timeout=45.0, seed=None) -> LLMResult:
        if "classify" in system.lower()[:200]:
            return LLMResult(text=json.dumps({"intent": "unknown", "product": "unknown", "severity": "medium", "sentiment": "neutral"}),
                             provider=self.name, model=self.model)
        blocks = re.split(r"\n(?=\[(?:TKT|KB)-[^\]]+\])", user)
        steps, first_id = [], None
        for b in blocks:
            m = re.match(r"\[((?:TKT|KB)-[^\]]+)\]", b)
            if not m:
                continue
            first_id = first_id or m.group(1)
            for line in re.findall(r"^\s*\d+\.\s+(.+)$", b, flags=re.M)[:4]:
                if len(steps) < 5:
                    steps.append({"text": line.strip(), "citations": [m.group(1)]})
            if steps:
                break
        out = {"issue_summary": "Customer reports a problem matching prior resolved cases.", "steps": steps,
               "escalate": not steps, "escalation_reason": None if steps else "No usable evidence.", "uncertainty": None}
        return LLMResult(text=json.dumps(out), provider=self.name, model=self.model)

    async def healthy(self) -> bool:
        return True


def build_llm(settings: Settings) -> ResilientLLM:
    providers = []
    for name in settings.provider_chain:
        if name == "ollama":
            providers.append(OllamaProvider(settings.ollama_base_url, settings.ollama_model, settings.llm_connect_timeout_seconds, settings.llm_num_ctx))
        elif name == "openai_compat" and settings.openai_compat_base_url and settings.openai_compat_model:
            providers.append(OpenAICompatProvider(settings.openai_compat_base_url, settings.openai_compat_api_key,
                                                  settings.openai_compat_model, settings.llm_connect_timeout_seconds))
        elif name == "mock":
            providers.append(MockProvider())
    return ResilientLLM(providers, settings)
