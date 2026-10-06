"""Resolution generation: grounded LLM generation + deterministic extractive fallback."""
from __future__ import annotations

import json
import re

from app.observability import tracing
from app.core.errors import ResolveIQError
from app.models.schemas import Classification, Resolution, RetrievedItem, Step
from app.rag.prompts import PromptBuild
from app.services.llm.base import LLMResult, ResilientLLM


class GenerationError(ResolveIQError):
    status_code, code = 502, "generation_failed"


def parse_resolution(text: str) -> Resolution:
    """Tolerant JSON extraction (strips <think> blocks / code fences); strict on structure."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise GenerationError("LLM returned no JSON object")
    try:
        data = json.loads(m.group(0))
        steps = []
        for st in data.get("steps") or []:
            if isinstance(st, str):
                steps.append(Step(text=st, citations=[]))
            else:
                cites = st.get("citations") or []
                steps.append(Step(text=str(st["text"]).strip(), citations=[str(c) for c in cites]))
        return Resolution(
            issue_summary=str(data.get("issue_summary") or "").strip(),
            steps=[s for s in steps if s.text],
            escalate=bool(data.get("escalate", False)),
            escalation_reason=data.get("escalation_reason") or None,
            uncertainty=data.get("uncertainty") or None,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:  # AttributeError: a step that is a number, not an object
        raise GenerationError(f"LLM JSON did not match the expected schema: {exc}") from exc


async def generate_with_llm(llm: ResilientLLM, build: PromptBuild, *, temperature: float = 0.1, seed: int | None = None) -> tuple[Resolution, LLMResult]:
    res = await llm.generate(build.system, build.user, json_mode=True, temperature=temperature, seed=seed)
    return parse_resolution(res.text), res


def _overlap(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


@tracing.traced("generate.extractive", attrs=lambda cls, evidence: {"resolveiq.evidence_items": len(evidence)},
                result=lambda r: {"resolveiq.steps": len(r.steps)})
def generate_extractive(cls: Classification, evidence: list[RetrievedItem]) -> Resolution:
    """No-LLM fallback: reuse the best ticket's recorded resolution steps and, if they add something,
    the best article's procedure. Every step is copied from (and cites) real evidence by construction."""
    steps: list[Step] = []
    tickets = [e for e in evidence if e.source_type == "ticket"]
    articles = [e for e in evidence if e.source_type == "article"]
    if tickets:
        for s in tickets[0].steps[:5]:
            steps.append(Step(text=s, citations=[tickets[0].source_id]))
    if articles:
        for s in articles[0].steps:
            if len(steps) >= 7:
                break
            if all(_overlap(s, x.text) < 0.6 for x in steps):
                steps.append(Step(text=s, citations=[articles[0].source_id]))
    head = tickets[0].resolution_summary if tickets else (articles[0].title if articles else "")
    return Resolution(
        issue_summary=f"{cls.product.replace('_', ' ')} / {cls.intent.replace('_', ' ')}: matches prior resolved case - {head}".strip(" -"),
        steps=steps, escalate=not steps,
        escalation_reason=None if steps else "No usable evidence to build a resolution.",
        uncertainty="Generated without an LLM (evidence-only mode); steps are reused verbatim from past resolutions.")
