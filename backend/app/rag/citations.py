"""Post-generation citation + grounding validation.

Guarantees: (1) the response never carries a citation id that was not actually retrieved,
(2) every step is flagged when it has no citation or its text is not supported by the cited evidence.
"""
from __future__ import annotations

import re

import numpy as np

from app.observability import tracing
from app.core.config import Settings
from app.models.schemas import Citation, Resolution, RetrievedItem, Step, ValidationReport
from app.observability.metrics import CITATION_FAILURES
from app.services.embedding import EmbeddingService

_ID = re.compile(r"[\[\]\s]")
_TOK = re.compile(r"[a-z0-9]+")
# Specifics an answer must never make up: where to go (links, e-mail), who to call or pay (long numbers), how much (amounts). Each one must appear in the evidence.
_SPECIFIC = re.compile(r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.-]+|[£$€]\s?\d[\d,.]*|\b\d[\d\s().-]{6,}\d\b")


def invented_specifics(step: str, units: list[str]) -> list[str]:
    """Links, addresses, long numbers and amounts that appear in `step` but nowhere in the cited evidence (compared with spaces and punctuation ignored)."""
    squash = lambda s: re.sub(r"[^a-z0-9£$€@.]", "", s.lower()).rstrip(".")  # noqa: E731
    haystack = squash(" ".join(units))
    out = []
    for m in _SPECIFIC.finditer(step):
        token = m.group(0).strip(" .,;:)")
        if squash(token) and squash(token) not in haystack:
            out.append(token[:60])
    return out


def normalize_id(raw: str, valid: dict[str, RetrievedItem]) -> str:
    c = _ID.sub("", str(raw))
    if c in valid:
        return c
    for v in valid:  # case-insensitive match only; we never "repair" an id to a different source
        if v.lower() == c.lower():
            return v
    return c


def evidence_units(item: RetrievedItem) -> list[str]:
    """The pieces of an evidence item a step could legitimately be derived from."""
    units = list(item.steps)
    if item.resolution_summary:
        units.append(item.resolution_summary)
    if item.source_type == "article":
        units.append(item.title)
        units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", item.text) if len(s.strip()) > 20)
    else:
        units.append(item.text)
    return [u for u in units if u]


def containment(step: str, units: list[str]) -> float:
    st = set(_TOK.findall(step.lower()))
    if not st:
        return 0.0
    ev = set(_TOK.findall(" ".join(units).lower()))
    return len(st & ev) / len(st)


@tracing.traced(
    "citations.validate",
    attrs=lambda resolution, evidence, embedder, settings: {"resolveiq.steps": len(resolution.steps), "resolveiq.evidence_items": len(evidence)},
    result=lambda r: {"resolveiq.validation.valid": r[2].valid, "resolveiq.validation.citation_coverage": r[2].citation_coverage,
                      "resolveiq.validation.grounded_ratio": r[2].grounded_ratio, "resolveiq.validation.invalid_citations": len(r[2].invalid_citations),
                      "resolveiq.validation.uncited_steps": len(r[2].uncited_steps),
                      "resolveiq.validation.unsupported_steps": len(r[2].unsupported_steps), "resolveiq.citations": len(r[1])})
async def validate_resolution(resolution: Resolution, evidence: list[RetrievedItem], embedder: EmbeddingService,
                              settings: Settings) -> tuple[Resolution, list[Citation], ValidationReport]:
    valid = {e.source_id: e for e in evidence}
    invalid: list[str] = []
    cleaned: list[Step] = []
    for st in resolution.steps:
        ids: list[str] = []
        for raw in st.citations:
            cid = normalize_id(raw, valid)
            if cid in valid:
                if cid not in ids:
                    ids.append(cid)
            else:
                invalid.append(str(raw))
        cleaned.append(Step(text=st.text, citations=ids))

    # grounding: embedding similarity of each step against the units of the sources it cites
    texts, owner = [], []
    for i, st in enumerate(cleaned):
        for cid in st.citations:
            for u in evidence_units(valid[cid]):
                texts.append(u)
                owner.append((i, cid))
    step_vecs = unit_vecs = None
    if cleaned and texts:
        mat = await embedder.embed_cached([s.text for s in cleaned] + texts)  # de-duplicated, and evidence sentences are cached across requests
        step_vecs, unit_vecs = mat[: len(cleaned)], mat[len(cleaned):]

    unsupported, uncited, invented = [], [], []
    for i, st in enumerate(cleaned):
        if not st.citations:
            uncited.append(i)
            st.grounded, st.grounding_score = False, 0.0
            continue
        idx = [j for j, (si, _) in enumerate(owner) if si == i]
        cos = float(np.max(unit_vecs[idx] @ step_vecs[i])) if idx else 0.0
        st.support = {cid: round(max(0.0, float(np.max(unit_vecs[[j for j in idx if owner[j][1] == cid]] @ step_vecs[i]))), 3) for cid in st.citations
                      if any(owner[j][1] == cid for j in idx)}   # per-citation evidence: how well each cited source backs this step
        units = [texts[j] for j in idx]
        cont = containment(st.text, units)
        st.grounding_score = round(max(cos, 0.0), 3)
        st.grounded = bool(cos >= settings.grounding_threshold or cont >= 0.75)
        fake = invented_specifics(st.text, units)
        if fake:   # a step that sounds right but sends the agent to a link, number or amount the evidence never mentioned is not grounded, however similar it reads
            st.grounded = False
            invented.extend(fake)
        if not st.grounded:
            unsupported.append(i)

    n = len(cleaned)
    grounded_ratio = (sum(1 for s in cleaned if s.grounded) / n) if n else 0.0
    coverage = ((n - len(uncited)) / n) if n else 0.0
    warnings: list[str] = []
    if invalid:
        warnings.append(f"LLM cited ids that were not retrieved and they were removed: {sorted(set(invalid))}")
        CITATION_FAILURES.labels("invalid_id").inc(len(invalid))
    if uncited:
        warnings.append(f"{len(uncited)} step(s) have no valid citation")
        CITATION_FAILURES.labels("uncited_step").inc(len(uncited))
    if unsupported:
        warnings.append(f"{len(unsupported)} step(s) are not supported by the text of the cited evidence")
        CITATION_FAILURES.labels("unsupported_step").inc(len(unsupported))
    if invented:
        warnings.append(f"a step contains a link, address, number or amount that the cited evidence does not contain: {sorted(set(invented))[:3]}")
        CITATION_FAILURES.labels("invented_detail").inc(len(invented))
    ok = (not invalid) and (not uncited) and (n == 0 or grounded_ratio >= settings.min_grounded_ratio)

    by_src: dict[str, list[int]] = {}
    for i, st in enumerate(cleaned):
        for cid in st.citations:
            by_src.setdefault(cid, []).append(i + 1)
    citations = [
        Citation(source_type=valid[cid].source_type, source_id=cid, title=valid[cid].title, score=valid[cid].score, cited_in_steps=steps)
        for cid, steps in by_src.items()
    ]
    report = ValidationReport(valid=ok, invalid_citations=sorted(set(invalid)), uncited_steps=[i + 1 for i in uncited],
                              unsupported_steps=[i + 1 for i in unsupported], grounded_ratio=round(grounded_ratio, 3),
                              citation_coverage=round(coverage, 3), warnings=warnings, invented_details=sorted(set(invented)),
                              citations_emitted=sum(len(st.citations) for st in resolution.steps))
    out = resolution.model_copy(update={"steps": cleaned})
    return out, citations, report
