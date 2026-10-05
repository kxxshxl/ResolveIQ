"""Post-generation citation + grounding validation.

Guarantees: (1) the response never carries a citation id that was not actually retrieved,
(2) every step is flagged when it has no citation or its text is not supported by the cited evidence.
"""
from __future__ import annotations

import asyncio
import re

import numpy as np

from app.core.config import Settings
from app.models.schemas import Citation, Resolution, RetrievedItem, Step, ValidationReport
from app.observability.metrics import CITATION_FAILURES
from app.services.embedding import EmbeddingService

_ID = re.compile(r"[\[\]\s]")
_TOK = re.compile(r"[a-z0-9]+")


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
        mat = await asyncio.to_thread(embedder.encode_sync, [s.text for s in cleaned] + texts)
        step_vecs, unit_vecs = mat[: len(cleaned)], mat[len(cleaned):]

    unsupported, uncited = [], []
    for i, st in enumerate(cleaned):
        if not st.citations:
            uncited.append(i)
            st.grounded, st.grounding_score = False, 0.0
            continue
        idx = [j for j, (si, _) in enumerate(owner) if si == i]
        cos = float(np.max(unit_vecs[idx] @ step_vecs[i])) if idx else 0.0
        units = [texts[j] for j in idx]
        cont = containment(st.text, units)
        st.grounding_score = round(max(cos, 0.0), 3)
        st.grounded = bool(cos >= settings.grounding_threshold or cont >= 0.75)
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
                              citation_coverage=round(coverage, 3), warnings=warnings,
                              citations_emitted=sum(len(st.citations) for st in resolution.steps))
    out = resolution.model_copy(update={"steps": cleaned})
    return out, citations, report
