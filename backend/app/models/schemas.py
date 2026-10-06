"""Pydantic contracts shared by the API, retrieval, RAG and evaluation layers."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SourceType = Literal["ticket", "article"]
Strategy = Literal["lexical", "bm25", "dense", "hybrid", "hybrid_reranked", "adaptive"]


# ----------------------------------------------------------------- classification
class Confidence(BaseModel):
    intent: float = 0.0
    product: float = 0.0
    severity: float = 0.0
    sentiment: float = 0.0


class Classification(BaseModel):
    intent: str
    product: str
    severity: str
    sentiment: str
    confidence: Confidence
    strategy: str = "ensemble"
    taxonomy_version: int | None = None


# ----------------------------------------------------------------- retrieval
class RetrievedItem(BaseModel):
    source_type: SourceType
    source_id: str
    title: str                      # ticket: resolution summary headline; article: title
    text: str                       # complaint text (ticket) or article content
    score: float                    # method-specific relevance score (see retrieval_method)
    rank: int
    retrieval_method: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    steps: list[str] = Field(default_factory=list)
    resolution_summary: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)  # per-stage scores for audit


class SearchFilters(BaseModel):
    intent: str | None = None
    product: str | None = None


# ----------------------------------------------------------------- ingestion
class TicketIn(BaseModel):
    ticket_id: str | None = Field(None, max_length=64)
    complaint_text: str = Field(..., min_length=5, max_length=8000)
    intent: str | None = None          # if omitted, the classifier fills it in
    product: str | None = None
    severity: str | None = None
    sentiment: str | None = None
    resolution_steps: list[str] = Field(..., min_length=1)
    resolution_summary: str = Field(..., min_length=3, max_length=2000)
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArticleIn(BaseModel):
    article_id: str | None = Field(None, max_length=64)
    title: str = Field(..., min_length=3, max_length=300)
    content: str = Field(..., min_length=20, max_length=20000)
    steps: list[str] = Field(default_factory=list)
    category: str
    product: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    source_type: SourceType
    source_id: str
    created: bool
    pii_redactions: dict[str, int] = Field(default_factory=dict)
    corpus_version: int
    embedding_ms: float
    warnings: list[str] = Field(default_factory=list)   # e.g. instruction-like text found in the document (it is neutralised before any model sees it)


class BatchIngestRequest(BaseModel):
    tickets: list[TicketIn] = Field(..., min_length=1, max_length=5000)


class TaxonomyLabelIn(BaseModel):
    dimension: Literal["intent", "product", "severity", "sentiment"] = "intent"
    label_id: str = Field(..., pattern=r"^[a-z][a-z0-9_]{1,63}$")
    description: str = Field(..., min_length=10)
    keywords: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    team: str | None = None


# ----------------------------------------------------------------- resolve
class ResolveRequest(BaseModel):
    complaint: str = Field(..., min_length=5, max_length=4000)
    strategy: Strategy | None = None
    use_metadata_filters: bool = False
    deterministic: bool = False   # temperature 0 and a fixed seed for the LLM, for reproducible evaluation and replay

    @field_validator("complaint")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("complaint must not be blank")
        return v


class Step(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    grounded: bool | None = None
    grounding_score: float | None = None
    support: dict[str, float] = Field(default_factory=dict)   # cited source id -> how well that source's text supports this step (cosine)


class Citation(BaseModel):
    source_type: SourceType
    source_id: str
    title: str
    score: float
    cited_in_steps: list[int] = Field(default_factory=list)


class ValidationReport(BaseModel):
    valid: bool
    invalid_citations: list[str] = Field(default_factory=list)   # ids the LLM cited that were not retrieved
    uncited_steps: list[int] = Field(default_factory=list)
    unsupported_steps: list[int] = Field(default_factory=list)   # cited but not backed by the cited text
    grounded_ratio: float = 0.0
    citation_coverage: float = 0.0
    citations_emitted: int = 0                                   # raw ids the generator produced (before sanitising)
    invented_details: list[str] = Field(default_factory=list)    # links, e-mail addresses, long numbers or amounts in a step that the cited evidence does not contain
    warnings: list[str] = Field(default_factory=list)


class EvidenceAssessment(BaseModel):
    confidence: float
    top_ticket_similarity: float
    top_article_similarity: float
    rerank_signal: float | None
    consensus: float
    sufficient: bool
    reason: str


class Resolution(BaseModel):
    issue_summary: str
    steps: list[Step]
    escalate: bool
    escalation_reason: str | None = None
    uncertainty: str | None = None


class StageRecord(BaseModel):
    """One stage of the pipeline as it ran: enough to debug or replay a case, never raw customer text."""
    name: str
    status: Literal["ok", "skipped", "degraded", "error"] = "ok"
    latency_ms: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class Provenance(BaseModel):
    """Everything needed to answer: which model, prompt, taxonomy, corpus and retrieval configuration produced this resolution?"""
    pipeline_version: str
    generator: str                                   # provider:model, "extractive" or "none"
    model: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    taxonomy_version: int | None = None
    corpus_version: int | None = None
    embedding_model: str | None = None
    reranker: str | None = None
    retrieval: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)   # temperature, seed, token counts, what went into the prompt


class LineageSource(BaseModel):
    id: str
    type: SourceType
    title: str
    excerpt: str
    rank: int
    method: str
    score: float
    stage_scores: dict[str, float] = Field(default_factory=dict)
    why: list[str] = Field(default_factory=list)      # plain-language reasons this source was retrieved
    intent: str | None = None
    matches_intent: bool = False
    selected: bool = False                            # chosen as evidence for generation
    in_prompt: bool = False                           # actually shown to the model
    cited_by_steps: list[int] = Field(default_factory=list)


class LineageStep(BaseModel):
    index: int
    text: str
    citations: list[str]
    grounded: bool | None = None
    grounding_score: float | None = None
    support: dict[str, float] = Field(default_factory=dict)


class LineageEdge(BaseModel):
    kind: Literal["extracted", "retrieved", "selected", "cited"]
    source: str
    target: str
    weight: float | None = None


class Lineage(BaseModel):
    """complaint -> attributes -> retrieved sources -> evidence -> steps -> citations -> validation, as data."""
    complaint: dict[str, Any]
    attributes: list[dict[str, Any]]
    sources: list[LineageSource]
    steps: list[LineageStep]
    edges: list[LineageEdge]
    signals: dict[str, Any]                           # safe, auditable quality signals (no model reasoning)
    checks: dict[str, bool]                           # invariants such as "every citation maps to a retrieved source"


class ResolveResponse(BaseModel):
    request_id: str
    trace_id: str
    status: Literal["resolved", "abstained", "degraded", "unreliable"]
    classification: Classification
    tickets: list[RetrievedItem]
    articles: list[RetrievedItem]
    resolution: Resolution
    citations: list[Citation]
    validation: ValidationReport
    evidence: EvidenceAssessment
    confidence: float
    generator: str                  # provider/model or "extractive"
    pii_redactions: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    latency_ms: dict[str, float]
    cached: bool = False
    provenance: Provenance | None = None
    trace: list[StageRecord] = Field(default_factory=list)
    lineage: Lineage | None = None


FEEDBACK_REASONS = ("wrong_intent", "steps_incorrect", "steps_missing", "irrelevant_source", "outdated_source", "unsafe", "too_vague", "other")


class FeedbackIn(BaseModel):
    request_id: str
    rating: Literal["helpful", "not_helpful"]
    comment: str | None = Field(None, max_length=2000)
    corrected_intent: str | None = None
    reasons: list[Literal["wrong_intent", "steps_incorrect", "steps_missing", "irrelevant_source", "outdated_source", "unsafe", "too_vague", "other"]] = Field(
        default_factory=list, max_length=8)
    rejected_sources: list[str] = Field(default_factory=list, max_length=20)   # ids of retrieved sources the agent judged irrelevant or wrong
    edited_steps: list[str] | None = Field(None, max_length=12)                 # the resolution as the agent would send it


class EvaluateRequest(BaseModel):
    suites: list[Literal["classification", "retrieval", "robustness", "rag", "e2e", "evolving", "discovery", "adaptive", "clustering"]] = Field(
        default_factory=lambda: ["classification", "retrieval"]
    )
    max_queries: int | None = Field(None, ge=1, le=1000)
