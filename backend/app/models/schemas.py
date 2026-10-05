"""Pydantic contracts shared by the API, retrieval, RAG and evaluation layers."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SourceType = Literal["ticket", "article"]
Strategy = Literal["lexical", "bm25", "dense", "hybrid", "hybrid_reranked"]


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


class FeedbackIn(BaseModel):
    request_id: str
    rating: Literal["helpful", "not_helpful"]
    comment: str | None = Field(None, max_length=2000)
    corrected_intent: str | None = None


class EvaluateRequest(BaseModel):
    suites: list[Literal["classification", "retrieval", "robustness", "rag", "e2e", "evolving", "discovery"]] = Field(
        default_factory=lambda: ["classification", "retrieval"]
    )
    max_queries: int | None = Field(None, ge=1, le=1000)
