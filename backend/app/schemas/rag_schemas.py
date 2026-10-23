
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# Enums
# ============================================================

class RAGStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VALIDATION_FAILED = "validation_failed"
    VALIDATION_ERROR = "validation_error"
    RETRIEVAL_FAILED = "retrieval_failed"
    ERROR = "error"


class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    HYBRID = "hybrid"


class CitationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"
    UNVERIFIED = "unverified"


# ============================================================
# API Request
# ============================================================

class RAGRequest(BaseModel):
    """Request body for the RAG question-answering endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    question: str = Field(
        ...,
        min_length=3,
        max_length=10000,
        description="Question to answer using retrieved documents.",
    )

    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional client-supplied correlation identifier.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of documents requested from retrieval.",
    )

    use_corrective_rag: bool = Field(
        default=True,
        description="Enable corrective retrieval and bounded query rewriting.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("question must not be empty.")

        return value.strip()

    @field_validator("request_id")
    @classmethod
    def validate_request_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(
                "request_id must not be blank when provided."
            )

        return value.strip() if value is not None else None


# ============================================================
# Citation and Evidence Schemas
# ============================================================

class CitationReference(BaseModel):
    """Reference to an evidence item used to support an answer."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Evidence identifier, for example E1.",
    )

    status: CitationStatus = Field(
        default=CitationStatus.UNVERIFIED,
    )

    claim: str | None = Field(
        default=None,
        max_length=5000,
        description="Answer claim supported by this evidence.",
    )


class RAGSource(BaseModel):
    """Retrieved source passage exposed in the API response."""

    model_config = ConfigDict(extra="ignore")

    evidence_id: str | None = None
    source_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None

    source: str = "unknown"
    page: str | int | None = None
    section: str | None = None

    content: str = Field(
        ...,
        description="Evidence passage used by the RAG pipeline.",
    )

    retrieval_score: float | None = None
    retrieval_method: str | None = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Source content must not be empty.")

        return value.strip()


# ============================================================
# Retrieval and Evaluation Details
# ============================================================

class RetrievalMetrics(BaseModel):
    """Summary of retrieval and corrective-retrieval execution."""

    model_config = ConfigDict(extra="forbid")

    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID

    retrieved_count: int = Field(default=0, ge=0)
    relevant_count: int = Field(default=0, ge=0)

    retry_count: int = Field(default=0, ge=0)
    query_rewritten: bool = False

    sufficient_evidence: bool = False

    @model_validator(mode="after")
    def validate_counts(self) -> "RetrievalMetrics":
        if self.relevant_count > self.retrieved_count:
            raise ValueError(
                "relevant_count cannot exceed retrieved_count."
            )

        return self


class GroundingAssessment(BaseModel):
    """Answer grounding and citation-validation result."""

    model_config = ConfigDict(extra="forbid")

    grounded: bool = False

    validation_status: str = Field(
        default="unverified",
        description=(
            "For example: passed, failed, validation_error, "
            "or unverified."
        ),
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
    )

    missing_citations: list[str] = Field(
        default_factory=list,
    )

    contradictory_evidence: list[str] = Field(
        default_factory=list,
    )

    validation_reason: str | None = None


class RAGExecutionMetrics(BaseModel):
    """Operational metrics for a RAG request."""

    model_config = ConfigDict(extra="forbid")

    execution_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
    )

    retrieval_time_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )

    generation_time_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )

    validation_time_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )

    input_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    output_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    total_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    total_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
    )


# ============================================================
# API Response
# ============================================================

class RAGResponse(BaseModel):
    """Standard response returned by the RAG endpoint."""

    model_config = ConfigDict(
        extra="ignore",
        use_enum_values=True,
    )

    request_id: str

    question: str

    answer: str

    status: RAGStatus = RAGStatus.COMPLETED

    grounded: bool = False

    sources: list[RAGSource] = Field(
        default_factory=list,
    )

    citations: list[CitationReference] = Field(
        default_factory=list,
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
    )

    retrieval: RetrievalMetrics = Field(
        default_factory=RetrievalMetrics,
    )

    grounding: GroundingAssessment = Field(
        default_factory=GroundingAssessment,
    )

    metrics: RAGExecutionMetrics = Field(
        default_factory=RAGExecutionMetrics,
    )

    error: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @model_validator(mode="after")
    def validate_grounding_consistency(self) -> "RAGResponse":
        if self.grounded != self.grounding.grounded:
            # Keep the top-level API field consistent with the
            # detailed grounding assessment.
            self.grounding.grounded = self.grounded

        if self.status == RAGStatus.INSUFFICIENT_EVIDENCE:
            self.grounded = False
            self.grounding.grounded = False

        return self


# ============================================================
# Health and Error Schemas
# ============================================================

class RAGHealthResponse(BaseModel):
    """Health-check response."""

    status: str = "healthy"
    service: str = "corrective-adaptive-rag"
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    dependencies: dict[str, str] = Field(
        default_factory=dict,
    )


class RAGErrorResponse(BaseModel):
    """Consistent error response."""

    request_id: str | None = None
    error_code: str
    message: str
    details: dict[str, Any] = Field(
        default_factory=dict,
    )

class IngestRequest(BaseModel):
    file_paths: list[str] = Field(
        default_factory=list,
        description="Paths relative to DOCUMENTS_DIR",
    )
    rebuild: bool = False


class IngestResponse(BaseModel):
    status: str
    files_processed: int
    pages_loaded: int
    chunks_created: int
    collection: str
    details: list[dict[str, Any]] = Field(default_factory=list)