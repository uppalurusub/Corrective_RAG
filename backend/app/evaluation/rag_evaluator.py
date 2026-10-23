
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================
# Evaluation Schemas
# ============================================================

@dataclass
class RetrievalEvaluation:
    retrieved_count: int = 0
    relevant_count: int = 0
    precision_at_k: float = 0.0
    recall_at_k: float | None = None
    reciprocal_rank: float = 0.0
    hit_at_k: float = 0.0
    mean_average_precision: float | None = None


@dataclass
class GroundingEvaluation:
    citation_count: int = 0
    valid_citation_count: int = 0
    invalid_citations: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    citation_precision: float = 0.0
    citation_coverage: float = 0.0
    grounded: bool = False
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass
class AnswerEvaluation:
    answer_present: bool = False
    answer_relevance_score: float | None = None
    faithfulness_score: float | None = None
    correctness_score: float | None = None
    evaluation_method: str = "deterministic_only"
    evaluation_error: str | None = None


@dataclass
class CorrectiveRAGEvaluation:
    retry_count: int = 0
    query_rewritten: bool = False
    initial_retrieval_sufficient: bool | None = None
    final_retrieval_sufficient: bool | None = None
    corrective_retrieval_used: bool = False


@dataclass
class RAGEvaluationResult:
    request_id: str | None
    retrieval: RetrievalEvaluation
    grounding: GroundingEvaluation
    answer: AnswerEvaluation
    corrective_rag: CorrectiveRAGEvaluation
    evaluation_time_seconds: float
    overall_score: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Optional Structured LLM Judge
# ============================================================

class LLMJudgeResult(BaseModel):
    """Structured result returned by the semantic evaluator."""

    answer_relevance_score: float = Field(ge=0.0, le=1.0)
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    correctness_score: float = Field(ge=0.0, le=1.0)

    unsupported_claims: list[str] = Field(default_factory=list)
    explanation: str = ""


# ============================================================
# RAG Evaluator
# ============================================================

class RAGEvaluator:
    """
    Evaluate RAG retrieval, citations, answer grounding, and
    corrective-retrieval behavior.

    Ground-truth relevance labels and expected answers are optional.
    Without them, recall and correctness cannot be measured reliably.

    The evaluator can run deterministic checks only or use an optional
    LLM judge for semantic answer assessment.
    """

    CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        enable_llm_judge: bool = False,
        relevance_threshold: float = 0.70,
        grounding_threshold: float = 0.80,
        max_context_chars: int = 30000,
    ) -> None:
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError(
                "relevance_threshold must be between 0 and 1."
            )

        if not 0.0 <= grounding_threshold <= 1.0:
            raise ValueError(
                "grounding_threshold must be between 0 and 1."
            )

        if max_context_chars < 1:
            raise ValueError(
                "max_context_chars must be positive."
            )

        if enable_llm_judge and llm is None:
            raise ValueError(
                "An LLM must be provided when enable_llm_judge=True."
            )

        self.llm = llm
        self.enable_llm_judge = enable_llm_judge
        self.relevance_threshold = relevance_threshold
        self.grounding_threshold = grounding_threshold
        self.max_context_chars = max_context_chars

        self._judge_chain = None

        if self.enable_llm_judge and self.llm is not None:
            self._judge_chain = (
                ChatPromptTemplate.from_messages([
                    (
                        "system",
                        """
You are an evaluator for retrieval-augmented generation (RAG).

Assess the answer only against the supplied evidence and question.

Score each field from 0.0 to 1.0:
- answer_relevance_score: whether the answer addresses the question.
- faithfulness_score: whether factual claims are supported by evidence.
- correctness_score: whether the answer is accurate based on the evidence.

Rules:
- Do not reward unsupported claims.
- Missing evidence is not proof that a claim is false.
- Do not use outside knowledge to fill evidence gaps.
- Identify concrete unsupported factual claims.
- A correct abstention can receive a high correctness score when
  the evidence is insufficient.
- Return only the requested structured result.
                        """.strip(),
                    ),
                    (
                        "human",
                        """
Question:
{question}

Answer:
{answer}

Evidence:
{context}
                        """.strip(),
                    ),
                ])
                | self.llm.with_structured_output(LLMJudgeResult)
            )

    # --------------------------------------------------------
    # Public evaluation API
    # --------------------------------------------------------

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        documents: Sequence[Document],
        request_id: str | None = None,
        expected_answer: str | None = None,
        relevant_document_ids: set[str] | None = None,
        expected_evidence_ids: set[str] | None = None,
        initial_retrieval_sufficient: bool | None = None,
        final_retrieval_sufficient: bool | None = None,
        retry_count: int = 0,
        query_rewritten: bool = False,
        corrective_retrieval_used: bool = False,
        validator_result: dict[str, Any] | None = None,
    ) -> RAGEvaluationResult:
        """
        Evaluate one RAG response.

        relevant_document_ids:
            IDs of documents judged relevant by human annotation or
            a trusted evaluation dataset. Needed for retrieval recall.

        expected_evidence_ids:
            Evidence IDs expected to support the answer. If supplied,
            citation coverage can be measured against these IDs.

        validator_result:
            Optional output from AnswerValidator.validate(). Prefer this
            for unsupported claims and grounding status, while keeping
            evaluator metrics independently calculated.
        """
        started_at = time.perf_counter()
        warnings: list[str] = []

        question = (question or "").strip()
        answer = (answer or "").strip()
        docs = list(documents or [])

        if not question:
            raise ValueError("question must not be empty.")

        if retry_count < 0:
            raise ValueError("retry_count cannot be negative.")

        retrieval = self.evaluate_retrieval(
            documents=docs,
            relevant_document_ids=relevant_document_ids,
        )

        grounding = self.evaluate_citations(
            answer=answer,
            documents=docs,
            expected_evidence_ids=expected_evidence_ids,
            validator_result=validator_result,
        )

        answer_eval = AnswerEvaluation(
            answer_present=bool(answer),
        )

        if self.enable_llm_judge and answer:
            try:
                judged = self._evaluate_answer_semantically(
                    question=question,
                    answer=answer,
                    documents=docs,
                )

                answer_eval.answer_relevance_score = (
                    judged.answer_relevance_score
                )
                answer_eval.faithfulness_score = (
                    judged.faithfulness_score
                )
                answer_eval.correctness_score = (
                    judged.correctness_score
                )
                answer_eval.evaluation_method = "llm_judge"

                if judged.unsupported_claims:
                    grounding.unsupported_claims = (
                        judged.unsupported_claims
                    )

                if judged.faithfulness_score < self.grounding_threshold:
                    grounding.grounded = False

            except Exception as exc:
                logger.exception(
                    "LLM judge evaluation failed | request_id=%s",
                    request_id,
                )
                answer_eval.evaluation_error = str(exc)
                answer_eval.evaluation_method = (
                    "deterministic_with_llm_judge_error"
                )
                warnings.append(
                    "Semantic evaluation failed; semantic scores "
                    "are unavailable."
                )

        if expected_answer:
            # Similarity against the expected answer is deliberately
            # not calculated with lexical overlap. Use a task-specific
            # semantic metric or a reference-based evaluation suite.
            warnings.append(
                "expected_answer was supplied, but reference-based "
                "correctness is not computed by this evaluator. "
                "Use a benchmark metric such as ROUGE, BERTScore, "
                "or a separately calibrated semantic judge."
            )

        if relevant_document_ids is None:
            warnings.append(
                "Retrieval recall and MAP are unavailable without "
                "ground-truth relevant_document_ids."
            )

        if expected_evidence_ids is None:
            warnings.append(
                "Citation coverage is based on observed citations "
                "unless expected_evidence_ids are supplied."
            )

        corrective = CorrectiveRAGEvaluation(
            retry_count=retry_count,
            query_rewritten=query_rewritten,
            initial_retrieval_sufficient=initial_retrieval_sufficient,
            final_retrieval_sufficient=final_retrieval_sufficient,
            corrective_retrieval_used=corrective_retrieval_used,
        )

        overall_score = self._calculate_overall_score(
            retrieval=retrieval,
            grounding=grounding,
            answer=answer_eval,
            has_retrieval_labels=relevant_document_ids is not None,
        )

        result = RAGEvaluationResult(
            request_id=request_id,
            retrieval=retrieval,
            grounding=grounding,
            answer=answer_eval,
            corrective_rag=corrective,
            evaluation_time_seconds=round(
                time.perf_counter() - started_at,
                6,
            ),
            overall_score=overall_score,
            warnings=warnings,
        )

        logger.info(
            "RAG evaluation completed | request_id=%s "
            "| retrieved=%d | precision_at_k=%.3f "
            "| citation_precision=%.3f | grounded=%s "
            "| overall_score=%s",
            request_id,
            retrieval.retrieved_count,
            retrieval.precision_at_k,
            grounding.citation_precision,
            grounding.grounded,
            overall_score,
        )

        return result

    # --------------------------------------------------------
    # Retrieval evaluation
    # --------------------------------------------------------

    def evaluate_retrieval(
        self,
        *,
        documents: Sequence[Document],
        relevant_document_ids: set[str] | None = None,
    ) -> RetrievalEvaluation:
        docs = list(documents or [])

        result = RetrievalEvaluation(
            retrieved_count=len(docs),
        )

        if not docs:
            return result

        if relevant_document_ids is None:
            return result

        relevant_ids = {
            str(item)
            for item in relevant_document_ids
        }

        retrieved_ids = [
            self._document_id(doc)
            for doc in docs
        ]

        relevance = [
            document_id in relevant_ids
            for document_id in retrieved_ids
        ]

        result.relevant_count = sum(relevance)
        result.precision_at_k = (
            result.relevant_count / len(docs)
        )

        result.recall_at_k = (
            len(set(
                document_id
                for document_id, is_relevant
                in zip(retrieved_ids, relevance)
                if is_relevant
            ))
            / len(relevant_ids)
            if relevant_ids
            else None
        )

        first_relevant_rank = next(
            (
                index
                for index, is_relevant
                in enumerate(relevance, start=1)
                if is_relevant
            ),
            None,
        )

        result.reciprocal_rank = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        result.hit_at_k = (
            1.0 if result.relevant_count > 0 else 0.0
        )

        precision_sum = 0.0
        relevant_seen = 0

        for rank, is_relevant in enumerate(relevance, start=1):
            if is_relevant:
                relevant_seen += 1
                precision_sum += relevant_seen / rank

        result.mean_average_precision = (
            precision_sum / min(len(relevant_ids), len(docs))
            if relevant_ids
            else None
        )

        return result

    # --------------------------------------------------------
    # Citation and grounding evaluation
    # --------------------------------------------------------

    def evaluate_citations(
        self,
        *,
        answer: str,
        documents: Sequence[Document],
        expected_evidence_ids: set[str] | None = None,
        validator_result: dict[str, Any] | None = None,
    ) -> GroundingEvaluation:
        docs = list(documents or [])

        evidence_ids = {
            self._evidence_id(doc, index)
            for index, doc in enumerate(docs, start=1)
        }

        citations = self.CITATION_PATTERN.findall(answer or "")
        unique_citations = set(citations)

        invalid = sorted(
            citation
            for citation in unique_citations
            if citation not in evidence_ids
        )

        valid = sorted(
            citation
            for citation in unique_citations
            if citation in evidence_ids
        )

        if unique_citations:
            citation_precision = (
                len(valid) / len(unique_citations)
            )
        else:
            citation_precision = 0.0

        if expected_evidence_ids:
            expected = {
                str(item)
                for item in expected_evidence_ids
            }
            citation_coverage = (
                len(unique_citations & expected) / len(expected)
                if expected
                else 0.0
            )
            missing = sorted(expected - unique_citations)
        else:
            # This is citation presence, not true evidence coverage.
            citation_coverage = 1.0 if unique_citations else 0.0
            missing = []

        validator = validator_result or {}
        unsupported = validator.get("unsupported_claims", [])
        validator_grounded = validator.get("grounded")

        grounded = bool(
            answer
            and unique_citations
            and not invalid
            and not unsupported
        )

        if validator_grounded is not None:
            grounded = grounded and bool(validator_grounded)

        return GroundingEvaluation(
            citation_count=len(unique_citations),
            valid_citation_count=len(valid),
            invalid_citations=invalid,
            missing_citations=missing,
            citation_precision=citation_precision,
            citation_coverage=citation_coverage,
            grounded=grounded,
            unsupported_claims=list(unsupported),
        )

    # --------------------------------------------------------
    # Optional LLM semantic evaluation
    # --------------------------------------------------------

    def _evaluate_answer_semantically(
        self,
        *,
        question: str,
        answer: str,
        documents: Sequence[Document],
    ) -> LLMJudgeResult:
        if self._judge_chain is None:
            raise RuntimeError("LLM judge is not configured.")

        context = self._build_judge_context(documents)

        result = self._judge_chain.invoke({
            "question": question,
            "answer": answer,
            "context": context,
        })

        if isinstance(result, LLMJudgeResult):
            return result

        return LLMJudgeResult.model_validate(result)

    def _build_judge_context(
        self,
        documents: Sequence[Document],
    ) -> str:
        parts: list[str] = []
        used_chars = 0

        for index, document in enumerate(documents, start=1):
            content = (document.page_content or "").strip()

            if not content:
                continue

            source = (
                document.metadata.get("source", "unknown")
                if document.metadata
                else "unknown"
            )

            part = (
                f"[E{index}] Source: {source}\n"
                f"{content}"
            )

            if used_chars + len(part) > self.max_context_chars:
                break

            parts.append(part)
            used_chars += len(part)

        return "\n\n".join(parts)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _document_id(document: Document) -> str:
        metadata = document.metadata or {}

        identifier = (
            metadata.get("chunk_id")
            or metadata.get("document_chunk_id")
            or metadata.get("document_id")
            or document.id
        )

        if identifier:
            return str(identifier)

        # Fallback is content-derived. Prefer stable IDs assigned
        # during ingestion for benchmark evaluation.
        import hashlib

        return hashlib.sha256(
            (document.page_content or "").encode("utf-8")
        ).hexdigest()[:24]

    @staticmethod
    def _evidence_id(
        document: Document,
        index: int,
    ) -> str:
        metadata = document.metadata or {}

        return str(
            metadata.get("evidence_id")
            or f"E{index}"
        )

    @staticmethod
    def _calculate_overall_score(
        *,
        retrieval: RetrievalEvaluation,
        grounding: GroundingEvaluation,
        answer: AnswerEvaluation,
        has_retrieval_labels: bool,
    ) -> float | None:
        """
        Calculate a transparent composite score only when semantic
        evaluation is available. The score is a project-specific
        indicator, not a standard RAG benchmark.
        """
        components: list[tuple[float, float]] = []

        if has_retrieval_labels:
            components.append((retrieval.precision_at_k, 0.25))

        components.append((grounding.citation_precision, 0.20))

        if answer.answer_relevance_score is not None:
            components.append((answer.answer_relevance_score, 0.25))

        if answer.faithfulness_score is not None:
            components.append((answer.faithfulness_score, 0.30))

        if not components:
            return None

        total_weight = sum(weight for _, weight in components)

        return round(
            sum(score * weight for score, weight in components)
            / total_weight,
            4,
        )


# ============================================================
# Convenience function
# ============================================================

def evaluate_rag_response(
    *,
    question: str,
    answer: str,
    documents: Sequence[Document],
    request_id: str | None = None,
    llm: BaseChatModel | None = None,
    enable_llm_judge: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    One-call evaluation helper.

    For repeated evaluations, instantiate RAGEvaluator once and reuse it.
    """
    evaluator = RAGEvaluator(
        llm=llm,
        enable_llm_judge=enable_llm_judge,
    )

    result = evaluator.evaluate(
        question=question,
        answer=answer,
        documents=documents,
        request_id=request_id,
        **kwargs,
    )

    return result.to_dict()
