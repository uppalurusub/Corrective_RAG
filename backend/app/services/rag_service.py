
import logging
import time
from typing import Any

from langchain_core.documents import Document

from app.implementations.rag_impl import RAGImplementation

logger = logging.getLogger(__name__)


class RAGService:
    """
    Application service for Corrective / Adaptive RAG.

    Responsibilities:
    - Validate service inputs.
    - Invoke the RAG implementation.
    - Normalize LangGraph results.
    - Preserve evidence and grounding metadata.
    - Measure execution time.
    - Propagate safe, meaningful failures.
    """

    def __init__(
        self,
        implementation: RAGImplementation | None = None,
    ) -> None:
        # The implementation should reuse initialized LLM,
        # embedding, and vector-store resources.
        self.implementation = (
            implementation or RAGImplementation()
        )

    def ask_question(
        self,
        question: str,
        request_id: str,
        top_k: int = 5,
        use_corrective_rag: bool = True,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()

        self._validate_request(
            question=question,
            request_id=request_id,
            top_k=top_k,
        )

        normalized_question = question.strip()

        logger.info(
            "RAG service started | request_id=%s | top_k=%d "
            "| corrective=%s",
            request_id,
            top_k,
            use_corrective_rag,
        )

        try:
            # Invoke the implementation, which owns graph execution.
            result = self.implementation.run(
                question=normalized_question,
                request_id=request_id,
                top_k=top_k,
                use_corrective_rag=use_corrective_rag,
            )

            if not isinstance(result, dict):
                raise TypeError(
                    "RAG implementation must return a dictionary"
                )

            answer = result.get("answer", "")
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(
                    "RAG implementation returned no valid answer"
                )

            sources = self._normalize_sources(
                result.get("sources", [])
            )

            grounded = result.get("grounded")
            if grounded is not None and not isinstance(
                grounded, bool
            ):
                grounded = None

            unsupported_claims = result.get(
                "unsupported_claims", []
            )
            if not isinstance(unsupported_claims, list):
                unsupported_claims = []

            elapsed = time.perf_counter() - started_at

            response = {
                "request_id": request_id,
                "question": normalized_question,
                "answer": answer.strip(),
                "status": result.get("status", "completed"),
                "sources": sources,
                "grounded": grounded,
                "unsupported_claims": unsupported_claims,
                "retrieval_strategy": result.get(
                    "retrieval_strategy"
                ),
                "retry_count": self._safe_int(
                    result.get("retry_count", 0)
                ),
                "execution_time_seconds": round(elapsed, 4),
            }

            logger.info(
                "RAG service completed | request_id=%s "
                "| status=%s | sources=%d | retries=%d "
                "| elapsed_seconds=%.4f",
                request_id,
                response["status"],
                len(sources),
                response["retry_count"],
                elapsed,
            )

            return response

        except Exception:
            logger.exception(
                "RAG service failed | request_id=%s",
                request_id,
            )
            raise

    @staticmethod
    def _validate_request(
        question: str,
        request_id: str,
        top_k: int,
    ) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must not be empty")

        if len(question.strip()) > 5000:
            raise ValueError("Question exceeds maximum length")

        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must not be empty")

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= 20
        ):
            raise ValueError("top_k must be between 1 and 20")

    @classmethod
    def _normalize_sources(
        cls,
        sources: Any,
    ) -> list[dict[str, Any]]:
        """
        Accept either serialized source dictionaries or
        LangChain Document instances.

        Source identifiers should originate from trusted retrieval
        metadata, not from generated answer text.
        """
        if sources is None:
            return []

        if not isinstance(sources, list):
            raise TypeError("sources must be a list")

        normalized: list[dict[str, Any]] = []

        for source in sources:
            if isinstance(source, Document):
                metadata = source.metadata or {}

                normalized.append({
                    "source_id": metadata.get("chunk_id"),
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "section": metadata.get("section"),
                    "content": source.page_content,
                    "score": cls._safe_float(
                        metadata.get("retrieval_score")
                    ),
                })

            elif isinstance(source, dict):
                normalized.append({
                    "source_id": source.get("source_id"),
                    "source": source.get("source"),
                    "page": source.get("page"),
                    "section": source.get("section"),
                    "content": source.get("content"),
                    "score": cls._safe_float(
                        source.get("score")
                    ),
                })

            else:
                raise TypeError(
                    "Each source must be a Document or dictionary"
                )

        return normalized

    @staticmethod
    def _safe_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        try:
            result = int(value)
            return max(result, 0)
        except (TypeError, ValueError, OverflowError):
            return 0

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            result = float(value)
            return result if result == result else None
        except (TypeError, ValueError, OverflowError):
            return None
