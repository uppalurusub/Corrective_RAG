
import logging
import time
from typing import Any

from langchain_core.documents import Document

from app.graphs.corrective_rag_graph import (
    build_corrective_rag_graph,
)

logger = logging.getLogger(__name__)
from app.config.llm_config import get_llm
llm = get_llm()

class RAGImplementation:
    """
    Implementation layer for Corrective / Adaptive RAG.

    Dependencies:
        llm:
            Initialized LangChain chat model.
        retrievers:
            Dictionary of initialized retrievers, for example:
            {
                "dense": dense_retriever,
                "hybrid": hybrid_retriever,
            }
        graph:
            Optional precompiled LangGraph instance.

    The LLM, retrievers, and graph should be reused across requests.
    """

    def __init__(
        self,
        llm: Any = None,
        retrievers: dict[str, Any] | None = None,
        graph: Any = None,
        max_retries: int = 2,
    ) -> None:
        #self.llm = llm
        self.llm = llm if llm is not None else get_llm()
        self.retrievers = retrievers
        self.max_retries = max_retries
        self.graph = graph

        if self.graph is None:
            self._initialize_graph()

    def _initialize_graph(self) -> None:
        """
        Compile the graph once during application initialization.

        Supply the LLM and retrievers through dependency injection.
        Do not silently create dummy retrievers or an unconfigured LLM.
        """
        if self.llm is None:
            raise ValueError(
                "An initialized LangChain LLM is required."
            )

        if not isinstance(self.retrievers, dict):
            raise ValueError(
                "Provide initialized retrievers as a dictionary."
            )

        if "hybrid" not in self.retrievers:
            raise ValueError(
                "A hybrid retriever is required by the current graph."
            )

        if not callable(getattr(self.retrievers["hybrid"], "invoke", None)):
            raise TypeError(
                "The hybrid retriever must implement invoke(query)."
            )

        if (
            "dense" in self.retrievers
            and not callable(
                getattr(self.retrievers["dense"], "invoke", None)
            )
        ):
            raise TypeError(
                "The dense retriever must implement invoke(query)."
            )

        self.graph = build_corrective_rag_graph(
            llm=self.llm,
            retrievers=self.retrievers,
            max_retries=self.max_retries,
        )

        logger.info(
            "Corrective RAG graph initialized | max_retries=%d",
            self.max_retries,
        )

    def run(
        self,
        question: str,
        request_id: str,
        top_k: int = 5,
        use_corrective_rag: bool = True,
    ) -> dict[str, Any]:
        """
        Invoke the compiled graph and normalize its output.

        Returns a dictionary compatible with RAGService.ask_question.
        """
        started_at = time.perf_counter()

        self._validate_inputs(
            question=question,
            request_id=request_id,
            top_k=top_k,
        )

        if self.graph is None:
            raise RuntimeError("RAG graph has not been initialized.")

        max_retries = self.max_retries if use_corrective_rag else 0

        initial_state = {
            "question": question.strip(),
            "request_id": request_id,
            "retries": 0,
            "max_retries": max_retries,
            "top_k": top_k,
        }

        logger.info(
            "RAG graph invocation started | request_id=%s "
            "| top_k=%d | corrective=%s",
            request_id,
            top_k,
            use_corrective_rag,
        )

        try:
            # Graph execution owns routing, retrieval, grading,
            # correction, generation, and grounding validation.
            state = self.graph.invoke(initial_state)

            if not isinstance(state, dict):
                raise TypeError(
                    "LangGraph must return a state dictionary."
                )

            answer = state.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(
                    "RAG graph returned an empty or invalid answer."
                )

            documents = state.get(
                "relevant_documents",
                state.get("documents", []),
            )

            sources = self._serialize_documents(documents)

            grounded = state.get("grounded")
            if not isinstance(grounded, bool):
                grounded = None

            unsupported_claims = state.get(
                "unsupported_claims", []
            )
            if not isinstance(unsupported_claims, list):
                unsupported_claims = []

            retries = self._safe_int(state.get("retries", 0))

            # Preserve the graph's explicit status where available.
            graph_status = state.get("status")
            if graph_status:
                result_status = graph_status
            elif not sources:
                result_status = "insufficient_evidence"
            elif grounded is True:
                result_status = "grounded"
            elif grounded is False:
                result_status = "grounding_failed"
            else:
                result_status = "completed_unverified"

            elapsed = time.perf_counter() - started_at

            result = {
                "request_id": request_id,
                "answer": answer.strip(),
                "status": result_status,
                "grounded": grounded,
                "unsupported_claims": unsupported_claims,
                "sources": sources,
                "retrieval_strategy": state.get("strategy"),
                "retry_count": retries,
                "execution_time_seconds": round(elapsed, 4),
            }

            logger.info(
                "RAG graph invocation completed | request_id=%s "
                "| status=%s | sources=%d | retries=%d "
                "| elapsed_seconds=%.4f",
                request_id,
                result_status,
                len(sources),
                retries,
                elapsed,
            )

            return result

        except Exception:
            logger.exception(
                "RAG graph invocation failed | request_id=%s",
                request_id,
            )
            raise

    @staticmethod
    def _validate_inputs(
        question: str,
        request_id: str,
        top_k: int,
    ) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be empty")

        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must not be empty")

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 20
        ):
            raise ValueError("top_k must be between 1 and 20")

    @classmethod
    def _serialize_documents(
        cls,
        documents: Any,
    ) -> list[dict[str, Any]]:
        """Convert retrieved LangChain Documents into API sources."""
        if documents is None:
            return []

        if not isinstance(documents, list):
            raise TypeError(
                "Graph document state must be a list."
            )

        sources = []

        for doc in documents:
            if not isinstance(doc, Document):
                raise TypeError(
                    "Graph returned an invalid document object."
                )

            metadata = doc.metadata or {}

            sources.append({
                "source_id": metadata.get("chunk_id")
                    or metadata.get("document_id"),
                "source": metadata.get("source")
                    or metadata.get("filename"),
                "page": metadata.get("page"),
                "section": metadata.get("section")
                    or metadata.get("heading"),
                "content": doc.page_content,
                "score": cls._safe_float(
                    metadata.get("retrieval_score")
                ),
            })

        return sources

    @staticmethod
    def _safe_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0

        try:
            return max(0, int(value))
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
