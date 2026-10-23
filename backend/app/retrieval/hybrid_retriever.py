
import hashlib
import logging
from collections import defaultdict
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict, Field, PrivateAttr

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval using dense vector search, BM25, and
    Reciprocal Rank Fusion (RRF).

    Parameters:
        vector_store:
            Initialized LangChain-compatible vector store,
            such as Chroma.
        documents:
            Documents used to build the BM25 index. Supply the same
            indexed corpus, or provide a prebuilt bm25_retriever.
        bm25_retriever:
            Optional prebuilt keyword retriever.
        dense_weight:
            Weight for dense retrieval in RRF.
        bm25_weight:
            Weight for BM25 retrieval in RRF.
        rrf_k:
            RRF rank constant; larger values reduce the influence
            of high-ranking positions.
        fetch_k:
            Number of candidates requested from each retriever.
        default_top_k:
            Number of fused results returned by default.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: Any = Field(exclude=True)
    bm25_retriever: Any = Field(default=None, exclude=True)

    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    rrf_k: int = 60
    fetch_k: int = 20
    default_top_k: int = 5

    _bm25_index: Any = PrivateAttr(default=None)

    def __init__(
        self,
        vector_store: Any,
        documents: Sequence[Document] | None = None,
        bm25_retriever: Any = None,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: int = 60,
        fetch_k: int = 20,
        default_top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            rrf_k=rrf_k,
            fetch_k=fetch_k,
            default_top_k=default_top_k,
            **kwargs,
        )

        if vector_store is None:
            raise ValueError("vector_store is required.")

        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError("Retrieval weights cannot be negative.")

        if dense_weight + bm25_weight <= 0:
            raise ValueError("At least one weight must be positive.")

        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        if fetch_k < 1 or default_top_k < 1:
            raise ValueError("Retrieval limits must be positive.")

        if bm25_retriever is not None:
            self._bm25_index = bm25_retriever

        elif documents:
            # Build this once, not on every query.
            try:
                from langchain_community.retrievers import BM25Retriever

                self._bm25_index = BM25Retriever.from_documents(
                    list(documents),
                    k=fetch_k,
                )
            except ImportError as exc:
                raise ImportError(
                    "Install compatible langchain-community and "
                    "rank_bm25 packages to use BM25 retrieval."
                ) from exc

        else:
            raise ValueError(
                "Provide documents to build BM25 or a prebuilt "
                "bm25_retriever."
            )

        logger.info(
            "Hybrid retriever initialized | fetch_k=%d | top_k=%d "
            "| dense_weight=%.2f | bm25_weight=%.2f",
            fetch_k,
            default_top_k,
            dense_weight,
            bm25_weight,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """LangChain standard retriever interface."""
        return self.search(
            query=query,
            top_k=self.default_top_k,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieve, fuse, deduplicate, and rank relevant documents.

        RRF scores are ranking scores, not calibrated probabilities.
        """
        query = self._validate_query(query)
        limit = top_k or self.default_top_k

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("top_k must be an integer.")

        if not 1 <= limit <= 100:
            raise ValueError("top_k must be between 1 and 100.")

        candidates = max(self.fetch_k, limit)

        dense_results = self._dense_search(
            query=query,
            fetch_k=candidates,
            metadata_filter=metadata_filter,
        )

        bm25_results = self._bm25_search(
            query=query,
            fetch_k=candidates,
            metadata_filter=metadata_filter,
        )

        fused = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        results = []

        for rank, item in enumerate(fused[:limit], start=1):
            metadata = dict(item["document"].metadata or {})
            metadata.update({
                "retrieval_method": "hybrid_rrf",
                "retrieval_rank": rank,
                "retrieval_score": item["rrf_score"],
                "retrieval_sources": item["retrieval_sources"],
            })

            # Return a copy so retrieval metadata does not mutate
            # the original indexed Document.
            results.append(
                Document(
                    page_content=item["document"].page_content,
                    metadata=metadata,
                    id=item["document"].id,
                )
            )

        logger.info(
            "Hybrid retrieval completed | dense=%d | bm25=%d "
            "| fused=%d | returned=%d",
            len(dense_results),
            len(bm25_results),
            len(fused),
            len(results),
        )

        return results

    def _dense_search(
        self,
        query: str,
        fetch_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[Document]:
        kwargs: dict[str, Any] = {"k": fetch_k}

        if metadata_filter:
            kwargs["filter"] = metadata_filter

        # Prefer a backend that supports server-side metadata filtering.
        if hasattr(self.vector_store, "similarity_search"):
            return self.vector_store.similarity_search(
                query,
                **kwargs,
            )

        raise TypeError(
            "vector_store must implement similarity_search()."
        )

    def _bm25_search(
        self,
        query: str,
        fetch_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[Document]:
        # BM25Retriever uses its configured k value.
        # The index is built once and reused.
        docs = self._bm25_index.invoke(query)

        if not isinstance(docs, list):
            raise TypeError("BM25 retriever must return a list.")

        if metadata_filter:
            docs = [
                doc for doc in docs
                if self._matches_metadata_filter(
                    doc.metadata or {},
                    metadata_filter,
                )
            ]

        return docs[:fetch_k]

    
    def _reciprocal_rank_fusion(
        self,
        dense_results: list[Document],
        bm25_results: list[Document],
    ) -> list[dict[str, Any]]:
        """Fuse dense and BM25 rankings using weighted RRF."""
        scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Document] = {}
        methods: dict[str, set[str]] = defaultdict(set)

        sources = (
            ("dense", dense_results, self.dense_weight),
            ("bm25", bm25_results, self.bm25_weight),
        )

        for method, docs, weight in sources:
            for rank, doc in enumerate(docs, start=1):
                key = self._document_key(doc)

                scores[key] += weight / (self.rrf_k + rank)
                methods[key].add(method)

                if key not in documents:
                    documents[key] = doc

        ordered_keys = sorted(
            scores,
            key=lambda key: scores[key],
            reverse=True,
        )

        return [
            {
                "document": documents[key],
                "rrf_score": scores[key],
                "retrieval_sources": sorted(methods[key]),
            }
            for key in ordered_keys
        ]


    def _get_dense_weight(self) -> float:
        return self.dense_weight

    def _get_bm25_weight(self) -> float:
        return self.bm25_weight

    @staticmethod
    def _document_key(doc: Document) -> str:
        """
        Prefer stable chunk IDs. Fall back to source metadata and
        content hashing when stable IDs are not available.
        """
        metadata = doc.metadata or {}

        stable_id = (
            metadata.get("chunk_id")
            or metadata.get("document_chunk_id")
            or doc.id
        )

        if stable_id:
            return str(stable_id)

        identity = "|".join([
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            doc.page_content,
        ])

        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _matches_metadata_filter(
        metadata: dict[str, Any],
        filters: dict[str, Any],
    ) -> bool:
        """
        Basic equality filtering for BM25 candidates.

        This does not implement the full Chroma filter syntax.
        """
        for key, expected in filters.items():
            actual = metadata.get(key)

            if isinstance(expected, dict):
                # Fail closed for unsupported filter operators.
                if len(expected) != 1:
                    return False

                operator, value = next(iter(expected.items()))

                if operator == "$eq":
                    if actual != value:
                        return False
                elif operator == "$in":
                    if not isinstance(value, list) or actual not in value:
                        return False
                else:
                    raise ValueError(
                        f"Unsupported BM25 filter operator: {operator}"
                    )
            elif actual != expected:
                return False

        return True

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty.")

        if len(query) > 5000:
            raise ValueError("query must not exceed 5000 characters.")

        return query.strip()
