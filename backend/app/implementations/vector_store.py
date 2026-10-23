from functools import lru_cache
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create and cache the configured embedding model."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


_store: Optional[Chroma] = None


def get_vector_store(force_recreate: bool = False) -> Chroma:
    """
    Return the cached Chroma vector store.

    `force_recreate=True` recreates the Python wrapper; it does not delete the
    persisted collection or re-embed documents.
    """
    global _store

    if force_recreate:
        _store = None

    if _store is None:
        settings.resolved_vector_db_path.mkdir(parents=True, exist_ok=True)
        _store = Chroma(
            collection_name=settings.chroma_collection,
            persist_directory=str(settings.resolved_vector_db_path),
            embedding_function=get_embeddings(),
        )

    return _store


def normalize_metadata_filter(filters: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """
    Convert an absent/empty filter to None.

    Chroma rejects `where={}`; passing None tells LangChain not to apply a
    metadata filter. Non-empty filters are passed through for Chroma validation.
    """
    if filters is None or filters == {}:
        return None
    return filters


def similarity_search_with_relevance_scores(
    query: str,
    k: int,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Run similarity search without passing an empty metadata filter to Chroma.
    """
    store = get_vector_store()
    normalized_filter = normalize_metadata_filter(filters)

    kwargs = {"k": max(1, int(k))}
    if normalized_filter is not None:
        kwargs["filter"] = normalized_filter

    return store.similarity_search_with_relevance_scores(query, **kwargs)
