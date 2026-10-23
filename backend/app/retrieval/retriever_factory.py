
import logging
import os
from pathlib import Path
from langchain_chroma import Chroma


from app.config.embedding_config import get_embeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.implementations.vector_store import get_vector_store
from app.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
print("PROJECT_ROOT: ",PROJECT_ROOT)

def build_retrievers() -> dict:
    # Reuse the exact store configuration used by ingestion.
    vector_store = get_vector_store()
    collection = vector_store._collection
    count = collection.count()

    logger.info(
        "RAG vector store | collection=%s | count=%d",
        collection.name,
        count,
    )

    if count == 0:
        raise RuntimeError(
            f"Chroma collection '{collection.name}' is empty. "
            "Run the document-ingestion pipeline before starting RAG."
        )

    dense_retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    stored = collection.get(
        include=["documents", "metadatas"]
    )

    texts = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    ids = stored.get("ids") or []

    documents = []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue

        metadata = (
            metadatas[i]
            if i < len(metadatas) and metadatas[i]
            else {}
        )

        document_id = ids[i] if i < len(ids) else None

        documents.append(
            Document(
                page_content=text,
                metadata=dict(metadata),
                id=document_id,
            )
        )

    if not documents:
        raise RuntimeError(
            "Chroma contains records but no readable text "
            "is available for BM25 indexing."
        )

    bm25_retriever = BM25Retriever.from_documents(
        documents,
        k=20,
    )

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        dense_weight=1.0,
        bm25_weight=1.0,
        rrf_k=60,
        fetch_k=20,
        default_top_k=5,
    )

    logger.info(
        "RAG retrievers initialized | dense=enabled "
        "| bm25=enabled | hybrid=enabled | chunks=%d",
        len(documents),
    )

    return {
        "dense": dense_retriever,
        "hybrid": hybrid_retriever,
    }
