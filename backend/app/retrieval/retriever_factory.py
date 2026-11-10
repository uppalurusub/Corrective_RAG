
import logging
import os
from pathlib import Path
from langchain_chroma import Chroma


from app.config.embedding_config import get_embeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.implementations.vector_store import get_vector_store
from app.retrieval.hybrid_retriever import HybridRetriever

from app.config.settings import settings
logger = settings.get_logger()

""" PROJECT_ROOT = Path(__file__).resolve().parents[3]
print("PROJECT_ROOT: ",PROJECT_ROOT) """

def build_retrievers() -> dict:

    vector_store = get_vector_store()
    collection = vector_store._collection

    count = collection.count()

    logger.info(
        "Chroma diagnostics | path={} | collection={} | count={}",
        settings.resolved_vector_db_path,
        collection.name,
        count,
    )

    if count == 0:
        raise RuntimeError(
            f"Chroma collection '{collection.name}' is empty. "
            "Ingest documents before starting the RAG application."
        )

    stored = collection.get(
        include=["documents", "metadatas"]
    )

    texts = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    ids = stored.get("ids") or []

    readable_count = 0
    empty_count = 0

    documents = []

    for i, text in enumerate(texts):

        if not isinstance(text, str) or not text.strip():
            empty_count += 1
            continue

        readable_count += 1

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

    logger.info(
        "Chroma document diagnostics | "
        "records={} | returned_texts={} | readable={} | empty={}",
        count,
        len(texts),
        readable_count,
        empty_count,
    )

    # Log a small metadata sample, not document contents.
    if metadatas:
        logger.info(
            "Chroma metadata sample | {}",
            metadatas[0],
        )

    if not documents:
        raise RuntimeError(
            "BM25 initialization failed: Chroma contains records "
            "but no readable document text. "
            f"Collection={collection.name}, "
            f"Path={settings.resolved_vector_db_path}, "
            f"Records={count}, "
            f"Returned texts={len(texts)}, "
            f"Readable texts={readable_count}. "
            "Verify ingestion path, collection name, and stored documents."
        )

    dense_retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.top_k}
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
        default_top_k=settings.top_k,
    )

    logger.info(
        "RAG retrievers initialized | "
        "dense=enabled | bm25=enabled | hybrid=enabled | chunks={}",
        len(documents),
    )

    return {
        "dense": dense_retriever,
        "hybrid": hybrid_retriever,
    }