
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config.settings import settings
logger = settings.get_logger()

from app.implementations.rag_impl import RAGImplementation
from app.services.rag_service import RAGService

        # Import your project's retriever factory.
        # This factory must initialize the vector store and return
        # {"hybrid": hybrid_retriever}.
from app.retrieval.retriever_factory import build_retrievers

# Load environment variables before initializing the LLM.
load_dotenv()




def configure_logging() -> None:
    """Configure application logging."""

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        force=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared RAG dependencies once per application process."""

    configure_logging()
    app.state.rag_ready = False
    app.state.rag_service = None

    logger.info("Starting Corrective / Adaptive RAG API")

    try:
        

        # 1. Initialize the LLM.
        llm = settings.get_model_client()

        if llm is None:
            raise RuntimeError(
                "get_llm() returned None. Check your LLM configuration."
            )

        logger.info("LLM initialized successfully")
        logger.info(
            "RAG configuration | documents_dir={} | vector_db_path={} | collection={}",
            settings.resolved_documents_dir,
            settings.resolved_vector_db_path,
            settings.chroma_collection,
        )

        # 2. Initialize retrievers once.
        retrievers = build_retrievers()

        if not isinstance(retrievers, dict):
            raise TypeError(
                "build_retrievers() must return a dictionary."
            )

        hybrid_retriever = retrievers.get("hybrid")

        if hybrid_retriever is None:
            raise RuntimeError(
                "Retriever factory did not provide a 'hybrid' retriever."
            )

        if not callable(getattr(hybrid_retriever, "invoke", None)):
            raise TypeError(
                "The hybrid retriever must implement invoke(query)."
            )

        # 3. Initialize the graph implementation using dependency injection.
        implementation = RAGImplementation(
            llm=llm,
            retrievers=retrievers,
            max_retries=int(os.getenv("RAG_MAX_RETRIES", "2")),
        )

        # 4. Inject the initialized implementation into the service.
        service = RAGService(
            implementation=implementation,
        )

        # 5. Store shared resources for API requests.
        app.state.llm = llm
        app.state.retrievers = retrievers
        app.state.rag_service = service

        # Optional service-level dependency health check.
        health_check = getattr(service, "health_check", None)

        if callable(health_check):
            result = health_check()

            if hasattr(result, "__await__"):
                result = await result

            if result is False:
                raise RuntimeError(
                    "RAG service health check failed."
                )

        app.state.rag_ready = True

        logger.info(
            "RAG application ready | retrievers=%s",
            list(retrievers.keys()),
        )

        yield

    except Exception:
        app.state.rag_ready = False
        logger.exception(
            "Failed to initialize or run the RAG application"
        )
        raise

    finally:
        app.state.rag_ready = False

        service = getattr(app.state, "rag_service", None)

        if service is not None:
            close_method = getattr(service, "close", None)

            if callable(close_method):
                try:
                    result = close_method()

                    if hasattr(result, "__await__"):
                        await result

                except Exception:
                    logger.exception(
                        "Failed to close RAG service cleanly"
                    )

        logger.info("RAG application shutdown complete")


app = FastAPI(
    title="Corrective / Adaptive RAG API",
    description=(
        "Evidence-grounded RAG with hybrid retrieval, relevance "
        "grading, corrective retrieval, and answer validation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# CORS configuration
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8501",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Request-ID",
    ],
)


# Register the API router.
from app.router.rag_router import router as rag_router

app.include_router(rag_router)


@app.get("/", tags=["Application"])
async def root() -> dict[str, Any]:
    return {
        "service": "corrective-adaptive-rag",
        "version": app.version,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "readiness": "/ready",
    }


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    """Liveness check; does not guarantee dependency availability."""

    return {
        "status": "healthy",
        "service": "corrective-adaptive-rag",
    }


@app.get("/ready", tags=["Health"])
async def readiness():
    """Readiness check after successful dependency initialization."""

    if not getattr(app.state, "rag_ready", False):
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": "corrective-adaptive-rag",
            },
        )

    return {
        "status": "ready",
        "service": "corrective-adaptive-rag",
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    """Return a generic error without exposing internal details."""

    logger.exception(
        "Unhandled API exception | method=%s | path=%s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected server error occurred.",
        },
    )