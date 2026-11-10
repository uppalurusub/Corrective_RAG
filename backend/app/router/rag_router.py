
import logging
from typing import Any
from uuid import uuid4
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from app.services.rag_service import RAGService
from app.schemas.rag_schemas import IngestResponse
from app.services.documents_service import DocumentsService
from app.config.dependencies import documents_service
from app.config.settings import settings
logger = settings.get_logger()


ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

router = APIRouter(
    prefix="/api/v1/rag",
    tags=["RAG"],
)


# ---------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------

class RAGQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Question to answer using retrieved documents",
    )
    request_id: str | None = Field(
        default=None,
        max_length=128,
    )
    top_k: int = Field(default=5, ge=1, le=20)
    use_corrective_rag: bool = True


class RAGSource(BaseModel):
    source_id: str | None = None
    source: str | None = None
    page: int | str | None = None
    section: str | None = None
    content: str | None = None
    score: float | None = None


class RAGQueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    status: str
    sources: list[RAGSource] = Field(default_factory=list)
    grounded: bool | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    retrieval_strategy: str | None = None
    retry_count: int = 0
    execution_time_seconds: float | None = None


class RAGHealthResponse(BaseModel):
    status: str
    service: str


# ---------------------------------------------------------
# Dependencies
# ---------------------------------------------------------

def get_rag_service(request: Request):
    """Return the shared RAG service initialized by main.py."""

    service = getattr(request.app.state, "rag_service", None)

    if service is None:
        raise RuntimeError(
            "RAG service is not initialized. "
            "Check application startup and dependency initialization."
        )

    return service

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@router.get(
    "/health",
    response_model=RAGHealthResponse,
    summary="Check RAG API health",
)
def health_check() -> RAGHealthResponse:
    return RAGHealthResponse(
        status="healthy",
        service="corrective-adaptive-rag",
    )

@router.post("/ingest", response_model=IngestResponse, summary="Upload and ingest one document")
async def ingest(
    file: UploadFile = File(..., description="Browse and select one PDF or TXT file"),
    rebuild: bool = Form(False, description="Set to true to rebuild the vector store after ingestion"),
    service: DocumentsService = Depends(documents_service),
)-> IngestResponse:
    """Save one uploaded PDF/TXT file, then pass its filename to the ingestion service."""
    try:
        original_name = Path(file.filename or "").name
        suffix = Path(original_name).suffix.lower()

        if not original_name or suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files are supported.",
            )

        content = await file.read(MAX_FILE_SIZE_BYTES + 1)
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The uploaded file exceeds the 25 MB limit.",
            )
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        documents_dir = settings.resolved_documents_dir
        documents_dir.mkdir(parents=True, exist_ok=True)
        destination = documents_dir / original_name
        destination.write_bytes(content)

        # Pass a list of saved filenames, never the UploadFile object.
        return service.ingest_files([original_name], rebuild=rebuild)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        ) from exc
    finally:
        await file.close()

@router.post(
    "/ask",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question using Corrective or Adaptive RAG",
)
def ask_question(
    payload: RAGQueryRequest,
    request: Request,
    service: RAGService = Depends(get_rag_service),
) -> RAGQueryResponse:
    request_id = payload.request_id or str(uuid4())

    # Attach correlation ID to request state for middleware/logging.
    request.state.request_id = request_id

    logger.info(
        "RAG request received | request_id=%s | top_k=%d "
        "| corrective=%s",
        request_id,
        payload.top_k,
        payload.use_corrective_rag,
    )

    try:
        # The service should invoke the graph and normalize its result.
        result: dict[str, Any] = service.ask_question(
            question=payload.question.strip(),
            request_id=request_id,
            top_k=payload.top_k,
            use_corrective_rag=payload.use_corrective_rag,
        )

        if not isinstance(result, dict):
            raise TypeError(
                "RAG service must return a dictionary"
            )

        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            logger.error(
                "RAG returned an empty or invalid answer | request_id=%s",
                request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="RAG pipeline returned an invalid answer.",
            )

        raw_sources = result.get("sources", [])
        if not isinstance(raw_sources, list):
            raw_sources = []

        sources = []
        for item in raw_sources:
            if isinstance(item, dict):
                sources.append(RAGSource(**item))

        response = RAGQueryResponse(
            request_id=request_id,
            question=payload.question.strip(),
            answer=answer,
            status=result.get("status", "completed"),
            sources=sources,
            grounded=result.get("grounded"),
            unsupported_claims=result.get(
                "unsupported_claims", []
            ),
            retrieval_strategy=result.get(
                "retrieval_strategy"
            ),
            retry_count=result.get("retry_count", 0),
            execution_time_seconds=result.get(
                "execution_time_seconds"
            ),
        )

        logger.info(
            "RAG request completed | request_id=%s | status=%s "
            "| source_count=%d | retries=%d",
            request_id,
            response.status,
            len(response.sources),
            response.retry_count,
        )

        return response

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "RAG request validation or processing failure | "
            "request_id=%s | error=%s",
            request_id,
            str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The RAG request could not be processed.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "RAG execution failed | request_id=%s",
            request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"RAG processing failed. Reference ID: {request_id}"
            ),
        ) from exc
