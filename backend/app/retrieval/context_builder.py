
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceItem:
    """A retrieved evidence passage with a stable citation ID."""

    evidence_id: str
    content: str
    source: str
    page: str
    section: str
    chunk_id: str
    retrieval_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "retrieval_score": self.retrieval_score,
        }

    def format_for_prompt(self) -> str:
        return (
            f"[{self.evidence_id}]\n"
            f"Source: {self.source}\n"
            f"Page: {self.page}\n"
            f"Section: {self.section}\n"
            f"Evidence:\n{self.content}"
        )


@dataclass(frozen=True)
class BuiltContext:
    """Context and the exact evidence mapping used to build it."""

    context_text: str
    evidence: tuple[EvidenceItem, ...]
    excluded_count: int
    total_context_characters: int

    def evidence_map(self) -> dict[str, EvidenceItem]:
        return {
            item.evidence_id: item
            for item in self.evidence
        }

    def to_sources(self) -> list[dict[str, Any]]:
        return [
            item.to_dict()
            for item in self.evidence
        ]


class ContextBuilder:
    """
    Construct bounded, citation-ready context from retrieved documents.

    Responsibilities:
    - Validate retrieved Document objects.
    - Deduplicate identical chunks.
    - Preserve source, page, section, and chunk metadata.
    - Enforce a context character budget.
    - Assign evidence IDs for citations.
    - Produce a reusable evidence mapping for answer validation.

    This component does not grade relevance or generate answers.
    """

    def __init__(
        self,
        max_context_chars: int = 30000,
        max_chunk_chars: int = 8000,
        max_documents: int = 12,
        min_chunk_chars: int = 1,
        include_scores: bool = False,
    ) -> None:
        if max_context_chars < 1000:
            raise ValueError(
                "max_context_chars must be at least 1000."
            )

        if max_chunk_chars < 1:
            raise ValueError(
                "max_chunk_chars must be positive."
            )

        if max_documents < 1:
            raise ValueError(
                "max_documents must be positive."
            )

        if min_chunk_chars < 1:
            raise ValueError(
                "min_chunk_chars must be positive."
            )

        self.max_context_chars = max_context_chars
        self.max_chunk_chars = max_chunk_chars
        self.max_documents = max_documents
        self.min_chunk_chars = min_chunk_chars
        self.include_scores = include_scores

    def build(
        self,
        documents: Sequence[Document],
    ) -> BuiltContext:
        """
        Build context while preserving document ranking order.

        Documents should normally be passed in their post-grading,
        post-reranking order.
        """
        if isinstance(documents, (str, bytes)):
            raise TypeError(
                "documents must be a sequence of Document objects."
            )

        if not isinstance(documents, Sequence):
            raise TypeError(
                "documents must be a sequence."
            )

        if not all(
            isinstance(doc, Document)
            for doc in documents
        ):
            raise TypeError(
                "Every item must be a LangChain Document."
            )

        seen: set[str] = set()
        selected: list[EvidenceItem] = []
        used_chars = 0
        excluded_count = 0

        for document in documents:
            if len(selected) >= self.max_documents:
                excluded_count += 1
                continue

            content = (document.page_content or "").strip()

            if len(content) < self.min_chunk_chars:
                excluded_count += 1
                continue

            identity = self._document_identity(document)

            if identity in seen:
                excluded_count += 1
                continue

            seen.add(identity)

            # Avoid splitting evidence silently. If a chunk exceeds
            # the per-chunk limit, exclude it and record the exclusion.
            if len(content) > self.max_chunk_chars:
                logger.warning(
                    "Context chunk excluded because it exceeds "
                    "max_chunk_chars | chunk_id=%s | chars=%d",
                    self._chunk_id(document),
                    len(content),
                )
                excluded_count += 1
                continue

            item = self._create_evidence_item(
                document=document,
                index=len(selected) + 1,
                content=content,
            )

            formatted = item.format_for_prompt()
            separator_chars = 2 if selected else 0
            required_chars = len(formatted) + separator_chars

            if used_chars + required_chars > self.max_context_chars:
                excluded_count += 1
                continue

            selected.append(item)
            used_chars += required_chars

        context_text = "\n\n".join(
            item.format_for_prompt()
            for item in selected
        )

        logger.info(
            "Context built | input_documents=%d | evidence=%d "
            "| excluded=%d | context_chars=%d",
            len(documents),
            len(selected),
            excluded_count,
            len(context_text),
        )

        return BuiltContext(
            context_text=context_text,
            evidence=tuple(selected),
            excluded_count=excluded_count,
            total_context_characters=len(context_text),
        )

    def _create_evidence_item(
        self,
        document: Document,
        index: int,
        content: str,
    ) -> EvidenceItem:
        metadata = document.metadata or {}

        raw_score = metadata.get("retrieval_score")
        score = self._safe_float(raw_score)

        return EvidenceItem(
            evidence_id=f"E{index}",
            content=content,
            source=self._safe_string(
                metadata.get("source")
                or metadata.get("filename")
                or metadata.get("document_name"),
                "unknown",
            ),
            page=self._safe_string(
                metadata.get("page"),
                "unknown",
            ),
            section=self._safe_string(
                metadata.get("section")
                or metadata.get("heading"),
                "unknown",
            ),
            chunk_id=self._chunk_id(document),
            retrieval_score=score if self.include_scores else None,
        )

    @staticmethod
    def _document_identity(document: Document) -> str:
        """
        Deduplicate exact content while retaining distinct passages
        with different content.
        """
        metadata = document.metadata or {}
        content_hash = hashlib.sha256(
            (document.page_content or "").strip().encode("utf-8")
        ).hexdigest()

        source = metadata.get("source", "")
        page = metadata.get("page", "")
        chunk_id = metadata.get("chunk_id")

        if chunk_id:
            return f"chunk:{source}:{chunk_id}:{content_hash}"

        return f"content:{source}:{page}:{content_hash}"

    @staticmethod
    def _chunk_id(document: Document) -> str:
        metadata = document.metadata or {}

        stable_id = (
            metadata.get("chunk_id")
            or metadata.get("document_chunk_id")
            or document.id
        )

        if stable_id:
            return str(stable_id)

        return hashlib.sha256(
            (document.page_content or "").encode("utf-8")
        ).hexdigest()[:24]

    @staticmethod
    def _safe_string(
        value: Any,
        default: str,
    ) -> str:
        if value is None:
            return default

        if isinstance(value, (str, int, float, bool)):
            return str(value)[:500]

        return default

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None

        try:
            result = float(value)

            if result != result:
                return None

            return result
        except (TypeError, ValueError, OverflowError):
            return None
