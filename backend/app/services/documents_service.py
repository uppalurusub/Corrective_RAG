from app.schemas.rag_schemas import IngestRequest
from app.implementations.ingestion_implementation import IngestionImplementation


class DocumentsService:
    def __init__(self, impl: IngestionImplementation):
        self.impl = impl

    def ingest(self, request: IngestRequest | None = None):
        """Ingest files already present in DOCUMENTS_DIR from an API request."""
        request = request or IngestRequest()
        return self.impl.ingest(request.file_paths, request.rebuild)

    def ingest_files(self, filenames: list[str], rebuild: bool = False):
        """Ingest uploaded filenames saved under DOCUMENTS_DIR."""
        if not filenames:
            raise ValueError("No uploaded filename was provided.")
        return self.impl.ingest(filenames, rebuild)
