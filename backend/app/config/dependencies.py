from functools import lru_cache
from app.implementations.ingestion_implementation import IngestionImplementation

from app.services.documents_service import DocumentsService

@lru_cache

@lru_cache
def ingestion_impl(): return IngestionImplementation()
@lru_cache

def documents_service(): return DocumentsService(ingestion_impl())
