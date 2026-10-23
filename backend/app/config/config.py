import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
    app_name: str = "End-to-End RAG API"
    log_level: str = "INFO"
    openrouter_api_key: str = ""
    fallback_model_1: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key_alt: str = ""
    openrouter_model: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    documents_dir: str = "./data/documents"
    vector_db_path: str = "./data/chroma"
    chroma_collection: str = "rag_documents"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    llm_max_tokens: int = 1200
    llm_temperature: float = 0
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-fastapi"
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "rag-evaluation"

    @property
    def resolved_api_key(self):
        return os.getenv("OPENROUTER_API_KEY") or self.openrouter_api_key or os.getenv("openrouter_api_key", "")

    @property
    def resolved_model(self):
        return os.getenv("OPENROUTER_MODEL") or self.openrouter_model or os.getenv("fallback_model_1") or self.fallback_model_1

    @property
    def resolved_base_url(self):
        return os.getenv("OPENROUTER_BASE_URL") or os.getenv("base_url") or self.openrouter_base_url or self.base_url

    @property
    def resolved_documents_dir(self):
        p = Path(self.documents_dir)
        return p if p.is_absolute() else (ROOT_DIR / p).resolve()

    @property
    def resolved_vector_db_path(self):
        p = Path(self.vector_db_path)
        return p if p.is_absolute() else (ROOT_DIR / p).resolve()

settings = Settings()

# LangChain/LangSmith instrumentation reads these environment variables directly.
if settings.langsmith_tracing:
    os.environ["LANGSMITH_TRACING"] = "true"
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
