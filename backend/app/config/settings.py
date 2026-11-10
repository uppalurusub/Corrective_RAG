"""Central application configuration, LLM client, usage tracking, and logging.

Use this module as the single settings source throughout the application:
    from app.config.settings import settings, LLMUsageTracker

Environment variables in .env override the defaults below.
"""

import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

# app/config/settings.py -> backend/ (assuming backend/app/config layout)
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "End-to-End RAG API"
    log_level: str = "INFO"
    environment: str = "development"

    # OpenRouter / LLM
    openrouter_api_key: str = ""
    openrouter_api_key_alt: str = ""
    openrouter_model: str = ""
    fallback_model_1: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    base_url: str = "https://openrouter.ai/api/v1"
    llm_max_tokens: int = 1200
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 2

    # OpenRouter pricing: USD per 1 million tokens for the configured model.
    llm_input_price_per_1m_usd: float = Field(
        default=0.10, validation_alias="LLM_INPUT_PRICE_PER_1M_USD"
    )
    llm_output_price_per_1m_usd: float = Field(
        default=0.32, validation_alias="LLM_OUTPUT_PRICE_PER_1M_USD"
    )

    # Embeddings and retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    documents_dir: str = "./data/documents"
    vector_db_path: str = "./data/chroma"
    chroma_collection: str = "rag_documents"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-fastapi"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # MLflow
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "rag-evaluation"

    # Logging
    log_dir: str = "./logs"

    @property
    def resolved_api_key(self) -> str:
        return self.openrouter_api_key or self.openrouter_api_key_alt

    @property
    def resolved_model(self) -> str:
        return self.openrouter_model or self.fallback_model_1

    @property
    def resolved_base_url(self) -> str:
        # openrouter_base_url is the canonical setting; base_url is retained
        # for compatibility with the former config.py.
        return (
            os.getenv("OPENROUTER_BASE_URL")
            or os.getenv("BASE_URL")
            or self.openrouter_base_url
            or self.base_url
        )

    @property
    def resolved_documents_dir(self) -> Path:
        path = Path(self.documents_dir).expanduser()
        return path if path.is_absolute() else (BACKEND_DIR / path).resolve()

    @property
    def resolved_vector_db_path(self) -> Path:
        path = Path(self.vector_db_path).expanduser()
        return path if path.is_absolute() else (BACKEND_DIR / path).resolve()

    @property
    def resolved_log_dir(self) -> Path:
        path = Path(self.log_dir).expanduser()
        return path if path.is_absolute() else (BACKEND_DIR / path).resolve()

    # Compatibility attributes used by the previous settings.py.
    @property
    def model(self) -> str:
        return self.resolved_model

    @property
    def protocol_document_path(self) -> str:
        return str(self.resolved_documents_dir)

    def get_logger(self, name: str = __name__):
        return logger.bind(module=name)

    def get_model_client(self):
        """Create a LangChain ChatOpenAI client configured for OpenRouter."""
        if not self.resolved_model:
            raise ValueError("OPENROUTER_MODEL (or FALLBACK_MODEL_1) is not configured")
        if not self.resolved_api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")

        # Import lazily so configuration and non-LLM modules can load without
        # initializing the provider client.
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            model=self.resolved_model,
            api_key=self.resolved_api_key,
            base_url=self.resolved_base_url,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            extra_body={"usage": {"include": True}},
            stream_usage=True,
            max_retries=self.llm_max_retries,
            timeout=self.llm_timeout_seconds,
        )
        logger.success("LangChain ChatOpenAI initialized | model={}", self.resolved_model)
        return client


settings = Settings()

# Export legacy module-level names for any remaining imports. Prefer settings.*.
MODEL = settings.resolved_model
OPENROUTER_API_KEY = settings.resolved_api_key
BASE_URL = settings.resolved_base_url
INPUT_PRICE_PER_1M = settings.llm_input_price_per_1m_usd
OUTPUT_PRICE_PER_1M = settings.llm_output_price_per_1m_usd
EMBEDDING_MODEL = settings.embedding_model
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
TOP_K = settings.top_k
VECTOR_DB_PATH = str(settings.resolved_vector_db_path)
PROTOCOL_DOCUMENT_PATH = str(settings.resolved_documents_dir)


class LLMUsageTracker(BaseCallbackHandler):
    """Thread-safe accumulator for token usage reported by LangChain callbacks.

    Attach an instance to the actual model invocation, for example:
        tracker = LLMUsageTracker(model=settings.resolved_model)
        response = llm.invoke(messages, config={"callbacks": [tracker]})

    The callback extracts provider usage when the callback response exposes it.
    Costs are estimates based on the configured per-million-token rates.
    """

    def __init__(
        self,
        model: str,
        input_price_per_1m: Optional[float] = None,
        output_price_per_1m: Optional[float] = None,
    ):
        super().__init__()
        self.model = model
        self.input_price_per_1m = (
            settings.llm_input_price_per_1m_usd
            if input_price_per_1m is None else float(input_price_per_1m)
        )
        self.output_price_per_1m = (
            settings.llm_output_price_per_1m_usd
            if output_price_per_1m is None else float(output_price_per_1m)
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_call_count = 0
        self.usage_reported_call_count = 0
        self._lock = threading.Lock()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage = self._extract_usage(response)
        with self._lock:
            self.llm_call_count += 1
            if usage is None:
                logger.warning("LLM invocation completed without reported token usage | model={}", self.model)
                return

            input_tokens, output_tokens = usage
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.usage_reported_call_count += 1
            logger.info(
                "LLM usage recorded | model={} | input_tokens={} | output_tokens={} | total_tokens={}",
                self.model, input_tokens, output_tokens, input_tokens + output_tokens,
            )

    @staticmethod
    def _extract_usage(response: Any) -> Optional[tuple[int, int]]:
        if response is None:
            return None

        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            parsed = LLMUsageTracker._parse_usage_dict(usage)
            if parsed is not None:
                return parsed

        for generation_group in getattr(response, "generations", []) or []:
            for generation in generation_group:
                message = getattr(generation, "message", None)
                if message is None:
                    continue

                usage_metadata = getattr(message, "usage_metadata", None)
                parsed = LLMUsageTracker._parse_usage_dict(usage_metadata)
                if parsed is not None:
                    return parsed

                response_metadata = getattr(message, "response_metadata", None) or {}
                if isinstance(response_metadata, dict):
                    parsed = LLMUsageTracker._parse_usage_dict(
                        response_metadata.get("token_usage") or response_metadata.get("usage")
                    )
                    if parsed is not None:
                        return parsed
        return None

    @staticmethod
    def _parse_usage_dict(usage: Any) -> Optional[tuple[int, int]]:
        if not isinstance(usage, dict) or not usage:
            return None
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if input_tokens is None or output_tokens is None:
            return None
        return int(input_tokens), int(output_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def input_cost_usd(self) -> float:
        return self.input_tokens * self.input_price_per_1m / 1_000_000

    @property
    def output_cost_usd(self) -> float:
        return self.output_tokens * self.output_price_per_1m / 1_000_000

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "model": self.model,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "input_cost_usd": self.input_cost_usd,
                "output_cost_usd": self.output_cost_usd,
                "total_cost_usd": self.total_cost_usd,
                "llm_call_count": self.llm_call_count,
                "usage_reported_call_count": self.usage_reported_call_count,
                "source": "langchain_callback_usage",
            }


class LoggerManager:
    """Compatibility wrapper around Loguru logging methods."""

    @staticmethod
    def info(message: str, *args: Any) -> None:
        logger.info(message, *args)

    @staticmethod
    def warning(message: str, *args: Any) -> None:
        logger.warning(message, *args)

    @staticmethod
    def error(message: str, *args: Any) -> None:
        logger.error(message, *args)

    @staticmethod
    def debug(message: str, *args: Any) -> None:
        logger.debug(message, *args)

    @staticmethod
    def critical(message: str, *args: Any) -> None:
        logger.critical(message, *args)


def configure_logging() -> None:
    """Configure console, application, and error log sinks once at startup."""
    log_dir = settings.resolved_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        str(log_dir / "application.log"),
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        level=settings.log_level.upper(),
        enqueue=True,
        backtrace=True,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    logger.add(
        str(log_dir / "error.log"),
        rotation="5 MB",
        retention="60 days",
        compression="zip",
        level="ERROR",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )


def configure_langsmith_environment() -> None:
    """Expose LangSmith settings to LangChain tracing via environment variables."""
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        if settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    else:
        os.environ.setdefault("LANGSMITH_TRACING", "false")


configure_logging()
configure_langsmith_environment()
