import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1200")),
        # OpenRouter can return provider usage/cost in the completion response.
        extra_body={"usage": {"include": True}},
    )

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
PROTOCOL_DOCUMENT_PATH = os.getenv("PROTOCOL_DOCUMENT_PATH")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

APP_NAME = os.getenv("APP_NAME", "Autonomous Clinical Research Agent")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")