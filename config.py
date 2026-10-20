"""Central configuration. All values come from environment variables / .env."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # reads .env from the working directory if present


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    google_api_key: str = ""
    chat_model: str = "gemini-3.5-flash"
    embedding_model: str = "gemini-embedding-001"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    # Minimum cosine similarity (1 - cosine distance) the BEST chunk must reach
    # before we call the LLM. Tune per embedding model - see README "Limitations".
    min_similarity: float = 0.45
    max_pdf_mb: int = 20
    max_pages: int = 200
    max_chunks: int = 1500  # hard cap on embedding calls per document
    max_retries: int = 4
    retry_base_delay: float = 1.0  # seconds; doubles each retry (1, 2, 4, 8)
    embed_batch_size: int = 50


def load_settings() -> Settings:
    key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if key.lower() in {"your_gemini_api_key_here", "changeme"}:
        key = ""  # placeholder from .env.example counts as missing
    d = Settings()
    return Settings(
        google_api_key=key,
        chat_model=os.getenv("CHAT_MODEL", d.chat_model),
        embedding_model=os.getenv("EMBEDDING_MODEL", d.embedding_model),
        chunk_size=_int("CHUNK_SIZE", d.chunk_size),
        chunk_overlap=_int("CHUNK_OVERLAP", d.chunk_overlap),
        top_k=_int("TOP_K", d.top_k),
        min_similarity=_float("MIN_SIMILARITY", d.min_similarity),
        max_pdf_mb=_int("MAX_PDF_MB", d.max_pdf_mb),
        max_pages=_int("MAX_PAGES", d.max_pages),
        max_chunks=_int("MAX_CHUNKS", d.max_chunks),
        max_retries=_int("MAX_RETRIES", d.max_retries),
        retry_base_delay=_float("RETRY_BASE_DELAY", d.retry_base_delay),
        embed_batch_size=_int("EMBED_BATCH_SIZE", d.embed_batch_size),
    )
