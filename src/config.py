"""Centralized environment-backed application configuration."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    source_pdf: Path = _path(os.getenv("SOURCE_PDF", "data/source/cisco_nexus_9000_troubleshooting.pdf"))
    chroma_persist_directory: Path = _path(os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma"))
    chroma_collection_name: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "cisco_nexus_9000_troubleshooting"
    )
    cache_path: Path = _path(os.getenv("CACHE_PATH", "data/cache/semantic_cache.json"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    factual_top_k: int = int(os.getenv("FACTUAL_TOP_K", "4"))
    broad_top_k: int = int(os.getenv("BROAD_TOP_K", "10"))
    final_context_chunks: int = int(os.getenv("FINAL_CONTEXT_CHUNKS", "5"))
    max_query_expansions: int = int(os.getenv("MAX_QUERY_EXPANSIONS", "3"))
    semantic_cache_threshold: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.85"))

    def validate(self, require_api_key: bool = True) -> None:
        if require_api_key and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to .env before asking a question.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")


settings = Settings()
