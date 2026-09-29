"""Persistent Chroma index access."""

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from .config import settings
from .embeddings import get_embeddings
from .ingestion import load_and_split_pdf


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_persist_directory),
    )


def index_pdf(pdf_path: Path | None = None) -> int:
    chunks = load_and_split_pdf(pdf_path)
    store = get_vectorstore()
    store.add_documents(chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks])
    return len(chunks)


def has_index() -> bool:
    try:
        return get_vectorstore()._collection.count() > 0
    except Exception:
        return False
