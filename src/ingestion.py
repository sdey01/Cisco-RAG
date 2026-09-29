"""PDF loading and configurable document chunking."""

import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .config import settings

logger = logging.getLogger(__name__)


def load_and_split_pdf(pdf_path: Path | None = None) -> list[Document]:
    path = pdf_path or settings.source_pdf
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    pages = PyPDFLoader(str(path)).load()
    for page in pages:
        page.metadata["source_name"] = path.name
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{path.name}:{index}"
    logger.info("Loaded %s pages and created %s chunks", len(pages), len(chunks))
    return chunks
