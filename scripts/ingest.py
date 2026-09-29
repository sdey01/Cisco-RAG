"""Build the persistent Chroma index from the configured Cisco PDF."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.vectorstore import index_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


if __name__ == "__main__":
    if not settings.source_pdf.exists():
        raise SystemExit(f"PDF not found: {settings.source_pdf}")
    print(f"Loading PDF: {settings.source_pdf}")
    count = index_pdf()
    print(f"Indexing complete. Chunks indexed: {count}")
