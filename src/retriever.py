"""Multi-query retrieval and stable chunk deduplication."""

from langchain_core.documents import Document


def deduplicate(documents: list[Document]) -> list[Document]:
    unique = []
    seen = set()
    for document in documents:
        key = document.metadata.get("chunk_id") or (
            document.metadata.get("source"), document.metadata.get("page"), document.page_content
        )
        if key not in seen:
            unique.append(document)
            seen.add(key)
    return unique


def retrieve(store, queries: list[str], top_k: int) -> list[Document]:
    documents = []
    for query in queries:
        documents.extend(store.similarity_search(query, k=top_k))
    return deduplicate(documents)
