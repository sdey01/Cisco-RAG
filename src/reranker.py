"""Lightweight embedding-based reranking for broad questions."""

import numpy as np


def rerank(question: str, documents, embeddings, top_k: int):
    if not documents:
        return []
    query_vector = np.asarray(embeddings.embed_query(question), dtype=float)
    document_vectors = np.asarray(
        embeddings.embed_documents([document.page_content for document in documents]), dtype=float
    )
    query_norm = np.linalg.norm(query_vector)
    scores = document_vectors @ query_vector / (np.linalg.norm(document_vectors, axis=1) * query_norm)
    ranked = sorted(zip(scores.tolist(), documents), key=lambda item: item[0], reverse=True)
    for score, document in ranked:
        document.metadata["rerank_score"] = round(float(score), 4)
    return [document for _, document in ranked[:top_k]]
