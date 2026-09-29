"""End-to-end cache, routing, retrieval, reranking, and answer generation."""

from dataclasses import asdict, dataclass
import logging
from time import perf_counter

from .cache import SemanticCache
from .config import settings
from .embeddings import get_embeddings
from .llm import get_llm
from .prompts import ANSWER_PROMPT
from .query_expansion import expand_query
from .reranker import rerank
from .retriever import retrieve
from .router import classify
from .vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    answer: str
    sources: list[dict]
    route: str
    cache_hit: bool
    expanded_queries: list[str]
    retrieved_chunks: int
    final_chunks: int
    reranked: bool
    latency_ms: float

    def as_dict(self):
        return asdict(self)


def _sources(documents) -> list[dict]:
    return [
        {
            "document": document.metadata.get("source_name", document.metadata.get("source", "source")),
            "page": int(document.metadata["page"]) + 1 if "page" in document.metadata else None,
            "chunk_id": document.metadata.get("chunk_id"),
        }
        for document in documents
    ]


def run(question: str) -> PipelineResult:
    started = perf_counter()
    settings.validate(require_api_key=True)
    embeddings = get_embeddings()
    cache = SemanticCache(settings.cache_path, settings.semantic_cache_threshold, embeddings)
    cached = cache.lookup(question)
    if cached:
        return PipelineResult(cached["answer"], cached.get("sources", []), cached.get("route", "broad"), True, [], 0, len(cached.get("sources", [])), False, round((perf_counter() - started) * 1000, 1))

    llm = get_llm()
    route = classify(question, llm)
    expansions = expand_query(question, llm, settings.max_query_expansions)
    queries = [question, *expansions]
    top_k = settings.factual_top_k if route == "factual" else settings.broad_top_k
    candidates = retrieve(get_vectorstore(), queries, top_k)
    if not candidates:
        answer = "I could not find relevant information in the indexed troubleshooting guide."
        return PipelineResult(answer, [], route, False, expansions, 0, 0, route == "broad", round((perf_counter() - started) * 1000, 1))
    final_documents = (
        rerank(question, candidates, embeddings, settings.final_context_chunks)
        if route == "broad"
        else candidates[: settings.final_context_chunks]
    )
    source_list = _sources(final_documents)
    context = "\n\n".join(
        f"[{source['document']} - page {source['page'] or 'unknown'}]\n{document.page_content}"
        for document, source in zip(final_documents, source_list)
    )
    try:
        response = llm.invoke(ANSWER_PROMPT.format(question=question, context=context))
        answer = str(getattr(response, "content", response))
    except Exception as exc:
        logger.exception("LLM answer generation failed")
        answer = f"The answer service could not complete the request: {exc}"
    cache.put(question, answer, source_list, route)
    return PipelineResult(answer, source_list, route, False, expansions, len(candidates), len(final_documents), route == "broad", round((perf_counter() - started) * 1000, 1))
