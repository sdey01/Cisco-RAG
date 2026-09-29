"""Streamlit UI for the Cisco Nexus 9000 troubleshooting assistant."""

import logging

import streamlit as st

from src.cache import cache_summary
from src.config import settings
from src.pipeline import run
from src.vectorstore import has_index

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Cisco Nexus 9000 Troubleshooting Assistant", page_icon="CN", layout="wide")


def rag_path(result) -> list[str]:
    if result.cache_hit:
        return ["Cache lookup", "Cache hit", "Return cached answer"]
    path = ["Cache lookup", "Cache miss", f"{result.route.title()} route", "Query expansion", "Vector retrieval"]
    if result.reranked:
        path.append("Embedding reranking")
    path.append("Groq answer generation")
    return path


if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.subheader("Workspace")
    if has_index():
        st.success("Chroma index is available")
    else:
        st.warning("No index found. Run `python scripts/ingest.py` first.")
    st.caption(f"Source: {settings.source_pdf.name}")
    st.divider()
    st.subheader("Latest cache status")
    if st.session_state.last_result:
        if st.session_state.last_result.cache_hit:
            st.success("CACHE HIT")
        else:
            st.info("CACHE MISS")
        st.caption("RAG path")
        st.write(" -> ".join(rag_path(st.session_state.last_result)))
    else:
        st.caption("Ask a question to see cache behavior.")
    st.subheader("Question history")
    if st.session_state.history:
        for index, (old_question, old_result) in enumerate(st.session_state.history, start=1):
            st.caption(f"{index}. {old_question}")
            st.caption(f"{old_result.route.upper()} | {'HIT' if old_result.cache_hit else 'MISS'} | {old_result.latency_ms} ms")
    else:
        st.caption("No questions asked in this session.")

left, center, right = st.columns([0.08, 0.62, 0.30])
with center:
    st.title("Cisco Nexus 9000 Troubleshooting Assistant")
    st.caption("Ask questions grounded in the Cisco Nexus 9000 Series Troubleshooting Guide.")
    question = st.text_area("Troubleshooting question", placeholder="How do I troubleshoot a Nexus 9000 interface that is not coming up?", height=120)
    ask = st.button("Ask", type="primary", disabled=not question.strip(), use_container_width=True)

if ask:
    if not has_index():
        st.error("The guide is not indexed yet. Run the ingestion command, then refresh this page.")
    else:
        with st.spinner("Searching the troubleshooting guide..."):
            try:
                result = run(question.strip())
                st.session_state.history.insert(0, (question.strip(), result))
                st.session_state.last_result = result
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

with center:
    if st.session_state.history:
        question_text, result = st.session_state.history[0]
        st.subheader("Current question")
        st.info(question_text)
        st.subheader("Answer")
        st.markdown(result.answer)
        st.caption(f"{'CACHE HIT' if result.cache_hit else 'CACHE MISS'} | {result.latency_ms} ms")
        st.subheader("RAG route")
        st.caption(f"Triggered pipeline: {result.route.upper()}")
        st.code(" -> ".join(rag_path(result)), language="text")
        st.subheader("Query expansion")
        if result.expanded_queries:
            for expanded_query in result.expanded_queries:
                st.write(f"- {expanded_query}")
        else:
            st.caption("No expansions were needed for this response, likely because it came from cache.")
        st.subheader("Sources")
        if result.sources:
            for index, source in enumerate(result.sources, start=1):
                page = f"Page {source['page']}" if source.get("page") else "Page unavailable"
                st.write(f"{index}. {source['document']} - {page}")
        else:
            st.info("No source chunks were available.")
        with st.expander("Retrieval diagnostics"):
            st.json({
                "route": result.route,
                "rag_path": rag_path(result),
                "cache": "HIT" if result.cache_hit else "MISS",
                "latency_ms": result.latency_ms,
                "retrieved_chunks": result.retrieved_chunks,
                "final_context_chunks": result.final_chunks,
                "reranked": result.reranked,
                "query_expansions": len(result.expanded_queries),
            })

with right:
    st.subheader("Semantic cache")
    st.caption(f"Threshold: {settings.semantic_cache_threshold:.2f}")
    entries = cache_summary(settings.cache_path)
    st.metric("Cached questions", len(entries))
    if entries:
        st.dataframe(entries, hide_index=True, use_container_width=True)
    else:
        st.caption("No cached answers yet. A cache entry appears after a successful cache miss.")
