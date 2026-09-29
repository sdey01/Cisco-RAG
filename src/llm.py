"""Groq LLM factory."""

from functools import lru_cache

from langchain_groq import ChatGroq

from .config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    settings.validate(require_api_key=True)
    return ChatGroq(model=settings.groq_model, temperature=0)
