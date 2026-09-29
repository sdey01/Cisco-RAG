"""LLM-assisted two-route intent classification with a conservative fallback."""

import re

from .prompts import CLASSIFIER_PROMPT


def parse_route(value: str) -> str:
    match = re.search(r"\b(factual|broad)\b", value.lower())
    return match.group(1) if match else "broad"


def classify(question: str, llm) -> str:
    try:
        response = llm.invoke(CLASSIFIER_PROMPT.format(question=question))
        content = getattr(response, "content", response)
        return parse_route(str(content))
    except Exception:
        return "broad"
