"""Query expansion parsing and validation."""

from .prompts import EXPANSION_PROMPT


def parse_expansions(value: str, original: str, limit: int) -> list[str]:
    seen = {original.strip().lower()}
    expansions = []
    for line in value.splitlines():
        query = line.strip().lstrip("-*").strip()
        if query and query.lower() not in seen and len(expansions) < limit:
            expansions.append(query)
            seen.add(query.lower())
    return expansions


def expand_query(question: str, llm, limit: int) -> list[str]:
    try:
        response = llm.invoke(EXPANSION_PROMPT.format(question=question, limit=limit))
        content = getattr(response, "content", response)
        return parse_expansions(str(content), question, limit)
    except Exception:
        return []
