from langchain_core.documents import Document

from src.query_expansion import parse_expansions
from src.retriever import deduplicate
from src.router import parse_route


def test_route_parser_defaults_to_broad():
    assert parse_route("factual") == "factual"
    assert parse_route("unknown response") == "broad"


def test_expansion_parser_limits_and_deduplicates():
    result = parse_expansions("- interface down\ninterface down\n- link failure\n- causes", "interface down", 2)
    assert result == ["link failure", "causes"]


def test_deduplicate_uses_chunk_id():
    first = Document(page_content="same", metadata={"chunk_id": "a"})
    duplicate = Document(page_content="same again", metadata={"chunk_id": "a"})
    assert deduplicate([first, duplicate]) == [first]
