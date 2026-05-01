from __future__ import annotations

from llm4rec.grounding.title import ground_title, normalize_title, token_overlap


CATALOG = [
    {"item_id": "i1", "title": "Alpha Movie"},
    {"item_id": "i2", "title": "Beta Movie"},
]


def test_title_grounding_exact_and_normalized() -> None:
    exact = ground_title("alpha movie", CATALOG)
    assert exact.grounding_success
    assert exact.grounded_item_id == "i1"
    normalized = ground_title("Alpha: Movie!", CATALOG)
    assert normalized.grounding_success
    assert normalized.grounding_method == "normalized"


def test_title_grounding_fuzzy_and_failed() -> None:
    fuzzy = ground_title("Alpha", CATALOG)
    assert fuzzy.grounding_success
    assert fuzzy.grounding_method == "token_overlap"
    failed = ground_title("Unknown Title", CATALOG)
    assert not failed.grounding_success
    empty = ground_title("", CATALOG)
    assert not empty.grounding_success
    assert empty.generated_title == ""
    assert normalize_title("Alpha: Movie!") == "alpha movie"
    assert token_overlap("Alpha Movie", "Alpha") == 0.5


def test_title_grounding_ambiguous_match_uses_deterministic_item_id_tiebreak() -> None:
    ambiguous_catalog = [
        {"item_id": "i2", "title": "Same Movie"},
        {"item_id": "i1", "title": "Same Movie"},
    ]
    result = ground_title("same movie", ambiguous_catalog)
    assert result.grounding_success
    assert result.grounded_item_id == "i1"
