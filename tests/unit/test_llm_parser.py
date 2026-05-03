from __future__ import annotations

from llm4rec.prompts.parsers import (
    parse_candidate_normalized_response,
    parse_generation_response,
    parse_llm_json,
    parse_rerank_response,
    parse_yes_no_response,
)


def test_parser_accepts_plain_fenced_and_surrounded_json() -> None:
    assert parse_llm_json('{"confidence": 0.5}').parse_success
    assert parse_llm_json('```json\n{"confidence": 0.5}\n```').parse_success
    assert parse_llm_json('prefix {"confidence": 0.5} suffix').parse_success


def test_parser_rejects_malformed_and_out_of_range_confidence() -> None:
    assert not parse_llm_json("not json").parse_success
    assert not parse_llm_json('{"confidence": 1.2}').parse_success
    assert not parse_llm_json('{"confidence": "0.5"}').parse_success
    assert not parse_llm_json('{"confidence": true}').parse_success


def test_typed_parsers_validate_expected_shapes() -> None:
    assert parse_generation_response('{"recommendation": "Alpha Movie", "confidence": 0.7}').parse_success
    assert parse_rerank_response('{"ranked_items": [{"title": "Alpha Movie", "confidence": 0.7}]}').parse_success
    assert parse_yes_no_response('{"answer": "yes", "confidence": 0.7}').parse_success
    assert parse_candidate_normalized_response(
        '{"options": [{"title": "Alpha Movie", "confidence": 0.7}], "normalized": true}'
    ).parse_success


def test_rerank_parser_recovers_closed_ranked_items_from_truncated_reason() -> None:
    parsed = parse_rerank_response(
        '{"ranked_items": ['
        '{"title": "Alpha Movie", "confidence": 0.9},'
        '{"title": "Beta Movie", "confidence": 0.7}'
        '], "reason": "provider output was truncated'
    )
    assert parsed.parse_success
    assert parsed.metadata["partial_recovery"] is True
    assert [row["title"] for row in parsed.data["ranked_items"]] == ["Alpha Movie", "Beta Movie"]


def test_typed_parsers_reject_wrong_or_empty_fields_and_keep_raw_output() -> None:
    malformed = parse_generation_response("prefix not-json")
    assert not malformed.parse_success
    assert malformed.raw_output == "prefix not-json"

    wrong_generation = parse_generation_response('{"title": "Alpha Movie", "confidence": 0.7}')
    assert not wrong_generation.parse_success
    assert "recommendation" in str(wrong_generation.error)

    empty_generation = parse_generation_response('{"recommendation": " ", "confidence": 0.7}')
    assert not empty_generation.parse_success

    empty_rerank = parse_rerank_response('{"ranked_items": []}')
    assert not empty_rerank.parse_success

    wrong_yes_no = parse_yes_no_response('{"answer": "maybe", "confidence": 0.5}')
    assert not wrong_yes_no.parse_success

    empty_options = parse_candidate_normalized_response('{"options": []}')
    assert not empty_options.parse_success
