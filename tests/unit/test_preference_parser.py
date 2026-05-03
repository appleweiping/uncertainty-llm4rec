"""Preference JSON parser tests."""

from __future__ import annotations

from llm4rec.prompts.preference_parser import parse_listwise_response, parse_pairwise_response


def test_parse_listwise_fenced_json():
    raw = """Here is JSON:
```json
{"ranking": [{"label": "A", "score": 0.9, "confidence": 0.8, "reason": "x"}], "uncertainty": {"overall_confidence": 0.7}}
```"""
    out = parse_listwise_response(raw, label_to_item_id={"A": "i1", "B": "i2"})
    assert out["ok"] is True
    assert out["ranking"][0]["item_id"] == "i1"


def test_parse_listwise_sanitizes_invalid_backslash_apostrophe():
    raw = r'{"ranking": [{"label": "A", "score": 0.9, "confidence": 0.8, "reason": "it\'s ok"}], "uncertainty": {}}'
    out = parse_listwise_response(raw, label_to_item_id={"A": "i1"})
    assert out["ok"] is True


def test_parse_listwise_rejects_unknown_label():
    raw = '{"ranking": [{"label": "Z", "score": 1, "confidence": 1, "reason": ""}]}'
    out = parse_listwise_response(raw, label_to_item_id={"A": "i1"})
    assert out["ok"] is False


def test_parse_pairwise():
    raw = '{"pairs": [{"left": "A", "right": "B", "winner": "B", "confidence": 0.6, "reason": ""}], "overall_confidence": 0.5}'
    out = parse_pairwise_response(raw, label_to_item_id={"A": "a", "B": "b"})
    assert out["ok"] is True
    assert out["pairs"][0]["winner_item_id"] == "b"
