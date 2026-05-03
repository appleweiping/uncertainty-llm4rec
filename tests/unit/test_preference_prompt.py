"""Preference prompt sanity (no raw global IDs in body when using anonymous labels)."""

from __future__ import annotations

from llm4rec.prompts.preference_templates import build_listwise_preference_prompt


def test_listwise_prompt_maps_labels():
    panel = {
        "user_id": "u",
        "example_id": "1:1",
        "panel_items": [
            {"item_id": "99", "title": "Alpha", "genre": "g1", "fallback_rank": 1, "fallback_score": 1.0, "popularity_bucket": "head", "source": ["fallback_top"]},
            {"item_id": "100", "title": "Beta", "genre": "g2", "fallback_rank": 2, "fallback_score": 0.5, "popularity_bucket": "mid", "source": ["fallback_top"]},
        ],
        "metadata": {"candidate_size": 500, "panel_size": 2, "seed": 13},
    }
    prompt, label_map = build_listwise_preference_prompt(history_titles=["Old"], panel=panel, show_fallback_rank=False)
    assert "A:" in prompt or "- A:" in prompt
    assert label_map["A"] == "99" and label_map["B"] == "100"
    assert "99" not in prompt and "100" not in prompt
