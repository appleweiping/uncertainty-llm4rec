from __future__ import annotations

import math

from llm4rec.metrics.diversity import category_entropy_at_k, intra_list_diversity, item_category_map


def test_category_based_intra_list_diversity() -> None:
    categories = {"i1": "A", "i2": "B", "i3": "A"}
    predictions = [{"predicted_items": ["i1", "i2", "i3"]}]
    assert intra_list_diversity(predictions, item_categories=categories, k=3) == 2 / 3


def test_category_entropy_uses_metadata_fallback() -> None:
    catalog = [{"item_id": "i1", "category": "A"}, {"item_id": "i2", "category": "B"}]
    categories = item_category_map(catalog)
    entropy = category_entropy_at_k([{"predicted_items": ["i1", "i2"]}], item_categories=categories, k=2)
    assert math.isclose(entropy, 1.0)
