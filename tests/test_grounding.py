from __future__ import annotations

from storyflow.grounding import TitleGrounder, ground_title
from storyflow.schemas import GroundingStatus, ItemCatalogRecord
from tests.fixtures.synthetic_observation import synthetic_catalog


def test_ground_title_exact_match() -> None:
    result = ground_title("The Matrix", synthetic_catalog(), prediction_id="p1")

    assert result.item_id == "item-head"
    assert result.status == GroundingStatus.EXACT
    assert result.score == 1.0
    assert result.is_grounded


def test_ground_title_normalized_exact_match() -> None:
    result = ground_title("matrix", synthetic_catalog(), prediction_id="p1")

    assert result.item_id == "item-head"
    assert result.status == GroundingStatus.NORMALIZED_EXACT
    assert result.score == 0.98


def test_ground_title_fuzzy_match() -> None:
    result = ground_title(
        "The Matrx",
        synthetic_catalog(),
        prediction_id="p1",
        fuzzy_threshold=0.80,
    )

    assert result.item_id == "item-head"
    assert result.status == GroundingStatus.FUZZY
    assert 0.80 <= result.score <= 1.0


def test_ground_title_ambiguous_normalized_match() -> None:
    catalog = (
        ItemCatalogRecord("item-1", "The Thing"),
        ItemCatalogRecord("item-2", "Thing"),
    )

    result = ground_title("thing", catalog, prediction_id="p1")

    assert result.item_id is None
    assert result.status == GroundingStatus.AMBIGUOUS
    assert not result.is_grounded


def test_title_grounder_reuses_catalog() -> None:
    grounder = TitleGrounder(synthetic_catalog())

    assert grounder.ground("Arrival", prediction_id="p1").item_id == "item-mid"
