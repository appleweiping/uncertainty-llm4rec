from __future__ import annotations

import pytest

from storyflow.schemas import (
    ConfidenceRecord,
    GenerativePredictionRecord,
    GroundedPredictionRecord,
    GroundingStatus,
    InteractionRecord,
    ItemCatalogRecord,
    ObservationExampleRecord,
    UserSequenceRecord,
)
from tests.fixtures.synthetic_observation import (
    synthetic_observation_example,
    synthetic_sequence,
)


def test_item_catalog_record_validates_required_fields() -> None:
    item = ItemCatalogRecord(item_id="item-1", title="The Matrix", popularity=10)

    assert item.item_id == "item-1"
    assert item.title == "The Matrix"

    with pytest.raises(ValueError):
        ItemCatalogRecord(item_id="", title="No id")
    with pytest.raises(ValueError):
        ItemCatalogRecord(item_id="item-2", title="", popularity=-1)


def test_user_sequence_requires_consistent_user_ids() -> None:
    sequence = synthetic_sequence()

    assert sequence.item_ids == ("item-tail", "item-mid")

    with pytest.raises(ValueError):
        UserSequenceRecord(
            user_id="user-1",
            interactions=(InteractionRecord("user-2", "item-1"),),
        )


def test_prediction_grounding_confidence_and_observation_records() -> None:
    prediction = GenerativePredictionRecord(
        prediction_id="pred-1",
        example_id="ex-1",
        user_id="user-1",
        generated_title="Primer",
    )
    grounded = GroundedPredictionRecord(
        prediction_id="pred-1",
        generated_title="Primer",
        normalized_title="primer",
        item_id="item-tail",
        status=GroundingStatus.EXACT,
        score=1.0,
        ambiguity=0.0,
    )
    confidence = ConfidenceRecord(
        prediction_id="pred-1",
        probability_correct=0.8,
        source="fixture",
    )
    example = ObservationExampleRecord(
        example_id="ex-1",
        user_sequence=synthetic_sequence(),
        target_item_ids=("item-tail",),
        prediction=prediction,
        grounded_prediction=grounded,
        confidence=confidence,
        correctness=1,
        is_synthetic=True,
    )

    assert grounded.is_grounded
    assert example.correctness == 1

    with pytest.raises(ValueError):
        ConfidenceRecord("pred-1", 1.2, "bad")


def test_synthetic_fixture_is_marked_synthetic() -> None:
    example = synthetic_observation_example()

    assert example.is_synthetic is True
    assert example.correctness == 1
