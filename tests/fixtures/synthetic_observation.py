"""Small synthetic fixture for tests only.

These records are not experimental data and must not be reported as results.
"""

from __future__ import annotations

from storyflow.schemas import (
    ConfidenceRecord,
    GenerativePredictionRecord,
    InteractionRecord,
    ItemCatalogRecord,
    ObservationExampleRecord,
    UserSequenceRecord,
)


def synthetic_catalog() -> tuple[ItemCatalogRecord, ...]:
    return (
        ItemCatalogRecord("item-head", "The Matrix", popularity=100),
        ItemCatalogRecord("item-mid", "Arrival", popularity=30),
        ItemCatalogRecord("item-tail", "Primer", popularity=3),
        ItemCatalogRecord("item-tail-2", "Coherence", popularity=2),
    )


def synthetic_sequence() -> UserSequenceRecord:
    return UserSequenceRecord(
        user_id="user-1",
        interactions=(
            InteractionRecord("user-1", "item-tail", timestamp=1),
            InteractionRecord("user-1", "item-mid", timestamp=2),
        ),
    )


def synthetic_prediction() -> GenerativePredictionRecord:
    return GenerativePredictionRecord(
        prediction_id="pred-1",
        example_id="example-1",
        user_id="user-1",
        generated_title="Primer",
        raw_text="Title: Primer\nProbability correct: 0.72",
        provider="synthetic",
        model="fixture",
    )


def synthetic_confidence() -> ConfidenceRecord:
    return ConfidenceRecord(
        prediction_id="pred-1",
        probability_correct=0.72,
        source="synthetic_fixture",
    )


def synthetic_observation_example() -> ObservationExampleRecord:
    return ObservationExampleRecord(
        example_id="example-1",
        user_sequence=synthetic_sequence(),
        target_item_ids=("item-tail",),
        prediction=synthetic_prediction(),
        confidence=synthetic_confidence(),
        correctness=1,
        is_synthetic=True,
    )
