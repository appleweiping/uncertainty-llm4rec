"""Core records for title-level generative recommendation observation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class GroundingStatus(str, Enum):
    """Catalog grounding outcome for a generated title."""

    EXACT = "exact"
    NORMALIZED_EXACT = "normalized_exact"
    FUZZY = "fuzzy"
    AMBIGUOUS = "ambiguous"
    UNGROUNDED = "ungrounded"


class PopularityBucket(str, Enum):
    """Head/mid/tail popularity group."""

    HEAD = "head"
    MID = "mid"
    TAIL = "tail"


def _require_non_empty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _copy_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return dict(metadata)


@dataclass(frozen=True, slots=True)
class ItemCatalogRecord:
    """One item available for grounding generated recommendation titles."""

    item_id: str
    title: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    popularity: float | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string("item_id", self.item_id)
        _require_non_empty_string("title", self.title)
        if self.popularity is not None and self.popularity < 0:
            raise ValueError("popularity must be non-negative when provided")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class InteractionRecord:
    """One user-item interaction from a sequential recommendation log."""

    user_id: str
    item_id: str
    timestamp: float | None = None
    event_type: str = "interaction"
    rating: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string("user_id", self.user_id)
        _require_non_empty_string("item_id", self.item_id)
        _require_non_empty_string("event_type", self.event_type)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class UserSequenceRecord:
    """Chronological interaction sequence for a single user."""

    user_id: str
    interactions: tuple[InteractionRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string("user_id", self.user_id)
        interactions = tuple(self.interactions)
        if not interactions:
            raise ValueError("interactions must contain at least one record")
        for interaction in interactions:
            if interaction.user_id != self.user_id:
                raise ValueError("all interactions must have the sequence user_id")
        object.__setattr__(self, "interactions", interactions)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(interaction.item_id for interaction in self.interactions)


@dataclass(frozen=True, slots=True)
class GenerativePredictionRecord:
    """Raw generated title before catalog grounding."""

    prediction_id: str
    example_id: str
    user_id: str
    generated_title: str
    rank: int = 1
    raw_text: str | None = None
    provider: str | None = None
    model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string("prediction_id", self.prediction_id)
        _require_non_empty_string("example_id", self.example_id)
        _require_non_empty_string("user_id", self.user_id)
        _require_non_empty_string("generated_title", self.generated_title)
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class GroundingCandidate:
    """One catalog candidate considered during title grounding."""

    item_id: str
    title: str
    normalized_title: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        _require_non_empty_string("item_id", self.item_id)
        _require_non_empty_string("title", self.title)
        _require_non_empty_string("normalized_title", self.normalized_title)
        _validate_probability("score", self.score)
        if self.rank < 1:
            raise ValueError("rank must be >= 1")


@dataclass(frozen=True, slots=True)
class GroundedPredictionRecord:
    """Generated title after attempting to ground it to the catalog."""

    prediction_id: str
    generated_title: str
    normalized_title: str
    item_id: str | None
    status: GroundingStatus
    score: float
    ambiguity: float
    candidates: tuple[GroundingCandidate, ...] = field(default_factory=tuple)
    second_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string("prediction_id", self.prediction_id)
        _require_non_empty_string("generated_title", self.generated_title)
        if self.item_id is not None:
            _require_non_empty_string("item_id", self.item_id)
        status = self.status
        if not isinstance(status, GroundingStatus):
            status = GroundingStatus(status)
        _validate_probability("score", self.score)
        _validate_probability("ambiguity", self.ambiguity)
        if self.second_score is not None:
            _validate_probability("second_score", self.second_score)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    @property
    def is_grounded(self) -> bool:
        return self.item_id is not None and self.status not in {
            GroundingStatus.AMBIGUOUS,
            GroundingStatus.UNGROUNDED,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceRecord:
    """Confidence that a grounded generated recommendation is correct."""

    prediction_id: str
    probability_correct: float
    source: str
    raw_confidence: str | float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string("prediction_id", self.prediction_id)
        _require_non_empty_string("source", self.source)
        _validate_probability("probability_correct", self.probability_correct)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ObservationExampleRecord:
    """One title-generation observation example with optional prediction state."""

    example_id: str
    user_sequence: UserSequenceRecord
    target_item_ids: tuple[str, ...]
    split: str = "test"
    prediction: GenerativePredictionRecord | None = None
    grounded_prediction: GroundedPredictionRecord | None = None
    confidence: ConfidenceRecord | None = None
    correctness: int | None = None
    is_synthetic: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string("example_id", self.example_id)
        _require_non_empty_string("split", self.split)
        target_item_ids = tuple(self.target_item_ids)
        if not target_item_ids:
            raise ValueError("target_item_ids must contain at least one item_id")
        for item_id in target_item_ids:
            _require_non_empty_string("target_item_id", item_id)
        if self.correctness not in (None, 0, 1):
            raise ValueError("correctness must be None, 0, or 1")
        object.__setattr__(self, "target_item_ids", target_item_ids)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))
