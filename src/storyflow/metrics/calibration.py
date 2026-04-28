"""Calibration and grounding metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from storyflow.schemas import GroundedPredictionRecord, PopularityBucket


def _as_float_list(name: str, values: Iterable[float]) -> list[float]:
    output = [float(value) for value in values]
    if not output:
        raise ValueError(f"{name} must not be empty")
    for value in output:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} values must be in [0, 1]")
    return output


def _as_label_list(labels: Iterable[int]) -> list[int]:
    output = [int(label) for label in labels]
    if not output:
        raise ValueError("labels must not be empty")
    for label in output:
        if label not in (0, 1):
            raise ValueError("labels must be binary 0/1 values")
    return output


def _validate_same_length(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        raise ValueError("inputs must have the same length")


def expected_calibration_error(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    n_bins: int = 10,
) -> float:
    """Compute fixed-width Expected Calibration Error for binary labels."""

    probs = _as_float_list("probabilities", probabilities)
    y = _as_label_list(labels)
    _validate_same_length(probs, y)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    total = len(probs)
    ece = 0.0
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == 0:
            member_indexes = [
                index
                for index, prob in enumerate(probs)
                if lower <= prob <= upper
            ]
        else:
            member_indexes = [
                index
                for index, prob in enumerate(probs)
                if lower < prob <= upper
            ]
        if not member_indexes:
            continue
        bin_confidence = sum(probs[index] for index in member_indexes) / len(
            member_indexes
        )
        bin_accuracy = sum(y[index] for index in member_indexes) / len(
            member_indexes
        )
        ece += (len(member_indexes) / total) * abs(
            bin_accuracy - bin_confidence
        )
    return ece


def brier_score(probabilities: Iterable[float], labels: Iterable[int]) -> float:
    """Compute binary Brier score."""

    probs = _as_float_list("probabilities", probabilities)
    y = _as_label_list(labels)
    _validate_same_length(probs, y)
    return sum((prob - label) ** 2 for prob, label in zip(probs, y)) / len(probs)


def cbu_tau(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    tau: float,
) -> float:
    """Correct-but-uncertain rate: P(confidence < tau | correct)."""

    probs = _as_float_list("probabilities", probabilities)
    y = _as_label_list(labels)
    _validate_same_length(probs, y)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    correct_probs = [prob for prob, label in zip(probs, y) if label == 1]
    if not correct_probs:
        return math.nan
    return sum(prob < tau for prob in correct_probs) / len(correct_probs)


def wbc_tau(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    tau: float,
) -> float:
    """Wrong-but-confident rate: P(confidence > tau | incorrect)."""

    probs = _as_float_list("probabilities", probabilities)
    y = _as_label_list(labels)
    _validate_same_length(probs, y)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    wrong_probs = [prob for prob, label in zip(probs, y) if label == 0]
    if not wrong_probs:
        return math.nan
    return sum(prob > tau for prob in wrong_probs) / len(wrong_probs)


def ground_hit_rate(
    grounded_predictions: Iterable[GroundedPredictionRecord | bool],
) -> float:
    """Share of generated titles grounded to a non-ambiguous catalog item."""

    records = list(grounded_predictions)
    if not records:
        raise ValueError("grounded_predictions must not be empty")
    hits = 0
    for record in records:
        if isinstance(record, GroundedPredictionRecord):
            hits += int(record.is_grounded)
        else:
            hits += int(bool(record))
    return hits / len(records)


def tail_underconfidence_gap(
    probabilities: Iterable[float],
    labels: Iterable[int],
    buckets: Iterable[PopularityBucket | str],
) -> float:
    """Mean correct-head confidence minus mean correct-tail confidence."""

    probs = _as_float_list("probabilities", probabilities)
    y = _as_label_list(labels)
    bucket_values = [
        bucket if isinstance(bucket, PopularityBucket) else PopularityBucket(bucket)
        for bucket in buckets
    ]
    if not bucket_values:
        raise ValueError("buckets must not be empty")
    if len(probs) != len(y) or len(probs) != len(bucket_values):
        raise ValueError("inputs must have the same length")

    correct_head = [
        prob
        for prob, label, bucket in zip(probs, y, bucket_values)
        if label == 1 and bucket == PopularityBucket.HEAD
    ]
    correct_tail = [
        prob
        for prob, label, bucket in zip(probs, y, bucket_values)
        if label == 1 and bucket == PopularityBucket.TAIL
    ]
    if not correct_head or not correct_tail:
        return math.nan
    return (sum(correct_head) / len(correct_head)) - (
        sum(correct_tail) / len(correct_tail)
    )
