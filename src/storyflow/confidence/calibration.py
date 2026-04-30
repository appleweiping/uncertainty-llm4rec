"""Calibration scaffold for CURE/TRUCE feature rows.

The calibrator in this module is deliberately small and deterministic. It
exists to enforce split provenance and a leakage-safe output contract before
the project adds learned exposure-counterfactual calibrators. It does not call
APIs, train models, or provide evidence that CURE/TRUCE improves results.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from storyflow.metrics import brier_score, expected_calibration_error
from storyflow.observation import read_jsonl, utc_now_iso, write_jsonl

CALIBRATOR_SCHEMA_VERSION = "cure_truce_calibrator_v1"
DEFAULT_PROBABILITY_SOURCE = "estimated_exposure_confidence"
SUPPORTED_PROBABILITY_SOURCES = {
    "estimated_exposure_confidence",
    "verbal_confidence",
    "generation_confidence",
    "grounding_confidence",
    "preference_score",
}


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _label_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    label = int(value)
    if label not in (0, 1):
        raise ValueError("calibration labels must be binary 0/1 values")
    return label


def _normalize_splits(splits: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(splits, str):
        raw_values = splits.split(",")
    else:
        raw_values = list(splits)
    normalized = tuple(str(value).strip() for value in raw_values if str(value).strip())
    if not normalized:
        raise ValueError("at least one split must be provided")
    return normalized


def row_split(row: Mapping[str, Any]) -> str:
    """Return the split recorded on a CURE/TRUCE feature row."""

    split = row.get("split")
    if split in (None, ""):
        feature = row.get("feature")
        if isinstance(feature, Mapping):
            metadata = feature.get("metadata")
            if isinstance(metadata, Mapping):
                split = metadata.get("split")
    if split in (None, ""):
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping):
            split = metadata.get("split")
    if split in (None, ""):
        raise ValueError("feature rows must record split provenance before calibration")
    return str(split)


def row_probability(
    row: Mapping[str, Any],
    *,
    source: str = DEFAULT_PROBABILITY_SOURCE,
) -> float | None:
    """Read a probability-like confidence value from a feature row."""

    if source not in SUPPORTED_PROBABILITY_SOURCES:
        raise ValueError(f"unsupported probability source: {source}")
    if source == "estimated_exposure_confidence":
        score = row.get("score")
        value = score.get(source) if isinstance(score, Mapping) else None
    else:
        feature = row.get("feature")
        value = feature.get(source) if isinstance(feature, Mapping) else None
    probability = _float_or_none(value)
    if probability is None:
        return None
    return _clip01(probability)


def row_label(row: Mapping[str, Any]) -> int | None:
    """Read the correctness label used as the first calibration target."""

    feature = row.get("feature")
    if isinstance(feature, Mapping):
        label = _label_or_none(feature.get("correctness_label"))
        if label is not None:
            return label
    return _label_or_none(row.get("correctness"))


def _bin_index(probability: float, *, n_bins: int) -> int:
    clipped = _clip01(probability)
    if clipped <= 0.0:
        return 0
    if clipped >= 1.0:
        return n_bins - 1
    return min(n_bins - 1, int(clipped * n_bins))


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    """One labeled feature row used for fitting or evaluating calibration."""

    row_index: int
    input_id: str | None
    split: str
    probability: float
    label: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """Fixed-width histogram bin fit on the selected fit split."""

    bin_index: int
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    empirical_accuracy: float | None
    calibrated_probability: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HistogramCalibrator:
    """A split-audited histogram calibrator scaffold."""

    schema_version: str
    probability_source: str
    fit_splits: tuple[str, ...]
    n_bins: int
    bins: tuple[CalibrationBin, ...]
    fit_count: int
    skipped_fit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "probability_source": self.probability_source,
            "fit_splits": list(self.fit_splits),
            "n_bins": self.n_bins,
            "bins": [bin_record.to_dict() for bin_record in self.bins],
            "fit_count": self.fit_count,
            "skipped_fit_count": self.skipped_fit_count,
        }

    def calibrate(self, probability: float) -> tuple[float, int, bool]:
        bin_index = _bin_index(probability, n_bins=self.n_bins)
        bin_record = self.bins[bin_index]
        if bin_record.calibrated_probability is None:
            return _clip01(probability), bin_index, True
        return bin_record.calibrated_probability, bin_index, False


def collect_calibration_examples(
    rows: Iterable[Mapping[str, Any]],
    *,
    splits: Sequence[str] | str,
    probability_source: str = DEFAULT_PROBABILITY_SOURCE,
) -> tuple[list[CalibrationExample], int]:
    """Collect labeled probability rows for the requested splits.

    Rows in the requested split with a missing label or missing probability are
    skipped and counted. Rows outside the requested split are ignored.
    """

    requested_splits = set(_normalize_splits(splits))
    examples: list[CalibrationExample] = []
    skipped = 0
    for index, row in enumerate(rows):
        split = row_split(row)
        if split not in requested_splits:
            continue
        probability = row_probability(row, source=probability_source)
        label = row_label(row)
        if probability is None or label is None:
            skipped += 1
            continue
        examples.append(
            CalibrationExample(
                row_index=index,
                input_id=str(row.get("input_id")) if row.get("input_id") is not None else None,
                split=split,
                probability=probability,
                label=label,
            )
        )
    return examples, skipped


def fit_histogram_calibrator(
    rows: Iterable[Mapping[str, Any]],
    *,
    fit_splits: Sequence[str] | str = ("train",),
    probability_source: str = DEFAULT_PROBABILITY_SOURCE,
    n_bins: int = 10,
) -> HistogramCalibrator:
    """Fit a fixed-width empirical calibrator on fit splits only."""

    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    normalized_fit_splits = _normalize_splits(fit_splits)
    rows_list = list(rows)
    examples, skipped = collect_calibration_examples(
        rows_list,
        splits=normalized_fit_splits,
        probability_source=probability_source,
    )
    if not examples:
        raise ValueError("no labeled fit rows are available for the requested fit splits")

    grouped: dict[int, list[CalibrationExample]] = {index: [] for index in range(n_bins)}
    for example in examples:
        grouped[_bin_index(example.probability, n_bins=n_bins)].append(example)

    bins: list[CalibrationBin] = []
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        members = grouped[index]
        if members:
            mean_probability = sum(example.probability for example in members) / len(members)
            empirical_accuracy = sum(example.label for example in members) / len(members)
            calibrated_probability = empirical_accuracy
        else:
            mean_probability = None
            empirical_accuracy = None
            calibrated_probability = None
        bins.append(
            CalibrationBin(
                bin_index=index,
                lower=lower,
                upper=upper,
                count=len(members),
                mean_probability=mean_probability,
                empirical_accuracy=empirical_accuracy,
                calibrated_probability=calibrated_probability,
            )
        )
    return HistogramCalibrator(
        schema_version=CALIBRATOR_SCHEMA_VERSION,
        probability_source=probability_source,
        fit_splits=normalized_fit_splits,
        n_bins=n_bins,
        bins=tuple(bins),
        fit_count=len(examples),
        skipped_fit_count=skipped,
    )


def summarize_calibration_examples(
    examples: Iterable[CalibrationExample],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Summarize ECE/Brier for labeled calibration examples."""

    records = list(examples)
    if not records:
        return {
            "count": 0,
            "mean_probability": None,
            "accuracy": None,
            "ece": None,
            "brier": None,
        }
    probabilities = [example.probability for example in records]
    labels = [example.label for example in records]
    return {
        "count": len(records),
        "mean_probability": sum(probabilities) / len(probabilities),
        "accuracy": sum(labels) / len(labels),
        "ece": expected_calibration_error(probabilities, labels, n_bins=n_bins),
        "brier": brier_score(probabilities, labels),
    }


def _summary_from_probabilities(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    n_bins: int,
) -> dict[str, Any]:
    probs = list(probabilities)
    y = list(labels)
    if not probs:
        return {
            "count": 0,
            "mean_probability": None,
            "accuracy": None,
            "ece": None,
            "brier": None,
        }
    return {
        "count": len(probs),
        "mean_probability": sum(probs) / len(probs),
        "accuracy": sum(y) / len(y),
        "ece": expected_calibration_error(probs, y, n_bins=n_bins),
        "brier": brier_score(probs, y),
    }


def apply_histogram_calibrator(
    rows: Iterable[Mapping[str, Any]],
    calibrator: HistogramCalibrator,
    *,
    eval_splits: Sequence[str] | str = ("validation", "test"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a fitted calibrator to evaluation splits and summarize outputs."""

    normalized_eval_splits = _normalize_splits(eval_splits)
    eval_split_set = set(normalized_eval_splits)
    output_rows: list[dict[str, Any]] = []
    source_probabilities: list[float] = []
    calibrated_probabilities: list[float] = []
    labels: list[int] = []
    missing_probability_count = 0
    missing_label_count = 0
    fallback_count = 0

    for row in rows:
        split = row_split(row)
        if split not in eval_split_set:
            continue
        output = dict(row)
        probability = row_probability(row, source=calibrator.probability_source)
        label = row_label(row)
        calibration_record: dict[str, Any] = {
            "schema_version": calibrator.schema_version,
            "probability_source": calibrator.probability_source,
            "fit_splits": list(calibrator.fit_splits),
            "eval_split": split,
            "source_probability": probability,
            "correctness_label": label,
            "calibrated_probability": None,
            "applied_bin_index": None,
            "empty_bin_fallback": False,
            "status": "calibrated",
        }
        if probability is None:
            missing_probability_count += 1
            calibration_record["status"] = "missing_probability"
        else:
            calibrated, bin_index, used_fallback = calibrator.calibrate(probability)
            calibration_record["calibrated_probability"] = calibrated
            calibration_record["applied_bin_index"] = bin_index
            calibration_record["empty_bin_fallback"] = used_fallback
            fallback_count += int(used_fallback)
            if label is not None:
                source_probabilities.append(probability)
                calibrated_probabilities.append(calibrated)
                labels.append(label)
        if label is None:
            missing_label_count += 1
        output["calibration"] = calibration_record
        output["is_experiment_result"] = False
        output_rows.append(output)

    split_counts = Counter(row["calibration"]["eval_split"] for row in output_rows)
    summary = {
        "eval_splits": list(normalized_eval_splits),
        "eval_row_count": len(output_rows),
        "eval_split_counts": dict(sorted(split_counts.items())),
        "labeled_eval_count": len(labels),
        "missing_eval_probability_count": missing_probability_count,
        "missing_eval_label_count": missing_label_count,
        "empty_bin_fallback_count": fallback_count,
        "source_probability_summary": _summary_from_probabilities(
            source_probabilities,
            labels,
            n_bins=calibrator.n_bins,
        ),
        "calibrated_probability_summary": _summary_from_probabilities(
            calibrated_probabilities,
            labels,
            n_bins=calibrator.n_bins,
        ),
    }
    return output_rows, summary


def calibrate_feature_rows(
    *,
    features_jsonl: str | Path,
    output_jsonl: str | Path,
    manifest_json: str | Path,
    fit_splits: Sequence[str] | str = ("train",),
    eval_splits: Sequence[str] | str = ("validation", "test"),
    probability_source: str = DEFAULT_PROBABILITY_SOURCE,
    n_bins: int = 10,
    allow_same_split_eval: bool = False,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Fit and apply the calibration scaffold with explicit split guards."""

    features_path = Path(features_jsonl)
    rows = read_jsonl(features_path)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        rows = rows[:max_examples]
    if not rows:
        raise ValueError("features_jsonl contains no rows")

    normalized_fit_splits = _normalize_splits(fit_splits)
    normalized_eval_splits = _normalize_splits(eval_splits)
    overlap = sorted(set(normalized_fit_splits) & set(normalized_eval_splits))
    if overlap and not allow_same_split_eval:
        raise ValueError(
            "fit_splits and eval_splits overlap; pass allow_same_split_eval only "
            "for explicitly labeled diagnostics"
        )

    split_counts = Counter(row_split(row) for row in rows)
    calibrator = fit_histogram_calibrator(
        rows,
        fit_splits=normalized_fit_splits,
        probability_source=probability_source,
        n_bins=n_bins,
    )
    fit_examples, skipped_fit_count = collect_calibration_examples(
        rows,
        splits=normalized_fit_splits,
        probability_source=probability_source,
    )
    output_rows, eval_summary = apply_histogram_calibrator(
        rows,
        calibrator,
        eval_splits=normalized_eval_splits,
    )
    if not output_rows:
        raise ValueError("no evaluation rows are available for the requested eval splits")

    write_jsonl(output_jsonl, output_rows)
    manifest = {
        "created_at_utc": utc_now_iso(),
        "calibrator_schema_version": CALIBRATOR_SCHEMA_VERSION,
        "features_jsonl": str(features_path),
        "output_jsonl": str(output_jsonl),
        "manifest_json": str(manifest_json),
        "probability_source": probability_source,
        "fit_splits": list(normalized_fit_splits),
        "eval_splits": list(normalized_eval_splits),
        "n_bins": n_bins,
        "allow_same_split_eval": allow_same_split_eval,
        "split_counts": dict(sorted(split_counts.items())),
        "fit_summary": {
            **summarize_calibration_examples(fit_examples, n_bins=n_bins),
            "skipped_fit_count": skipped_fit_count,
        },
        "eval_summary": eval_summary,
        "calibrator": calibrator.to_dict(),
        "leakage_guard": {
            "fit_eval_overlap": overlap,
            "same_split_eval_allowed": allow_same_split_eval,
            "label_source": "feature.correctness_label",
            "requires_split_provenance": True,
            "default_refuses_fit_eval_overlap": True,
        },
        "api_called": False,
        "model_training": False,
        "server_executed": False,
        "is_experiment_result": False,
        "note": (
            "Split-audited CURE/TRUCE calibration scaffold. This fits a "
            "deterministic histogram mapping on the requested fit split only; "
            "it is not a learned method result or paper evidence."
        ),
    }
    manifest_path = Path(manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "CALIBRATOR_SCHEMA_VERSION",
    "DEFAULT_PROBABILITY_SOURCE",
    "SUPPORTED_PROBABILITY_SOURCES",
    "CalibrationBin",
    "CalibrationExample",
    "HistogramCalibrator",
    "apply_histogram_calibrator",
    "calibrate_feature_rows",
    "collect_calibration_examples",
    "fit_histogram_calibrator",
    "row_label",
    "row_probability",
    "row_split",
    "summarize_calibration_examples",
]
