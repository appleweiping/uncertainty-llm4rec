"""Popularity residual scaffold for CURE/TRUCE feature rows.

This module fits a split-audited popularity-only confidence baseline and
subtracts it from evaluation rows. The output is a deconfounding contract for
later learned CURE/TRUCE modules; it does not call APIs, train models, or
produce paper evidence.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from storyflow.confidence.calibration import (
    DEFAULT_PROBABILITY_SOURCE,
    SUPPORTED_PROBABILITY_SOURCES,
    row_label,
    row_probability,
    row_split,
)
from storyflow.observation import read_jsonl, utc_now_iso, write_jsonl

POPULARITY_RESIDUAL_SCHEMA_VERSION = "cure_truce_popularity_residual_v1"
POPULARITY_RESIDUAL_METHOD = "bucket_mean"
POPULARITY_BUCKETS = ("head", "mid", "tail", "unknown")


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


def _normalize_splits(splits: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(splits, str):
        raw_values = splits.split(",")
    else:
        raw_values = list(splits)
    normalized = tuple(str(value).strip() for value in raw_values if str(value).strip())
    if not normalized:
        raise ValueError("at least one split must be provided")
    return normalized


def _normalize_bucket(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    normalized = str(value).strip().lower()
    if normalized in {"head", "mid", "tail"}:
        return normalized
    return "unknown"


def row_popularity_bucket(row: Mapping[str, Any]) -> str:
    """Return the popularity bucket recorded on a feature row."""

    feature = row.get("feature")
    if isinstance(feature, Mapping):
        bucket = _normalize_bucket(feature.get("popularity_bucket"))
        if bucket != "unknown":
            return bucket
        metadata = feature.get("metadata")
        if isinstance(metadata, Mapping):
            bucket = _normalize_bucket(metadata.get("generated_popularity_bucket"))
            if bucket != "unknown":
                return bucket

    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("generated_popularity_bucket", "popularity_bucket"):
            bucket = _normalize_bucket(metadata.get(key))
            if bucket != "unknown":
                return bucket

    return "unknown"


def row_popularity_percentile(row: Mapping[str, Any]) -> float | None:
    """Return a [0, 1] popularity percentile if one is available."""

    feature = row.get("feature")
    if isinstance(feature, Mapping):
        percentile = _float_or_none(feature.get("popularity_percentile"))
        if percentile is not None:
            return _clip01(percentile)
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        percentile = _float_or_none(metadata.get("popularity_percentile"))
        if percentile is not None:
            return _clip01(percentile)
    return None


@dataclass(frozen=True, slots=True)
class PopularityResidualExample:
    """One feature row used to fit a popularity-only confidence baseline."""

    row_index: int
    input_id: str | None
    split: str
    probability: float
    popularity_bucket: str
    popularity_percentile: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PopularityResidualBin:
    """Mean confidence observed for one popularity bucket on fit splits."""

    popularity_bucket: str
    count: int
    mean_probability: float
    mean_popularity_percentile: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PopularityResidualModel:
    """Split-audited popularity-only confidence baseline."""

    schema_version: str
    method: str
    probability_source: str
    fit_splits: tuple[str, ...]
    global_mean_probability: float
    bins: tuple[PopularityResidualBin, ...]
    fit_count: int
    skipped_fit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "probability_source": self.probability_source,
            "fit_splits": list(self.fit_splits),
            "global_mean_probability": self.global_mean_probability,
            "bins": [bin_record.to_dict() for bin_record in self.bins],
            "fit_count": self.fit_count,
            "skipped_fit_count": self.skipped_fit_count,
        }

    def baseline_probability(self, popularity_bucket: str) -> tuple[float, bool]:
        bucket = _normalize_bucket(popularity_bucket)
        for bin_record in self.bins:
            if bin_record.popularity_bucket == bucket:
                return bin_record.mean_probability, False
        return self.global_mean_probability, True


def collect_popularity_residual_examples(
    rows: Iterable[Mapping[str, Any]],
    *,
    splits: Sequence[str] | str,
    probability_source: str = DEFAULT_PROBABILITY_SOURCE,
) -> tuple[list[PopularityResidualExample], int]:
    """Collect probability rows for fitting a popularity-only baseline."""

    if probability_source not in SUPPORTED_PROBABILITY_SOURCES:
        raise ValueError(f"unsupported probability source: {probability_source}")
    requested_splits = set(_normalize_splits(splits))
    examples: list[PopularityResidualExample] = []
    skipped = 0
    for index, row in enumerate(rows):
        split = row_split(row)
        if split not in requested_splits:
            continue
        probability = row_probability(row, source=probability_source)
        if probability is None:
            skipped += 1
            continue
        examples.append(
            PopularityResidualExample(
                row_index=index,
                input_id=str(row.get("input_id")) if row.get("input_id") is not None else None,
                split=split,
                probability=probability,
                popularity_bucket=row_popularity_bucket(row),
                popularity_percentile=row_popularity_percentile(row),
            )
        )
    return examples, skipped


def fit_popularity_residual_model(
    rows: Iterable[Mapping[str, Any]],
    *,
    fit_splits: Sequence[str] | str = ("train",),
    probability_source: str = DEFAULT_PROBABILITY_SOURCE,
) -> PopularityResidualModel:
    """Fit a bucket-mean popularity-only confidence baseline on fit splits."""

    normalized_fit_splits = _normalize_splits(fit_splits)
    rows_list = list(rows)
    examples, skipped = collect_popularity_residual_examples(
        rows_list,
        splits=normalized_fit_splits,
        probability_source=probability_source,
    )
    if not examples:
        raise ValueError("no fit rows with probabilities are available for popularity residualization")

    by_bucket: dict[str, list[PopularityResidualExample]] = defaultdict(list)
    for example in examples:
        by_bucket[example.popularity_bucket].append(example)

    bins: list[PopularityResidualBin] = []
    ordered_buckets = [bucket for bucket in POPULARITY_BUCKETS if bucket in by_bucket]
    ordered_buckets.extend(sorted(bucket for bucket in by_bucket if bucket not in POPULARITY_BUCKETS))
    for bucket in ordered_buckets:
        members = by_bucket[bucket]
        percentiles = [
            member.popularity_percentile
            for member in members
            if member.popularity_percentile is not None
        ]
        bins.append(
            PopularityResidualBin(
                popularity_bucket=bucket,
                count=len(members),
                mean_probability=sum(member.probability for member in members) / len(members),
                mean_popularity_percentile=(
                    sum(percentiles) / len(percentiles) if percentiles else None
                ),
            )
        )

    return PopularityResidualModel(
        schema_version=POPULARITY_RESIDUAL_SCHEMA_VERSION,
        method=POPULARITY_RESIDUAL_METHOD,
        probability_source=probability_source,
        fit_splits=normalized_fit_splits,
        global_mean_probability=sum(example.probability for example in examples) / len(examples),
        bins=tuple(bins),
        fit_count=len(examples),
        skipped_fit_count=skipped,
    )


def _mean(values: Iterable[float]) -> float | None:
    records = list(values)
    if not records:
        return None
    return sum(records) / len(records)


def _summarize_applied_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    residuals = [
        row["popularity_residual_confidence"]
        for row in rows
        if row.get("popularity_residual_confidence") is not None
    ]
    source_probabilities = [
        row["source_probability"] for row in rows if row.get("source_probability") is not None
    ]
    baseline_probabilities = [
        row["popularity_baseline_probability"]
        for row in rows
        if row.get("popularity_baseline_probability") is not None
    ]
    deconfounded_probabilities = [
        row["deconfounded_confidence_proxy"]
        for row in rows
        if row.get("deconfounded_confidence_proxy") is not None
    ]
    return {
        "count": len(rows),
        "mean_source_probability": _mean(source_probabilities),
        "mean_popularity_baseline_probability": _mean(baseline_probabilities),
        "mean_popularity_residual_confidence": _mean(residuals),
        "mean_abs_popularity_residual_confidence": _mean(abs(value) for value in residuals),
        "mean_deconfounded_confidence_proxy": _mean(deconfounded_probabilities),
    }


def apply_popularity_residual_model(
    rows: Iterable[Mapping[str, Any]],
    model: PopularityResidualModel,
    *,
    eval_splits: Sequence[str] | str = ("validation", "test"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a popularity residual model to evaluation splits."""

    normalized_eval_splits = _normalize_splits(eval_splits)
    eval_split_set = set(normalized_eval_splits)
    output_rows: list[dict[str, Any]] = []
    applied_records: list[dict[str, Any]] = []
    missing_probability_count = 0
    global_fallback_count = 0
    labeled_eval_count = 0

    for row in rows:
        split = row_split(row)
        if split not in eval_split_set:
            continue
        output = dict(row)
        bucket = row_popularity_bucket(row)
        percentile = row_popularity_percentile(row)
        probability = row_probability(row, source=model.probability_source)
        label = row_label(row)
        if label is not None:
            labeled_eval_count += 1

        baseline, used_global_fallback = model.baseline_probability(bucket)
        residual: float | None = None
        deconfounded_proxy: float | None = None
        status = "residualized"
        if probability is None:
            missing_probability_count += 1
            status = "missing_probability"
        else:
            residual = probability - baseline
            deconfounded_proxy = _clip01(model.global_mean_probability + residual)
            global_fallback_count += int(used_global_fallback)

        residual_record: dict[str, Any] = {
            "schema_version": model.schema_version,
            "method": model.method,
            "probability_source": model.probability_source,
            "fit_splits": list(model.fit_splits),
            "eval_split": split,
            "source_probability": probability,
            "correctness_label": label,
            "popularity_bucket": bucket,
            "popularity_percentile": percentile,
            "popularity_baseline_probability": baseline,
            "global_baseline_probability": model.global_mean_probability,
            "used_global_bucket_fallback": used_global_fallback,
            "popularity_residual_confidence": residual,
            "deconfounded_confidence_proxy": deconfounded_proxy,
            "status": status,
        }
        output["popularity_residualization"] = residual_record
        output["is_experiment_result"] = False
        output_rows.append(output)
        applied_records.append(residual_record)

    split_counts = Counter(row["popularity_residualization"]["eval_split"] for row in output_rows)
    bucket_counts = Counter(
        row["popularity_residualization"]["popularity_bucket"] for row in output_rows
    )
    by_bucket = {
        bucket: _summarize_applied_records(
            row["popularity_residualization"]
            for row in output_rows
            if row["popularity_residualization"]["popularity_bucket"] == bucket
        )
        for bucket in sorted(bucket_counts)
    }
    by_label = {
        str(label): _summarize_applied_records(
            row["popularity_residualization"]
            for row in output_rows
            if row["popularity_residualization"]["correctness_label"] == label
        )
        for label in (0, 1)
    }
    summary = {
        "eval_splits": list(normalized_eval_splits),
        "eval_row_count": len(output_rows),
        "eval_split_counts": dict(sorted(split_counts.items())),
        "eval_bucket_counts": dict(sorted(bucket_counts.items())),
        "labeled_eval_count": labeled_eval_count,
        "missing_eval_probability_count": missing_probability_count,
        "global_bucket_fallback_count": global_fallback_count,
        "overall": _summarize_applied_records(applied_records),
        "by_popularity_bucket": by_bucket,
        "by_correctness_label": by_label,
    }
    return output_rows, summary


def residualize_feature_rows(
    *,
    features_jsonl: str | Path,
    output_jsonl: str | Path,
    manifest_json: str | Path,
    fit_splits: Sequence[str] | str = ("train",),
    eval_splits: Sequence[str] | str = ("validation", "test"),
    probability_source: str = DEFAULT_PROBABILITY_SOURCE,
    allow_same_split_eval: bool = False,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Fit/apply popularity residualization with explicit split guards."""

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
    model = fit_popularity_residual_model(
        rows,
        fit_splits=normalized_fit_splits,
        probability_source=probability_source,
    )
    output_rows, eval_summary = apply_popularity_residual_model(
        rows,
        model,
        eval_splits=normalized_eval_splits,
    )
    if not output_rows:
        raise ValueError("no evaluation rows are available for the requested eval splits")

    write_jsonl(output_jsonl, output_rows)
    manifest = {
        "created_at_utc": utc_now_iso(),
        "popularity_residual_schema_version": POPULARITY_RESIDUAL_SCHEMA_VERSION,
        "method": POPULARITY_RESIDUAL_METHOD,
        "features_jsonl": str(features_path),
        "output_jsonl": str(output_jsonl),
        "manifest_json": str(manifest_json),
        "probability_source": probability_source,
        "fit_splits": list(normalized_fit_splits),
        "eval_splits": list(normalized_eval_splits),
        "max_examples": max_examples,
        "allow_same_split_eval": allow_same_split_eval,
        "split_counts": dict(sorted(split_counts.items())),
        "fit_summary": {
            "fit_row_count": model.fit_count,
            "skipped_fit_count": model.skipped_fit_count,
            "global_mean_probability": model.global_mean_probability,
            "bucket_counts": {
                bin_record.popularity_bucket: bin_record.count for bin_record in model.bins
            },
        },
        "eval_summary": eval_summary,
        "model": model.to_dict(),
        "leakage_guard": {
            "fit_eval_overlap": overlap,
            "same_split_eval_allowed": allow_same_split_eval,
            "requires_split_provenance": True,
            "default_refuses_fit_eval_overlap": True,
            "label_source": "feature.correctness_label when available; not required for fitting",
        },
        "api_called": False,
        "model_training": False,
        "server_executed": False,
        "is_experiment_result": False,
        "note": (
            "Split-audited CURE/TRUCE popularity residual scaffold. It fits a "
            "popularity-bucket mean confidence baseline on the requested fit split "
            "and applies it to evaluation splits only; it is not a learned method "
            "result or paper evidence."
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
    "POPULARITY_RESIDUAL_METHOD",
    "POPULARITY_RESIDUAL_SCHEMA_VERSION",
    "PopularityResidualBin",
    "PopularityResidualExample",
    "PopularityResidualModel",
    "apply_popularity_residual_model",
    "collect_popularity_residual_examples",
    "fit_popularity_residual_model",
    "residualize_feature_rows",
    "row_popularity_bucket",
    "row_popularity_percentile",
]
