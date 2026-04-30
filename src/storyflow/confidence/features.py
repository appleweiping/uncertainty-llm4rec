"""Build CURE/TRUCE feature records from grounded observation outputs.

The builder converts API, mock, server, or baseline grounded predictions into
the shared ``ExposureConfidenceFeatures`` schema. It is a feature-contract
layer only: it does not call APIs, train calibrators, or turn observation rows
into paper evidence.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from storyflow.confidence.exposure import (
    CureTruceWeights,
    ExposureConfidenceFeatures,
    score_cure_truce_candidate,
)
from storyflow.observation import load_catalog_rows, read_jsonl, utc_now_iso, write_jsonl

FEATURE_SCHEMA_VERSION = "cure_truce_feature_v1"


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _clip01(value: float | None) -> float | None:
    if value is None:
        return None
    return min(1.0, max(0.0, value))


def _bool_grounded(row: Mapping[str, Any]) -> bool:
    status = str(row.get("grounding_status") or "").lower()
    return bool(row.get("grounded_item_id")) and status not in {
        "ambiguous",
        "ungrounded",
        "out_of_catalog",
        "parse_failed",
    }


@dataclass(frozen=True, slots=True)
class CatalogFeatureIndex:
    """Catalog lookup used to attach generated-item popularity features."""

    rows_by_item_id: Mapping[str, Mapping[str, Any]]
    popularity_percentile_by_item_id: Mapping[str, float]

    @classmethod
    def from_csv(cls, catalog_csv: str | Path) -> "CatalogFeatureIndex":
        rows = load_catalog_rows(catalog_csv)
        rows_by_item_id = {str(row["item_id"]): dict(row) for row in rows}
        popularities = sorted(float(row.get("popularity") or 0.0) for row in rows)
        if not popularities:
            return cls(rows_by_item_id={}, popularity_percentile_by_item_id={})
        minimum = popularities[0]
        maximum = popularities[-1]
        percentile_by_id: dict[str, float] = {}
        for row in rows:
            item_id = str(row["item_id"])
            popularity = float(row.get("popularity") or 0.0)
            if maximum == minimum:
                percentile = 0.5
            else:
                percentile = bisect.bisect_right(popularities, popularity) / len(popularities)
            percentile_by_id[item_id] = min(1.0, max(0.0, percentile))
        return cls(
            rows_by_item_id=rows_by_item_id,
            popularity_percentile_by_item_id=percentile_by_id,
        )


def _input_rows_by_id(input_jsonl: str | Path | None) -> dict[str, dict[str, Any]]:
    if input_jsonl is None:
        return {}
    return {str(row.get("input_id")): row for row in read_jsonl(input_jsonl)}


def _generated_popularity_features(
    row: Mapping[str, Any],
    *,
    catalog: CatalogFeatureIndex | None,
) -> tuple[float | None, str | None, dict[str, Any]]:
    item_id = row.get("grounded_item_id")
    item_key = str(item_id) if item_id not in (None, "") else None
    metadata: dict[str, Any] = {
        "generated_popularity": None,
        "generated_popularity_bucket": None,
        "popularity_source": "unknown",
    }
    if item_key and catalog is not None and item_key in catalog.rows_by_item_id:
        catalog_row = catalog.rows_by_item_id[item_key]
        percentile = catalog.popularity_percentile_by_item_id.get(item_key)
        bucket = str(catalog_row.get("popularity_bucket") or "") or None
        metadata.update(
            {
                "generated_popularity": catalog_row.get("popularity"),
                "generated_popularity_bucket": bucket,
                "popularity_source": "catalog_grounded_item",
            }
        )
        return percentile, bucket, metadata

    if item_key and item_key == str(row.get("target_item_id") or ""):
        bucket = str(row.get("target_popularity_bucket") or "") or None
        metadata.update(
            {
                "generated_popularity": row.get("target_popularity"),
                "generated_popularity_bucket": bucket,
                "popularity_source": "target_popularity_when_correct",
            }
        )
        return None, bucket, metadata

    return None, None, metadata


def _preference_score(row: Mapping[str, Any]) -> tuple[float, str]:
    for key in ("preference_score", "utility_score", "ranker_score"):
        value = _float_or_none(row.get(key))
        if value is not None:
            clipped = _clip01(value)
            if clipped is not None:
                return clipped, key

    score_source = str(row.get("baseline_score_source") or "")
    baseline_score = _float_or_none(row.get("baseline_score"))
    if baseline_score is not None and score_source.startswith("ranking_jsonl"):
        return 1.0 / (1.0 + math.exp(-baseline_score)), "ranking_jsonl_sigmoid_baseline_score"

    rank = _float_or_none(row.get("baseline_selected_rank"))
    if rank is not None and rank >= 1:
        return min(1.0, 1.0 / math.sqrt(rank)), "baseline_rank_proxy"

    return 0.5, "neutral_missing_preference_score"


def _generation_confidence(row: Mapping[str, Any]) -> tuple[float | None, str | None]:
    for key in (
        "generation_confidence",
        "token_confidence",
        "sequence_probability",
        "mean_token_probability",
        "sampling_consistency",
    ):
        value = _float_or_none(row.get(key))
        if value is not None:
            return _clip01(value), key
    return None, None


def _history_features(
    row: Mapping[str, Any],
    input_record: Mapping[str, Any] | None,
) -> tuple[float | None, float | None, str]:
    item_id = row.get("grounded_item_id")
    if not item_id or input_record is None:
        return None, None, "missing_input_history"
    history_ids = {str(value) for value in input_record.get("history_item_ids", [])}
    if not history_ids:
        return None, None, "empty_input_history"
    in_history = str(item_id) in history_ids
    return (1.0 if in_history else 0.0), (0.0 if in_history else 1.0), "history_item_membership"


def _is_correctness_label(value: Any) -> int | None:
    if value is None or value == "":
        return None
    label = int(value)
    if label not in (0, 1):
        raise ValueError("correctness must be 0 or 1 when present")
    return label


def feature_from_grounded_row(
    row: Mapping[str, Any],
    *,
    catalog: CatalogFeatureIndex | None = None,
    input_record: Mapping[str, Any] | None = None,
) -> tuple[ExposureConfidenceFeatures, dict[str, Any]]:
    """Convert one grounded observation row into CURE/TRUCE features."""

    user_id = str(row.get("user_id") or "")
    if not user_id:
        raise ValueError("grounded row missing user_id")
    grounded = _bool_grounded(row)
    grounded_item_id = str(row.get("grounded_item_id")) if row.get("grounded_item_id") else None
    item_id = grounded_item_id if grounded else None
    popularity_percentile, popularity_bucket, popularity_metadata = _generated_popularity_features(
        row,
        catalog=catalog,
    )
    preference_score, preference_source = _preference_score(row)
    generation_confidence, generation_source = _generation_confidence(row)
    history_alignment, novelty_score, history_source = _history_features(row, input_record)
    metadata = {
        **popularity_metadata,
        "input_id": row.get("input_id"),
        "example_id": row.get("example_id"),
        "split": row.get("split"),
        "provider": row.get("provider"),
        "model": row.get("model"),
        "baseline": row.get("baseline"),
        "target_item_id": row.get("target_item_id"),
        "target_title": row.get("target_title"),
        "target_popularity": row.get("target_popularity"),
        "target_popularity_bucket": row.get("target_popularity_bucket"),
        "grounding_status": row.get("grounding_status"),
        "preference_score_source": preference_source,
        "generation_confidence_source": generation_source,
        "history_feature_source": history_source,
        "is_experiment_result": bool(row.get("is_experiment_result", False)),
    }
    features = ExposureConfidenceFeatures(
        user_id=user_id,
        item_id=item_id,
        generated_title=str(row.get("generated_title") or "") or None,
        preference_score=preference_score,
        verbal_confidence=_clip01(_float_or_none(row.get("confidence"))),
        generation_confidence=generation_confidence,
        grounding_confidence=_clip01(_float_or_none(row.get("grounding_score"))),
        grounding_ambiguity=_clip01(_float_or_none(row.get("grounding_ambiguity"))),
        popularity_percentile=popularity_percentile,
        popularity_bucket=popularity_bucket,
        history_alignment=history_alignment,
        novelty_score=novelty_score,
        correctness_label=_is_correctness_label(row.get("correctness")),
        is_grounded=grounded,
        metadata=metadata,
    )
    return features, metadata


def confidence_feature_record(
    row: Mapping[str, Any],
    *,
    catalog: CatalogFeatureIndex | None = None,
    input_record: Mapping[str, Any] | None = None,
    weights: CureTruceWeights | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable feature row plus deterministic scaffold score."""

    features, metadata = feature_from_grounded_row(
        row,
        catalog=catalog,
        input_record=input_record,
    )
    score = score_cure_truce_candidate(features, weights)
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_id": row.get("input_id"),
        "example_id": row.get("example_id"),
        "user_id": row.get("user_id"),
        "split": row.get("split"),
        "provider": row.get("provider"),
        "model": row.get("model"),
        "baseline": row.get("baseline"),
        "grounded_item_id": row.get("grounded_item_id"),
        "target_item_id": row.get("target_item_id"),
        "feature": features.to_dict(),
        "score": score.to_dict(),
        "metadata": metadata,
        "is_experiment_result": False,
    }


def build_confidence_features(
    *,
    grounded_jsonl: str | Path,
    output_jsonl: str | Path,
    manifest_json: str | Path,
    catalog_csv: str | Path | None = None,
    input_jsonl: str | Path | None = None,
    max_examples: int | None = None,
    weights: CureTruceWeights | None = None,
) -> dict[str, Any]:
    """Build feature JSONL and manifest from grounded predictions."""

    grounded_path = Path(grounded_jsonl)
    rows = read_jsonl(grounded_path)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        rows = rows[:max_examples]
    if not rows:
        raise ValueError("grounded_jsonl contains no rows")
    catalog = CatalogFeatureIndex.from_csv(catalog_csv) if catalog_csv is not None else None
    inputs_by_id = _input_rows_by_id(input_jsonl)

    feature_rows = [
        confidence_feature_record(
            row,
            catalog=catalog,
            input_record=inputs_by_id.get(str(row.get("input_id"))),
            weights=weights,
        )
        for row in rows
    ]
    write_jsonl(output_jsonl, feature_rows)

    action_counts = Counter(str(row["score"]["action"]) for row in feature_rows)
    popularity_sources = Counter(
        str(row["metadata"].get("popularity_source") or "unknown") for row in feature_rows
    )
    provider_counts = Counter(str(row.get("provider") or "unknown") for row in feature_rows)
    manifest = {
        "created_at_utc": utc_now_iso(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "grounded_jsonl": str(grounded_path),
        "input_jsonl": str(input_jsonl) if input_jsonl is not None else None,
        "catalog_csv": str(catalog_csv) if catalog_csv is not None else None,
        "output_jsonl": str(output_jsonl),
        "manifest_json": str(manifest_json),
        "requested_row_count": len(rows),
        "feature_count": len(feature_rows),
        "grounded_feature_count": sum(bool(row["feature"]["is_grounded"]) for row in feature_rows),
        "ungrounded_feature_count": sum(
            not bool(row["feature"]["is_grounded"]) for row in feature_rows
        ),
        "missing_generated_popularity_count": sum(
            row["metadata"].get("popularity_source") == "unknown" for row in feature_rows
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "popularity_source_counts": dict(sorted(popularity_sources.items())),
        "provider_counts": dict(sorted(provider_counts.items())),
        "api_called": False,
        "model_training": False,
        "server_executed": False,
        "is_experiment_result": False,
        "note": (
            "CURE/TRUCE feature extraction from existing grounded predictions. "
            "No API call, model training, or paper result."
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
    "FEATURE_SCHEMA_VERSION",
    "CatalogFeatureIndex",
    "build_confidence_features",
    "confidence_feature_record",
    "feature_from_grounded_row",
]
