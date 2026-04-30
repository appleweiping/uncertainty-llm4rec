"""Analysis utilities for grounded generative observation outputs."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from storyflow.metrics import (
    brier_score,
    cbu_tau,
    expected_calibration_error,
    ground_hit_rate,
    tail_underconfidence_gap,
    wbc_tau,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_path = Path(path)
    if not input_path.exists():
        return rows
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        return {}
    return json.loads(input_path.read_text(encoding="utf-8"))


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _mean(values: Iterable[float]) -> float | None:
    collected = [float(value) for value in values]
    if not collected:
        return None
    return sum(collected) / len(collected)


def _rate(values: Iterable[bool | int]) -> float | None:
    collected = [int(bool(value)) for value in values]
    if not collected:
        return None
    return sum(collected) / len(collected)


def _confidence(row: dict[str, Any]) -> float:
    return float(row.get("confidence") or 0.0)


def _correctness(row: dict[str, Any]) -> int:
    return int(row.get("correctness") or 0)


def _is_grounded(row: dict[str, Any]) -> bool:
    return bool(row.get("grounded_item_id"))


def _bucket(row: dict[str, Any]) -> str:
    return str(row.get("target_popularity_bucket") or "unknown")


def _repeat_group(row: dict[str, Any]) -> str:
    if "target_in_history" not in row:
        return "unknown"
    return "repeat_target" if bool(row.get("target_in_history")) else "non_repeat_target"


def _list_field(row: dict[str, Any], field: str) -> list[Any]:
    value = row.get(field)
    return value if isinstance(value, list) else []


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _counter(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return dict(sorted(counts.items()))


def reliability_bins(
    rows: Iterable[dict[str, Any]],
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Build fixed-width reliability diagram bins."""

    records = list(rows)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    output: list[dict[str, Any]] = []
    total = len(records)
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == 0:
            members = [
                row for row in records if lower <= _confidence(row) <= upper
            ]
        else:
            members = [
                row for row in records if lower < _confidence(row) <= upper
            ]
        mean_conf = _mean(_confidence(row) for row in members)
        accuracy = _rate(_correctness(row) for row in members)
        output.append(
            {
                "bin_index": bin_index,
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "fraction": (len(members) / total) if total else 0.0,
                "mean_confidence": mean_conf,
                "accuracy": accuracy,
                "calibration_error": (
                    abs(mean_conf - accuracy)
                    if mean_conf is not None and accuracy is not None
                    else None
                ),
            }
        )
    return output


def reliability_by_popularity_bucket(
    rows: Iterable[dict[str, Any]],
    *,
    n_bins: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    records = list(rows)
    return {
        bucket: reliability_bins(
            [row for row in records if _bucket(row) == bucket],
            n_bins=n_bins,
        )
        for bucket in ("head", "mid", "tail")
    }


def _status_counts(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field) or "unknown") for row in rows)
    return dict(sorted(counts.items()))


def candidate_diagnostic_rows(
    grounded_rows: Iterable[dict[str, Any]],
    *,
    input_rows: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Join grounded outputs with diagnostic candidate-set input metadata."""

    inputs_by_id = {
        str(row.get("input_id")): row
        for row in (input_rows or [])
        if row.get("input_id") is not None
    }
    rows: list[dict[str, Any]] = []
    for grounded in grounded_rows:
        input_id = str(grounded.get("input_id") or "")
        source = inputs_by_id.get(input_id, {})
        candidate_ids = [str(value) for value in _list_field(source, "catalog_candidate_item_ids")]
        candidate_titles = [str(value) for value in _list_field(source, "catalog_candidate_titles")]
        candidate_buckets = [
            str(value) for value in _list_field(source, "catalog_candidate_popularity_buckets")
        ]
        candidate_scores = [
            _float_or_none(value) for value in _list_field(source, "catalog_candidate_scores")
        ]
        policy = source.get("candidate_policy")
        policy_dict = policy if isinstance(policy, dict) else {}
        grounded_item_id = grounded.get("grounded_item_id")
        grounded_id = str(grounded_item_id) if grounded_item_id is not None else None
        target_item_id = source.get("target_item_id", grounded.get("target_item_id"))
        target_id = str(target_item_id) if target_item_id is not None else None
        history_ids = {str(value) for value in _list_field(source, "history_item_ids")}

        selected_index: int | None = None
        if grounded_id is not None and grounded_id in candidate_ids:
            selected_index = candidate_ids.index(grounded_id)
        target_in_candidates = (
            target_id in candidate_ids if target_id is not None and candidate_ids else None
        )
        selected_score = (
            candidate_scores[selected_index]
            if selected_index is not None and selected_index < len(candidate_scores)
            else None
        )
        selected_bucket = (
            candidate_buckets[selected_index]
            if selected_index is not None and selected_index < len(candidate_buckets)
            else None
        )
        selected_title = (
            candidate_titles[selected_index]
            if selected_index is not None and selected_index < len(candidate_titles)
            else None
        )
        top_score = candidate_scores[0] if candidate_scores else None
        top_bucket = candidate_buckets[0] if candidate_buckets else None
        rows.append(
            {
                "input_id": grounded.get("input_id"),
                "example_id": grounded.get("example_id"),
                "user_id": grounded.get("user_id"),
                "prompt_template": source.get(
                    "prompt_template",
                    grounded.get("prompt_template"),
                ),
                "candidate_policy_name": policy_dict.get(
                    "name",
                    policy_dict.get("candidate_policy", source.get("candidate_policy")),
                ),
                "candidate_context_available": bool(candidate_ids),
                "candidate_count": len(candidate_ids),
                "target_item_id": target_item_id,
                "target_in_candidates": target_in_candidates,
                "target_excluded_from_candidates": (
                    not target_in_candidates if target_in_candidates is not None else None
                ),
                "candidate_correctness_interpretable": not bool(
                    policy_dict.get(
                        "correctness_not_interpretable_without_unbiased_candidate_generation"
                    )
                ),
                "generated_title": grounded.get("generated_title"),
                "grounded_item_id": grounded_item_id,
                "grounding_status": grounded.get("grounding_status"),
                "confidence": grounded.get("confidence"),
                "correctness": grounded.get("correctness"),
                "target_popularity_bucket": grounded.get("target_popularity_bucket"),
                "generated_in_candidate_set": selected_index is not None,
                "selected_candidate_rank": (
                    selected_index + 1 if selected_index is not None else None
                ),
                "selected_candidate_score": selected_score,
                "selected_candidate_bucket": selected_bucket,
                "selected_candidate_title": selected_title,
                "selected_history_item": (
                    grounded_id in history_ids if grounded_id is not None else False
                ),
                "top_candidate_score": top_score,
                "top_candidate_bucket": top_bucket,
                "input_row_joined": bool(source),
            }
        )
    return rows


def candidate_diagnostic_summary(
    grounded_rows: Iterable[dict[str, Any]],
    *,
    input_rows: Iterable[dict[str, Any]] | None = None,
    high_confidence_tau: float = 0.7,
) -> dict[str, Any]:
    """Summarize whether candidate-prompt outputs select from the provided set."""

    rows = candidate_diagnostic_rows(grounded_rows, input_rows=input_rows)
    context_rows = [row for row in rows if row["candidate_context_available"]]
    selected_rows = [row for row in context_rows if row["generated_in_candidate_set"]]
    grounded_not_selected_rows = [
        row
        for row in context_rows
        if row.get("grounded_item_id") and not row["generated_in_candidate_set"]
    ]
    ungrounded_rows = [
        row for row in context_rows if not row.get("grounded_item_id")
    ]
    rank_values = [
        int(row["selected_candidate_rank"])
        for row in selected_rows
        if row.get("selected_candidate_rank") is not None
    ]
    selected_scores = [
        float(row["selected_candidate_score"])
        for row in selected_rows
        if row.get("selected_candidate_score") is not None
    ]
    selected_confidences = [
        float(row.get("confidence") or 0.0)
        for row in selected_rows
        if row.get("selected_candidate_score") is not None
    ]
    top_scores = [
        float(row["top_candidate_score"])
        for row in context_rows
        if row.get("top_candidate_score") is not None
    ]
    top_confidences = [
        float(row.get("confidence") or 0.0)
        for row in context_rows
        if row.get("top_candidate_score") is not None
    ]

    bucket_summaries: dict[str, dict[str, Any]] = {}
    for bucket in ("head", "mid", "tail", "unknown"):
        bucket_rows = [
            row
            for row in selected_rows
            if str(row.get("selected_candidate_bucket") or "unknown") == bucket
        ]
        if not bucket_rows:
            continue
        bucket_summaries[bucket] = {
            "count": len(bucket_rows),
            "mean_confidence": _mean(float(row.get("confidence") or 0.0) for row in bucket_rows),
            "correctness_rate": _rate(int(row.get("correctness") or 0) for row in bucket_rows),
            "mean_selected_candidate_rank": _mean(
                float(row.get("selected_candidate_rank") or 0.0) for row in bucket_rows
            ),
            "mean_selected_candidate_score": _mean(
                float(row.get("selected_candidate_score") or 0.0)
                for row in bucket_rows
                if row.get("selected_candidate_score") is not None
            ),
            "wrong_high_confidence_count": sum(
                int(row.get("correctness") or 0) == 0
                and float(row.get("confidence") or 0.0) >= high_confidence_tau
                for row in bucket_rows
            ),
        }

    rank_histogram = _counter(
        f"rank_{row['selected_candidate_rank']}"
        if row.get("selected_candidate_rank") is not None
        else (
            "ungrounded"
            if not row.get("grounded_item_id")
            else "grounded_not_in_candidates"
        )
        for row in context_rows
    )
    target_excluded_count = sum(
        row.get("target_in_candidates") is False for row in context_rows
    )
    summary = {
        "candidate_context_available": bool(context_rows),
        "count": len(rows),
        "input_joined_count": sum(row["input_row_joined"] for row in rows),
        "rows_with_candidate_context": len(context_rows),
        "rows_without_candidate_context": len(rows) - len(context_rows),
        "target_in_candidates_count": sum(
            row.get("target_in_candidates") is True for row in context_rows
        ),
        "target_excluded_from_candidates_count": target_excluded_count,
        "target_excluded_from_candidates_rate": (
            target_excluded_count / len(context_rows) if context_rows else None
        ),
        "generated_in_candidate_set_count": len(selected_rows),
        "generated_in_candidate_set_rate": (
            len(selected_rows) / len(context_rows) if context_rows else None
        ),
        "grounded_not_in_candidate_set_count": len(grounded_not_selected_rows),
        "ungrounded_with_candidate_context_count": len(ungrounded_rows),
        "selected_history_item_count": sum(
            row.get("selected_history_item") for row in context_rows
        ),
        "selected_history_item_rate": (
            sum(row.get("selected_history_item") for row in context_rows)
            / len(context_rows)
            if context_rows
            else None
        ),
        "mean_selected_candidate_rank": _mean(float(value) for value in rank_values),
        "selected_candidate_rank_histogram": rank_histogram,
        "selected_candidate_bucket_counts": _counter(
            row.get("selected_candidate_bucket") or "unknown"
            for row in selected_rows
        ),
        "selected_candidate_bucket_summary": bucket_summaries,
        "candidate_policy_counts": _counter(
            row.get("candidate_policy_name") or "unknown" for row in context_rows
        ),
        "prompt_template_counts": _counter(
            row.get("prompt_template") or "unknown" for row in context_rows
        ),
        "selected_candidate_score_confidence_slope": _slope(
            selected_scores,
            selected_confidences,
        ),
        "top_candidate_score_confidence_slope": _slope(
            top_scores,
            top_confidences,
        ),
        "target_correctness_interpretable_as_recommendation_accuracy": not any(
            not row.get("candidate_correctness_interpretable") for row in context_rows
        ),
        "note": (
            "Candidate diagnostics measure whether candidate-prompt outputs are "
            "groundable to the provided catalog context. If the target item is "
            "excluded from candidates, target-hit correctness is not a "
            "recommendation-accuracy metric."
        ),
    }
    return summary


def bucket_summary(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records = list(rows)
    summary: dict[str, dict[str, Any]] = {}
    for bucket in ("head", "mid", "tail", "unknown"):
        bucket_rows = [row for row in records if _bucket(row) == bucket]
        if not bucket_rows:
            continue
        summary[bucket] = {
            "count": len(bucket_rows),
            "mean_confidence": _mean(_confidence(row) for row in bucket_rows),
            "correctness_rate": _rate(_correctness(row) for row in bucket_rows),
            "ground_hit_rate": _rate(_is_grounded(row) for row in bucket_rows),
            "mean_grounding_score": _mean(
                float(row.get("grounding_score") or 0.0) for row in bucket_rows
            ),
            "wrong_high_confidence_count": sum(
                _correctness(row) == 0 and _confidence(row) >= 0.7
                for row in bucket_rows
            ),
            "correct_low_confidence_count": sum(
                _correctness(row) == 1 and _confidence(row) < 0.5
                for row in bucket_rows
            ),
        }
    return summary


def _slice_summary(
    rows: Iterable[dict[str, Any]],
    *,
    low_confidence_tau: float,
    high_confidence_tau: float,
    n_bins: int,
) -> dict[str, Any]:
    records = list(rows)
    probabilities = [_confidence(row) for row in records]
    labels = [_correctness(row) for row in records]
    grounded_flags = [_is_grounded(row) for row in records]
    if not records:
        return {
            "count": 0,
            "mean_confidence": None,
            "correctness_rate": None,
            "ground_hit_rate": None,
            "mean_grounding_score": None,
            "ece": None,
            "brier": None,
            "cbu_tau": None,
            "wbc_tau": None,
            "tail_underconfidence_gap": None,
            "wrong_high_confidence_count": 0,
            "correct_low_confidence_count": 0,
            "popularity_bucket_counts": {},
            "grounding_status_counts": {},
        }
    return {
        "count": len(records),
        "mean_confidence": _mean(probabilities),
        "correctness_rate": _rate(labels),
        "ground_hit_rate": _rate(grounded_flags),
        "mean_grounding_score": _mean(
            float(row.get("grounding_score") or 0.0) for row in records
        ),
        "ece": expected_calibration_error(probabilities, labels, n_bins=n_bins),
        "brier": brier_score(probabilities, labels),
        "cbu_tau": _finite_or_none(
            cbu_tau(probabilities, labels, tau=low_confidence_tau)
        ),
        "wbc_tau": _finite_or_none(
            wbc_tau(probabilities, labels, tau=high_confidence_tau)
        ),
        "tail_underconfidence_gap": _finite_or_none(
            _tail_underconfidence_gap_or_nan(records)
        ),
        "wrong_high_confidence_count": sum(
            _correctness(row) == 0 and _confidence(row) >= high_confidence_tau
            for row in records
        ),
        "correct_low_confidence_count": sum(
            _correctness(row) == 1 and _confidence(row) < low_confidence_tau
            for row in records
        ),
        "popularity_bucket_counts": _status_counts(
            records,
            "target_popularity_bucket",
        ),
        "grounding_status_counts": _status_counts(records, "grounding_status"),
    }


def repeat_target_summary(
    rows: Iterable[dict[str, Any]],
    *,
    low_confidence_tau: float = 0.5,
    high_confidence_tau: float = 0.7,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Summarize observation outputs by target-in-history repeat status."""

    records = list(rows)
    groups: dict[str, list[dict[str, Any]]] = {
        "all": records,
        "non_repeat_target": [
            row for row in records if _repeat_group(row) == "non_repeat_target"
        ],
        "repeat_target": [
            row for row in records if _repeat_group(row) == "repeat_target"
        ],
        "unknown": [
            row for row in records if _repeat_group(row) == "unknown"
        ],
        "same_timestamp_repeat_target": [
            row for row in records if bool(row.get("target_same_timestamp_as_history"))
        ],
    }
    summary = {
        group: _slice_summary(
            group_rows,
            low_confidence_tau=low_confidence_tau,
            high_confidence_tau=high_confidence_tau,
            n_bins=n_bins,
        )
        for group, group_rows in groups.items()
    }
    summary["repeat_metadata_presence"] = {
        "rows_with_target_in_history_field": sum(
            "target_in_history" in row for row in records
        ),
        "rows_with_target_history_occurrence_count": sum(
            "target_history_occurrence_count" in row for row in records
        ),
        "rows_with_same_timestamp_field": sum(
            "target_same_timestamp_as_history" in row for row in records
        ),
    }
    summary["note"] = (
        "Repeat-target slices separate ordinary next-item probes from repeat "
        "purchase or duplicate-review diagnostics. They are not conclusions by "
        "themselves."
    )
    return summary


def _slope(xs: list[float], ys: list[float]) -> dict[str, float | None]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return {"slope": None, "intercept": None, "correlation": None}
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0:
        return {"slope": None, "intercept": None, "correlation": None}
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = covariance / var_x
    intercept = mean_y - slope * mean_x
    correlation = (
        covariance / math.sqrt(var_x * var_y) if var_y > 0 else None
    )
    return {
        "slope": slope,
        "intercept": intercept,
        "correlation": correlation,
    }


def popularity_confidence_slope(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Estimate confidence-popularity coupling with lightweight stdlib math."""

    records = [
        row
        for row in rows
        if row.get("target_popularity") is not None
        and row.get("confidence") is not None
    ]
    xs = [math.log1p(float(row.get("target_popularity") or 0.0)) for row in records]
    ys = [_confidence(row) for row in records]
    univariate = _slope(xs, ys)

    grouped_means: dict[int, tuple[float, float]] = {}
    for label in (0, 1):
        group_rows = [row for row in records if _correctness(row) == label]
        grouped_means[label] = (
            _mean(math.log1p(float(row.get("target_popularity") or 0.0)) for row in group_rows)
            or 0.0,
            _mean(_confidence(row) for row in group_rows) or 0.0,
        )
    residual_xs: list[float] = []
    residual_ys: list[float] = []
    for row, x, y in zip(records, xs, ys):
        mean_x, mean_y = grouped_means[_correctness(row)]
        residual_xs.append(x - mean_x)
        residual_ys.append(y - mean_y)
    controlled = _slope(residual_xs, residual_ys)
    return {
        "n": len(records),
        "x": "log1p(target_popularity)",
        "y": "confidence",
        "univariate": univariate,
        "correctness_residualized": controlled,
        "note": (
            "Exploratory stdlib slope for analysis sanity. It is not a causal "
            "claim or paper result."
        ),
    }


def observation_source_profile(
    rows: Iterable[dict[str, Any]],
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify an analyzed run without upgrading it to a result claim."""

    records = list(rows)
    manifest = manifest or {}
    first = records[0] if records else {}
    provider = manifest.get("provider") or first.get("provider")
    baseline = manifest.get("baseline") or first.get("baseline")
    model = manifest.get("model") or first.get("model") or baseline
    dry_run = manifest.get("dry_run", first.get("dry_run"))
    api_called = manifest.get("api_called", first.get("api_called"))
    run_stage = manifest.get("run_stage") or manifest.get("stage")
    server_executed = bool(
        manifest.get("server_executed")
        or manifest.get("model_inference_run")
        or manifest.get("server_run")
    )
    model_training = bool(
        manifest.get("model_training")
        or manifest.get("model_training_run")
        or manifest.get("training_run")
    )

    if str(provider) == "baseline" or baseline:
        source_kind = "baseline_observation"
        confidence_semantics = "non_calibrated_baseline_proxy"
        claim_scope = "baseline_sanity_or_protocol_artifact"
        confidence_is_calibrated = False
    elif str(provider) == "mock":
        source_kind = "mock_observation"
        confidence_semantics = "mock_sanity_confidence"
        claim_scope = "schema_sanity"
        confidence_is_calibrated = False
    elif dry_run is True:
        source_kind = "api_dry_run"
        confidence_semantics = "dry_run_placeholder_confidence"
        claim_scope = "schema_sanity"
        confidence_is_calibrated = False
    elif server_executed:
        source_kind = "server_observation"
        confidence_semantics = "model_reported_confidence"
        claim_scope = str(run_stage or "server_artifact")
        confidence_is_calibrated = False
    elif api_called is True:
        source_kind = "api_observation"
        confidence_semantics = "model_reported_confidence"
        claim_scope = str(run_stage or "pilot_or_full_artifact")
        confidence_is_calibrated = False
    else:
        source_kind = "observation_artifact"
        confidence_semantics = "unknown_or_unlabeled_confidence"
        claim_scope = "unlabeled_artifact"
        confidence_is_calibrated = False

    return {
        "source_kind": source_kind,
        "claim_scope": claim_scope,
        "provider": provider,
        "model": model,
        "baseline": baseline,
        "run_stage": run_stage,
        "dry_run": dry_run,
        "api_called": api_called,
        "server_executed": server_executed,
        "model_training": model_training,
        "confidence_semantics": confidence_semantics,
        "confidence_is_calibrated": confidence_is_calibrated,
        "title_grounding_required": True,
        "is_experiment_result": False,
        "is_paper_result": False,
    }


def observation_claim_guardrails(
    *,
    source_profile: dict[str, Any],
    candidate_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record conservative claim boundaries for an analysis artifact."""

    candidate_summary = candidate_summary or {}
    source_kind = str(source_profile.get("source_kind") or "")
    target_accuracy_allowed = bool(
        candidate_summary.get(
            "target_correctness_interpretable_as_recommendation_accuracy",
            True,
        )
    )
    if source_kind in {"mock_observation", "api_dry_run"}:
        target_accuracy_allowed = False
    return {
        "is_experiment_result": False,
        "is_paper_result": False,
        "grounding_required_before_correctness": True,
        "confidence_is_calibrated": bool(source_profile.get("confidence_is_calibrated")),
        "confidence_requires_calibration": True,
        "baseline_confidence_is_proxy": source_kind == "baseline_observation",
        "target_correctness_interpretable_as_recommendation_accuracy": target_accuracy_allowed,
        "requires_source_manifest_for_claims": True,
        "requires_grounded_predictions_for_claims": True,
        "forbidden_claims": [
            "method_improves_performance",
            "full_result_without_manifest",
            "server_run_without_logs",
            "calibrated_confidence_without_calibrator",
        ],
    }


def risk_case_slices(
    rows: Iterable[dict[str, Any]],
    *,
    low_confidence_tau: float = 0.5,
    high_confidence_tau: float = 0.7,
    max_cases: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    records = list(rows)

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_id": row.get("input_id"),
            "example_id": row.get("example_id"),
            "user_id": row.get("user_id"),
            "provider": row.get("provider"),
            "model": row.get("model"),
            "generated_title": row.get("generated_title"),
            "target_title": row.get("target_title"),
            "grounded_item_id": row.get("grounded_item_id"),
            "grounding_status": row.get("grounding_status"),
            "grounding_score": row.get("grounding_score"),
            "confidence": row.get("confidence"),
            "correctness": row.get("correctness"),
            "target_popularity": row.get("target_popularity"),
            "target_popularity_bucket": row.get("target_popularity_bucket"),
            "target_in_history": row.get("target_in_history"),
            "target_history_occurrence_count": row.get(
                "target_history_occurrence_count"
            ),
            "target_same_timestamp_as_history": row.get(
                "target_same_timestamp_as_history"
            ),
            "history_duplicate_item_count": row.get("history_duplicate_item_count"),
        }

    wrong_high = sorted(
        [
            row
            for row in records
            if _correctness(row) == 0 and _confidence(row) >= high_confidence_tau
        ],
        key=lambda row: (-_confidence(row), str(row.get("input_id") or "")),
    )
    correct_low = sorted(
        [
            row
            for row in records
            if _correctness(row) == 1 and _confidence(row) < low_confidence_tau
        ],
        key=lambda row: (_confidence(row), str(row.get("input_id") or "")),
    )
    grounding_failures = [
        row for row in records if not _is_grounded(row)
    ]
    return {
        "wrong_high_confidence": [compact(row) for row in wrong_high[:max_cases]],
        "correct_low_confidence": [compact(row) for row in correct_low[:max_cases]],
        "grounding_failures": [compact(row) for row in grounding_failures[:max_cases]],
    }


def _tail_underconfidence_gap_or_nan(rows: Iterable[dict[str, Any]]) -> float:
    valid_rows = [
        row for row in rows if _bucket(row) in {"head", "mid", "tail"}
    ]
    if not valid_rows:
        return math.nan
    return tail_underconfidence_gap(
        [_confidence(row) for row in valid_rows],
        [_correctness(row) for row in valid_rows],
        [_bucket(row) for row in valid_rows],
    )


def summarize_observation_records(
    grounded_rows: Iterable[dict[str, Any]],
    *,
    failed_rows: Iterable[dict[str, Any]] | None = None,
    manifest: dict[str, Any] | None = None,
    input_rows: Iterable[dict[str, Any]] | None = None,
    low_confidence_tau: float = 0.5,
    high_confidence_tau: float = 0.7,
    n_bins: int = 10,
    max_cases: int = 20,
) -> dict[str, Any]:
    records = list(grounded_rows)
    failures = list(failed_rows or [])
    manifest = manifest or {}
    if not records and not failures:
        raise ValueError("analysis requires grounded rows or failed rows")

    probabilities = [_confidence(row) for row in records]
    labels = [_correctness(row) for row in records]
    grounded_flags = [_is_grounded(row) for row in records]

    confidence_metrics: dict[str, Any] = {}
    if records:
        confidence_metrics = {
            "ece": expected_calibration_error(probabilities, labels, n_bins=n_bins),
            "brier": brier_score(probabilities, labels),
            "cbu_tau": _finite_or_none(
                cbu_tau(probabilities, labels, tau=low_confidence_tau)
            ),
            "wbc_tau": _finite_or_none(
                wbc_tau(probabilities, labels, tau=high_confidence_tau)
            ),
            "tail_underconfidence_gap": _finite_or_none(
                _tail_underconfidence_gap_or_nan(records)
            ),
        }

    parse_failures = [
        row for row in failures if str(row.get("failure_stage")) == "parse"
    ]
    provider_failures = [
        row for row in failures if str(row.get("failure_stage")) == "provider"
    ]
    source_profile = observation_source_profile(records, manifest=manifest)
    candidate_summary = candidate_diagnostic_summary(
        records,
        input_rows=input_rows,
        high_confidence_tau=high_confidence_tau,
    )
    claim_guardrails = observation_claim_guardrails(
        source_profile=source_profile,
        candidate_summary=candidate_summary,
    )
    summary = {
        "created_at_utc": utc_now_iso(),
        "provider": source_profile.get("provider"),
        "model": source_profile.get("model"),
        "baseline": source_profile.get("baseline"),
        "dry_run": source_profile.get("dry_run"),
        "api_called": source_profile.get("api_called"),
        "input_jsonl": manifest.get("input_jsonl"),
        "source_output_dir": manifest.get("output_dir"),
        "source_profile": source_profile,
        "claim_guardrails": claim_guardrails,
        "count": len(records),
        "failed_count": len(failures),
        "parse_failure_count": len(parse_failures),
        "provider_failure_count": len(provider_failures),
        "ground_hit": ground_hit_rate(grounded_flags) if records else None,
        "correctness": _rate(labels),
        "mean_confidence": _mean(probabilities),
        "confidence_metrics": confidence_metrics,
        "low_confidence_tau": low_confidence_tau,
        "high_confidence_tau": high_confidence_tau,
        "grounding_summary": {
            "status_counts": _status_counts(records, "grounding_status"),
            "failure_count": sum(not flag for flag in grounded_flags),
            "mean_grounding_score": _mean(
                float(row.get("grounding_score") or 0.0) for row in records
            ),
        },
        "parse_summary": {
            "parse_strategy_counts": _status_counts(records, "parse_strategy"),
            "failure_stage_counts": _status_counts(failures, "failure_stage"),
            "parse_failure_examples": parse_failures[:max_cases],
        },
        "bucket_summary": bucket_summary(records),
        "repeat_target_summary": repeat_target_summary(
            records,
            low_confidence_tau=low_confidence_tau,
            high_confidence_tau=high_confidence_tau,
            n_bins=n_bins,
        ),
        "candidate_diagnostic_summary": candidate_summary,
        "quadrant_counts": {
            "correct_confident": sum(
                label == 1 and prob >= high_confidence_tau
                for prob, label in zip(probabilities, labels)
            ),
            "correct_low_confidence": sum(
                label == 1 and prob < low_confidence_tau
                for prob, label in zip(probabilities, labels)
            ),
            "wrong_high_confidence": sum(
                label == 0 and prob >= high_confidence_tau
                for prob, label in zip(probabilities, labels)
            ),
            "wrong_low_confidence": sum(
                label == 0 and prob < low_confidence_tau
                for prob, label in zip(probabilities, labels)
            ),
            "intermediate_confidence": sum(
                low_confidence_tau <= prob < high_confidence_tau
                for prob in probabilities
            ),
        },
        "popularity_confidence_slope": popularity_confidence_slope(records),
        "reliability_overall": reliability_bins(records, n_bins=n_bins),
        "reliability_by_popularity_bucket": reliability_by_popularity_bucket(
            records,
            n_bins=n_bins,
        ),
        "risk_cases": risk_case_slices(
            records,
            low_confidence_tau=low_confidence_tau,
            high_confidence_tau=high_confidence_tau,
            max_cases=max_cases,
        ),
        "is_experiment_result": False,
        "note": (
            "Analysis output is a reproducibility artifact. Mock/dry-run inputs "
            "are not paper evidence; real pilot/full status must come from the "
            "source manifest."
        ),
    }
    return summary


def observation_analysis_markdown(summary: dict[str, Any]) -> str:
    metrics = summary.get("confidence_metrics", {})
    bucket_rows = []
    for bucket, row in summary.get("bucket_summary", {}).items():
        bucket_rows.append(
            "| {bucket} | {count} | {conf} | {corr} | {ground} | {whc} | {clc} |".format(
                bucket=bucket,
                count=row.get("count"),
                conf=row.get("mean_confidence"),
                corr=row.get("correctness_rate"),
                ground=row.get("ground_hit_rate"),
                whc=row.get("wrong_high_confidence_count"),
                clc=row.get("correct_low_confidence_count"),
            )
        )
    repeat_rows = []
    for group in (
        "non_repeat_target",
        "repeat_target",
        "same_timestamp_repeat_target",
        "unknown",
    ):
        row = summary.get("repeat_target_summary", {}).get(group)
        if not row or not row.get("count"):
            continue
        repeat_rows.append(
            "| {group} | {count} | {conf} | {corr} | {ground} | {whc} | {clc} |".format(
                group=group,
                count=row.get("count"),
                conf=row.get("mean_confidence"),
                corr=row.get("correctness_rate"),
                ground=row.get("ground_hit_rate"),
                whc=row.get("wrong_high_confidence_count"),
                clc=row.get("correct_low_confidence_count"),
            )
        )
    slope = summary.get("popularity_confidence_slope", {})
    univariate = slope.get("univariate", {})
    controlled = slope.get("correctness_residualized", {})
    candidate = summary.get("candidate_diagnostic_summary", {})
    source_profile = summary.get("source_profile", {})
    guardrails = summary.get("claim_guardrails", {})
    lines = [
        "# Observation Analysis Report",
        "",
        "This report summarizes grounded title-generation observation outputs. If the source run is mock or dry-run, this is only a schema/analysis sanity artifact and not paper evidence.",
        "",
        "## Source",
        "",
        f"- Provider: {summary.get('provider')}",
        f"- Model: {summary.get('model')}",
        f"- Baseline: {summary.get('baseline')}",
        f"- Source kind: {source_profile.get('source_kind')}",
        f"- Confidence semantics: {source_profile.get('confidence_semantics')}",
        f"- Dry-run: {summary.get('dry_run')}",
        f"- API called: {summary.get('api_called')}",
        f"- Source output dir: {summary.get('source_output_dir')}",
        f"- Is paper result: {guardrails.get('is_paper_result')}",
        f"- Confidence calibrated: {guardrails.get('confidence_is_calibrated')}",
        "",
        "## Summary",
        "",
        f"- Grounded rows: {summary.get('count')}",
        f"- Failed rows: {summary.get('failed_count')}",
        f"- Parse failures: {summary.get('parse_failure_count')}",
        f"- GroundHit: {summary.get('ground_hit')}",
        f"- Correctness: {summary.get('correctness')}",
        f"- Mean confidence: {summary.get('mean_confidence')}",
        f"- ECE: {metrics.get('ece')}",
        f"- Brier: {metrics.get('brier')}",
        f"- CBU_tau: {metrics.get('cbu_tau')}",
        f"- WBC_tau: {metrics.get('wbc_tau')}",
        f"- Tail Underconfidence Gap: {metrics.get('tail_underconfidence_gap')}",
        "",
        "## Quadrants",
        "",
    ]
    for key, value in summary.get("quadrant_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Popularity Buckets",
            "",
            "| bucket | count | mean confidence | correctness | GroundHit | wrong-high-conf | correct-low-conf |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *bucket_rows,
            "",
            "## Repeat Target Slices",
            "",
            "| repeat group | count | mean confidence | correctness | GroundHit | wrong-high-conf | correct-low-conf |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *(repeat_rows or ["| none | 0 |  |  |  |  |  |"]),
            "",
            "## Popularity-Confidence Slope",
            "",
            f"- Univariate slope: {univariate.get('slope')}",
            f"- Univariate correlation: {univariate.get('correlation')}",
            f"- Correctness-residualized slope: {controlled.get('slope')}",
            f"- Correctness-residualized correlation: {controlled.get('correlation')}",
            "",
            "## Candidate Prompt Diagnostics",
            "",
            f"- Candidate context available: {candidate.get('candidate_context_available')}",
            f"- Rows with candidate context: {candidate.get('rows_with_candidate_context')}",
            f"- Generated-in-candidate-set rate: {candidate.get('generated_in_candidate_set_rate')}",
            f"- Target-excluded rate: {candidate.get('target_excluded_from_candidates_rate')}",
            f"- Selected history-item rate: {candidate.get('selected_history_item_rate')}",
            f"- Mean selected candidate rank: {candidate.get('mean_selected_candidate_rank')}",
            f"- Target correctness interpretable as recommendation accuracy: {candidate.get('target_correctness_interpretable_as_recommendation_accuracy')}",
            "",
            "## Failure Summary",
            "",
            f"- Grounding status counts: {summary.get('grounding_summary', {}).get('status_counts')}",
            f"- Parse strategy counts: {summary.get('parse_summary', {}).get('parse_strategy_counts')}",
            f"- Failure stage counts: {summary.get('parse_summary', {}).get('failure_stage_counts')}",
            "",
        "## Reminder",
        "",
            "Do not treat mock, dry-run, synthetic, or baseline-proxy outputs as calibrated confidence or paper results.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_observation_run(
    *,
    grounded_jsonl: str | Path,
    output_dir: str | Path,
    failed_jsonl: str | Path | None = None,
    manifest_json: str | Path | None = None,
    input_jsonl: str | Path | None = None,
    low_confidence_tau: float = 0.5,
    high_confidence_tau: float = 0.7,
    n_bins: int = 10,
    max_cases: int = 20,
) -> dict[str, Any]:
    grounded_path = Path(grounded_jsonl)
    failed_path = Path(failed_jsonl) if failed_jsonl else None
    manifest_path = Path(manifest_json) if manifest_json else None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    grounded_rows = read_jsonl(grounded_path)
    failed_rows = read_jsonl(failed_path) if failed_path else []
    manifest = load_json(manifest_path) if manifest_path else {}
    input_path = Path(input_jsonl) if input_jsonl else None
    if input_path is None and manifest.get("input_jsonl"):
        input_path = Path(str(manifest["input_jsonl"]))
    input_rows = read_jsonl(input_path) if input_path and input_path.exists() else []
    summary = summarize_observation_records(
        grounded_rows,
        failed_rows=failed_rows,
        manifest=manifest,
        input_rows=input_rows,
        low_confidence_tau=low_confidence_tau,
        high_confidence_tau=high_confidence_tau,
        n_bins=n_bins,
        max_cases=max_cases,
    )

    summary.update(
        {
            "grounded_jsonl": str(grounded_path),
            "failed_jsonl": str(failed_path) if failed_path else None,
            "source_manifest_json": str(manifest_path) if manifest_path else None,
            "source_input_jsonl": str(input_path) if input_path else None,
        }
    )
    summary_path = output_path / "analysis_summary.json"
    reliability_path = output_path / "reliability_diagram.json"
    bucket_path = output_path / "bucket_summary.json"
    repeat_path = output_path / "repeat_summary.json"
    candidate_summary_path = output_path / "candidate_diagnostic_summary.json"
    candidate_cases_path = output_path / "candidate_diagnostic_cases.jsonl"
    risk_path = output_path / "risk_cases.jsonl"
    report_path = output_path / "report.md"
    manifest_out_path = output_path / "analysis_manifest.json"

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    reliability_path.write_text(
        json.dumps(
            {
                "overall": summary["reliability_overall"],
                "by_popularity_bucket": summary["reliability_by_popularity_bucket"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    bucket_path.write_text(
        json.dumps(
            summary["bucket_summary"],
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    repeat_path.write_text(
        json.dumps(
            summary["repeat_target_summary"],
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    candidate_summary_path.write_text(
        json.dumps(
            summary["candidate_diagnostic_summary"],
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_jsonl(
        candidate_cases_path,
        candidate_diagnostic_rows(grounded_rows, input_rows=input_rows),
    )
    risk_rows: list[dict[str, Any]] = []
    for slice_name, rows in summary["risk_cases"].items():
        for row in rows:
            risk_rows.append({"slice": slice_name, **row})
    write_jsonl(risk_path, risk_rows)
    report_path.write_text(
        observation_analysis_markdown(summary),
        encoding="utf-8",
    )

    analysis_manifest = {
        "created_at_utc": utc_now_iso(),
        "analysis_dir": str(output_path),
        "summary": str(summary_path),
        "reliability_diagram": str(reliability_path),
        "bucket_summary": str(bucket_path),
        "repeat_summary": str(repeat_path),
        "candidate_diagnostic_summary": str(candidate_summary_path),
        "candidate_diagnostic_cases": str(candidate_cases_path),
        "risk_cases": str(risk_path),
        "report": str(report_path),
        "source_grounded_jsonl": str(grounded_path),
        "source_failed_jsonl": str(failed_path) if failed_path else None,
        "source_manifest_json": str(manifest_path) if manifest_path else None,
        "source_input_jsonl": str(input_path) if input_path else None,
        "provider": summary.get("provider"),
        "model": summary.get("model"),
        "baseline": summary.get("baseline"),
        "source_kind": summary.get("source_profile", {}).get("source_kind"),
        "claim_scope": summary.get("source_profile", {}).get("claim_scope"),
        "confidence_semantics": summary.get("source_profile", {}).get(
            "confidence_semantics"
        ),
        "confidence_is_calibrated": summary.get("source_profile", {}).get(
            "confidence_is_calibrated"
        ),
        "claim_guardrails": summary.get("claim_guardrails"),
        "dry_run": summary.get("dry_run"),
        "api_called": summary.get("api_called"),
        "count": summary.get("count"),
        "failed_count": summary.get("failed_count"),
        "is_experiment_result": False,
        "note": "Analysis artifact only; mock/dry-run inputs are not paper evidence.",
    }
    manifest_out_path.write_text(
        json.dumps(analysis_manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return analysis_manifest
