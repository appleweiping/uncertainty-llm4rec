"""Shared confidence diagnostics for CURE/TRUCE feature-row contracts."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

from storyflow.metrics import selective_risk_summary


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return min(1.0, max(0.0, numeric))


def _label_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    label = int(value)
    if label not in (0, 1):
        raise ValueError("selective-risk labels must be 0, 1, or missing")
    return label


def _compact_summary(probabilities: list[float], labels: list[int]) -> dict[str, Any]:
    summary = selective_risk_summary(probabilities, labels)
    return {
        key: value
        for key, value in summary.items()
        if key != "curve"
    }


def selective_risk_diagnostics(
    entries: Iterable[Mapping[str, Any]],
    *,
    confidence_key: str = "selected_confidence",
    label_key: str = "correctness_label",
    bucket_key: str = "popularity_bucket",
    requested_confidence_source: str | None = None,
) -> dict[str, Any]:
    """Summarize selective risk overall and by popularity bucket.

    The compact form intentionally omits the full curve so rerank/triage
    manifests stay small. The observation-analysis layer still writes the full
    curve when a run-level artifact is needed.
    """

    probabilities: list[float] = []
    labels: list[int] = []
    by_bucket: dict[str, dict[str, list[float] | list[int]]] = defaultdict(
        lambda: {"probabilities": [], "labels": []}
    )
    total_entries = 0
    skipped_missing_confidence = 0
    skipped_missing_label = 0

    for entry in entries:
        total_entries += 1
        probability = _float_or_none(entry.get(confidence_key))
        label = _label_or_none(entry.get(label_key))
        if probability is None:
            skipped_missing_confidence += 1
            continue
        if label is None:
            skipped_missing_label += 1
            continue
        probabilities.append(probability)
        labels.append(label)
        bucket = str(entry.get(bucket_key) or "unknown")
        by_bucket[bucket]["probabilities"].append(probability)
        by_bucket[bucket]["labels"].append(label)

    diagnostics: dict[str, Any] = {
        "diagnostic": "selective_risk",
        "requested_confidence_source": requested_confidence_source,
        "total_entry_count": total_entries,
        "labeled_entry_count": len(labels),
        "skipped_missing_confidence_count": skipped_missing_confidence,
        "skipped_missing_label_count": skipped_missing_label,
        "api_called": False,
        "model_training": False,
        "server_executed": False,
        "is_experiment_result": False,
        "note": (
            "Compact AURC/selective-risk diagnostic over the selected "
            "confidence source. It is a decision-support signal for "
            "triage/rerank plumbing, not exposure-utility evidence."
        ),
    }
    if labels:
        diagnostics["overall"] = _compact_summary(probabilities, labels)
        diagnostics["by_popularity_bucket"] = {
            bucket: _compact_summary(
                values["probabilities"],  # type: ignore[arg-type]
                values["labels"],  # type: ignore[arg-type]
            )
            for bucket, values in sorted(by_bucket.items())
            if values["labels"]
        }
    else:
        diagnostics["overall"] = {
            "status": "missing_labeled_confidence_rows",
            "count": 0,
        }
        diagnostics["by_popularity_bucket"] = {}
    return diagnostics


__all__ = ["selective_risk_diagnostics"]
