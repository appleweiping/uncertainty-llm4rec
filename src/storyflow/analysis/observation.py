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
    summary = {
        "created_at_utc": utc_now_iso(),
        "provider": manifest.get("provider") or (records[0].get("provider") if records else None),
        "model": manifest.get("model") or (records[0].get("model") if records else None),
        "dry_run": manifest.get("dry_run", records[0].get("dry_run") if records else None),
        "api_called": manifest.get("api_called"),
        "input_jsonl": manifest.get("input_jsonl"),
        "source_output_dir": manifest.get("output_dir"),
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
    slope = summary.get("popularity_confidence_slope", {})
    univariate = slope.get("univariate", {})
    controlled = slope.get("correctness_residualized", {})
    lines = [
        "# Observation Analysis Report",
        "",
        "This report summarizes grounded title-generation observation outputs. If the source run is mock or dry-run, this is only a schema/analysis sanity artifact and not paper evidence.",
        "",
        "## Source",
        "",
        f"- Provider: {summary.get('provider')}",
        f"- Model: {summary.get('model')}",
        f"- Dry-run: {summary.get('dry_run')}",
        f"- API called: {summary.get('api_called')}",
        f"- Source output dir: {summary.get('source_output_dir')}",
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
            "## Popularity-Confidence Slope",
            "",
            f"- Univariate slope: {univariate.get('slope')}",
            f"- Univariate correlation: {univariate.get('correlation')}",
            f"- Correctness-residualized slope: {controlled.get('slope')}",
            f"- Correctness-residualized correlation: {controlled.get('correlation')}",
            "",
            "## Failure Summary",
            "",
            f"- Grounding status counts: {summary.get('grounding_summary', {}).get('status_counts')}",
            f"- Parse strategy counts: {summary.get('parse_summary', {}).get('parse_strategy_counts')}",
            f"- Failure stage counts: {summary.get('parse_summary', {}).get('failure_stage_counts')}",
            "",
            "## Reminder",
            "",
            "Do not treat mock, dry-run, or synthetic outputs as real model behavior or paper results.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_observation_run(
    *,
    grounded_jsonl: str | Path,
    output_dir: str | Path,
    failed_jsonl: str | Path | None = None,
    manifest_json: str | Path | None = None,
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
    summary = summarize_observation_records(
        grounded_rows,
        failed_rows=failed_rows,
        manifest=manifest,
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
        }
    )
    summary_path = output_path / "analysis_summary.json"
    reliability_path = output_path / "reliability_diagram.json"
    bucket_path = output_path / "bucket_summary.json"
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
        "risk_cases": str(risk_path),
        "report": str(report_path),
        "source_grounded_jsonl": str(grounded_path),
        "source_failed_jsonl": str(failed_path) if failed_path else None,
        "source_manifest_json": str(manifest_path) if manifest_path else None,
        "provider": summary.get("provider"),
        "model": summary.get("model"),
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
