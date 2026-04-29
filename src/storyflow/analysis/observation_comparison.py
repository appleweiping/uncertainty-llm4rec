"""Comparison utilities for completed observation analysis summaries."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        return {}
    return json.loads(input_path.read_text(encoding="utf-8"))


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(numerator: Any, denominator: Any) -> float | None:
    numerator_number = _number(numerator)
    denominator_number = _number(denominator)
    if numerator_number is None or not denominator_number:
        return None
    return float(numerator_number) / float(denominator_number)


def _delta(value: Any, baseline: Any) -> float | None:
    value_number = _number(value)
    baseline_number = _number(baseline)
    if value_number is None or baseline_number is None:
        return None
    return float(value_number) - float(baseline_number)


def _counter_value(mapping: dict[str, Any] | None, key: str) -> int:
    if not isinstance(mapping, dict):
        return 0
    value = mapping.get(key)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def observation_comparison_row(
    *,
    label: str,
    analysis_summary: dict[str, Any],
    case_review_summary: dict[str, Any] | None = None,
    analysis_summary_path: str | Path | None = None,
    case_review_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Flatten one run's analysis/case-review summaries for comparison."""

    candidate = analysis_summary.get("candidate_diagnostic_summary")
    candidate = candidate if isinstance(candidate, dict) else {}
    confidence = analysis_summary.get("confidence_metrics")
    confidence = confidence if isinstance(confidence, dict) else {}
    grounding = analysis_summary.get("grounding_summary")
    grounding = grounding if isinstance(grounding, dict) else {}
    quadrants = analysis_summary.get("quadrant_counts")
    quadrants = quadrants if isinstance(quadrants, dict) else {}
    bucket_counts = analysis_summary.get("repeat_target_summary")
    bucket_counts = analysis_summary.get("bucket_summary")
    bucket_counts = bucket_counts if isinstance(bucket_counts, dict) else {}
    case_review_summary = case_review_summary or {}
    taxonomy = case_review_summary.get("taxonomy_counts")
    taxonomy = taxonomy if isinstance(taxonomy, dict) else {}
    tags = case_review_summary.get("tag_counts")
    tags = tags if isinstance(tags, dict) else {}

    count = analysis_summary.get("count")
    grounding_failure_count = grounding.get("failure_count")
    candidate_context_available = bool(candidate.get("candidate_context_available"))
    target_in_candidates_count = candidate.get("target_in_candidates_count")
    target_correctness_interpretable = candidate.get(
        "target_correctness_interpretable_as_recommendation_accuracy"
    )
    if not candidate_context_available:
        target_correctness_interpretable = None

    return {
        "label": label,
        "provider": analysis_summary.get("provider"),
        "model": analysis_summary.get("model"),
        "dry_run": analysis_summary.get("dry_run"),
        "api_called": analysis_summary.get("api_called"),
        "count": count,
        "failed_count": analysis_summary.get("failed_count"),
        "parse_failure_count": analysis_summary.get("parse_failure_count"),
        "provider_failure_count": analysis_summary.get("provider_failure_count"),
        "ground_hit": analysis_summary.get("ground_hit"),
        "correctness": analysis_summary.get("correctness"),
        "mean_confidence": analysis_summary.get("mean_confidence"),
        "ece": confidence.get("ece"),
        "brier": confidence.get("brier"),
        "cbu_tau": confidence.get("cbu_tau"),
        "wbc_tau": confidence.get("wbc_tau"),
        "tail_underconfidence_gap": confidence.get("tail_underconfidence_gap"),
        "wrong_high_confidence_count": quadrants.get("wrong_high_confidence"),
        "correct_low_confidence_count": quadrants.get("correct_low_confidence"),
        "wrong_low_confidence_count": quadrants.get("wrong_low_confidence"),
        "correct_confident_count": quadrants.get("correct_confident"),
        "grounding_failure_count": grounding_failure_count,
        "grounding_failure_rate": _rate(grounding_failure_count, count),
        "grounding_status_counts": grounding.get("status_counts") or {},
        "target_bucket_counts": {
            bucket: _nested(bucket_counts, bucket, "count")
            for bucket in ("head", "mid", "tail")
            if _nested(bucket_counts, bucket, "count") is not None
        },
        "candidate_context_available": candidate_context_available,
        "candidate_policy_counts": candidate.get("candidate_policy_counts") or {},
        "rows_with_candidate_context": candidate.get("rows_with_candidate_context"),
        "target_in_candidates_count": target_in_candidates_count,
        "target_excluded_from_candidates_rate": candidate.get(
            "target_excluded_from_candidates_rate"
        ),
        "generated_in_candidate_set_count": candidate.get(
            "generated_in_candidate_set_count"
        ),
        "generated_in_candidate_set_rate": candidate.get(
            "generated_in_candidate_set_rate"
        ),
        "grounded_not_in_candidate_set_count": candidate.get(
            "grounded_not_in_candidate_set_count"
        ),
        "ungrounded_with_candidate_context_count": candidate.get(
            "ungrounded_with_candidate_context_count"
        ),
        "selected_history_item_rate": candidate.get("selected_history_item_rate"),
        "mean_selected_candidate_rank": candidate.get("mean_selected_candidate_rank"),
        "selected_candidate_bucket_counts": candidate.get(
            "selected_candidate_bucket_counts"
        )
        or {},
        "target_correctness_interpretable_as_recommendation_accuracy": (
            target_correctness_interpretable
        ),
        "case_ungrounded_high_confidence_count": _counter_value(
            taxonomy,
            "ungrounded_high_confidence",
        ),
        "case_ungrounded_low_confidence_count": _counter_value(
            taxonomy,
            "ungrounded_low_confidence",
        ),
        "case_wrong_high_confidence_count": _counter_value(
            taxonomy,
            "wrong_high_confidence",
        ),
        "case_wrong_low_confidence_count": _counter_value(
            taxonomy,
            "wrong_low_confidence",
        ),
        "case_self_verified_wrong_count": _counter_value(
            tags,
            "self_verified_wrong",
        ),
        "case_generated_more_popular_than_target_count": _counter_value(
            tags,
            "generated_more_popular_than_target",
        ),
        "case_wrong_high_confidence_generated_head_count": _counter_value(
            tags,
            "wrong_high_confidence_generated_head",
        ),
        "analysis_summary_path": (
            str(analysis_summary_path) if analysis_summary_path is not None else None
        ),
        "case_review_summary_path": (
            str(case_review_summary_path)
            if case_review_summary_path is not None
            else None
        ),
    }


def compare_observation_summaries(
    runs: Iterable[dict[str, Any]],
    *,
    source_label: str | None = None,
) -> dict[str, Any]:
    """Compare multiple completed observation analysis summaries."""

    rows = [
        observation_comparison_row(
            label=str(run["label"]),
            analysis_summary=run["analysis_summary"],
            case_review_summary=run.get("case_review_summary"),
            analysis_summary_path=run.get("analysis_summary_path"),
            case_review_summary_path=run.get("case_review_summary_path"),
        )
        for run in runs
    ]
    if not rows:
        raise ValueError("comparison requires at least one run")

    baseline = rows[0]
    for row in rows:
        row["delta_ground_hit_vs_first"] = _delta(
            row.get("ground_hit"),
            baseline.get("ground_hit"),
        )
        row["delta_mean_confidence_vs_first"] = _delta(
            row.get("mean_confidence"),
            baseline.get("mean_confidence"),
        )
        row["delta_wbc_tau_vs_first"] = _delta(
            row.get("wbc_tau"),
            baseline.get("wbc_tau"),
        )
        row["delta_grounding_failure_rate_vs_first"] = _delta(
            row.get("grounding_failure_rate"),
            baseline.get("grounding_failure_rate"),
        )

    candidate_rows = [row for row in rows if row["candidate_context_available"]]
    target_excluding_rows = [
        row
        for row in candidate_rows
        if row.get("target_excluded_from_candidates_rate") == 1.0
    ]
    best_ground_hit = max(rows, key=lambda row: _number(row.get("ground_hit")) or -1)
    lowest_grounding_failure = min(
        rows,
        key=lambda row: (
            _number(row.get("grounding_failure_rate"))
            if _number(row.get("grounding_failure_rate")) is not None
            else 10**9
        ),
    )
    best_candidate_adherence = (
        max(
            candidate_rows,
            key=lambda row: _number(row.get("generated_in_candidate_set_rate")) or -1,
        )
        if candidate_rows
        else None
    )
    lowest_wbc = min(
        rows,
        key=lambda row: (
            _number(row.get("wbc_tau"))
            if _number(row.get("wbc_tau")) is not None
            else 10**9
        ),
    )
    comparison_allowed = not any(
        row.get("target_correctness_interpretable_as_recommendation_accuracy") is False
        for row in rows
    )

    return {
        "created_at_utc": utc_now_iso(),
        "source_label": source_label,
        "run_count": len(rows),
        "rows": rows,
        "diagnostic_takeaways": {
            "highest_ground_hit_label": best_ground_hit["label"],
            "lowest_grounding_failure_label": lowest_grounding_failure["label"],
            "highest_candidate_adherence_label": (
                best_candidate_adherence["label"]
                if best_candidate_adherence is not None
                else None
            ),
            "lowest_wbc_tau_label": lowest_wbc["label"],
            "target_excluding_candidate_run_count": len(target_excluding_rows),
            "candidate_run_count": len(candidate_rows),
        },
        "claim_guardrails": {
            "comparison_scope": (
                "Prompt/candidate/grounding QA over completed observation "
                "summaries. Candidate-constrained runs may exclude the held-out "
                "target by design."
            ),
            "recommendation_accuracy_comparison_allowed": comparison_allowed,
            "is_paper_result": False,
            "do_not_claim": [
                "method improvement",
                "recommendation accuracy for target-excluding candidate prompts",
                "server or training result",
            ],
        },
        "note": (
            "This comparison reads existing analysis artifacts and does not "
            "call APIs, download data, train models, or inspect raw responses."
        ),
    }


def observation_comparison_markdown(comparison: dict[str, Any]) -> str:
    rows = comparison.get("rows", [])
    lines = [
        "# Observation Run Comparison",
        "",
        "This report compares completed analysis summaries. It is a reproducibility and QA artifact, not a paper-result table.",
        "",
        "## Scope Guardrails",
        "",
        f"- Source label: {comparison.get('source_label')}",
        f"- Recommendation-accuracy comparison allowed: {comparison.get('claim_guardrails', {}).get('recommendation_accuracy_comparison_allowed')}",
        f"- Paper result: {comparison.get('claim_guardrails', {}).get('is_paper_result')}",
        f"- Scope: {comparison.get('claim_guardrails', {}).get('comparison_scope')}",
        "",
        "## Summary Table",
        "",
        "| label | count | failed | GroundHit | mean conf | WBC_tau | grounding failure rate | generated in candidate set | target excluded rate | mean selected rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {count} | {failed} | {ground_hit} | {conf} | {wbc} | {gfail} | {candidate_rate} | {target_excluded} | {rank} |".format(
                label=row.get("label"),
                count=row.get("count"),
                failed=row.get("failed_count"),
                ground_hit=row.get("ground_hit"),
                conf=row.get("mean_confidence"),
                wbc=row.get("wbc_tau"),
                gfail=row.get("grounding_failure_rate"),
                candidate_rate=row.get("generated_in_candidate_set_rate"),
                target_excluded=row.get("target_excluded_from_candidates_rate"),
                rank=row.get("mean_selected_candidate_rank"),
            )
        )
    takeaways = comparison.get("diagnostic_takeaways", {})
    lines.extend(
        [
            "",
            "## Diagnostic Takeaways",
            "",
            f"- Highest GroundHit in this comparison: {takeaways.get('highest_ground_hit_label')}",
            f"- Lowest grounding failure rate in this comparison: {takeaways.get('lowest_grounding_failure_label')}",
            f"- Highest candidate adherence in candidate-context runs: {takeaways.get('highest_candidate_adherence_label')}",
            f"- Lowest WBC_tau in this comparison: {takeaways.get('lowest_wbc_tau_label')}",
            f"- Target-excluding candidate runs: {takeaways.get('target_excluding_candidate_run_count')}",
            "",
            "## Required Interpretation",
            "",
            "- Do not read target-hit correctness from target-excluding candidate prompts as recommendation accuracy.",
            "- Use this comparison to choose prompt/grounding gates before larger API spend.",
            "- Any paper claim still needs a dedicated protocol, manifests, and reviewer-facing scope language.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_observation_comparison(
    *,
    runs: Iterable[dict[str, Any]],
    output_dir: str | Path,
    source_label: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison = compare_observation_summaries(runs, source_label=source_label)

    summary_path = output_path / "comparison_summary.json"
    rows_path = output_path / "comparison_rows.jsonl"
    table_path = output_path / "comparison_table.csv"
    report_path = output_path / "report.md"
    manifest_path = output_path / "comparison_manifest.json"

    summary_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_jsonl(rows_path, comparison["rows"])
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "label",
            "count",
            "failed_count",
            "ground_hit",
            "mean_confidence",
            "wbc_tau",
            "grounding_failure_rate",
            "generated_in_candidate_set_rate",
            "target_excluded_from_candidates_rate",
            "mean_selected_candidate_rank",
            "delta_ground_hit_vs_first",
            "delta_mean_confidence_vs_first",
            "delta_wbc_tau_vs_first",
            "delta_grounding_failure_rate_vs_first",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison["rows"]:
            writer.writerow({field: row.get(field) for field in fieldnames})
    report_path.write_text(
        observation_comparison_markdown(comparison),
        encoding="utf-8",
    )

    manifest = {
        "created_at_utc": comparison["created_at_utc"],
        "comparison_dir": str(output_path),
        "summary": str(summary_path),
        "rows": str(rows_path),
        "table": str(table_path),
        "report": str(report_path),
        "source_label": source_label,
        "run_count": comparison["run_count"],
        "is_paper_result": False,
        "note": comparison["note"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
