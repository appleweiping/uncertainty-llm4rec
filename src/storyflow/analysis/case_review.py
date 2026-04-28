"""Case review and failure taxonomy for observation pilot outputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from storyflow.analysis.observation import load_json, read_jsonl, write_jsonl
from storyflow.observation import load_catalog_rows


YES_VALUES = {"yes", "y", "true", "likely", "correct"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_yes(value: Any) -> bool:
    return str(value or "").strip().lower() in YES_VALUES


def _tail(values: list[Any], max_items: int) -> list[Any]:
    if max_items < 1:
        return []
    return values[-max_items:]


def _load_input_rows(input_jsonl: str | Path | None) -> dict[str, dict[str, Any]]:
    if not input_jsonl:
        return {}
    path = Path(input_jsonl)
    if not path.exists():
        return {}
    return {str(row.get("input_id")): row for row in read_jsonl(path)}


def _catalog_by_id_from_inputs(input_rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    catalog_csv: str | None = None
    for row in input_rows.values():
        source = row.get("source") or {}
        if source.get("catalog_csv"):
            catalog_csv = str(source["catalog_csv"])
            break
    if not catalog_csv or not Path(catalog_csv).exists():
        return {}
    return {str(row["item_id"]): row for row in load_catalog_rows(catalog_csv)}


def _primary_label(
    row: dict[str, Any],
    *,
    low_confidence_tau: float,
    high_confidence_tau: float,
) -> str:
    if row.get("failure_stage"):
        return f"{row.get('failure_stage')}_failure"

    confidence = _safe_float(row.get("confidence"))
    correctness = _safe_int(row.get("correctness"))
    grounded = bool(row.get("grounded_item_id"))
    if not grounded:
        if confidence >= high_confidence_tau:
            return "ungrounded_high_confidence"
        if confidence < low_confidence_tau:
            return "ungrounded_low_confidence"
        return "ungrounded_intermediate_confidence"
    if correctness == 1:
        if confidence >= high_confidence_tau:
            return "correct_high_confidence"
        if confidence < low_confidence_tau:
            return "correct_low_confidence"
        return "correct_intermediate_confidence"
    if confidence >= high_confidence_tau:
        return "wrong_high_confidence"
    if confidence < low_confidence_tau:
        return "wrong_low_confidence"
    return "wrong_intermediate_confidence"


def _taxonomy_tags(
    row: dict[str, Any],
    *,
    generated_item: dict[str, Any] | None,
    low_confidence_tau: float,
    high_confidence_tau: float,
    ambiguity_tau: float,
    popularity_ratio_tau: float,
) -> list[str]:
    tags = [
        _primary_label(
            row,
            low_confidence_tau=low_confidence_tau,
            high_confidence_tau=high_confidence_tau,
        )
    ]
    confidence = _safe_float(row.get("confidence"))
    correctness = _safe_int(row.get("correctness"))
    grounded = bool(row.get("grounded_item_id"))
    target_bucket = str(row.get("target_popularity_bucket") or "unknown")
    generated_bucket = str((generated_item or {}).get("popularity_bucket") or "unknown")
    target_popularity = _safe_float(row.get("target_popularity"))
    generated_popularity = _safe_float((generated_item or {}).get("popularity"))

    if row.get("failure_stage"):
        tags.append("pipeline_failure")
        return sorted(set(tags))
    if _is_yes(row.get("is_likely_correct")) and correctness == 0:
        tags.append("self_verified_wrong")
    if not grounded:
        tags.append("grounding_failure")
    if grounded and str(row.get("grounding_status") or "") == "fuzzy":
        tags.append("fuzzy_grounding")
    if _safe_float(row.get("grounding_ambiguity"), default=0.0) >= ambiguity_tau:
        tags.append("grounding_ambiguous")
    if target_bucket in {"head", "mid", "tail"}:
        tags.append(f"target_{target_bucket}")
    if generated_bucket in {"head", "mid", "tail"}:
        tags.append(f"generated_{generated_bucket}")
    if (
        generated_popularity > 0
        and target_popularity >= 0
        and generated_popularity >= max(1.0, target_popularity) * popularity_ratio_tau
    ):
        tags.append("generated_more_popular_than_target")
    if generated_bucket == "head" and target_bucket == "tail":
        tags.append("generated_head_target_tail")
    if correctness == 0 and confidence >= high_confidence_tau and generated_bucket == "head":
        tags.append("wrong_high_confidence_generated_head")
    if correctness == 1 and confidence < low_confidence_tau and target_bucket == "tail":
        tags.append("correct_low_confidence_tail")
    return sorted(set(tags))


def _case_recommended_actions(primary_label: str, tags: Iterable[str]) -> list[str]:
    tag_set = {str(tag) for tag in tags}
    actions: list[str] = []
    if primary_label.startswith("parse_"):
        actions.append("fix_parser_or_forced_json_prompt")
    if primary_label.startswith("provider_"):
        actions.append("inspect_provider_retry_rate_limit_or_network")
    if "grounding_failure" in tag_set:
        actions.append("inspect_catalog_coverage_and_title_normalization")
    if primary_label == "ungrounded_high_confidence":
        actions.append("tighten_prompt_to_catalog_groundable_titles")
        actions.append("add_or_improve_retrieval_assisted_grounding")
    if "grounding_ambiguous" in tag_set or "fuzzy_grounding" in tag_set:
        actions.append("review_grounding_thresholds_and_candidate_margin")
    if "self_verified_wrong" in tag_set:
        actions.append("audit_self_verification_confidence_prompt")
    if (
        "generated_more_popular_than_target" in tag_set
        or "generated_head_target_tail" in tag_set
        or "wrong_high_confidence_generated_head" in tag_set
    ):
        actions.append("prioritize_popularity_confidence_residual_analysis")
    if primary_label == "wrong_high_confidence":
        actions.append("prioritize_overconfidence_case_review")
    if "correct_low_confidence_tail" in tag_set:
        actions.append("prioritize_tail_underconfidence_calibration")
    if primary_label == "correct_low_confidence":
        actions.append("preserve_hard_positive_before_naive_pruning")
    if not actions:
        actions.append("monitor_in_aggregate")
    return sorted(set(actions))


def _case_record(
    row: dict[str, Any],
    *,
    input_record: dict[str, Any] | None,
    generated_item: dict[str, Any] | None,
    low_confidence_tau: float,
    high_confidence_tau: float,
    ambiguity_tau: float,
    popularity_ratio_tau: float,
    max_history_titles: int,
) -> dict[str, Any]:
    history_titles = list((input_record or {}).get("history_item_titles") or [])
    primary_label = _primary_label(
        row,
        low_confidence_tau=low_confidence_tau,
        high_confidence_tau=high_confidence_tau,
    )
    taxonomy_tags = _taxonomy_tags(
        row,
        generated_item=generated_item,
        low_confidence_tau=low_confidence_tau,
        high_confidence_tau=high_confidence_tau,
        ambiguity_tau=ambiguity_tau,
        popularity_ratio_tau=popularity_ratio_tau,
    )
    return {
        "input_id": row.get("input_id"),
        "example_id": row.get("example_id"),
        "user_id": row.get("user_id") or (input_record or {}).get("user_id"),
        "split": row.get("split") or (input_record or {}).get("split"),
        "primary_failure_type": primary_label,
        "taxonomy_tags": taxonomy_tags,
        "recommended_actions": _case_recommended_actions(primary_label, taxonomy_tags),
        "history_length": row.get("history_length") or (input_record or {}).get("history_length"),
        "history_titles_tail": _tail(history_titles, max_history_titles),
        "generated_title": row.get("generated_title"),
        "target_title": row.get("target_title") or (input_record or {}).get("target_title"),
        "confidence": row.get("confidence"),
        "is_likely_correct": row.get("is_likely_correct"),
        "correctness": row.get("correctness"),
        "grounded_item_id": row.get("grounded_item_id"),
        "grounded_title": (generated_item or {}).get("title"),
        "grounding_status": row.get("grounding_status"),
        "grounding_score": row.get("grounding_score"),
        "grounding_ambiguity": row.get("grounding_ambiguity"),
        "generated_popularity": (generated_item or {}).get("popularity"),
        "generated_popularity_bucket": (generated_item or {}).get("popularity_bucket"),
        "target_item_id": row.get("target_item_id") or (input_record or {}).get("target_item_id"),
        "target_popularity": row.get("target_popularity") or (input_record or {}).get("target_popularity"),
        "target_popularity_bucket": row.get("target_popularity_bucket")
        or (input_record or {}).get("target_popularity_bucket"),
        "parse_strategy": row.get("parse_strategy"),
        "cache_hit": row.get("cache_hit"),
        "failure_stage": row.get("failure_stage"),
        "error": row.get("error"),
        "usage": row.get("usage"),
    }


def _case_priority(row: dict[str, Any]) -> tuple[int, float, str]:
    priority = {
        "wrong_high_confidence": 0,
        "ungrounded_high_confidence": 1,
        "parse_failure": 2,
        "provider_failure": 3,
        "correct_low_confidence": 4,
        "wrong_intermediate_confidence": 5,
    }.get(str(row.get("primary_failure_type")), 9)
    return (priority, -_safe_float(row.get("confidence")), str(row.get("input_id") or ""))


def summarize_case_review(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(cases)
    taxonomy_counts = Counter(str(row.get("primary_failure_type")) for row in records)
    tag_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    bucket_taxonomy: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        for tag in row.get("taxonomy_tags") or []:
            tag_counts[str(tag)] += 1
        for action in row.get("recommended_actions") or []:
            action_counts[str(action)] += 1
        bucket = str(row.get("target_popularity_bucket") or "unknown")
        bucket_taxonomy[bucket][str(row.get("primary_failure_type"))] += 1
    return {
        "created_at_utc": utc_now_iso(),
        "case_count": len(records),
        "taxonomy_counts": dict(sorted(taxonomy_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "recommended_next_actions": [
            action
            for action, _ in sorted(
                action_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:8]
        ],
        "target_bucket_taxonomy_counts": {
            bucket: dict(sorted(counter.items()))
            for bucket, counter in sorted(bucket_taxonomy.items())
        },
        "wrong_high_confidence_count": taxonomy_counts.get("wrong_high_confidence", 0),
        "ungrounded_high_confidence_count": taxonomy_counts.get("ungrounded_high_confidence", 0),
        "correct_low_confidence_count": taxonomy_counts.get("correct_low_confidence", 0),
        "self_verified_wrong_count": tag_counts.get("self_verified_wrong", 0),
        "generated_more_popular_than_target_count": tag_counts.get(
            "generated_more_popular_than_target",
            0,
        ),
        "wrong_high_confidence_generated_head_count": tag_counts.get(
            "wrong_high_confidence_generated_head",
            0,
        ),
        "is_experiment_result": False,
        "note": (
            "Case review is a diagnostic artifact for pilot triage. It does not "
            "make paper claims by itself."
        ),
    }


def case_review_markdown(summary: dict[str, Any], cases: Iterable[dict[str, Any]]) -> str:
    records = list(cases)
    taxonomy_rows = [
        f"| {name} | {count} |"
        for name, count in summary.get("taxonomy_counts", {}).items()
    ]
    tag_rows = [
        f"| {name} | {count} |"
        for name, count in summary.get("tag_counts", {}).items()
    ]
    action_rows = [
        f"| {name} | {count} |"
        for name, count in summary.get("action_counts", {}).items()
    ]
    top_cases = records[:10]
    case_rows = []
    for row in top_cases:
        history = " | ".join(str(title) for title in row.get("history_titles_tail") or [])
        case_rows.append(
            "| {case_type} | {conf} | {gen} | {ground} | {target} | {bucket} | {history} |".format(
                case_type=row.get("primary_failure_type"),
                conf=row.get("confidence"),
                gen=str(row.get("generated_title") or "").replace("|", "/"),
                ground=str(row.get("grounded_title") or row.get("grounded_item_id") or "").replace("|", "/"),
                target=str(row.get("target_title") or "").replace("|", "/"),
                bucket=row.get("target_popularity_bucket"),
                history=history.replace("|", "/"),
            )
        )
    return "\n".join(
        [
            "# Observation Pilot Case Review",
            "",
            "This report is for pilot diagnosis and failure taxonomy. It is not a paper result.",
            "",
            "## Summary",
            "",
            f"- Case count: {summary.get('case_count')}",
            f"- Wrong-high-confidence: {summary.get('wrong_high_confidence_count')}",
            f"- Ungrounded-high-confidence: {summary.get('ungrounded_high_confidence_count')}",
            f"- Correct-low-confidence: {summary.get('correct_low_confidence_count')}",
            f"- Self-verified wrong: {summary.get('self_verified_wrong_count')}",
            f"- Generated more popular than target: {summary.get('generated_more_popular_than_target_count')}",
            "",
            "## Primary Taxonomy",
            "",
            "| type | count |",
            "| --- | ---: |",
            *taxonomy_rows,
            "",
            "## Overlay Tags",
            "",
            "| tag | count |",
            "| --- | ---: |",
            *tag_rows,
            "",
            "## Recommended Next Actions",
            "",
            *[f"- {action}" for action in summary.get("recommended_next_actions", [])],
            "",
            "## Action Counts",
            "",
            "| action | count |",
            "| --- | ---: |",
            *action_rows,
            "",
            "## Priority Cases",
            "",
            "| type | confidence | generated | grounded | target | target bucket | history tail |",
            "| --- | ---: | --- | --- | --- | --- | --- |",
            *case_rows,
            "",
            "## Caveat",
            "",
            "Use this file to debug prompts, grounding, and sampling. Do not treat a small pilot taxonomy as a full model behavior claim.",
        ]
    ) + "\n"


def review_observation_cases(
    *,
    grounded_jsonl: str | Path,
    output_dir: str | Path,
    input_jsonl: str | Path | None = None,
    failed_jsonl: str | Path | None = None,
    manifest_json: str | Path | None = None,
    low_confidence_tau: float = 0.5,
    high_confidence_tau: float = 0.7,
    ambiguity_tau: float = 0.75,
    popularity_ratio_tau: float = 2.0,
    max_history_titles: int = 8,
) -> dict[str, Any]:
    grounded_path = Path(grounded_jsonl)
    manifest = load_json(manifest_json) if manifest_json else {}
    if input_jsonl is None:
        input_jsonl = manifest.get("input_jsonl")
    input_rows = _load_input_rows(input_jsonl)
    catalog_by_id = _catalog_by_id_from_inputs(input_rows)
    grounded_rows = read_jsonl(grounded_path)
    failed_rows = read_jsonl(failed_jsonl) if failed_jsonl and Path(failed_jsonl).exists() else []

    cases: list[dict[str, Any]] = []
    for row in grounded_rows:
        input_record = input_rows.get(str(row.get("input_id")))
        generated_item = catalog_by_id.get(str(row.get("grounded_item_id")))
        cases.append(
            _case_record(
                row,
                input_record=input_record,
                generated_item=generated_item,
                low_confidence_tau=low_confidence_tau,
                high_confidence_tau=high_confidence_tau,
                ambiguity_tau=ambiguity_tau,
                popularity_ratio_tau=popularity_ratio_tau,
                max_history_titles=max_history_titles,
            )
        )
    for row in failed_rows:
        input_record = input_rows.get(str(row.get("input_id")))
        cases.append(
            _case_record(
                row,
                input_record=input_record,
                generated_item=None,
                low_confidence_tau=low_confidence_tau,
                high_confidence_tau=high_confidence_tau,
                ambiguity_tau=ambiguity_tau,
                popularity_ratio_tau=popularity_ratio_tau,
                max_history_titles=max_history_titles,
            )
        )
    cases = sorted(cases, key=_case_priority)
    summary = summarize_case_review(cases)
    summary.update(
        {
            "provider": manifest.get("provider"),
            "model": manifest.get("model"),
            "api_called": manifest.get("api_called"),
            "dry_run": manifest.get("dry_run"),
            "source_grounded_jsonl": str(grounded_path),
            "source_failed_jsonl": str(failed_jsonl) if failed_jsonl else None,
            "source_input_jsonl": str(input_jsonl) if input_jsonl else None,
            "source_manifest_json": str(manifest_json) if manifest_json else None,
            "low_confidence_tau": low_confidence_tau,
            "high_confidence_tau": high_confidence_tau,
            "ambiguity_tau": ambiguity_tau,
            "popularity_ratio_tau": popularity_ratio_tau,
        }
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "case_review_summary.json"
    cases_path = output_path / "case_review_cases.jsonl"
    report_path = output_path / "case_review.md"
    manifest_path = output_path / "case_review_manifest.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    write_jsonl(cases_path, cases)
    report_path.write_text(
        case_review_markdown(summary, cases),
        encoding="utf-8",
    )
    review_manifest = {
        "created_at_utc": utc_now_iso(),
        "case_review_dir": str(output_path),
        "summary": str(summary_path),
        "cases": str(cases_path),
        "report": str(report_path),
        "provider": summary.get("provider"),
        "model": summary.get("model"),
        "api_called": summary.get("api_called"),
        "dry_run": summary.get("dry_run"),
        "case_count": summary.get("case_count"),
        "is_experiment_result": False,
        "note": "Pilot case review diagnostic; not paper evidence.",
    }
    manifest_path.write_text(
        json.dumps(review_manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return review_manifest
