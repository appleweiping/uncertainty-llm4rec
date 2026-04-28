"""Grounding ambiguity diagnostics for processed catalogs and observation runs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from storyflow.grounding import normalize_title


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def load_catalog_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            title = str(row.get("title") or "")
            normalized = str(row.get("title_normalized") or "").strip()
            if not normalized and title.strip():
                normalized = normalize_title(title)
            rows.append(
                {
                    **row,
                    "item_id": str(row.get("item_id") or ""),
                    "title": title,
                    "title_normalized": normalized,
                    "popularity": int(float(row.get("popularity") or 0)),
                    "popularity_bucket": str(row.get("popularity_bucket") or "unknown"),
                }
            )
    return rows


def duplicate_title_groups(
    catalog_rows: Iterable[dict[str, Any]],
    *,
    max_groups: int = 50,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog_rows:
        normalized = str(row.get("title_normalized") or "").strip()
        if normalized:
            grouped[normalized].append(dict(row))

    duplicate_groups: list[dict[str, Any]] = []
    for normalized, rows in grouped.items():
        if len(rows) <= 1:
            continue
        rows_sorted = sorted(
            rows,
            key=lambda row: (-int(row.get("popularity") or 0), str(row.get("item_id") or "")),
        )
        bucket_counts = Counter(str(row.get("popularity_bucket") or "unknown") for row in rows_sorted)
        duplicate_groups.append(
            {
                "normalized_title": normalized,
                "count": len(rows_sorted),
                "max_popularity": max(int(row.get("popularity") or 0) for row in rows_sorted),
                "bucket_counts": dict(sorted(bucket_counts.items())),
                "items": [
                    {
                        "item_id": row.get("item_id"),
                        "title": row.get("title"),
                        "popularity": int(row.get("popularity") or 0),
                        "popularity_bucket": row.get("popularity_bucket"),
                    }
                    for row in rows_sorted
                ],
            }
        )
    duplicate_groups.sort(
        key=lambda group: (-int(group["count"]), -int(group["max_popularity"]), group["normalized_title"])
    )
    return duplicate_groups[:max_groups]


def catalog_grounding_summary(catalog_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(catalog_rows)
    normalized_counts = Counter(str(row.get("title_normalized") or "") for row in rows)
    duplicate_group_count = sum(1 for title, count in normalized_counts.items() if title and count > 1)
    duplicate_item_count = sum(count for title, count in normalized_counts.items() if title and count > 1)
    empty_title_count = sum(not str(row.get("title") or "").strip() for row in rows)
    bucket_counts = Counter(str(row.get("popularity_bucket") or "unknown") for row in rows)
    return {
        "catalog_item_count": len(rows),
        "empty_title_count": empty_title_count,
        "duplicate_normalized_group_count": duplicate_group_count,
        "duplicate_normalized_item_count": duplicate_item_count,
        "duplicate_item_fraction": (duplicate_item_count / len(rows)) if rows else 0.0,
        "catalog_bucket_counts": dict(sorted(bucket_counts.items())),
    }


def _candidate_scores(row: dict[str, Any]) -> tuple[float | None, float | None]:
    candidates = row.get("grounding_candidates") or []
    if isinstance(candidates, list) and candidates:
        scores = [float(candidate.get("score") or 0.0) for candidate in candidates[:2]]
        top_score = scores[0]
        second_score = scores[1] if len(scores) > 1 else None
        return top_score, second_score
    top_score = row.get("grounding_score")
    second_score = row.get("grounding_second_score")
    if top_score is None:
        return None, None
    return float(top_score), float(second_score) if second_score is not None else None


def _candidate_margin(row: dict[str, Any]) -> float | None:
    top_score, second_score = _candidate_scores(row)
    if top_score is None or second_score is None:
        return None
    return float(top_score) - float(second_score)


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return float(SequenceMatcher(None, left, right).ratio())


def _quick_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return float(SequenceMatcher(None, left, right).quick_ratio())


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _catalog_candidates(
    generated_title: str,
    catalog_rows: Iterable[dict[str, Any]],
    *,
    max_candidates: int = 3,
) -> list[dict[str, Any]]:
    generated_normalized = normalize_title(generated_title)
    if not generated_normalized:
        return []
    generated_tokens = set(generated_normalized.split())
    catalog = list(catalog_rows)
    use_token_prefilter = len(catalog) > 500 and bool(generated_tokens)
    candidates: list[dict[str, Any]] = []
    for row in catalog:
        title = str(row.get("title") or "")
        normalized = str(row.get("title_normalized") or "").strip() or normalize_title(title)
        if not normalized:
            continue
        if use_token_prefilter and not (generated_tokens & set(normalized.split())):
            continue
        score = _quick_similarity(generated_normalized, normalized)
        candidates.append(
            {
                "item_id": row.get("item_id"),
                "title": title,
                "score": score,
                "title_normalized": normalized,
                "popularity": int(row.get("popularity") or 0),
                "popularity_bucket": row.get("popularity_bucket"),
            }
        )
    candidates.sort(
        key=lambda candidate: (
            -float(candidate["score"]),
            -int(candidate.get("popularity") or 0),
            str(candidate.get("item_id") or ""),
        )
    )
    shortlist_size = min(len(candidates), max(max_candidates * 20, 50))
    rescored_candidates: list[dict[str, Any]] = []
    for candidate in candidates[:shortlist_size]:
        rescored_candidates.append(
            {
                **candidate,
                "score": _similarity(generated_normalized, str(candidate["title_normalized"])),
            }
        )
    rescored_candidates.sort(
        key=lambda candidate: (
            -float(candidate["score"]),
            -int(candidate.get("popularity") or 0),
            str(candidate.get("item_id") or ""),
        )
    )
    return [
        {
            **{key: value for key, value in candidate.items() if key != "title_normalized"},
            "rank": index + 1,
        }
        for index, candidate in enumerate(rescored_candidates[:max_candidates])
    ]


def _compact_low_margin_case(row: dict[str, Any]) -> dict[str, Any]:
    top_score, second_score = _candidate_scores(row)
    candidates = row.get("grounding_candidates") or []
    return {
        "input_id": row.get("input_id"),
        "example_id": row.get("example_id"),
        "user_id": row.get("user_id"),
        "generated_title": row.get("generated_title"),
        "target_title": row.get("target_title"),
        "confidence": row.get("confidence"),
        "correctness": row.get("correctness"),
        "grounded_item_id": row.get("grounded_item_id"),
        "grounding_status": row.get("grounding_status"),
        "grounding_score": top_score,
        "grounding_second_score": second_score,
        "grounding_margin": _candidate_margin(row),
        "grounding_ambiguity": row.get("grounding_ambiguity"),
        "target_popularity_bucket": row.get("target_popularity_bucket"),
        "top_candidates": [
            {
                "item_id": candidate.get("item_id"),
                "title": candidate.get("title"),
                "score": candidate.get("score"),
                "rank": candidate.get("rank"),
            }
            for candidate in candidates[:3]
        ],
    }


def _failure_actions(categories: Iterable[str]) -> list[str]:
    category_set = set(categories)
    actions: list[str] = []
    if "generated_title_too_generic" in category_set:
        actions.append("tighten_prompt_for_specific_catalog_title")
    if "near_miss_candidate" in category_set:
        actions.append("inspect_grounding_threshold_and_normalization")
    if "weak_candidate_overlap" in category_set:
        actions.append("inspect_prompt_catalog_alignment")
    if "no_catalog_support" in category_set:
        actions.append("consider_catalog_constrained_gate_or_retrieval_context")
    if "duplicate_title_risk" in category_set:
        actions.append("review_duplicate_title_disambiguation")
    if "target_title_near_generated" in category_set:
        actions.append("inspect_target_near_miss_grounding")
    if "high_confidence_ungrounded" in category_set:
        actions.append("prioritize_case_review_before_api_scale")
    if not actions:
        actions.append("manual_review")
    return sorted(dict.fromkeys(actions))


def classify_grounding_failure(
    row: dict[str, Any],
    *,
    catalog_rows: Iterable[dict[str, Any]] = (),
    duplicate_normalized_titles: set[str] | None = None,
    near_miss_threshold: float = 0.70,
    weak_match_threshold: float = 0.40,
    high_confidence_threshold: float = 0.70,
    max_candidates: int = 3,
) -> dict[str, Any]:
    """Classify one non-grounded prediction into actionable diagnostic buckets."""

    duplicate_normalized_titles = duplicate_normalized_titles or set()
    generated_title = str(row.get("generated_title") or "")
    target_title = str(row.get("target_title") or "")
    generated_normalized = normalize_title(generated_title)
    target_normalized = normalize_title(target_title)
    existing_candidates = row.get("grounding_candidates") or []
    catalog_candidates = (
        [
            {
                "item_id": candidate.get("item_id"),
                "title": candidate.get("title"),
                "score": float(candidate.get("score") or 0.0),
                "rank": candidate.get("rank") or index + 1,
            }
            for index, candidate in enumerate(existing_candidates[:max_candidates])
        ]
        if isinstance(existing_candidates, list) and existing_candidates
        else _catalog_candidates(
            generated_title,
            catalog_rows,
            max_candidates=max_candidates,
        )
    )
    top_score = float(catalog_candidates[0]["score"]) if catalog_candidates else 0.0
    second_score = float(catalog_candidates[1]["score"]) if len(catalog_candidates) > 1 else None
    confidence = _as_float(row.get("confidence"))
    target_similarity = _similarity(generated_normalized, target_normalized)
    token_count = len(generated_normalized.split()) if generated_normalized else 0

    categories: list[str] = []
    if confidence is not None and confidence >= high_confidence_threshold:
        categories.append("high_confidence_ungrounded")
    if generated_normalized in duplicate_normalized_titles:
        categories.append("duplicate_title_risk")
    if token_count <= 2 or len(generated_normalized) < 8:
        categories.append("generated_title_too_generic")
    if top_score >= near_miss_threshold:
        categories.append("near_miss_candidate")
    elif top_score >= weak_match_threshold:
        categories.append("weak_candidate_overlap")
    else:
        categories.append("no_catalog_support")
    if target_similarity >= near_miss_threshold:
        categories.append("target_title_near_generated")
    else:
        categories.append("target_title_distant")

    return {
        "input_id": row.get("input_id"),
        "example_id": row.get("example_id"),
        "user_id": row.get("user_id"),
        "generated_title": generated_title,
        "target_title": target_title,
        "generated_title_normalized": generated_normalized,
        "target_title_normalized": target_normalized,
        "confidence": confidence,
        "correctness": row.get("correctness"),
        "grounding_status": row.get("grounding_status"),
        "grounded_item_id": row.get("grounded_item_id"),
        "grounding_score": row.get("grounding_score"),
        "target_popularity_bucket": row.get("target_popularity_bucket"),
        "generated_target_similarity": target_similarity,
        "top_candidate_score": top_score,
        "second_candidate_score": second_score,
        "top_candidates": catalog_candidates,
        "failure_categories": categories,
        "recommended_actions": _failure_actions(categories),
    }


def grounding_failure_review(
    grounded_rows: Iterable[dict[str, Any]],
    catalog_rows: Iterable[dict[str, Any]],
    duplicate_groups: Iterable[dict[str, Any]],
    *,
    near_miss_threshold: float = 0.70,
    weak_match_threshold: float = 0.40,
    high_confidence_threshold: float = 0.70,
    max_cases: int = 50,
) -> dict[str, Any]:
    rows = list(grounded_rows)
    catalog = list(catalog_rows)
    duplicate_normalized_titles = {
        str(group.get("normalized_title") or "")
        for group in duplicate_groups
        if str(group.get("normalized_title") or "")
    }
    failure_rows = [
        row
        for row in rows
        if not row.get("grounded_item_id")
        or str(row.get("grounding_status") or "").lower()
        in {"ambiguous", "ungrounded", "out_of_catalog", "parse_failed"}
    ]
    cases = [
        classify_grounding_failure(
            row,
            catalog_rows=catalog,
            duplicate_normalized_titles=duplicate_normalized_titles,
            near_miss_threshold=near_miss_threshold,
            weak_match_threshold=weak_match_threshold,
            high_confidence_threshold=high_confidence_threshold,
        )
        for row in failure_rows
    ]
    cases.sort(
        key=lambda row: (
            -float(row.get("confidence") or 0.0),
            -float(row.get("top_candidate_score") or 0.0),
            str(row.get("input_id") or ""),
        )
    )
    category_counts = Counter(
        category for case in cases for category in case.get("failure_categories", [])
    )
    action_counts = Counter(action for case in cases for action in case.get("recommended_actions", []))
    status_counts = Counter(str(row.get("grounding_status") or "unknown") for row in failure_rows)
    return {
        "failure_row_count": len(failure_rows),
        "failure_fraction": (len(failure_rows) / len(rows)) if rows else 0.0,
        "near_miss_threshold": near_miss_threshold,
        "weak_match_threshold": weak_match_threshold,
        "high_confidence_threshold": high_confidence_threshold,
        "status_counts": dict(sorted(status_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "recommended_action_counts": dict(sorted(action_counts.items())),
        "cases": cases[:max_cases],
    }


def grounding_margin_summary(
    grounded_rows: Iterable[dict[str, Any]],
    *,
    margin_threshold: float = 0.03,
    max_cases: int = 50,
) -> dict[str, Any]:
    rows = list(grounded_rows)
    margins = [margin for row in rows if (margin := _candidate_margin(row)) is not None]
    low_margin_rows = [
        row
        for row in rows
        if (margin := _candidate_margin(row)) is not None and margin <= margin_threshold
    ]
    status_counts = Counter(str(row.get("grounding_status") or "unknown") for row in rows)
    low_margin_status_counts = Counter(
        str(row.get("grounding_status") or "unknown") for row in low_margin_rows
    )
    low_margin_cases = sorted(
        (_compact_low_margin_case(row) for row in low_margin_rows),
        key=lambda row: (
            float(row["grounding_margin"]) if row["grounding_margin"] is not None else 1.0,
            str(row.get("input_id") or ""),
        ),
    )
    return {
        "grounded_row_count": len(rows),
        "rows_with_top_two_scores": len(margins),
        "margin_threshold": margin_threshold,
        "low_margin_count": len(low_margin_rows),
        "low_margin_fraction": (len(low_margin_rows) / len(margins)) if margins else 0.0,
        "min_margin": min(margins) if margins else None,
        "mean_margin": (sum(margins) / len(margins)) if margins else None,
        "status_counts": dict(sorted(status_counts.items())),
        "low_margin_status_counts": dict(sorted(low_margin_status_counts.items())),
        "low_margin_cases": low_margin_cases[:max_cases],
    }


def grounding_diagnostics_markdown(summary: dict[str, Any]) -> str:
    catalog = summary["catalog"]
    margins = summary.get("observation_margins")
    failures = summary.get("grounding_failures")
    lines = [
        f"# Grounding Diagnostics: {summary.get('dataset')} / {summary.get('processed_suffix')}",
        "",
        "This report audits catalog title ambiguity and optional observation grounding margins. It is diagnostic only and not paper evidence.",
        "",
        "## Catalog",
        "",
        f"- Catalog items: {catalog['catalog_item_count']}",
        f"- Empty titles: {catalog['empty_title_count']}",
        f"- Duplicate normalized title groups: {catalog['duplicate_normalized_group_count']}",
        f"- Items inside duplicate groups: {catalog['duplicate_normalized_item_count']}",
        f"- Duplicate item fraction: {catalog['duplicate_item_fraction']:.6f}",
        f"- Catalog bucket counts: {catalog['catalog_bucket_counts']}",
        "",
        "## Top Duplicate Normalized Titles",
        "",
        "| normalized title | count | max popularity | bucket counts |",
        "| --- | ---: | ---: | --- |",
    ]
    for group in summary.get("duplicate_groups", [])[:10]:
        lines.append(
            "| {title} | {count} | {pop} | {buckets} |".format(
                title=group["normalized_title"],
                count=group["count"],
                pop=group["max_popularity"],
                buckets=group["bucket_counts"],
            )
        )
    if margins is not None:
        lines.extend(
            [
                "",
                "## Observation Grounding Margins",
                "",
                f"- Grounded rows: {margins['grounded_row_count']}",
                f"- Rows with top-two scores: {margins['rows_with_top_two_scores']}",
                f"- Margin threshold: {margins['margin_threshold']}",
                f"- Low-margin rows: {margins['low_margin_count']}",
                f"- Low-margin fraction: {margins['low_margin_fraction']}",
                f"- Min margin: {margins['min_margin']}",
                f"- Mean margin: {margins['mean_margin']}",
                f"- Status counts: {margins['status_counts']}",
                f"- Low-margin status counts: {margins['low_margin_status_counts']}",
            ]
        )
    if failures is not None:
        lines.extend(
            [
                "",
                "## Grounding Failure Taxonomy",
                "",
                f"- Failure rows: {failures['failure_row_count']}",
                f"- Failure fraction: {failures['failure_fraction']}",
                f"- Near-miss threshold: {failures['near_miss_threshold']}",
                f"- Weak-match threshold: {failures['weak_match_threshold']}",
                f"- High-confidence threshold: {failures['high_confidence_threshold']}",
                f"- Failure status counts: {failures['status_counts']}",
                f"- Failure category counts: {failures['category_counts']}",
                f"- Recommended action counts: {failures['recommended_action_counts']}",
                "",
                "Top cases are sorted by confidence and catalog-candidate score for manual QA.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "Duplicate normalized titles and low candidate margins indicate grounding risk. They do not prove model behavior or recommendation quality by themselves.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_grounding_diagnostics(
    *,
    catalog_csv: str | Path,
    output_dir: str | Path,
    dataset: str | None = None,
    processed_suffix: str | None = None,
    grounded_jsonl: str | Path | None = None,
    manifest_json: str | Path | None = None,
    margin_threshold: float = 0.03,
    near_miss_threshold: float = 0.70,
    weak_match_threshold: float = 0.40,
    high_confidence_threshold: float = 0.70,
    max_groups: int = 50,
    max_cases: int = 50,
) -> dict[str, Any]:
    catalog_path = Path(catalog_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    catalog_rows = load_catalog_csv(catalog_path)
    duplicate_groups = duplicate_title_groups(catalog_rows, max_groups=max_groups)
    grounded_rows = read_jsonl(grounded_jsonl) if grounded_jsonl else []
    manifest = (
        json.loads(Path(manifest_json).read_text(encoding="utf-8"))
        if manifest_json and Path(manifest_json).exists()
        else {}
    )
    margin_info = (
        grounding_margin_summary(
            grounded_rows,
            margin_threshold=margin_threshold,
            max_cases=max_cases,
        )
        if grounded_jsonl
        else None
    )
    failure_info = (
        grounding_failure_review(
            grounded_rows,
            catalog_rows,
            duplicate_groups,
            near_miss_threshold=near_miss_threshold,
            weak_match_threshold=weak_match_threshold,
            high_confidence_threshold=high_confidence_threshold,
            max_cases=max_cases,
        )
        if grounded_jsonl
        else None
    )
    summary = {
        "created_at_utc": utc_now_iso(),
        "dataset": dataset,
        "processed_suffix": processed_suffix,
        "catalog_csv": str(catalog_path),
        "grounded_jsonl": str(grounded_jsonl) if grounded_jsonl else None,
        "source_manifest_json": str(manifest_json) if manifest_json else None,
        "provider": manifest.get("provider"),
        "model": manifest.get("model"),
        "dry_run": manifest.get("dry_run"),
        "api_called": manifest.get("api_called"),
        "catalog": catalog_grounding_summary(catalog_rows),
        "duplicate_groups": duplicate_groups,
        "observation_margins": margin_info,
        "grounding_failures": failure_info,
        "is_experiment_result": False,
        "note": "Grounding diagnostics only. Not model behavior or paper evidence.",
    }

    summary_path = output_path / "grounding_diagnostics_summary.json"
    duplicates_path = output_path / "duplicate_title_groups.jsonl"
    low_margin_path = output_path / "low_margin_cases.jsonl"
    failure_cases_path = output_path / "grounding_failure_cases.jsonl"
    report_path = output_path / "grounding_diagnostics_report.md"
    manifest_out_path = output_path / "grounding_diagnostics_manifest.json"

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    write_jsonl(duplicates_path, duplicate_groups)
    write_jsonl(
        low_margin_path,
        margin_info["low_margin_cases"] if margin_info is not None else [],
    )
    write_jsonl(
        failure_cases_path,
        failure_info["cases"] if failure_info is not None else [],
    )
    report_path.write_text(grounding_diagnostics_markdown(summary), encoding="utf-8")
    analysis_manifest = {
        "created_at_utc": utc_now_iso(),
        "analysis_dir": str(output_path),
        "summary": str(summary_path),
        "duplicate_title_groups": str(duplicates_path),
        "low_margin_cases": str(low_margin_path),
        "grounding_failure_cases": str(failure_cases_path),
        "report": str(report_path),
        "source_catalog_csv": str(catalog_path),
        "source_grounded_jsonl": str(grounded_jsonl) if grounded_jsonl else None,
        "source_manifest_json": str(manifest_json) if manifest_json else None,
        "dataset": dataset,
        "processed_suffix": processed_suffix,
        "provider": summary.get("provider"),
        "model": summary.get("model"),
        "dry_run": summary.get("dry_run"),
        "api_called": summary.get("api_called"),
        "is_experiment_result": False,
        "note": "Grounding diagnostics artifact only.",
    }
    manifest_out_path.write_text(
        json.dumps(analysis_manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return analysis_manifest
