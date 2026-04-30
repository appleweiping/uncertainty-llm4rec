"""Validation manifests for external baseline ranking artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from storyflow.baselines.observation import parse_ranking_candidates
from storyflow.observation import load_catalog_rows, read_jsonl, utc_now_iso

SCHEMA_VERSION = "baseline_artifact_validation_v1"


def _resolve_existing_path(path_value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    if base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    return path


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_entries(record: dict[str, Any]) -> tuple[str | None, list[Any], list[Any]]:
    if isinstance(record.get("ranked_items"), list):
        return "ranked_items", record["ranked_items"], []
    if isinstance(record.get("recommendations"), list):
        return "recommendations", record["recommendations"], []
    if isinstance(record.get("ranked_item_ids"), list):
        scores = record.get("scores", record.get("ranked_scores", []))
        return "ranked_item_ids", record["ranked_item_ids"], scores if isinstance(scores, list) else []
    if isinstance(record.get("item_ids"), list):
        scores = record.get("scores", [])
        return "item_ids", record["item_ids"], scores if isinstance(scores, list) else []
    return None, [], []


def _candidate_item_id(entry: Any) -> str | None:
    if isinstance(entry, dict):
        for key in ("item_id", "id", "asin", "catalog_item_id"):
            value = entry.get(key)
            if value not in (None, ""):
                return str(value)
        return None
    if entry in (None, ""):
        return None
    return str(entry)


def _score_like(value: Any) -> bool:
    if value in (None, ""):
        return True
    try:
        score = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(score)


def _problem(
    problems: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    input_id: str | None = None,
    line_number: int | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {"code": code, "message": message}
    if input_id is not None:
        row["input_id"] = input_id
    if line_number is not None:
        row["line_number"] = line_number
    if detail:
        row["detail"] = detail
    problems.append(row)


def _problem_summary(problems: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(problem["code"]) for problem in problems))


def _source_catalog_path(input_rows: list[dict[str, Any]], input_path: Path) -> Path | None:
    for row in input_rows:
        source = row.get("source")
        if isinstance(source, dict) and source.get("catalog_csv"):
            return _resolve_existing_path(source["catalog_csv"], base_dir=input_path.parent)
    return None


def default_baseline_artifact_manifest_path(
    *,
    ranking_jsonl: str | Path,
    root: str | Path = ".",
) -> Path:
    ranking_path = Path(ranking_jsonl)
    stem = ranking_path.stem or "ranking_artifact"
    return Path(root) / "outputs" / "baseline_artifact_validation" / stem / "manifest.json"


def validate_baseline_artifact(
    *,
    ranking_jsonl: str | Path,
    input_jsonl: str | Path,
    baseline_family: str,
    output_manifest_json: str | Path | None = None,
    model_family: str | None = None,
    run_label: str | None = None,
    dataset: str | None = None,
    processed_suffix: str | None = None,
    split: str | None = None,
    trained_splits: list[str] | None = None,
    seed: int | None = None,
    config_path: str | Path | None = None,
    source_manifest_json: str | Path | None = None,
    max_examples: int | None = None,
    strict: bool = False,
    allow_missing_inputs: bool = False,
    fail_on_extra_rankings: bool = False,
) -> dict[str, Any]:
    """Validate an external ranking artifact before title-level observation.

    The validator does not execute a recommender, call an API, or train a model.
    It checks whether local ranked item IDs can be adapted into the shared
    title-grounding observation path with enough provenance to be reproducible.
    """

    ranking_path = Path(ranking_jsonl)
    input_path = Path(input_jsonl)
    if not ranking_path.exists():
        raise FileNotFoundError(f"ranking_jsonl not found: {ranking_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_path}")
    if not baseline_family:
        raise ValueError("baseline_family must not be empty")

    input_rows = read_jsonl(input_path)
    if max_examples is not None:
        input_rows = input_rows[:max_examples]
    ranking_rows = read_jsonl(ranking_path)

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []

    if not input_rows:
        _problem(errors, code="empty_input_jsonl", message="input_jsonl contains no records")
    if not ranking_rows:
        _problem(errors, code="empty_ranking_jsonl", message="ranking_jsonl contains no records")

    input_by_id: dict[str, dict[str, Any]] = {}
    duplicate_input_rows: list[str] = []
    for row in input_rows:
        input_id = str(row.get("input_id") or "")
        if not input_id:
            _problem(errors, code="input_missing_input_id", message="input row is missing input_id")
            continue
        if input_id in input_by_id:
            duplicate_input_rows.append(input_id)
        input_by_id[input_id] = row
    for input_id in sorted(set(duplicate_input_rows)):
        _problem(errors, code="duplicate_input_row", input_id=input_id, message="input_jsonl repeats input_id")

    catalog_path = _source_catalog_path(input_rows, input_path)
    catalog_by_id: dict[str, dict[str, Any]] = {}
    if catalog_path is not None and catalog_path.exists():
        catalog_by_id = {str(row["item_id"]): row for row in load_catalog_rows(catalog_path)}
    else:
        _problem(
            warnings,
            code="catalog_unavailable",
            message="input source catalog_csv could not be resolved; catalog-id checks are skipped",
            detail={"catalog_csv": str(catalog_path) if catalog_path is not None else None},
        )

    ranking_by_input: dict[str, dict[str, Any]] = {}
    duplicate_ranking_ids: list[str] = []
    format_counts: Counter[str] = Counter()
    candidate_count_distribution: Counter[str] = Counter()
    candidate_total = 0
    unknown_catalog_item_count = 0
    history_overlap_count = 0
    no_valid_unseen_count = 0
    target_item_in_ranking_count = 0

    for line_number, record in enumerate(ranking_rows, start=1):
        input_id = str(record.get("input_id") or "")
        if not input_id:
            _problem(
                errors,
                code="ranking_missing_input_id",
                line_number=line_number,
                message="ranking row is missing input_id",
            )
            continue
        if input_id in ranking_by_input:
            duplicate_ranking_ids.append(input_id)
        ranking_by_input[input_id] = record

        shape_key, entries, scores = _candidate_entries(record)
        if shape_key is None:
            _problem(
                errors,
                code="ranking_missing_candidates",
                input_id=input_id,
                line_number=line_number,
                message="ranking row has no supported candidate list",
            )
            continue
        format_counts[shape_key] += 1
        if shape_key in {"ranked_item_ids", "item_ids"} and scores and len(scores) != len(entries):
            _problem(
                errors if strict else warnings,
                code="score_length_mismatch",
                input_id=input_id,
                line_number=line_number,
                message="scores length does not match ranked item ids length",
                detail={"candidate_count": len(entries), "score_count": len(scores)},
            )
        if not entries:
            _problem(
                errors,
                code="empty_ranking_candidates",
                input_id=input_id,
                line_number=line_number,
                message="ranking row has an empty candidate list",
            )
            continue

        raw_ids = [_candidate_item_id(entry) for entry in entries]
        missing_item_ids = [index + 1 for index, item_id in enumerate(raw_ids) if item_id is None]
        if missing_item_ids:
            _problem(
                errors,
                code="candidate_missing_item_id",
                input_id=input_id,
                line_number=line_number,
                message="one or more candidates are missing item ids",
                detail={"candidate_positions": missing_item_ids[:20]},
            )
        compact_ids = [item_id for item_id in raw_ids if item_id is not None]
        duplicate_candidate_ids = [
            item_id for item_id, count in Counter(compact_ids).items() if count > 1
        ]
        if duplicate_candidate_ids:
            _problem(
                errors if strict else warnings,
                code="duplicate_candidate_item_id",
                input_id=input_id,
                line_number=line_number,
                message="ranking row repeats candidate item ids; adapter will deduplicate",
                detail={"item_ids": sorted(duplicate_candidate_ids)[:20]},
            )

        for index, entry in enumerate(entries):
            if isinstance(entry, dict):
                for key in ("score", "logit", "rating", "relevance_score"):
                    if key in entry and not _score_like(entry.get(key)):
                        _problem(
                            errors if strict else warnings,
                            code="nonnumeric_score",
                            input_id=input_id,
                            line_number=line_number,
                            message="candidate score is not numeric",
                            detail={"candidate_position": index + 1, "score_key": key},
                        )
        for index, score in enumerate(scores):
            if not _score_like(score):
                _problem(
                    errors if strict else warnings,
                    code="nonnumeric_score",
                    input_id=input_id,
                    line_number=line_number,
                    message="score list contains a nonnumeric score",
                    detail={"score_position": index + 1},
                )

        try:
            parsed_candidates = parse_ranking_candidates(record)
        except ValueError as exc:
            _problem(
                errors,
                code="ranking_parse_error",
                input_id=input_id,
                line_number=line_number,
                message=str(exc),
            )
            continue
        candidate_count_distribution[str(len(parsed_candidates))] += 1
        candidate_total += len(parsed_candidates)

        input_record = input_by_id.get(input_id)
        history_ids = (
            {str(item_id) for item_id in input_record.get("history_item_ids", [])}
            if input_record is not None
            else set()
        )
        target_item_id = str(input_record.get("target_item_id") or "") if input_record else ""
        valid_unseen = 0
        for candidate in parsed_candidates:
            if catalog_by_id and candidate.item_id not in catalog_by_id:
                unknown_catalog_item_count += 1
            if candidate.item_id in history_ids:
                history_overlap_count += 1
            if target_item_id and candidate.item_id == target_item_id:
                target_item_in_ranking_count += 1
            if (not catalog_by_id or candidate.item_id in catalog_by_id) and candidate.item_id not in history_ids:
                valid_unseen += 1
        if input_record is not None and valid_unseen == 0:
            no_valid_unseen_count += 1
            _problem(
                errors if strict else warnings,
                code="no_valid_unseen_catalog_item",
                input_id=input_id,
                line_number=line_number,
                message="all ranked candidates are history items or outside the catalog",
            )

    for input_id in sorted(set(duplicate_ranking_ids)):
        _problem(
            errors,
            code="duplicate_ranking_input_id",
            input_id=input_id,
            message="ranking_jsonl repeats input_id",
        )

    input_ids = set(input_by_id)
    ranking_ids = set(ranking_by_input)
    missing_inputs = sorted(input_ids - ranking_ids)
    extra_rankings = sorted(ranking_ids - input_ids)
    if missing_inputs and not allow_missing_inputs:
        _problem(
            errors,
            code="missing_input_ranking",
            message="ranking_jsonl does not cover every selected input_id",
            detail={"count": len(missing_inputs), "examples": missing_inputs[:20]},
        )
    elif missing_inputs:
        _problem(
            warnings,
            code="missing_input_ranking_allowed",
            message="ranking_jsonl is missing selected input_id values, but allow_missing_inputs is set",
            detail={"count": len(missing_inputs), "examples": missing_inputs[:20]},
        )
    if extra_rankings:
        _problem(
            errors if fail_on_extra_rankings else warnings,
            code="extra_ranking_input",
            message="ranking_jsonl contains input_id values outside the selected input_jsonl slice",
            detail={"count": len(extra_rankings), "examples": extra_rankings[:20]},
        )

    if catalog_by_id and unknown_catalog_item_count:
        _problem(
            errors if strict else warnings,
            code="unknown_catalog_item_id",
            message="ranking_jsonl contains item ids that are not in the input catalog",
            detail={"count": unknown_catalog_item_count},
        )
    if history_overlap_count:
        _problem(
            warnings,
            code="history_item_in_ranking",
            message="ranking_jsonl contains already-seen history items; adapter will filter them",
            detail={"count": history_overlap_count},
        )
    if target_item_in_ranking_count:
        _problem(
            infos,
            code="target_item_in_ranking",
            message="held-out target item appears in at least one ranking list; this is a diagnostic, not a leak claim",
            detail={"count": target_item_in_ranking_count},
        )

    input_split_counts = Counter(str(row.get("split") or "unknown") for row in input_rows)
    input_dataset_counts = Counter(str(row.get("dataset") or "unknown") for row in input_rows)
    input_suffix_counts = Counter(str(row.get("processed_suffix") or "unknown") for row in input_rows)
    if split and input_split_counts and set(input_split_counts) != {split}:
        _problem(
            errors if strict else warnings,
            code="split_mismatch",
            message="selected inputs contain splits that differ from declared split",
            detail={"declared_split": split, "input_split_counts": dict(input_split_counts)},
        )
    if dataset and input_dataset_counts and set(input_dataset_counts) != {dataset}:
        _problem(
            errors if strict else warnings,
            code="dataset_mismatch",
            message="selected inputs contain datasets that differ from declared dataset",
            detail={"declared_dataset": dataset, "input_dataset_counts": dict(input_dataset_counts)},
        )
    if processed_suffix and input_suffix_counts and set(input_suffix_counts) != {processed_suffix}:
        _problem(
            errors if strict else warnings,
            code="processed_suffix_mismatch",
            message="selected inputs contain processed suffixes that differ from declared suffix",
            detail={
                "declared_processed_suffix": processed_suffix,
                "input_processed_suffix_counts": dict(input_suffix_counts),
            },
        )

    config_hash = None
    if config_path is not None:
        config_hash = _file_sha256(_resolve_existing_path(config_path, base_dir=input_path.parent))
        if config_hash is None:
            _problem(
                warnings,
                code="config_path_unavailable",
                message="declared config path could not be hashed",
                detail={"config_path": str(config_path)},
            )
    source_manifest_hash = None
    if source_manifest_json is not None:
        source_manifest_hash = _file_sha256(
            _resolve_existing_path(source_manifest_json, base_dir=input_path.parent)
        )
        if source_manifest_hash is None:
            _problem(
                warnings,
                code="source_manifest_unavailable",
                message="declared source manifest could not be hashed",
                detail={"source_manifest_json": str(source_manifest_json)},
            )

    status = "failed" if errors else "warning" if warnings else "passed"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "validation_status": status,
        "strict": strict,
        "input_jsonl": str(input_path),
        "ranking_jsonl": str(ranking_path),
        "ranking_jsonl_sha256": _file_sha256(ranking_path),
        "input_jsonl_sha256": _file_sha256(input_path),
        "catalog_csv": str(catalog_path) if catalog_path is not None else None,
        "catalog_item_count": len(catalog_by_id),
        "baseline_family": baseline_family,
        "model_family": model_family or baseline_family,
        "run_label": run_label,
        "dataset": dataset,
        "processed_suffix": processed_suffix,
        "split": split,
        "trained_splits": trained_splits or [],
        "seed": seed,
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": config_hash,
        "source_manifest_json": str(source_manifest_json) if source_manifest_json is not None else None,
        "source_manifest_sha256": source_manifest_hash,
        "max_examples": max_examples,
        "coverage": {
            "selected_input_count": len(input_rows),
            "ranking_record_count": len(ranking_rows),
            "matched_input_count": len(input_ids & ranking_ids),
            "missing_input_count": len(missing_inputs),
            "extra_ranking_count": len(extra_rankings),
            "allow_missing_inputs": allow_missing_inputs,
            "fail_on_extra_rankings": fail_on_extra_rankings,
        },
        "quality": {
            "candidate_total": candidate_total,
            "candidate_count_distribution": dict(candidate_count_distribution),
            "format_counts": dict(format_counts),
            "unknown_catalog_item_count": unknown_catalog_item_count,
            "history_overlap_count": history_overlap_count,
            "no_valid_unseen_catalog_item_count": no_valid_unseen_count,
            "target_item_in_ranking_count": target_item_in_ranking_count,
        },
        "input_slice": {
            "split_counts": dict(input_split_counts),
            "dataset_counts": dict(input_dataset_counts),
            "processed_suffix_counts": dict(input_suffix_counts),
        },
        "problem_summary": {
            "errors": _problem_summary(errors),
            "warnings": _problem_summary(warnings),
            "infos": _problem_summary(infos),
        },
        "problem_examples": {
            "errors": errors[:100],
            "warnings": warnings[:100],
            "infos": infos[:100],
        },
        "validator_api_called": False,
        "validator_model_training": False,
        "validator_server_executed": False,
        "validator_downloaded_data": False,
        "is_experiment_result": False,
        "grounding_required_before_correctness": True,
        "note": (
            "Validation manifest for a local external baseline ranking artifact. "
            "It does not execute the recommender, call an API, train a model, "
            "or establish paper evidence."
        ),
    }

    if output_manifest_json is not None:
        output_path = Path(output_manifest_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest["output_manifest_json"] = str(output_path)
        output_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    return manifest
