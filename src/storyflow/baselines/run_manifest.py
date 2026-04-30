"""Validation for upstream baseline ranking run manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from storyflow.observation import utc_now_iso

SCHEMA_VERSION = "baseline_ranking_run_manifest_v1"
VALIDATION_SCHEMA_VERSION = "baseline_ranking_run_manifest_validation_v1"
SUPPORTED_RANKING_SCHEMAS = {"ranking_jsonl_v1", "storyflow_ranking_jsonl_v1"}

REQUIRED_FIELDS = {
    "schema_version",
    "baseline_family",
    "model_family",
    "run_label",
    "dataset",
    "processed_suffix",
    "train_splits",
    "evaluation_split",
    "input_jsonl",
    "ranking_jsonl",
    "ranking_output_schema",
    "command",
    "git_commit",
    "seed",
    "grounding_required_before_correctness",
    "uses_heldout_targets_for_training",
}


def _resolve_path(path_value: str | Path, *, manifest_dir: Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    manifest_relative = manifest_dir / path
    if manifest_relative.exists():
        return manifest_relative
    return repo_root / path


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _problem(
    problems: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    detail: dict[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {"code": code, "message": message}
    if detail:
        row["detail"] = detail
    problems.append(row)


def _summary(problems: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(problem["code"]) for problem in problems))


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def default_baseline_run_manifest_validation_path(
    *,
    manifest_json: str | Path,
    root: str | Path = ".",
) -> Path:
    manifest_path = Path(manifest_json)
    stem = manifest_path.stem or "baseline_run_manifest"
    return Path(root) / "outputs" / "baseline_run_manifest_validation" / stem / "validation.json"


def validate_baseline_run_manifest(
    *,
    manifest_json: str | Path,
    output_validation_json: str | Path | None = None,
    strict: bool = False,
    require_artifact_paths: bool = True,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Validate provenance for an upstream baseline ranking run.

    This validates the manifest of a run that produced a ranking JSONL artifact.
    It does not train or execute the ranker and it does not establish a paper
    result. The ranking JSONL must still pass `validate_baseline_artifact.py`
    before entering title-level grounding.
    """

    root = Path(repo_root)
    manifest_path = Path(manifest_json)
    if not manifest_path.exists():
        raise FileNotFoundError(f"baseline run manifest not found: {manifest_path}")
    manifest_dir = manifest_path.parent
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("baseline run manifest must be a JSON object")

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []

    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        _problem(
            errors,
            code="missing_required_fields",
            message="baseline run manifest is missing required fields",
            detail={"fields": missing},
        )

    if str(record.get("schema_version") or "") != SCHEMA_VERSION:
        _problem(
            errors if strict else warnings,
            code="schema_version_mismatch",
            message="baseline run manifest schema_version does not match the current contract",
            detail={"expected": SCHEMA_VERSION, "actual": record.get("schema_version")},
        )

    for field in ("baseline_family", "model_family", "run_label", "dataset", "processed_suffix"):
        if not str(record.get(field) or "").strip():
            _problem(
                errors,
                code=f"empty_{field}",
                message=f"{field} must be a non-empty string",
            )

    train_splits = _as_str_list(record.get("train_splits"))
    evaluation_split = str(record.get("evaluation_split") or "")
    validation_splits = _as_str_list(record.get("validation_splits"))
    if not train_splits:
        _problem(errors, code="empty_train_splits", message="train_splits must not be empty")
    if not evaluation_split:
        _problem(errors, code="empty_evaluation_split", message="evaluation_split must not be empty")
    if evaluation_split and evaluation_split in set(train_splits):
        _problem(
            errors,
            code="evaluation_split_overlaps_train",
            message="evaluation_split must not be part of train_splits",
            detail={"evaluation_split": evaluation_split, "train_splits": train_splits},
        )
    overlap_validation = sorted(set(train_splits) & set(validation_splits))
    if overlap_validation:
        _problem(
            errors,
            code="validation_split_overlaps_train",
            message="validation_splits must not overlap train_splits",
            detail={"overlap": overlap_validation},
        )

    ranking_schema = str(record.get("ranking_output_schema") or "")
    if ranking_schema not in SUPPORTED_RANKING_SCHEMAS:
        _problem(
            errors if strict else warnings,
            code="unsupported_ranking_output_schema",
            message="ranking_output_schema is not a supported Storyflow ranking JSONL schema",
            detail={"supported": sorted(SUPPORTED_RANKING_SCHEMAS), "actual": ranking_schema},
        )

    if record.get("grounding_required_before_correctness") is not True:
        _problem(
            errors,
            code="grounding_guard_not_true",
            message="grounding_required_before_correctness must be true",
        )
    if record.get("uses_heldout_targets_for_training") is not False:
        _problem(
            errors,
            code="heldout_target_training_leakage_guard_failed",
            message="uses_heldout_targets_for_training must be false",
        )
    if record.get("api_called") is True:
        _problem(
            warnings,
            code="api_called_in_baseline_run",
            message="baseline run manifest declares api_called=true; this must be separately approved",
        )
    if record.get("is_paper_result") is True or record.get("is_experiment_result") is True:
        _problem(
            errors if strict else warnings,
            code="result_claim_in_source_manifest",
            message="source ranking run must not self-declare paper or experiment results before grounded analysis",
        )

    try:
        seed = int(record.get("seed"))
        if seed < 0:
            raise ValueError
    except (TypeError, ValueError):
        _problem(errors, code="invalid_seed", message="seed must be a non-negative integer")

    command = str(record.get("command") or "").strip()
    if not command:
        _problem(errors, code="empty_command", message="command must record the upstream run command")
    if "--execute-api" in command:
        _problem(
            warnings,
            code="execute_api_flag_in_baseline_command",
            message="baseline manifest command contains --execute-api; verify this is not an LLM/API run",
        )
    git_commit = str(record.get("git_commit") or "").strip()
    if len(git_commit) < 7:
        _problem(errors, code="invalid_git_commit", message="git_commit must be at least a short commit hash")

    path_fields = [
        "input_jsonl",
        "ranking_jsonl",
        "config_path",
        "processed_manifest",
        "train_manifest",
        "stdout_log",
        "stderr_log",
    ]
    path_info: dict[str, dict[str, Any]] = {}
    for field in path_fields:
        value = record.get(field)
        if not value:
            if field in {"config_path", "processed_manifest"}:
                _problem(
                    warnings,
                    code=f"missing_{field}",
                    message=f"{field} is recommended for baseline reproducibility",
                )
            continue
        resolved = _resolve_path(value, manifest_dir=manifest_dir, repo_root=root)
        digest = _sha256(resolved)
        path_info[field] = {
            "declared": str(value),
            "resolved": str(resolved),
            "exists": resolved.exists(),
            "sha256": digest,
        }
        if require_artifact_paths and not resolved.exists():
            _problem(
                errors if strict else warnings,
                code=f"{field}_missing",
                message=f"{field} path could not be resolved locally",
                detail={"declared": str(value), "resolved": str(resolved)},
            )

    if "ranking_jsonl" in path_info and "input_jsonl" in path_info:
        _problem(
            infos,
            code="next_validator",
            message=(
                "After this run manifest passes, validate the ranking JSONL "
                "with scripts/validate_baseline_artifact.py before grounded observation."
            ),
        )

    status = "failed" if errors else "warning" if warnings else "passed"
    validation = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "validation_status": status,
        "strict": strict,
        "require_artifact_paths": require_artifact_paths,
        "manifest_json": str(manifest_path),
        "manifest_json_sha256": _sha256(manifest_path),
        "source_schema_version": record.get("schema_version"),
        "baseline_family": record.get("baseline_family"),
        "model_family": record.get("model_family"),
        "run_label": record.get("run_label"),
        "dataset": record.get("dataset"),
        "processed_suffix": record.get("processed_suffix"),
        "train_splits": train_splits,
        "validation_splits": validation_splits,
        "evaluation_split": evaluation_split,
        "ranking_output_schema": record.get("ranking_output_schema"),
        "path_info": path_info,
        "problem_summary": {
            "errors": _summary(errors),
            "warnings": _summary(warnings),
            "infos": _summary(infos),
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
        "grounding_required_before_correctness": True,
        "is_experiment_result": False,
        "note": (
            "Baseline source run manifest validation only. This does not train "
            "or execute the baseline and does not create a paper result."
        ),
    }
    if output_validation_json is not None:
        output_path = Path(output_validation_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        validation["output_validation_json"] = str(output_path)
        output_path.write_text(
            json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    return validation
