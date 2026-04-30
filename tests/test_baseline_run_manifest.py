from __future__ import annotations

import json
import uuid
from pathlib import Path

from scripts.validate_baseline_run_manifest import main as validate_run_manifest_main
from storyflow.baselines import validate_baseline_run_manifest


def _workspace_tmp(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_text(path: Path, text: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _valid_run_manifest(workspace: Path) -> dict[str, object]:
    input_jsonl = _write_text(workspace / "inputs.jsonl", '{"input_id":"x"}\n')
    ranking_jsonl = _write_text(workspace / "rankings.jsonl", '{"input_id":"x","ranked_item_ids":["i1"]}\n')
    config_path = _write_text(workspace / "config.yaml", "seed: 7\n")
    processed_manifest = _write_text(workspace / "preprocess_manifest.json", '{"dataset":"synthetic_fixture"}')
    train_manifest = _write_text(workspace / "train_manifest.json", '{"model":"SASRec"}')
    stdout_log = _write_text(workspace / "stdout.log", "completed\n")
    stderr_log = _write_text(workspace / "stderr.log", "")
    return {
        "schema_version": "baseline_ranking_run_manifest_v1",
        "baseline_family": "sasrec",
        "model_family": "SASRec",
        "run_label": "sasrec_fixture",
        "dataset": "synthetic_fixture",
        "processed_suffix": "tiny",
        "train_splits": ["train"],
        "validation_splits": ["validation"],
        "evaluation_split": "test",
        "input_jsonl": str(input_jsonl),
        "ranking_jsonl": str(ranking_jsonl),
        "ranking_output_schema": "ranking_jsonl_v1",
        "config_path": str(config_path),
        "processed_manifest": str(processed_manifest),
        "train_manifest": str(train_manifest),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "command": "python scripts/server/train_sasrec.py --config fixture.yaml",
        "git_commit": "500564b",
        "seed": 7,
        "grounding_required_before_correctness": True,
        "uses_heldout_targets_for_training": False,
        "api_called": False,
        "server_executed": True,
        "model_training_run": True,
        "is_experiment_result": False,
        "is_paper_result": False,
    }


def test_validate_baseline_run_manifest_passes_and_hashes_paths() -> None:
    workspace = _workspace_tmp("baseline_run_manifest_valid")
    manifest_json = workspace / "run_manifest.json"
    output_validation_json = workspace / "validation.json"
    manifest_json.write_text(
        json.dumps(_valid_run_manifest(workspace), indent=2),
        encoding="utf-8",
    )

    validation = validate_baseline_run_manifest(
        manifest_json=manifest_json,
        output_validation_json=output_validation_json,
        strict=True,
    )

    assert validation["validation_status"] == "passed"
    assert validation["path_info"]["input_jsonl"]["sha256"]
    assert validation["path_info"]["ranking_jsonl"]["sha256"]
    assert validation["validator_api_called"] is False
    assert validation["validator_model_training"] is False
    assert validation["validator_server_executed"] is False
    assert validation["is_experiment_result"] is False
    assert output_validation_json.exists()


def test_validate_baseline_run_manifest_flags_split_and_claim_failures() -> None:
    workspace = _workspace_tmp("baseline_run_manifest_invalid")
    manifest_json = workspace / "run_manifest.json"
    record = _valid_run_manifest(workspace)
    record.update(
        {
            "train_splits": ["train", "test"],
            "evaluation_split": "test",
            "ranking_output_schema": "unknown_schema",
            "grounding_required_before_correctness": False,
            "uses_heldout_targets_for_training": True,
            "seed": -1,
            "git_commit": "abc",
            "is_experiment_result": True,
        }
    )
    manifest_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

    validation = validate_baseline_run_manifest(
        manifest_json=manifest_json,
        strict=True,
        require_artifact_paths=False,
    )
    errors = validation["problem_summary"]["errors"]

    assert validation["validation_status"] == "failed"
    assert errors["evaluation_split_overlaps_train"] == 1
    assert errors["unsupported_ranking_output_schema"] == 1
    assert errors["grounding_guard_not_true"] == 1
    assert errors["heldout_target_training_leakage_guard_failed"] == 1
    assert errors["invalid_seed"] == 1
    assert errors["invalid_git_commit"] == 1
    assert errors["result_claim_in_source_manifest"] == 1


def test_validate_baseline_run_manifest_cli_does_not_require_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    workspace = _workspace_tmp("baseline_run_manifest_cli")
    manifest_json = workspace / "run_manifest.json"
    output_validation_json = workspace / "cli_validation.json"
    manifest_json.write_text(
        json.dumps(_valid_run_manifest(workspace), indent=2),
        encoding="utf-8",
    )

    code = validate_run_manifest_main(
        [
            "--manifest-json",
            str(manifest_json),
            "--output-validation-json",
            str(output_validation_json),
            "--strict",
        ]
    )
    validation = json.loads(output_validation_json.read_text(encoding="utf-8"))

    assert code == 0
    assert validation["validation_status"] == "passed"
    assert validation["validator_api_called"] is False
