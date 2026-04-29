from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from scripts.server.run_qwen3_observation import main as qwen_server_main
from storyflow.observation import write_jsonl
from storyflow.server import (
    build_qwen_server_observation_plan,
    default_qwen_server_output_dir,
    load_qwen_server_config,
)


def _workspace_tmp(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _input_jsonl(workspace: Path, *, n: int = 2) -> Path:
    catalog_csv = workspace / "item_catalog.csv"
    with catalog_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "title",
                "title_normalized",
                "popularity",
                "popularity_bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "item_id": "item-head",
                    "title": "Head Serum",
                    "title_normalized": "head serum",
                    "popularity": 100,
                    "popularity_bucket": "head",
                },
                {
                    "item_id": "item-tail",
                    "title": "Tail Balm",
                    "title_normalized": "tail balm",
                    "popularity": 3,
                    "popularity_bucket": "tail",
                },
            ]
        )
    rows = []
    for index in range(n):
        rows.append(
            {
                "input_id": f"fixture:tiny:{index}:hash",
                "dataset": "synthetic_fixture",
                "processed_suffix": "tiny",
                "example_id": f"u{index}:1",
                "user_id": f"u{index}",
                "split": "test",
                "history_item_ids": ["item-tail"],
                "history_item_titles": ["Tail Balm"],
                "history_timestamps": [1],
                "history_length": 1,
                "target_item_id": "item-head",
                "target_title": "Head Serum",
                "target_timestamp": 2,
                "target_popularity": 100,
                "target_popularity_bucket": "head",
                "prompt_template": "forced_json",
                "prompt": "Return JSON for the next title.",
                "prompt_hash": f"hash-{index}",
                "source": {
                    "catalog_csv": str(catalog_csv),
                    "processed_dir": str(workspace),
                    "observation_examples": str(workspace / "observation_examples.jsonl"),
                },
            }
        )
    input_path = workspace / "inputs.jsonl"
    write_jsonl(input_path, rows)
    return input_path


def test_qwen_server_config_loads_plan_only_contract() -> None:
    config = load_qwen_server_config("configs/server/qwen3_8b_observation.yaml")

    assert config["backend"] == "qwen_server"
    assert config["model_alias"] == "qwen3_8b"
    assert config["guards"]["server_execution_required"] is True
    assert config["output_contract"]["grounded_predictions"] == "grounded_predictions.jsonl"


def test_qwen_server_plan_writes_api_compatible_contract() -> None:
    workspace = _workspace_tmp("qwen_plan")
    input_jsonl = _input_jsonl(workspace, n=2)
    output_dir = workspace / "plan"

    manifest = build_qwen_server_observation_plan(
        config_path="configs/server/qwen3_8b_observation.yaml",
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        max_examples=2,
        run_label="unit-qwen-plan",
        run_stage="planned",
    )

    request_rows = [
        json.loads(line)
        for line in (output_dir / "request_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contract = json.loads((output_dir / "expected_output_contract.json").read_text(encoding="utf-8"))
    assert manifest["api_called"] is False
    assert manifest["server_executed"] is False
    assert manifest["model_inference_run"] is False
    assert manifest["model_training"] is False
    assert manifest["output_schema_matches_api_observation"] is True
    assert request_rows[0]["metadata"]["execution_mode"] == "server_plan_only"
    assert contract["schema_family"] == "api_observation_compatible"
    assert contract["grounding_required_before_correctness"] is True
    assert (output_dir / "server_command_plan.md").exists()


def test_qwen_server_cli_default_is_plan_only() -> None:
    workspace = _workspace_tmp("qwen_cli_plan")
    input_jsonl = _input_jsonl(workspace, n=1)
    output_dir = workspace / "cli_plan"

    code = qwen_server_main(
        [
            "--config",
            "configs/server/qwen3_8b_observation.yaml",
            "--input-jsonl",
            str(input_jsonl),
            "--output-dir",
            str(output_dir),
            "--max-examples",
            "1",
            "--run-label",
            "cli-plan",
        ]
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert code == 0
    assert manifest["execution_mode"] == "server_plan_only"
    assert manifest["api_called"] is False
    assert manifest["server_executed"] is False
    assert not (output_dir / "raw_responses.jsonl").exists()


def test_default_qwen_output_dir_is_under_ignored_outputs() -> None:
    path = default_qwen_server_output_dir(
        input_jsonl=Path("outputs/observation_inputs/dataset/run/test_forced_json.jsonl"),
        model_alias="qwen3_8b",
        root=Path("."),
    )

    assert path.parts[:2] == ("outputs", "server_observations")
    assert "qwen3_8b" in path.parts
