from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

import pytest

from scripts.server.run_qwen3_lora_sft import main as lora_main
from storyflow.observation import write_jsonl
from storyflow.training import (
    build_qwen_lora_training_plan,
    load_qwen_lora_training_config,
    run_qwen_lora_training,
)


def _workspace(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _input_jsonl(workspace: Path) -> Path:
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
        writer.writerow(
            {
                "item_id": "item-head",
                "title": "Head Serum",
                "title_normalized": "head serum",
                "popularity": 10,
                "popularity_bucket": "head",
            }
        )
    input_path = workspace / "inputs.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "input_id": "fixture:tiny:0",
                "dataset": "synthetic_fixture",
                "processed_suffix": "tiny",
                "example_id": "u1:1",
                "user_id": "u1",
                "split": "train",
                "history_item_titles": ["Tail Balm"],
                "target_item_id": "item-head",
                "target_title": "Head Serum",
                "target_popularity": 10,
                "target_popularity_bucket": "head",
                "prompt_template": "forced_json",
                "prompt": "Return JSON.",
                "prompt_hash": "hash",
                "source": {"catalog_csv": str(catalog_csv)},
            }
        ],
    )
    return input_path


def _config(workspace: Path, input_jsonl: Path) -> Path:
    config_path = workspace / "qwen_lora.yaml"
    config_path.write_text(
        "\n".join(
            [
                "backend: qwen_lora_sft",
                "model_name: Qwen/Qwen3-8B",
                "model_alias: qwen3_8b_lora_sft",
                "stage: sft_baseline",
                "seed: 7",
                "data:",
                f"  train_input_jsonl: {input_jsonl.as_posix()}",
                "  validation_input_jsonl: null",
                "  response_policy: target_title_json_confidence_1",
                "training:",
                f"  output_dir: {(workspace / 'train_out').as_posix()}",
                "  num_train_epochs: 1",
                "lora:",
                "  r: 8",
                "  alpha: 16",
                "  dropout: 0.05",
                "guards:",
                "  server_execution_required: true",
                "  no_local_codex_training: true",
                "output_contract:",
                "  plan_manifest: train_manifest.json",
                "  command_plan: train_command_plan.md",
                "  config_snapshot: config_snapshot.json",
                "  expected_artifacts: expected_training_artifacts.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_qwen_lora_training_config_loads() -> None:
    config = load_qwen_lora_training_config("configs/server/qwen3_8b_lora_sft.yaml")

    assert config["backend"] == "qwen_lora_sft"
    assert config["model_alias"] == "qwen3_8b_lora_sft"
    assert config["guards"]["server_execution_required"] is True


def test_qwen_lora_plan_writes_contract_without_training() -> None:
    workspace = _workspace("qwen_lora_plan")
    input_jsonl = _input_jsonl(workspace)
    config_path = _config(workspace, input_jsonl)
    output_dir = workspace / "plan"

    manifest = build_qwen_lora_training_plan(
        config_path=config_path,
        output_dir=output_dir,
    )

    assert manifest["api_called"] is False
    assert manifest["server_executed"] is False
    assert manifest["model_training"] is False
    assert manifest["train_input_exists"] is True
    assert manifest["claim_scope"] == "plan_only_not_training_not_paper_evidence"
    assert (output_dir / "train_command_plan.md").exists()
    assert (output_dir / "expected_training_artifacts.json").exists()
    assert (output_dir / "sft_preview_rows.jsonl").exists()


def test_qwen_lora_cli_default_is_plan_only() -> None:
    workspace = _workspace("qwen_lora_cli")
    input_jsonl = _input_jsonl(workspace)
    config_path = _config(workspace, input_jsonl)
    output_dir = workspace / "cli_plan"

    code = lora_main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    manifest = json.loads((output_dir / "train_manifest.json").read_text(encoding="utf-8"))

    assert code == 0
    assert manifest["model_training"] is False
    assert manifest["server_executed"] is False


def test_qwen_lora_refuses_local_execute_server() -> None:
    workspace = _workspace("qwen_lora_refuse")
    input_jsonl = _input_jsonl(workspace)
    config_path = _config(workspace, input_jsonl)

    with pytest.raises(RuntimeError, match="server-only"):
        run_qwen_lora_training(config_path=config_path, execute_server=True)
