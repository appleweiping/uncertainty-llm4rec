from __future__ import annotations

import json
import uuid
from pathlib import Path

from scripts.build_local_deepseek_experiment_plan import (
    build_local_deepseek_experiment_plan,
    main as plan_main,
)


def _workspace(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_local_deepseek_plan_is_non_executing_and_defers_server(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    plan = build_local_deepseek_experiment_plan(
        datasets=["amazon_reviews_2023_tiny"],
        processed_suffix="full",
        split="test",
        gate_size=5,
        candidate_count=20,
        repeat_target_policy="exclude",
        provider_config_path=Path("configs/providers/deepseek.yaml"),
        run_stage="smoke",
        rate_limit=10,
        max_concurrency=1,
        budget_label=None,
        run_label_prefix="unit_local_plan",
    )

    assert plan["api_called"] is False
    assert plan["server_executed"] is False
    assert plan["model_training"] is False
    assert plan["server_deferred"] is True
    assert plan["small_model_training_deferred"] is True
    assert plan["is_experiment_result"] is False
    assert plan["api_key_value_printed"] is False
    assert "budget_label" in plan["missing_execution_confirmations"]
    assert "environment variable DEEPSEEK_API_KEY" in plan["missing_execution_confirmations"]

    dataset = plan["datasets"][0]
    assert dataset["dataset"] == "amazon_reviews_2023_tiny"
    assert "--repeat-target-policy exclude" in dataset["build_gate_inputs_command"]
    assert len(dataset["prompt_variants"]) == 3

    forced = next(
        variant
        for variant in dataset["prompt_variants"]
        if variant["prompt_template"] == "forced_json"
    )
    assert forced["input_jsonl"].endswith("test_gate5_no_repeat_forced_json.jsonl")
    assert forced["api_called"] is False
    assert "--dry-run" in forced["safe_preflight_commands"][1]
    assert "--execute-api" in forced["approval_required_execute_command"]
    assert "--budget-label <budget-label>" in forced["approval_required_execute_command"]
    assert all("qwen" not in command.lower() for command in forced["post_api_commands"])


def test_local_deepseek_plan_cli_writes_json_and_markdown(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key-for-tests")
    workspace = _workspace("local_deepseek_plan")

    code = plan_main(
        [
            "--dataset",
            "amazon_reviews_2023_beauty",
            "--gate-size",
            "5",
            "--run-stage",
            "pilot",
            "--rate-limit",
            "10",
            "--max-concurrency",
            "1",
            "--budget-label",
            "unit-test-budget",
            "--run-label-prefix",
            "unit_local_plan",
            "--output-dir",
            str(workspace),
        ]
    )

    manifest = json.loads((workspace / "local_deepseek_experiment_plan.json").read_text(encoding="utf-8"))
    report = (workspace / "local_deepseek_experiment_plan.md").read_text(encoding="utf-8")

    assert code == 0
    assert manifest["provider"] == "deepseek"
    assert manifest["model"] == "deepseek-v4-flash"
    assert manifest["api_key_env_present"] is True
    assert "budget_label" not in manifest["missing_execution_confirmations"]
    assert "environment variable DEEPSEEK_API_KEY" not in manifest["missing_execution_confirmations"]
    assert any("current-turn explicit approval" in item for item in manifest["missing_execution_confirmations"])
    assert manifest["outputs"]["json"].endswith("local_deepseek_experiment_plan.json")
    assert "Approval-required execution" in report
    assert "--execute-api" in report
    assert "server deferred: True" in report
