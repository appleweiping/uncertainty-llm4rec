from __future__ import annotations

import json
import uuid
from pathlib import Path

from scripts.build_expansion_run_packet import build_expansion_run_packet, main as packet_main


def _workspace(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_api_run_packet_is_non_executing_and_lists_missing_confirmations() -> None:
    packet = build_expansion_run_packet(
        "api_provider",
        run_label="deepseek_smoke_packet",
        provider="deepseek",
        model="deepseek-v4-flash",
        sample_size=5,
        rate_limit=10,
        max_concurrency=1,
        provider_config="configs/providers/deepseek.yaml",
        input_jsonl="outputs/observation_inputs/example.jsonl",
    )

    assert packet["api_called"] is False
    assert packet["server_executed"] is False
    assert packet["model_training"] is False
    assert packet["is_experiment_result"] is False
    assert packet["grounding_required_before_correctness"] is True
    assert packet["ready_for_execution_after_user_approval"] is False
    assert "budget_label" in packet["missing_confirmations"]
    assert "environment variable exists" in packet["missing_confirmations"]
    assert "--execute-api" in packet["commands"]["approval_required_execute"]
    assert any("Do not describe this packet as an executed run" in item for item in packet["forbidden_claims"])


def test_amazon_full_prepare_packet_can_be_complete_but_still_requires_approval() -> None:
    packet = build_expansion_run_packet(
        "amazon_full_prepare",
        run_label="books_full_prepare_packet",
        dataset="amazon_reviews_2023_books",
        reviews_jsonl="data/raw/amazon_reviews_2023_books/Books.jsonl",
        metadata_jsonl="data/raw/amazon_reviews_2023_books/meta_Books.jsonl",
        processed_suffix="full",
        license_accepted=True,
        machine_disk_budget="server with sufficient disk",
        execution_location="server",
    )

    assert packet["ready_for_execution_after_user_approval"] is True
    assert packet["full_data_processed"] is False
    assert packet["data_downloaded"] is False
    assert "--allow-full" in packet["commands"]["approval_required_execute"]
    assert "preprocess_manifest.json" in packet["expected_artifacts"]


def test_run_packet_cli_writes_json_and_markdown() -> None:
    workspace = _workspace("run_packet")

    code = packet_main(
        [
            "--track",
            "baseline_artifact",
            "--run-label",
            "sasrec_packet",
            "--input-jsonl",
            "outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json.jsonl",
            "--dataset",
            "amazon_reviews_2023_beauty",
            "--processed-suffix",
            "full",
            "--split",
            "test",
            "--baseline-family",
            "sasrec",
            "--model-family",
            "sequential",
            "--ranking-jsonl",
            "runs/baselines/sasrec/ranking.jsonl",
            "--run-manifest-json",
            "runs/baselines/sasrec/run_manifest.json",
            "--output-dir",
            str(workspace),
        ]
    )

    manifest = json.loads((workspace / "expansion_run_packet.json").read_text(encoding="utf-8"))
    report = (workspace / "expansion_run_packet.md").read_text(encoding="utf-8")

    assert code == 0
    assert manifest["track"] == "baseline_artifact"
    assert manifest["baseline_training_run"] is False
    assert "training machine/server provenance" in manifest["missing_confirmations"]
    assert "validate_baseline_run_manifest.py" in report
    assert "run packet only" in report
