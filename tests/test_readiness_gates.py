from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from scripts.check_api_pilot_readiness import main as readiness_main
from scripts.inspect_amazon_category_matrix import main as amazon_matrix_main
from scripts.prepare_amazon_reviews_2023 import main as prepare_amazon_main
from storyflow.data import inspect_amazon_config, prepare_amazon_from_jsonl, resolve_existing_raw_path
from storyflow.data.amazon import write_amazon_readiness_report
from storyflow.observation import write_jsonl
from storyflow.providers import check_api_pilot_readiness
from storyflow.utils.config import load_simple_yaml


def _workspace(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _provider_config(path: Path) -> Path:
    config_path = path / "deepseek.yaml"
    config_path.write_text(
        "\n".join(
            [
                "provider_name: deepseek",
                "provider_family: openai_compatible",
                "model_name: deepseek-v4-flash",
                "api_key_env: DEEPSEEK_TEST_KEY",
                "base_url: https://api.deepseek.com",
                "endpoint: /chat/completions",
                "requires_endpoint_confirmation: false",
                "timeout_seconds: 60",
                "max_concurrency: 1",
                "temperature: 0.0",
                "max_tokens: 256",
                "retry:",
                "  max_attempts: 3",
                "  backoff_seconds: 1.0",
                "rate_limit:",
                "  requests_per_minute: 10",
                "cache:",
                "  enabled: true",
                f"  cache_dir: {(path / 'cache').as_posix()}",
                "dry_run:",
                "  default: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _input_jsonl(path: Path, *, n: int = 5) -> Path:
    catalog_csv = path / "item_catalog.csv"
    with catalog_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "title", "title_normalized", "popularity", "popularity_bucket"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "i1",
                "title": "Fixture Item",
                "title_normalized": "fixture item",
                "popularity": 1,
                "popularity_bucket": "head",
            }
        )
    rows = [
        {
            "input_id": f"input-{index}",
            "prompt": "Recommend next title.",
            "prompt_hash": f"hash-{index}",
            "prompt_template": "forced_json",
            "target_title": "Fixture Item",
            "source": {"catalog_csv": str(catalog_csv)},
        }
        for index in range(n)
    ]
    input_path = path / "inputs.jsonl"
    write_jsonl(input_path, rows)
    return input_path


def test_api_readiness_blocks_without_approval_or_env(monkeypatch) -> None:
    workspace = _workspace("api_readiness_blocked")
    config_path = _provider_config(workspace)
    input_path = _input_jsonl(workspace)
    monkeypatch.delenv("DEEPSEEK_TEST_KEY", raising=False)

    manifest = check_api_pilot_readiness(
        provider_config_path=config_path,
        input_jsonl=input_path,
        sample_size=5,
        stage="smoke",
    )

    assert manifest["status"] == "blocked"
    assert manifest["api_called"] is False
    assert any("approved_model" in blocker for blocker in manifest["blockers"])
    assert any("DEEPSEEK_TEST_KEY" in blocker for blocker in manifest["blockers"])


def test_api_readiness_ready_with_explicit_gates(monkeypatch) -> None:
    workspace = _workspace("api_readiness_ready")
    config_path = _provider_config(workspace)
    input_path = _input_jsonl(workspace)
    monkeypatch.setenv("DEEPSEEK_TEST_KEY", "not-a-real-key-for-tests")

    manifest = check_api_pilot_readiness(
        provider_config_path=config_path,
        input_jsonl=input_path,
        sample_size=5,
        stage="smoke",
        approved_provider="deepseek",
        approved_model="deepseek-v4-flash",
        approved_rate_limit=10,
        approved_max_concurrency=1,
        approved_budget_label="unit-test-budget",
        execute_api_intended=True,
    )

    assert manifest["status"] == "ready_for_execute_api"
    assert manifest["api_called"] is False
    assert manifest["api_key_value_printed"] is False
    assert "--execute-api" in manifest["command_template_after_approval"]
    assert "--max-concurrency 1" in manifest["command_template_after_approval"]
    assert "--budget-label unit-test-budget" in manifest["command_template_after_approval"]


def test_api_readiness_allows_explicit_over_20_pilot(monkeypatch) -> None:
    workspace = _workspace("api_readiness_over20")
    config_path = _provider_config(workspace)
    input_path = _input_jsonl(workspace, n=30)
    monkeypatch.setenv("DEEPSEEK_TEST_KEY", "not-a-real-key-for-tests")

    blocked = check_api_pilot_readiness(
        provider_config_path=config_path,
        input_jsonl=input_path,
        sample_size=30,
        stage="pilot",
        approved_provider="deepseek",
        approved_model="deepseek-v4-flash",
        approved_rate_limit=10,
        approved_max_concurrency=2,
        approved_budget_label="unit-test-budget",
        execute_api_intended=True,
    )
    ready = check_api_pilot_readiness(
        provider_config_path=config_path,
        input_jsonl=input_path,
        sample_size=30,
        stage="pilot",
        approved_provider="deepseek",
        approved_model="deepseek-v4-flash",
        approved_rate_limit=10,
        approved_max_concurrency=2,
        approved_budget_label="unit-test-budget",
        execute_api_intended=True,
        allow_over_20=True,
    )

    assert blocked["status"] == "blocked"
    assert any("sample_size" in blocker for blocker in blocked["blockers"])
    assert ready["status"] == "ready_for_execute_api"
    assert ready["allow_over_20"] is True
    assert ready["approved_max_concurrency"] == 2
    assert "--max-concurrency 2" in ready["command_template_after_approval"]
    assert ready["warnings"]


def test_api_readiness_supports_full_stage(monkeypatch) -> None:
    workspace = _workspace("api_readiness_full")
    config_path = _provider_config(workspace)
    input_path = _input_jsonl(workspace, n=185)
    monkeypatch.setenv("DEEPSEEK_TEST_KEY", "not-a-real-key-for-tests")

    manifest = check_api_pilot_readiness(
        provider_config_path=config_path,
        input_jsonl=input_path,
        sample_size=185,
        stage="full",
        approved_provider="deepseek",
        approved_model="deepseek-v4-flash",
        approved_rate_limit=30,
        approved_max_concurrency=3,
        approved_budget_label="unit-test-full-budget",
        execute_api_intended=True,
    )

    assert manifest["status"] == "ready_for_execute_api"
    assert manifest["stage"] == "full"
    assert manifest["sample_size"] == 185
    assert manifest["allow_over_20"] is False
    assert "--max-examples 185" in manifest["command_template_after_approval"]
    assert "--rate-limit 30" in manifest["command_template_after_approval"]
    assert any("full stage" in warning for warning in manifest["warnings"])


def test_api_readiness_cli_writes_manifest(monkeypatch) -> None:
    workspace = _workspace("api_readiness_cli")
    config_path = _provider_config(workspace)
    input_path = _input_jsonl(workspace)
    output_dir = workspace / "out"
    monkeypatch.setenv("DEEPSEEK_TEST_KEY", "not-a-real-key-for-tests")

    code = readiness_main(
        [
            "--provider-config",
            str(config_path),
            "--input-jsonl",
            str(input_path),
            "--sample-size",
            "5",
            "--approved-provider",
            "deepseek",
            "--approved-model",
            "deepseek-v4-flash",
            "--approved-rate-limit",
            "10",
            "--approved-max-concurrency",
            "2",
            "--approved-budget-label",
            "unit-test-budget",
            "--execute-api-intended",
            "--allow-over-20",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    manifest = json.loads((output_dir / "readiness_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ready_for_execute_api"
    assert manifest["allow_over_20"] is True
    assert manifest["approved_max_concurrency"] == 2


def test_amazon_local_path_resolution_and_schema_sample() -> None:
    workspace = _workspace("amazon_readiness")
    raw_dir = workspace / "raw"
    raw_dir.mkdir()
    reviews = raw_dir / "Tiny.jsonl"
    metadata = raw_dir / "meta_Tiny.jsonl"
    write_jsonl(
        reviews,
        [
            {
                "user_id": "u1",
                "parent_asin": "p1",
                "rating": 5,
                "timestamp": 1,
                "text": "nice",
            }
        ],
    )
    write_jsonl(
        metadata,
        [{"parent_asin": "p1", "title": "Tiny Item", "categories": ["A"], "store": "S"}],
    )
    config = {
        "name": "amazon_reviews_2023_tiny",
        "category_name": "Tiny",
        "hf_dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "hf_review_config": "raw_review_Tiny",
        "hf_meta_config": "raw_meta_Tiny",
        "source_url": "https://example.invalid",
        "raw_reviews_path": str(reviews),
        "raw_metadata_path": str(metadata),
        "user_id_field": "user_id",
        "item_id_field": "parent_asin",
        "rating_field": "rating",
        "timestamp_field": "timestamp",
        "review_text_field": "text",
        "metadata_join_key": "parent_asin",
        "title_field": "title",
    }

    assert resolve_existing_raw_path(config, "raw_reviews_path") == reviews
    manifest = inspect_amazon_config(config, sample_records=1)
    assert manifest["status"] == "local_raw_available"
    assert manifest["raw_reviews_path_exists"] is True
    assert manifest["review_schema_sample"]["sample_records_read"] == 1
    assert "user_id" in manifest["review_schema_sample"]["fields_seen"]


def test_amazon_readiness_report_uses_readable_chinese() -> None:
    workspace = _workspace("amazon_readiness_report")
    report_path = workspace / "readiness_report.md"
    manifest = {
        "dataset": "amazon_reviews_2023_tiny",
        "hf_dataset": "fixture",
        "category_name": "Tiny",
        "status": "local_raw_available",
        "full_download_attempted": False,
        "full_processed": False,
        "raw_reviews_path_exists": True,
        "raw_metadata_path_exists": True,
        "resume_command": "python scripts/inspect_amazon_reviews_2023.py --dataset tiny",
        "full_mode_command": "python scripts/prepare_amazon_reviews_2023.py --dataset tiny --allow-full",
        "warnings": [],
    }

    write_amazon_readiness_report(manifest, report_path)
    text = report_path.read_text(encoding="utf-8")

    assert "本报告只说明入口和可恢复状态" in text
    assert "full processed: False" in text
    assert "涓" not in text
    assert "鎵" not in text


def test_amazon_prepare_dry_run_writes_readable_chinese_report() -> None:
    workspace = _workspace("amazon_prepare_dry_run_report")

    code = prepare_amazon_main(
        [
            "--dataset",
            "amazon_reviews_2023_beauty",
            "--reviews-jsonl",
            str(workspace / "missing_reviews.jsonl"),
            "--metadata-jsonl",
            str(workspace / "missing_meta.jsonl"),
            "--dry-run",
        ]
    )

    report_path = (
        Path("outputs")
        / "amazon_reviews_2023"
        / "amazon_reviews_2023_beauty"
        / "prepare"
        / "prepare_readiness_report.md"
    )
    text = report_path.read_text(encoding="utf-8")
    assert code == 0
    assert "当前没有执行 full Amazon preprocessing" in text
    assert "该报告不是 full processed result" in text
    assert "涓" not in text
    assert "鎵" not in text


def test_local_amazon_configs_have_guarded_full_and_sample_commands() -> None:
    datasets = [
        "amazon_reviews_2023_beauty",
        "amazon_reviews_2023_digital_music",
        "amazon_reviews_2023_handmade",
        "amazon_reviews_2023_health",
        "amazon_reviews_2023_video_games",
        "amazon_reviews_2023_sports",
        "amazon_reviews_2023_books",
    ]
    for dataset in datasets:
        config = load_simple_yaml(Path("configs") / "datasets" / f"{dataset}.yaml")
        full_command = str(config.get("full_mode_command_template") or "")
        sample_command = str(config.get("local_sample_command_template") or "")
        assert "--allow-full" in full_command
        assert "--sample-mode" in sample_command
        assert "--max-records" in sample_command


def test_amazon_category_matrix_writes_readiness_artifacts() -> None:
    workspace = _workspace("amazon_category_matrix")
    output_dir = workspace / "matrix"

    code = amazon_matrix_main(
        [
            "--datasets",
            "amazon_reviews_2023_video_games",
            "amazon_reviews_2023_books",
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "amazon_category_matrix.json").read_text(encoding="utf-8"))
    report = (output_dir / "amazon_category_matrix.md").read_text(encoding="utf-8")
    csv_text = (output_dir / "amazon_category_matrix.csv").read_text(encoding="utf-8")

    assert code == 0
    assert manifest["api_called"] is False
    assert manifest["server_executed"] is False
    assert manifest["full_download_attempted"] is False
    assert manifest["is_experiment_result"] is False
    assert manifest["dataset_count"] == 2
    assert {row["dataset"] for row in manifest["records"]} == {
        "amazon_reviews_2023_video_games",
        "amazon_reviews_2023_books",
    }
    assert all("--allow-full" in row["full_mode_command_template"] for row in manifest["records"])
    assert "readiness artifact only" in report
    assert "amazon_reviews_2023_books" in csv_text


def test_amazon_sample_prepare_writes_manifest_and_popularity() -> None:
    workspace = _workspace("amazon_sample_prepare")
    raw_dir = workspace / "raw"
    processed_dir = workspace / "processed"
    raw_dir.mkdir()
    reviews = raw_dir / "Tiny.jsonl"
    metadata = raw_dir / "meta_Tiny.jsonl"
    write_jsonl(
        reviews,
        [
            {"user_id": "u1", "parent_asin": "p1", "rating": 5, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "p2", "rating": 5, "timestamp": 2},
            {"user_id": "u1", "parent_asin": "p3", "rating": 4, "timestamp": 3},
            {"user_id": "u2", "parent_asin": "p1", "rating": 5, "timestamp": 1},
            {"user_id": "u2", "parent_asin": "p2", "rating": 4, "timestamp": 4},
        ],
    )
    write_jsonl(
        metadata,
        [
            {"parent_asin": "unused", "title": "Unused Item"},
            {"parent_asin": "p1", "title": "Tiny Cleanser", "categories": ["Beauty"], "store": "A"},
            {"parent_asin": "p2", "title": "Tiny Lotion", "categories": ["Beauty"], "store": "B"},
            {"parent_asin": "p3", "title": "Tiny Brush", "categories": ["Beauty"], "store": "C"},
        ],
    )
    config = {
        "name": "amazon_reviews_2023_tiny",
        "category_name": "Tiny",
        "source_name": "fixture",
        "source_url": "https://example.invalid",
        "processed_dir": str(processed_dir),
        "user_id_field": "user_id",
        "item_id_field": "parent_asin",
        "rating_field": "rating",
        "timestamp_field": "timestamp",
        "metadata_join_key": "parent_asin",
        "title_field": "title",
        "preprocess_min_user_interactions": 1,
        "preprocess_user_k_core": 1,
        "preprocess_item_k_core": 1,
        "preprocess_min_history": 1,
        "preprocess_max_history": 5,
        "preprocess_split_policy": "global_chronological",
        "head_fraction": 0.4,
        "tail_fraction": 0.4,
    }

    summary = prepare_amazon_from_jsonl(
        config=config,
        reviews_jsonl=reviews,
        metadata_jsonl=metadata,
        output_suffix="sample_test",
        max_records=5,
    )

    manifest = json.loads((summary.output_dir / "preprocess_manifest.json").read_text(encoding="utf-8"))
    assert summary.example_count > 0
    assert manifest["is_sample_result"] is True
    assert manifest["is_full_result"] is False
    assert manifest["is_experiment_result"] is False
    assert manifest["result_scope"] == "local_sample"
    assert manifest["metadata_stats"]["matched_item_count"] == 3
    assert (summary.output_dir / "item_popularity.csv").exists()
