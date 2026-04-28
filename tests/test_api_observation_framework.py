from __future__ import annotations

import csv
import json
import urllib.error
import uuid
from pathlib import Path

import pytest

from scripts.run_api_observation import run_api_observation
from storyflow.data import inspect_amazon_config
from storyflow.observation import read_jsonl, write_jsonl
from storyflow.observation_parsing import (
    normalize_confidence,
    parse_observation_response,
)
from storyflow.providers import ResponseCache, build_cache_key, load_provider_config
from storyflow.utils.config import load_simple_yaml


def _workspace(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _provider_config(path: Path, *, cache_dir: Path) -> Path:
    config_path = path / "provider.yaml"
    config_path.write_text(
        "\n".join(
            [
                "provider_name: dry_provider",
                "provider_family: openai_compatible",
                "model_name: TODO_CONFIRM_MODEL",
                "api_key_env: MISSING_DRY_RUN_KEY",
                "base_url: TODO_CONFIRM_BASE_URL",
                "endpoint: TODO_CONFIRM_ENDPOINT",
                "requires_endpoint_confirmation: true",
                "timeout_seconds: 10",
                "max_concurrency: 1",
                "temperature: 0.0",
                "max_tokens: 128",
                "retry:",
                "  max_attempts: 2",
                "  backoff_seconds: 0.01",
                "rate_limit:",
                "  requests_per_minute: 60",
                "cache:",
                "  enabled: true",
                f"  cache_dir: {cache_dir.as_posix()}",
                "dry_run:",
                "  default: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _input_jsonl(path: Path, *, n: int = 3) -> Path:
    catalog_csv = path / "item_catalog.csv"
    with catalog_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "title", "title_normalized", "genres", "popularity", "popularity_bucket"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "item_id": "item-1",
                    "title": "The Matrix",
                    "title_normalized": "matrix",
                    "genres": "Sci-Fi",
                    "popularity": 10,
                    "popularity_bucket": "head",
                },
                {
                    "item_id": "item-2",
                    "title": "Arrival",
                    "title_normalized": "arrival",
                    "genres": "Sci-Fi",
                    "popularity": 3,
                    "popularity_bucket": "tail",
                },
            ]
        )
    rows = []
    for index in range(n):
        rows.append(
            {
                "input_id": f"input-{index}",
                "dataset": "fixture",
                "processed_suffix": "tiny",
                "example_id": f"user:{index}",
                "user_id": "user",
                "split": "test",
                "history_item_ids": ["item-2"],
                "history_item_titles": ["Arrival"],
                "history_timestamps": [1],
                "history_length": 1,
                "target_item_id": "item-1",
                "target_title": "The Matrix",
                "target_timestamp": 2 + index,
                "target_popularity": 10,
                "target_popularity_bucket": "head",
                "prompt_template": "forced_json",
                "prompt": "Generate JSON for The Matrix",
                "prompt_hash": f"hash-{index}",
                "source": {
                    "catalog_csv": str(catalog_csv),
                    "processed_dir": str(path),
                    "observation_examples": str(path / "observation_examples.jsonl"),
                },
            }
        )
    input_path = path / "inputs.jsonl"
    write_jsonl(input_path, rows)
    return input_path


def test_provider_config_loading() -> None:
    config = load_provider_config("configs/providers/deepseek.yaml")

    assert config.provider_name == "deepseek"
    assert config.api_key_env == "DEEPSEEK_API_KEY"
    assert config.dry_run_default is True
    assert config.model_name == "deepseek-v4-flash"
    assert config.endpoint_is_placeholder is False


def test_missing_api_key_dry_run_does_not_trigger_real_call(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = _workspace("api_missing_key")
    config_path = _provider_config(workspace, cache_dir=workspace / "cache")
    input_path = _input_jsonl(workspace, n=1)
    monkeypatch.delenv("MISSING_DRY_RUN_KEY", raising=False)

    manifest = run_api_observation(
        provider_config_path=config_path,
        input_jsonl=input_path,
        output_dir=workspace / "run",
        max_examples=1,
        dry_run=True,
        resume=True,
        max_concurrency=None,
        rate_limit=None,
    )

    assert manifest["api_called"] is False
    assert manifest["dry_run"] is True


def test_cache_key_is_deterministic() -> None:
    first = build_cache_key(
        provider="p",
        model="m",
        prompt_template="forced_json",
        temperature=0.0,
        input_hash="abc",
    )
    second = build_cache_key(
        provider="p",
        model="m",
        prompt_template="forced_json",
        temperature=0.0,
        input_hash="abc",
    )

    assert first == second
    assert len(first) == 64


def test_response_cache_hit_roundtrip() -> None:
    workspace = _workspace("cache_roundtrip")
    cache = ResponseCache(workspace / "cache")
    cache.put("key", {"raw_text": "value"})

    assert cache.get("key") == {"raw_text": "value"}
    assert cache.get("missing") is None


def test_api_runner_cache_hit_and_resume() -> None:
    workspace = _workspace("api_runner")
    config_path = _provider_config(workspace, cache_dir=workspace / "cache")
    input_path = _input_jsonl(workspace, n=3)
    run_dir = workspace / "run"

    first = run_api_observation(
        provider_config_path=config_path,
        input_jsonl=input_path,
        output_dir=run_dir,
        max_examples=2,
        dry_run=True,
        resume=True,
        max_concurrency=None,
        rate_limit=None,
    )
    second = run_api_observation(
        provider_config_path=config_path,
        input_jsonl=input_path,
        output_dir=run_dir,
        max_examples=3,
        dry_run=True,
        resume=True,
        max_concurrency=None,
        rate_limit=None,
    )

    assert first["newly_processed_count"] == 2
    assert second["newly_processed_count"] == 1
    assert second["total_grounded_count"] == 3

    cache_run_dir = workspace / "cache_run"
    cache_manifest = run_api_observation(
        provider_config_path=config_path,
        input_jsonl=input_path,
        output_dir=cache_run_dir,
        max_examples=1,
        dry_run=True,
        resume=False,
        max_concurrency=None,
        rate_limit=None,
    )
    raw_rows = read_jsonl(cache_run_dir / "raw_responses.jsonl")
    assert cache_manifest["newly_processed_count"] == 1
    assert raw_rows[0]["cache_hit"] is True


def test_parser_strict_json_fenced_json_and_regex() -> None:
    strict = parse_observation_response(
        '{"generated_title":"The Matrix","is_likely_correct":"YES","confidence":0.7}'
    )
    fenced = parse_observation_response(
        '```json\n{"generated_title":"Arrival","is_likely_correct":"no","confidence":"33%"}\n```'
    )
    regex = parse_observation_response("Title: Primer\nCorrect: yes\nConfidence: 72%")

    assert strict.success and strict.parse_strategy == "strict_json"
    assert fenced.success and fenced.confidence == pytest.approx(0.33)
    assert regex.success and regex.generated_title == "Primer"


def test_confidence_normalization_and_parse_failure() -> None:
    assert normalize_confidence("72%") == pytest.approx(0.72)
    assert normalize_confidence(72) == pytest.approx(0.72)

    failure = parse_observation_response('{"generated_title":"","confidence":0.5}')
    assert failure.success is False
    assert failure.error


def test_amazon_config_and_inspector_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_simple_yaml("configs/datasets/amazon_reviews_2023_beauty.yaml")
    assert config["category_name"] == "All_Beauty"
    assert config["metadata_join_key"] == "parent_asin"

    def _raise(*args, **kwargs):
        raise urllib.error.URLError("network unavailable")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    manifest = inspect_amazon_config(config, check_online=True)
    assert manifest["status"] == "online_check_failed"
    assert manifest["full_download_attempted"] is False
    assert manifest["full_processed"] is False
    assert manifest["warnings"]
