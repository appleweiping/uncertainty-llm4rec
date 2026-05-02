from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REAL_TEMPLATES = [
    "configs/experiments/real_main_movielens_template.yaml",
    "configs/experiments/real_main_amazon_template.yaml",
    "configs/experiments/real_llm_api_template.yaml",
    "configs/experiments/real_ours_method_template.yaml",
    "configs/experiments/real_ablation_template.yaml",
    "configs/experiments/real_observation_template.yaml",
    "configs/experiments/real_multiseed_template.yaml",
]


def test_validate_experiment_ready_accepts_safe_ours_template() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_experiment_ready.py",
            "--config",
            "configs/experiments/real_ours_method_template.yaml",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ready"] is True
    assert payload["checks"]["method_configured"] is True
    assert payload["checks"]["metrics_configured"] is True
    assert payload["checks"]["split_strategy_configured"] is True
    assert payload["checks"]["candidate_protocol_configured"] is True
    assert payload["checks"]["data_protocol_configured"] is True
    assert payload["checks"]["leakage_safeguards_present"] is True
    assert payload["checks"]["safety_defaults_safe"] is True
    assert payload["dry_run"] is True
    assert payload["requires_confirm"] is True


def test_validate_experiment_ready_accepts_all_real_templates() -> None:
    for config_path in REAL_TEMPLATES:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_experiment_ready.py",
                "--config",
                config_path,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        payload = json.loads(result.stdout)
        assert result.returncode == 0, payload
        assert payload["ready"] is True, payload
        assert payload["checks"]["safety_defaults_safe"] is True
        assert payload["checks"]["leakage_safeguards_present"] is True


def test_validate_experiment_ready_rejects_missing_method(tmp_path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        "\n".join(
            [
                "dry_run: true",
                "requires_confirm: true",
                "dataset:",
                "  name: tiny",
                "  processed_dir: TBD",
                "split: test",
                "candidate:",
                "  protocol: full",
                "metrics: [ranking]",
                "safety:",
                "  leakage_safeguards: true",
                "output:",
                f"  output_dir: {tmp_path}",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/validate_experiment_ready.py", "--config", str(config)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["ready"] is False
    assert "check failed: method_configured" in payload["errors"]


def test_validate_experiment_ready_rejects_missing_split_strategy(tmp_path) -> None:
    config = tmp_path / "bad_split.yaml"
    config.write_text(
        "\n".join(
            [
                "dry_run: true",
                "requires_confirm: true",
                "seed: 13",
                "dataset:",
                "  name: tiny",
                "  processed_dir: TBD",
                "split: test",
                "candidate:",
                "  protocol: full",
                "  seed: 13",
                "  set_path: TBD",
                "  target_policy: include_heldout_target_for_main_eval",
                "  target_excluding_protocol: diagnostic_only_not_accuracy",
                "  same_set_for_comparable_baselines: true",
                "data_protocol:",
                "  interaction_schema: [user_id, item_id, timestamp, rating, domain]",
                "  item_schema: [item_id, title, description, category, brand, domain, raw_text]",
                "  user_item_id_mapping: required_saved_in_processed_artifacts",
                "  timestamp_handling: required_for_temporal_or_document_leave_one_out_order",
                "  train_popularity_source: train_split_only",
                "  domain_field: domain",
                "  candidate_set_saved_path: TBD",
                "method:",
                "  name: popularity",
                "metrics: [ranking]",
                "safety:",
                "  allow_api_calls: false",
                "  allow_download: false",
                "  allow_training: false",
                "  leakage_safeguards: true",
                "output:",
                f"  output_dir: {tmp_path}",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/validate_experiment_ready.py", "--config", str(config)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["ready"] is False
    assert "check failed: split_strategy_configured" in payload["errors"]
