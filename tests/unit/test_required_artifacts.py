from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_list_required_artifacts_for_config() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/list_required_artifacts.py",
            "--config",
            "configs/experiments/real_ours_method_template.yaml",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "predictions.jsonl" in payload["required_artifacts"]
    assert "metrics.json" in payload["required_artifacts"]
    assert payload["planned_output_dir"] == "outputs/real_runs/ours_method"
    assert payload["template"] is True


def test_list_required_artifacts_uses_r2_parent_prefix(tmp_path) -> None:
    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        "\n".join(
            [
                "name: r2_fixture",
                "type: tiny_csv",
                "interactions_path: tests/fixtures/tiny_interactions.csv",
                "items_path: tests/fixtures/tiny_items.csv",
                f"processed_dir: {tmp_path / 'processed'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "r2.yaml"
    config_path.write_text(
        "\n".join(
            [
                "template: false",
                "dry_run: false",
                "requires_confirm: false",
                "seed: 13",
                "seeds: [13]",
                "run_name: r2_fixture_full",
                "output_dir: outputs/runs",
                "baselines: [popularity, bm25]",
                "dataset:",
                f"  config_path: {dataset_config}",
                f"  processed_dir: {tmp_path / 'processed'}",
                "output:",
                "  run_name: r2_fixture_full",
                "  output_dir: outputs/runs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/list_required_artifacts.py", "--config", str(config_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    planned = {path.replace("\\", "/") for path in payload["planned_run_dirs"]}
    assert "outputs/runs/r2_fixture_full_popularity_seed13" in planned
    assert "outputs/runs/r2_fixture_full_bm25_seed13" in planned


def test_list_required_artifacts_reports_missing_run_files(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("seed: 13\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/list_required_artifacts.py", "--run-dir", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "resolved_config.yaml" in payload["present_required_artifacts"]
    assert "predictions.jsonl" in payload["missing_required_artifacts"]
