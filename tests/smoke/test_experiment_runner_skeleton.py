from __future__ import annotations

import json
from pathlib import Path

from llm4rec.experiments.runner import run_all


def test_experiment_runner_skeleton_writes_run_artifacts(tmp_path: Path) -> None:
    dataset_config = tmp_path / "tiny.yaml"
    processed_dir = tmp_path / "processed"
    dataset_config.write_text(
        "\n".join(
            [
                "name: tiny",
                "type: tiny_csv",
                "domain: tiny",
                "seed: 7",
                "interactions_path: tests/fixtures/tiny_interactions.csv",
                "items_path: tests/fixtures/tiny_items.csv",
                f"processed_dir: {processed_dir.as_posix()}",
                "split:",
                "  strategy: leave_one_out",
                "  min_history: 1",
                "candidate:",
                "  protocol: full",
                "  include_history: false",
                "  sample_size: null",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    experiment_config = tmp_path / "smoke.yaml"
    run_root = tmp_path / "runs"
    experiment_config.write_text(
        "\n".join(
            [
                "seed: 7",
                "run_name: smoke_skeleton_test",
                f"output_dir: {run_root.as_posix()}",
                "domain: tiny",
                "dataset:",
                "  name: tiny",
                f"  config_path: {dataset_config.as_posix()}",
                f"  processed_dir: {processed_dir.as_posix()}",
                "split: test",
                "candidate:",
                "  protocol: full",
                "method: skeleton",
                "top_k: [1, 5]",
                "logging:",
                "  level: INFO",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = run_all(experiment_config)
    run_dir = Path(result["run_dir"])
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "logs.txt").exists()
    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "artifacts").is_dir()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["aggregate"]["recall@1"] == 1.0
    assert metrics["aggregate"]["validity_rate"] == 1.0
